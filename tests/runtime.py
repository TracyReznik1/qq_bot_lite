from __future__ import annotations

import atexit
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from typing import Any, Callable, Iterator
from unittest import mock

import requests


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TEST_GEMINI_API_KEY = (
    "AIza" + "A1b2_C3d4-E5f6_G7h8-I9j0_K1l2M3n4"
)
COMPACT_HARD_SECRET_CASES = (
    ("otp_zh", "验证码654321", "654321"),
    ("otp_en", "OTP482915", "482915"),
    ("password_zh", "密码hunter2", "hunter2"),
    ("payment_password", "支付密码123456", "123456"),
    ("bank_account", "银行账号9876543210", "9876543210"),
    ("bank_account_alt", "银行帐号1357902468", "1357902468"),
    ("bank_account_word", "银行账户112233445566", "112233445566"),
    ("payment_account", "支付账号2468013579", "2468013579"),
    ("cvv", "CVV123", "123"),
    ("cvc", "CVC987", "987"),
    ("api_key_zh", f"API密钥{TEST_GEMINI_API_KEY}", TEST_GEMINI_API_KEY),
    ("gemini_bare", TEST_GEMINI_API_KEY, TEST_GEMINI_API_KEY),
)
ORDINARY_COMPACT_SECRET_PHRASES = (
    "验证码功能说明",
    "OTP流程设计",
    "密码学基础",
    "支付密码功能",
    "银行账号管理",
    "银行帐号变更流程",
    "银行账户有两个",
    "支付账号页面",
    "CVV规则",
    "CVC校验流程",
    "API密钥轮换方案",
    "AIza前缀说明",
)
REPOSITORY_DEFAULT_MEMORY_DB = (
    REPOSITORY_ROOT / "qqbot_data" / "memory.sqlite3"
).resolve()
_guard_patchers: list[mock._patch] = []


def _resolved_database_path(database: Any) -> Path | None:
    if isinstance(database, int):
        return None
    value = os.fspath(database)
    if value == ":memory:":
        return None
    if value.startswith("file:"):
        value = value[5:].split("?", 1)[0]
    try:
        return Path(value).resolve()
    except (OSError, RuntimeError, ValueError):
        return None


def guarded_sqlite_connect(
    original: Callable[..., sqlite3.Connection],
    database: Any,
    *args: Any,
    **kwargs: Any,
) -> sqlite3.Connection:
    resolved = _resolved_database_path(database)
    if resolved == REPOSITORY_DEFAULT_MEMORY_DB:
        raise AssertionError(
            "tests must not open the repository default memory database: "
            f"{resolved}"
        )
    return original(database, *args, **kwargs)


def guarded_http_request(
    _original: Callable[..., Any],
    _session: requests.Session,
    method: str,
    url: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    del args, kwargs
    raise AssertionError(
        "tests must mock the external transport boundary: "
        f"{str(method).upper()} {url}"
    )


def install_test_guards() -> None:
    if _guard_patchers:
        return
    original_connect = sqlite3.connect
    original_request = requests.sessions.Session.request
    patchers = [
        mock.patch.object(
            sqlite3,
            "connect",
            lambda database, *args, **kwargs: guarded_sqlite_connect(
                original_connect,
                database,
                *args,
                **kwargs,
            ),
        ),
        mock.patch.object(
            requests.sessions.Session,
            "request",
            lambda session, method, url, *args, **kwargs: (
                guarded_http_request(
                    original_request,
                    session,
                    method,
                    url,
                    *args,
                    **kwargs,
                )
            ),
        ),
    ]
    for patcher in patchers:
        patcher.start()
    _guard_patchers.extend(patchers)


def uninstall_test_guards() -> None:
    while _guard_patchers:
        _guard_patchers.pop().stop()


def _stop_global_memory_service() -> None:
    service_module = sys.modules.get("src.memory.service")
    if service_module is None:
        return
    service = getattr(service_module, "_global_memory_service", None)
    if service is not None:
        service.stop()
    service_module._global_memory_service = None


def reset_runtime_singletons() -> None:
    _stop_global_memory_service()

    llm_module = sys.modules.get("src.services.llm_client")
    if llm_module is not None:
        llm_module._llm_client = None
        llm_module._memory_llm_client = None

    persona_module = sys.modules.get("src.persona")
    if persona_module is not None:
        persona_module.get_persona.cache_clear()

    chat_module = sys.modules.get("src.chat.chat_service")
    if chat_module is not None:
        with chat_module.chat_history_lock:
            chat_module.chat_history.clear()

    main_module = sys.modules.get("src.main")
    if main_module is not None:
        main_module._startup_initialized = False


@dataclass
class IsolatedRuntime:
    data_dir: Path
    config: Any
    _services: list[Any] = field(default_factory=list)

    def track_service(self, service: Any) -> Any:
        self._services.append(service)
        return service

    def stop_workers(self) -> None:
        while self._services:
            self._services.pop().stop()
        _stop_global_memory_service()


@contextmanager
def isolated_runtime(
    data_dir: str | os.PathLike[str] | None = None,
) -> Iterator[IsolatedRuntime]:
    temporary_root = (
        tempfile.TemporaryDirectory(prefix="qqbot-lite-runtime-")
        if data_dir is None
        else None
    )
    root = Path(
        temporary_root.name if temporary_root is not None else data_dir
    ).resolve()
    root.mkdir(parents=True, exist_ok=True)

    from src import config as config_module

    old_config = config_module.config
    with (
        mock.patch.dict(os.environ, {"DATA_DIR": str(root)}),
        ExitStack() as stack,
    ):
        current_config = config_module.Config()
        for module in tuple(sys.modules.values()):
            if (
                module is not None
                and str(getattr(module, "__name__", "")).startswith("src")
                and getattr(module, "config", None) is old_config
            ):
                stack.enter_context(
                    mock.patch.object(module, "config", current_config)
                )
        stack.enter_context(
            mock.patch.object(
                config_module,
                "config",
                current_config,
            )
        )
        reset_runtime_singletons()
        runtime = IsolatedRuntime(root, current_config)
        try:
            yield runtime
        finally:
            runtime.stop_workers()
            reset_runtime_singletons()
    if temporary_root is not None:
        temporary_root.cleanup()


atexit.register(uninstall_test_guards)
atexit.register(reset_runtime_singletons)
