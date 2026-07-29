import hmac
import logging
import re
import time
from threading import Lock
from typing import Any

from flask import Flask, request

from src.chat.chat_service import generate_reply
from src.commands import CommandContext, handle_command
from src.commands.renderer import PersonaCommandRenderer
from src.config import BASE_DIR, config
from src.persona import get_persona
from src.messaging import (
    MessageQueue,
    enqueue_message,
    get_event_memory_scope_key,
    get_event_session_key,
    mark_message_seen,
)
from src.memory.models import MemoryContext, MemoryEvent
from src.memory.service import get_memory_service
from src.router import route_message
from src.services.image_input_service import (
    ImageInputError,
    load_chat_images,
    parse_image_message,
)
from src.services.llm_client import ImageRecognitionUnavailable
from src.services.llm_client import get_llm_client
from src.services.onebot_client import OneBotClient
from src.utils.data_migration import LEGACY_DATA_DIR_NAME, migrate_legacy_data


app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("qq-bot")
CALLBACK_SECRET_HEADER = "X-QQBOT-Callback-Secret"
LEGACY_CALLBACK_SECRET_HEADER = "X-ATRI-Callback-Secret"

MAX_PROCESSED_MESSAGE_IDS = 500
_startup_initialized = False
_startup_lock = Lock()

onebot = OneBotClient(config)
command_renderer = PersonaCommandRenderer(model=get_llm_client())


def _register_pending_memory_sequence(data: dict[str, Any]) -> None:
    scope_key = get_event_memory_scope_key(data)
    sequence = int(data.get("_qqbot_sequence") or 0)
    if scope_key and sequence > 0:
        get_memory_service().register_pending_sequence(scope_key, sequence)


def _clear_pending_memory_sequence(data: dict[str, Any]) -> None:
    scope_key = get_event_memory_scope_key(data)
    sequence = int(data.get("_qqbot_sequence") or 0)
    if scope_key and sequence > 0:
        get_memory_service().clear_pending_sequence(scope_key, sequence)


message_queue = MessageQueue(
    max_workers=config.message_workers,
    max_processed_message_ids=MAX_PROCESSED_MESSAGE_IDS,
    on_accepted=_register_pending_memory_sequence,
    on_rejected=_clear_pending_memory_sequence,
)


CQ_REPLY_PATTERN = re.compile(
    r"\[CQ:reply,(?:[^\]]*,)?id=([^,\]]+)[^\]]*\]",
    re.IGNORECASE,
)


def get_reply_message_id(data: dict[str, Any], raw_message: str) -> str | None:
    message = data.get("message")
    if isinstance(message, list):
        for segment in message:
            if not isinstance(segment, dict) or segment.get("type") != "reply":
                continue
            segment_data = segment.get("data")
            if isinstance(segment_data, dict):
                message_id = str(
                    segment_data.get("id")
                    or segment_data.get("message_id")
                    or ""
                ).strip()
                if message_id:
                    return message_id
    match = CQ_REPLY_PATTERN.search(raw_message)
    if match:
        return match.group(1).strip()
    message_id = str(data.get("reply_to_message_id") or "").strip()
    return message_id or None


def startup() -> None:
    global _startup_initialized
    if _startup_initialized:
        return

    with _startup_lock:
        if _startup_initialized:
            return

        get_persona()
        default_data_dir = (BASE_DIR / "qqbot_data").resolve()
        if config.data_dir.resolve() == default_data_dir:
            migrate_legacy_data(
                BASE_DIR / LEGACY_DATA_DIR_NAME,
                config.data_dir,
                config.history_turns,
            )
        get_memory_service().start()
        _startup_initialized = True


def strip_bot_mention(raw_msg: str, self_id: str) -> tuple[bool, str]:
    import re

    pattern = re.compile(rf"\[CQ:at,qq={re.escape(self_id)}(?:,[^\]]*)?\]")
    if pattern.search(raw_msg):
        stripped = pattern.sub("", raw_msg).strip()
        stripped = re.sub(r"^(?:\[CQ:reply,[^\]]+\]\s*)+", "", stripped).strip()
        return True, stripped
    return False, raw_msg.strip()


def split_reply(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []

    limit = max(config.max_reply_chars, 200)
    parts = []
    while len(text) > limit:
        cut = max(text.rfind("\n", 0, limit), text.rfind("。", 0, limit), text.rfind("，", 0, limit))
        if cut < limit // 2:
            parts.append(text[:limit].strip())
            text = text[limit:].strip()
        else:
            parts.append(text[: cut + 1].strip())
            text = text[cut + 1 :].strip()
    if text:
        parts.append(text)
    return parts


def send_reply(target_id: Any, text: str, is_group: bool) -> None:
    parts = split_reply(text)
    if not parts:
        logger.info("Reply skipped: empty text target_id=%s is_group=%s", target_id, is_group)
        return

    logger.info(
        "Sending reply target_id=%s is_group=%s parts=%s chars=%s",
        target_id,
        is_group,
        len(parts),
        len(text or ""),
    )
    for index, part in enumerate(parts, 1):
        logger.info(
            "Sending reply part target_id=%s is_group=%s part=%s/%s chars=%s",
            target_id,
            is_group,
            index,
            len(parts),
            len(part),
        )
        onebot.send_msg(target_id, part, is_group=is_group)
        time.sleep(0.2)


def _process_message(data: dict[str, Any]) -> None:
    uid = str(data.get("user_id", ""))
    raw_msg = str(data.get("raw_message", "")).strip()
    if not uid or not raw_msg:
        logger.info(
            "Message ignored: missing uid or raw_message message_type=%s group_id=%s user_id=%s message_id=%s",
            data.get("message_type"),
            data.get("group_id"),
            data.get("user_id"),
            data.get("message_id"),
        )
        return

    is_group = data.get("message_type") == "group"
    self_id = str(data.get("self_id", ""))
    target_id = data.get("group_id") if is_group else uid
    session_key = get_event_session_key(data) or ""

    if is_group:
        logger.info(
            "Group message received group_id=%s user_id=%s self_id=%s message_id=%s require_group_at=%s",
            data.get("group_id"),
            uid,
            self_id,
            data.get("message_id"),
            config.require_group_at,
        )

    if is_group and config.require_group_at:
        mentioned, raw_msg = strip_bot_mention(raw_msg, self_id)
        logger.info(
            "Group mention check group_id=%s user_id=%s self_id=%s message_id=%s mentioned=%s",
            data.get("group_id"),
            uid,
            self_id,
            data.get("message_id"),
            mentioned,
        )
        if not mentioned:
            logger.info(
                "Group message ignored: bot not mentioned group_id=%s user_id=%s message_id=%s",
                data.get("group_id"),
                uid,
                data.get("message_id"),
            )
            return
        if not raw_msg:
            logger.info(
                "Group message ignored: mention-only message group_id=%s user_id=%s message_id=%s",
                data.get("group_id"),
                uid,
                data.get("message_id"),
            )
            return

    raw_msg = CQ_REPLY_PATTERN.sub("", raw_msg).strip()

    try:
        parsed_message = parse_image_message(data, raw_msg)
        route_text = parsed_message.text or (
            "[图片]" if parsed_message.image_urls else ""
        )
        if not route_text:
            return

        route = route_message(route_text)
        logger.info(
            "Message routed session_key=%s is_group=%s handler=%s command=%s query_chars=%s",
            session_key,
            is_group,
            route.handler,
            route.command,
            len(route.query or ""),
        )
        mem_ctx = MemoryContext(
            user_id=uid,
            session_key=session_key,
            is_group=is_group,
            group_id=str(data.get("group_id")) if is_group else None,
        )
        if route.handler == "command":
            result = handle_command(
                route,
                CommandContext(
                    uid=uid,
                    session_key=session_key,
                    raw_message=route_text,
                    memory_context=mem_ctx,
                    message_id=str(data.get("message_id") or ""),
                ),
                renderer=command_renderer,
            )
            logger.info(
                "Command handled session_key=%s command=%s handled=%s reply_chars=%s",
                session_key,
                route.command,
                result.handled,
                len(result.reply or ""),
            )
            if result.handled and result.reply:
                send_reply(target_id, result.reply, is_group)
            return

        reply_to_message_id = get_reply_message_id(
            data,
            str(data.get("raw_message") or ""),
        )
        reply_to_user_id = (
            onebot.get_message_author(reply_to_message_id)
            if reply_to_message_id
            else None
        )
        if reply_to_user_id is None:
            reply_to_user_id = (
                str(data.get("reply_to_user_id"))
                if data.get("reply_to_user_id")
                else None
            )
        mem_event = MemoryEvent(
            context=mem_ctx,
            message_id=str(data.get("message_id") or ""),
            sequence=int(data.get("_qqbot_sequence") or 0),
            text=parsed_message.text,
            image_count=len(parsed_message.image_urls),
            mentioned_qq_ids=tuple(str(qid) for qid in data.get("mentioned_qq_ids") or ()),
            reply_to_message_id=reply_to_message_id,
            reply_to_user_id=reply_to_user_id,
        )
        memory_service = get_memory_service()
        job_id: int | None = None
        try:
            job_id = memory_service.stage_event(mem_event)
        except Exception as error:
            scope_type, scope_id = mem_ctx.primary_scope
            logger.error(
                "Memory stage failed scope_key=%s sequence=%s error_type=%s",
                f"{scope_type}:{scope_id}",
                mem_event.sequence,
                type(error).__name__,
            )
        image_data_urls: list[str] = []

        try:
            logger.info("Generating chat reply session_key=%s is_group=%s", session_key, is_group)
            image_data_urls = load_chat_images(
                parsed_message.image_urls,
                image_file_ids=parsed_message.image_file_ids,
                image_url_resolver=onebot.get_image_url,
            )
            reply = generate_reply(
                session_key,
                parsed_message.text,
                image_data_urls=image_data_urls,
            )
            logger.info("Chat reply generated session_key=%s reply_chars=%s", session_key, len(reply or ""))
            send_reply(target_id, reply, is_group)
        finally:
            if job_id is not None:
                try:
                    memory_service.release_job(job_id, image_data_urls)
                except Exception as error:
                    logger.error(
                        "Memory release failed job_id=%s error_type=%s",
                        job_id,
                        type(error).__name__,
                    )
    except ImageInputError as error:
        send_reply(target_id, str(error), is_group)
    except ImageRecognitionUnavailable as error:
        logger.info("Image recognition unavailable session_key=%s", session_key)
        send_reply(target_id, str(error), is_group)
    except RuntimeError as error:
        logger.error(
            "Configuration error error_type=%s",
            type(error).__name__,
        )
        send_reply(target_id, f"配置还没好：{error}", is_group)
    except Exception as error:
        logger.error(
            "Message handling failed error_type=%s",
            type(error).__name__,
        )
        send_reply(target_id, "我这边处理失败了，先缓一缓再试。", is_group)


def process_message(data: dict[str, Any]) -> None:
    scope_key = get_event_memory_scope_key(data)
    sequence = int(data.get("_qqbot_sequence") or 0)
    try:
        _process_message(data)
    finally:
        if scope_key and sequence > 0:
            try:
                get_memory_service().clear_pending_sequence(scope_key, sequence)
            except Exception as error:
                logger.error(
                    "Memory sequence cleanup failed scope_key=%s sequence=%s error_type=%s",
                    scope_key,
                    sequence,
                    type(error).__name__,
                )


def process_message_safely(data: dict[str, Any]) -> None:
    try:
        process_message(data)
    except Exception as error:
        logger.error(
            "Background message processing failed error_type=%s",
            type(error).__name__,
        )


def is_callback_authorized() -> bool:
    secret = config.callback_secret.strip()
    if not secret:
        return True

    authorization = request.headers.get("Authorization", "").strip()
    callback_secret = request.headers.get(CALLBACK_SECRET_HEADER, "").strip()
    legacy_secret = request.headers.get(LEGACY_CALLBACK_SECRET_HEADER, "").strip()
    if hmac.compare_digest(authorization, f"Bearer {secret}") or hmac.compare_digest(
        callback_secret,
        secret,
    ):
        return True
    if legacy_secret and hmac.compare_digest(legacy_secret, secret):
        logger.warning("Legacy callback header accepted; update OneBot configuration")
        return True
    return False


@app.route("/", methods=["POST"])
def onebot_event() -> dict[str, str] | tuple[dict[str, str], int]:
    if not is_callback_authorized():
        logger.warning("Rejected unauthorized OneBot callback")
        return {"status": "forbidden"}, 403

    startup()
    data = request.get_json(silent=True) or {}
    if data.get("post_type") == "message":
        seen = mark_message_seen(data, message_queue)
        logger.info(
            "OneBot message callback message_type=%s group_id=%s user_id=%s message_id=%s accepted=%s",
            data.get("message_type"),
            data.get("group_id"),
            data.get("user_id"),
            data.get("message_id"),
            seen,
        )
        if seen:
            enqueue_message(data, message_queue, process_message_safely)
    return {"status": "ok"}


@app.route("/health", methods=["GET"])
def health() -> dict[str, Any]:
    persona = get_persona()
    return {
        "status": "ok",
        "bot_name": persona.name,
        "chat_models": [
            {"provider": item.provider, "model": item.model}
            for item in config.chat_models
        ],
        "gemini_configured": bool(config.gemini_api_key),
        "deepseek_configured": bool(config.deepseek_api_key),
        "onebot_url": config.onebot_url,
        "require_group_at": config.require_group_at,
    }


def run() -> None:
    startup()
    logger.info("Starting %s on %s:%s", get_persona().name, config.host, config.port)
    app.run(host=config.host, port=config.port)


if __name__ == "__main__":
    run()
