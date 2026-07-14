import hmac
import logging
import time
from threading import Lock
from typing import Any

from flask import Flask, request

from src.chat.chat_service import generate_reply
from src.chat.memory import migrate_legacy_memory_files
from src.commands import CommandContext, handle_command
from src.config import BASE_DIR, config
from src.messaging import (
    MessageQueue,
    enqueue_message,
    get_event_session_key,
    mark_message_seen,
)
from src.router import route_message
from src.services.image_input_service import (
    ImageInputError,
    load_chat_images,
    parse_image_message,
)
from src.services.llm_client import ImageRecognitionUnavailable
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

message_queue = MessageQueue(
    max_workers=4,
    max_processed_message_ids=MAX_PROCESSED_MESSAGE_IDS,
)


def message_preview(text: str, limit: int = 80) -> str:
    preview = " ".join(str(text or "").split())
    if len(preview) > limit:
        return preview[:limit] + "..."
    return preview


def startup() -> None:
    global _startup_initialized
    if _startup_initialized:
        return

    with _startup_lock:
        if _startup_initialized:
            return

        default_data_dir = (BASE_DIR / "qqbot_data").resolve()
        if config.data_dir.resolve() == default_data_dir:
            migrate_legacy_data(
                BASE_DIR / LEGACY_DATA_DIR_NAME,
                config.data_dir,
                config.history_turns,
                config.memory_limit,
            )
        migrate_legacy_memory_files()
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
            "Sending reply part target_id=%s is_group=%s part=%s/%s chars=%s preview=%r",
            target_id,
            is_group,
            index,
            len(parts),
            len(part),
            message_preview(part),
        )
        onebot.send_msg(target_id, part, is_group=is_group)
        time.sleep(0.2)


def process_message(data: dict[str, Any]) -> None:
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

    try:
        parsed_message = parse_image_message(data, raw_msg)
        route_text = parsed_message.text or (
            "[图片]" if parsed_message.image_urls else ""
        )
        if not route_text:
            return

        route = route_message(route_text)
        logger.info(
            "Message routed session_key=%s is_group=%s handler=%s command=%s query=%r",
            session_key,
            is_group,
            route.handler,
            route.command,
            message_preview(route.query),
        )
        if route.handler == "command":
            result = handle_command(
                route,
                CommandContext(uid=uid, session_key=session_key, raw_message=route_text),
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
    except ImageInputError as error:
        send_reply(target_id, str(error), is_group)
    except ImageRecognitionUnavailable as error:
        logger.info("Image recognition unavailable session_key=%s", session_key)
        send_reply(target_id, str(error), is_group)
    except RuntimeError as error:
        logger.exception("Configuration error")
        send_reply(target_id, f"配置还没好：{error}", is_group)
    except Exception:
        logger.exception("Message handling failed")
        send_reply(target_id, "我这边处理失败了，先缓一缓再试。", is_group)


def process_message_safely(data: dict[str, Any]) -> None:
    try:
        process_message(data)
    except Exception:
        logger.exception("Background message processing failed")


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
    return {
        "status": "ok",
        "bot_name": config.bot_name,
        "gemini_configured": bool(config.gemini_api_key),
        "deepseek_configured": bool(config.deepseek_api_key),
        "onebot_url": config.onebot_url,
        "require_group_at": config.require_group_at,
    }


def run() -> None:
    startup()
    logger.info("Starting %s on %s:%s", config.bot_name, config.host, config.port)
    app.run(host=config.host, port=config.port)


if __name__ == "__main__":
    run()
