from typing import Any
import logging

import requests

from src.config import Config

logger = logging.getLogger("qq-bot")


class OneBotClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.cfg.onebot_access_token:
            headers["Authorization"] = f"Bearer {self.cfg.onebot_access_token}"
        return headers

    def get_image_url(self, file_id: str) -> str:
        response = requests.post(
            f"{self.cfg.onebot_url}/get_image",
            json={"file": file_id},
            headers=self._headers(),
            timeout=self.cfg.request_timeout,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        url = str(data.get("url") or "").strip() if isinstance(data, dict) else ""
        if payload.get("retcode") != 0 or not url:
            raise RuntimeError("OneBot could not resolve the received image")
        return url

    def get_message_author(self, message_id: str) -> str | None:
        try:
            response = requests.post(
                f"{self.cfg.onebot_url}/get_msg",
                json={"message_id": message_id},
                headers=self._headers(),
                timeout=self.cfg.request_timeout,
            )
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data") if isinstance(payload, dict) else None
            sender = data.get("sender") if isinstance(data, dict) else None
            user_id = (
                sender.get("user_id")
                if isinstance(sender, dict)
                else None
            )
            author = str(user_id or "").strip()
            if (
                not isinstance(payload, dict)
                or payload.get("retcode") != 0
                or not author.isdigit()
            ):
                logger.warning(
                    "OneBot get message author failed message_id=%s error_type=%s",
                    message_id,
                    "InvalidAuthorData",
                )
                return None
            return author
        except Exception as error:
            logger.warning(
                "OneBot get message author failed message_id=%s error_type=%s",
                message_id,
                type(error).__name__,
            )
            return None

    def send_msg(self, target_id: Any, message: str, is_group: bool = False) -> None:
        message = (message or "").strip()
        if not message:
            logger.info("OneBot send skipped: empty message target_id=%s is_group=%s", target_id, is_group)
            return

        endpoint = "send_group_msg" if is_group else "send_private_msg"
        payload_key = "group_id" if is_group else "user_id"
        payload = {payload_key: target_id, "message": message}
        try:
            logger.info(
                "OneBot send request endpoint=%s target_id=%s is_group=%s chars=%s",
                endpoint,
                target_id,
                is_group,
                len(message),
            )
            response = requests.post(
                f"{self.cfg.onebot_url}/{endpoint}",
                json=payload,
                headers=self._headers(),
                timeout=self.cfg.request_timeout,
            )
            response.raise_for_status()
            logger.info(
                "OneBot send success endpoint=%s target_id=%s is_group=%s status_code=%s",
                endpoint,
                target_id,
                is_group,
                response.status_code,
            )
        except Exception:
            logger.exception(
                "OneBot send failed endpoint=%s target_id=%s is_group=%s chars=%s",
                endpoint,
                target_id,
                is_group,
                len(message),
            )
