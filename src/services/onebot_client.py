from typing import Any
import logging

import requests
from pathlib import Path

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

    def send_image(self, target_id: Any, image_path: str, is_group: bool = False) -> None:
        """Send an image via CQ image ``file://`` URI.

        First edition uses ``[CQ:image,file=<uri>]``.  Some OneBot clients
        (e.g. NapCat) may not support ``file://`` URIs — in that case a
        base64 fallback should be added in a future iteration.
        """
        resolved = Path(image_path).resolve()
        if not resolved.exists():
            logger.warning(
                "OneBot image send skipped: file not found path=%s", resolved
            )
            return
        uri = resolved.as_uri()
        message = f"[CQ:image,file={uri}]"
        self.send_msg(target_id, message, is_group=is_group)
