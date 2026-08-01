from src.chat.chat_service import generate_reply
from src.services.search_service import normalize_search_query


def search_reply(query: str, session_key: str, raw_message: str) -> str:
    normalized = normalize_search_query(query)
    if not normalized:
        return "想搜什么？比如：/search DeepSeek 最新消息"
    return generate_reply(
        session_key,
        normalized,
        force_search=True,
        history_text=raw_message,
    )
