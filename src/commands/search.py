from src.chat.chat_service import generate_reply
from src.commands import CommandContext
from src.search.simple.models import SearchMode
from src.services.search_service import normalize_search_query


def search_reply(query: str, context: CommandContext) -> str:
    normalized = normalize_search_query(query)
    if not normalized and not context.image_data_urls:
        return "想搜什么？比如：/search DeepSeek 最新消息，也可以附带图片。"
    return generate_reply(
        context.memory_context,
        normalized,
        list(context.image_data_urls),
        mode=SearchMode.STANDARD,
        history_text=context.raw_message,
    )
