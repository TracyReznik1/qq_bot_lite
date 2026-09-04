from src.chat.chat_service import generate_reply
from src.commands import CommandContext
from src.search.simple.models import SearchMode


def skip_reply(query: str, context: CommandContext) -> str:
    text = " ".join(str(query or "").split())
    if not text and not context.image_data_urls:
        return "用法：/skip <内容>，也可以附带图片。"
    return generate_reply(
        context.memory_context,
        text,
        list(context.image_data_urls),
        mode=SearchMode.SKIP,
        history_text=context.raw_message,
    )
