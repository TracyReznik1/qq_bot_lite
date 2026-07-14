from src.chat.chat_service import generate_reply
from src.services.search_service import has_search_results, normalize_search_query, search


def search_reply(query: str, session_key: str, raw_message: str) -> str:
    query = normalize_search_query(query)
    if not query:
        return "想搜什么？比如：/search DeepSeek 最新消息"

    search_result = search(query)
    if not has_search_results(search_result):
        tool_context = (
            "这是 /search 命令的搜索失败结果。请按当前角色设定回答用户："
            "说明没有搜到可靠结果，所以不知道或无法确认；不要猜测，不要编造成确定事实。\n"
            f"搜索状态：\n{search_result.text}"
        )
    else:
        tool_context = f"网页搜索结果：\n{search_result.text}"

    return generate_reply(session_key, raw_message, tool_context)
