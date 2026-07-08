from src.chat.chat_service import generate_reply
from src.services.search_service import has_search_results, search
from src.services.search_service import normalize_search_query
from src.services.url_fetch_service import extract_first_url, fetch_url


def search_reply(query: str, session_key: str, raw_message: str) -> str:
    """qqbot_lite: /search command — web search with optional URL fetch fallback."""
    query = normalize_search_query(query)
    if not query:
        return "想搜什么？比如：/search DeepSeek 最新消息"

    # If the query contains a URL, fetch it directly as an internal convenience.
    # The /url command itself is not registered, so users cannot trigger URL
    # fetching through a dedicated command.
    if extract_first_url(query):
        fetch_result = fetch_url(query)
        if fetch_result.ok:
            tool_context = (
                "这是 /search 命令收到 URL 后的直读结果。请按 ATRI 的角色设定回答用户："
                "结合网页标题和正文摘录总结重点；如果摘录不足以支撑结论，要明确说明不确定。\n"
                f"URL 直读结果：\n{fetch_result.text}"
            )
        else:
            tool_context = (
                "这是 /search 命令收到 URL 后的读取失败结果。请按 ATRI 的角色设定回答用户："
                "说明网页没有读到可靠内容，所以无法确认；不要猜测，不要编造页面内容。\n"
                f"URL 直读结果：\n{fetch_result.text}"
            )
        return generate_reply(session_key, raw_message, tool_context)

    search_result = search(query)
    if not has_search_results(search_result):
        tool_context = (
            "这是 /search 命令的搜索失败结果。请按 ATRI 的角色设定回答用户："
            "说明没有搜到可靠结果，所以不知道或无法确认；不要猜测，不要编造成确定事实。\n"
            f"搜索状态：\n{search_result.text}"
        )
    else:
        tool_context = f"网页搜索结果：\n{search_result.text}"

    return generate_reply(session_key, raw_message, tool_context)
