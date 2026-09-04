from src.persona import get_persona


def help_text() -> str:
    persona = get_persona()
    return (
        f"我是 {persona.name}，默认进行日常对话，必要时自动联网检索或解读网页。\n\n"
        "常用指令：\n"
        "/search <关键词>：强制在线搜索并汇总回答\n"
        "/skip [问题]：跳过在线搜索，直接回答\n"
        "/remember <内容>：保存你的个人偏好或专属记忆\n"
        "/globalremember <内容>：保存全局共享记忆（需管理员）\n"
        "/memories [查询词]：查看或检索已保存的记忆\n"
        "/forget <记忆ID或内容>：删除指定记忆\n"
        "/reset：清空当前会话上下文历史\n"
        "/help：查看本帮助信息\n\n"
        "能力与限制：\n"
        "• 群聊中默认需要 @ 我触发。\n"
        "• 聊天中包含网页链接（URL）时，会自动读取网页正文并进行总结或解答。\n"
        "• 可以直接发送图片，或发送图片加文字，由模型识别并回答。\n"
        "• 不支持图片生成、图片编辑、主动发图、视频理解，也不支持天气或B站等独立工具插件。"
    )
