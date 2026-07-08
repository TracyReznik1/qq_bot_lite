from src.config import config


def help_text() -> str:
    return (
        f"我是 {config.bot_name}（qqbot_lite 严格版），默认先聊天，必要时才会联网搜索。\n"
        "用法示例：\n"
        "你好\n"
        "kskbl 是什么意思\n"
        "/search DeepSeek 最新消息\n"
        "/remember 我喜欢简洁回答\n"
        "/globalremember 所有人都知道的设定（管理员）\n"
        "/reset\n"
        "/help\n"
        "群聊里默认需要 @ 我。\n"
        "lite 版只保留联网搜索能力，不包含天气、B站、URL直读、视频理解、图片生成等功能。"
    )
