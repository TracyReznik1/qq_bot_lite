from src.chat.chat_service import reset_history


def reset_context(session_key: str) -> str:
    reset_history(session_key)
    return "当前会话上下文已清空。"
