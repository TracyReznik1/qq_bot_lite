from dataclasses import dataclass


@dataclass(frozen=True)
class Route:
    handler: str
    action: str
    query: str
    command: str = ""


def route_message(text: str) -> Route:
    normalized = text.strip()
    if not normalized:
        return Route(handler="chat", action="empty", query="")

    if not normalized.startswith("/"):
        return Route(handler="chat", action="chat", query=normalized)

    command_text = normalized[1:].strip()
    if not command_text:
        return Route(handler="command", action="command", query="", command="")

    command, _, query = command_text.partition(" ")
    command = command.lower()
    return Route(handler="command", action="command", query=query.strip(), command=command)
