import sys

from src.model_config import ModelConfigurationError


def load_application():
    from src import main

    return main


def main() -> int:
    try:
        application = load_application()
    except ModelConfigurationError as error:
        print(f"模型配置错误：{error}", file=sys.stderr)
        print(
            "正确格式：CHAT_MODELS=gemini:模型名,deepseek:模型名",
            file=sys.stderr,
        )
        return 2

    application.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
