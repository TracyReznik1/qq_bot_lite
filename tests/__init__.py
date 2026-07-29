import atexit
import os
import tempfile


os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ["CHAT_MODELS"] = "gemini:test-gemini"
os.environ["MEMORY_MODELS"] = ""
os.environ["GEMINI_API_KEY"] = "test-gemini-key"

_temporary_data_root = tempfile.TemporaryDirectory(
    prefix="qqbot-lite-tests-"
)
atexit.register(_temporary_data_root.cleanup)
os.environ["DATA_DIR"] = _temporary_data_root.name

from tests.runtime import install_test_guards


install_test_guards()
