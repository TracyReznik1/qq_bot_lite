import sys

from src import main as _main


if __name__ == "__main__":
    _main.run()
else:
    sys.modules[__name__] = _main
