import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

import requests

from tests.runtime import (
    REPOSITORY_ROOT,
    REPOSITORY_DEFAULT_MEMORY_DB,
    guarded_http_request,
    guarded_sqlite_connect,
    isolated_runtime,
)


class TestRuntimeTests(unittest.TestCase):
    def test_inherited_data_dir_is_replaced_by_controlled_test_root(self):
        with tempfile.TemporaryDirectory() as root:
            inherited_data_dir = Path(root) / "sentinel-real-data"
            inherited_data_dir.mkdir()
            sentinel = inherited_data_dir / "DO_NOT_TOUCH.txt"
            sentinel.write_text("untouched", encoding="utf-8")
            script = (
                "from pathlib import Path\n"
                "import tests\n"
                "from src.config import config\n"
                f"inherited = Path({str(inherited_data_dir)!r}).resolve()\n"
                "assert config.data_dir.resolve() != inherited, "
                "(config.data_dir, inherited)\n"
                "print(config.data_dir)\n"
            )
            environment = os.environ.copy()
            environment["DATA_DIR"] = str(inherited_data_dir)

            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=REPOSITORY_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(
                0,
                result.returncode,
                msg=f"stdout={result.stdout}\nstderr={result.stderr}",
            )
            self.assertEqual("untouched", sentinel.read_text(encoding="utf-8"))
            self.assertEqual(
                {"DO_NOT_TOUCH.txt"},
                {entry.name for entry in inherited_data_dir.iterdir()},
            )

    def test_repository_default_memory_database_is_blocked_before_open(self):
        def unexpected_open(*_args, **_kwargs):
            raise RuntimeError("database open escaped the test guard")

        with self.assertRaisesRegex(
            AssertionError,
            "repository default memory database",
        ):
            guarded_sqlite_connect(
                unexpected_open,
                REPOSITORY_DEFAULT_MEMORY_DB,
            )

    def test_external_transport_is_blocked_before_network(self):
        def unexpected_request(*_args, **_kwargs):
            raise RuntimeError("provider request escaped the test guard")

        with self.assertRaisesRegex(
            AssertionError,
            "mock the external transport boundary",
        ):
            guarded_http_request(
                unexpected_request,
                requests.Session(),
                "POST",
                "https://generativelanguage.googleapis.com/v1/models/test",
            )

    def test_isolated_runtime_stops_tracked_memory_workers(self):
        from src.memory.service import MemoryService

        with isolated_runtime() as runtime:
            service = runtime.track_service(MemoryService())
            service.start(worker_count=0)
            data_dir = runtime.config.data_dir
            self.assertEqual(runtime.data_dir, data_dir)
            self.assertTrue(service._running)

        self.assertFalse(service._running)
        self.assertEqual([], service._workers)


if __name__ == "__main__":
    unittest.main()
