"""Run the repository's stdlib unittest suite."""

from __future__ import annotations

import unittest
from pathlib import Path


def main() -> int:
    tests = Path(__file__).resolve().parent.parent / "tests"
    suite = unittest.defaultTestLoader.discover(str(tests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
