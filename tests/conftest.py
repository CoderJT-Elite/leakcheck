# AI-written by Antigravity (Gemini) for John Tewolde, 2026-11-04. John must read and understand it before using it.
"""
Pytest configuration and wall-clock timeout wrapper for leakage_tool test suite.
"""

import sys
import time
from pathlib import Path
import pytest

# Ensure both repo root (parent of leakage_tool) and tool root (leakage_tool itself) are on sys.path
_this_dir = Path(__file__).resolve().parent
_tool_root = _this_dir.parent
_parent_root = _tool_root.parent

for _p in [str(_parent_root), str(_tool_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture(autouse=True)
def per_test_timeout_wrapper(request):
    """
    Wall-clock timeout wrapper ensuring no test runs indefinitely.
    Fails the test if execution exceeds 30.0 seconds.
    """
    start_time = time.monotonic()
    yield
    elapsed = time.monotonic() - start_time
    max_timeout = 30.0  # seconds
    if elapsed > max_timeout:
        pytest.fail(f"Test '{request.node.name}' exceeded wall-clock timeout: {elapsed:.2f}s > {max_timeout:.1f}s")
