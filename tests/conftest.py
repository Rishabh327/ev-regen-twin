"""Shared pytest fixtures. Adds the project root to sys.path so `core` and `sdv` import."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import load_config  # noqa: E402
from core.drive_cycles import load_cycle  # noqa: E402


@pytest.fixture(scope="session")
def cfg() -> dict[str, Any]:
    return load_config()


@pytest.fixture(scope="session")
def wltp():
    return load_cycle("WLTP")


@pytest.fixture(scope="session")
def city():
    return load_cycle("CITY")
