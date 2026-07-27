"""Atomic phased-sweep state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fish_vlm.utils.io import read_json, write_json


def load_state(path: str | Path) -> dict[str, Any]:
    """Load state or return a new manifest."""
    state_path = Path(path)
    return read_json(state_path) if state_path.exists() else {"version": 1, "phases": {}}


def save_state(path: str | Path, state: dict[str, Any]) -> None:
    """Atomically persist sweep state."""
    write_json(path, state)

