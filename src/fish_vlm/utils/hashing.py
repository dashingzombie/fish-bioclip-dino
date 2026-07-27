"""Stable hashes used for scientific compatibility checks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


def stable_json_hash(value: Any) -> str:
    """Return a SHA-256 hash of canonical JSON."""
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ordered_names_hash(names: Sequence[str]) -> str:
    """Hash an ordered class list."""
    return stable_json_hash(list(names))


def prompts_hash(prompts: Mapping[str, str], species_names: Sequence[str]) -> str:
    """Hash prompts in explicit species order."""
    return stable_json_hash([{"species": name, "prompt": prompts[name]} for name in species_names])

