"""Safe filesystem and serialisation helpers."""

from __future__ import annotations

import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any


def ensure_parent(path: str | Path) -> Path:
    """Create the parent directory and return a normalised path."""
    result = Path(path)
    result.parent.mkdir(parents=True, exist_ok=True)
    return result


def atomic_write_text(path: str | Path, text: str) -> None:
    """Atomically replace a UTF-8 text file."""
    destination = ensure_parent(path)
    fd, temporary = tempfile.mkstemp(dir=destination.parent, prefix=f".{destination.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_json(path: str | Path, value: Any, *, indent: int = 2) -> None:
    """Write JSON deterministically and atomically."""
    atomic_write_text(path, json.dumps(value, indent=indent, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: str | Path) -> Any:
    """Read a JSON document."""
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def read_pickle(path: str | Path) -> Any:
    """Read a trusted local pickle supplied by the competition organiser."""
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def torch_save_atomic(value: Any, path: str | Path) -> None:
    """Atomically save a PyTorch object without importing torch at module import."""
    import torch

    destination = ensure_parent(path)
    fd, temporary = tempfile.mkstemp(dir=destination.parent, prefix=f".{destination.name}.")
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise

