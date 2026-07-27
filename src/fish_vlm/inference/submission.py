"""Submission merge with collision protection."""

from __future__ import annotations

from pathlib import Path

from fish_vlm.utils.io import read_json, write_json


def merge_predictions(
    test_path: str | Path,
    unseen_path: str | Path,
    output_path: str | Path,
) -> dict[str, str]:
    """Merge official split predictions, rejecting filename overlap."""
    test = read_json(test_path)
    unseen = read_json(unseen_path)
    if not isinstance(test, dict) or not isinstance(unseen, dict):
        raise TypeError("Prediction files must be JSON objects")
    duplicate = sorted(set(test) & set(unseen))
    if duplicate:
        raise ValueError(f"Duplicate filenames across prediction files: {duplicate}")
    merged = {**test, **unseen}
    write_json(output_path, merged)
    return merged

