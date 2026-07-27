"""Submission merge with collision protection."""

from __future__ import annotations

import os
import tempfile
import zipfile
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


def package_submission(
    submission_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Create a deterministic ZIP containing only ``prediction.json``."""
    source = Path(submission_path)
    destination = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(f"Submission JSON does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    )
    os.close(descriptor)
    try:
        info = zipfile.ZipInfo("prediction.json", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        with zipfile.ZipFile(
            temporary_name,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            archive.writestr(info, source.read_bytes())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination
