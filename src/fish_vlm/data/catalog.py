"""Competition file catalog and split metadata normalisation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from fish_vlm.config import data_path
from fish_vlm.utils.io import read_json, read_pickle


def load_labels(config: dict[str, Any]) -> dict[str, str]:
    """Load filename-to-species labels."""
    value = read_json(data_path(config, "labels_json"))
    if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
        raise TypeError("label_train.json must map string filenames to string species")
    return value


def _filename_from_record(record: Any) -> str:
    if isinstance(record, str):
        return record
    if isinstance(record, Mapping):
        for key in ("filename", "file_name", "image", "image_id", "path"):
            if key in record:
                return Path(str(record[key])).name
    raise ValueError(f"Cannot determine filename from split record: {record!r}")


def split_filenames(path: str | Path) -> list[str]:
    """Extract ordered unique filenames from common organiser pickle formats."""
    value = read_pickle(path)
    if isinstance(value, pd.DataFrame):
        records: Iterable[Any] = value.to_dict("records")
    elif isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value):
            records = value.keys()
        else:
            records = value.values()
    elif isinstance(value, Iterable):
        records = value
    else:
        raise TypeError(f"Unsupported split object: {type(value).__name__}")
    names = [_filename_from_record(item) for item in records]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate filenames in split: {path}")
    return names


def official_split_counts(config: dict[str, Any]) -> tuple[int, int]:
    """Return actual official seen-test and unseen image counts."""
    return (
        len(split_filenames(data_path(config, "test_split"))),
        len(split_filenames(data_path(config, "unseen_split"))),
    )

