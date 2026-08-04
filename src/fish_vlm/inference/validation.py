"""Strict official submission validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fish_vlm.config import data_path
from fish_vlm.data.catalog import split_filenames
from fish_vlm.training.train import ensure_partitions


def _read_json_reject_duplicates(path: str | Path) -> dict[str, Any]:
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle, object_pairs_hook=hook)
    if not isinstance(value, dict):
        raise TypeError("Submission must be a JSON object")
    return value


def validate_submission(path: str | Path, config: dict[str, Any]) -> dict[str, int]:
    """Validate exact coverage/vocabulary and the configured candidate policy."""
    submission = _read_json_reject_duplicates(path)
    test_names = split_filenames(data_path(config, "test_split"))
    unseen_names = split_filenames(data_path(config, "unseen_split"))
    expected = set(test_names) | set(unseen_names)
    actual = set(submission)
    if expected != actual:
        raise ValueError(
            f"Submission filename mismatch; missing={sorted(expected-actual)}, extra={sorted(actual-expected)}"
        )
    if not all(isinstance(value, str) for value in submission.values()):
        raise TypeError("Every prediction must be a species string")
    partitions = ensure_partitions(config)
    vocabulary = set(partitions.all_species)
    invalid = sorted({value for value in submission.values() if value not in vocabulary})
    if invalid:
        raise ValueError(f"Predictions outside all_classes.pkl: {invalid}")
    if not config.get("inference", {}).get("generalised_enabled", False):
        bad_test = sorted(name for name in test_names if submission[name] not in set(partitions.seen_species))
        bad_unseen = sorted(name for name in unseen_names if submission[name] not in set(partitions.unseen_species))
        if bad_test or bad_unseen:
            raise ValueError(f"Candidate partition violations; test={bad_test}, unseen={bad_unseen}")
    return {"test_images": len(test_names), "unseen_images": len(unseen_names), "total_images": len(expected)}
