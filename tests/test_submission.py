from __future__ import annotations

import json
import pickle
import zipfile
from pathlib import Path

import pytest

from fish_vlm.inference.submission import merge_predictions, package_submission
from fish_vlm.inference.validation import validate_submission


def _config(root: Path) -> dict:
    return {
        "data": {
            "root_dir": str(root),
            "labels_json": "label_train.json",
            "all_classes_pickle": "all_classes.pkl",
            "test_split": "splits/test.pkl",
            "unseen_split": "splits/unseen.pkl",
            "processed_dir": "processed",
        }
    }


def test_submission_merge_validation_and_duplicate_detection(tmp_path: Path) -> None:
    (tmp_path / "splits").mkdir()
    (tmp_path / "label_train.json").write_text(json.dumps({"train.jpg": "Seen fish"}))
    with (tmp_path / "all_classes.pkl").open("wb") as handle:
        pickle.dump(["Seen fish", "Unseen fish"], handle)
    with (tmp_path / "splits/test.pkl").open("wb") as handle:
        pickle.dump(["test.jpg"], handle)
    with (tmp_path / "splits/unseen.pkl").open("wb") as handle:
        pickle.dump(["unseen.jpg"], handle)
    test = tmp_path / "test.json"
    unseen = tmp_path / "unseen.json"
    test.write_text(json.dumps({"test.jpg": "Seen fish"}))
    unseen.write_text(json.dumps({"unseen.jpg": "Unseen fish"}))
    output = tmp_path / "prediction.json"
    merge_predictions(test, unseen, output)
    assert validate_submission(output, _config(tmp_path))["total_images"] == 2
    output.write_text('{"test.jpg":"Seen fish","test.jpg":"Seen fish","unseen.jpg":"Unseen fish"}')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        validate_submission(output, _config(tmp_path))


def test_merge_rejects_cross_split_collision(tmp_path: Path) -> None:
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    first.write_text('{"same.jpg":"A"}')
    second.write_text('{"same.jpg":"B"}')
    with pytest.raises(ValueError, match="Duplicate filenames"):
        merge_predictions(first, second, tmp_path / "out.json")


def test_submission_zip_is_deterministic_and_has_one_root_file(
    tmp_path: Path,
) -> None:
    submission = tmp_path / "prediction.json"
    submission.write_text('{"fish.jpg":"A fish"}\n', encoding="utf-8")
    first = package_submission(submission, tmp_path / "first.zip")
    second = package_submission(submission, tmp_path / "second.zip")
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == ["prediction.json"]
        assert archive.read("prediction.json") == submission.read_bytes()
