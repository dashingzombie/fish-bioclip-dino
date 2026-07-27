from __future__ import annotations

import json
from pathlib import Path

import pytest

from fish_vlm.data.descriptions import clean_description, prepare_canonical_prompts


def test_prompt_cleaning_removes_nonvisual_and_vague_sentences() -> None:
    audit = clean_description(
        "Oreochromis mossambicus",
        "It has a compressed silver body and dark stripes. It inhabits rivers. "
        "The eyes appear normal. Males develop pointed fins.",
    )
    assert "compressed silver body" in audit.canonical_prompt
    assert "pointed fins" in audit.canonical_prompt
    assert "inhabits rivers" not in audit.canonical_prompt
    assert len(audit.removed_sentences) == 2


def test_manual_override_wins_and_missing_description_fails(tmp_path: Path) -> None:
    descriptions = tmp_path / "descriptions.json"
    overrides = tmp_path / "overrides.json"
    descriptions.write_text(json.dumps({"A fish": "It is blue."}), encoding="utf-8")
    overrides.write_text(json.dumps({"A fish": "Curated visible prompt."}), encoding="utf-8")
    prompts = prepare_canonical_prompts(
        descriptions, overrides, tmp_path / "prompts.json", tmp_path / "audit.jsonl",
        expected_species=["A fish"],
    )
    assert prompts["A fish"] == "Curated visible prompt."
    audit = json.loads((tmp_path / "audit.jsonl").read_text().strip())
    assert "manual_override" in audit["warnings"]
    with pytest.raises(ValueError, match="Missing descriptions"):
        prepare_canonical_prompts(
            descriptions, overrides, tmp_path / "other.json", tmp_path / "other.jsonl",
            expected_species=["A fish", "Missing fish"],
        )

