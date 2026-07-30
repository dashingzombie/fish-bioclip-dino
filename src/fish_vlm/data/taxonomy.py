"""Taxonomic metadata used by prompts, metrics, and hard-negative mining."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fish_vlm.config import data_path
from fish_vlm.utils.io import read_json

FAMILY_PATTERN = re.compile(
    r"\bfamily\s+([A-Z][A-Za-z-]*idae)\b",
    flags=re.IGNORECASE,
)


def genus_for_species(species: str) -> str:
    """Derive the genus from a binomial-or-longer scientific name."""
    parts = species.split()
    if not parts:
        raise ValueError("Species name cannot be empty")
    return parts[0]


def _family_from_description(value: Any) -> str | None:
    if isinstance(value, dict):
        family = value.get("family")
        if isinstance(family, str) and family.strip():
            return family.strip()
        text = " ".join(
            str(value.get(key, ""))
            for key in ("taxonomy", "description", "text")
        )
    else:
        text = str(value)
    match = FAMILY_PATTERN.search(text)
    return None if match is None else match.group(1)


def load_family_mapping(
    config: dict[str, Any],
    species_names: list[str],
) -> dict[str, str]:
    """Load families from explicit metadata, falling back to descriptions."""
    mapping: dict[str, str] = {}
    taxonomy_value = config.get("data", {}).get("taxonomy_json")
    if taxonomy_value:
        taxonomy_path = data_path(config, "taxonomy_json")
        if taxonomy_path.is_file():
            raw = read_json(taxonomy_path)
            if not isinstance(raw, dict):
                raise TypeError("taxonomy_json must contain a JSON object")
            for species, value in raw.items():
                if isinstance(value, str):
                    mapping[str(species)] = value
                elif isinstance(value, dict) and isinstance(
                    value.get("family"), str
                ):
                    mapping[str(species)] = value["family"]

    descriptions_path = data_path(config, "descriptions_json")
    if Path(descriptions_path).is_file():
        descriptions = read_json(descriptions_path)
        if isinstance(descriptions, dict):
            for species in species_names:
                if species in mapping:
                    continue
                family = _family_from_description(
                    descriptions.get(species, "")
                )
                if family is not None:
                    mapping[species] = family
    return {
        species: mapping[species]
        for species in species_names
        if species in mapping
    }
