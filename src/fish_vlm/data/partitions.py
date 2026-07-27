"""Seen/unseen class partition construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fish_vlm.utils.io import read_json, read_pickle, write_json


@dataclass(frozen=True)
class ClassPartitions:
    """Deterministically ordered species partitions and mappings."""

    seen_species: list[str]
    unseen_species: list[str]
    all_species: list[str]

    @staticmethod
    def _mappings(names: list[str], prefix: str) -> dict[str, Any]:
        return {
            f"{prefix}_species_to_index": {name: index for index, name in enumerate(names)},
            f"index_to_{prefix}_species": {str(index): name for index, name in enumerate(names)},
        }

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "seen_species": self.seen_species,
            "unseen_species": self.unseen_species,
            "all_species": self.all_species,
        }
        result.update(self._mappings(self.seen_species, "seen"))
        result.update(self._mappings(self.unseen_species, "unseen"))
        result.update(self._mappings(self.all_species, "all"))
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ClassPartitions":
        return cls(
            seen_species=list(value["seen_species"]),
            unseen_species=list(value["unseen_species"]),
            all_species=list(value["all_species"]),
        )


def build_class_partitions(labels: dict[str, str], all_classes: list[str]) -> ClassPartitions:
    """Build partitions, rejecting labels outside the official vocabulary."""
    vocabulary = {str(name) for name in all_classes}
    seen = {str(name) for name in labels.values()}
    unknown = sorted(seen - vocabulary)
    if unknown:
        raise ValueError(f"Labelled species absent from all_classes.pkl: {unknown}")
    all_species = sorted(vocabulary)
    return ClassPartitions(
        seen_species=sorted(seen),
        unseen_species=sorted(vocabulary - seen),
        all_species=all_species,
    )


def create_and_save_partitions(
    labels_path: str | Path,
    all_classes_path: str | Path,
    output_path: str | Path,
) -> ClassPartitions:
    """Read organiser files, derive partitions and persist the complete mapping."""
    labels = read_json(labels_path)
    all_classes = read_pickle(all_classes_path)
    if not isinstance(labels, dict) or not isinstance(all_classes, (list, tuple, set)):
        raise TypeError("Labels must be a JSON object and all classes must be a sequence")
    partitions = build_class_partitions(labels, list(all_classes))
    write_json(output_path, partitions.to_dict())
    return partitions


def load_partitions(path: str | Path) -> ClassPartitions:
    """Load persisted partitions."""
    return ClassPartitions.from_dict(read_json(path))

