"""Leakage-safe class-level pseudo-unseen splits."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fish_vlm.utils.hashing import stable_json_hash
from fish_vlm.utils.io import write_json


@dataclass(frozen=True)
class PseudoUnseenSplit:
    """Class partition used for zero-shot validation."""

    strategy: str
    seed: int
    training_species: list[str]
    evaluation_species: list[str]

    @property
    def split_hash(self) -> str:
        return stable_json_hash(
            {
                "strategy": self.strategy,
                "seed": self.seed,
                "training_species": self.training_species,
                "evaluation_species": self.evaluation_species,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "seed": self.seed,
            "training_species": self.training_species,
            "evaluation_species": self.evaluation_species,
            "training_species_hash": stable_json_hash(self.training_species),
            "split_hash": self.split_hash,
        }


def make_pseudo_unseen_split(
    seen_species: list[str],
    *,
    strategy: str,
    holdout_fraction: float,
    seed: int,
) -> PseudoUnseenSplit:
    """Create deterministic species- or genus-level holdout partitions."""
    names = sorted(set(seen_species))
    if len(names) < 2:
        raise ValueError("At least two seen species are required")
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between zero and one")
    rng = random.Random(seed)
    if strategy == "species_holdout":
        shuffled = names.copy()
        rng.shuffle(shuffled)
        count = max(1, min(len(names) - 1, round(len(names) * holdout_fraction)))
        evaluation = sorted(shuffled[:count])
    elif strategy == "genus_holdout":
        genera = sorted({name.split()[0] for name in names})
        if len(genera) < 2:
            raise ValueError("Genus holdout requires at least two genera")
        rng.shuffle(genera)
        count = max(1, min(len(genera) - 1, round(len(genera) * holdout_fraction)))
        held_genera = set(genera[:count])
        evaluation = [name for name in names if name.split()[0] in held_genera]
    else:
        raise ValueError("strategy must be species_holdout or genus_holdout")
    training = sorted(set(names) - set(evaluation))
    if set(training) & set(evaluation) or set(training) | set(evaluation) != set(names):
        raise RuntimeError("Pseudo-unseen split invariant violated")
    return PseudoUnseenSplit(strategy, seed, training, sorted(evaluation))


def assert_no_pseudo_unseen_leakage(
    training_filenames: list[str],
    labels: dict[str, str],
    evaluation_species: list[str],
) -> None:
    """Reject any loader input containing held-out species."""
    leaked = sorted(
        {
            labels[name]
            for name in training_filenames
            if name in labels and labels[name] in set(evaluation_species)
        }
    )
    if leaked:
        raise ValueError(f"Pseudo-unseen species leaked into training: {leaked}")


def save_pseudo_unseen_splits(
    seen_species: list[str],
    output_dir: str | Path,
    *,
    strategy: str,
    holdout_fraction: float,
    seeds: list[int],
) -> list[PseudoUnseenSplit]:
    """Create and persist each configured class-level split."""
    output = Path(output_dir)
    splits = [
        make_pseudo_unseen_split(
            seen_species, strategy=strategy, holdout_fraction=holdout_fraction, seed=seed
        )
        for seed in seeds
    ]
    for split in splits:
        write_json(output / f"{strategy}_seed_{split.seed}.json", split.to_dict())
    return splits

