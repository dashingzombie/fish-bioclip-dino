import pytest

from fish_vlm.data.pseudo_unseen import (
    assert_no_pseudo_unseen_leakage,
    make_pseudo_unseen_split,
)


def test_pseudo_unseen_is_deterministic_and_leakage_safe() -> None:
    species = [f"Genus{i // 2} species{i}" for i in range(10)]
    first = make_pseudo_unseen_split(
        species, strategy="species_holdout", holdout_fraction=0.2, seed=42
    )
    second = make_pseudo_unseen_split(
        list(reversed(species)), strategy="species_holdout", holdout_fraction=0.2, seed=42
    )
    assert first == second
    assert not set(first.training_species) & set(first.evaluation_species)
    labels = {"a.jpg": first.evaluation_species[0]}
    with pytest.raises(ValueError, match="leaked"):
        assert_no_pseudo_unseen_leakage(["a.jpg"], labels, first.evaluation_species)
    genus = make_pseudo_unseen_split(
        species, strategy="genus_holdout", holdout_fraction=0.2, seed=7
    )
    held_genera = {name.split()[0] for name in genus.evaluation_species}
    assert not any(name.split()[0] in held_genera for name in genus.training_species)

