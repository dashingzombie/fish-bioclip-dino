"""Controlled BioCLIP text conditions and prototype ensembles."""

from __future__ import annotations

import re
from typing import Any

import torch
import torch.nn.functional as F

from fish_vlm.data.descriptions import clean_description
from fish_vlm.data.taxonomy import genus_for_species
from fish_vlm.models.bioclip import encode_bioclip_text

PROMPT_CONDITIONS = (
    "scientific_name",
    "taxonomic_hierarchy",
    "morphology_only",
    "morphology_taxonomy",
    "full_description",
)


def _description_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("description", "text", "morphology", "full_description"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate
    return str(value)


def _morphology(
    species: str,
    raw_description: Any,
    *,
    family: str | None = None,
) -> str:
    audit = clean_description(species, _description_text(raw_description))
    prefix = (
        f"A photograph of {species}, a fish species in the genus "
        f"{genus_for_species(species)}."
    )
    morphology = re.sub(
        r"\s+",
        " ",
        audit.canonical_prompt.removeprefix(prefix).strip(),
    )
    morphology = re.sub(
        re.escape(species),
        "this fish",
        morphology,
        flags=re.IGNORECASE,
    )
    for taxon in (genus_for_species(species), family):
        if taxon:
            morphology = re.sub(
                rf"\b{re.escape(taxon)}\b",
                "",
                morphology,
                flags=re.IGNORECASE,
            )
    morphology = re.sub(
        r"\b(?:genus|family)\s+[A-Za-z-]+\b",
        "",
        morphology,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", morphology).strip()


def build_prompt_conditions(
    species_names: list[str],
    descriptions: dict[str, Any],
    canonical_prompts: dict[str, str],
    *,
    family_by_species: dict[str, str] | None = None,
) -> dict[str, dict[str, str]]:
    """Build the five required prompt conditions in one deterministic order."""
    families = family_by_species or {}
    conditions = {name: {} for name in PROMPT_CONDITIONS}
    for species in species_names:
        if species not in descriptions:
            raise ValueError(f"Missing description for {species}")
        if species not in canonical_prompts:
            raise ValueError(f"Missing canonical prompt for {species}")
        genus = genus_for_species(species)
        family = families.get(species)
        taxonomy = f"{species}, a fish species in the genus {genus}"
        if family:
            taxonomy += f" and family {family}"
        morphology = _morphology(
            species,
            descriptions[species],
            family=family,
        )
        conditions["scientific_name"][species] = (
            f"A photograph of {species}."
        )
        conditions["taxonomic_hierarchy"][species] = (
            f"A photograph of {taxonomy}."
        )
        conditions["morphology_only"][species] = (
            "A photograph of a fish."
            + (f" {morphology}" if morphology else "")
        )
        conditions["morphology_taxonomy"][species] = (
            f"A photograph of {taxonomy}."
            + (f" {morphology}" if morphology else "")
        )
        conditions["full_description"][species] = canonical_prompts[species]
    return conditions


@torch.no_grad()
def encode_condition_prototypes(
    prompts: dict[str, str],
    species_names: list[str],
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    device: torch.device | str,
    batch_size: int,
) -> torch.Tensor:
    """Encode one ordered prototype matrix without training."""
    chunks: list[torch.Tensor] = []
    for start in range(0, len(species_names), batch_size):
        texts = [
            prompts[name]
            for name in species_names[start : start + batch_size]
        ]
        chunks.append(
            encode_bioclip_text(
                model, tokenizer(texts).to(device)
            )
        )
    if not chunks:
        raise ValueError("Cannot encode an empty prototype set")
    return torch.cat(chunks)


def ensemble_prototypes(
    condition_prototypes: dict[str, torch.Tensor],
    weights: dict[str, float],
) -> torch.Tensor:
    """Average prompt prototypes with explicit non-negative condition weights."""
    unknown = sorted(set(weights) - set(condition_prototypes))
    if unknown:
        raise ValueError(f"Unknown prototype conditions: {unknown}")
    positive = {
        name: float(weight)
        for name, weight in weights.items()
        if float(weight) > 0
    }
    if not positive:
        raise ValueError("At least one prototype-ensemble weight must be positive")
    reference_shape = next(iter(condition_prototypes.values())).shape
    if any(
        condition_prototypes[name].shape != reference_shape
        for name in positive
    ):
        raise ValueError("Prototype condition matrices have incompatible shapes")
    total = sum(positive.values())
    ensemble = sum(
        (weight / total) * condition_prototypes[name].float()
        for name, weight in positive.items()
    )
    return F.normalize(ensemble, dim=-1)
