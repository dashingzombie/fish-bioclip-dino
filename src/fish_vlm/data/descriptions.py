"""Deterministic conversion of supplied descriptions into canonical prompts."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fish_vlm.utils.io import atomic_write_text, read_json, write_json

DEFAULT_NONVISUAL_KEYWORDS = (
    "mtdna", "dna", "chromosome", "genetic marker", "distribution", "distributed",
    "native to", "habitat", "inhabits", "occurs in", "ecology", "diet", "feeds on", "behaviour", "behavior",
    "conservation status", "endangered", "least concern",
)
DEFAULT_VAGUE_PATTERNS = (
    r"\beyes? (?:appear|appears|are|is) normal\b",
    r"\bmouth (?:appear|appears|are|is) normal\b",
    r"\bno distinctive features?\b",
    r"\bfeatures? (?:are|is) unnoted\b",
    r"\bthese characteristics assist identification\b",
)


@dataclass(frozen=True)
class PromptAudit:
    """Auditable output for one species prompt."""

    scientific_name: str
    raw_description: str
    canonical_prompt: str
    removed_sentences: list[str]
    token_count: int
    was_truncated: bool
    warnings: list[str]


def split_sentences(text: str) -> list[str]:
    """Split plain descriptions predictably without external NLP dependencies."""
    compact = re.sub(r"\s+", " ", text).strip()
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", compact) if part.strip()]


def clean_description(
    scientific_name: str,
    raw_description: str,
    *,
    nonvisual_keywords: tuple[str, ...] = DEFAULT_NONVISUAL_KEYWORDS,
    vague_patterns: tuple[str, ...] = DEFAULT_VAGUE_PATTERNS,
    max_tokens: int = 220,
) -> PromptAudit:
    """Filter non-visual/vague sentences and build exactly one prompt."""
    if not scientific_name.strip():
        raise ValueError("Scientific name cannot be empty")
    genus = scientific_name.split()[0]
    kept: list[str] = []
    removed: list[str] = []
    for sentence in split_sentences(raw_description):
        lower = sentence.casefold()
        if any(keyword.casefold() in lower for keyword in nonvisual_keywords) or any(
            re.search(pattern, lower) for pattern in vague_patterns
        ):
            removed.append(sentence)
        else:
            kept.append(sentence)
    warnings: list[str] = []
    if not kept:
        warnings.append("no_visual_description_retained")
    prefix = f"A photograph of {scientific_name}, a fish species in the genus {genus}."
    body = " ".join(kept)
    words = (prefix + (" " + body if body else "")).split()
    was_truncated = len(words) > max_tokens
    if was_truncated:
        words = words[:max_tokens]
        warnings.append("truncated_to_token_limit")
    canonical = " ".join(words)
    return PromptAudit(
        scientific_name=scientific_name,
        raw_description=raw_description,
        canonical_prompt=canonical,
        removed_sentences=removed,
        token_count=len(words),
        was_truncated=was_truncated,
        warnings=warnings,
    )


def prepare_canonical_prompts(
    descriptions_path: str | Path,
    overrides_path: str | Path,
    prompts_output: str | Path,
    audit_output: str | Path,
    *,
    expected_species: list[str] | None = None,
    max_tokens: int = 220,
) -> dict[str, str]:
    """Prepare prompts, applying manually authored full-prompt overrides last."""
    descriptions = read_json(descriptions_path)
    if not isinstance(descriptions, dict):
        raise TypeError("descriptions_all.json must be a JSON object")
    overrides_file = Path(overrides_path)
    overrides = read_json(overrides_file) if overrides_file.exists() else {}
    if not isinstance(overrides, dict) or not all(isinstance(v, str) for v in overrides.values()):
        raise TypeError("Prompt overrides must map species to prompt strings")
    species = sorted(expected_species if expected_species is not None else descriptions)
    missing = sorted(set(species) - set(descriptions) - set(overrides))
    if missing:
        raise ValueError(f"Missing descriptions for species: {missing}")
    unknown_overrides = sorted(set(overrides) - set(species))
    if expected_species is not None and unknown_overrides:
        raise ValueError(f"Overrides contain unknown species: {unknown_overrides}")
    prompts: dict[str, str] = {}
    audits: list[PromptAudit] = []
    for name in species:
        raw = str(descriptions.get(name, ""))
        audit = clean_description(name, raw, max_tokens=max_tokens)
        if name in overrides:
            override = re.sub(r"\s+", " ", overrides[name]).strip()
            if not override:
                raise ValueError(f"Manual override is empty for {name}")
            audit = PromptAudit(
                scientific_name=name,
                raw_description=raw,
                canonical_prompt=override,
                removed_sentences=audit.removed_sentences,
                token_count=len(override.split()),
                was_truncated=False,
                warnings=[*audit.warnings, "manual_override"],
            )
        prompts[name] = audit.canonical_prompt
        audits.append(audit)
    write_json(prompts_output, prompts)
    lines = "".join(json.dumps(asdict(item), ensure_ascii=False, sort_keys=True) + "\n" for item in audits)
    atomic_write_text(audit_output, lines)
    return prompts
