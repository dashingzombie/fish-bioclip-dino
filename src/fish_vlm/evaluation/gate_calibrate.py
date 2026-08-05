"""Runtime collection and calibration for the DINO/BioCLIP confidence gate."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from fish_vlm.config import data_path
from fish_vlm.data.catalog import (
    load_labels,
    official_split_counts,
    split_filenames,
)
from fish_vlm.evaluation.calibration import fit_temperature
from fish_vlm.evaluation.gating import (
    fit_confidence_gate,
    threshold_for_acceptance_rate,
)
from fish_vlm.inference.bioclip_checkpoint import (
    load_finetuned_bioclip_visual,
)
from fish_vlm.training.checkpoint import (
    checkpoint_training_species,
    load_checkpoint,
)
from fish_vlm.training.distributed import DistributedContext
from fish_vlm.training.train import (
    _data_processed_path,
    _pseudo_split_path,
    _split_labelled_filenames,
    build_runtime,
    load_candidate_prototypes,
    make_loader,
)
from fish_vlm.utils.hashing import prompts_hash, stable_json_hash
from fish_vlm.utils.io import read_json, write_json


def validate_hybrid_checkpoint(checkpoint: dict[str, Any]) -> None:
    """Reject checkpoints that could have modified the BioCLIP fallback."""
    resolved = checkpoint.get("resolved_configuration")
    if not isinstance(resolved, dict):
        raise ValueError("Hybrid checkpoint lacks its resolved configuration")
    if resolved.get("training", {}).get("stage") != "dino_seen_classifier":
        raise ValueError("Hybrid checkpoint was not trained as a DINO classifier")
    model = resolved.get("model", {})
    if (
        model.get("dino", {}).get("trainable_scope") != "full"
        or model.get("tuning_mode", "frozen") != "frozen"
        or not model.get("bioclip", {}).get("freeze_image_encoder", True)
        or not model.get("bioclip", {}).get("freeze_text_encoder", True)
    ):
        raise ValueError(
            "Hybrid checkpoint does not prove full-DINO/frozen-BioCLIP training"
        )
    if checkpoint.get("active_losses") != ["supervised_species"]:
        raise ValueError(
            "Hybrid checkpoint must use only supervised seen-species loss"
        )


def _threshold_grid(config: dict[str, Any]) -> list[float]:
    gate = config.get("hybrid_gate", {})
    if "thresholds" in gate:
        values = [float(value) for value in gate["thresholds"]]
    else:
        minimum = float(gate.get("threshold_min", 0.0))
        maximum = float(gate.get("threshold_max", 1.0))
        steps = int(gate.get("threshold_steps", 201))
        if steps < 2:
            raise ValueError("hybrid_gate.threshold_steps must be at least 2")
        values = [
            minimum + (maximum - minimum) * index / (steps - 1)
            for index in range(steps)
        ]
    if not values or min(values) < 0.0 or max(values) > 1.0:
        raise ValueError("Every configured gate threshold must be in [0, 1]")
    return values


@torch.no_grad()
def _collect_gate_validation(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    bioclip_checkpoint_path: str | Path | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[str],
    list[str],
    dict[str, Any],
    dict[str, Any] | None,
]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = build_runtime(config, device=device)
    prototypes, candidate_species, cache = load_candidate_prototypes(
        config, bundle, "seen", device=device
    )
    canonical_prompts = read_json(
        _data_processed_path(config, "canonical_prompts.json")
    )
    canonical_prompt_hash = prompts_hash(
        canonical_prompts, bundle.partitions.all_species
    )
    checkpoint = load_checkpoint(
        checkpoint_path,
        bundle.model,
        expected_seen_species=bundle.partitions.seen_species,
        expected_unseen_species=bundle.partitions.unseen_species,
        expected_text_prototype_hash=cache["prompt_hash"],
        expected_canonical_prompt_hash=canonical_prompt_hash,
        expected_dino_model_name=str(config["model"]["dino"]["name"]),
        expected_dino_checkpoint_source=bundle.dino_source,
        expected_bioclip_checkpoint=bundle.bioclip_checkpoint,
        strict=False,
    )
    validate_hybrid_checkpoint(checkpoint)
    supervised_species = checkpoint_training_species(
        checkpoint, seen_species=bundle.partitions.seen_species
    )
    finetuned_bioclip = None
    if bioclip_checkpoint_path is not None:
        if bundle.model.bioclip is None:
            raise ValueError("Fine-tuned BioCLIP fallback is unavailable")
        finetuned_bioclip = load_finetuned_bioclip_visual(
            bioclip_checkpoint_path,
            bundle.model.bioclip,
            expected_seen_species=bundle.partitions.seen_species,
            expected_unseen_species=bundle.partitions.unseen_species,
            expected_training_species=supervised_species,
            expected_text_prototype_hash=cache["prompt_hash"],
            expected_canonical_prompt_hash=canonical_prompt_hash,
            expected_bioclip_checkpoint=bundle.bioclip_checkpoint,
        )
    labels = load_labels(config)
    _, validation_names = _split_labelled_filenames(
        split_filenames(data_path(config, "train_split")),
        labels,
        int(config["seed"]),
        float(config["validation"].get("seen_fraction", 0.1)),
    )
    loader, _ = make_loader(
        validation_names,
        config,
        bundle,
        labels,
        candidate_species,
        training=False,
        context=DistributedContext(0, 1, 0, device),
    )
    supervised_logits: list[torch.Tensor] = []
    bioclip_logits: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    bundle.model.eval()
    for batch in loader:
        output = bundle.model(
            batch["dino_image"].to(device),
            prototypes,
            batch["bioclip_image"].to(device),
        )
        if output.supervised_logits is None:
            raise ValueError("Gate calibration requires the DINO supervised head")
        if output.bioclip_logits is None:
            raise ValueError("Gate calibration requires frozen native BioCLIP")
        supervised_logits.append(output.supervised_logits.cpu())
        bioclip_logits.append(output.bioclip_logits.cpu())
        targets.append(batch["species_index"])
    if not targets:
        raise ValueError("Gate calibration validation split is empty")
    return (
        torch.cat(supervised_logits),
        torch.cat(bioclip_logits),
        torch.cat(targets),
        candidate_species,
        supervised_species,
        checkpoint,
        finetuned_bioclip,
    )


def _read_gate(path: str | Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise TypeError("Gate calibration must be a JSON object")
    supplied_hash = value.get("hash")
    unhashed = {key: item for key, item in value.items() if key != "hash"}
    if supplied_hash != stable_json_hash(unhashed):
        raise ValueError("Gate calibration hash is invalid")
    return value


def calibrate_gate_checkpoint(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    threshold_source: str | Path | None = None,
    bioclip_checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Select a pseudo-unseen threshold or transfer it to the final model."""
    (
        supervised_logits,
        bioclip_logits,
        targets,
        candidate_species,
        supervised_species,
        checkpoint,
        finetuned_bioclip,
    ) = _collect_gate_validation(
        config,
        checkpoint_path,
        bioclip_checkpoint_path,
    )
    candidate_to_index = {
        name: index for index, name in enumerate(candidate_species)
    }
    supervised_indices = [
        candidate_to_index[name] for name in supervised_species
    ]

    if threshold_source is None:
        pseudo_path = _pseudo_split_path(config)
        if pseudo_path is None or not pseudo_path.exists():
            raise FileNotFoundError(
                "Gate threshold selection requires a configured pseudo-unseen split"
            )
        pseudo = read_json(pseudo_path)
        if list(pseudo["training_species"]) != supervised_species:
            raise ValueError(
                "Pseudo-unseen training species do not match the DINO classifier"
            )
        known_indices = {
            candidate_to_index[name] for name in supervised_species
        }
        known_mask = torch.tensor(
            [int(target) in known_indices for target in targets],
            dtype=torch.bool,
        )
        test_count, unseen_count = official_split_counts(config)
        fit = fit_confidence_gate(
            supervised_logits,
            bioclip_logits,
            targets,
            supervised_class_indices=supervised_indices,
            known_mask=known_mask,
            thresholds=_threshold_grid(config),
            selection_metric=str(
                config.get("hybrid_gate", {}).get(
                    "selection_metric", "estimated_overall_accuracy"
                )
            ),
            official_seen_count=test_count,
            official_unseen_count=unseen_count,
        )
        parameters = {
            "threshold": fit.threshold,
            "supervised_temperature": fit.supervised_temperature,
        }
        metrics: dict[str, Any] = asdict(fit)
        threshold_origin = {
            "kind": "species_disjoint_pseudo_unseen",
            "pseudo_unseen_split": str(pseudo_path),
            "pseudo_unseen_split_hash": pseudo["split_hash"],
        }
    else:
        source = _read_gate(threshold_source)
        source_bioclip_checkpoint = source.get("metadata", {}).get(
            "bioclip_fallback_checkpoint"
        )
        if (bioclip_checkpoint_path is None) != (
            source_bioclip_checkpoint is None
        ):
            raise ValueError(
                "Threshold source and final gate must use the same BioCLIP "
                "fallback variant"
            )
        if set(supervised_species) != set(candidate_species):
            raise ValueError(
                "Final gate recalibration requires DINO trained on every seen species"
            )
        subset_targets = torch.tensor(
            [supervised_indices.index(int(target)) for target in targets],
            dtype=torch.long,
        )
        temperature = fit_temperature(supervised_logits, subset_targets)
        confidence = torch.softmax(
            supervised_logits.float() / temperature, dim=-1
        ).max(dim=-1).values
        target_acceptance = float(
            source["metrics"]["known_dino_route_fraction"]
        )
        transferred_threshold = threshold_for_acceptance_rate(
            confidence, target_acceptance
        )
        parameters = {
            "threshold": transferred_threshold,
            "supervised_temperature": temperature,
        }
        metrics = {
            "known_temperature_fit_images": int(len(targets)),
            "target_known_dino_route_fraction": target_acceptance,
            "final_known_dino_route_fraction": float(
                (confidence >= transferred_threshold).float().mean()
            ),
            "threshold_selection_metric": source["metrics"][
                "selection_metric"
            ],
            "threshold_selection_value": source["metrics"][
                "selection_value"
            ],
        }
        threshold_origin = {
            "kind": "transferred_from_species_disjoint_calibration",
            "source": str(threshold_source),
            "source_hash": source["hash"],
            "source_selected_threshold": float(
                source["parameters"]["threshold"]
            ),
            "transfer_method": "known_acceptance_quantile",
        }

    report: dict[str, Any] = {
        "parameters": parameters,
        "metrics": metrics,
        "metadata": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_step": int(checkpoint["step"]),
            "candidate_species": candidate_species,
            "supervised_species": supervised_species,
            "threshold_origin": threshold_origin,
            "official_unseen_labels_used": False,
            "fallback": (
                "finetuned_bioclip_scientific_name_all_candidates"
                if bioclip_checkpoint_path is not None
                else "pretrained_bioclip_scientific_name_all_candidates"
            ),
            "bioclip_fallback_checkpoint": (
                None
                if bioclip_checkpoint_path is None
                else str(bioclip_checkpoint_path)
            ),
            "bioclip_fallback_checkpoint_step": (
                None
                if finetuned_bioclip is None
                else int(finetuned_bioclip["step"])
            ),
        },
    }
    report["hash"] = stable_json_hash(report)
    write_json(output_path, report)
    return report


def load_gate_calibration(path: str | Path) -> dict[str, Any]:
    """Load and verify a confidence-gate report."""
    return _read_gate(path)
