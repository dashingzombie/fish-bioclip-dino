"""Deterministic phased sweeps for the joint supervised-text stage."""

from __future__ import annotations

import copy
import itertools
import json
import shlex
import statistics
from pathlib import Path
from typing import Any

import yaml

from fish_vlm.config import deep_merge, load_config, validate_config
from fish_vlm.slurm.launcher import launch_slurm, submit_slurm_script
from fish_vlm.sweeps.ranking import rank_results
from fish_vlm.sweeps.state import load_state, save_state
from fish_vlm.utils.hashing import stable_json_hash
from fish_vlm.utils.io import atomic_write_text, read_json, write_json
from fish_vlm.workflow import submit_bootstrap_pipeline


PHASE_ORDER = ("loss", "optimiser", "architecture", "training")
PHASE_CONFIGS = {
    "loss": "phase1_loss.yaml",
    "optimiser": "phase2_optimiser.yaml",
    "architecture": "phase3_architecture.yaml",
    "training": "phase4_training.yaml",
    "confirmation": "confirm_top.yaml",
}
SELECTION_METRIC = "estimated_overall_accuracy"
HARMONIC_METRIC = "seen_unseen_harmonic_mean"
DEFAULT_OUTPUT_ROOT = Path("outputs/sweep_pipelines/joint_supervised_text")
_CONFIG_ROOT = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "sweeps"
    / "joint_supervised_text"
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPOSITORY_ROOT / "scripts" / "run_joint_sweeps.py"
_RUN_NAME_KEYS = {
    "loss.dino_text_classification.weight": "dino_w",
    "loss.bioclip_image_teacher.weight": "teacher_w",
    "loss.supervised_species.weight": "supervised_w",
    "training.lr": "lr",
    "training.weight_decay": "wd",
    "projector_lr_multiplier": "projector_mult",
    "dino_last_block_lr_multiplier": "dino_mult",
    "supervised_head_lr_multiplier": "head_mult",
    "model.projector.hidden_dim": "hidden",
    "model.projector.dropout": "dropout",
    "model.temperature": "temperature",
    "loss.branch_consistency.enabled": "consistency",
    "loss.branch_consistency.weight": "consistency_w",
    "loss.branch_consistency.method": "consistency_method",
    "training.batch_size": "batch",
    "training.gradient_accumulation_steps": "accum",
    "training.max_steps": "steps",
}


def _nested_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for dotted, value in overrides.items():
        target = result
        parts = dotted.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return result


def _phase_path(phase: str) -> Path:
    return _CONFIG_ROOT / PHASE_CONFIGS[phase]


def _clean_config(config: dict[str, Any]) -> dict[str, Any]:
    cleaned = copy.deepcopy(config)
    cleaned.pop("_config_path", None)
    cleaned.pop("sweep", None)
    return cleaned


def _value_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return format(value, ".6g")
    return str(value).replace(" ", "_")


def run_name(phase: str, parameters: dict[str, Any]) -> str:
    """Encode every varied parameter in a stable W&B-safe run name."""
    parts = [phase]
    for key in sorted(parameters):
        label = _RUN_NAME_KEYS.get(key, key.replace(".", "_"))
        parts.append(f"{label}={_format_value(parameters[key])}")
    return "-".join(parts)


def _cartesian(parameters: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = sorted(parameters)
    values = [list(parameters[key]) for key in keys]
    return [
        dict(zip(keys, combination, strict=True))
        for combination in itertools.product(*values)
    ]


def _candidate_pairs(candidate: dict[str, Any]) -> set[tuple[str, str, str, str]]:
    items = sorted((key, _value_key(value)) for key, value in candidate.items())
    return {
        (left_key, left_value, right_key, right_value)
        for (left_key, left_value), (right_key, right_value) in itertools.combinations(
            items, 2
        )
    }


def _select_representative(
    candidates: list[dict[str, Any]],
    *,
    target_runs: int,
    baseline: dict[str, Any],
) -> list[dict[str, Any]]:
    """Greedily cover pairs, then balance value frequency deterministically."""
    ordered = sorted(candidates, key=lambda item: stable_json_hash(item))
    if target_runs > len(ordered):
        raise ValueError("Representative target exceeds valid candidate count")
    baseline_match = next(
        (item for item in ordered if item == baseline),
        None,
    )
    if baseline_match is None:
        raise ValueError("Configured sweep baseline is not a valid candidate")
    selected = [baseline_match]
    remaining = [item for item in ordered if item != baseline_match]
    covered = _candidate_pairs(baseline_match)
    counts = {
        (key, _value_key(value)): 1 for key, value in baseline_match.items()
    }
    while len(selected) < target_runs:
        best_index = 0
        best_score: tuple[int, int, int] | None = None
        for index, candidate in enumerate(remaining):
            pairs = _candidate_pairs(candidate)
            new_pairs = len(pairs - covered)
            unseen_values = sum(
                counts.get((key, _value_key(value)), 0) == 0
                for key, value in candidate.items()
            )
            existing_frequency = sum(
                counts.get((key, _value_key(value)), 0)
                for key, value in candidate.items()
            )
            score = (new_pairs, unseen_values, -existing_frequency)
            if best_score is None or score > best_score:
                best_index, best_score = index, score
        chosen = remaining.pop(best_index)
        selected.append(chosen)
        covered.update(_candidate_pairs(chosen))
        for key, value in chosen.items():
            token = (key, _value_key(value))
            counts[token] = counts.get(token, 0) + 1
    return selected


def _phase_candidates(
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sweep = config["sweep"]
    phase = str(sweep["phase"])
    parameters = {
        key: list(values) for key, values in sweep.get("parameters", {}).items()
    }
    baseline = dict(sweep.get("baseline", {}))
    skipped: list[dict[str, Any]] = []

    if phase == "architecture":
        architecture = _cartesian(parameters)
        consistency = sweep["branch_consistency"]
        candidates: list[dict[str, Any]] = []
        for candidate in architecture:
            candidates.append(
                {
                    **candidate,
                    "loss.branch_consistency.enabled": False,
                }
            )
            for weight, method in itertools.product(
                consistency["weight"],
                consistency["method"],
            ):
                candidates.append(
                    {
                        **candidate,
                        "loss.branch_consistency.enabled": True,
                        "loss.branch_consistency.weight": weight,
                        "loss.branch_consistency.method": method,
                    }
                )
    else:
        if phase == "optimiser":
            optional = sweep.get("optional_component_multipliers", {})
            required = optional.get("require_existing_parameter_groups", [])
            existing = set(
                config["training"].get("parameter_groups", {})
            )
            if required and set(required).issubset(existing):
                parameters.update(
                    {
                        key: list(values)
                        for key, values in optional.get(
                            "parameters", {}
                        ).items()
                    }
                )
                baseline.update(optional.get("baseline", {}))
        candidates = _cartesian(parameters)

    if phase == "training":
        max_local_batch = int(
            sweep.get("memory_safety", {}).get(
                "max_local_batch_size",
                0,
            )
        )
        safe: list[dict[str, Any]] = []
        for candidate in candidates:
            local_batch = int(candidate["training.batch_size"])
            if max_local_batch and local_batch > max_local_batch:
                skipped.append(
                    {
                        "phase": phase,
                        "parameters": candidate,
                        "reason": (
                            f"estimated local batch {local_batch} exceeds "
                            f"safe limit {max_local_batch}"
                        ),
                    }
                )
            else:
                safe.append(candidate)
        candidates = safe

    target = int(sweep.get("target_runs", len(candidates)))
    if target < len(candidates):
        candidates = _select_representative(
            candidates,
            target_runs=target,
            baseline=baseline,
        )
    return candidates, skipped


def _actual_overrides(
    parameters: dict[str, Any],
) -> dict[str, Any]:
    overrides = {
        key: value
        for key, value in parameters.items()
        if not key.endswith("_lr_multiplier")
    }
    if any(key.endswith("_lr_multiplier") for key in parameters):
        learning_rate = float(parameters["training.lr"])
        mapping = {
            "projector_lr_multiplier": "training.parameter_groups.projector.lr",
            "dino_last_block_lr_multiplier": (
                "training.parameter_groups.dino_last_block.lr"
            ),
            "supervised_head_lr_multiplier": (
                "training.parameter_groups.supervised_head.lr"
            ),
        }
        for multiplier_key, output_key in mapping.items():
            overrides[output_key] = (
                learning_rate * float(parameters[multiplier_key])
            )
    return overrides


def _learning_rate_metadata(config: dict[str, Any]) -> dict[str, float]:
    training = config["training"]
    base_lr = float(training["lr"])
    groups = training.get("parameter_groups", {})
    return {
        "base": base_lr,
        "projector": float(groups.get("projector", {}).get("lr", base_lr)),
        "dino_last_block": float(
            groups.get("dino_last_block", {}).get("lr", 1e-5)
        ),
        "supervised_head": float(
            groups.get("supervised_head", {}).get("lr", base_lr)
        ),
    }


def _apply_run_metadata(
    config: dict[str, Any],
    *,
    phase: str,
    name: str,
    parameters: dict[str, Any],
    run_id: str,
) -> None:
    world_size = int(config["slurm"].get("gpus", 1))
    batch_size = int(config["training"]["batch_size"])
    accumulation = int(
        config["training"].get("gradient_accumulation_steps", 1)
    )
    config["wandb"]["name"] = name
    config["wandb"]["group"] = f"joint-supervised-text-{phase}"
    config["wandb"]["tags"] = list(
        dict.fromkeys(
            [
                *config["wandb"].get("tags", []),
                "joint-sweep",
                phase,
            ]
        )
    )
    config["sweep_metadata"] = {
        "run_id": run_id,
        "phase": phase,
        "parameters": parameters,
        "local_batch_size": batch_size,
        "world_size": world_size,
        "gradient_accumulation_steps": accumulation,
        "effective_global_batch_size": (
            batch_size * world_size * accumulation
        ),
        "learning_rates": _learning_rate_metadata(config),
    }


def _phase_config(phase: str) -> dict[str, Any]:
    return load_config(_phase_path(phase))


def _index_path(root: Path) -> Path:
    return root / "run_index.json"


def _load_index(root: Path) -> dict[str, Any]:
    path = _index_path(root)
    if path.exists():
        return read_json(path)
    return {"version": 1, "runs": [], "skipped": []}


def _save_index(root: Path, index: dict[str, Any]) -> None:
    index["runs"] = sorted(
        index["runs"],
        key=lambda item: (
            PHASE_ORDER.index(item["phase"])
            if item["phase"] in PHASE_ORDER
            else len(PHASE_ORDER),
            item["name"],
        ),
    )
    write_json(_index_path(root), index)


def _metrics_for_run(run: dict[str, Any]) -> dict[str, Any] | None:
    path = Path(run["metrics_path"])
    return read_json(path) if path.exists() else None


def _run_completed(run: dict[str, Any]) -> bool:
    return (
        Path(run["metrics_path"]).is_file()
        and Path(run["checkpoint_path"]).is_file()
        and Path(run["resolved_config_path"]).is_file()
    )


def _refresh_status(run: dict[str, Any]) -> None:
    run["status"] = "completed" if _run_completed(run) else run.get(
        "status", "planned"
    )


def _rank_search_runs(
    index: dict[str, Any],
    *,
    phase: str | None = None,
) -> list[dict[str, Any]]:
    eligible: list[dict[str, Any]] = []
    lookup: dict[str, dict[str, Any]] = {}
    for run in index["runs"]:
        if run["phase"] not in PHASE_ORDER:
            continue
        if phase is not None and run["phase"] != phase:
            continue
        metrics = _metrics_for_run(run)
        if metrics is None or not _run_completed(run):
            continue
        lookup[run["id"]] = run
        eligible.append(
            {
                "name": run["name"],
                "id": run["id"],
                "metrics": metrics,
            }
        )
    return [
        {**lookup[result["id"]], "metrics": result["metrics"]}
        for result in rank_results(eligible, SELECTION_METRIC)
    ]


def _best_parent(index: dict[str, Any], phase: str) -> dict[str, Any] | None:
    ranked = _rank_search_runs(index, phase=phase)
    return ranked[0] if ranked else None


def _resolved_yaml(config: dict[str, Any]) -> str:
    return yaml.safe_dump(
        _clean_config(config),
        sort_keys=True,
        allow_unicode=True,
    )


def _materialize_phase(
    phase: str,
    *,
    root: Path,
    index: dict[str, Any],
) -> tuple[list[dict[str, Any]], str | None]:
    phase_config = _phase_config(phase)
    sweep = phase_config["sweep"]
    parent_phase = sweep.get("parent_phase")
    parent: dict[str, Any] | None = None
    if parent_phase:
        parent = _best_parent(index, str(parent_phase))
        if parent is None:
            return [], (
                f"{phase} requires a completed {parent_phase} phase "
                "before it can inherit the best configuration"
            )
        base = yaml.safe_load(
            Path(parent["resolved_config_path"]).read_text(encoding="utf-8")
        )
    else:
        base = _clean_config(phase_config)
    base = deep_merge(
        base,
        _nested_overrides(dict(sweep.get("fixed", {}))),
    )
    candidates, skipped = _phase_candidates(phase_config)
    generated: list[dict[str, Any]] = []
    existing = {run["id"]: run for run in index["runs"]}
    for parameters in candidates:
        identity = {
            "phase": phase,
            "parent_run_id": parent["id"] if parent else None,
            "parameters": parameters,
        }
        run_id = stable_json_hash(identity)[:16]
        name = run_name(phase, parameters)
        output_dir = (root / "runs" / phase / run_id).resolve()
        config_path = (root / "configs" / phase / f"{run_id}.yaml").resolve()
        resolved_path = output_dir / "resolved_config.yaml"
        resolved = deep_merge(
            base,
            _nested_overrides(_actual_overrides(parameters)),
        )
        resolved["seed"] = 42
        resolved["output_dir"] = str(output_dir)
        resolved["training"]["checkpoint_name"] = "best.pt"
        resolved["training"]["metrics_name"] = "best.json"
        _apply_run_metadata(
            resolved,
            phase=phase,
            name=name,
            parameters=parameters,
            run_id=run_id,
        )
        validate_config(resolved)
        text = _resolved_yaml(resolved)
        atomic_write_text(config_path, text)
        atomic_write_text(resolved_path, text)
        record = {
            "id": run_id,
            "phase": phase,
            "name": name,
            "parent_run_id": parent["id"] if parent else None,
            "parameters": parameters,
            "config_path": str(config_path),
            "resolved_config_path": str(resolved_path),
            "output_dir": str(output_dir),
            "metrics_path": str(output_dir / "metrics" / "best.json"),
            "checkpoint_path": str(
                output_dir / "checkpoints" / "best.pt"
            ),
            "status": existing.get(run_id, {}).get("status", "planned"),
        }
        for key in ("job_id", "array_task_id"):
            if key in existing.get(run_id, {}):
                record[key] = existing[run_id][key]
        _refresh_status(record)
        existing[run_id] = record
        generated.append(record)
    index["runs"] = list(existing.values())
    index["skipped"] = [
        item
        for item in index.get("skipped", [])
        if item.get("phase") != phase
    ] + skipped
    _save_index(root, index)
    return generated, None


def _confirmation_candidates(
    *,
    top: int,
    root: Path,
    index: dict[str, Any],
) -> list[dict[str, Any]]:
    ranked = _rank_search_runs(index)
    if len(ranked) < top:
        raise ValueError(
            f"Confirmation requested top {top}, but only "
            f"{len(ranked)} completed search runs are available"
        )
    confirm = _phase_config("confirmation")
    sweep = confirm["sweep"]
    seeds = [int(seed) for seed in sweep["seeds"]]
    existing = {run["id"]: run for run in index["runs"]}
    generated: list[dict[str, Any]] = []
    for rank, source in enumerate(ranked[:top], start=1):
        source_config = yaml.safe_load(
            Path(source["resolved_config_path"]).read_text(encoding="utf-8")
        )
        source_parameters = dict(source["parameters"])
        for seed in seeds:
            identity = {
                "phase": "confirmation",
                "source_run_id": source["id"],
                "seed": seed,
            }
            run_id = stable_json_hash(identity)[:16]
            parameters = {
                **source_parameters,
                "seed": seed,
            }
            name = f"confirm{rank}-{run_name(source['phase'], parameters)}"
            output_dir = (
                root / "runs" / "confirmation" / run_id
            ).resolve()
            config_path = (
                root / "configs" / "confirmation" / f"{run_id}.yaml"
            ).resolve()
            resolved_path = output_dir / "resolved_config.yaml"
            resolved = deep_merge(
                source_config,
                _nested_overrides(dict(sweep.get("fixed", {}))),
            )
            resolved["seed"] = seed
            resolved["output_dir"] = str(output_dir)
            resolved["training"]["checkpoint_name"] = "best.pt"
            resolved["training"]["metrics_name"] = "best.json"
            _apply_run_metadata(
                resolved,
                phase="confirmation",
                name=name,
                parameters=parameters,
                run_id=run_id,
            )
            resolved["sweep_metadata"]["source_run_id"] = source["id"]
            resolved["sweep_metadata"]["source_rank"] = rank
            validate_config(resolved)
            text = _resolved_yaml(resolved)
            atomic_write_text(config_path, text)
            atomic_write_text(resolved_path, text)
            record = {
                "id": run_id,
                "phase": "confirmation",
                "name": name,
                "source_run_id": source["id"],
                "source_rank": rank,
                "seed": seed,
                "parameters": parameters,
                "config_path": str(config_path),
                "resolved_config_path": str(resolved_path),
                "output_dir": str(output_dir),
                "metrics_path": str(output_dir / "metrics" / "best.json"),
                "checkpoint_path": str(
                    output_dir / "checkpoints" / "best.pt"
                ),
                "status": existing.get(run_id, {}).get(
                    "status", "planned"
                ),
            }
            for key in ("job_id", "array_task_id"):
                if key in existing.get(run_id, {}):
                    record[key] = existing[run_id][key]
            _refresh_status(record)
            existing[run_id] = record
            generated.append(record)
    index["runs"] = list(existing.values())
    _save_index(root, index)
    return generated


def _submit_array(
    runs: list[dict[str, Any]],
    *,
    root: Path,
    index: dict[str, Any],
    max_concurrent: int,
    resume: bool,
    dependency: str | None = None,
) -> str | None:
    pending = [run for run in runs if not _run_completed(run)]
    if not pending:
        return None
    phase = pending[0]["phase"]
    state_path = root / "state.json"
    state = load_state(state_path)
    phase_state = state["phases"].get(phase, {})
    if phase_state.get("job_ids") and not resume:
        raise ValueError(
            f"Phase {phase} was already submitted; pass --resume to "
            "submit only its incomplete runs"
        )
    launch_config = load_config(pending[0]["config_path"])
    launch_config["slurm"]["array_configs"] = [
        run["config_path"] for run in pending
    ]
    launch_config["slurm"]["array_max_concurrent"] = max_concurrent
    launch_config["slurm"]["job_name"] = f"fish-joint-{phase}"
    launch_config["slurm"]["script_dir"] = str(
        (root / "slurm").resolve()
    )
    launch_config["slurm"]["log_dir"] = str(
        (root / "slurm" / "logs").resolve()
    )
    launch_config["slurm"]["dependency"] = dependency
    job_id = launch_slurm(launch_config, dry_run=False).split(";", 1)[0]
    for task_id, run in enumerate(pending):
        run["job_id"] = job_id
        run["array_task_id"] = task_id
        run["status"] = "submitted"
    phase_state.setdefault("job_ids", []).append(job_id)
    phase_state["last_submitted_run_ids"] = [run["id"] for run in pending]
    phase_state["max_concurrent"] = max_concurrent
    state["phases"][phase] = phase_state
    save_state(state_path, state)
    _save_index(root, index)
    return job_id


def _controller_script(
    config: dict[str, Any],
    *,
    job_name: str,
    command: list[str],
    root: Path,
) -> str:
    """Render a short dependent controller through the existing Slurm profile."""
    slurm = config["slurm"]
    log_dir = (root / "slurm" / "logs").resolve()
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={job_name}",
        "#SBATCH --nodes=1",
        "#SBATCH --gpus=1",
        "#SBATCH --cpus-per-task=1",
        "#SBATCH --mem=8G",
        "#SBATCH --time=00:30:00",
    ]
    for key in ("account", "partition"):
        if slurm.get(key):
            lines.append(f"#SBATCH --{key}={slurm[key]}")
    lines.extend(
        [
            f"#SBATCH --output={log_dir}/%x-%j.out",
            f"#SBATCH --error={log_dir}/%x-%j.err",
            "",
            "set -euo pipefail",
            f"cd {shlex.quote(str(_REPOSITORY_ROOT))}",
        ]
    )
    if slurm.get("environment_activate"):
        lines.append(str(slurm["environment_activate"]))
    lines.append(shlex.join(command))
    return "\n".join(lines) + "\n"


def _submit_controller(
    *,
    root: Path,
    index: dict[str, Any],
    after_job_id: str,
    completed_phase: str,
    max_concurrent: int,
) -> str:
    """Submit the controller that advances to the next dependent phase."""
    if completed_phase == "loss":
        arguments = ["--phase", "optimiser", "--submit"]
        controller_name = "optimiser"
    elif completed_phase == "optimiser":
        arguments = ["--phase", "architecture", "--submit"]
        controller_name = "architecture"
    elif completed_phase == "architecture":
        arguments = ["--phase", "training", "--submit"]
        controller_name = "training"
    elif completed_phase == "training":
        arguments = ["--confirm-top", "8", "--submit"]
        controller_name = "confirmation"
    elif completed_phase == "confirmation":
        arguments = ["--report-only"]
        controller_name = "report"
    else:
        raise ValueError(f"Cannot chain after unknown phase {completed_phase!r}")

    arguments.extend(
        [
            "--resume",
            "--auto-chain",
            "--max-concurrent",
            str(max_concurrent),
            "--output-root",
            str(root),
        ]
    )
    command = [
        "python",
        str(_SCRIPT_PATH),
        *arguments,
    ]
    reference = next(
        (
            run
            for run in index["runs"]
            if run["phase"] == completed_phase
        ),
        None,
    )
    if reference is None:
        reference = index["runs"][0]
    config = load_config(reference["config_path"])
    script = _controller_script(
        config,
        job_name=f"fish-joint-next-{controller_name}",
        command=command,
        root=root,
    )
    (root / "slurm" / "logs").mkdir(parents=True, exist_ok=True)
    controller_id = submit_slurm_script(
        script,
        root / "slurm" / "controllers" / f"after-{completed_phase}.sh",
        dependency=after_job_id,
    )
    state_path = root / "state.json"
    state = load_state(state_path)
    phase_state = state["phases"].setdefault(completed_phase, {})
    phase_state.setdefault("controller_job_ids", []).append(controller_id)
    save_state(state_path, state)
    return controller_id


def _summary_rows(index: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in index["runs"]:
        _refresh_status(run)
        metrics = _metrics_for_run(run)
        rows.append(
            {
                "run_name": run["name"],
                "parameters": run["parameters"],
                "score": (
                    None
                    if metrics is None
                    else metrics.get(SELECTION_METRIC)
                ),
                "best_step": (
                    None if metrics is None else metrics.get("best_step")
                ),
                "checkpoint": run["checkpoint_path"],
                "status": run["status"],
            }
        )
    return rows


def format_summary_table(rows: list[dict[str, Any]]) -> str:
    """Return the requested compact human-readable run summary."""
    headers = ("run", "parameters", "score", "best_step", "checkpoint")
    rendered: list[tuple[str, ...]] = []
    for row in rows:
        score = (
            "-"
            if row["score"] is None
            else f"{float(row['score']):.6f}"
        )
        step = (
            "-"
            if row["best_step"] is None
            else str(int(float(row["best_step"])))
        )
        rendered.append(
            (
                str(row["run_name"]),
                json.dumps(
                    row["parameters"],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                score,
                step,
                str(row["checkpoint"]),
            )
        )
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rendered))
        if rendered
        else len(headers[index])
        for index in range(len(headers))
    ]
    lines = [
        "  ".join(
            header.ljust(widths[index])
            for index, header in enumerate(headers)
        ),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "  ".join(
            value.ljust(widths[index])
            for index, value in enumerate(row)
        )
        for row in rendered
    )
    return "\n".join(lines)


def confirmation_ranking(index: dict[str, Any]) -> list[dict[str, Any]]:
    """Rank fully repeated configurations by the three required statistics."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in index["runs"]:
        if run["phase"] != "confirmation":
            continue
        metrics = _metrics_for_run(run)
        if metrics is not None and _run_completed(run):
            grouped.setdefault(run["source_run_id"], []).append(metrics)
    ranked: list[dict[str, Any]] = []
    for source_id, metrics in grouped.items():
        if len(metrics) != 3:
            continue
        overall = [float(item[SELECTION_METRIC]) for item in metrics]
        harmonic = [float(item[HARMONIC_METRIC]) for item in metrics]
        ranked.append(
            {
                "source_run_id": source_id,
                "mean_estimated_overall_accuracy": statistics.fmean(
                    overall
                ),
                "mean_seen_unseen_harmonic_mean": statistics.fmean(
                    harmonic
                ),
                "worst_seed_estimated_overall_accuracy": min(overall),
            }
        )
    return sorted(
        ranked,
        key=lambda item: (
            item["mean_estimated_overall_accuracy"],
            item["mean_seen_unseen_harmonic_mean"],
            item["worst_seed_estimated_overall_accuracy"],
        ),
        reverse=True,
    )


def run_joint_sweeps(
    *,
    phase: str | None = None,
    confirm_top: int | None = None,
    submit: bool = False,
    dry_run: bool = False,
    max_concurrent: int = 8,
    resume: bool = False,
    auto_chain: bool = False,
    everything: bool = False,
    report_only: bool = False,
    pipeline_config: str | Path = "configs/pipeline.yaml",
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Materialise, resume, summarize, and optionally submit sweep arrays."""
    if submit and dry_run:
        raise ValueError("--submit and --dry-run are mutually exclusive")
    if max_concurrent < 1:
        raise ValueError("--max-concurrent must be at least one")
    if confirm_top is not None and confirm_top < 1:
        raise ValueError("--confirm-top must be at least one")
    if everything and confirm_top is not None:
        raise ValueError("--everything cannot be combined with --confirm-top")
    root = Path(output_root).resolve()
    index = _load_index(root)
    generated: list[dict[str, Any]] = []
    blocked: list[str] = []
    job_ids: list[str] = []
    controller_job_ids: list[str] = []
    bootstrap: dict[str, Any] | None = None

    if report_only:
        rows = _summary_rows(index)
        _save_index(root, index)
        print(format_summary_table(rows))
        confirmation = confirmation_ranking(index)
        if confirmation:
            print("\nconfirmation ranking")
            print(json.dumps(confirmation, indent=2, sort_keys=True))
        return {
            "dry_run": dry_run,
            "submitted": False,
            "generated_runs": 0,
            "job_ids": [],
            "controller_job_ids": [],
            "blocked": [],
            "run_index": str(_index_path(root)),
            "completed_runs": sum(
                row["status"] == "completed" for row in rows
            ),
            "confirmation_ranking": confirmation,
        }

    initial_dependency: str | None = None
    if everything:
        pipeline = load_config(pipeline_config)
        if dry_run:
            bootstrap_plan = submit_bootstrap_pipeline(
                pipeline,
                dry_run=True,
                gpus=int(pipeline["slurm"].get("gpus", 4)),
            )
            bootstrap = {
                "dry_run": True,
                "workflow_hash": bootstrap_plan["workflow_hash"],
                "jobs": [
                    {
                        "name": job["name"],
                        "gpus": job["gpus"],
                        "depends_on": job["depends_on"],
                    }
                    for job in bootstrap_plan["jobs"]
                ],
            }
        elif submit:
            state_path = root / "state.json"
            state = load_state(state_path)
            master = state.setdefault("master_pipeline", {})
            if master.get("jobs"):
                if not resume:
                    raise ValueError(
                        "The master pipeline was already submitted; "
                        "pass --resume to continue incomplete work"
                    )
                bootstrap = {
                    "mode": "slurm",
                    "status": "submitted",
                    "jobs": master["jobs"],
                }
            else:
                bootstrap = submit_bootstrap_pipeline(
                    pipeline,
                    dry_run=False,
                    gpus=int(pipeline["slurm"].get("gpus", 4)),
                )
                master["jobs"] = bootstrap["jobs"]
                master["workflow_hash"] = bootstrap["workflow_hash"]
                save_state(state_path, state)
            initial_dependency = str(
                bootstrap["jobs"]["finalisation"]
            )
        phase = "loss"
        auto_chain = True

    if confirm_top is not None:
        generated = _confirmation_candidates(
            top=confirm_top,
            root=root,
            index=index,
        )
        if submit:
            job_id = _submit_array(
                generated,
                root=root,
                index=index,
                max_concurrent=max_concurrent,
                resume=resume,
            )
            if job_id:
                job_ids.append(job_id)
                if auto_chain:
                    controller_job_ids.append(
                        _submit_controller(
                            root=root,
                            index=index,
                            after_job_id=job_id,
                            completed_phase="confirmation",
                            max_concurrent=max_concurrent,
                        )
                    )
    else:
        selected_phases = (
            PHASE_ORDER if phase == "all" else (phase or "loss",)
        )
        for selected in selected_phases:
            phase_runs, reason = _materialize_phase(
                selected,
                root=root,
                index=index,
            )
            if reason:
                blocked.append(reason)
                break
            generated.extend(phase_runs)
            if submit:
                job_id = _submit_array(
                    phase_runs,
                    root=root,
                    index=index,
                    max_concurrent=max_concurrent,
                    resume=resume,
                    dependency=initial_dependency,
                )
                if job_id:
                    job_ids.append(job_id)
                    if auto_chain:
                        controller_job_ids.append(
                            _submit_controller(
                                root=root,
                                index=index,
                                after_job_id=job_id,
                                completed_phase=selected,
                                max_concurrent=max_concurrent,
                            )
                        )
                elif (
                    auto_chain
                    and initial_dependency
                    and all(_run_completed(run) for run in phase_runs)
                ):
                    controller_job_ids.append(
                        _submit_controller(
                            root=root,
                            index=index,
                            after_job_id=initial_dependency,
                            completed_phase=selected,
                            max_concurrent=max_concurrent,
                        )
                    )
                initial_dependency = None
            if selected_phases == PHASE_ORDER and not all(
                _run_completed(run) for run in phase_runs
            ):
                blocked.append(
                    f"{selected} was materialized"
                    + (" and submitted" if submit else "")
                    + "; rerun --phase all after it completes to advance"
                )
                break

    _save_index(root, index)
    rows = _summary_rows(index)
    _save_index(root, index)
    print(format_summary_table(rows))
    confirmation = confirmation_ranking(index)
    if confirmation:
        print("\nconfirmation ranking")
        print(json.dumps(confirmation, indent=2, sort_keys=True))
    return {
        "dry_run": dry_run,
        "submitted": submit,
        "generated_runs": len(generated),
        "job_ids": job_ids,
        "controller_job_ids": controller_job_ids,
        "blocked": blocked,
        "run_index": str(_index_path(root)),
        "completed_runs": sum(row["status"] == "completed" for row in rows),
        "confirmation_ranking": confirmation,
        "bootstrap": bootstrap,
    }
