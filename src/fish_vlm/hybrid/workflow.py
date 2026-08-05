"""One resumable run point for four DINO/BioCLIP hybrid submissions."""

from __future__ import annotations

import copy
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from fish_vlm.config import load_config, validate_config
from fish_vlm.slurm.launcher import submit_slurm_script
from fish_vlm.utils.io import atomic_write_text, read_json, write_json


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ENTRYPOINT = REPOSITORY_ROOT / "scripts" / "run_hybrid_pipeline.py"


def _read_spec(path: str | Path) -> dict[str, Any]:
    spec_path = Path(path)
    with spec_path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError("Hybrid sweep specification must be a YAML mapping")
    if not isinstance(value.get("finetuned_bioclip_config"), str):
        raise ValueError(
            "Hybrid sweep requires finetuned_bioclip_config"
        )
    recipes = value.get("recipes")
    if not isinstance(recipes, list) or not recipes:
        raise ValueError("Hybrid sweep requires at least one recipe")
    names: list[str] = []
    for recipe in recipes:
        if not isinstance(recipe, dict) or not isinstance(recipe.get("name"), str):
            raise TypeError("Every hybrid recipe requires a string name")
        name = recipe["name"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError(f"Unsafe hybrid recipe name: {name!r}")
        names.append(name)
        if not isinstance(recipe.get("overrides", {}), dict):
            raise TypeError(f"Hybrid recipe overrides must be a mapping: {name}")
    if len(names) != len(set(names)):
        raise ValueError("Hybrid recipe names must be unique")
    return value


def _set_dotted(config: dict[str, Any], key: str, value: Any) -> None:
    target = config
    parts = key.split(".")
    if any(not part for part in parts):
        raise ValueError(f"Invalid dotted override: {key!r}")
    for part in parts[:-1]:
        child = target.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"Override crosses a non-mapping field: {key}")
        target = child
    target[parts[-1]] = value


def _clean_config(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result.pop("_config_path", None)
    return result


def _write_config(path: Path, config: dict[str, Any]) -> None:
    validate_config(config)
    atomic_write_text(
        path,
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
    )


def materialize_hybrid_plan(spec_path: str | Path) -> dict[str, Any]:
    """Write deterministic resolved configs and return the inspectable plan."""
    spec = _read_spec(spec_path)
    base_path = Path(str(spec["base_config"]))
    if not base_path.is_absolute():
        base_path = REPOSITORY_ROOT / base_path
    base = _clean_config(load_config(base_path))
    bioclip_path = Path(str(spec["finetuned_bioclip_config"]))
    if not bioclip_path.is_absolute():
        bioclip_path = REPOSITORY_ROOT / bioclip_path
    root = Path(str(spec.get("output_root", "outputs/hybrid")))
    if not root.is_absolute():
        root = REPOSITORY_ROOT / root
    runs: list[dict[str, Any]] = []
    for recipe in spec["recipes"]:
        name = str(recipe["name"])
        run_root = root / "calibration" / name
        config = copy.deepcopy(base)
        for key, value in sorted(recipe.get("overrides", {}).items()):
            _set_dotted(config, str(key), value)
        config["output_dir"] = str(run_root)
        config["training"]["checkpoint_name"] = "best.pt"
        config["training"]["metrics_name"] = "best.json"
        config["training"]["resume_checkpoint"] = None
        pseudo = config["validation"]["pseudo_unseen"]
        pseudo["enabled"] = True
        pseudo["split_seed"] = 42
        pseudo["split_path"] = "auto"
        config["validation"]["selection_metric"] = "seen_accuracy"
        config["wandb"]["name"] = name
        config_path = run_root / "resolved_config.yaml"
        _write_config(config_path, config)
        runs.append(
            {
                "name": name,
                "parameters": recipe.get("overrides", {}),
                "config": str(config_path),
                "checkpoint": str(run_root / "checkpoints" / "best.pt"),
                "training_metrics": str(run_root / "metrics" / "best.json"),
                "gate": str(run_root / "gate.json"),
            }
        )
    bioclip_root = root / "bioclip" / "calibration"
    bioclip_config = _clean_config(load_config(bioclip_path))
    bioclip_config["output_dir"] = str(bioclip_root)
    bioclip_config["training"]["checkpoint_name"] = "best.pt"
    bioclip_config["training"]["metrics_name"] = "best.json"
    bioclip_config["training"]["resume_checkpoint"] = None
    bioclip_pseudo = bioclip_config["validation"]["pseudo_unseen"]
    bioclip_pseudo["enabled"] = True
    bioclip_pseudo["split_seed"] = 42
    bioclip_pseudo["split_path"] = "auto"
    bioclip_config["validation"][
        "selection_metric"
    ] = "seen_unseen_harmonic_mean"
    bioclip_config_path = bioclip_root / "resolved_config.yaml"
    _write_config(bioclip_config_path, bioclip_config)
    finetuned_bioclip = {
        "config": str(bioclip_config_path),
        "checkpoint": str(bioclip_root / "checkpoints" / "best.pt"),
        "training_metrics": str(bioclip_root / "metrics" / "best.json"),
        "gate": str(bioclip_root / "gate.json"),
    }
    plan = {
        "version": 2,
        "spec": str(Path(spec_path)),
        "base_config": str(base_path),
        "output_root": str(root),
        "runs": runs,
        "finetuned_bioclip": finetuned_bioclip,
        "submission_outputs": {
            name: str(root / "submissions" / name / "submission.zip")
            for name in (
                "pretrained_bioclip_hard_routed",
                "pretrained_bioclip_confidence_gated",
                "finetuned_bioclip_hard_routed",
                "finetuned_bioclip_confidence_gated",
            )
        },
    }
    write_json(root / "plan.json", plan)
    return plan


def _python_command(*arguments: str) -> list[str]:
    return [sys.executable, "-m", "fish_vlm.cli", *arguments]


def _train_command(config_path: str, gpus: int) -> list[str]:
    return [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={gpus}",
        "-m",
        "fish_vlm.cli",
        "train",
        "--config",
        config_path,
    ]


def _run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def _step(
    name: str,
    command: list[str],
    outputs: list[str | Path],
    *,
    resume: bool,
    state: dict[str, Any],
    state_path: Path,
) -> None:
    output_paths = [Path(path) for path in outputs]
    if resume and output_paths and all(path.is_file() for path in output_paths):
        state["steps"][name] = {"status": "reused", "command": command}
        write_json(state_path, state)
        return
    if not resume and any(path.exists() for path in output_paths):
        raise FileExistsError(
            f"Step {name} already has output; rerun with --resume"
        )
    _run_command(command)
    missing = [str(path) for path in output_paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Step {name} did not create outputs: {missing}")
    state["steps"][name] = {"status": "completed", "command": command}
    write_json(state_path, state)


def select_best_recipe(plan: dict[str, Any]) -> dict[str, Any]:
    """Rank completed recipe gates by the configured leakage-safe objective."""
    ranked: list[tuple[tuple[float, float, float, str], dict[str, Any]]] = []
    for run in plan["runs"]:
        gate = read_json(run["gate"])
        metrics = gate["metrics"]
        ranked.append(
            (
                (
                    float(metrics["selection_value"]),
                    float(metrics["pseudo_unseen_accuracy"]),
                    float(metrics["known_accuracy"]),
                    str(run["name"]),
                ),
                {**run, "gate_report": gate},
            )
        )
    if not ranked:
        raise ValueError("No completed hybrid recipe gates are available")
    return max(ranked, key=lambda item: item[0])[1]


def _materialize_final_configs(
    plan: dict[str, Any], selected: dict[str, Any]
) -> dict[str, str]:
    root = Path(plan["output_root"])
    final_root = root / "final"
    final = _clean_config(load_config(selected["config"]))
    final["output_dir"] = str(final_root)
    final["validation"]["pseudo_unseen"]["enabled"] = False
    final["validation"]["pseudo_unseen"]["split_path"] = None
    final["validation"]["selection_metric"] = "seen_accuracy"
    final["wandb"]["name"] = f"final-{selected['name']}"
    final_path = final_root / "resolved_config.yaml"
    _write_config(final_path, final)

    hard_seen = copy.deepcopy(final)
    hard_seen["model"]["dino"]["trainable_scope"] = "frozen"
    hard_seen["training"]["stage"] = "projection_only"
    hard_seen["inference"]["generalised_enabled"] = False
    hard_seen["inference"]["hybrid_dino_seen_checkpoint"] = True
    hard_seen["inference"].pop("training_free_native", None)
    hard_seen["inference"]["test"] = {
        "candidate_set": "seen",
        "mode": "supervised",
    }
    hard_seen_path = final_root / "inference_hard_seen.yaml"
    _write_config(hard_seen_path, hard_seen)

    hard_unseen = copy.deepcopy(hard_seen)
    hard_unseen["model"]["supervised_head"]["enabled"] = False
    hard_unseen["inference"]["training_free_native"] = True
    hard_unseen["inference"]["test"] = {
        "candidate_set": "unseen",
        "mode": "bioclip_native",
    }
    hard_unseen["inference"]["unseen"] = {
        "candidate_set": "unseen",
        "mode": "bioclip_native",
    }
    hard_unseen_path = final_root / "inference_hard_unseen_pretrained.yaml"
    _write_config(hard_unseen_path, hard_unseen)

    hard_unseen_finetuned = copy.deepcopy(hard_unseen)
    hard_unseen_finetuned["inference"].pop("training_free_native", None)
    hard_unseen_finetuned_path = (
        final_root / "inference_hard_unseen_finetuned.yaml"
    )
    _write_config(hard_unseen_finetuned_path, hard_unseen_finetuned)

    gated = copy.deepcopy(hard_seen)
    gated["inference"]["generalised_enabled"] = True
    gated["inference"]["test"] = {
        "candidate_set": "all",
        "mode": "bioclip_native",
    }
    gated["inference"]["unseen"] = {
        "candidate_set": "all",
        "mode": "bioclip_native",
    }
    gated_path = final_root / "inference_gated.yaml"
    _write_config(gated_path, gated)
    return {
        "final": str(final_path),
        "hard_seen": str(hard_seen_path),
        "hard_unseen_pretrained": str(hard_unseen_path),
        "hard_unseen_finetuned": str(hard_unseen_finetuned_path),
        "gated": str(gated_path),
    }


def _materialize_final_bioclip_config(plan: dict[str, Any]) -> str:
    root = Path(plan["output_root"])
    final_root = root / "bioclip" / "final"
    config = _clean_config(
        load_config(plan["finetuned_bioclip"]["config"])
    )
    config["output_dir"] = str(final_root)
    config["training"]["checkpoint_name"] = "best.pt"
    config["training"]["metrics_name"] = "best.json"
    config["training"]["resume_checkpoint"] = None
    config["validation"]["pseudo_unseen"]["enabled"] = False
    config["validation"]["pseudo_unseen"]["split_path"] = None
    config["validation"]["selection_metric"] = "seen_accuracy"
    config["wandb"]["name"] = "bioclip-long-final-all-seen"
    path = final_root / "resolved_config.yaml"
    _write_config(path, config)
    return str(path)


def planned_commands(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the static portion of the plan for dry-run inspection."""
    first_config = plan["runs"][0]["config"]
    config = load_config(first_config)
    gpus = int(config["slurm"].get("gpus", 1))
    commands: list[dict[str, Any]] = []
    for command in (
        "prepare-prompts",
        "build-text-prototypes",
        "make-pseudo-unseen",
        "build-image-cache",
        "build-teacher-cache",
    ):
        commands.append(
            {"step": command, "command": _python_command(command, "--config", first_config)}
        )
    for run in plan["runs"]:
        commands.extend(
            [
                {
                    "step": f"train:{run['name']}",
                    "command": _train_command(run["config"], gpus),
                },
                {
                    "step": f"calibrate-gate:{run['name']}",
                    "command": _python_command(
                        "calibrate-gate",
                        "--config",
                        run["config"],
                        "--checkpoint",
                        run["checkpoint"],
                        "--output",
                        run["gate"],
                    ),
                },
            ]
        )
    bioclip = plan["finetuned_bioclip"]
    commands.append(
        {
            "step": "train:bioclip-long-pseudo-unseen",
            "command": _train_command(bioclip["config"], gpus),
        }
    )
    commands.append(
        {
            "step": "select-dino-calibrate-two-gates-and-run-finals",
            "command": ["dynamic-after-calibration"],
        }
    )
    return commands


def run_hybrid_pipeline(spec_path: str | Path, *, resume: bool = False) -> dict[str, Any]:
    """Execute long DINO/BioCLIP fits and four deterministic ZIPs."""
    plan = materialize_hybrid_plan(spec_path)
    root = Path(plan["output_root"])
    state_path = root / "state.json"
    state = read_json(state_path) if state_path.exists() and resume else {
        "version": 1,
        "steps": {},
    }
    first_config = plan["runs"][0]["config"]
    base = load_config(first_config)
    gpus = int(base["slurm"].get("gpus", 1))

    # Cache builders are identity-validating and safely reuse valid artifacts.
    for command_name in (
        "prepare-prompts",
        "build-text-prototypes",
        "make-pseudo-unseen",
        "build-image-cache",
        "build-teacher-cache",
    ):
        _run_command(_python_command(command_name, "--config", first_config))

    for run in plan["runs"]:
        _step(
            f"train:{run['name']}",
            _train_command(run["config"], gpus),
            [run["checkpoint"], run["training_metrics"]],
            resume=resume,
            state=state,
            state_path=state_path,
        )
        _step(
            f"calibrate-gate:{run['name']}",
            _python_command(
                "calibrate-gate",
                "--config",
                run["config"],
                "--checkpoint",
                run["checkpoint"],
                "--output",
                run["gate"],
            ),
            [run["gate"]],
            resume=resume,
            state=state,
            state_path=state_path,
        )

    bioclip_calibration = plan["finetuned_bioclip"]
    _step(
        "train:bioclip-long-pseudo-unseen",
        _train_command(bioclip_calibration["config"], gpus),
        [
            bioclip_calibration["checkpoint"],
            bioclip_calibration["training_metrics"],
        ],
        resume=resume,
        state=state,
        state_path=state_path,
    )

    selected = select_best_recipe(plan)
    write_json(
        root / "selection.json",
        {
            "name": selected["name"],
            "parameters": selected["parameters"],
            "gate": selected["gate"],
            "gate_metrics": selected["gate_report"]["metrics"],
        },
    )
    configs = _materialize_final_configs(plan, selected)
    final_checkpoint = root / "final" / "checkpoints" / "best.pt"
    final_metrics = root / "final" / "metrics" / "best.json"
    pretrained_final_gate = root / "final" / "gate_pretrained_bioclip.json"
    _step(
        "train:final-all-seen",
        _train_command(configs["final"], gpus),
        [final_checkpoint, final_metrics],
        resume=resume,
        state=state,
        state_path=state_path,
    )
    _step(
        "calibrate-gate:pretrained-bioclip-final-temperature",
        _python_command(
            "calibrate-gate",
            "--config",
            configs["final"],
            "--checkpoint",
            str(final_checkpoint),
            "--threshold-source",
            selected["gate"],
            "--output",
            str(pretrained_final_gate),
        ),
        [pretrained_final_gate],
        resume=resume,
        state=state,
        state_path=state_path,
    )

    finetuned_calibration_gate = Path(bioclip_calibration["gate"])
    _step(
        "calibrate-gate:finetuned-bioclip-pseudo-unseen",
        _python_command(
            "calibrate-gate",
            "--config",
            selected["config"],
            "--checkpoint",
            selected["checkpoint"],
            "--bioclip-checkpoint",
            bioclip_calibration["checkpoint"],
            "--output",
            str(finetuned_calibration_gate),
        ),
        [finetuned_calibration_gate],
        resume=resume,
        state=state,
        state_path=state_path,
    )

    final_bioclip_config = _materialize_final_bioclip_config(plan)
    final_bioclip_checkpoint = (
        root / "bioclip" / "final" / "checkpoints" / "best.pt"
    )
    final_bioclip_metrics = root / "bioclip" / "final" / "metrics" / "best.json"
    _step(
        "train:bioclip-long-final-all-seen",
        _train_command(final_bioclip_config, gpus),
        [final_bioclip_checkpoint, final_bioclip_metrics],
        resume=resume,
        state=state,
        state_path=state_path,
    )
    finetuned_final_gate = root / "final" / "gate_finetuned_bioclip.json"
    _step(
        "calibrate-gate:finetuned-bioclip-final-temperature",
        _python_command(
            "calibrate-gate",
            "--config",
            configs["final"],
            "--checkpoint",
            str(final_checkpoint),
            "--bioclip-checkpoint",
            str(final_bioclip_checkpoint),
            "--threshold-source",
            str(finetuned_calibration_gate),
            "--output",
            str(finetuned_final_gate),
        ),
        [finetuned_final_gate],
        resume=resume,
        state=state,
        state_path=state_path,
    )

    submission_specs = {
        "pretrained_bioclip_hard_routed": {
            "test_config": configs["hard_seen"],
            "unseen_config": configs["hard_unseen_pretrained"],
            "gated": False,
            "gate": None,
            "bioclip_checkpoint": None,
        },
        "pretrained_bioclip_confidence_gated": {
            "test_config": configs["gated"],
            "unseen_config": configs["gated"],
            "gated": True,
            "gate": str(pretrained_final_gate),
            "bioclip_checkpoint": None,
        },
        "finetuned_bioclip_hard_routed": {
            "test_config": configs["hard_seen"],
            "unseen_config": configs["hard_unseen_finetuned"],
            "gated": False,
            "gate": None,
            "bioclip_checkpoint": str(final_bioclip_checkpoint),
        },
        "finetuned_bioclip_confidence_gated": {
            "test_config": configs["gated"],
            "unseen_config": configs["gated"],
            "gated": True,
            "gate": str(finetuned_final_gate),
            "bioclip_checkpoint": str(final_bioclip_checkpoint),
        },
    }
    for submission_name, submission in submission_specs.items():
        destination = root / "submissions" / submission_name
        test_predictions = destination / "test.json"
        unseen_predictions = destination / "unseen.json"
        merged = destination / "prediction.json"
        zipped = destination / "submission.zip"
        for split, output in (
            ("test", test_predictions),
            ("unseen", unseen_predictions),
        ):
            config_key = f"{split}_config"
            if submission["gated"]:
                command = _python_command(
                    "infer-gated",
                    "--config",
                    str(submission[config_key]),
                    "--checkpoint",
                    str(final_checkpoint),
                    "--gate",
                    str(submission["gate"]),
                    "--split",
                    split,
                    "--output",
                    str(output),
                )
            else:
                command = _python_command(
                    "infer",
                    "--config",
                    str(submission[config_key]),
                    "--checkpoint",
                    str(final_checkpoint),
                    "--split",
                    split,
                    "--output",
                    str(output),
                )
            if submission["bioclip_checkpoint"] is not None and (
                submission["gated"] or split == "unseen"
            ):
                command.extend(
                    [
                        "--bioclip-checkpoint",
                        str(submission["bioclip_checkpoint"]),
                    ]
                )
            _step(
                f"infer:{submission_name}:{split}",
                command,
                [output],
                resume=resume,
                state=state,
                state_path=state_path,
            )
        _step(
            f"merge:{submission_name}",
            _python_command(
                "merge-submission",
                "--test",
                str(test_predictions),
                "--unseen",
                str(unseen_predictions),
                "--output",
                str(merged),
            ),
            [merged],
            resume=resume,
            state=state,
            state_path=state_path,
        )
        _run_command(
            _python_command(
                "validate-submission",
                "--submission",
                str(merged),
                "--config",
                str(submission["test_config"]),
            )
        )
        _step(
            f"package:{submission_name}",
            _python_command(
                "package-submission",
                "--submission",
                str(merged),
                "--output",
                str(zipped),
            ),
            [zipped],
            resume=resume,
            state=state,
            state_path=state_path,
        )

    summary = {
        "selected_recipe": selected["name"],
        "selected_parameters": selected["parameters"],
        "pretrained_threshold_selection": selected["gate_report"]["metrics"],
        "finetuned_threshold_selection": read_json(
            finetuned_calibration_gate
        )["metrics"],
        "final_gates": {
            "pretrained_bioclip": read_json(pretrained_final_gate),
            "finetuned_bioclip": read_json(finetuned_final_gate),
        },
        "finetuned_bioclip_checkpoint": str(final_bioclip_checkpoint),
        "submissions": plan["submission_outputs"],
        "official_unseen_labels_used_for_selection": False,
    }
    write_json(root / "summary.json", summary)
    return summary


def render_hybrid_slurm_script(
    spec_path: str | Path,
    *,
    resume: bool,
) -> str:
    """Render one allocation that owns the complete resumable pipeline."""
    plan = materialize_hybrid_plan(spec_path)
    config = load_config(plan["runs"][0]["config"])
    slurm = config["slurm"]
    command = [
        "python",
        str(ENTRYPOINT),
        "--spec",
        str(spec_path),
        "--run",
    ]
    if resume:
        command.append("--resume")
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={slurm.get('job_name', 'fish-dino-bioclip-hybrid')}",
        "#SBATCH --nodes=1",
        f"#SBATCH --gpus={int(slurm.get('gpus', 1))}",
        f"#SBATCH --cpus-per-task={int(slurm.get('cpus', 16))}",
        f"#SBATCH --mem={slurm.get('memory', '96G')}",
        f"#SBATCH --time={slurm.get('time_limit', '24:00:00')}",
    ]
    for key in ("account", "partition"):
        if slurm.get(key):
            lines.append(f"#SBATCH --{key}={slurm[key]}")
    log_dir = str(slurm.get("log_dir", "outputs/hybrid/slurm"))
    work_dir = Path(str(slurm.get("work_dir", REPOSITORY_ROOT)))
    if not work_dir.is_absolute():
        work_dir = (REPOSITORY_ROOT / work_dir).resolve()
    lines.extend(
        [
            f"#SBATCH --output={log_dir}/%x-%j.out",
            f"#SBATCH --error={log_dir}/%x-%j.err",
            "",
            "set -euo pipefail",
            f"cd {shlex.quote(str(work_dir))}",
        ]
    )
    if slurm.get("environment_activate"):
        lines.append(str(slurm["environment_activate"]))
    lines.extend(
        [
            "export OMP_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            "export TORCH_NCCL_ASYNC_ERROR_HANDLING=1",
            shlex.join(command),
        ]
    )
    return "\n".join(lines) + "\n"


def submit_hybrid_pipeline(
    spec_path: str | Path,
    *,
    resume: bool,
) -> str:
    """Submit the single hybrid allocation and return its Slurm job ID."""
    plan = materialize_hybrid_plan(spec_path)
    config = load_config(plan["runs"][0]["config"])
    slurm = config["slurm"]
    script = render_hybrid_slurm_script(spec_path, resume=resume)
    script_dir = Path(str(slurm.get("script_dir", "outputs/hybrid/slurm")))
    log_dir = Path(str(slurm.get("log_dir", "outputs/hybrid/slurm")))
    if not script_dir.is_absolute():
        script_dir = REPOSITORY_ROOT / script_dir
    if not log_dir.is_absolute():
        log_dir = REPOSITORY_ROOT / log_dir
    script_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return submit_slurm_script(
        script,
        script_dir / "hybrid-pipeline.sh",
    )


def plan_as_json(spec_path: str | Path) -> str:
    """Return the complete static dry-run plan plus rendered Slurm script."""
    plan = materialize_hybrid_plan(spec_path)
    return json.dumps(
        {
            "plan": plan,
            "commands": planned_commands(plan),
            "slurm_script": render_hybrid_slurm_script(
                spec_path, resume=False
            ),
        },
        indent=2,
        sort_keys=True,
    )
