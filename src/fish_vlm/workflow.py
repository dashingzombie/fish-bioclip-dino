"""One-command local or dependency-chained SLURM pipeline execution."""

from __future__ import annotations

import subprocess
import sys
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fish_vlm.inference.validation import validate_submission
from fish_vlm.slurm.templates import render_workflow_batch_script
from fish_vlm.utils.hashing import stable_json_hash
from fish_vlm.utils.io import read_json, write_json, atomic_write_text


@dataclass(frozen=True)
class WorkflowStep:
    """One named command in the end-to-end pipeline."""

    name: str
    command: list[str]
    group: str
    gpus: int


def _root(config: dict[str, Any]) -> Path:
    config_path = Path(config["_config_path"]).resolve()
    return config_path.parent.parent if config_path.parent.name == "configs" else Path.cwd()


def _resolved(root: Path, value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else (root / path).resolve())


def build_workflow_steps(
    config: dict[str, Any],
    *,
    python_executable: str | None = None,
    slurm: bool = False,
) -> list[WorkflowStep]:
    """Build the exact ordered commands shared by local and SLURM modes."""
    workflow = config["workflow"]
    root = _root(config)
    python = "python" if slurm else (python_executable or sys.executable)
    base_config = _resolved(root, workflow.get("preparation_config", "configs/pipeline.yaml"))
    local_gpus = int(workflow.get("local_gpus", 1))
    training_gpus = int(config["slurm"].get("gpus", 2)) if slurm else local_gpus
    prepare_gpus = int(workflow.get("preparation_gpus", 1))
    final_gpus = int(workflow.get("final_gpus", 1))

    def cli(*arguments: str) -> list[str]:
        return [python, "-m", "fish_vlm.cli", *arguments]

    steps = [
        WorkflowStep(
            "prepare_prompts",
            cli("prepare-prompts", "--config", base_config),
            "preparation",
            prepare_gpus,
        ),
        WorkflowStep(
            "build_text_prototypes",
            cli("build-text-prototypes", "--config", base_config),
            "preparation",
            prepare_gpus,
        ),
        WorkflowStep(
            "make_pseudo_unseen",
            cli("make-pseudo-unseen", "--config", base_config),
            "preparation",
            prepare_gpus,
        ),
        WorkflowStep(
            "build_image_cache",
            cli("build-image-cache", "--config", base_config),
            "preparation",
            prepare_gpus,
        ),
        WorkflowStep(
            "build_teacher_cache",
            cli("build-teacher-cache", "--config", base_config),
            "preparation",
            prepare_gpus,
        ),
    ]
    for stage in workflow["training_stages"]:
        stage_config = _resolved(root, stage["config"])
        if training_gpus > 1:
            command = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={training_gpus}",
                "-m",
                "fish_vlm.cli",
                "train",
                "--config",
                stage_config,
            ]
        else:
            command = cli("train", "--config", stage_config)
        steps.append(
            WorkflowStep(
                str(stage["name"]),
                command,
                f"training_{stage['name']}",
                training_gpus,
            )
        )
        stage_name = str(stage["name"])
        stage_checkpoint = _resolved(root, stage["checkpoint"])
        stage_seen_config = _resolved(root, stage["seen_inference_config"])
        stage_unseen_config = _resolved(
            root,
            stage.get(
                "unseen_inference_config",
                workflow["unseen_inference_config"],
            ),
        )
        stage_output_dir = Path(
            _resolved(root, workflow["stage_submission_dir"])
        ) / stage_name
        stage_calibration = str(stage_output_dir / "calibration.json")
        stage_test = str(stage_output_dir / "test.json")
        stage_unseen = str(stage_output_dir / "unseen.json")
        stage_submission = str(stage_output_dir / "prediction.json")
        stage_zip = str(stage_output_dir / "submission.zip")
        stage_group = f"training_{stage_name}"
        steps.extend(
            [
                WorkflowStep(
                    f"{stage_name}_calibrate",
                    cli(
                        "calibrate",
                        "--config",
                        stage_seen_config,
                        "--checkpoint",
                        stage_checkpoint,
                        "--output",
                        stage_calibration,
                    ),
                    stage_group,
                    training_gpus,
                ),
                WorkflowStep(
                    f"{stage_name}_infer_test",
                    cli(
                        "infer",
                        "--config",
                        stage_seen_config,
                        "--checkpoint",
                        stage_checkpoint,
                        "--calibration",
                        stage_calibration,
                        "--output",
                        stage_test,
                    ),
                    stage_group,
                    training_gpus,
                ),
                WorkflowStep(
                    f"{stage_name}_infer_unseen",
                    cli(
                        "infer",
                        "--config",
                        stage_unseen_config,
                        "--checkpoint",
                        stage_checkpoint,
                        "--calibration",
                        stage_calibration,
                        "--output",
                        stage_unseen,
                    ),
                    stage_group,
                    training_gpus,
                ),
                WorkflowStep(
                    f"{stage_name}_merge_submission",
                    cli(
                        "merge-submission",
                        "--test",
                        stage_test,
                        "--unseen",
                        stage_unseen,
                        "--output",
                        stage_submission,
                    ),
                    stage_group,
                    training_gpus,
                ),
                WorkflowStep(
                    f"{stage_name}_validate_submission",
                    cli(
                        "validate-submission",
                        "--submission",
                        stage_submission,
                        "--config",
                        base_config,
                    ),
                    stage_group,
                    training_gpus,
                ),
                WorkflowStep(
                    f"{stage_name}_package_submission",
                    cli(
                        "package-submission",
                        "--submission",
                        stage_submission,
                        "--output",
                        stage_zip,
                    ),
                    stage_group,
                    training_gpus,
                ),
            ]
        )
    checkpoint = _resolved(root, workflow["final_checkpoint"])
    seen_config = _resolved(root, workflow["seen_inference_config"])
    unseen_config = _resolved(root, workflow["unseen_inference_config"])
    calibration = _resolved(root, workflow["calibration_output"])
    test_predictions = _resolved(root, workflow["test_predictions"])
    unseen_predictions = _resolved(root, workflow["unseen_predictions"])
    submission = _resolved(root, workflow["submission_output"])
    final_metrics = _resolved(root, workflow["final_metrics"])
    steps.extend(
        [
            WorkflowStep(
                "evaluate_final",
                cli(
                    "evaluate",
                    "--config",
                    seen_config,
                    "--checkpoint",
                    checkpoint,
                    "--output",
                    final_metrics,
                ),
                "finalisation",
                final_gpus,
            ),
            WorkflowStep(
                "calibrate",
                cli(
                    "calibrate",
                    "--config",
                    seen_config,
                    "--checkpoint",
                    checkpoint,
                    "--output",
                    calibration,
                ),
                "finalisation",
                final_gpus,
            ),
            WorkflowStep(
                "infer_test",
                cli(
                    "infer",
                    "--config",
                    seen_config,
                    "--checkpoint",
                    checkpoint,
                    "--calibration",
                    calibration,
                    "--output",
                    test_predictions,
                ),
                "finalisation",
                final_gpus,
            ),
            WorkflowStep(
                "infer_unseen",
                cli(
                    "infer",
                    "--config",
                    unseen_config,
                    "--checkpoint",
                    checkpoint,
                    "--calibration",
                    calibration,
                    "--output",
                    unseen_predictions,
                ),
                "finalisation",
                final_gpus,
            ),
            WorkflowStep(
                "merge_submission",
                cli(
                    "merge-submission",
                    "--test",
                    test_predictions,
                    "--unseen",
                    unseen_predictions,
                    "--output",
                    submission,
                ),
                "finalisation",
                final_gpus,
            ),
            WorkflowStep(
                "validate_submission",
                cli(
                    "validate-submission",
                    "--submission",
                    submission,
                    "--config",
                    base_config,
                ),
                "finalisation",
                final_gpus,
            ),
            WorkflowStep(
                "package_submission",
                cli(
                    "package-submission",
                    "--submission",
                    submission,
                    "--output",
                    str(Path(submission).with_name("submission.zip")),
                ),
                "finalisation",
                final_gpus,
            ),
            WorkflowStep(
                "write_summary",
                cli("pipeline-summary", "--config", base_config),
                "finalisation",
                final_gpus,
            ),
        ]
    )
    return steps


def _workflow_state_path(config: dict[str, Any]) -> Path:
    output = Path(config.get("output_dir", "outputs"))
    if not output.is_absolute():
        output = _root(config) / output
    return output / "pipeline" / "workflow_state.json"


def run_local_workflow(
    config: dict[str, Any],
    *,
    dry_run: bool,
    force: bool,
) -> dict[str, Any]:
    """Run every command sequentially, stopping immediately on failure."""
    steps = build_workflow_steps(config)
    commands = [{"name": step.name, "command": step.command} for step in steps]
    workflow_hash = stable_json_hash(commands)
    if dry_run:
        return {
            "mode": "local",
            "dry_run": True,
            "workflow_hash": workflow_hash,
            "steps": commands,
        }
    state_path = _workflow_state_path(config)
    state = read_json(state_path) if state_path.exists() and not force else None
    if state is not None and state.get("workflow_hash") != workflow_hash:
        raise ValueError(
            "Existing workflow state belongs to a different command plan; use --force"
        )
    if state is None:
        state = {"mode": "local", "workflow_hash": workflow_hash, "steps": {}}
    for step in steps:
        if state["steps"].get(step.name, {}).get("status") == "completed" and not force:
            continue
        state["steps"][step.name] = {"status": "running", "command": step.command}
        write_json(state_path, state)
        try:
            subprocess.run(step.command, check=True, cwd=_root(config))
        except subprocess.CalledProcessError as error:
            state["steps"][step.name] = {
                "status": "failed",
                "command": step.command,
                "returncode": error.returncode,
            }
            state["status"] = "failed"
            write_json(state_path, state)
            raise
        state["steps"][step.name] = {"status": "completed", "command": step.command}
        write_json(state_path, state)
    state["status"] = "completed"
    write_json(state_path, state)
    return state


def run_slurm_workflow(
    config: dict[str, Any],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    """Submit grouped jobs linked by strict ``afterok`` dependencies."""
    steps = build_workflow_steps(config, slurm=True)
    groups: list[tuple[str, int, list[list[str]]]] = []
    for step in steps:
        if groups and groups[-1][0] == step.group:
            groups[-1][2].append(step.command)
        else:
            groups.append((step.group, step.gpus, [step.command]))
    rendered = [
        {
            "name": name,
            "gpus": gpus,
            "depends_on": None if index == 0 else groups[index - 1][0],
            "script": render_workflow_batch_script(
                config,
                job_name=f"fish-{name.replace('_', '-')}",
                commands=commands,
                gpus=gpus,
                cache_scope=(
                    "shared"
                    if name == "preparation"
                    else "finalisation"
                    if name == "finalisation"
                    else "training"
                ),
                stage_images=name == "preparation",
            ),
        }
        for index, (name, gpus, commands) in enumerate(groups)
    ]
    workflow_hash = stable_json_hash(
        [{"name": job["name"], "script": job["script"]} for job in rendered]
    )
    if dry_run:
        return {
            "mode": "slurm",
            "dry_run": True,
            "workflow_hash": workflow_hash,
            "jobs": rendered,
        }
    script_dir = Path(config["slurm"].get("script_dir", "outputs/slurm"))
    if not script_dir.is_absolute():
        script_dir = _root(config) / script_dir
    script_dir = script_dir / "pipeline"
    job_ids: dict[str, str] = {}
    previous_job_id: str | None = None
    for job in rendered:
        script_path = script_dir / f"{job['name']}.sh"
        atomic_write_text(script_path, str(job["script"]))
        command = ["sbatch", "--parsable"]
        if previous_job_id is not None:
            command.append(f"--dependency=afterok:{previous_job_id}")
        command.append(str(script_path))
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        job_id = result.stdout.strip().split(";", maxsplit=1)[0]
        if not job_id:
            raise RuntimeError(f"sbatch returned no job ID for {job['name']}")
        job_ids[str(job["name"])] = job_id
        previous_job_id = job_id
    state = {
        "mode": "slurm",
        "status": "submitted",
        "workflow_hash": workflow_hash,
        "jobs": job_ids,
    }
    write_json(_workflow_state_path(config), state)
    return state


def run_all(
    config: dict[str, Any],
    *,
    mode: str,
    dry_run: bool = False,
    force: bool = False,
    gpus: int | None = None,
) -> dict[str, Any]:
    """Run or submit the entire pipeline in one call."""
    if gpus is not None:
        if gpus < 1:
            raise ValueError("gpus must be at least one")
        config = copy.deepcopy(config)
        if mode == "local":
            config["workflow"]["local_gpus"] = gpus
        elif mode == "slurm":
            config["slurm"]["gpus"] = gpus
    if mode == "local":
        return run_local_workflow(config, dry_run=dry_run, force=force)
    if mode == "slurm":
        if force:
            raise ValueError("--force is only meaningful for local workflow state")
        return run_slurm_workflow(config, dry_run=dry_run)
    raise ValueError("mode must be local or slurm")


def write_pipeline_summary(config: dict[str, Any]) -> dict[str, Any]:
    """Write one compact result document for human interpretation."""
    workflow = config["workflow"]
    root = _root(config)
    stages: dict[str, Any] = {}
    for stage in workflow["training_stages"]:
        metrics_path = Path(_resolved(root, stage["metrics"]))
        checkpoint_path = Path(_resolved(root, stage["checkpoint"]))
        stage_submission_zip = (
            Path(_resolved(root, workflow["stage_submission_dir"]))
            / str(stage["name"])
            / "submission.zip"
        )
        if not metrics_path.exists():
            raise FileNotFoundError(f"Stage metrics are missing: {metrics_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Stage checkpoint is missing: {checkpoint_path}")
        if not stage_submission_zip.exists():
            raise FileNotFoundError(
                f"Stage submission ZIP is missing: {stage_submission_zip}"
            )
        stages[str(stage["name"])] = {
            "metrics_path": str(metrics_path),
            "checkpoint": str(checkpoint_path),
            "submission_zip": str(stage_submission_zip),
            "results": read_json(metrics_path),
        }
    final_metrics_path = Path(_resolved(root, workflow["final_metrics"]))
    calibration_path = Path(_resolved(root, workflow["calibration_output"]))
    submission_path = Path(_resolved(root, workflow["submission_output"]))
    final_metrics = read_json(final_metrics_path)
    validation = validate_submission(submission_path, config)
    branch_accuracies = {
        key.removesuffix("_accuracy"): value
        for key, value in final_metrics.items()
        if key.endswith("_accuracy")
        and not key.endswith("balanced_accuracy")
        and not key.endswith("top5_accuracy")
    }
    best_branch = max(branch_accuracies, key=branch_accuracies.get) if branch_accuracies else None
    summary = {
        "status": "complete",
        "selection_metric": config["validation"]["selection_metric"],
        "final_checkpoint": _resolved(root, workflow["final_checkpoint"]),
        "stages": stages,
        "final_evaluation": final_metrics,
        "calibration": read_json(calibration_path),
        "submission": validation,
        "interpretation": {
            "best_final_seen_branch": best_branch,
            "best_final_seen_accuracy": (
                branch_accuracies[best_branch] if best_branch is not None else None
            ),
            "official_test_mode": config["inference"]["test"]["mode"],
            "official_unseen_mode": config["inference"]["unseen"]["mode"],
        },
    }
    output_path = Path(_resolved(root, workflow["summary_output"]))
    write_json(output_path, summary)
    return summary
