"""Internal bootstrap workflow used by the unified sweep entry point."""

from __future__ import annotations

import copy
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fish_vlm.inference.validation import validate_submission
from fish_vlm.slurm.launcher import submit_slurm_script
from fish_vlm.slurm.templates import render_workflow_batch_script
from fish_vlm.utils.hashing import stable_json_hash
from fish_vlm.utils.io import read_json, write_json


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


def build_slurm_workflow_steps(
    config: dict[str, Any],
) -> list[WorkflowStep]:
    """Build the exact ordered bootstrap commands for Slurm."""
    workflow = config["workflow"]
    root = _root(config)
    python = "python"
    base_config = _resolved(root, workflow.get("preparation_config", "configs/pipeline.yaml"))
    training_gpus = int(config["slurm"].get("gpus", 2))
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
        WorkflowStep(
            "evaluate_bioclip_zero_shot",
            cli(
                "evaluate",
                "--config",
                base_config,
                "--output",
                _resolved(
                    root,
                    workflow.get(
                        "zero_shot_output",
                        "outputs/metrics/bioclip_zero_shot.json",
                    ),
                ),
            ),
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
        if stage_name == "joint_supervised_text":
            steps.append(
                WorkflowStep(
                    "verify_unseen_inference",
                    cli(
                        "verify-unseen-inference",
                        "--config",
                        _resolved(root, workflow["unseen_inference_config"]),
                        "--checkpoint",
                        stage_checkpoint,
                        "--output",
                        _resolved(
                            root,
                            workflow.get(
                                "unseen_audit_output",
                                "outputs/metrics/unseen_inference_audit.json",
                            ),
                        ),
                    ),
                    stage_group,
                    training_gpus,
                )
            )
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
        if stage_name == "joint_supervised_text":
            steps.append(
                WorkflowStep(
                    "evaluate_dino_stages",
                    cli(
                        "evaluate-stages",
                        "--config",
                        base_config,
                        "--output",
                        _resolved(
                            root,
                            workflow.get(
                                "stage_comparison_output",
                                "outputs/metrics/stage_comparison.json",
                            ),
                        ),
                    ),
                    stage_group,
                    training_gpus,
                )
            )
    seen_checkpoint = _resolved(
        root,
        workflow.get(
            "final_seen_checkpoint", workflow["final_checkpoint"]
        ),
    )
    unseen_checkpoint = _resolved(
        root,
        workflow.get(
            "final_unseen_checkpoint", workflow["final_checkpoint"]
        ),
    )
    seen_config = _resolved(root, workflow["seen_inference_config"])
    unseen_config = _resolved(root, workflow["unseen_inference_config"])
    calibration = _resolved(root, workflow["calibration_output"])
    unseen_calibration = _resolved(
        root,
        workflow.get(
            "unseen_calibration_output",
            "outputs/metrics/unseen_calibration.json",
        ),
    )
    model_selection = _resolved(
        root,
        workflow.get(
            "model_selection_output",
            "outputs/metrics/model_selection.json",
        ),
    )
    test_predictions = _resolved(root, workflow["test_predictions"])
    unseen_predictions = _resolved(root, workflow["unseen_predictions"])
    submission = _resolved(root, workflow["submission_output"])
    final_metrics = _resolved(root, workflow["final_metrics"])
    steps.extend(
        [
            WorkflowStep(
                "select_models",
                cli(
                    "select-models",
                    "--config",
                    base_config,
                    "--output",
                    model_selection,
                ),
                "finalisation",
                final_gpus,
            ),
            WorkflowStep(
                "evaluate_final",
                cli(
                    "evaluate",
                    "--config",
                    seen_config,
                    "--checkpoint",
                    seen_checkpoint,
                    "--selection-report",
                    model_selection,
                    "--purpose",
                    "seen",
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
                    seen_checkpoint,
                    "--selection-report",
                    model_selection,
                    "--purpose",
                    "seen",
                    "--output",
                    calibration,
                ),
                "finalisation",
                final_gpus,
            ),
            WorkflowStep(
                "calibrate_unseen_checkpoint",
                cli(
                    "calibrate",
                    "--config",
                    seen_config,
                    "--checkpoint",
                    unseen_checkpoint,
                    "--selection-report",
                    model_selection,
                    "--purpose",
                    "unseen",
                    "--output",
                    unseen_calibration,
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
                    seen_checkpoint,
                    "--selection-report",
                    model_selection,
                    "--purpose",
                    "seen",
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
                    unseen_checkpoint,
                    "--selection-report",
                    model_selection,
                    "--purpose",
                    "unseen",
                    "--calibration",
                    unseen_calibration,
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


_ACTIVE_SLURM_STATES = {
    "CONFIGURING",
    "COMPLETING",
    "PENDING",
    "RUNNING",
    "SUSPENDED",
}


def _slurm_job_states(job_ids: list[str]) -> dict[str, str]:
    """Return base Slurm states for the requested non-array job IDs."""
    if not job_ids:
        return {}
    result = subprocess.run(
        [
            "sacct",
            "--noheader",
            "--parsable2",
            "--jobs",
            ",".join(job_ids),
            "--format=JobIDRaw,State",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    requested = set(job_ids)
    states: dict[str, str] = {}
    for line in result.stdout.splitlines():
        fields = line.strip().split("|", maxsplit=2)
        if len(fields) < 2 or fields[0] not in requested:
            continue
        states[fields[0]] = fields[1].split(maxsplit=1)[0].rstrip("+")
    return states


def submit_bootstrap_pipeline(
    config: dict[str, Any],
    *,
    dry_run: bool,
    gpus: int | None = None,
    existing_jobs: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Render, submit, or resume bootstrap jobs linked by strict dependencies."""
    if gpus is not None:
        if gpus < 1:
            raise ValueError("gpus must be at least one")
        config = copy.deepcopy(config)
        config["slurm"]["gpus"] = gpus
    steps = build_slurm_workflow_steps(config)
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
    start_index = 0
    resumed_from: str | None = None
    slurm_states: dict[str, str] = {}
    if existing_jobs:
        existing_ids = [
            str(existing_jobs[str(job["name"])])
            for job in rendered
            if str(job["name"]) in existing_jobs
        ]
        slurm_states = _slurm_job_states(existing_ids)
        for index, job in enumerate(rendered):
            name = str(job["name"])
            job_id = str(existing_jobs.get(name, ""))
            state = slurm_states.get(job_id, "UNKNOWN")
            if state == "COMPLETED":
                job_ids[name] = job_id
                start_index = index + 1
                continue
            if state in _ACTIVE_SLURM_STATES:
                state = {
                    "mode": "slurm",
                    "status": "submitted",
                    "workflow_hash": workflow_hash,
                    "jobs": {
                        str(key): str(value)
                        for key, value in existing_jobs.items()
                    },
                    "slurm_states": slurm_states,
                }
                write_json(_workflow_state_path(config), state)
                return state
            start_index = index
            resumed_from = name
            break
        else:
            state = {
                "mode": "slurm",
                "status": "complete",
                "workflow_hash": workflow_hash,
                "jobs": job_ids,
                "slurm_states": slurm_states,
            }
            write_json(_workflow_state_path(config), state)
            return state
    previous_job_id: str | None = None
    for job in rendered[start_index:]:
        script_path = script_dir / f"{job['name']}.sh"
        job_id = submit_slurm_script(
            str(job["script"]),
            script_path,
            dependency=previous_job_id,
        )
        if not job_id:
            raise RuntimeError(f"sbatch returned no job ID for {job['name']}")
        job_ids[str(job["name"])] = job_id
        previous_job_id = job_id
    state = {
        "mode": "slurm",
        "status": "resubmitted" if resumed_from is not None else "submitted",
        "workflow_hash": workflow_hash,
        "jobs": job_ids,
    }
    if resumed_from is not None:
        state["resumed_from"] = resumed_from
        state["previous_slurm_states"] = slurm_states
    write_json(_workflow_state_path(config), state)
    return state


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
    unseen_calibration_path = Path(
        _resolved(root, workflow["unseen_calibration_output"])
    )
    stage_comparison_path = Path(
        _resolved(root, workflow["stage_comparison_output"])
    )
    model_selection_path = Path(
        _resolved(root, workflow["model_selection_output"])
    )
    zero_shot_path = Path(_resolved(root, workflow["zero_shot_output"]))
    submission_path = Path(_resolved(root, workflow["submission_output"]))
    final_metrics = read_json(final_metrics_path)
    validation = validate_submission(submission_path, config)
    branch_accuracies = {
        key.removesuffix("_accuracy"): value
        for key, value in final_metrics.items()
        if key.endswith("_accuracy")
        and not key.endswith("balanced_accuracy")
        and not key.endswith("top5_accuracy")
        and isinstance(value, (int, float))
    }
    selected_branch = final_metrics.get("selection_branch")
    best_branch = (
        str(selected_branch)
        if selected_branch in branch_accuracies
        else max(branch_accuracies, key=branch_accuracies.get)
        if branch_accuracies
        else None
    )
    summary = {
        "status": "complete",
        "selection_metric": config["validation"]["selection_metric"],
        "purpose_checkpoint_defaults": {
            "seen": _resolved(root, workflow["final_seen_checkpoint"]),
            "unseen": _resolved(
                root, workflow["final_unseen_checkpoint"]
            ),
            "joint": _resolved(root, workflow["final_joint_checkpoint"]),
        },
        "purpose_checkpoint_selection": read_json(
            model_selection_path
        )["selection"],
        "stages": stages,
        "bioclip_zero_shot": read_json(zero_shot_path),
        "dino_stage_comparison": read_json(stage_comparison_path),
        "all_model_selection": read_json(model_selection_path),
        "final_evaluation": final_metrics,
        "calibration": read_json(calibration_path),
        "unseen_checkpoint_calibration": read_json(
            unseen_calibration_path
        ),
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
