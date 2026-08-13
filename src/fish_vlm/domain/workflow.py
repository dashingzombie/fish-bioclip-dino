"""Parallel one-GPU GenomeDK DAG for all-image domain adaptation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from fish_vlm.config import load_config
from fish_vlm.slurm.launcher import submit_slurm_script
from fish_vlm.slurm.templates import render_workflow_batch_script
from fish_vlm.utils.hashing import stable_json_hash
from fish_vlm.utils.io import write_json


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _absolute(value: str | Path) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else REPOSITORY_ROOT / path)


def build_all_data_plan(
    config_path: str | Path, *, resume: bool = False
) -> dict[str, Any]:
    """Render the exact dependency graph without submitting it."""
    common_path = _absolute(config_path)
    config = load_config(common_path)
    dino_domain = _absolute("configs/all_data/dino_domain.yaml")
    bioclip_domain = _absolute("configs/all_data/bioclip_domain.yaml")
    dino_finetune = _absolute("configs/all_data/dino_finetune.yaml")
    seen_inference = _absolute("configs/all_data/inference_seen.yaml")
    unseen_inference = _absolute("configs/all_data/inference_unseen.yaml")
    output = REPOSITORY_ROOT / "outputs/all_data"
    dino_domain_checkpoint = output / "dino_domain/checkpoints/best.pt"
    bioclip_checkpoint = output / "bioclip_domain/checkpoints/best.pt"
    dino_checkpoint = output / "dino_final/checkpoints/best.pt"
    test_predictions = output / "submission/test.json"
    unseen_predictions = output / "submission/unseen.json"
    merged = output / "submission/prediction.json"
    zipped = output / "submission/submission.zip"

    def cli(*arguments: str) -> list[str]:
        return ["python", "-m", "fish_vlm.cli", *arguments]

    jobs: list[dict[str, Any]] = [
        {
            "name": "prepare-metadata",
            "depends_on": [],
            "commands": [
                cli("prepare-prompts", "--config", common_path),
                cli("make-pseudo-unseen", "--config", common_path),
                cli("prepare-domain-data", "--config", common_path),
            ],
            "cache_scope": "shared",
            "stage_images": False,
            "gpus": 0,
            "partition": str(
                config["domain"]["cpu_jobs"].get("partition", "normal")
            ),
            "resources": dict(config["domain"]["cpu_jobs"]),
        },
        {
            "name": "bioclip-assets",
            "depends_on": ["prepare-metadata"],
            "commands": [
                cli("build-text-prototypes", "--config", common_path),
                cli("build-all-teacher-cache", "--config", common_path),
            ],
            "cache_scope": "shared",
            "stage_images": True,
            "gpus": 1,
            "partition": str(config["slurm"]["partition"]),
        },
        {
            "name": "dino-domain",
            "depends_on": ["prepare-metadata"],
            "commands": [
                cli(
                    "train-dino-domain",
                    "--config",
                    dino_domain,
                    *(["--resume"] if resume else []),
                )
            ],
            "cache_scope": "dino_domain",
            "stage_images": True,
            "gpus": 1,
            "partition": str(config["slurm"]["partition"]),
        },
        {
            "name": "bioclip-domain",
            "depends_on": ["bioclip-assets"],
            "commands": [
                cli(
                    "train-bioclip-domain",
                    "--config",
                    bioclip_domain,
                    *(["--resume"] if resume else []),
                )
            ],
            "cache_scope": "training",
            "stage_images": True,
            "gpus": 1,
            "partition": str(config["slurm"]["partition"]),
        },
        {
            "name": "dino-seen-finetune",
            "depends_on": ["dino-domain", "bioclip-assets"],
            "commands": [cli("train", "--config", dino_finetune)],
            "cache_scope": "training",
            "stage_images": True,
            "gpus": 1,
            "partition": str(config["slurm"]["partition"]),
        },
        {
            "name": "infer-seen",
            "depends_on": ["dino-seen-finetune"],
            "commands": [
                cli(
                    "infer",
                    "--config",
                    seen_inference,
                    "--checkpoint",
                    str(dino_checkpoint),
                    "--split",
                    "test",
                    "--output",
                    str(test_predictions),
                ),
            ],
            "cache_scope": "finalisation",
            "stage_images": True,
            "gpus": 1,
            "partition": str(config["slurm"]["partition"]),
        },
        {
            "name": "infer-unseen",
            "depends_on": ["dino-seen-finetune", "bioclip-domain"],
            "commands": [
                cli(
                    "infer",
                    "--config",
                    unseen_inference,
                    "--checkpoint",
                    str(dino_checkpoint),
                    "--bioclip-checkpoint",
                    str(bioclip_checkpoint),
                    "--split",
                    "unseen",
                    "--output",
                    str(unseen_predictions),
                ),
            ],
            "cache_scope": "finalisation",
            "stage_images": True,
            "gpus": 1,
            "partition": str(config["slurm"]["partition"]),
        },
        {
            "name": "package",
            "depends_on": ["infer-seen", "infer-unseen"],
            "commands": [
                cli(
                    "merge-submission",
                    "--test",
                    str(test_predictions),
                    "--unseen",
                    str(unseen_predictions),
                    "--output",
                    str(merged),
                ),
                cli(
                    "validate-submission",
                    "--config",
                    common_path,
                    "--submission",
                    str(merged),
                ),
                cli(
                    "package-submission",
                    "--submission",
                    str(merged),
                    "--output",
                    str(zipped),
                ),
            ],
            "cache_scope": "shared",
            "stage_images": False,
            "gpus": 0,
            "partition": str(
                config["domain"]["cpu_jobs"].get("partition", "normal")
            ),
            "resources": dict(config["domain"]["cpu_jobs"]),
        },
    ]
    for job in jobs:
        job_config = copy.deepcopy(config)
        job_config["slurm"]["partition"] = job["partition"]
        if job.get("resources"):
            job_config["workflow"].update(
                {
                    key: value
                    for key, value in job["resources"].items()
                    if key in {"cpus", "memory", "time_limit"}
                }
            )
        job["script"] = render_workflow_batch_script(
            job_config,
            job_name=f"fish-all-{job['name']}",
            commands=job["commands"],
            gpus=int(job["gpus"]),
            cache_scope=job["cache_scope"],
            stage_images=bool(job["stage_images"]),
        )
    plan = {
        "version": 1,
        "config": common_path,
        "scheduler": "GenomeDK Slurm",
        "resource_contract": {
            "cpu_jobs": "partition normal with no GPU request",
            "gpu_jobs": "one GPU per job on the configured GPU partition",
        },
        "parallel_branches": [
            ["dino-domain", "bioclip-assets"],
            ["dino-seen-finetune", "bioclip-domain"],
            ["infer-seen", "infer-unseen"],
        ],
        "official_test_and_unseen_use": "unlabeled domain adaptation only",
        "jobs": jobs,
        "outputs": {
            "dino_domain_checkpoint": str(dino_domain_checkpoint),
            "bioclip_domain_checkpoint": str(bioclip_checkpoint),
            "dino_seen_checkpoint": str(dino_checkpoint),
            "prediction": str(merged),
            "submission_zip": str(zipped),
        },
    }
    plan["plan_hash"] = stable_json_hash(
        {key: value for key, value in plan.items() if key != "plan_hash"}
    )
    write_json(output / "plan.json", plan)
    return plan


def submit_all_data_plan(
    config_path: str | Path, *, resume: bool = False
) -> dict[str, Any]:
    """Submit the DAG; sibling domain jobs share only the prepare dependency."""
    plan = build_all_data_plan(config_path, resume=resume)
    config = load_config(plan["config"])
    script_dir = REPOSITORY_ROOT / config["slurm"].get(
        "script_dir", "outputs/all_data/slurm"
    )
    log_dir = REPOSITORY_ROOT / config["slurm"].get(
        "log_dir", "outputs/all_data/slurm"
    )
    script_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    job_ids: dict[str, str] = {}
    for job in plan["jobs"]:
        dependencies = [job_ids[name] for name in job["depends_on"]]
        job_ids[job["name"]] = submit_slurm_script(
            job["script"],
            script_dir / f"{job['name']}.sh",
            dependency=":".join(dependencies) if dependencies else None,
        )
    state = {
        "status": "submitted",
        "plan_hash": plan["plan_hash"],
        "jobs": job_ids,
    }
    write_json(REPOSITORY_ROOT / "outputs/all_data/submission_state.json", state)
    return state


def plan_as_json(config_path: str | Path, *, resume: bool = False) -> str:
    return json.dumps(build_all_data_plan(config_path, resume=resume), indent=2)
