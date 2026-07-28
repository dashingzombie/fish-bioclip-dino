from __future__ import annotations

import subprocess
import copy
from pathlib import Path

from fish_vlm.config import load_config
from fish_vlm.workflow import (
    build_slurm_workflow_steps,
    submit_bootstrap_pipeline,
)


def test_bootstrap_contains_complete_ordered_pipeline() -> None:
    config = load_config("configs/pipeline.yaml")
    steps = build_slurm_workflow_steps(config)
    names = [step.name for step in steps]
    assert names[:5] == [
        "prepare_prompts",
        "build_text_prototypes",
        "make_pseudo_unseen",
        "build_image_cache",
        "build_teacher_cache",
    ]
    for stage in (
        "projection_only",
        "final_block",
        "joint_supervised_text",
        "bioclip_adapter",
    ):
        assert stage in names
        assert f"{stage}_package_submission" in names
        assert names.index(stage) < names.index(f"{stage}_package_submission")
    assert names[-4:] == [
        "merge_submission",
        "validate_submission",
        "package_submission",
        "write_summary",
    ]
    projection = next(step for step in steps if step.name == "projection_only")
    assert projection.command[0] == "torchrun"
    assert "--nproc_per_node=4" in projection.command


def test_slurm_dry_run_never_calls_sbatch(monkeypatch) -> None:
    config = load_config("configs/pipeline.yaml")

    def forbidden(*args, **kwargs):
        raise AssertionError("subprocess.run must not be called during dry-run")

    monkeypatch.setattr(subprocess, "run", forbidden)
    result = submit_bootstrap_pipeline(config, dry_run=True, gpus=3)
    assert result["dry_run"] is True
    assert len(result["jobs"]) == 6
    assert result["jobs"][0]["depends_on"] is None
    assert result["jobs"][1]["depends_on"] == "preparation"
    assert result["jobs"][1]["gpus"] == 3
    assert "--dependency" not in result["jobs"][0]["script"]
    preparation = result["jobs"][0]["script"]
    training = result["jobs"][1]["script"]
    assert 'FISH_VLM_CACHE_DIR="${SHARED_CACHE_DIR}"' in preparation
    assert 'tar --directory="${SHARED_IMAGES_DIR}"' in preparation
    assert 'export FISH_VLM_IMAGES_DIR' in preparation
    assert 'FISH_VLM_CACHE_DIR="${NODE_TMPDIR}"/fish-vlm-cache' in training
    assert "CACHE_COPY_PIDS=()" in training
    assert 'cp --archive --reflink=auto "${source_path}"' in training
    assert "image_transforms/test/dino.npy" in training
    assert 'export HF_HOME="${FISH_VLM_CACHE_DIR}/huggingface"' in training
    assert 'export TORCH_HOME="${FISH_VLM_CACHE_DIR}/torch"' in training


def test_slurm_submission_uses_afterok_job_ids(
    monkeypatch, tmp_path: Path
) -> None:
    config = copy.deepcopy(load_config("configs/pipeline.yaml"))
    config["slurm"]["script_dir"] = str(tmp_path / "scripts")
    config["output_dir"] = str(tmp_path / "outputs")
    calls: list[list[str]] = []

    def completed(command, **kwargs):
        calls.append(command)
        job_id = str(100 + len(calls) - 1)
        return subprocess.CompletedProcess(command, 0, stdout=f"{job_id}\n", stderr="")

    monkeypatch.setattr(subprocess, "run", completed)
    result = submit_bootstrap_pipeline(config, dry_run=False)
    assert result["jobs"]["preparation"] == "100"
    assert result["jobs"]["finalisation"] == "105"
    assert calls[0][:2] == ["sbatch", "--parsable"]
    assert "--dependency=afterok:100" in calls[1]
    assert "--dependency=afterok:104" in calls[-1]
