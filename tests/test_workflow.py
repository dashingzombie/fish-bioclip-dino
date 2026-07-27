from __future__ import annotations

import subprocess
import copy
from pathlib import Path

from fish_vlm.config import load_config
from fish_vlm.workflow import build_workflow_steps, run_all


def test_local_workflow_contains_complete_ordered_pipeline() -> None:
    config = load_config("configs/pipeline.yaml")
    steps = build_workflow_steps(config, python_executable="python-test")
    names = [step.name for step in steps]
    assert names[:4] == [
        "prepare_prompts",
        "build_text_prototypes",
        "make_pseudo_unseen",
        "build_teacher_cache",
    ]
    assert names[4:8] == [
        "projection_only",
        "final_block",
        "joint_supervised_text",
        "bioclip_adapter",
    ]
    assert names[-3:] == [
        "merge_submission",
        "validate_submission",
        "write_summary",
    ]
    result = run_all(config, mode="local", dry_run=True)
    assert result["dry_run"] is True
    assert len(result["steps"]) == 15
    multi_gpu = run_all(config, mode="local", dry_run=True, gpus=2)
    projection = next(
        step for step in multi_gpu["steps"] if step["name"] == "projection_only"
    )
    assert projection["command"][0] == "torchrun"
    assert "--nproc_per_node=2" in projection["command"]


def test_slurm_dry_run_never_calls_sbatch(monkeypatch) -> None:
    config = load_config("configs/pipeline.yaml")

    def forbidden(*args, **kwargs):
        raise AssertionError("subprocess.run must not be called during dry-run")

    monkeypatch.setattr(subprocess, "run", forbidden)
    result = run_all(config, mode="slurm", dry_run=True, gpus=3)
    assert result["dry_run"] is True
    assert len(result["jobs"]) == 6
    assert result["jobs"][0]["depends_on"] is None
    assert result["jobs"][1]["depends_on"] == "preparation"
    assert result["jobs"][1]["gpus"] == 3
    assert "--dependency" not in result["jobs"][0]["script"]


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
    result = run_all(config, mode="slurm")
    assert result["jobs"]["preparation"] == "100"
    assert result["jobs"]["finalisation"] == "105"
    assert calls[0][:2] == ["sbatch", "--parsable"]
    assert "--dependency=afterok:100" in calls[1]
    assert "--dependency=afterok:104" in calls[-1]
