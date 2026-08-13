from __future__ import annotations

import subprocess

import pytest

from fish_vlm.config import load_config
from fish_vlm.slurm.launcher import submit_slurm_script


def test_image_staging_reports_progress_and_verifies_file_count() -> None:
    from fish_vlm.slurm.templates import render_workflow_batch_script

    config = load_config("configs/all_data/common.yaml")
    script = render_workflow_batch_script(
        config,
        job_name="staging-test",
        commands=[["true"]],
        gpus=1,
        stage_images=True,
    )
    assert "Starting node-local image staging" in script
    assert 'NODE_STAGE_DIR="/tmp/${NODE_STAGE_KEY}"' in script
    assert 'exec 9>"${NODE_STAGE_LOCK}"' in script
    assert "flock 9" in script
    assert "Reusing node-local image staging" in script
    assert ".fish-vlm-stage-complete" in script
    assert "--checkpoint=10000" in script
    assert "staging source checkpoint %u" in script
    assert "Image staging count mismatch" in script
    assert "Completed node-local image staging" in script


def test_dependency_aware_submission_records_parsable_job_id(
    monkeypatch,
    tmp_path,
) -> None:
    calls: list[list[str]] = []

    def completed(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="12345;cluster\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", completed)
    job_id = submit_slurm_script(
        "#!/usr/bin/env bash\ntrue\n",
        tmp_path / "job.sh",
        dependency="999",
    )
    assert job_id == "12345"
    assert calls == [
        [
            "sbatch",
            "--parsable",
            "--dependency=afterok:999",
            str(tmp_path / "job.sh"),
        ]
    ]


def test_submission_error_includes_slurm_stderr(
    monkeypatch,
    tmp_path,
) -> None:
    def failed(command, **kwargs):
        raise subprocess.CalledProcessError(
            1,
            command,
            stderr="sbatch: error: Invalid job id specified",
        )

    monkeypatch.setattr(subprocess, "run", failed)
    with pytest.raises(RuntimeError, match="Invalid job id specified"):
        submit_slurm_script(
            "#!/usr/bin/env bash\ntrue\n",
            tmp_path / "job.sh",
            dependency="999",
        )
