from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import yaml
import pytest

from fish_vlm.config import load_config
from fish_vlm.sweeps.joint import (
    _phase_candidates,
    _phase_config,
    confirmation_ranking,
    run_joint_sweeps,
)
from fish_vlm.utils.io import write_json


def test_makefile_exposes_only_the_unified_pipeline_targets() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    assert "everything:" in makefile
    assert "everything-dry-run:" in makefile
    assert "everything-resume:" in makefile
    assert "scripts/run_joint_sweeps.py --everything --submit" in makefile
    assert "scripts/run_all.py" not in makefile
    assert "run-all-" not in makefile
    assert "sweep-dry-run:" not in makefile


def _index(root: Path) -> dict:
    return json.loads((root / "run_index.json").read_text(encoding="utf-8"))


def _complete(run: dict, *, score: float, harmonic: float = 0.5) -> None:
    checkpoint = Path(run["checkpoint_path"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.touch()
    write_json(
        run["metrics_path"],
        {
            "estimated_overall_accuracy": score,
            "seen_unseen_harmonic_mean": harmonic,
            "best_step": 2750,
        },
    )


def test_phase_generation_counts_and_conditional_overrides() -> None:
    expected = {
        "loss": 30,
        "optimiser": 15,
        "architecture": 20,
        "training": 18,
    }
    for phase, count in expected.items():
        candidates, skipped = _phase_candidates(_phase_config(phase))
        assert len(candidates) == count
        assert not skipped

    loss, _ = _phase_candidates(_phase_config("loss"))
    for key, values in _phase_config("loss")["sweep"]["parameters"].items():
        counts = Counter(candidate[key] for candidate in loss)
        assert set(counts) == set(values)
        assert min(counts.values()) > 1

    optimiser, _ = _phase_candidates(_phase_config("optimiser"))
    assert all(
        not any(key.endswith("_lr_multiplier") for key in candidate)
        for candidate in optimiser
    )

    architecture, _ = _phase_candidates(_phase_config("architecture"))
    for candidate in architecture:
        if not candidate["loss.branch_consistency.enabled"]:
            assert "loss.branch_consistency.weight" not in candidate
            assert "loss.branch_consistency.method" not in candidate
        else:
            assert candidate["loss.branch_consistency.method"] == "js"
            assert "loss.branch_consistency.weight" in candidate


def test_deterministic_unique_names_and_resolved_config(
    tmp_path: Path,
    capsys,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    run_joint_sweeps(phase="loss", dry_run=True, output_root=first_root)
    capsys.readouterr()
    run_joint_sweeps(phase="loss", dry_run=True, output_root=second_root)
    capsys.readouterr()
    first = _index(first_root)["runs"]
    second = _index(second_root)["runs"]
    assert [run["id"] for run in first] == [run["id"] for run in second]
    assert [run["name"] for run in first] == [run["name"] for run in second]
    assert len({run["name"] for run in first}) == 30

    resolved = load_config(first[0]["config_path"])
    assert resolved["seed"] == 42
    assert resolved["training"]["max_steps"] == 10000
    assert resolved["training"]["validation_interval_steps"] == 250
    assert (
        resolved["training"]["early_stopping_patience_evaluations"] == 6
    )
    assert (
        resolved["validation"]["selection_metric"]
        == "estimated_overall_accuracy"
    )
    assert resolved["sweep_metadata"]["effective_global_batch_size"] == 2048
    raw = yaml.safe_load(
        Path(first[0]["config_path"]).read_text(encoding="utf-8")
    )
    assert "defaults" not in raw


def test_phase_inherits_best_previous_resolved_configuration(
    tmp_path: Path,
    capsys,
) -> None:
    root = tmp_path / "sweep"
    run_joint_sweeps(phase="loss", dry_run=True, output_root=root)
    capsys.readouterr()
    index = _index(root)
    selected = index["runs"][0]
    _complete(selected, score=0.9)

    result = run_joint_sweeps(
        phase="optimiser",
        dry_run=True,
        output_root=root,
    )
    capsys.readouterr()
    assert result["generated_runs"] == 15
    updated = _index(root)
    optimiser = [
        run for run in updated["runs"] if run["phase"] == "optimiser"
    ]
    assert {run["parent_run_id"] for run in optimiser} == {selected["id"]}
    resolved = load_config(optimiser[0]["config_path"])
    for key, value in selected["parameters"].items():
        section, component, field = key.split(".")
        assert resolved[section][component][field] == value


def test_confirmation_materialises_three_seeds_for_top_eight(
    tmp_path: Path,
    capsys,
) -> None:
    root = tmp_path / "sweep"
    run_joint_sweeps(phase="loss", dry_run=True, output_root=root)
    capsys.readouterr()
    index = _index(root)
    for rank, run in enumerate(index["runs"][:8], start=1):
        _complete(run, score=1.0 - rank / 100, harmonic=0.8)
    result = run_joint_sweeps(
        confirm_top=8,
        dry_run=True,
        output_root=root,
    )
    capsys.readouterr()
    assert result["generated_runs"] == 24
    confirmations = [
        run
        for run in _index(root)["runs"]
        if run["phase"] == "confirmation"
    ]
    assert len(confirmations) == 24
    assert {run["seed"] for run in confirmations} == {7, 42, 123}
    assert len({run["name"] for run in confirmations}) == 24
    for run in confirmations:
        resolved = load_config(run["config_path"])
        assert resolved["validation"]["pseudo_unseen"]["split_seed"] == 42

    source_ids = list(dict.fromkeys(run["source_run_id"] for run in confirmations))
    for run in confirmations:
        if run["source_run_id"] == source_ids[0]:
            _complete(run, score=0.8, harmonic=0.7)
        elif run["source_run_id"] == source_ids[1]:
            _complete(run, score=0.7, harmonic=0.9)
    ranking = confirmation_ranking(_index(root))
    assert ranking[0]["source_run_id"] == source_ids[0]
    assert ranking[0]["mean_estimated_overall_accuracy"] == pytest.approx(0.8)
    assert ranking[0]["worst_seed_estimated_overall_accuracy"] == pytest.approx(
        0.8
    )


def test_dry_run_does_not_invoke_slurm(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run must not invoke the Slurm launcher")

    monkeypatch.setattr(
        "fish_vlm.sweeps.joint.launch_slurm",
        forbidden,
    )
    result = run_joint_sweeps(
        phase="loss",
        dry_run=True,
        output_root=tmp_path / "sweep",
    )
    capsys.readouterr()
    assert result["job_ids"] == []


def test_array_submission_records_jobs_and_resume_skips_completed(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    root = tmp_path / "sweep"
    submitted_sizes: list[int] = []

    def fake_launch(config, *, dry_run):
        assert dry_run is False
        submitted_sizes.append(len(config["slurm"]["array_configs"]))
        return str(1200 + len(submitted_sizes))

    monkeypatch.setattr(
        "fish_vlm.sweeps.joint.launch_slurm",
        fake_launch,
    )
    first = run_joint_sweeps(
        phase="loss",
        submit=True,
        max_concurrent=8,
        output_root=root,
    )
    capsys.readouterr()
    assert first["job_ids"] == ["1201"]
    assert submitted_sizes == [30]
    index = _index(root)
    assert {run["job_id"] for run in index["runs"]} == {"1201"}
    assert {run["array_task_id"] for run in index["runs"]} == set(range(30))

    with pytest.raises(ValueError, match="--resume"):
        run_joint_sweeps(
            phase="loss",
            submit=True,
            max_concurrent=8,
            output_root=root,
        )
    capsys.readouterr()

    completed = _index(root)["runs"][0]
    _complete(completed, score=0.9)
    resumed = run_joint_sweeps(
        phase="loss",
        submit=True,
        resume=True,
        max_concurrent=4,
        output_root=root,
    )
    capsys.readouterr()
    assert resumed["job_ids"] == ["1202"]
    assert submitted_sizes == [30, 29]
    state = json.loads((root / "state.json").read_text(encoding="utf-8"))
    assert state["phases"]["loss"]["job_ids"] == ["1201", "1202"]
    assert state["phases"]["loss"]["max_concurrent"] == 4


def test_everything_dry_run_plans_pipeline_without_submission(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    calls: list[dict] = []

    def fake_bootstrap(config, **kwargs):
        calls.append(kwargs)
        return {
            "dry_run": True,
            "workflow_hash": "workflow",
            "jobs": [
                {
                    "name": "preparation",
                    "gpus": 1,
                    "depends_on": None,
                }
            ],
        }

    def forbidden(*args, **kwargs):
        raise AssertionError("everything dry-run must not submit jobs")

    monkeypatch.setattr(
        "fish_vlm.sweeps.joint.submit_bootstrap_pipeline",
        fake_bootstrap,
    )
    monkeypatch.setattr("fish_vlm.sweeps.joint.launch_slurm", forbidden)
    monkeypatch.setattr(
        "fish_vlm.sweeps.joint.submit_slurm_script",
        forbidden,
    )
    result = run_joint_sweeps(
        everything=True,
        dry_run=True,
        output_root=tmp_path / "sweep",
    )
    capsys.readouterr()
    assert calls == [{"dry_run": True, "gpus": 4}]
    assert result["generated_runs"] == 30
    assert result["job_ids"] == []
    assert result["controller_job_ids"] == []


def test_everything_submits_pipeline_array_and_next_controller(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    observed: dict[str, object] = {}

    def fake_bootstrap(config, **kwargs):
        assert kwargs == {"dry_run": False, "gpus": 4}
        return {
            "mode": "slurm",
            "status": "submitted",
            "workflow_hash": "workflow",
            "jobs": {
                "preparation": "800",
                "training_projection_only": "801",
                "training_final_block": "802",
                "training_joint_supervised_text": "803",
                "training_bioclip_adapter": "804",
                "finalisation": "805",
            },
        }

    def fake_launch(config, *, dry_run):
        assert dry_run is False
        observed["array_dependency"] = config["slurm"]["dependency"]
        observed["array_size"] = len(config["slurm"]["array_configs"])
        return "900"

    def fake_submit(script, path, *, dependency):
        observed["controller_dependency"] = dependency
        observed["controller_script"] = script
        return "901"

    monkeypatch.setattr(
        "fish_vlm.sweeps.joint.submit_bootstrap_pipeline",
        fake_bootstrap,
    )
    monkeypatch.setattr("fish_vlm.sweeps.joint.launch_slurm", fake_launch)
    monkeypatch.setattr(
        "fish_vlm.sweeps.joint.submit_slurm_script",
        fake_submit,
    )
    result = run_joint_sweeps(
        everything=True,
        submit=True,
        max_concurrent=8,
        output_root=tmp_path / "sweep",
    )
    capsys.readouterr()
    assert result["job_ids"] == ["900"]
    assert result["controller_job_ids"] == ["901"]
    assert observed["array_dependency"] == "805"
    assert observed["array_size"] == 30
    assert observed["controller_dependency"] == "900"
    assert "--phase optimiser" in str(observed["controller_script"])
    state = json.loads(
        (tmp_path / "sweep" / "state.json").read_text(encoding="utf-8")
    )
    assert state["master_pipeline"]["jobs"]["finalisation"] == "805"
