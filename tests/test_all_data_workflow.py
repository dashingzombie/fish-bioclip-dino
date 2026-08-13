"""GenomeDK DAG and configuration contracts for the all-data workflow."""

from __future__ import annotations

from fish_vlm.config import load_config
from fish_vlm.domain.workflow import build_all_data_plan


def test_all_data_plan_parallelises_independent_one_gpu_branches() -> None:
    plan = build_all_data_plan("configs/all_data/common.yaml")
    jobs = {job["name"]: job for job in plan["jobs"]}
    for name in ("prepare-metadata", "package"):
        assert jobs[name]["gpus"] == 0
        assert jobs[name]["partition"] == "normal"
        assert "#SBATCH --partition=normal" in jobs[name]["script"]
        assert "#SBATCH --gpus=" not in jobs[name]["script"]
    for name, job in jobs.items():
        if name in {"prepare-metadata", "package"}:
            continue
        assert job["gpus"] == 1
        assert job["partition"] == "gpu-h200"
        assert "#SBATCH --partition=gpu-h200" in job["script"]
        assert "#SBATCH --gpus=1" in job["script"]
    assert jobs["dino-domain"]["depends_on"] == ["prepare-metadata"]
    assert jobs["bioclip-assets"]["depends_on"] == ["prepare-metadata"]
    assert jobs["bioclip-domain"]["depends_on"] == ["bioclip-assets"]
    assert set(jobs["dino-seen-finetune"]["depends_on"]) == {
        "dino-domain",
        "bioclip-assets",
    }
    assert jobs["infer-seen"]["depends_on"] == ["dino-seen-finetune"]
    assert set(jobs["infer-unseen"]["depends_on"]) == {
        "dino-seen-finetune",
        "bioclip-domain",
    }
    assert set(jobs["package"]["depends_on"]) == {
        "infer-seen",
        "infer-unseen",
    }
    assert all("--missing-image-cache-only" not in job["script"] for job in jobs.values())


def test_submission_uses_independent_sibling_jobs_not_a_serial_chain(
    monkeypatch,
) -> None:
    submitted: list[tuple[str, str | None]] = []

    def fake_submit(script, path, *, dependency=None):
        del script
        name = path.stem
        submitted.append((name, dependency))
        return str(100 + len(submitted) - 1)

    monkeypatch.setattr(
        "fish_vlm.domain.workflow.submit_slurm_script", fake_submit
    )
    from fish_vlm.domain.workflow import submit_all_data_plan

    result = submit_all_data_plan("configs/all_data/common.yaml")
    dependencies = dict(submitted)
    assert dependencies["prepare-metadata"] is None
    assert dependencies["bioclip-assets"] == "100"
    assert dependencies["dino-domain"] == "100"
    assert dependencies["bioclip-domain"] == "101"
    assert set(dependencies["dino-seen-finetune"].split(":")) == {
        "101",
        "102",
    }
    assert dependencies["infer-seen"] == "104"
    assert set(dependencies["infer-unseen"].split(":")) == {"103", "104"}
    assert set(dependencies["package"].split(":")) == {"105", "106"}
    assert result["jobs"]["bioclip-assets"] == "101"
    assert result["jobs"]["dino-domain"] == "102"


def test_bioclip_domain_configuration_has_no_classifier_head() -> None:
    config = load_config("configs/all_data/bioclip_domain.yaml")
    assert not config["model"]["bioclip_classifier"]["enabled"]
    assert config["model"]["bioclip"]["freeze_text_encoder"]
    assert not config["model"]["bioclip"]["freeze_image_encoder"]
    assert config["domain"]["validation_fraction"] == 0.10
    assert config["domain"]["bioclip"]["early_stopping_patience_epochs"] == 30


def test_dino_seen_configuration_uses_class_aware_epoch_validation() -> None:
    config = load_config("configs/all_data/dino_finetune.yaml")
    assert config["model"]["supervised_head"]["enabled"]
    assert not config["model"]["bioclip_classifier"]["enabled"]
    assert config["training"]["class_balanced_sampling"]
    assert config["training"]["validation_interval_epochs"] == 1.0
    assert config["training"]["early_stopping_patience_evaluations"] == 30
