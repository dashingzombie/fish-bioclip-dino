"""GenomeDK DAG and configuration contracts for the all-data workflow."""

from __future__ import annotations

from fish_vlm.config import load_config
from fish_vlm.domain.workflow import build_all_data_plan


def test_all_data_plan_parallelises_independent_one_gpu_branches() -> None:
    plan = build_all_data_plan("configs/all_data/common.yaml")
    jobs = {job["name"]: job for job in plan["jobs"]}
    assert all(job["gpus"] == 1 for job in jobs.values())
    assert all("#SBATCH --gpus=1" in job["script"] for job in jobs.values())
    assert jobs["dino-domain"]["depends_on"] == ["prepare-metadata"]
    assert jobs["bioclip-assets"]["depends_on"] == ["prepare-metadata"]
    assert jobs["bioclip-domain"]["depends_on"] == ["bioclip-assets"]
    assert set(jobs["dino-seen-finetune"]["depends_on"]) == {
        "dino-domain",
        "bioclip-assets",
    }
    assert set(jobs["finalise"]["depends_on"]) == {
        "dino-seen-finetune",
        "bioclip-domain",
    }
    assert all("--missing-image-cache-only" not in job["script"] for job in jobs.values())


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
