from __future__ import annotations

from typing import Any

from fish_vlm.config import load_config
from fish_vlm.training.wandb_logging import (
    ScientificWandbLogger,
    compact_epoch_payload,
    interpretable_metric_name,
)


class FakeRun:
    def __init__(self) -> None:
        self.summary: dict[str, Any] = {}
        self.logged: list[dict[str, Any]] = []
        self.defined: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.finished = False

    def define_metric(self, *args: Any, **kwargs: Any) -> None:
        self.defined.append((args, kwargs))

    def log(self, value: dict[str, Any]) -> None:
        self.logged.append(value)

    def finish(self) -> None:
        self.finished = True


class FakeWandb:
    def __init__(self) -> None:
        self.run = FakeRun()
        self.init_kwargs: dict[str, Any] = {}

    def init(self, **kwargs: Any) -> FakeRun:
        self.init_kwargs = kwargs
        return self.run


def test_metric_names_are_direct_and_interpretable() -> None:
    assert (
        interpretable_metric_name("pseudo_unseen_fused_text_balanced_accuracy")
        == "validation/pseudo_unseen/fused_text/balanced_accuracy"
    )
    compact = compact_epoch_payload(
        epoch=2,
        training_losses={"loss": 1.2, "dino_text_classification": 0.8},
        metrics={
            "estimated_overall_accuracy": 0.7,
            "dino_text_accuracy": 0.6,
        },
        learning_rates={"projector": 1e-4},
        throughput=50.0,
        gpu_peak_memory_bytes=None,
        detailed=False,
    )
    assert compact["loss/train/total"] == 1.2
    assert compact["score/estimated_overall_accuracy"] == 0.7
    assert "validation/seen/dino_text/accuracy" not in compact


def test_wandb_logger_uses_small_periodic_payload_and_best_summary() -> None:
    config = load_config("configs/train/projection_only.yaml")
    fake = FakeWandb()
    logger = ScientificWandbLogger(
        config, trainable_parameters=123, wandb_module=fake
    )
    metrics = {
        "estimated_overall_accuracy": 0.7,
        "seen_accuracy": 0.8,
        "pseudo_unseen_accuracy": 0.6,
        "dino_text_accuracy": 0.75,
    }
    logger.log_epoch(
        epoch=1,
        training_losses={"loss": 1.0},
        metrics=metrics,
        learning_rates={"projector": 1e-4},
        throughput=12.0,
        gpu_peak_memory_bytes=None,
        improved=False,
    )
    assert "score/estimated_overall_accuracy" in fake.run.logged[-1]
    assert "validation/seen/dino_text/accuracy" not in fake.run.logged[-1]
    logger.log_epoch(
        epoch=2,
        training_losses={"loss": 0.9},
        metrics=metrics,
        learning_rates={"projector": 1e-4},
        throughput=13.0,
        gpu_peak_memory_bytes=1024**3,
        improved=True,
    )
    assert fake.run.logged[-1]["validation/seen/dino_text/accuracy"] == 0.75
    logger.record_best(epoch=2, metrics=metrics)
    assert fake.run.summary["best/epoch"] == 2
    assert fake.run.summary["best/score/estimated_overall_accuracy"] == 0.7
    assert "resolved_configuration" not in fake.init_kwargs["config"]
    logger.finish()
    assert fake.run.finished

