"""Compact, interpretable W&B logging without model artifacts."""

from __future__ import annotations

from typing import Any


_METRIC_SUFFIXES = ("balanced_accuracy", "top5_accuracy", "macro_f1", "accuracy")
_SCORE_KEYS = {
    "estimated_overall_accuracy": "score/estimated_overall_accuracy",
    "seen_unseen_harmonic_mean": "score/seen_unseen_harmonic_mean",
    "seen_accuracy": "accuracy/seen/selected",
    "pseudo_unseen_accuracy": "accuracy/pseudo_unseen/selected",
}


def interpretable_metric_name(key: str) -> str:
    """Map internal metric keys to stable W&B panel paths."""
    if key in _SCORE_KEYS:
        return _SCORE_KEYS[key]
    split = "seen"
    remainder = key
    if key.startswith("pseudo_unseen_"):
        split = "pseudo_unseen"
        remainder = key.removeprefix("pseudo_unseen_")
    for suffix in _METRIC_SUFFIXES:
        marker = f"_{suffix}"
        if remainder.endswith(marker):
            branch = remainder[: -len(marker)]
            return f"validation/{split}/{branch}/{suffix}"
    return f"validation/other/{key}"


def compact_step_payload(
    *,
    step: int,
    training_losses: dict[str, float],
    metrics: dict[str, float],
    learning_rates: dict[str, float],
    throughput: float,
    gpu_peak_memory_bytes: int | None,
    detailed: bool,
) -> dict[str, float | int]:
    """Create one concise scalar payload for a validation step."""
    payload: dict[str, float | int] = {
        "step": step,
        "system/throughput_images_per_second": throughput,
    }
    for name, value in training_losses.items():
        readable = "total" if name == "loss" else name
        payload[f"loss/train/{readable}"] = value
    for name, value in learning_rates.items():
        payload[f"optimization/learning_rate/{name}"] = value
    if gpu_peak_memory_bytes is not None:
        payload["system/gpu_peak_memory_gib"] = gpu_peak_memory_bytes / (1024**3)
    for key, output_name in _SCORE_KEYS.items():
        if key in metrics:
            payload[output_name] = metrics[key]
    if detailed:
        for key, value in metrics.items():
            if key not in _SCORE_KEYS:
                payload[interpretable_metric_name(key)] = value
    return payload


def scientific_run_config(config: dict[str, Any]) -> dict[str, Any]:
    """Keep only decision-relevant configuration in the W&B run config."""
    training = config["training"]
    return {
        "seed": config["seed"],
        "stage": training["stage"],
        "max_steps": training["max_steps"],
        "validation_interval_steps": training["validation_interval_steps"],
        "batch_size": training["batch_size"],
        "optimizer": "AdamW",
        "learning_rate": training["lr"],
        "weight_decay": training["weight_decay"],
        "amp_dtype": training.get("amp_dtype"),
        "dino": config["model"]["dino"],
        "projector": config["model"]["projector"],
        "bioclip_image_path": config["model"]["bioclip_image_path"],
        "supervised_head": config["model"]["supervised_head"],
        "losses": config["loss"],
        "pseudo_unseen": config["validation"]["pseudo_unseen"],
        "selection_metric": config["validation"]["selection_metric"],
        "selection_branch": config["validation"].get("selection_branch"),
        "fusion": config["fusion"],
        "sweep": config.get("sweep_metadata"),
        "runtime_fallbacks": config.get("_runtime_fallbacks", []),
    }


class ScientificWandbLogger:
    """Log compact step scalars and a single best-result summary."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        trainable_parameters: int,
        wandb_module: Any | None = None,
    ) -> None:
        if wandb_module is None:
            import wandb as wandb_module

        wandb_config = config["wandb"]
        stage = str(config["training"]["stage"])
        default_name = f"{stage}-seed-{int(config['seed'])}"
        self._detailed_every_steps = max(
            1, int(wandb_config.get("log_detailed_every_steps", 500))
        )
        self.run = wandb_module.init(
            project=wandb_config["project"],
            name=wandb_config.get("name") or default_name,
            group=wandb_config.get("group") or "multimodal-pipeline",
            job_type=stage,
            tags=list(wandb_config.get("tags", ["fish", "dino", "bioclip"])),
            mode=wandb_config.get("mode", "online"),
            config=scientific_run_config(config),
        )
        self.run.define_metric("step")
        self.run.define_metric("*", step_metric="step")
        self.run.summary["model/trainable_parameters"] = int(trainable_parameters)
        self.run.summary["model/dino_name"] = config["model"]["dino"]["name"]
        self.run.summary["model/bioclip_checkpoint"] = config["model"]["bioclip"]["checkpoint"]
        self.run.summary["selection/metric"] = config["validation"]["selection_metric"]
        self.run.summary["output/checkpoint"] = (
            str(config.get("output_dir", "outputs"))
            + "/checkpoints/"
            + config["training"].get("checkpoint_name", "best.pt")
        )

    def log_step(
        self,
        *,
        step: int,
        training_losses: dict[str, float],
        metrics: dict[str, float],
        learning_rates: dict[str, float],
        throughput: float,
        gpu_peak_memory_bytes: int | None,
        improved: bool,
    ) -> None:
        """Log always-on decisions and periodic/new-best branch details."""
        detailed = (
            improved
            or step == 0
            or step % self._detailed_every_steps == 0
        )
        self.run.log(
            compact_step_payload(
                step=step,
                training_losses=training_losses,
                metrics=metrics,
                learning_rates=learning_rates,
                throughput=throughput,
                gpu_peak_memory_bytes=gpu_peak_memory_bytes,
                detailed=detailed,
            )
        )

    def record_best(self, *, step: int, metrics: dict[str, float]) -> None:
        """Replace summary values with the latest best validation result."""
        self.run.summary["best/step"] = int(step)
        for key, value in metrics.items():
            self.run.summary[f"best/{interpretable_metric_name(key)}"] = float(value)

    def finish(self) -> None:
        """Finish without uploading model or checkpoint artifacts."""
        self.run.finish()
