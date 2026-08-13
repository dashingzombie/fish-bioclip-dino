"""DINO-style all-image self-distillation without a species classifier."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from fish_vlm.config import data_path
from fish_vlm.data.catalog import load_labels
from fish_vlm.domain.datasets import DomainMultiViewDataset, collate_domain_views
from fish_vlm.domain.transforms import build_dino_domain_transforms
from fish_vlm.models.dino import load_dino, pooled_features
from fish_vlm.training.early_stopping import EarlyStopping
from fish_vlm.utils.io import read_json, torch_save_atomic, write_json
from fish_vlm.utils.seed import seed_everything


class DinoProjectionHead(nn.Module):
    """Disposable self-distillation head; never used as a classifier."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(F.normalize(value.float(), dim=-1))


def dino_cross_view_loss(
    student_outputs: list[torch.Tensor],
    teacher_outputs: list[torch.Tensor],
    *,
    center: torch.Tensor,
    student_temperature: float,
    teacher_temperature: float,
) -> torch.Tensor:
    """Cross entropy between global teacher crops and non-matching student crops."""
    teacher_probabilities = [
        F.softmax((value.float() - center) / teacher_temperature, dim=-1).detach()
        for value in teacher_outputs
    ]
    student_log_probabilities = [
        F.log_softmax(value.float() / student_temperature, dim=-1)
        for value in student_outputs
    ]
    terms: list[torch.Tensor] = []
    for teacher_index, teacher in enumerate(teacher_probabilities):
        for student_index, student in enumerate(student_log_probabilities):
            if student_index == teacher_index:
                continue
            terms.append(torch.sum(-teacher * student, dim=-1).mean())
    if not terms:
        raise ValueError("DINO loss requires at least two views")
    return torch.stack(terms).mean()


@torch.no_grad()
def _ema_update(teacher: nn.Module, student: nn.Module, momentum: float) -> None:
    for teacher_parameter, student_parameter in zip(
        teacher.parameters(), student.parameters(), strict=True
    ):
        teacher_parameter.mul_(momentum).add_(
            student_parameter.detach(), alpha=1.0 - momentum
        )


def _forward_views(
    backbone: nn.Module,
    head: nn.Module,
    views: list[torch.Tensor],
    device: torch.device,
) -> list[torch.Tensor]:
    batch_size = len(views[0])
    combined = torch.cat([view.to(device, non_blocking=True) for view in views])
    outputs = head(pooled_features(backbone, combined))
    return list(outputs.split(batch_size))


@torch.no_grad()
def _validation_loss(
    student: nn.Module,
    student_head: nn.Module,
    teacher: nn.Module,
    teacher_head: nn.Module,
    loader: DataLoader,
    center: torch.Tensor,
    device: torch.device,
    config: dict[str, Any],
) -> float:
    student.eval()
    student_head.eval()
    teacher.eval()
    teacher_head.eval()
    values: list[float] = []
    for batch in loader:
        student_outputs = _forward_views(
            student, student_head, batch["views"], device
        )
        teacher_outputs = _forward_views(
            teacher, teacher_head, batch["views"][:2], device
        )
        loss = dino_cross_view_loss(
            student_outputs,
            teacher_outputs,
            center=center,
            student_temperature=float(config["student_temperature"]),
            teacher_temperature=float(config["teacher_temperature"]),
        )
        values.append(float(loss))
    if not values:
        raise ValueError("DINO adaptation validation split is empty")
    return sum(values) / len(values)


def train_dino_domain(config: dict[str, Any], *, resume: bool = False) -> dict[str, Any]:
    """Adapt DINO on labeled and unlabeled images; save only the backbone."""
    if config["model"].get("supervised_head", {}).get("enabled", False):
        raise ValueError("DINO domain adaptation forbids a species classifier head")
    if config["model"].get("bioclip_classifier", {}).get("enabled", False):
        raise ValueError("DINO domain adaptation forbids a BioCLIP classifier head")
    seed_everything(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain = config["domain"]
    training = domain["dino"]
    manifest = read_json(domain["manifest_path"])
    student, feature_dim, source = load_dino(config["model"]["dino"])
    student = student.to(device)
    head = DinoProjectionHead(
        feature_dim,
        int(training["head_hidden_dim"]),
        int(training["head_output_dim"]),
    ).to(device)
    teacher = copy.deepcopy(student).to(device).eval().requires_grad_(False)
    teacher_head = copy.deepcopy(head).to(device).eval().requires_grad_(False)
    train_transform, validation_transform = build_dino_domain_transforms(
        student,
        global_crops=int(training["global_crops"]),
        local_crops=int(training["local_crops"]),
    )
    images_dir = data_path(config, "images_dir")
    train_dataset = DomainMultiViewDataset(
        list(manifest["adaptation_train"]), images_dir, train_transform
    )
    validation_dataset = DomainMultiViewDataset(
        list(manifest["adaptation_validation"]), images_dir, validation_transform
    )
    workers = int(training.get("num_workers", 8))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=device.type == "cuda",
        drop_last=len(train_dataset) >= int(training["batch_size"]),
        collate_fn=collate_domain_views,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(training.get("eval_batch_size", training["batch_size"])),
        shuffle=False,
        num_workers=max(0, workers // 2),
        pin_memory=device.type == "cuda",
        collate_fn=collate_domain_views,
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": student.parameters(), "lr": float(training["backbone_lr"])},
            {"params": head.parameters(), "lr": float(training["head_lr"])},
        ],
        weight_decay=float(training["weight_decay"]),
    )
    max_epochs = int(training["max_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs
    )
    center = torch.zeros(
        1, int(training["head_output_dim"]), device=device
    )
    early = EarlyStopping(int(training["early_stopping_patience_epochs"]))
    output_dir = Path(domain["dino_output_dir"])
    best_path = output_dir / "checkpoints" / "best.pt"
    last_path = output_dir / "checkpoints" / "last.pt"
    start_epoch = 0
    if resume and last_path.is_file():
        state = torch.load(last_path, map_location=device, weights_only=False)
        student.load_state_dict(state["model_state"])
        head.load_state_dict(state["student_head_state"])
        teacher.load_state_dict(state["teacher_state"])
        teacher_head.load_state_dict(state["teacher_head_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        center.copy_(state["center"])
        early = EarlyStopping(**state["early_stopping"])
        start_epoch = int(state["epoch"])
    use_amp = bool(training.get("use_amp", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16
    metrics: dict[str, Any] = {}
    center_momentum = float(training["center_momentum"])
    base_momentum = float(training["teacher_momentum"])
    total_steps = max(1, max_epochs * len(train_loader))
    step = start_epoch * len(train_loader)
    for epoch in range(start_epoch, max_epochs):
        student.train()
        head.train()
        running = 0.0
        samples = 0
        for batch in train_loader:
            progress = step / total_steps
            momentum = 1.0 - (1.0 - base_momentum) * (
                math.cos(math.pi * progress) + 1.0
            ) / 2.0
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                student_outputs = _forward_views(
                    student, head, batch["views"], device
                )
                with torch.no_grad():
                    teacher_outputs = _forward_views(
                        teacher, teacher_head, batch["views"][:2], device
                    )
                loss = dino_cross_view_loss(
                    student_outputs,
                    teacher_outputs,
                    center=center,
                    student_temperature=float(training["student_temperature"]),
                    teacher_temperature=float(training["teacher_temperature"]),
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [*student.parameters(), *head.parameters()],
                float(training.get("gradient_clip_norm", 3.0)),
            )
            optimizer.step()
            _ema_update(teacher, student, momentum)
            _ema_update(teacher_head, head, momentum)
            with torch.no_grad():
                batch_center = torch.cat(teacher_outputs).mean(dim=0, keepdim=True)
                center.mul_(center_momentum).add_(
                    batch_center, alpha=1.0 - center_momentum
                )
            batch_size = len(batch["filename"])
            running += float(loss) * batch_size
            samples += batch_size
            step += 1
        scheduler.step()
        validation_loss = _validation_loss(
            student,
            head,
            teacher,
            teacher_head,
            validation_loader,
            center,
            device,
            training,
        )
        improved, should_stop = early.update(-validation_loss)
        metrics = {
            "epoch": epoch + 1,
            "train_self_distillation_loss": running / max(1, samples),
            "validation_self_distillation_loss": validation_loss,
            "selection_value": -validation_loss,
            "manifest_hash": manifest["manifest_hash"],
            "classifier_head_used": False,
            "projection_head_discarded": True,
        }
        common = {
            "model_state": student.state_dict(),
            "epoch": epoch + 1,
            "metrics": metrics,
            "dino_model_name": config["model"]["dino"]["name"],
            "dino_checkpoint_source": source,
            "domain_manifest_hash": manifest["manifest_hash"],
            "official_test_labels_loaded": False,
            "official_unseen_labels_loaded": False,
        }
        torch_save_atomic(
            {
                **common,
                "student_head_state": head.state_dict(),
                "teacher_state": teacher.state_dict(),
                "teacher_head_state": teacher_head.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "center": center.detach().cpu(),
                "early_stopping": {
                    "patience": early.patience,
                    "min_delta": early.min_delta,
                    "best": early.best,
                    "bad_evaluations": early.bad_evaluations,
                },
            },
            last_path,
        )
        if improved:
            torch_save_atomic(common, best_path)
            write_json(output_dir / "metrics" / "best.json", metrics)
        if should_stop:
            break
    return metrics
