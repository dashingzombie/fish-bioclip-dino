"""AdamW parameter groups for staged optimisation."""

from __future__ import annotations

import torch
from torch import nn

from fish_vlm.models.dino import _last_block
from fish_vlm.models.multimodal import FishMultimodalModel


def _parameters(module: nn.Module | None) -> list[nn.Parameter]:
    return [] if module is None else [p for p in module.parameters() if p.requires_grad]


def build_optimizer(model: FishMultimodalModel, config: dict) -> torch.optim.AdamW:
    """Build non-overlapping stage-aware AdamW groups."""
    stage = config["stage"]
    base_lr = float(config["lr"])
    base_wd = float(config["weight_decay"])
    groups: list[dict] = []

    def add(name: str, params: list[nn.Parameter], lr: float, weight_decay: float) -> None:
        if params:
            groups.append({"name": name, "params": params, "lr": lr, "weight_decay": weight_decay})

    parameter_groups = config.get("parameter_groups", {})
    projector_cfg = parameter_groups.get("projector", {})
    add("projector", _parameters(model.projector), float(projector_cfg.get("lr", base_lr)), float(projector_cfg.get("weight_decay", base_wd)))
    scale_cfg = parameter_groups.get("logit_scale", {})
    add("logit_scale", _parameters(model.logit_scale), float(scale_cfg.get("lr", base_lr)), float(scale_cfg.get("weight_decay", 0.0)))
    if stage in {"final_block", "joint_supervised_text"}:
        dino_cfg = parameter_groups.get("dino_last_block", {})
        dino_params = _parameters(_last_block(model.dino))
        for name in ("norm", "fc_norm"):
            normalisation = getattr(model.dino, name, None)
            dino_params.extend(_parameters(normalisation))
        seen: set[int] = set()
        dino_params = [p for p in dino_params if not (id(p) in seen or seen.add(id(p)))]
        add("dino_last_block", dino_params, float(dino_cfg.get("lr", 1e-5)), float(dino_cfg.get("weight_decay", 1e-4)))
    add("supervised_head", _parameters(model.supervised_head), base_lr, base_wd)
    add("bioclip_adapter", _parameters(model.bioclip_adapter), base_lr, base_wd)
    if not groups:
        raise ValueError("No trainable parameters for optimizer")
    all_ids = [id(parameter) for group in groups for parameter in group["params"]]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("A parameter appears in multiple optimiser groups")
    return torch.optim.AdamW(groups)

