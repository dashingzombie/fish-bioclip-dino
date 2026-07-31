"""One-at-a-time hard-negative selection for text classification."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def hard_negative_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    strategy: str,
    top_k: int,
    class_groups: list[int] | torch.Tensor | None = None,
    prototype_similarity: torch.Tensor | None = None,
) -> torch.Tensor:
    """Keep the positive and controlled hard negatives for each example."""
    if top_k < 1:
        raise ValueError("hard-negative top_k must be at least one")
    batch, classes = logits.shape
    if classes < 2:
        return F.cross_entropy(logits.float(), targets.long())
    targets = targets.long()
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(1, targets[:, None], True)
    if strategy == "visually_similar" and prototype_similarity is not None:
        similarity = prototype_similarity.to(logits.device).float()
        if similarity.shape != (classes, classes):
            raise ValueError("Visual similarity matrix has the wrong shape")
        candidates = similarity.index_select(0, targets).clone()
        candidates.scatter_(1, targets[:, None], float("-inf"))
        indices = candidates.topk(min(top_k, classes - 1), dim=-1).indices
        mask.scatter_(1, indices, True)
    elif strategy in {"visually_similar", "model_score"}:
        candidates = logits.detach().float().clone()
        candidates.scatter_(1, targets[:, None], float("-inf"))
        indices = candidates.topk(min(top_k, classes - 1), dim=-1).indices
        mask.scatter_(1, indices, True)
    elif strategy == "text_similar":
        if prototype_similarity is None:
            raise ValueError("text_similar negatives require prototype similarity")
        similarity = prototype_similarity.to(logits.device).float()
        if similarity.shape != (classes, classes):
            raise ValueError("Prototype similarity matrix has the wrong shape")
        candidates = similarity.index_select(0, targets).clone()
        candidates.scatter_(1, targets[:, None], float("-inf"))
        indices = candidates.topk(min(top_k, classes - 1), dim=-1).indices
        mask.scatter_(1, indices, True)
    elif strategy in {"same_genus", "same_family"}:
        if class_groups is None:
            raise ValueError(f"{strategy} negatives require taxonomy groups")
        groups = torch.as_tensor(
            class_groups, device=logits.device, dtype=torch.long
        )
        if groups.ndim != 1 or len(groups) != classes:
            raise ValueError("Taxonomy groups must match logit columns")
        if groups.min().item() < -1:
            raise ValueError(
                "Taxonomy group indices must be -1 or non-negative"
            )
        target_groups = groups[targets]
        group_mask = (
            (groups.unsqueeze(0) == target_groups.unsqueeze(1))
            & (groups.unsqueeze(0) >= 0)
            & (target_groups.unsqueeze(1) >= 0)
        )
        group_mask.scatter_(1, targets[:, None], False)
        ranked = logits.detach().float().masked_fill(~group_mask, float("-inf"))
        available = group_mask.sum(dim=-1)
        for row in range(batch):
            count = min(top_k, int(available[row].item()))
            if count:
                mask[row, ranked[row].topk(count).indices] = True
    else:
        raise ValueError(f"Unknown hard-negative strategy: {strategy}")
    no_negative = mask.sum(dim=-1) == 1
    if bool(no_negative.any()):
        fallback = logits.detach().float().clone()
        fallback.scatter_(1, targets[:, None], float("-inf"))
        indices = fallback.topk(min(top_k, classes - 1), dim=-1).indices
        for row in no_negative.nonzero(as_tuple=False).flatten().tolist():
            mask[row] = False
            mask[row, targets[row]] = True
            mask[row, indices[row]] = True
    selected = logits.float().masked_fill(~mask, float("-inf"))
    return F.cross_entropy(selected, targets)
