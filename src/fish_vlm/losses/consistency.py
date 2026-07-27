"""Optional probability consistency across text branches."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def branch_consistency_loss(
    first_logits: torch.Tensor,
    second_logits: torch.Tensor,
    method: str = "js",
) -> torch.Tensor:
    """Compute symmetric KL or Jensen-Shannon divergence in float32."""
    first_log = F.log_softmax(first_logits.float(), dim=-1)
    second_log = F.log_softmax(second_logits.float(), dim=-1)
    first = first_log.exp()
    second = second_log.exp()
    if method == "symmetric_kl":
        return 0.5 * (
            F.kl_div(first_log, second, reduction="batchmean")
            + F.kl_div(second_log, first, reduction="batchmean")
        )
    if method == "js":
        mean = 0.5 * (first + second)
        log_mean = mean.clamp_min(1e-12).log()
        return 0.5 * (
            F.kl_div(log_mean, first, reduction="batchmean")
            + F.kl_div(log_mean, second, reduction="batchmean")
        )
    raise ValueError("Consistency method must be js or symmetric_kl")

