"""Single-node torchrun/DDP helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    """Process identity and device chosen from torchrun environment."""

    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialise_distributed(enabled: bool, backend: str = "nccl") -> DistributedContext:
    """Initialise a torchrun process group when requested."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if enabled and world_size > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        if backend == "nccl" and not torch.cuda.is_available():
            raise RuntimeError("NCCL distributed training requires CUDA")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        dist.init_process_group(backend=backend)
        return DistributedContext(dist.get_rank(), dist.get_world_size(), local_rank, device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DistributedContext(0, 1, 0, device)


def cleanup_distributed() -> None:
    """Destroy an active process group."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """Sum a tensor across ranks, preserving input when not distributed."""
    result = tensor.clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result


def gather_objects(value: object) -> list[object]:
    """Gather arbitrary metric payloads on every rank."""
    if not (dist.is_available() and dist.is_initialized()):
        return [value]
    gathered: list[object] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    return gathered

