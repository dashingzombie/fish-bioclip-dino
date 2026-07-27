"""SLURM batch-script rendering."""

from __future__ import annotations

import shlex
from typing import Any


def render_batch_script(config: dict[str, Any], training_config: str) -> str:
    """Render a one-node torchrun job with explicit resources."""
    slurm = config["slurm"]
    gpus = int(slurm.get("gpus", 2))
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={slurm.get('job_name', 'fish-vlm')}",
        "#SBATCH --nodes=1",
        f"#SBATCH --gres=gpu:{gpus}",
        f"#SBATCH --cpus-per-task={int(slurm.get('cpus', 16))}",
        f"#SBATCH --mem={slurm.get('memory', '64G')}",
        f"#SBATCH --time={slurm.get('time_limit', '24:00:00')}",
        f"#SBATCH --output={slurm.get('log_dir', 'outputs/slurm')}/%x-%j.out",
    ]
    for optional, directive in (("partition", "partition"), ("account", "account")):
        if slurm.get(optional):
            lines.append(f"#SBATCH --{directive}={slurm[optional]}")
    lines.extend(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(str(slurm.get('work_dir', '.')))}",
        ]
    )
    modules = slurm.get("modules", [])
    if modules:
        lines.append("module purge")
        lines.extend(f"module load {shlex.quote(str(module))}" for module in modules)
    if slurm.get("environment_activate"):
        lines.append(str(slurm["environment_activate"]))
    lines.append(
        f"torchrun --standalone --nproc_per_node={gpus} -m fish_vlm.cli train "
        f"--config {shlex.quote(training_config)}"
    )
    return "\n".join(lines) + "\n"


def render_workflow_batch_script(
    config: dict[str, Any],
    *,
    job_name: str,
    commands: list[list[str]],
    gpus: int,
) -> str:
    """Render one dependency-chain workflow job containing explicit commands."""
    slurm = config["slurm"]
    workflow = config["workflow"]
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={job_name}",
        "#SBATCH --nodes=1",
    ]
    if gpus > 0:
        lines.append(f"#SBATCH --gres=gpu:{gpus}")
    lines.extend(
        [
            f"#SBATCH --cpus-per-task={int(workflow.get('cpus', slurm.get('cpus', 16)))}",
            f"#SBATCH --mem={workflow.get('memory', slurm.get('memory', '64G'))}",
            f"#SBATCH --time={workflow.get('time_limit', slurm.get('time_limit', '24:00:00'))}",
            f"#SBATCH --output={slurm.get('log_dir', 'outputs/slurm')}/%x-%j.out",
        ]
    )
    for optional in ("partition", "account"):
        if slurm.get(optional):
            lines.append(f"#SBATCH --{optional}={slurm[optional]}")
    lines.extend(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(str(slurm.get('work_dir', '.')))}",
        ]
    )
    modules = slurm.get("modules", [])
    if modules:
        lines.append("module purge")
        lines.extend(f"module load {shlex.quote(str(module))}" for module in modules)
    if slurm.get("environment_activate"):
        lines.append(str(slurm["environment_activate"]))
    lines.extend(shlex.join(command) for command in commands)
    return "\n".join(lines) + "\n"
