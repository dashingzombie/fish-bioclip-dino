"""SLURM batch-script rendering."""

from __future__ import annotations

import shlex
from typing import Any


def render_batch_script(config: dict[str, Any], training_config: str) -> str:
    """Render a one-node torchrun job, optionally as a SLURM array."""
    slurm = config["workflow"]
    gpus = int(slurm.get("gpus", 2))

    array_configs = [
        str(path) for path in slurm.training_stages.get("config", [])
    ]
    is_array = bool(array_configs)

    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={slurm.get('job_name', 'fish-vlm')}",
        f"#SBATCH --account={slurm.get('account', 'worm-species')}",
        f"#SBATCH --partition={slurm.get('partition', 'gpu-l40s,gpu-h200')}",
        "#SBATCH --nodes=1",
        f"#SBATCH --gres=gpu:{gpus}",
        f"#SBATCH --cpus-per-task={int(slurm.get('cpus', 16))}",
        f"#SBATCH --mem={slurm.get('memory', '64G')}",
        f"#SBATCH --time={slurm.get('time_limit', '24:00:00')}",
    ]


    if is_array:
        array_spec = f"0-{len(array_configs) - 1}"

        max_parallel = slurm.get("array_max_parallel")
        if max_parallel is not None:
            max_parallel = int(max_parallel)
            if max_parallel < 1:
                raise ValueError("slurm.array_max_parallel must be at least 1")
            array_spec += f"%{max_parallel}"

        lines.append(f"#SBATCH --array={array_spec}")
        lines.append(
            f"#SBATCH --output="
            f"{slurm.get('log_dir', 'outputs/slurm')}/%x-%A_%a.out"
            f"#SBATCH --error="
            f"{slurm.get('log_dir', 'outputs/slurm')}/%x-%A_%a.err"
        )
    else:
        lines.append(
            f"#SBATCH --output="
            f"{slurm.get('log_dir', 'outputs/slurm')}/%x-%j.out"
            f"#SBATCH --error="
            f"{slurm.get('log_dir', 'outputs/slurm')}/%x-%j.err"
        )

    lines.extend(
        [
            "",
            "set -euo pipefail",
            f"cd {shlex.quote(str(slurm.get('work_dir', '.')))}",
        ]
    )

    modules = slurm.get("modules", [])
    if modules:
        lines.append("module purge")
        lines.extend(
            f"module load {shlex.quote(str(module))}"
            for module in modules
        )

    if environment_activate := slurm.get("environment_activate"):
        lines.append(str(environment_activate))

    if is_array:
        lines.append("")
        lines.append("TRAINING_CONFIGS=(")
        lines.extend(
            f"    {shlex.quote(path)}"
            for path in array_configs
        )
        lines.append(")")
        lines.append(
            'TRAINING_CONFIG="${TRAINING_CONFIGS[$SLURM_ARRAY_TASK_ID]}"'
        )
        lines.append(
            'echo "Array task: ${SLURM_ARRAY_TASK_ID}"'
        )
        lines.append(
            'echo "Training config: ${TRAINING_CONFIG}"'
        )
    else:
        lines.append(
            f"TRAINING_CONFIG={shlex.quote(training_config)}"
        )

    lines.append(
        f'torchrun --standalone --nproc_per_node={gpus} '
        f'-m fish_vlm.cli train --config "$TRAINING_CONFIG"'
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
        f"#SBATCH --account={slurm.get('account', 'worm-species')}",
        f"#SBATCH --job-name={job_name}",
        "#SBATCH --nodes=1",
        f"#SBATCH --partition={slurm.get('partition', 'gpu-l40s,gpu-h200')}"
    ]
    if gpus > 0:
        lines.append(f"#SBATCH --gres=gpu:{gpus}")
    lines.extend(
        [
            f"#SBATCH --cpus-per-task={int(workflow.get('cpus', slurm.get('cpus', 16)))}",
            f"#SBATCH --mem={workflow.get('memory', slurm.get('memory', '64G'))}",
            f"#SBATCH --time={workflow.get('time_limit', slurm.get('time_limit', '24:00:00'))}",
            f"#SBATCH --output={slurm.get('log_dir', 'outputs/slurm')}/%x-%j.out",
            f"#SBATCH --error={slurm.get('log_dir', 'outputs/slurm')}/%x-%j.err",
        ]
    )
    
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
