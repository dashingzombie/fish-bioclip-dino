"""SLURM batch-script rendering."""

from __future__ import annotations

import shlex
from typing import Any


def _append_optional_directives(lines: list[str], slurm: dict[str, Any]) -> None:
    for key in ("account", "partition"):
        if slurm.get(key):
            lines.append(f"#SBATCH --{key}={slurm[key]}")


def _cache_setup_lines(
    config: dict[str, Any],
    *,
    node_local: bool,
) -> list[str]:
    """Point all artifact/model caches at shared or node-local storage."""
    cache_dir = shlex.quote(str(config.get("cache_dir", "cache")))
    #transfer local images to shared cache dir if node-local caching is enabled
    
    lines = [
        "",
        f"SHARED_CACHE_DIR={cache_dir}",
        'if [[ "${SHARED_CACHE_DIR}" != /* ]]; then',
        '    SHARED_CACHE_DIR="${PWD}/${SHARED_CACHE_DIR}"',
        "fi",
    ]
    if node_local:
        tmpdir = ''
        node_cache_name = shlex.quote(
            str(config["slurm"].get("node_cache_dir", "fish-vlm-cache"))
        )
        lines.extend(
            [
                ': "${TMPDIR:?TMPDIR must be set for node-local cache staging}"',
                'if [[ ! -d "${SHARED_CACHE_DIR}" ]]; then',
                '    echo "Shared cache directory does not exist: ${SHARED_CACHE_DIR}" >&2',
                "    exit 1",
                "fi",
                f"FISH_VLM_CACHE_DIR=\"${{TMPDIR}}\"/{node_cache_name}",
                'mkdir -p "${FISH_VLM_CACHE_DIR}"',
                'cp -a "${SHARED_CACHE_DIR}/." "${FISH_VLM_CACHE_DIR}/"',
            ]
        )
    else:
        lines.extend(
            [
                'mkdir -p "${SHARED_CACHE_DIR}"',
                'FISH_VLM_CACHE_DIR="${SHARED_CACHE_DIR}"',
            ]
        )
    lines.extend(
        [
            'export FISH_VLM_CACHE_DIR',
            'export HF_HOME="${FISH_VLM_CACHE_DIR}/huggingface"',
            'export TORCH_HOME="${FISH_VLM_CACHE_DIR}/torch"',
            'mkdir -p "${HF_HOME}" "${TORCH_HOME}"',
        ]
    )
    return lines


def _append_runtime_setup(
    lines: list[str],
    config: dict[str, Any],
    *,
    node_local_cache: bool,
) -> None:
    slurm = config["slurm"]
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
            f"module load {shlex.quote(str(module))}" for module in modules
        )
    if slurm.get("environment_activate"):
        lines.append(str(slurm["environment_activate"]))
    lines.extend(_cache_setup_lines(config, node_local=node_local_cache))


def render_batch_script(config: dict[str, Any], training_config: str) -> str:
    """Render a one-node torchrun job, optionally as a SLURM array."""
    slurm = config["slurm"]
    gpus = int(slurm.get("gpus", 1))
    array_configs = [str(path) for path in slurm.get("array_configs", [])]
    is_array = bool(array_configs)
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={slurm.get('job_name', 'fish-vlm')}",
        "#SBATCH --nodes=1",
        f"#SBATCH --gres=gpu:{gpus}",
        f"#SBATCH --cpus-per-task={int(slurm.get('cpus', 16))}",
        f"#SBATCH --mem={slurm.get('memory', '64G')}",
        f"#SBATCH --time={slurm.get('time_limit', '24:00:00')}",
    ]
    _append_optional_directives(lines, slurm)
    log_dir = slurm.get("log_dir", "outputs/slurm")
    if is_array:
        array_spec = f"0-{len(array_configs) - 1}"
        max_parallel = slurm.get("array_max_parallel")
        if max_parallel is not None:
            max_parallel = int(max_parallel)
            if max_parallel < 1:
                raise ValueError("slurm.array_max_parallel must be at least 1")
            array_spec += f"%{max_parallel}"
        lines.extend(
            [
                f"#SBATCH --array={array_spec}",
                f"#SBATCH --output={log_dir}/%x-%A_%a.out",
                f"#SBATCH --error={log_dir}/%x-%A_%a.err",
            ]
        )
    else:
        lines.extend(
            [
                f"#SBATCH --output={log_dir}/%x-%j.out",
                f"#SBATCH --error={log_dir}/%x-%j.err",
            ]
        )

    _append_runtime_setup(
        lines,
        config,
        node_local_cache=bool(slurm.get("stage_cache_on_node", True)),
    )
    if is_array:
        lines.extend(["", "TRAINING_CONFIGS=("])
        lines.extend(f"    {shlex.quote(path)}" for path in array_configs)
        lines.extend(
            [
                ")",
                'TRAINING_CONFIG="${TRAINING_CONFIGS[$SLURM_ARRAY_TASK_ID]}"',
                'echo "Array task: ${SLURM_ARRAY_TASK_ID}"',
                'echo "Training config: ${TRAINING_CONFIG}"',
            ]
        )
    else:
        lines.append(f"TRAINING_CONFIG={shlex.quote(training_config)}")
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
    node_local_cache: bool = True,
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
        ]
    )
    _append_optional_directives(lines, slurm)
    log_dir = slurm.get("log_dir", "outputs/slurm")
    lines.extend(
        [
            f"#SBATCH --output={log_dir}/%x-%j.out",
            f"#SBATCH --error={log_dir}/%x-%j.err",
        ]
    )
    _append_runtime_setup(
        lines,
        config,
        node_local_cache=node_local_cache,
    )
    lines.extend(["", *(shlex.join(command) for command in commands)])
    return "\n".join(lines) + "\n"
