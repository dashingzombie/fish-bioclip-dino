"""SLURM batch-script rendering."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any


def _append_optional_directives(lines: list[str], slurm: dict[str, Any]) -> None:
    for key in ("account", "partition"):
        if slurm.get(key):
            lines.append(f"#SBATCH --{key}={slurm[key]}")


def _node_tmpdir_lines() -> list[str]:
    return [
        'NODE_TMPDIR="${TMPDIR:-${SLURM_TMPDIR:-}}"',
        'if [[ -z "${NODE_TMPDIR}" ]]; then',
        '    echo "TMPDIR or SLURM_TMPDIR must be set for node-local staging" >&2',
        "    exit 1",
        "fi",
    ]


def _cache_items(scope: str) -> list[str]:
    if scope == "training":
        return [
            "huggingface",
            "torch",
            "text",
            "bioclip_images/train_embeddings.pt",
            "image_transforms/train/manifest.json",
            "image_transforms/train/dino.npy",
            "image_transforms/train/bioclip.npy",
            "image_transforms/test/manifest.json",
            "image_transforms/test/dino.npy",
            "image_transforms/test/bioclip.npy",
            "image_transforms/unseen/manifest.json",
            "image_transforms/unseen/dino.npy",
            "image_transforms/unseen/bioclip.npy",
        ]
    if scope == "finalisation":
        return [
            "huggingface",
            "torch",
            "text",
            "image_transforms/train/manifest.json",
            "image_transforms/train/dino.npy",
            "image_transforms/train/bioclip.npy",
            "image_transforms/test/manifest.json",
            "image_transforms/test/dino.npy",
            "image_transforms/test/bioclip.npy",
            "image_transforms/unseen/manifest.json",
            "image_transforms/unseen/dino.npy",
            "image_transforms/unseen/bioclip.npy",
        ]
    if scope != "shared":
        raise ValueError(f"Unknown cache staging scope: {scope}")
    return []


def _cache_setup_lines(config: dict[str, Any], *, scope: str) -> list[str]:
    """Stage only cache entries consumed by this job, in parallel."""
    cache_dir = shlex.quote(str(config.get("cache_dir", "cache")))
    lines = [
        "",
        f"SHARED_CACHE_DIR={cache_dir}",
        'if [[ "${SHARED_CACHE_DIR}" != /* ]]; then',
        '    SHARED_CACHE_DIR="${PWD}/${SHARED_CACHE_DIR}"',
        "fi",
    ]
    if scope == "shared":
        lines.extend(
            [
                'mkdir -p "${SHARED_CACHE_DIR}"',
                'FISH_VLM_CACHE_DIR="${SHARED_CACHE_DIR}"',
            ]
        )
    else:
        lines.extend(_node_tmpdir_lines())
        node_cache_name = shlex.quote(
            str(config["slurm"].get("node_cache_dir", "fish-vlm-cache"))
        )
        lines.extend(
            [
                'if [[ ! -d "${SHARED_CACHE_DIR}" ]]; then',
                '    echo "Shared cache directory does not exist: ${SHARED_CACHE_DIR}" >&2',
                "    exit 1",
                "fi",
                f'FISH_VLM_CACHE_DIR="${{NODE_TMPDIR}}"/{node_cache_name}',
                'mkdir -p "${FISH_VLM_CACHE_DIR}"',
                "CACHE_ITEMS=(",
                *(
                    f"    {shlex.quote(relative)}"
                    for relative in _cache_items(scope)
                ),
                ")",
                "CACHE_COPY_PIDS=()",
                'for relative in "${CACHE_ITEMS[@]}"; do',
                '    source_path="${SHARED_CACHE_DIR}/${relative}"',
                '    if [[ ! -e "${source_path}" ]]; then',
                "        continue",
                "    fi",
                '    destination_path="${FISH_VLM_CACHE_DIR}/${relative}"',
                '    mkdir -p "$(dirname "${destination_path}")"',
                '    cp --archive --reflink=auto "${source_path}" "${destination_path}" &',
                '    CACHE_COPY_PIDS+=("$!")',
                "done",
                'for pid in "${CACHE_COPY_PIDS[@]}"; do',
                '    wait "${pid}"',
                "done",
            ]
        )
    lines.extend(
        [
            "export FISH_VLM_CACHE_DIR",
            'export HF_HOME="${FISH_VLM_CACHE_DIR}/huggingface"',
            'export TORCH_HOME="${FISH_VLM_CACHE_DIR}/torch"',
            'mkdir -p "${HF_HOME}" "${TORCH_HOME}"',
        ]
    )
    return lines


def _image_setup_lines(config: dict[str, Any]) -> list[str]:
    """Stream only split-referenced raw images to node-local NVMe."""
    data = config["data"]
    images_dir = Path(str(data["images_dir"]))
    if not images_dir.is_absolute():
        images_dir = Path(str(data["root_dir"])) / images_dir
    config_path = shlex.quote(str(config["_config_path"]))
    lines = ["", *_node_tmpdir_lines()]
    lines.extend(
        [
            f"SHARED_IMAGES_DIR={shlex.quote(str(images_dir))}",
            'if [[ ! -d "${SHARED_IMAGES_DIR}" ]]; then',
            '    echo "Shared image directory does not exist: ${SHARED_IMAGES_DIR}" >&2',
            "    exit 1",
            "fi",
            'IMAGE_LIST="${NODE_TMPDIR}/fish-vlm-required-images.nul"',
            'FISH_VLM_IMAGES_DIR="${NODE_TMPDIR}/fish-vlm-images"',
            'mkdir -p "${FISH_VLM_IMAGES_DIR}"',
            f'python -m fish_vlm.cli list-images --config {config_path} '
            f'--output "${{IMAGE_LIST}}" --missing-image-cache-only',
            'tar --directory="${SHARED_IMAGES_DIR}" --create --file=- '
            '--null --verbatim-files-from --files-from="${IMAGE_LIST}" '
            '| tar --directory="${FISH_VLM_IMAGES_DIR}" --extract --file=-',
            "export FISH_VLM_IMAGES_DIR",
        ]
    )
    return lines


def _append_runtime_setup(
    lines: list[str],
    config: dict[str, Any],
    *,
    cache_scope: str,
    stage_images: bool,
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
    lines.extend(
        [
            "export OMP_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            "export TORCH_NCCL_ASYNC_ERROR_HANDLING=1",
        ]
    )
    lines.extend(_cache_setup_lines(config, scope=cache_scope))
    if stage_images:
        lines.extend(_image_setup_lines(config))


def render_batch_script(config: dict[str, Any], training_config: str) -> str:
    """Render a one-node, all-GPU torchrun job."""
    slurm = config["slurm"]
    gpus = int(slurm.get("gpus", 1))
    array_configs = [
        str(path) for path in slurm.get("array_configs", [])
    ]
    max_concurrent = int(slurm.get("array_max_concurrent", 0))
    if max_concurrent < 0:
        raise ValueError("slurm.array_max_concurrent cannot be negative")
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={slurm.get('job_name', 'fish-vlm')}",
        "#SBATCH --nodes=1",
        f"#SBATCH --gpus={gpus}",
        f"#SBATCH --cpus-per-task={int(slurm.get('cpus', 16))}",
        f"#SBATCH --mem={slurm.get('memory', '64G')}",
        f"#SBATCH --time={slurm.get('time_limit', '24:00:00')}",
    ]
    if array_configs:
        concurrency = f"%{max_concurrent}" if max_concurrent else ""
        lines.append(
            f"#SBATCH --array=0-{len(array_configs) - 1}{concurrency}"
        )
    _append_optional_directives(lines, slurm)
    log_dir = slurm.get("log_dir", "outputs/slurm")
    log_suffix = "%A_%a" if array_configs else "%j"
    lines.extend(
        [
            f"#SBATCH --output={log_dir}/%x-{log_suffix}.out",
            f"#SBATCH --error={log_dir}/%x-{log_suffix}.err",
        ]
    )
    _append_runtime_setup(
        lines,
        config,
        cache_scope="training",
        stage_images=False,
    )
    if array_configs:
        lines.extend(
            [
                "SWEEP_CONFIGS=(",
                *(f"    {shlex.quote(path)}" for path in array_configs),
                ")",
                'TRAINING_CONFIG="${SWEEP_CONFIGS[${SLURM_ARRAY_TASK_ID}]}"',
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
    cache_scope: str = "training",
    stage_images: bool = False,
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
        lines.append(f"#SBATCH --gpus={gpus}")
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
        cache_scope=cache_scope,
        stage_images=stage_images,
    )
    lines.extend(["", *(shlex.join(command) for command in commands)])
    return "\n".join(lines) + "\n"
