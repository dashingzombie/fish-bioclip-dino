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
    if scope == "dino_domain":
        return ["torch"]
    if scope == "training":
        return [
            "huggingface",
            "torch",
            "text",
            "bioclip_images/train_embeddings.pt",
            "bioclip_images/all_embeddings.pt",
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
    """Stage the all-data image union once per physical node."""
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
            f'python -m fish_vlm.cli list-images --config {config_path} '
            f'--output "${{IMAGE_LIST}}"',
            'IMAGE_COUNT="$(python -c \'import pathlib, sys; print(pathlib.Path(sys.argv[1]).read_bytes().count(b"\\0"))\' "${IMAGE_LIST}")"',
            'IMAGE_LIST_HASH="$(sha256sum "${IMAGE_LIST}" | cut -d " " -f 1)"',
            'NODE_STAGE_KEY="fish-vlm-${USER}-${IMAGE_LIST_HASH}"',
            'NODE_STAGE_DIR="/tmp/${NODE_STAGE_KEY}"',
            'NODE_STAGE_LOCK="/tmp/${NODE_STAGE_KEY}.lock"',
            'NODE_STAGE_MARKER="${NODE_STAGE_DIR}/.fish-vlm-stage-complete"',
            'echo "Waiting for node-local staging lock: ${NODE_STAGE_LOCK}"',
            'exec 9>"${NODE_STAGE_LOCK}"',
            "flock 9",
            'EXPECTED_MARKER="${IMAGE_LIST_HASH} ${IMAGE_COUNT}"',
            'if [[ -f "${NODE_STAGE_MARKER}" ]] && [[ "$(<"${NODE_STAGE_MARKER}")" == "${EXPECTED_MARKER}" ]]; then',
            '    STAGED_COUNT="$(find "${NODE_STAGE_DIR}" -type f ! -name .fish-vlm-stage-complete -printf x | wc -c)"',
            'else',
            '    STAGED_COUNT=0',
            'fi',
            'if [[ "${STAGED_COUNT}" -eq "${IMAGE_COUNT}" ]]; then',
            '    echo "Reusing node-local image staging: ${STAGED_COUNT} files at ${NODE_STAGE_DIR}"',
            'else',
            '    if [[ -e "${NODE_STAGE_DIR}" ]]; then',
            '        STALE_STAGE="${NODE_STAGE_DIR}.stale-${SLURM_JOB_ID}-$(date +%s)"',
            '        echo "Moving incomplete node-local staging aside: ${STALE_STAGE}"',
            '        mv "${NODE_STAGE_DIR}" "${STALE_STAGE}"',
            '    fi',
            '    BUILD_STAGE="$(mktemp -d "/tmp/${NODE_STAGE_KEY}.building-${SLURM_JOB_ID}.XXXXXX")"',
            '    STAGING_STARTED="$(date +%s)"',
            '    echo "Starting node-local image staging: ${IMAGE_COUNT} files"',
            '    tar --directory="${SHARED_IMAGES_DIR}" --create --file=- '
            '    --null --verbatim-files-from --files-from="${IMAGE_LIST}" '
            '    --checkpoint=10000 '
            "    --checkpoint-action='echo=staging source checkpoint %u' "
            '    | tar --directory="${BUILD_STAGE}" --extract --file=-',
            '    STAGING_FINISHED="$(date +%s)"',
            '    STAGED_COUNT="$(find "${BUILD_STAGE}" -type f -printf x | wc -c)"',
            '    if [[ "${STAGED_COUNT}" -ne "${IMAGE_COUNT}" ]]; then',
            '        echo "Image staging count mismatch: expected ${IMAGE_COUNT}, found ${STAGED_COUNT}" >&2',
            "        exit 1",
            "    fi",
            '    printf "%s\\n" "${EXPECTED_MARKER}" > "${BUILD_STAGE}/.fish-vlm-stage-complete"',
            '    mv "${BUILD_STAGE}" "${NODE_STAGE_DIR}"',
            '    echo "Completed node-local image staging: ${STAGED_COUNT} files in $((STAGING_FINISHED - STAGING_STARTED)) seconds"',
            "fi",
            "flock -u 9",
            'FISH_VLM_IMAGES_DIR="${NODE_STAGE_DIR}"',
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
