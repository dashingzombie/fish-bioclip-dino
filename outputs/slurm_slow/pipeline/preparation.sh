#!/usr/bin/env bash
#SBATCH --job-name=fish-preparation
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --account=worm-species
#SBATCH --partition=gpu-h200,gpu-l40s
#SBATCH --output=outputs/slurm_slow/%x-%j.out
#SBATCH --error=outputs/slurm_slow/%x-%j.err

set -euo pipefail
cd .
source .venv/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

SHARED_CACHE_DIR=cache
if [[ "${SHARED_CACHE_DIR}" != /* ]]; then
    SHARED_CACHE_DIR="${PWD}/${SHARED_CACHE_DIR}"
fi
mkdir -p "${SHARED_CACHE_DIR}"
FISH_VLM_CACHE_DIR="${SHARED_CACHE_DIR}"
export FISH_VLM_CACHE_DIR
export HF_HOME="${FISH_VLM_CACHE_DIR}/huggingface"
export TORCH_HOME="${FISH_VLM_CACHE_DIR}/torch"
mkdir -p "${HF_HOME}" "${TORCH_HOME}"

NODE_TMPDIR="${TMPDIR:-${SLURM_TMPDIR:-}}"
if [[ -z "${NODE_TMPDIR}" ]]; then
    echo "TMPDIR or SLURM_TMPDIR must be set for node-local staging" >&2
    exit 1
fi
SHARED_IMAGES_DIR=/faststorage/project/worm-species/fish-data/images
if [[ ! -d "${SHARED_IMAGES_DIR}" ]]; then
    echo "Shared image directory does not exist: ${SHARED_IMAGES_DIR}" >&2
    exit 1
fi
IMAGE_LIST="${NODE_TMPDIR}/fish-vlm-required-images.nul"
FISH_VLM_IMAGES_DIR="${NODE_TMPDIR}/fish-vlm-images"
mkdir -p "${FISH_VLM_IMAGES_DIR}"
python -m fish_vlm.cli list-images --config /faststorage/project/worm-species/fish-bioclip-dino/configs/pipeline.yaml --output "${IMAGE_LIST}" --missing-image-cache-only
tar --directory="${SHARED_IMAGES_DIR}" --create --file=- --null --verbatim-files-from --files-from="${IMAGE_LIST}" | tar --directory="${FISH_VLM_IMAGES_DIR}" --extract --file=-
export FISH_VLM_IMAGES_DIR

python -m fish_vlm.cli prepare-prompts --config /faststorage/project/worm-species/fish-bioclip-dino/configs/pipeline.yaml
python -m fish_vlm.cli build-text-prototypes --config /faststorage/project/worm-species/fish-bioclip-dino/configs/pipeline.yaml
python -m fish_vlm.cli make-pseudo-unseen --config /faststorage/project/worm-species/fish-bioclip-dino/configs/pipeline.yaml
python -m fish_vlm.cli build-image-cache --config /faststorage/project/worm-species/fish-bioclip-dino/configs/pipeline.yaml
python -m fish_vlm.cli build-teacher-cache --config /faststorage/project/worm-species/fish-bioclip-dino/configs/pipeline.yaml
