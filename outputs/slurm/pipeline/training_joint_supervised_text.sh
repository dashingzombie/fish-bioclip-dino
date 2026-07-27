#!/usr/bin/env bash
#SBATCH --account=worm-species
#SBATCH --job-name=fish-training-joint-supervised-text
#SBATCH --nodes=1
#SBATCH --partition=gpu-l40s,gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
set -euo pipefail
cd .
source .venv/bin/activate

SHARED_CACHE_DIR=cache
if [[ "${SHARED_CACHE_DIR}" != /* ]]; then
    SHARED_CACHE_DIR="${PWD}/${SHARED_CACHE_DIR}"
fi
: "${SLURM_TMPDIR:?SLURM_TMPDIR must be set for node-local cache staging}"
if [[ ! -d "${SHARED_CACHE_DIR}" ]]; then
    echo "Shared cache directory does not exist: ${SHARED_CACHE_DIR}" >&2
    exit 1
fi
FISH_VLM_CACHE_DIR="${SLURM_TMPDIR}"/fish-vlm-cache
mkdir -p "${FISH_VLM_CACHE_DIR}"
cp -a "${SHARED_CACHE_DIR}/." "${FISH_VLM_CACHE_DIR}/"
export FISH_VLM_CACHE_DIR
export HF_HOME="${FISH_VLM_CACHE_DIR}/huggingface"
export TORCH_HOME="${FISH_VLM_CACHE_DIR}/torch"
mkdir -p "${HF_HOME}" "${TORCH_HOME}"

python -m fish_vlm.cli train --config /faststorage/project/worm-species/fish-bioclip-dino/configs/train/joint_supervised_text.yaml
