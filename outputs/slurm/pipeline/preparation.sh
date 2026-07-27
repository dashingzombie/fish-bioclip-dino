#!/usr/bin/env bash
#SBATCH --account=worm-species
#SBATCH --job-name=fish-preparation
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
mkdir -p "${SHARED_CACHE_DIR}"
FISH_VLM_CACHE_DIR="${SHARED_CACHE_DIR}"
export FISH_VLM_CACHE_DIR
export HF_HOME="${FISH_VLM_CACHE_DIR}/huggingface"
export TORCH_HOME="${FISH_VLM_CACHE_DIR}/torch"
mkdir -p "${HF_HOME}" "${TORCH_HOME}"

python -m fish_vlm.cli prepare-prompts --config /faststorage/project/worm-species/fish-bioclip-dino/configs/pipeline.yaml
python -m fish_vlm.cli build-text-prototypes --config /faststorage/project/worm-species/fish-bioclip-dino/configs/pipeline.yaml
python -m fish_vlm.cli make-pseudo-unseen --config /faststorage/project/worm-species/fish-bioclip-dino/configs/pipeline.yaml
python -m fish_vlm.cli build-teacher-cache --config /faststorage/project/worm-species/fish-bioclip-dino/configs/pipeline.yaml
