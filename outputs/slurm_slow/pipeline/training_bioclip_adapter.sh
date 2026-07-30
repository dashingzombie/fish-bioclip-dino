#!/usr/bin/env bash
#SBATCH --job-name=fish-training-bioclip-adapter
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
NODE_TMPDIR="${TMPDIR:-${SLURM_TMPDIR:-}}"
if [[ -z "${NODE_TMPDIR}" ]]; then
    echo "TMPDIR or SLURM_TMPDIR must be set for node-local staging" >&2
    exit 1
fi
if [[ ! -d "${SHARED_CACHE_DIR}" ]]; then
    echo "Shared cache directory does not exist: ${SHARED_CACHE_DIR}" >&2
    exit 1
fi
FISH_VLM_CACHE_DIR="${NODE_TMPDIR}"/fish-vlm-cache
mkdir -p "${FISH_VLM_CACHE_DIR}"
CACHE_ITEMS=(
    huggingface
    torch
    text
    bioclip_images/train_embeddings.pt
    image_transforms/train/manifest.json
    image_transforms/train/dino.npy
    image_transforms/train/bioclip.npy
    image_transforms/test/manifest.json
    image_transforms/test/dino.npy
    image_transforms/test/bioclip.npy
    image_transforms/unseen/manifest.json
    image_transforms/unseen/dino.npy
    image_transforms/unseen/bioclip.npy
)
CACHE_COPY_PIDS=()
for relative in "${CACHE_ITEMS[@]}"; do
    source_path="${SHARED_CACHE_DIR}/${relative}"
    if [[ ! -e "${source_path}" ]]; then
        continue
    fi
    destination_path="${FISH_VLM_CACHE_DIR}/${relative}"
    mkdir -p "$(dirname "${destination_path}")"
    cp --archive --reflink=auto "${source_path}" "${destination_path}" &
    CACHE_COPY_PIDS+=("$!")
done
for pid in "${CACHE_COPY_PIDS[@]}"; do
    wait "${pid}"
done
export FISH_VLM_CACHE_DIR
export HF_HOME="${FISH_VLM_CACHE_DIR}/huggingface"
export TORCH_HOME="${FISH_VLM_CACHE_DIR}/torch"
mkdir -p "${HF_HOME}" "${TORCH_HOME}"

python -m fish_vlm.cli train --config /faststorage/project/worm-species/fish-bioclip-dino/configs/train/bioclip_adapter.yaml
python -m fish_vlm.cli calibrate --config /faststorage/project/worm-species/fish-bioclip-dino/configs/inference/seen_adapter.yaml --checkpoint /faststorage/project/worm-species/fish-bioclip-dino/outputs/checkpoints/best-adapter.pt --output /faststorage/project/worm-species/fish-bioclip-dino/outputs/submissions/stages/bioclip_adapter/calibration.json
python -m fish_vlm.cli infer --config /faststorage/project/worm-species/fish-bioclip-dino/configs/inference/seen_adapter.yaml --checkpoint /faststorage/project/worm-species/fish-bioclip-dino/outputs/checkpoints/best-adapter.pt --calibration /faststorage/project/worm-species/fish-bioclip-dino/outputs/submissions/stages/bioclip_adapter/calibration.json --output /faststorage/project/worm-species/fish-bioclip-dino/outputs/submissions/stages/bioclip_adapter/test.json
python -m fish_vlm.cli infer --config /faststorage/project/worm-species/fish-bioclip-dino/configs/inference/unseen_adapter.yaml --checkpoint /faststorage/project/worm-species/fish-bioclip-dino/outputs/checkpoints/best-adapter.pt --calibration /faststorage/project/worm-species/fish-bioclip-dino/outputs/submissions/stages/bioclip_adapter/calibration.json --output /faststorage/project/worm-species/fish-bioclip-dino/outputs/submissions/stages/bioclip_adapter/unseen.json
python -m fish_vlm.cli merge-submission --test /faststorage/project/worm-species/fish-bioclip-dino/outputs/submissions/stages/bioclip_adapter/test.json --unseen /faststorage/project/worm-species/fish-bioclip-dino/outputs/submissions/stages/bioclip_adapter/unseen.json --output /faststorage/project/worm-species/fish-bioclip-dino/outputs/submissions/stages/bioclip_adapter/prediction.json
python -m fish_vlm.cli validate-submission --submission /faststorage/project/worm-species/fish-bioclip-dino/outputs/submissions/stages/bioclip_adapter/prediction.json --config /faststorage/project/worm-species/fish-bioclip-dino/configs/pipeline.yaml
python -m fish_vlm.cli package-submission --submission /faststorage/project/worm-species/fish-bioclip-dino/outputs/submissions/stages/bioclip_adapter/prediction.json --output /faststorage/project/worm-species/fish-bioclip-dino/outputs/submissions/stages/bioclip_adapter/submission.zip
