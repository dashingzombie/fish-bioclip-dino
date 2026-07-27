#!/usr/bin/env bash
#SBATCH --account=worm-species
#SBATCH --job-name=fish-finalisation
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

python -m fish_vlm.cli evaluate --config /faststorage/project/worm-species/fish-bioclip-dino/configs/inference/seen.yaml --checkpoint /faststorage/project/worm-species/fish-bioclip-dino/outputs/checkpoints/best.pt --output /faststorage/project/worm-species/fish-bioclip-dino/outputs/metrics/final_seen_evaluation.json
python -m fish_vlm.cli calibrate --config /faststorage/project/worm-species/fish-bioclip-dino/configs/inference/seen.yaml --checkpoint /faststorage/project/worm-species/fish-bioclip-dino/outputs/checkpoints/best.pt --output /faststorage/project/worm-species/fish-bioclip-dino/outputs/metrics/calibration.json
python -m fish_vlm.cli infer --config /faststorage/project/worm-species/fish-bioclip-dino/configs/inference/seen.yaml --checkpoint /faststorage/project/worm-species/fish-bioclip-dino/outputs/checkpoints/best.pt --calibration /faststorage/project/worm-species/fish-bioclip-dino/outputs/metrics/calibration.json --output /faststorage/project/worm-species/fish-bioclip-dino/outputs/predictions/test.json
python -m fish_vlm.cli infer --config /faststorage/project/worm-species/fish-bioclip-dino/configs/inference/unseen.yaml --checkpoint /faststorage/project/worm-species/fish-bioclip-dino/outputs/checkpoints/best.pt --calibration /faststorage/project/worm-species/fish-bioclip-dino/outputs/metrics/calibration.json --output /faststorage/project/worm-species/fish-bioclip-dino/outputs/predictions/unseen.json
python -m fish_vlm.cli merge-submission --test /faststorage/project/worm-species/fish-bioclip-dino/outputs/predictions/test.json --unseen /faststorage/project/worm-species/fish-bioclip-dino/outputs/predictions/unseen.json --output /faststorage/project/worm-species/fish-bioclip-dino/outputs/submissions/prediction.json
python -m fish_vlm.cli validate-submission --submission /faststorage/project/worm-species/fish-bioclip-dino/outputs/submissions/prediction.json --config /faststorage/project/worm-species/fish-bioclip-dino/configs/pipeline.yaml
python -m fish_vlm.cli pipeline-summary --config /faststorage/project/worm-species/fish-bioclip-dino/configs/pipeline.yaml
