#!/usr/bin/env bash
#SBATCH --account=worm-species
#SBATCH --job-name=fish-preparation
#SBATCH --nodes=1
#SBATCH --partition=gpu-l40s,gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
set -euo pipefail
cd .
source .venv/bin/activate
python -m fish_vlm.cli prepare-prompts --config /faststorage/project/worm-species/fish-bioclip-dino/configs/pipeline.yaml
python -m fish_vlm.cli build-text-prototypes --config /faststorage/project/worm-species/fish-bioclip-dino/configs/pipeline.yaml
python -m fish_vlm.cli make-pseudo-unseen --config /faststorage/project/worm-species/fish-bioclip-dino/configs/pipeline.yaml
python -m fish_vlm.cli build-teacher-cache --config /faststorage/project/worm-species/fish-bioclip-dino/configs/pipeline.yaml
