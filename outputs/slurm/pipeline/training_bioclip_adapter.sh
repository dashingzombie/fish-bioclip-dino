#!/usr/bin/env bash
#SBATCH --account=worm-species
#SBATCH --job-name=fish-training-bioclip-adapter
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
python -m fish_vlm.cli train --config /faststorage/project/worm-species/fish-bioclip-dino/configs/train/bioclip_adapter.yaml
