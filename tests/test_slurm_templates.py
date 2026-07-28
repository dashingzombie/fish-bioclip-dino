from __future__ import annotations

from fish_vlm.config import load_config
from fish_vlm.slurm.templates import render_batch_script


def test_single_job_renderer_uses_slurm_config_and_node_cache() -> None:
    config = load_config("configs/slurm/genome.yaml")
    script = render_batch_script(config, config["_config_path"])

    assert "#SBATCH --gpus=4" in script
    assert "#SBATCH --account=worm-species" in script
    assert "#SBATCH --output=outputs/slurm/%x-%j.out\n" in script
    assert "#SBATCH --error=outputs/slurm/%x-%j.err\n" in script
    assert "#SBATCH --partition=gpu-h200" in script
    assert 'FISH_VLM_CACHE_DIR="${NODE_TMPDIR}"/fish-vlm-cache' in script
    assert 'cp --archive --reflink=auto "${source_path}"' in script
    assert "--nproc_per_node=4" in script


def test_array_renderer_uses_concurrency_and_task_config() -> None:
    config = load_config("configs/slurm/genome.yaml")
    config["slurm"]["array_configs"] = [
        "/work/configs/one.yaml",
        "/work/configs/two.yaml",
        "/work/configs/three.yaml",
    ]
    config["slurm"]["array_max_concurrent"] = 2
    script = render_batch_script(config, config["_config_path"])

    assert "#SBATCH --array=0-2%2" in script
    assert "#SBATCH --output=outputs/slurm/%x-%A_%a.out" in script
    assert "SWEEP_CONFIGS=(" in script
    assert "/work/configs/one.yaml" in script
    assert 'TRAINING_CONFIG="${SWEEP_CONFIGS[${SLURM_ARRAY_TASK_ID}]}"' in script
