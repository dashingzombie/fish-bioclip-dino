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
