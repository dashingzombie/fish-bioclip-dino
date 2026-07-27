"""Safe SLURM dry-run and submission."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

from fish_vlm.slurm.templates import render_batch_script
from fish_vlm.utils.io import atomic_write_text


def launch_slurm(config: dict[str, Any], *, dry_run: bool) -> str:
    """Render a script; submit only when explicitly not in dry-run mode."""
    training_config = config.get("_config_path", "configs/base.yaml")
    script = render_batch_script(config, training_config)
    if dry_run:
        return script
    output_dir = Path(config["slurm"].get("script_dir", "outputs/slurm"))
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = output_dir / "fish-vlm-job.sh"
    atomic_write_text(script_path, script)
    result = subprocess.run(
        ["sbatch", "--parsable", str(script_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()

