"""Safe SLURM dry-run and submission."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from fish_vlm.slurm.templates import render_batch_script
from fish_vlm.utils.io import atomic_write_text


def launch_slurm(config: dict[str, Any], *, dry_run: bool) -> str:
    """Render a script; submit only when explicitly not in dry-run mode."""
    slurm = config["slurm"]
    training_config = config.get("_config_path", "configs/base.yaml")

    script = render_batch_script(config, training_config)

    if dry_run:
        return script

    script_dir = Path(slurm.get("script_dir", "outputs/slurm"))
    log_dir = Path(slurm.get("log_dir", "outputs/slurm"))

    script_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    is_array = bool(slurm.get("array_configs"))
    job_name = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "-",
        str(slurm.get("job_name", "fish-vlm")),
    ).strip("-") or "fish-vlm"
    script_name = f"{job_name}-{'array' if is_array else 'job'}.sh"
    script_path = script_dir / script_name

    atomic_write_text(script_path, script)

    result = subprocess.run(
        ["sbatch", "--parsable", str(script_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    return result.stdout.strip()
