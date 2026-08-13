"""Safe SLURM dry-run and submission."""

from __future__ import annotations

import subprocess
from pathlib import Path

from fish_vlm.utils.io import atomic_write_text


def submit_slurm_script(
    script: str,
    path: str | Path,
    *,
    dependency: str | None = None,
) -> str:
    """Persist and submit one script with an optional strict dependency."""
    script_path = Path(path)
    script_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(script_path, script)
    command = ["sbatch", "--parsable"]
    if dependency:
        command.append(f"--dependency=afterok:{dependency}")
    command.append(str(script_path))
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or str(error)).strip()
        raise RuntimeError(
            f"Slurm submission failed for {script_path}: {detail}"
        ) from error
    return result.stdout.strip().split(";", 1)[0]
