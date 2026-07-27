"""Phased sweep planner with small, explicit experiment sets."""

from __future__ import annotations

import itertools
import subprocess
from pathlib import Path
from typing import Any

import yaml

from fish_vlm.sweeps.state import load_state, save_state
from fish_vlm.utils.hashing import stable_json_hash
from fish_vlm.utils.io import atomic_write_text, write_json


def _expand_phase(phase: dict[str, Any]) -> list[dict[str, Any]]:
    parameters = phase.get("parameters", {})
    keys = sorted(parameters)
    values = [parameters[key] if isinstance(parameters[key], list) else [parameters[key]] for key in keys]
    return [
        {"phase": phase["name"], "overrides": dict(zip(keys, combination, strict=True))}
        for combination in itertools.product(*values)
    ] or [{"phase": phase["name"], "overrides": {}}]


def _nested_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for dotted, value in overrides.items():
        target = result
        parts = dotted.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return result


def run_pipeline(config: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    """Materialise one phase at a time; optionally submit its explicit configs."""
    sweep = config["sweep"]
    root = Path(config.get("output_dir", "outputs")) / "sweep_pipelines" / sweep.get("name", "multimodal")
    state_path = root / "state.json"
    state = load_state(state_path)
    manifest: list[dict[str, Any]] = []
    for phase_index, phase in enumerate(sweep["phases"], start=1):
        experiments = _expand_phase(phase)
        for experiment_index, experiment in enumerate(experiments, start=1):
            experiment["id"] = stable_json_hash(experiment)[:12]
            experiment["phase_index"] = phase_index
            experiment["experiment_index"] = experiment_index
        manifest.extend(experiments)
        if not dry_run and phase.get("submit", False):
            for experiment in experiments:
                config_path = root / "configs" / f"{experiment['id']}.yaml"
                generated = {
                    "defaults": [str(Path(config["_config_path"]).resolve())],
                    **_nested_overrides(experiment["overrides"]),
                }
                atomic_write_text(config_path, yaml.safe_dump(generated, sort_keys=True))
                subprocess.run(
                    ["python", "-m", "fish_vlm.cli", "slurm", "--config", str(config_path)],
                    check=True,
                )
        state["phases"][phase["name"]] = {
            "planned": len(experiments),
            "status": "planned" if dry_run else "materialised",
        }
    write_json(root / "manifest.json", manifest)
    save_state(state_path, state)
    return {"dry_run": dry_run, "experiments": len(manifest), "manifest": str(root / "manifest.json")}
