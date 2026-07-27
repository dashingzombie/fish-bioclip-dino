"""Run the complete fish DINOv3–BioCLIP pipeline locally or on SLURM."""

import sys

from fish_vlm.cli import main

raise SystemExit(main(["run-all", *sys.argv[1:]]))
