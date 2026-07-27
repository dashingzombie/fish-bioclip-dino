"""Evaluate a checkpoint or Stage 0 baseline."""
import sys
from fish_vlm.cli import main
raise SystemExit(main(["evaluate", *sys.argv[1:]]))
