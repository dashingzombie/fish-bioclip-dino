"""Predict an official split."""
import sys
from fish_vlm.cli import main
raise SystemExit(main(["infer", *sys.argv[1:]]))
