"""Fit branch calibration."""
import sys
from fish_vlm.cli import main
raise SystemExit(main(["calibrate", *sys.argv[1:]]))
