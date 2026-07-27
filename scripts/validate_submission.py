"""Validate a merged submission."""
import sys
from fish_vlm.cli import main
raise SystemExit(main(["validate-submission", *sys.argv[1:]]))
