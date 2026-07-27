"""Merge split predictions."""
import sys
from fish_vlm.cli import main
raise SystemExit(main(["merge-submission", *sys.argv[1:]]))
