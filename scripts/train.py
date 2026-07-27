"""Run staged training."""
import sys
from fish_vlm.cli import main
raise SystemExit(main(["train", *sys.argv[1:]]))
