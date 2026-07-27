"""Prepare canonical prompts."""
import sys
from fish_vlm.cli import main
raise SystemExit(main(["prepare-prompts", *sys.argv[1:]]))
