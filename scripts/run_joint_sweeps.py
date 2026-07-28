"""Run phased joint-supervised-text sweeps through the main CLI."""

import sys

from fish_vlm.cli import main

raise SystemExit(main(["joint-sweep", *sys.argv[1:]]))
