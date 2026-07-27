"""Build BioCLIP text caches."""
import sys
from fish_vlm.cli import main
raise SystemExit(main(["build-text-prototypes", *sys.argv[1:]]))
