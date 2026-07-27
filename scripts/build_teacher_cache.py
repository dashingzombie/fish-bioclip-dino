"""Build BioCLIP image-teacher cache."""
import sys
from fish_vlm.cli import main
raise SystemExit(main(["build-teacher-cache", *sys.argv[1:]]))
