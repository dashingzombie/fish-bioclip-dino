"""Build class-level validation splits."""
import sys
from fish_vlm.cli import main
raise SystemExit(main(["make-pseudo-unseen", *sys.argv[1:]]))
