"""Public entry point for the all-image DINO/BioCLIP workflow."""

from __future__ import annotations

import argparse
import json

from fish_vlm.domain.workflow import plan_as_json, submit_all_data_plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/all_data/common.yaml")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--submit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        if args.resume:
            parser.error("--resume is only valid with --submit")
        print(plan_as_json(args.config))
    else:
        print(json.dumps(submit_all_data_plan(args.config, resume=args.resume), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
