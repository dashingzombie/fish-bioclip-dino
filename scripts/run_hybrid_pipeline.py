"""Single public run point for DINO/BioCLIP hybrid submissions."""

from __future__ import annotations

import argparse
import json

from fish_vlm.hybrid.workflow import (
    plan_as_json,
    run_hybrid_pipeline,
    submit_hybrid_pipeline,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="configs/hybrid/sweep.yaml")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--submit", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        if args.resume:
            parser.error("--resume is only valid with --submit or --run")
        print(plan_as_json(args.spec))
    elif args.submit:
        print(json.dumps({"job_id": submit_hybrid_pipeline(args.spec, resume=args.resume)}))
    else:
        print(json.dumps(run_hybrid_pipeline(args.spec, resume=args.resume), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
