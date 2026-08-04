# DINO-seen / BioCLIP-unseen fish classifier

This branch has one production entry point and produces two directly comparable
submission ZIPs from one final DINO checkpoint:

1. **Hard routed:** full-backbone DINO species classification for the official
   known/test images; frozen BioCLIP 2 scientific-name similarity restricted to
   unseen species for official unseen images.
2. **Confidence gated:** use the DINO known-species prediction when its
   calibrated confidence exceeds a selected threshold; otherwise use frozen
   BioCLIP 2 scientific-name similarity across **all** candidate species.

No official unseen image label is loaded or used for training, recipe selection,
temperature fitting, or threshold selection.

## One run point

Set `data.root_dir` in `configs/base.yaml`, then inspect the exact six-recipe
full-DINO sweep and the single Slurm allocation without calling `sbatch`:

```bash
make hybrid-dry-run
```

Submit the complete workflow:

```bash
make hybrid
```

Resume after a failed allocation without repeating completed checkpoints,
calibrations, predictions, or ZIP packaging:

```bash
make hybrid-resume
```

All three commands call only
`scripts/run_hybrid_pipeline.py`. To execute inside an allocation instead of
submitting one, run:

```bash
python scripts/run_hybrid_pipeline.py \
  --spec configs/hybrid/sweep.yaml --run
```

## Leakage-safe fitting order

The runner performs the following sequence:

1. Prepare class partitions, scientific-name text prototypes, the fixed seed-42
   species holdout, and deterministic image caches.
2. Train six full-backbone DINO classifiers. Each calibration classifier omits
   every image belonging to its pseudo-unseen species.
3. For each recipe, temperature-calibrate DINO on known validation images and
   search 201 confidence thresholds on known versus pseudo-unseen validation
   images. Rank recipes by estimated overall accuracy using only the official
   split *counts*.
4. Retrain the selected recipe from pretrained DINO on every seen species.
5. Refit only the final DINO temperature on held-out seen images. Transfer the
   known-species acceptance rate selected in step 3 onto the final classifier's
   confidence scale; never retune the gate on official unseen images.
6. Generate, validate, and deterministically package both submissions.

The sweep is defined in [configs/hybrid/sweep.yaml](configs/hybrid/sweep.yaml).
It tests DINO backbone learning rates `1e-6`, `3e-6`, and `1e-5`, crossed with
classifier learning rates `1e-4` and `3e-4`. BioCLIP remains frozen and uses the
prompt `A photograph of <scientific name>.` for every fallback comparison.

## Outputs

The final artifacts are:

```text
outputs/hybrid/
├── plan.json
├── state.json
├── selection.json
├── summary.json
├── calibration/<recipe>/
│   ├── resolved_config.yaml
│   ├── checkpoints/best.pt
│   ├── metrics/best.json
│   └── gate.json
├── final/
│   ├── resolved_config.yaml
│   ├── checkpoints/best.pt
│   └── gate.json
└── submissions/
    ├── hard_routed/{prediction.json,submission.zip}
    └── confidence_gated/{prediction.json,submission.zip}
```

`gate.json` is content-hashed and records the exact supervised species,
checkpoint step, threshold origin, temperature, pseudo-unseen split hash, and
the explicit fact that official unseen labels were not used. A gated submission
allows any species from `all_classes.pkl`; the hard-routed submission retains
strict seen/unseen candidate-partition validation.

## Validation boundary

Run repository tests with:

```bash
make test
```

Tests and `make hybrid-dry-run` validate orchestration, class ordering, gate
selection, configuration, and ZIP contracts. They do not constitute real DINO
training, pretrained-model execution, GPU/DDP execution, Slurm submission, or a
competition result.
