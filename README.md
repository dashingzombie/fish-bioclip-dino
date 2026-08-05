# DINO-seen / BioCLIP-unseen fish classifier

This branch has one production entry point and produces four directly comparable
submission ZIPs from one final DINO checkpoint:

1. **Pretrained BioCLIP, hard routed:** full-backbone DINO species
   classification for known/test images; pretrained BioCLIP 2 scientific-name
   similarity restricted to unseen species for unseen images.
2. **Pretrained BioCLIP, confidence gated:** use DINO when its calibrated
   known-species confidence is high; otherwise use pretrained BioCLIP 2
   scientific-name similarity across **all** candidates.
3. **Fine-tuned BioCLIP, hard routed:** the same final DINO checkpoint, with a
   long visual-only BioCLIP fine-tune as the unseen fallback.
4. **Fine-tuned BioCLIP, confidence gated:** the same DINO checkpoint and the
   fine-tuned BioCLIP visual tower, with an independently fitted leakage-safe
   confidence gate.

No official unseen image label is loaded or used for training, recipe selection,
temperature fitting, or threshold selection.

## One run point

Set `data.root_dir` in `configs/base.yaml`, then inspect the exact six-recipe
full-DINO sweep, long BioCLIP fine-tune, four submissions, and single Slurm
allocation without calling `sbatch`:

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

1. Prepare class partitions, scientific-name text prototypes, BioCLIP teacher
   features, the fixed seed-42 species holdout, and deterministic image caches.
2. Train six full-backbone DINO classifiers for up to 16,000 steps with label
   smoothing. Each calibration classifier omits every image belonging to its
   pseudo-unseen species.
3. Train a visual-only BioCLIP 2 calibration model for up to 20,000 steps on
   the same non-held-out species. The text encoder stays frozen; scientific-name
   alignment and pretrained-image distillation remain active.
4. For each DINO recipe, temperature-calibrate DINO on known validation images and
   search 201 confidence thresholds on known versus pseudo-unseen validation
   images. Rank recipes by estimated overall accuracy using only the official
   split *counts*.
5. Fit a separate fine-tuned-BioCLIP gate on that same pseudo-unseen validation
   split, then retrain the selected DINO recipe and BioCLIP fine-tune from their
   original pretrained weights on every seen species.
6. Refit only the final DINO temperature on held-out seen images. Transfer each
   gate's known-species acceptance rate onto the final classifier's confidence
   scale; never retune either gate on official unseen images.
7. Generate, validate, and deterministically package all four submissions.

The sweep is defined in [configs/hybrid/sweep.yaml](configs/hybrid/sweep.yaml).
It tests DINO backbone learning rates `1e-6`, `3e-6`, and `1e-5`, crossed with
classifier learning rates `1e-4` and `3e-4`. Both BioCLIP variants use the prompt
`A photograph of <scientific name>.`; fine-tuning changes only its visual tower,
and checkpoint loading rejects any changed text/nonvisual weights.

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
├── bioclip/
│   ├── calibration/{resolved_config.yaml,checkpoints/best.pt,gate.json}
│   └── final/{resolved_config.yaml,checkpoints/best.pt,gate.json}
├── final/
│   ├── resolved_config.yaml
│   ├── checkpoints/best.pt
│   └── gate.json
└── submissions/
    ├── pretrained_bioclip_hard_routed/{prediction.json,submission.zip}
    ├── pretrained_bioclip_confidence_gated/{prediction.json,submission.zip}
    ├── finetuned_bioclip_hard_routed/{prediction.json,submission.zip}
    └── finetuned_bioclip_confidence_gated/{prediction.json,submission.zip}
```

`gate.json` is content-hashed and records the exact supervised species,
checkpoint step, threshold origin, temperature, pseudo-unseen split hash, and
the explicit fact that official unseen labels were not used. Gated submissions
allow any species from `all_classes.pkl`; hard-routed submissions retain strict
seen/unseen candidate-partition validation.

## Validation boundary

Run repository tests with:

```bash
make test
```

Tests and `make hybrid-dry-run` validate orchestration, class ordering, gate
selection, configuration, and ZIP contracts. They do not constitute real DINO
training, pretrained-model execution, GPU/DDP execution, Slurm submission, or a
competition result.
