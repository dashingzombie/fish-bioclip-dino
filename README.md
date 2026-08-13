# All-image DINO / BioCLIP fish pipeline

This branch has one recommended GenomeDK workflow:

```bash
make all-data-dry-run
make all-data
make all-data-resume
```

`all-data-dry-run` writes and prints the complete Slurm DAG without calling
`sbatch`. `all-data` submits it. `all-data-resume` submits the same DAG with
epoch-level resume enabled for the two domain-adaptation jobs. No GPU work is
submitted by tests or dry-runs.

## Scientific contract

The workflow deliberately separates labels from images:

| Input | DINO domain stage | BioCLIP domain stage | Supervised DINO stage |
|---|---|---|---|
| Labeled seen image | self-distillation only | frozen-text alignment, original-BioCLIP distillation, view consistency | seen classifier |
| Image without text/label | self-distillation only | original-BioCLIP distillation and view consistency only | excluded |
| Text for a class without images | not used | frozen text prototype and inference candidate | excluded |

Only `label_train.json` entries whose filenames belong to `train.pkl` may be
used as labels. Images from `test.pkl` and `unseen.pkl` participate only as
unlabeled domain-adaptation inputs. The generated
`outputs/all_data/domain_manifest.json` records that boundary, and
`outputs/all_data/prompt_inventory.json` lists every class, prompt condition,
image count, and paired training image.

The labeled seen split uses class-aware 10% validation. A species represented
by one image is always training-only; every other species retains at least one
training image. The same deterministic split is used throughout.

## Model stages

1. Prepare class partitions, five-condition BioCLIP prompt ensembles, the
   species-disjoint pseudo-unseen split, the missing-modality manifest, and
   original-BioCLIP teacher embeddings for every image.
2. Adapt DINOv3 using two global and four constrained local fish views. The
   training projection head is discarded and no species classifier exists in
   this stage. Early stopping is evaluated each epoch with patience 30.
3. In parallel with DINO, adapt BioCLIP without a classifier head. Its text
   encoder stays frozen. Labeled images use text-prototype alignment; every
   image uses pretrained-image distillation and view consistency. Fine-tuning
   begins with two final visual blocks, then exposes the full visual tower.
   Selection uses the harmonic mean of class-balanced seen and
   species-disjoint pseudo-unseen accuracy, with patience 30 epochs.
4. Add a seen-species classifier only to DINO and fine-tune it with
   class-balanced sampling and conservative fish augmentation. Validation is
   once per epoch with patience 30.
5. Hard-route `test.pkl` to DINO and `unseen.pkl` to classifier-free BioCLIP
   similarity restricted to unseen candidate text prototypes. Merge, validate,
   and package one deterministic submission ZIP.

## GenomeDK parallelism

GenomeDK's submission filter requires an explicit GPU request even for metadata
jobs, so every job requests exactly one GPU. DINO adaptation starts concurrently with BioCLIP asset construction;
after each branch advances, DINO seen fine-tuning can overlap BioCLIP visual
adaptation. Final inference waits for both resulting checkpoints.

```text
prepare-metadata (1 GPU; required by GenomeDK policy)
├── dino-domain (1 GPU) ───────── dino-seen-finetune (1 GPU) ──┐
└── bioclip-assets (1 GPU) ────── bioclip-domain (1 GPU) ───────┤
                                                               └── finalise (1 GPU)
```

Configurations live under `configs/all_data/`. The current augmentation,
learning-rate, loss-weight, validation, and patience values are all explicit in
`configs/all_data/common.yaml` and `configs/all_data/dino_finetune.yaml`.

## Validation boundary

```bash
make test
make all-data-dry-run
```

Tests and dry-runs validate code, data-role, dependency, and output contracts.
They do not establish pretrained-model availability, GPU memory fit, GenomeDK
queue behavior, completed training, or competition accuracy.

The earlier six-recipe gated workflow remains available through `make
hybrid-dry-run`, `make hybrid`, and `make hybrid-resume` for recovery and
comparison, but it is not the recommended entry point on this branch.
