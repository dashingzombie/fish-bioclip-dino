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

Metadata and packaging use GenomeDK's CPU `normal` partition without requesting
a GPU. Every model/cache/inference job requests one GPU from `gpu-h200`. DINO
adaptation starts concurrently with BioCLIP asset construction;
after each branch advances, DINO seen fine-tuning can overlap BioCLIP visual
adaptation. Final inference waits for both resulting checkpoints.

These are separate dependency-aware `sbatch` jobs, intentionally not a Slurm
array: their commands, resources, and parent dependencies differ. Jobs with the
same satisfied parent are submitted independently and may occupy different GPU
nodes at the same time.

Each GPU job stages its own copy of the 99,924-image union because node-local
NVMe is not shared between nodes. When multiple jobs land on the same node,
they use a dataset-hashed `flock` under `/tmp`: one job stages and validates the
images, and the others reuse that completed directory. The logs print lock,
reuse/start, periodic tar checkpoint, verified file-count, and elapsed-time
messages.

```text
prepare-metadata (CPU, normal)
├── dino-domain (1 GPU) ───────── dino-seen-finetune (1 GPU) ──┐
└── bioclip-assets (1 GPU) ────── bioclip-domain (1 GPU) ───────┤
                                                               ├── infer-seen (1 GPU) ───┐
                                                               └── infer-unseen (1 GPU) ──┤
                                                                                            └── package (CPU, normal)
```

Configurations live under `configs/all_data/`. Augmentation, learning-rate,
loss-weight, validation, and patience values are explicit in
`configs/all_data/common.yaml` and `configs/all_data/dino_finetune.yaml`.

For routine resource changes, edit only `configs/all_data/resources.yaml`. It
contains the CPU/GPU partitions, cores, memory, time limits, worker counts, and
DINO-domain, BioCLIP-domain, and supervised-DINO batch parameters.

W&B is enabled for all three training jobs: DINO domain adaptation, BioCLIP
domain adaptation, and supervised DINO fine-tuning. Each gets a separate run
under the `fish-dino-bioclip` project and logs epoch/step losses, validation
and early-stopping selection metrics, learning rates, throughput, peak GPU
memory, and the best-checkpoint path. Checkpoints remain on GenomeDK; they are not
uploaded as W&B artifacts.

## Validation boundary

```bash
make test
make all-data-dry-run
```

Tests and dry-runs validate code, data-role, dependency, and output contracts.
They do not establish pretrained-model availability, GPU memory fit, GenomeDK
queue behavior, completed training, or competition accuracy.
