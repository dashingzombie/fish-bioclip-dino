# Fish DINOv3–BioCLIP

Production-oriented seen/unseen fish classification by aligning DINOv3 image
features with the frozen BioCLIP embedding space. The same canonical text
prototype matrix is used by the projected DINO branch and by BioCLIP's own
frozen image encoder. Their calibrated probabilities can be fused, and seen
classification may additionally use a supervised cosine head.

The implementation never assumes official `test.pkl` or `unseen.pkl` labels.
Unseen classes enter training artifacts only as text: one supplied description,
one deterministic canonical prompt, and one frozen BioCLIP text prototype.

## Run everything with one command

After setting `data.root_dir` in `configs/base.yaml`, the complete workflow can
run locally:

```bash
python scripts/run_all.py \
  --config configs/pipeline.yaml \
  --mode local \
  --gpus 1
```

or as an ordered SLURM dependency chain:

```bash
python scripts/run_all.py \
  --config configs/pipeline.yaml \
  --mode slurm \
  --gpus 4
```

Inspect either execution plan without running commands or calling `sbatch`:

```bash
python scripts/run_all.py --config configs/pipeline.yaml --mode local --gpus 1 --dry-run
python scripts/run_all.py --config configs/pipeline.yaml --mode slurm --gpus 4 --dry-run
```

The workflow performs prompt and deterministic image-cache preparation, all
three prototype-alignment/joint stages, the separate BioCLIP-adapter experiment,
final evaluation, calibration, inference, submission validation, and summary
generation. Every training stage also writes a deterministic `submission.zip`
below `outputs/submissions/stages/<stage>/`. Local state is saved in
`outputs/pipeline/workflow_state.json`; rerunning resumes completed steps.
Use `--force` to rerun every local step.

SLURM creates six jobs connected by `afterok`: preparation, four training jobs,
and finalisation. A failed job therefore prevents dependent jobs from using
partial outputs.

The final human-readable result is
`outputs/metrics/pipeline_summary.json`. It contains stage results, final branch
metrics, fitted calibration, submission counts, the best final seen branch, and
the exact modes used for official test and unseen prediction.

## Installation

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Edit `data.root_dir` in `configs/base.yaml` or a composed configuration. The
expected organiser layout is:

```text
data/
├── label_train.json
├── descriptions.json
├── all_classes.pkl
├── manual/canonical_prompt_overrides.json
├── splits/train.pkl
├── splits/test.pkl
├── splits/unseen.pkl
└── images/
```

## Preparation

The first command creates sorted seen/unseen/all class partitions, all forward
and reverse mappings, one prompt per species, and a JSONL audit. Sentence
filtering is deterministic and configurable in
`fish_vlm.data.descriptions`; a full-prompt manual override always wins.

```bash
python -m fish_vlm.cli prepare-prompts --config configs/base.yaml
python -m fish_vlm.cli build-text-prototypes --config configs/base.yaml
python -m fish_vlm.cli make-pseudo-unseen --config configs/base.yaml
python -m fish_vlm.cli build-image-cache --config configs/base.yaml
python -m fish_vlm.cli build-teacher-cache --config configs/base.yaml
```

Preparation creates the three text-prototype caches, the image-teacher cache,
and memory-mapped float16 DINO/BioCLIP transform caches for train, test, and
unseen splits. Every builder validates its destination before model loading. A
valid cache is reused; an incompatible cache fails instead of being overwritten.
Model downloads are kept below `cache/` through `HF_HOME` and `TORCH_HOME`.

The teacher cache reads only filenames from `train.pkl`, requires labels for
every included image, uses the deterministic BioCLIP evaluation transform, and
stores float16 embeddings converted to float32 for losses. Unit tests substitute
tiny deterministic encoders and perform no network access.

The default training configuration selects by
`estimated_overall_accuracy`. It therefore resolves the seed-42 split written
by `make-pseudo-unseen` and fails clearly if that command was skipped.

## Model branches

- **DINO-text:** `timm.create_model(name, pretrained=True, num_classes=0)`;
  features are obtained through `forward_features` followed by
  `forward_head(..., pre_logits=True)`. This
  `forward_features_then_forward_head_pre_logits` pooling identity is saved in
  checkpoints. A linear or MLP projector maps the dynamically inferred DINO
  dimension into the dynamically inferred BioCLIP dimension and unit-normalises
  in float32.
- **BioCLIP-native:** `open_clip.create_model_and_transforms(
  "hf-hub:imageomics/bioclip-2")` supplies both the frozen encoder and its own
  image transform. Its image embeddings are unit-normalised before prototype
  similarity. Adapter mode adds a zero-initialised residual MLP while the
  encoder stays frozen.
- **Supervised seen head:** a learned cosine classifier consumes unprojected
  DINO features. It is never accepted in unseen inference.

The dataset decodes each source image once but separately applies the DINO and
BioCLIP transforms. A DINO-normalised tensor is never reused by BioCLIP.

## Stages and training

Stage 0 is a native BioCLIP zero-shot evaluation:

```bash
python -m fish_vlm.cli evaluate --config configs/base.yaml
```

Stages 1–4:

```bash
torchrun --standalone --nproc_per_node=4 -m fish_vlm.cli train \
  --config configs/train/projection_only.yaml

torchrun --standalone --nproc_per_node=4 -m fish_vlm.cli train \
  --config configs/train/final_block.yaml

torchrun --standalone --nproc_per_node=4 -m fish_vlm.cli train \
  --config configs/train/joint_supervised_text.yaml

torchrun --standalone --nproc_per_node=4 -m fish_vlm.cli train \
  --config configs/train/bioclip_adapter.yaml
```

The later stage configurations specify `training.resume_checkpoint`; adjust
that path after retaining the preceding stage's best checkpoint. Stage 1 trains
only the projector and logit scale. Stage 2 additionally trains the final DINO
block and final normalisation at separate rates. Stage 3 adds the supervised
head. Stage 4 trains only the residual BioCLIP adapter.

The exact implemented objective is a non-normalised weighted sum of enabled
terms:

- DINO prototype cross-entropy;
- cosine (`1 - cosine`) or optional symmetric batch-contrastive image-teacher
  alignment;
- ordinary supervised seen-head cross-entropy;
- BioCLIP-adapter prototype cross-entropy;
- optional symmetric KL or Jensen–Shannon branch consistency.

Defaults are `1.0 * DINO text CE + 0.25 * cosine teacher`; joint training adds
`0.5 * supervised CE`. Similarities, normalisation, softmax/KL and
log-sum-exp-sensitive work are float32 even under AMP. BioCLIP parameters have
`requires_grad=False` and cannot enter AdamW groups.

Training is step-based: the four stages use 1200, 800, 800, and 600 optimizer
steps, with validation every 100 steps. The Genome profile targets one complete
`gpu-h200` node: four Hopper GPUs, 128 CPU cores, and 700 GB RAM. It uses BF16,
a 512-image microbatch per GPU, no gradient accumulation, static-graph DDP,
gradient bucket views, TF32, and fixed-shape cuDNN benchmarking. The global
training batch is 2048 images per optimizer step.

FSDP is intentionally not used. DINO and BioCLIP are frozen for most stages and
the complete model fits easily on each GPU; sharding would add parameter
all-gathers without removing a memory bottleneck. Cached-teacher stages also
skip the unused BioCLIP image forward.

## Pseudo-unseen validation

`species_holdout` removes complete species from every training loader.
`genus_holdout` removes complete genera. Seeds 7, 42 and 123 are generated.
Checkpoints store the exact ordered training-species list, its hash, and the
pseudo-split hash. Calibration fits the subset supervised head only on eligible
targets, expands its probabilities into the full seen ordering, and then fuses
them with full-space text probabilities. Checkpoint loading rejects missing or
incompatible class identities.

Seen and pseudo-unseen evaluation reports branch accuracy, balanced accuracy,
macro-F1 and safe top-5. Selection also records:

```text
H = 2 * seen_accuracy * pseudo_unseen_accuracy
    / (seen_accuracy + pseudo_unseen_accuracy)

estimated = (len(test.pkl) * seen_accuracy
             + len(unseen.pkl) * pseudo_unseen_accuracy)
            / (len(test.pkl) + len(unseen.pkl))
```

The default selector is `estimated_overall_accuracy`. The official image-only
splits are used only for their actual counts, never for labels or selection.

## Calibration, inference, and submission

Independent validation temperatures are fitted for DINO-text,
BioCLIP-native, and (when present) the supervised head. Grid-selected convex
weights then combine probabilities—not unrelated raw logits. Calibration JSON
contains its own content hash.

```bash
python -m fish_vlm.cli calibrate --config configs/base.yaml \
  --checkpoint outputs/checkpoints/best.pt

python -m fish_vlm.cli infer --config configs/inference/seen.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --output outputs/predictions/test.json

python -m fish_vlm.cli infer --config configs/inference/unseen.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --output outputs/predictions/unseen.json

python -m fish_vlm.cli merge-submission \
  --test outputs/predictions/test.json \
  --unseen outputs/predictions/unseen.json \
  --output outputs/submissions/prediction.json

python -m fish_vlm.cli validate-submission \
  --submission outputs/submissions/prediction.json \
  --config configs/base.yaml
```

Seen mode defaults to supervised-plus-text. Unseen mode restricts candidates to
unseen prototypes and defaults to fused text; supervised modes are rejected.
Generalised all-class inference exists in `configs/inference/generalised.yaml`
and is opt-in.

## DDP, SLURM, sweeps, and tests

All four GPUs train concurrently through one `torchrun` process group. Exact
nonlinear metrics gather predictions across ranks, and only rank zero writes
checkpoints/metrics or starts W&B.

```bash
python -m fish_vlm.cli slurm --config configs/slurm/genome.yaml --dry-run
python -m fish_vlm.cli slurm --config configs/slurm/genome.yaml
python -m fish_vlm.cli sweep \
  --config configs/sweeps/multimodal_pipeline.yaml --dry-run
pytest
```

SLURM dry-run only renders text and never invokes `sbatch`. Preparation creates a
NUL-delimited list from the three official split files and streams only those
raw images to node-local NVMe with one tar pipeline. It builds persistent
deterministic transform caches from that local copy. Training jobs do not copy
raw images; they copy only the exact model, prototype, teacher, and split-cache
files they consume. Large cache files are copied concurrently with
`cp --archive --reflink=auto`, and all reads then use node-local storage.

The focused joint-supervised-text sweep uses four sequential search phases:
loss weights, optimizer settings, projector/regularisation, and batch/training
duration. Search runs always use seed 42, select
`estimated_overall_accuracy`, validate every 250 optimizer steps, stop after six
non-improving evaluations, and otherwise run for 10,000 steps except when the
duration phase explicitly varies `max_steps`.

```bash
python scripts/run_joint_sweeps.py --phase loss --dry-run
python scripts/run_joint_sweeps.py --phase loss --submit --max-concurrent 8
python scripts/run_joint_sweeps.py --phase all --submit --max-concurrent 8
python scripts/run_joint_sweeps.py --confirm-top 8 --submit --max-concurrent 8
```

Each phase is one Slurm array with an optional `%N` concurrency limit. Phase
two inherits the best completed phase-one resolved configuration; later phases
follow the same rule. Consequently, `--phase all` submits or materialises the
next ready phase and stops until its metrics exist. Rerun the same command with
`--resume` to submit only incomplete runs from a previously submitted phase.

The deterministic reduced grids contain 30 loss runs, 15 optimizer runs, 20
architecture/consistency runs, and 18 batch/duration runs. The current
optimizer has independent absolute learning rates for the projector and final
DINO block, but not the supervised head, so the optional three-component
multiplier sweep remains disabled rather than changing optimizer behaviour for
the sweep.

Every run receives its own output directory containing `resolved_config.yaml`,
`metrics/best.json`, and `checkpoints/best.pt`. The shared
`run_index.json` records parameters, status, array task, job ID, score, best
step, and paths. W&B names contain every varied parameter, while
`sweep_metadata` records local/effective global batch sizes and the actual
component learning rates. After search, confirmation repeats the top eight
configurations at seeds 7, 42, and 123 and ranks complete triples by mean
estimated overall accuracy, mean harmonic mean, then worst-seed overall
accuracy.

## W&B result layout

Each training stage creates one clearly named W&B run in the
`multimodal-pipeline` group. W&B receives:

- total and component training losses;
- the selection score, selected seen accuracy, selected pseudo-unseen accuracy,
  harmonic mean, and estimated overall accuracy at validation steps;
- full per-branch accuracy, balanced accuracy, macro-F1, and top-5 every 500
  steps or when a new best checkpoint is found;
- learning rates, throughput, peak GPU GiB, and trainable parameter count;
- one `best/...` summary containing the final best scientific result.

Metric names use direct paths such as
`validation/pseudo_unseen/fused_text/accuracy` and
`score/estimated_overall_accuracy`. Resolved filesystem/SLURM noise is excluded
from W&B config. Checkpoints and model artifacts are never uploaded.

## Limitations

- Real pretrained execution requires network/cache access to the configured
  timm DINOv3 and Hugging Face BioCLIP artifacts.
- DINO architecture support for `final_block` requires a timm model exposing
  `blocks`, `stages`, or `layers`, plus `norm` or `fc_norm` when present.
- The initial implementation deliberately does not fine-tune BioCLIP itself,
  sample random prompt bags, or use official image-only labels.
- Multi-node DDP is outside scope; the launcher targets one node and a
  configurable GPU count.
