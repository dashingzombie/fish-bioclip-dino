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
  --gpus 2
```

Inspect either execution plan without running commands or calling `sbatch`:

```bash
python scripts/run_all.py --config configs/pipeline.yaml --mode local --gpus 1 --dry-run
python scripts/run_all.py --config configs/pipeline.yaml --mode slurm --gpus 2 --dry-run
```

The workflow performs prompt preparation, all three prototype-alignment/joint
stages, the separate BioCLIP-adapter experiment, final evaluation, calibration,
seen and unseen inference, submission merge, strict validation, and summary
generation. Local state is saved in
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
├── descriptions_all.json
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
python -m fish_vlm.cli build-teacher-cache --config configs/base.yaml
python -m fish_vlm.cli make-pseudo-unseen --config configs/base.yaml
```

The cache builder downloads BioCLIP only when the real preparation command is
run. Unit tests substitute tiny deterministic encoders and perform no network
access. The teacher cache reads only filenames from `train.pkl`, requires labels
for every included image, uses the deterministic BioCLIP evaluation transform,
and stores float16 embeddings converted to float32 for losses.

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
  "hf-hub:imageomics/bioclip")` supplies both the frozen encoder and its own
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
torchrun --standalone --nproc_per_node=2 -m fish_vlm.cli train \
  --config configs/train/projection_only.yaml

torchrun --standalone --nproc_per_node=2 -m fish_vlm.cli train \
  --config configs/train/final_block.yaml

torchrun --standalone --nproc_per_node=2 -m fish_vlm.cli train \
  --config configs/train/joint_supervised_text.yaml

torchrun --standalone --nproc_per_node=2 -m fish_vlm.cli train \
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

## Pseudo-unseen validation

`species_holdout` removes complete species from every training loader.
`genus_holdout` removes complete genera. Seeds 7, 42 and 123 are generated.
Checkpoints store both the exact training-species hash and pseudo-split hash;
loaders and checkpoint loading reject leakage or incompatible identities.

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

Distributed samplers call `set_epoch`, exact nonlinear metrics gather predictions
across ranks, and only rank zero writes checkpoints/metrics or starts W&B.

```bash
python -m fish_vlm.cli slurm --config configs/slurm/ghpc.yaml --dry-run
python -m fish_vlm.cli slurm --config configs/slurm/ghpc.yaml
python -m fish_vlm.cli sweep \
  --config configs/sweeps/multimodal_pipeline.yaml --dry-run
pytest
```

SLURM dry-run only renders text and never invokes `sbatch`. The sweep is phased
across baseline, projector, objective, learning rate, teacher weight, DINO
adaptation, supervised head, BioCLIP adapter, inference branch, calibration and
pseudo-unseen seed. It does not expose official labels.

## W&B result layout

Each training stage creates one clearly named W&B run in the
`multimodal-pipeline` group. W&B receives:

- total and component training losses;
- the selection score, selected seen accuracy, selected pseudo-unseen accuracy,
  harmonic mean, and estimated overall accuracy every epoch;
- full per-branch accuracy, balanced accuracy, macro-F1, and top-5 only on the
  first epoch, every fifth epoch, or when a new best checkpoint is found;
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
- Multi-node DDP is outside scope; the launcher targets one node and two GPUs.
