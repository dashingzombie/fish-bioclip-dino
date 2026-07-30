# Fish DINOv3–BioCLIP

Production-oriented seen/unseen fish classification with projected DINOv3 and
native BioCLIP 2 branches. BioCLIP supports training-free zero-shot inference,
a seen-only linear probe, an adapter, one-block partial image-encoder tuning,
and explicit full image-encoder fine-tuning. Their calibrated probabilities can
be fused, while purpose-specific checkpoints are selected independently for
seen, unseen, and joint prediction.

The implementation never assumes official `test.pkl` or `unseen.pkl` labels.
Unseen classes enter training artifacts only as text: one supplied description,
one deterministic canonical prompt, and one frozen BioCLIP text prototype.

## Run everything with one command

After setting `data.root_dir` in `configs/base.yaml`, inspect the complete
SLURM workflow without submitting jobs:

```bash
make everything-dry-run
```

Submit preparation, the ordered baseline and ablation stages,
calibration/inference/submission, all four sweep phases, top-eight multi-seed
confirmation, and the final report:

```bash
make everything MAX_CONCURRENT=8
```

Resume the same run without resubmitting completed work:

```bash
make everything-resume MAX_CONCURRENT=8
```

The bootstrap workflow creates 16 jobs connected by `afterok`: preparation,
14 ordered training jobs, and finalisation. Preparation builds deterministic
caches and copies them back to persistent cache storage before the node-local
working directory disappears. Each training stage writes a deterministic
`submission.zip` below `outputs/submissions/stages/<stage>/`. A failed job
prevents dependent jobs and sweeps from using partial outputs.

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

Text-prototype caches use schema version 2. They store fixed tokenizer tokens
and text-encoder embeddings in addition to the prompt and checkpoint identity.
Inference reproduces both probes with the live BioCLIP 2 tokenizer and text
encoder. A legacy or changed cache is rejected and must be archived before
building a replacement.

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

The training-free BioCLIP 2 evaluation reports seen and pseudo-unseen results
for scientific name, taxonomic hierarchy, morphology-only,
morphology-plus-taxonomy, and full-description prompts:

```bash
make evaluate-zero-shot
```

Evaluate the three existing DINO checkpoints independently and write the
purpose-specific selection report:

```bash
make evaluate-stages
```

The comparison includes `best-stage1.pt`, `best-stage2.pt`, and `best.pt`;
records seen, pseudo-unseen, harmonic-mean, text-retrieval, genus, and family
accuracy; and chooses separate seen, unseen, and joint checkpoints. Inference
and calibration accept `--selection-report` and `--purpose`.

After all baselines finish, `make select-models` compares every model family,
including each training-free prompt condition, and writes
`outputs/metrics/model_selection.json`. Seen selection uses seen accuracy,
unseen selection uses pseudo-unseen accuracy, and joint selection uses their
harmonic mean; the final workflow consumes these three choices independently.

The remaining model baselines have direct targets:

```bash
make train-bioclip-linear
make train-bioclip-adapter
make train-bioclip-partial
make train-alignment-preserving
make train-bioclip-full
```

The linear classifier is used only for seen prediction. Its unseen path uses
the unmodified native BioCLIP embedding. Partial tuning initially unfreezes one
final visual block and uses supervised, image-text, and pretrained-embedding
distillation losses with separate backbone and head learning rates. Full
BioCLIP image-encoder tuning is explicit and remains the final ablation.
`make train-alignment-final-block` is intentionally excluded from the automatic
sequence; run it only after the frozen alignment-preserving result confirms
stable pseudo-unseen performance.

The exact implemented objective is a non-normalised weighted sum of enabled
terms:

- DINO prototype cross-entropy;
- cosine (`1 - cosine`) or optional symmetric batch-contrastive image-teacher
  alignment;
- ordinary supervised seen-head cross-entropy;
- native or adapter BioCLIP prototype cross-entropy;
- BioCLIP linear-classifier cross-entropy;
- pretrained BioCLIP embedding distillation;
- DINO representation distillation from `best-stage2.pt`;
- genus and family supervision;
- one controlled hard-negative strategy;
- optional symmetric KL or Jensen–Shannon branch consistency.

Defaults are `1.0 * DINO text CE + 0.25 * cosine teacher`. The
alignment-preserving joint configuration starts with species `0.5`, text
`1.0`, distillation `0.5`, genus `0.1`, and family `0.05`. Similarities,
normalisation, softmax/KL and log-sum-exp-sensitive work are float32 even under
AMP. BioCLIP stays frozen except in the explicit partial/full tuning stages.

Training is step-based and each stage controls its own maximum steps and
validation interval. The Genome profile targets one complete `gpu-h200` node:
four Hopper GPUs, 128 CPU cores, and 700 GB RAM. It uses BF16, a 512-image
microbatch per GPU, no gradient accumulation, static-graph DDP, gradient bucket
views, TF32, and fixed-shape cuDNN benchmarking. The global training batch is
2048 images per optimizer step.

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

Independent validation temperatures are fitted for DINO-text, BioCLIP-native,
and the applicable supervised head. Grid-selected convex weights combine
probabilities—not unrelated raw logits. DINO/BioCLIP fusion weights and the
seen-class penalty `calibration_gamma` are selected only on the configured
species-disjoint validation split. Calibration JSON contains its own content
hash.

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

Prompt ensembles and hard negatives are separate controlled ablations under
`configs/ablations/`. The workflow runs same-genus, same-family, text-similar,
and visually similar negatives one at a time, followed by equal multi-prompt
averaging and weighted morphology/taxonomy prototypes.

## Required unseen-inference gate

Before training or interpreting a new architecture, validate the existing
checkpoint against the exact organiser data, class partitions, BioCLIP 2 text
cache, and runtime model identities:

```bash
make verify-unseen-inference \
  CHECKPOINT=outputs/checkpoints/best.pt
```

The command writes `outputs/metrics/unseen_inference_audit.json`. It fails on a
stale candidate partition or ordering map, a tokenizer/text-encoder mismatch,
an incompatible checkpoint, missing or conflicting projector weights, score
column/label drift, or any constructed seen classifier in unseen-only mode. The
report includes the ordered unseen species, their ordering hash, and:

```text
random_accuracy = 1 / number_of_candidate_species
```

Similarity logits now always derive from:

```python
scores = normalise(image_embeddings) @ normalise(text_embeddings).T
```

## DDP, SLURM, sweeps, and tests

All four GPUs train concurrently through one `torchrun` process group. Exact
nonlinear metrics gather predictions across ranks, and only rank zero writes
checkpoints/metrics or starts W&B.

```bash
python -m fish_vlm.cli slurm --config configs/slurm/genome.yaml --dry-run
python -m fish_vlm.cli slurm --config configs/slurm/genome.yaml
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
make everything-dry-run
make everything MAX_CONCURRENT=8
make everything-resume MAX_CONCURRENT=8

# Individual/manual phase control remains available:
python scripts/run_joint_sweeps.py --phase loss --dry-run
python scripts/run_joint_sweeps.py --phase loss --submit --max-concurrent 8
python scripts/run_joint_sweeps.py --phase all --submit --max-concurrent 8
python scripts/run_joint_sweeps.py --confirm-top 8 --submit --max-concurrent 8
```

`make everything` is the single-command path. It first submits the full
preparation/training/calibration/inference/submission workflow. The loss array
depends on that workflow's finalisation job. Short `afterok` controller jobs
then rank each completed phase and submit optimizer, architecture, training,
top-eight confirmation, and the final report automatically.

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
accuracy. Confirmation varies the optimisation seed while fixing
`validation.pseudo_unseen.split_seed: 42`, so every run uses the compatible
stage-2 parent and the same species-disjoint benchmark.

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
- Full fine-tuning affects the BioCLIP image encoder only; the pretrained text
  encoder remains fixed so prototype identity stays auditable.
- Prompt ensembles are deterministic; random prompt bags and official
  image-only labels are not used.
- Multi-node DDP is outside scope; the launcher targets one node and a
  configurable GPU count.
