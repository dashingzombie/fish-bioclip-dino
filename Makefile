PYTHON ?= python
CONFIG ?= configs/base.yaml
CHECKPOINT ?= outputs/checkpoints/best.pt
GPUS ?= 4
MAX_CONCURRENT ?= 8
PIPELINE_CONFIG ?= configs/pipeline.yaml
SWEEP_OUTPUT ?= outputs/sweep_pipelines/joint_supervised_text

.PHONY: install test prepare-prompts build-text-prototypes build-image-cache build-teacher-cache \
	pseudo-unseen train evaluate-zero-shot evaluate evaluate-stages calibrate infer-test infer-unseen \
	select-models \
	train-bioclip-linear train-bioclip-adapter train-bioclip-partial train-alignment-preserving \
	train-alignment-final-block train-bioclip-full \
	verify-unseen-inference submission validate-submission slurm-dry-run slurm-submit \
	everything everything-dry-run everything-resume

install:
	$(PYTHON) -m pip install -e '.[dev]'

test:
	$(PYTHON) -m pytest

prepare-prompts:
	$(PYTHON) -m fish_vlm.cli prepare-prompts --config $(CONFIG)

build-text-prototypes:
	$(PYTHON) -m fish_vlm.cli build-text-prototypes --config $(CONFIG)

build-image-cache:
	$(PYTHON) -m fish_vlm.cli build-image-cache --config $(CONFIG)

build-teacher-cache:
	$(PYTHON) -m fish_vlm.cli build-teacher-cache --config $(CONFIG)

pseudo-unseen:
	$(PYTHON) -m fish_vlm.cli make-pseudo-unseen --config $(CONFIG)

train:
	torchrun --standalone --nproc_per_node=$(GPUS) -m fish_vlm.cli train --config $(CONFIG)

train-bioclip-linear:
	$(MAKE) train CONFIG=configs/train/bioclip_linear_probe.yaml

train-bioclip-adapter:
	$(MAKE) train CONFIG=configs/train/bioclip_adapter.yaml

train-bioclip-partial:
	$(MAKE) train CONFIG=configs/train/bioclip_partial_finetune.yaml

train-alignment-preserving:
	$(MAKE) train CONFIG=configs/train/joint_alignment_preserving.yaml

train-alignment-final-block:
	$(MAKE) train CONFIG=configs/train/joint_alignment_final_block.yaml

train-bioclip-full:
	$(MAKE) train CONFIG=configs/train/bioclip_full_finetune.yaml

evaluate-zero-shot:
	$(PYTHON) -m fish_vlm.cli evaluate --config $(CONFIG)

evaluate:
	$(PYTHON) -m fish_vlm.cli evaluate --config $(CONFIG) --checkpoint $(CHECKPOINT)

evaluate-stages:
	$(PYTHON) -m fish_vlm.cli evaluate-stages \
		--config $(PIPELINE_CONFIG) \
		--output outputs/metrics/stage_comparison.json

select-models:
	$(PYTHON) -m fish_vlm.cli select-models \
		--config $(PIPELINE_CONFIG) \
		--output outputs/metrics/model_selection.json

calibrate:
	$(PYTHON) -m fish_vlm.cli calibrate --config $(CONFIG) --checkpoint $(CHECKPOINT)

infer-test:
	$(PYTHON) -m fish_vlm.cli infer --config configs/inference/seen.yaml \
		--checkpoint $(CHECKPOINT) --output outputs/predictions/test.json

infer-unseen:
	$(PYTHON) -m fish_vlm.cli infer --config configs/inference/unseen.yaml \
		--checkpoint $(CHECKPOINT) --output outputs/predictions/unseen.json

verify-unseen-inference:
	$(PYTHON) -m fish_vlm.cli verify-unseen-inference \
		--config configs/inference/unseen.yaml --checkpoint $(CHECKPOINT)

submission:
	$(PYTHON) -m fish_vlm.cli merge-submission \
		--test outputs/predictions/test.json \
		--unseen outputs/predictions/unseen.json \
		--output outputs/submissions/prediction.json
	$(PYTHON) -m fish_vlm.cli package-submission \
		--submission outputs/submissions/prediction.json \
		--output outputs/submissions/submission.zip

validate-submission:
	$(PYTHON) -m fish_vlm.cli validate-submission \
		--submission outputs/submissions/prediction.json --config $(CONFIG)

slurm-dry-run:
	$(PYTHON) -m fish_vlm.cli slurm --config configs/slurm/genome.yaml --dry-run

slurm-submit:
	$(PYTHON) -m fish_vlm.cli slurm --config configs/slurm/genome.yaml

everything:
	$(PYTHON) scripts/run_joint_sweeps.py --everything --submit \
		--max-concurrent $(MAX_CONCURRENT) \
		--pipeline-config $(PIPELINE_CONFIG) \
		--output-root $(SWEEP_OUTPUT)

everything-dry-run:
	$(PYTHON) scripts/run_joint_sweeps.py --everything --dry-run \
		--max-concurrent $(MAX_CONCURRENT) \
		--pipeline-config $(PIPELINE_CONFIG) \
		--output-root $(SWEEP_OUTPUT)

everything-resume:
	$(PYTHON) scripts/run_joint_sweeps.py --everything --submit --resume \
		--max-concurrent $(MAX_CONCURRENT) \
		--pipeline-config $(PIPELINE_CONFIG) \
		--output-root $(SWEEP_OUTPUT)
