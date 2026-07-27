PYTHON ?= python
CONFIG ?= configs/base.yaml
CHECKPOINT ?= outputs/checkpoints/best.pt
GPUS ?= 1

.PHONY: install test prepare-prompts build-text-prototypes build-teacher-cache \
	pseudo-unseen train evaluate-zero-shot evaluate calibrate infer-test infer-unseen \
	submission validate-submission slurm-dry-run slurm-submit sweep-dry-run \
	run-all-local run-all-local-dry-run run-all-slurm run-all-slurm-dry-run

install:
	$(PYTHON) -m pip install -e '.[dev]'

test:
	$(PYTHON) -m pytest

prepare-prompts:
	$(PYTHON) -m fish_vlm.cli prepare-prompts --config $(CONFIG)

build-text-prototypes:
	$(PYTHON) -m fish_vlm.cli build-text-prototypes --config $(CONFIG)

build-teacher-cache:
	$(PYTHON) -m fish_vlm.cli build-teacher-cache --config $(CONFIG)

pseudo-unseen:
	$(PYTHON) -m fish_vlm.cli make-pseudo-unseen --config $(CONFIG)

train:
	torchrun --standalone --nproc_per_node=$(GPUS) -m fish_vlm.cli train --config $(CONFIG)

evaluate-zero-shot:
	$(PYTHON) -m fish_vlm.cli evaluate --config $(CONFIG)

evaluate:
	$(PYTHON) -m fish_vlm.cli evaluate --config $(CONFIG) --checkpoint $(CHECKPOINT)

calibrate:
	$(PYTHON) -m fish_vlm.cli calibrate --config $(CONFIG) --checkpoint $(CHECKPOINT)

infer-test:
	$(PYTHON) -m fish_vlm.cli infer --config configs/inference/seen.yaml \
		--checkpoint $(CHECKPOINT) --output outputs/predictions/test.json

infer-unseen:
	$(PYTHON) -m fish_vlm.cli infer --config configs/inference/unseen.yaml \
		--checkpoint $(CHECKPOINT) --output outputs/predictions/unseen.json

submission:
	$(PYTHON) -m fish_vlm.cli merge-submission \
		--test outputs/predictions/test.json \
		--unseen outputs/predictions/unseen.json \
		--output outputs/submissions/prediction.json

validate-submission:
	$(PYTHON) -m fish_vlm.cli validate-submission \
		--submission outputs/submissions/prediction.json --config $(CONFIG)

slurm-dry-run:
	$(PYTHON) -m fish_vlm.cli slurm --config configs/slurm/genome.yaml --dry-run

slurm-submit:
	$(PYTHON) -m fish_vlm.cli slurm --config configs/slurm/genome.yaml

sweep-dry-run:
	$(PYTHON) -m fish_vlm.cli sweep --config configs/sweeps/multimodal_pipeline.yaml --dry-run

run-all-local:
	$(PYTHON) scripts/run_all.py --config configs/pipeline.yaml --mode local --gpus $(GPUS)

run-all-local-dry-run:
	$(PYTHON) scripts/run_all.py --config configs/pipeline.yaml --mode local --gpus $(GPUS) --dry-run

run-all-slurm:
	$(PYTHON) scripts/run_all.py --config configs/pipeline.yaml --mode slurm --gpus $(GPUS)

run-all-slurm-dry-run:
	$(PYTHON) scripts/run_all.py --config configs/pipeline.yaml --mode slurm --gpus $(GPUS) --dry-run
