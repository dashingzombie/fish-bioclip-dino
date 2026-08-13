PYTHON ?= python
HYBRID_SPEC ?= configs/hybrid/sweep.yaml

.PHONY: install test hybrid hybrid-dry-run hybrid-resume all-data all-data-dry-run all-data-resume

install:
	$(PYTHON) -m pip install -e '.[dev]'

test:
	$(PYTHON) -m pytest

# The only production run point on this branch. It submits one resumable Slurm
# allocation; that allocation owns preparation, the sweep, final training,
# inference, strict validation, and all four ZIP files.
hybrid:
	$(PYTHON) scripts/run_hybrid_pipeline.py --spec $(HYBRID_SPEC) --submit

hybrid-dry-run:
	$(PYTHON) scripts/run_hybrid_pipeline.py --spec $(HYBRID_SPEC) --dry-run

hybrid-resume:
	$(PYTHON) scripts/run_hybrid_pipeline.py --spec $(HYBRID_SPEC) --submit --resume

# Classifier-free all-image adaptation followed by DINO seen fine-tuning and
# hard-routed BioCLIP unseen inference. Every Slurm job requests one GPU; the
# independent DINO and BioCLIP branches are submitted concurrently.
all-data:
	PYTHONPATH=src $(PYTHON) scripts/run_all_data_pipeline.py --submit

all-data-dry-run:
	PYTHONPATH=src $(PYTHON) scripts/run_all_data_pipeline.py --dry-run

all-data-resume:
	PYTHONPATH=src $(PYTHON) scripts/run_all_data_pipeline.py --submit --resume
