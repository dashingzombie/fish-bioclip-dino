PYTHON ?= python

.PHONY: install test all-data all-data-dry-run all-data-resume

install:
	$(PYTHON) -m pip install -e '.[dev]'

test:
	$(PYTHON) -m pytest

# Classifier-free all-image adaptation followed by DINO seen fine-tuning and
# hard-routed BioCLIP unseen inference. Every Slurm job requests one GPU; the
# independent DINO and BioCLIP branches are submitted concurrently.
all-data:
	PYTHONPATH=src $(PYTHON) scripts/run_all_data_pipeline.py --submit

all-data-dry-run:
	PYTHONPATH=src $(PYTHON) scripts/run_all_data_pipeline.py --dry-run

all-data-resume:
	PYTHONPATH=src $(PYTHON) scripts/run_all_data_pipeline.py --submit --resume
