PYTHON ?= python
HYBRID_SPEC ?= configs/hybrid/sweep.yaml

.PHONY: install test hybrid hybrid-dry-run hybrid-resume

install:
	$(PYTHON) -m pip install -e '.[dev]'

test:
	$(PYTHON) -m pytest

# The only production run point on this branch. It submits one resumable Slurm
# allocation; that allocation owns preparation, the sweep, final training,
# inference, strict validation, and both ZIP files.
hybrid:
	$(PYTHON) scripts/run_hybrid_pipeline.py --spec $(HYBRID_SPEC) --submit

hybrid-dry-run:
	$(PYTHON) scripts/run_hybrid_pipeline.py --spec $(HYBRID_SPEC) --dry-run

hybrid-resume:
	$(PYTHON) scripts/run_hybrid_pipeline.py --spec $(HYBRID_SPEC) --submit --resume
