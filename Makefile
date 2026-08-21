PYTHON ?= .venv/bin/python
PYTHON_BOOTSTRAP ?= $(shell command -v python3.12 2>/dev/null || command -v python3.11 2>/dev/null || command -v python3.10 2>/dev/null || command -v python3)
PIP := $(PYTHON) -m pip

.PHONY: setup setup-model setup-evaluation setup-separation download-models verify-models download-evaluation evaluate usability-report product-validation-report test lint run benchmark autopilot-observe autopilot-safe-fix

setup:
	$(PYTHON_BOOTSTRAP) -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

setup-model:
	$(PIP) install -e ".[model]"

setup-evaluation:
	$(PIP) install -e ".[evaluation]"

setup-separation:
	$(PIP) install -e ".[separation]"

download-models:
	$(PYTHON) scripts/download_models.py

verify-models:
	$(PYTHON) scripts/verify_models.py

download-evaluation:
	$(PYTHON) scripts/download_evaluation_subset.py

evaluate:
	$(PYTHON) scripts/run_evaluation.py

usability-report:
	$(PYTHON) scripts/generate_usability_report.py

product-validation-report:
	$(PYTHON) scripts/generate_product_validation_report.py

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m mypy app

run:
	$(PYTHON) -m uvicorn app.main:app --host 127.0.0.1 --port 8000

benchmark:
	$(PYTHON) scripts/benchmark_local.py $(ARGS)

autopilot-observe:
	$(PYTHON) scripts/autopilot.py observe

autopilot-safe-fix:
	$(PYTHON) scripts/autopilot.py safe-fix
