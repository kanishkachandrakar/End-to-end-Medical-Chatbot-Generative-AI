# Shortcuts for what CI runs, so a local check matches the one that gates a push.
# `make help` lists them.
#
# Run `make install` first. These targets use whichever python is on PATH, and
# the wrong one fails obscurely -- pytest reports "unrecognized arguments:
# --cov" when pytest-cov is missing rather than saying so. Override it with
# `make test PYTHON=.venv/bin/python` if the right interpreter is not first.
PYTHON ?= python

.DEFAULT_GOAL := help
.PHONY: help install test fast lint fix check run index dry-run image clean

help:  ## Show this list
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime and test dependencies
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt ruff

test:  ## Run the whole suite with the coverage gate CI applies
	$(PYTHON) -m pytest --cov-fail-under=95

fast:  ## Run the suite without the ten-second langchain_community import
	$(PYTHON) -m pytest -m "not slow" --no-cov

lint:  ## Check formatting, imports, types and the non-Python files
	ruff check .
	$(PYTHON) -m mypy src app.py store_index.py
	node --check static/chat.js
	bash -n scripts/deploy_space.sh

fix:  ## Apply the lint fixes that are safe to apply
	ruff check --fix .

check: lint test  ## Everything CI checks, except the image build

run:  ## Serve the app on localhost:8080
	$(PYTHON) app.py

dry-run:  ## Report what an index rebuild would produce, touching nothing
	$(PYTHON) store_index.py --dry-run

index:  ## Rebuild the index from scratch, clearing duplicates
	$(PYTHON) store_index.py --recreate

image:  ## Build the container and check it serves a request
	docker build -t medibot:local .
	docker run --rm --entrypoint python medibot:local -c \
		"from src.webapp import create_app; \
		 c = create_app(type('S', (), {'invoke': lambda s, p: {'answer': 'ok'}})()).test_client(); \
		 assert c.get('/healthz').status_code in (200, 503); print('image serves')"

clean:  ## Remove caches and coverage data
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
