# Shortcuts for what CI runs, so a local check matches the one that gates a push.
# `make help` lists them.
#
# Run `make install` first. These targets use whichever python is on PATH, and
# the wrong one fails obscurely -- pytest reports "unrecognized arguments:
# --cov" when pytest-cov is missing rather than saying so. Override it with
# `make test PYTHON=.venv/bin/python` if the right interpreter is not first.
PYTHON ?= python

# coverage traces every imported module, and importing langchain costs about
# fourteen seconds of that under the default C tracer. sys.monitoring does the
# same job for a fraction of it, and exists from Python 3.12 -- so pick the
# core the interpreter supports rather than hardcoding either.
COVERAGE_CORE := $(shell $(PYTHON) -c 'import sys; print("sysmon" if sys.version_info >= (3, 12) else "ctrace")')
export COVERAGE_CORE

.DEFAULT_GOAL := help
.PHONY: help install test fast lint fix hooks check run index dry-run image clean

help:  ## Show this list
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime and test dependencies
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

# --cov before --cov-fail-under, never before a path: see pyproject.toml.
test:  ## Run the whole suite with the coverage gate CI applies
	$(PYTHON) -m pytest --cov --cov-fail-under=95

fast:  ## Run the suite without the slow-marked files or coverage
	$(PYTHON) -m pytest -m "not slow"

lint:  ## Check formatting, imports, types and the non-Python files
	ruff check .
	$(PYTHON) -m mypy src app.py store_index.py
	node --check static/chat.js
	bash -n scripts/deploy_space.sh
	# Catches a syntax error in a module the suite never imports, as CI does.
	$(PYTHON) -m compileall -q app.py store_index.py template.py setup.py \
		gunicorn.conf.py src
	# The same Dockerfile lint CI runs; it is what found the shell-form CMD
	# that was swallowing SIGTERM.
	docker buildx build --check .

fix:  ## Apply the lint fixes that are safe to apply
	ruff check --fix .

hooks:  ## Run the pre-commit hooks, as CI does
	SKIP=ruff $(PYTHON) -m pre_commit run --all-files

check: lint hooks test  ## Everything CI checks, except the image build

run:  ## Serve the app on localhost:8080
	$(PYTHON) app.py

dry-run:  ## Report what an index rebuild would produce, touching nothing
	$(PYTHON) store_index.py --dry-run

index:  ## Rebuild the index from scratch, clearing duplicates
	$(PYTHON) store_index.py --recreate

image:  ## Build the container and run the checks CI runs against it
	docker build -t medibot:local .
	# Same budget as CI: losing the CPU-only torch wheel is silent otherwise.
	@bytes=$$(docker image inspect medibot:local --format '{{.Size}}'); \
		printf 'image is %s MB\n' "$$((bytes / 1024 / 1024))"; \
		test "$$bytes" -le $$((3 * 1024 * 1024 * 1024)) \
		|| { echo "image is over its 3GB budget -- is a CUDA torch back?"; exit 1; }
	# -W error, as CI does: these are the import paths the suite cannot reach.
	docker run --rm --entrypoint python medibot:local -W error -c \
		"from src.webapp import create_app; \
		 c = create_app(type('S', (), {'invoke': lambda s, p: {'answer': 'ok'}})()).test_client(); \
		 assert c.get('/healthz').status_code in (200, 503); print('image serves')"
	docker run --rm --entrypoint python medibot:local -W error -c \
		"from src.config import EMBED_DIM, EMBED_MODEL; \
		 from src.helper import download_hugging_face_embeddings as g; \
		 n = len(g().embed_query('hypertension')); \
		 print(f'{EMBED_MODEL} -> {n} dimensions, expected {EMBED_DIM}'); \
		 assert n == EMBED_DIM"
	docker run --rm --entrypoint python medibot:local -c \
		"import importlib.util as u; \
		 s = u.spec_from_file_location('g', 'gunicorn.conf.py'); \
		 m = u.module_from_spec(s); s.loader.exec_module(m); \
		 print('bind', m.bind, 'workers', m.workers); \
		 assert m.bind.endswith(':7860') and m.workers == 1"

clean:  ## Remove caches and coverage data
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
