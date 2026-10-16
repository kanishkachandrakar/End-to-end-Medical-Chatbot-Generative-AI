# Contributing

A learning project, so the bar is "does it work and can the next person tell
why" rather than anything ceremonial. What follows is mostly the shape of the
checks, because a few of them exist for reasons that are not obvious.

## Getting set up

```bash
make install        # runtime + test dependencies
make check          # lint, types and the suite, exactly as CI runs them
```

`make help` lists the rest. `make fast` skips the files that pay for heavyweight
imports and real git work.

The suite itself takes about five seconds; coverage used to add fifteen on top,
all of it spent tracing the langchain import tree. `make` sets `COVERAGE_CORE`
to `sysmon` where the interpreter has `sys.monitoring` (3.12 and later), which
removes almost all of that. If you run `pytest` directly and it feels slow,
that is why.

Optionally `pip install pre-commit && pre-commit install` runs the quick checks
before each commit. CI runs the same hooks, so the only thing you lose by not
installing them is finding out later.

`ruff` is a dev requirement and the hook runs it from the environment rather
than pinning its own copy — three places used to choose a version
independently and the hook's was seven releases behind, which let it pass code
CI rejected. A test fails if a pinned mirror comes back.

More generally: **nothing in CI installs a tool that isn't in
`requirements-dev.txt`.** Adding a step that needs a tool is the moment `pip
install` is at hand, and the result passes while recording nothing about the
version used — it happened with ruff, then with pre-commit three days later. A
test reads the `pip install` lines out of the workflow and the Makefile and
fails on any package not declared.

## What CI checks, and why each one is there

The fast job is lint, mypy and the suite on Python 3.10 and 3.12. The slow job
builds the container and then uses it, and that distinction matters:

- **It builds the image, boots it, and requests `/healthz` over HTTP.** This
  project has twice reached a state where the image built cleanly and the app
  would not start, both times because of a dependency resolution the unit suite
  could not see. "It builds" is not "it runs".
- **It embeds a word inside the container and checks the vector length.** The
  suite never loads the model, so nothing else would notice an embedding
  library changing its output size — which invalidates every vector in the
  Pinecone index at once.
- **It holds the image to a size budget.** Losing the CPU-only torch wheel is
  silent: it builds, it runs, it is four times larger.

## Things that are deliberate

Worth knowing before changing them:

- **`src/config.py` reads no secrets.** That is what lets it be imported in
  tests and CI without a `.env`, and a test asserts it.
- **`create_app()` takes the chain as an argument** rather than building it.
  That is why the routes can be tested at all; `app.py` builds the real one.
- **Every requirement has a lower bound.** Not style: without one, pip resolved
  `langchain-pinecone` backwards to a 2023 release and the image installed it
  without complaint.
- **Every dependency is imported or justified.** `requirements.txt` is checked
  in both directions: nothing may be imported undeclared, and nothing may be
  declared without either being imported or appearing in the `JUSTIFIED` set in
  `tests/test_declared_dependencies.py` with the reason. Four are there now —
  gunicorn, sentence-transformers, langchain and pypdf — each for a different
  reason worth knowing before removing it.
- **Warnings are errors.** A library announces a move with a warning before
  it makes one, and this project has been broken twice by a move it was not
  watching for. One narrow exception is allowed, for langchain-community's
  sunset notice, and the image job additionally runs the model construction
  under `-W error` because the suite never builds it.
- **`src/loaders.py` is the only module importing `langchain-community`,** and
  a test enforces that. The package is sunset; its embeddings half already
  moved to `langchain-huggingface`, and the PDF loaders stay there, isolated,
  until they have a standalone home. If that happens, delete the module and
  the test with it rather than leaving either passing vacuously.
- **`research/trials.ipynb` is not maintained.** It records how the pipeline was
  prototyped; `src/` is the code that runs.
- **The rate limit is global, not per visitor.** Behind a proxy every request
  shares one address, so a per-IP limit would be one bucket anyway.
- **`/livez` and `/healthz` are deliberately different.** Liveness knows nothing
  about the index or the chain; readiness reports both. A restart cannot
  repopulate an index, so anything that restarts on failure must watch the
  former. A test asserts `/livez` mentions neither.
- **`store_index.py` does everything that can refuse before it reads the PDF.**
  Connecting, the existence check, the confirmation and the dimension
  comparison all come first, so a misconfiguration costs seconds rather than
  the minutes it takes to parse and embed the book.

## Secrets

Both of this project's API keys are in its public git history. See
[SECURITY.md](SECURITY.md). A test fails on any tracked file containing
something key-shaped, so adding one is caught at commit time rather than later.
