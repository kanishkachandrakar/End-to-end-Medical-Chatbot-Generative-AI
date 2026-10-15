# Changelog

Grouped by theme rather than by release — there are no releases yet. The
entries worth knowing about are the behavioural ones; the rest is in the git
log.

## Unreleased

### Fixed — things that were broken

- **The indexer never uploaded anything.** `store_index.py` created the Pinecone
  index and exited; the `from_documents` call that embeds and upserts the chunks
  only existed in `research/trials.ipynb`. Anyone following the README ended up
  with an empty index and a bot answering from the model's own knowledge.
- **Re-indexing duplicated the whole book.** Vector ids were fresh UUIDs on every
  run, so the live index accumulated six copies of all 5,860 chunks. Because
  duplicates embed to identical vectors, a top-3 search returned the same chunk
  three times — the model got a third of the context it was configured for. Ids
  are now a hash of the chunk text, making a re-run an overwrite.
- **The Docker image did not build.** `--index-url` replaces PyPI rather than
  adding to it, and the PyTorch CPU index does not serve every pure-Python
  dependency of torch. Found by the CI build job on its first run, after eight
  days of the Dockerfile linting clean.
- **`sentence-transformers` was pinned to 2.2.2**, which imports a function
  `huggingface_hub` removed in 0.26 — so any clean install failed at startup.
- **Application logs were discarded.** Flask leaves `app.logger` at the root
  logger's `WARNING` outside debug mode, so every `logger.info` call was dropped
  under gunicorn, including the index vector count at startup.
- **Hardcoded API keys** in `app.py`, `store_index.py` and the notebook, replaced
  by `.env`. The keys that were committed remain in this repository's history and
  should be treated as public.
- **A broken avatar path**, **quirks-mode rendering** from tags above the
  doctype, **jQuery loaded three times**, and **HTML injection** in the chat log,
  which built bubbles by string concatenation and passed them through
  `$.parseHTML`.
- **The app could not start at all on a clean install**, twice over, and both
  times the image built successfully first. `langchain` 1.0 moved
  `create_retrieval_chain` into a separate `langchain-classic` package, and
  separately pip resolved `langchain-pinecone` *backwards* to 0.0.1 — a 2023
  release with no `PineconeVectorStore` — because `pinecone[grpc]` constrained
  the `pinecone` version and backtracking was easier than failing. Neither was
  reproducible from an existing virtualenv. `requirements.txt` now carries
  bounds in both directions, with the reason beside each.
- **`langchain-core` and `pydantic` were imported but never declared**, working
  only because something else pulled them in.
- **The embeddings were coming from a sunset package.** `langchain-community`
  announces on import that it is no longer actively maintained, and had
  deprecated `HuggingFaceEmbeddings` in favour of the standalone
  `langchain-huggingface`. This is the third instance of the pattern that broke
  this project twice — a langchain package relocating something — and the first
  one caught before it bit. The loaders, which have no standalone home yet, are
  confined to `src/loaders.py` by a test.
- **Four tests held a file open.** Flask's test client does not close a static
  response for you, so each run leaked handles and reported four
  ResourceWarnings — which also blocked treating warnings as errors.
- **The notebook still imported the module langchain 1.0 removed.** The same
  breakage as `app.py`, fixed there two days earlier and missed here — the
  notebook is marked as a prototype, which is a reason not to rewrite it and
  not a reason to leave an import in it that cannot resolve. Its imports are
  now resolved by the test suite too.
- **Static asset URLs were keyed on modification time.** An mtime belongs to
  the checkout, not the file: every deploy rewrites them, so each release
  invalidated assets whose bytes had not changed, and two replicas built
  separately would advertise different URLs for identical content. The key is
  now a content hash.
- **`pre-commit` was installed ad hoc in CI** — the exact pattern removed for
  ruff three days earlier, reintroduced by the commit that put the hooks into
  CI. Both are declared now, and a test fails on the next one.
- **The base image pin had nothing watching it.** The Dockerfile was pinned by
  digest and both the commit and SECURITY.md said dependabot proposes the
  updates — but only the pip and github-actions ecosystems were configured, so
  nothing watched the Dockerfile. That made the pin worse than the moving tag
  it replaced: a tag at least collects security patches, a digest nobody
  updates collects nothing.
- **Three independent ruff versions.** The pre-commit hook pinned a mirror
  seven minor releases behind the one CI and the Makefile used, so the commit
  hook could pass code CI then rejected — the single failure a commit hook
  exists to prevent. ruff is now a declared dev requirement that all three use.
- **A degraded index marked the container unhealthy.** `/healthz` was changed
  to answer 503 when the index is empty or the chain could not be built, and the
  container's `HEALTHCHECK` still pointed at it — so an app serving perfectly
  well with nothing to retrieve from was reported unhealthy, and anything
  watching container health would restart it. Repeatedly, since a restart
  cannot repopulate an index. Liveness and readiness are now separate: `/livez`
  for "is it up", `/healthz` for "can it answer".
- **The indexer did its checks after the slow work.** The dimension comparison
  claimed to run up front and ran after the 15MB PDF had been parsed and the
  embedding model loaded. Everything that can refuse now happens before the PDF
  is opened.
- **The degraded banner cried wolf.** Any failed `/healthz` — including one
  dropped request — told every visitor the service was down. An unexplained
  failure is now confirmed against `/livez` first.
- **Startup could hang indefinitely.** Reading the index size happens before the
  port is bound and the Pinecone client has no timeout of its own, so a network
  that accepted the connection and then went quiet would stall the boot — which
  from outside is a crash loop that never recovers. It is now abandoned after
  fifteen seconds and `/healthz` reports `vectors=unknown`.
- **A crash loop when a dependency was down at startup.** If Pinecone or
  HuggingFace was unreachable as the worker booted, building the retrieval
  chain raised, the worker died, gunicorn restarted it and it raised again —
  a Space that never came up, with `/healthz` unreachable so nothing could say
  why. It now starts anyway and reports `degraded … chain-unavailable`.

### Changed — behaviour

- `POST /get` is POST-only. Answering costs a Groq call and spends the shared
  rate limit, so it should not be reachable by a crawler or a link prefetch.
- The demo answers at most `RATE_LIMIT_PER_MINUTE` questions a minute across all
  visitors. The limit is global, not per visitor: behind the Spaces proxy every
  request shares one address, and a forwarded header would be trivially spoofed.
- Questions longer than `MAX_QUESTION_CHARS` are rejected before the model is
  called, and the input box mirrors the limit.
- Failures are told apart: a Groq rate limit returns 429 and says to wait, a
  timeout 504, anything else 502. Each reply quotes a request id that appears in
  the log.
- The system prompt no longer tells the model to "google the terms" when it does
  not know — it has no web access, so that only licensed invention.
- Questions are no longer written to the log. On a public URL they are strangers'
  health questions, and a Space's log is retained.
- Debug mode is off unless `FLASK_DEBUG` is set. It used to be hardcoded on,
  alongside `host="0.0.0.0"`, which exposes the Werkzeug console to the network.

### Added

- Deployment: a `Dockerfile` serving the app through gunicorn, a Hugging Face
  Space configuration, `scripts/deploy_space.sh`, and `DEPLOY.md`.
- `GET /healthz`, which reports the revision the running image was built from,
  and `GET /robots.txt`.
- Security headers including a CSP, with tests that hold the policy to the
  origins the page actually loads from — in both directions.
- Errors answer in plain text rather than Flask's HTML pages, which the chat log
  would otherwise paste verbatim into a bubble.
- Each request gets an id, returned in `X-Request-Id`, written to the log beside
  how long the answer took, and quoted in failure replies so it can be reported.
- `python store_index.py --dry-run` reports what a rebuild would produce without
  touching Pinecone; `--limit N` runs the whole pipeline over the first N chunks
  as a cheap end-to-end check.
- A test suite covering everything in `src/`, with lint, coverage, a real image
  build, and the JavaScript and shell script checked, on Python 3.10 and 3.12.
- [SECURITY.md](SECURITY.md), recording the exposed keys and what a public
  deployment of this does and does not defend against.
- `python store_index.py --recreate` deletes the index before rebuilding, which
  is the only way to clear vectors written under ids the script no longer
  generates — the notebook's duplicates.
- `GET /healthz` reports the vector count and answers 503 `index-empty` when
  there is nothing to retrieve from, so the failure that looks like success is
  visible to a monitor.
- A `Makefile` wrapping the checks CI runs, and a pre-commit configuration for
  the fast half of them.
- `store_index.py` reads the index back after upserting and waits for the count
  to settle, so its last word is what arrived rather than what it sent.
- A warning at startup when the two API keys look swapped — both are set and
  non-empty in that case, so nothing else notices until the first request.
- `GET /favicon.ico`, and `no-store` on the page that names the year-cached
  asset URLs.
- The chat page asks `/healthz` on load and shows a banner when answers would
  not be grounded, rather than letting the first question come back as an error.
- The resolved configuration is logged in one line at startup; ten of those
  settings are environment-overridable and a wrong one was invisible.
- `Retry-After` on a rate-limited reply, computed from the bucket rather than
  guessed, plus `X-RateLimit-Limit` and `X-RateLimit-Remaining` on every answer
  so a client can pace itself instead of discovering the limit.
- `GET /livez`.
- `Retry-After` on a refusal while degraded, deliberately longer than the
  rate-limit wait: that one clears by itself and this one does not.
- Progress while rebuilding the index: a rate and an estimate per batch, with
  the PDF load and the model load timed separately.
- [CONTRIBUTING.md](CONTRIBUTING.md), and a supply-chain section in
  [SECURITY.md](SECURITY.md) that says what is *not* in place as well as what
  is.
- `--yes` skips the confirmation `--recreate` now asks for before deleting an
  index, and `store_index.py` refuses up front if the existing index's
  dimension does not match the embedding model.

### Changed — infrastructure

- CI builds the image and then uses it: the app is served out of it and asked
  for a page, the embedding dimension is checked against `EMBED_DIM`, and the
  image size is held to a budget. The dimension check exists because the unit
  suite never loads the model, so nothing else would notice an embedding
  library upgrade invalidating every vector in the index.
- The suite runs on Python 3.10 and 3.12, and the JavaScript and shell script
  are checked.
- Static assets are fingerprinted by modification time and cached for a year.
- Request bodies above `MAX_CONTENT_BYTES` are refused by Flask before being
  parsed, rather than after.
- Gunicorn calls `app:create()` as an application factory, so importing
  `app.py` runs nothing — which is what finally brought it under test.
- CI boots the container and requests `/healthz` over HTTP, which is the only
  check that exercises gunicorn, its config and the factory together. It needs
  no credentials because the app degrades rather than refusing to start.
- mypy runs over `src/`, `app.py` and `store_index.py`, with the real langchain
  packages installed in CI rather than stubbed — which is what surfaced two
  wrong `ChatGroq` argument names and a loose `Sequence` annotation.
- Tests resolve every name `app.py` imports against the installed packages, so
  the next time a library moves something it fails in the unit suite rather
  than three minutes into the image job.
- The base image is pinned by digest, and every requirement has a lower bound,
  enforced by a test — a bare package name is what let pip resolve backwards.
- The deploy script is executed in the suite against a local repository
  standing in for the Space. It had never been run by anything.
- A test fails on any tracked file containing a key-shaped string.
- Gunicorn's worker class is named rather than inferred from the thread count.
- A test derives the set of states `/healthz` can report and requires the chat
  page and DEPLOY.md to cover every one, in both directions.
- The GitHub Actions are current, the dependency floors are raised to what is
  actually installed, and the five dependabot pull requests those bumps came
  from are closed.
- Warnings are errors in the suite, with one narrow exception for
  langchain-community's sunset notice, and both in-image python steps run under
  `-W error` — they construct the embedding model and the Flask app, which the
  suite never does, so that is where a deprecation would otherwise be printed
  into a build log and scrolled past.
- The asset cache-busting moved to `src/assets.py`, where the year-long cache
  lifetime and the fingerprinting that makes it safe are one call rather than
  two separable settings.
- CI runs with a read-only token and a timeout on every job; the defaults were
  write access and six hours.
- `make check` covers what CI covers, asserted by a test that compares all
  seven commands rather than the three it used to.
- `prepare_index()` and `confirm_deletion()` are out of `store_index.main()`
  and tested directly, so a decision made before the PDF is opened no longer
  needs a PDF to exercise.
- Coverage uses `sys.monitoring` where the interpreter has it. It had been
  spending about fifteen seconds tracing the langchain import tree and
  discarding the result — a fixed cost, so it fell entirely on single-file
  runs. `make fast` is now about a second; the full suite with the gate is
  five.
- CI runs the pre-commit hooks, so the whitespace, large-file and
  private-key checks apply to everyone rather than whoever opted in.
- The container boot check also requires `/livez` to answer 200 while
  `/healthz` reports 503 — the asymmetry those two exist for — and reads the
  container's own health status, which is the only thing that exercises the
  `HEALTHCHECK` command.
- `.dockerignore` is covered by a test in both directions: nothing the app
  needs may be excluded, nothing it cannot use may be included.
- An `.editorconfig`, checked against the tree rather than asserted: three
  indentation conventions here are deliberate and one of them is load-bearing
  for the template contract tests.
- `src/config.py` cross-checks the settings whose meaning is relative to each
  other, and `--recreate` asks for confirmation before deleting an index.
- `/healthz` reports one of three states; see [DEPLOY.md](DEPLOY.md).

### Known limitations

See the section of the same name in [README.md](README.md). The short version:
one 1999 reference book as the only source, no conversation memory, and nothing
enforcing the "not medical advice" framing beyond the prompt asking for it.
