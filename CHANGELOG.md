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
- mypy runs over `src/`, `app.py` and `store_index.py`.
- `/healthz` reports one of three states; see [DEPLOY.md](DEPLOY.md).

### Known limitations

See the section of the same name in [README.md](README.md). The short version:
one 1999 reference book as the only source, no conversation memory, and nothing
enforcing the "not medical advice" framing beyond the prompt asking for it.
