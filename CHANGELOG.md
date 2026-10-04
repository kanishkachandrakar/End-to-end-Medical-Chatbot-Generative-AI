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
- `GET /healthz`, `GET /robots.txt`, security headers including a CSP, and
  `python store_index.py --dry-run`.
- A test suite covering everything in `src/`, with lint, coverage and a real
  image build in CI.

### Known limitations

See the section of the same name in [README.md](README.md). The short version:
one 1999 reference book as the only source, no conversation memory, and nothing
enforcing the "not medical advice" framing beyond the prompt asking for it.
