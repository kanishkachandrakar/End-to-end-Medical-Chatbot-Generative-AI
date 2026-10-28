# Changelog

Grouped by theme rather than exhaustively. The entries worth knowing about are
the behavioural ones; the rest is in the git log.

## Unreleased

### Fixed

- **A truncated reply published the model's chain of thought.**
  `strip_reasoning` matched `<think>…</think>` and nothing else, so an
  unterminated block went to the visitor intact, tag and all — and that is the
  shape a cut-short reply has, since deepseek-r1 reasons at length and meeting
  the token ceiling or the 60-second Groq timeout leaves the opening tag emitted
  and the closing one never written. A test asserted that behaviour on the
  grounds that "a lone opening tag must not swallow the answer"; the tag is
  unclosed *because* generation stopped, so there is no answer after it to
  protect. Both unbalanced cases are stripped now, which leaves an empty string
  that the route already answers with NO_ANSWER.
- **The development server offered your Groq budget to the local network.**
  `python app.py` bound `0.0.0.0` while holding the real keys, and `POST /get`
  takes no credentials — so anyone on a café or office network could query the
  index and spend the quota. The Makefile said "Serve the app on
  localhost:8080"; the README said `0.0.0.0` and called it reachable from other
  machines. Loopback now, through a `HOST` setting; the container is untouched,
  because inside one `0.0.0.0` is the only useful thing to bind.
- **The revision and config lines never reached a Space's log.** A gunicorn
  worker leaves the root logger at WARNING with no handlers, and a module logger
  has no level of its own, so both `logger.info` calls in `create()` were
  dropped — the two lines whose entire purpose is making a misconfigured deploy
  visible. `src/webapp.py` fixes exactly this for `app.logger` and says why;
  nobody connected the two. The test that covered it passed throughout, because
  `caplog.at_level(logging.INFO)` supplies the one condition production lacks.
- **The index size failed silently and reported only that it had.** Every
  exception became `None` with the reason discarded, so a slow control plane, a
  revoked key, a renamed index and a dead region all produced one line: "could
  not read stats". The chain is built separately and lazily, so it can succeed
  while this fails, leaving that line as the only sign.
- **Two packages were imported by name and never declared** — werkzeug, for
  `HTTPException`, and langchain-text-splitters, declared only as a dev
  dependency while `store_index.py` imports it. `requirements.txt` states the
  policy on this twice in its own comments. The test file that should have
  caught them opened by claiming "the other direction is already checked:
  nothing may be imported without being declared", and no such check existed
  anywhere.
- **`pip install -e .` installed a package called `tests`.** A bare
  `find_packages()` takes every directory with an `__init__.py`, and tests/ has
  one so the suite can import its stubs, so `top_level.txt` declared both `src`
  and `tests` — a package with about the most collidable name available. The
  image was unaffected only because `.dockerignore` excludes tests/.
- **`--limit 0` was rejected after the whole PDF had been parsed.** The flag
  exists to make a cheap end-to-end check possible before a full rebuild, and
  validating it cost the expensive half of that rebuild. argparse rejects it
  now, with usage and exit 2 like any other bad argument.
- **The index rebuild could wait forever with nothing printed.**
  `create_index` and `delete_index` were called with no timeout, and the SDK
  documents that as waiting indefinitely. `confirm_deletion` already refuses to
  create that state, in the same file, because it is indistinguishable from the
  slow indexing it precedes.
- **A banner branch that could never fire**, holding its own copy of the
  wording. chat.js searched a 200 `/healthz` body for `index-empty`, which a 200
  body cannot contain — every degraded state is a 503. So one of two copies of a
  visitor-facing sentence was unreachable.
- **A stylesheet rule that has never matched anything.** `.container` against a
  template whose only container is `container-fluid`, and `align-content` does
  nothing on a non-flex element anyway. An earlier round's cleanup missed it by
  asking whether the class name appeared in the page: "container" appears inside
  "container-fluid".

- **The deploy pushed eighteen `.pyc` files and kept what the repository had
  deleted.** `cp -R src dest/` merges into an existing directory rather than
  replacing it, so a Space held the union of every deploy ever made; and it
  copied the working tree, so a local `__pycache__` — which any test run
  produces — went out with it, bytecode from two interpreters, neither of them
  the image's. They also broke the "nothing to push" shortcut, since running
  the suite was enough to change them. Each entry is now cleared and written
  from `git archive HEAD`, which also makes the deploy agree with its own
  stamp: the revision baked into the Dockerfile is `HEAD`, and the code shipped
  beside it used to be whatever the tree held.
- **The Space's image contained the Space's git history.** `.dockerignore` was
  not in the payload, and a `docker build` with no `.dockerignore` excludes
  nothing — measured on a throwaway context: `.git` lands in the image. `.git`
  is the first line of this repository's copy, which is why CI never showed it.
- **`.dockerignore` patterns are anchored at the context root**, so
  `__pycache__/` excluded the one beside the Dockerfile and took nothing out of
  `src/`. The image carried `src/__pycache__/*.cpython-312.pyc` — in an image
  built on 3.10, whose Dockerfile sets `PYTHONDONTWRITEBYTECODE=1` on purpose.
  The four artefact patterns carry `**/` now. The test helper that should have
  caught it modelled docker as more generous than it is: it also compared
  basenames, which made `__pycache__/` look like it covered any depth.
- **128MB of local mypy cache was going into the image**, along with
  `.pytest_cache`, `.ruff_cache` and a `.coverage` holding absolute paths from
  whoever last ran the suite. More than the embedding weights that are baked in
  deliberately. CI never carried it, because its checkout is fresh — so the
  affected builds were the local ones, which is what `make image` exists to
  produce in CI's place, and what the 3GB budget was being measured against.
  2.3GB to 2.15GB. `make clean` was missing `.mypy_cache` too, while claiming
  to remove the caches.
- **Five files reached CI's image and not the Space's** — `store_index.py`,
  `template.py`, the `Makefile`, `.env.example`, `.gitattributes` — none of
  which either image needs. Excluded, which makes both smaller rather than both
  larger.
- **The CSP permitted `data:` images and nothing loads one.** Not the template,
  `chat.js` or `style.css`, and not bootstrap 4.1.3 or Font Awesome 5.5.0,
  whose stylesheets were fetched and searched for `url(data:` — zero matches.
  Both checks on unused sources matched `https://[a-z0-9.-]+`, and a scheme has
  no host, so neither could see it.
- **A body ceiling of 501 bytes passed validation while refusing legal
  questions.** `MAX_QUESTION_CHARS` counts characters and `MAX_CONTENT_BYTES`
  caps bytes, with UTF-8 and percent-encoding in between, so comparing the two
  numbers compares nothing. A 500-character question measures 504 bytes in
  ASCII letters and 6004 in emoji — twelve bytes per character. The bound is the
  worst case now, derived rather than guessed. The shipped defaults were always
  fine; this matters to anyone setting either variable, which is why they are
  variables.
- **Static assets were cached for a year without the content hash that makes a
  year safe** — `/favicon.ico`, which is a route and so never fingerprinted; a
  bare `/static/…` URL; and a file whose digest could not be read, where the
  URL carries no `?v=` and nothing can ever invalidate it. That last case was
  permanent, because `digest()` memoised its failures: one `OSError` was
  remembered for the life of the process. Only successes are cached now, and
  the long lifetime requires the hash to be in the URL.
- **`make lint` ran `bash -n` on the deploy script and not `shellcheck`**,
  which CI runs — so `make check` could pass while CI failed on a finding
  nobody local could see. It matters this round, which rewrote that script's
  copy loop into a `git archive | tar -x` pipeline. Guarded, because not every
  machine has shellcheck, and announced rather than skipped quietly.

- **Every error raised before a route ran came back as HTML.** The plain-text
  error handler was registered for 404, 405 and 500 — the statuses a route can
  produce. Setting `MAX_CONTENT_LENGTH` makes Werkzeug raise 413 before any
  route is reached, so that one arrived as a `<!doctype html>` document, which
  the page appends to the chat log verbatim. Registered against `HTTPException`
  now, so the list cannot fall behind another way to fail; redirects pass
  through untouched, since `RequestRedirect` is one too and its `Location`
  header is the whole response. There had been a test for the oversize body: it
  asserted the status and stopped, while the 404 and 405 tests beside it both
  assert the content type.
- **One status meant two different sentences.** 413 from the route said "that
  question is too long"; 413 from Werkzeug said "Request Entity Too Large."
  Both statuses that overlap the app's own vocabulary are now worded from
  `src/messages.py`, which restores that module's claim to hold everything a
  visitor can be shown — a status name reaching a chat bubble had made it
  false. A mistyped URL still answers "Not Found.", for the one audience that
  never sees a chat bubble.
- **The page displayed whatever came back.** The failure path handed
  `jqXHR.responseText` straight to the chat log. The server is not the only
  thing that can produce a body: a Space's router answers with its own HTML
  page while the container is starting, restarting or asleep, and a 502 from in
  front of the app is the likeliest failure this page will ever see. It now
  shows a body only when the server declared `text/plain` and it is short
  enough to be one of the sentences the server can send.
- **Naming one test file ran all 490 of them.** `--cov` takes an optional
  value, so the bare one in `addopts` consumed the path as its argument: with
  nothing positional left pytest fell back to `testpaths`, and coverage was
  pointed at a test file it then reported as never imported. 0.4 seconds of
  work took 22, and no coverage report was produced at all. Nothing was ever
  red, which is why it lasted.
- **CI proved an image built from a context the Space does not have.** The
  image job is the only check that the image a Space builds will build, and it
  runs against this repository; the Space builds against whatever the deploy
  script pushed. pyproject.toml was in one context and not the other — no
  difference today, and a broken Space the moment `setup.py` is dropped for
  PEP 621 metadata, with CI still green.
- **`store_index.py` was outside the coverage gate** — 99 statements, so the
  figure above the floor described two thirds of the shipped code.
- **.dockerignore excluded two files that no longer exist.** It kept
  `pytest.ini` and `ruff.toml` out of the image long after both tools' settings
  moved into pyproject.toml, and said it was keeping the tooling config out
  while the file holding all of it went in.
- Three statements in the README, CONTRIBUTING and `make help` that the
  coverage change itself made untrue, and two reasons given in this round's own
  commit messages that were wrong in their detail.

- **Two more tests asserted nothing.** Byte-identical bodies, different names,
  both relying on a fixture's stubs raising if touched — a real guarantee, but
  an invisible one, and the second test's name described what the fixture
  provides rather than anything the test did. Merged, with the claim visible.
- **Five of fourteen modules were missing from the README's structure tree.**
  The other three hand-kept lists of those modules each have a test requiring
  completeness, and each has caught an omission; this one had no test.
- **Two tests had never run.** Both were indented one level too far and became
  methods of a stub class, and pytest collects test methods only from classes
  named `Test*`. One of them asserted that reading the index size cannot hang
  the startup; making it run showed that it could. `ThreadPoolExecutor`'s
  context manager calls `shutdown(wait=True)` on exit, so the timeout fired and
  the call then waited for exactly the hang it had just abandoned — measured at
  0.2s to time out and 3.0s to return. Coverage stayed at 100% throughout,
  because other tests exercised those lines without asserting what these two
  asserted.
- **The notebook's embeddings were still coming from the sunset package.** The
  third repeat this fortnight of a fix applied where the problem was noticed
  rather than everywhere it existed. The check that forbids it now reads the
  notebook too, and forbids the retired submodule everywhere rather than
  permitting it per module.
- **`store_index.py`'s Pinecone client was untyped**, which hid that
  `confirm_upsert` is handed a value that is `None` on one path. It never is in
  practice — the dry run returns first — but two branches apart that is a
  connection a reader has to reconstruct and a checker cannot. Now stated.

### Added

- A check that nothing is imported without being declared — the direction that
  was claimed to exist and did not.
- Checks that every CDN tag carries a subresource integrity hash, in an
  algorithm browsers accept, fetched with `crossorigin` so the hash is actually
  enforced. All three hashes were verified against the live CDNs while writing
  it; that check is not in the suite, which passes with no network.
- A check that the chain Protocol is satisfied by all three things passed to
  `create_app` — the stub, the refusing stand-in, and langchain's Runnable.
  mypy does not run over tests/, so the Protocol had been documenting an
  intention and enforcing nothing. What is asserted is that one positional
  argument binds, not that the parameter shares a name, since the real chain's
  is `input` where the Protocol says `payload`.
- Checks that style.css and the markup agree in both directions, and that the
  five ids chat.js reaches for exist in the template — `$("#serviceBanner")` on
  a missing element is an empty jQuery set on which every method succeeds and
  does nothing. The case that makes it worth having is a misspelling:
  `msg_cotainer` is wrong identically in chat.js and style.css, so correcting
  one file looks like tidying and unstyles every message in the log.
- Checks that every degraded state answers a failure status and no healthy
  reply names a degraded reason, which is what the banner logic depends on.
- A check that the stubbed Pinecone client's signatures match the real one's;
  the fake took `delete_index(self, name)` where the client takes a timeout too.

- The deploy says when the working tree is not what it ships, naming the files
  — `git status --porcelain` on the payload, so untracked files appear too,
  which is the half a `git diff` would miss and the half that produced the
  `.pyc` problem. A warning rather than a refusal: deploying `HEAD` from a tree
  carrying unrelated edits is a normal thing to want.
- A check that nothing tracked reaches CI's image that the Space's image lacks,
  which is the difference CI structurally cannot see, and one that the Space
  carries the exclusion rules its own build reads.
- Checks that no tool cache or git-ignored artefact reaches the image: a fixed
  list, which is what runs in CI where the checkout is clean, and a scan of the
  working copy, which catches what nobody thought to list.
- A check that `make clean` removes every cache `.dockerignore` names.
- A CSP check that tokenises the policy into (directive, source) pairs and
  rejects `data:`, `blob:`, `*` and both `unsafe-*` keywords wherever they
  appear. It is the second attempt: the first asked whether each source
  appeared in the page's text and skipped the ones that did, which excused
  `data:` immediately, because `chat.js` contains `data: { msg: question }`.

- A check that the CI workflow's eight `run:` blocks are valid shell. They are
  107 lines of loops, `case` statements and nested quoting that only ever
  execute on a runner, so a missing `fi` was found by pushing; each block is
  now extracted and passed through `bash -n`, with the failure naming the step.
  Alongside it: no action pinned to a moving ref, no tabs, a `timeout-minutes`
  for every job, and the token still read-only.
- A check that the files the app serves parse — HTML tag balance and a
  doctype first, stylesheet braces, the notebook as one JSON document carrying
  no outputs, and every `filename=` in the template naming a file that exists.
  All of these had been run by hand for weeks.
- A check that every shipped module is either measured by coverage or named as
  unmeasured with a reason, in both directions.
- A check that the coverage floor only appears on a command that also collects
  coverage — `--cov-fail-under` is accepted and silently ignored without it, so
  a step that lost the flag would pass and stop checking — and that `--cov`
  never precedes a non-option. Both were wrong when first written: each
  searched the whole file and was satisfied by the comment explaining the rule
  rather than by any command.
- A check that no `.dockerignore` pattern names something absent, with the ten
  entries that exist to exclude what a working copy grows listed as such.
- A check that the deploy payload covers every file the image is built from.
- `make lint` runs the Dockerfile lint CI runs.

- `X-App-Revision` on every response, beside `X-Request-Id`: a reply kept in a
  bug report can be traced to a request and to the build that produced it.
- A check for a test that asserts nothing, with the four that legitimately
  assert by not raising listed and explained — the adjacent failure to a test
  that never runs, and indistinguishable from the outside.
- A check that the README's structure tree lists every module, in both
  directions, and one that the Makefile's `PYTHON=` override still works.
- A check for a test hidden inside a helper class, which is how the two above
  went unnoticed, and one for the notebook's JSON, since every edit to it in
  this history was made by a script and more than one of those got it wrong.
- Checks for the things that silently do nothing: a dependency declared after
  its code is gone, a test file pytest will not collect, a test function
  shadowed by a second definition of the same name, and a version that
  disagrees with the changelog.

### Changed

- The deploy ships the payload as of `HEAD` rather than the working tree, so
  committing first is now required to deploy a change. Documented in DEPLOY.md
  with the reason, which is that the revision `/healthz` reports is `HEAD`.

- Coverage is collected by `make test` and the CI step rather than by every
  `pytest` invocation, which is what lets a single file be a single file. The
  source list stays in `[tool.coverage.run]`, stated once.

- Four constants that were only read by their own module are named as
  internal; a module's public names are what other code is invited to depend
  on.
- The dependency checks build the distribution map once instead of per
  package, which was 4.5 seconds of an 11-second suite in a file that is not
  marked slow. CI caches the pre-commit environments.
- The text a visitor can be shown is collected in `src/messages.py`, with the
  rule it follows written down — no internal vocabulary — and a test holding it
  there, the banner strings in `chat.js` included.
- `make image` runs the checks CI runs against the built image, rather than one
  of the six, and a test compares the two including the size budget as a number.
- `store_index.main()` is five calls: the index preparation, the confirmation
  and now the upsert loop are each their own function and tested directly.
- `src/webapp.py` is 220 lines from 295: the asset cache-busting and the
  request lifecycle are their own modules, each holding settings that only
  make sense together.
- Notebooks are never merged automatically — git's line-based merge of one
  JSON document produces files that are plausibly broken rather than
  conflicts to resolve.

## 0.1.0 — 2026-10-16

The first version that deploys and is checked end to end: the image builds and
is booted in CI, the retrieval path is tested, and the two API keys are the only
thing standing between a fresh clone and a running Space.

Everything below was fixed or added on the way there.

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
