# Security

## Credentials exposed in this repository's history

Pinecone and Groq API keys were committed to this repository and remain
reachable in its git history. Removing them from the current files does not
remove them from the history, and this repository is public — so those keys
should be treated as disclosed, regardless of what the working tree contains.

**If you are the owner of this project: rotate both keys.** That is the only
step that actually ends the exposure.

- Groq: <https://console.groq.com/keys>
- Pinecone: <https://app.pinecone.io> → API Keys

Rotating is enough. Rewriting the history is optional and has a real cost: it
changes every commit hash after the first affected one, which breaks every
existing clone, fork and link to a commit. If you want it anyway:

```bash
pip install git-filter-repo
git filter-repo --replace-text <(printf 'pcsk_***==>REDACTED\ngsk_***==>REDACTED\n')
git push --force --all
```

Replace `pcsk_***` and `gsk_***` with the literal key strings. Do it on a fresh
clone, and expect to re-create any forks. Note that GitHub keeps unreachable
objects accessible for a period after a force push, so rotation still comes
first.

## How secrets are handled now

- `.env` is gitignored, and `.env.example` carries names with no values.
- `src/config.py` reads no secrets at all, which is what lets it be imported in
  tests and CI; the keys are read in `app.py` and `store_index.py` through
  `src/env.py`.
- On Hugging Face Spaces both keys are set as *secrets*, not variables, so they
  are not shown to visitors. See [DEPLOY.md](DEPLOY.md).
- Nothing logs a key, and questions are not logged either — on a public
  deployment those are strangers' health questions.

## Supply chain

The dependencies are the part of this project most likely to change without
anyone deciding to change it, and twice they have broken it outright. Both
times the install succeeded and the image built; only starting the app showed
the problem.

What is in place now:

- **Every requirement has a lower and, where a major release has broken us
  before, an upper bound** — with the reason beside it in `requirements.txt`. A
  bare package name lets the resolver pick anything, which is how
  `langchain-pinecone` 0.0.1 came to be installed.
- **The base image is pinned by digest**, so a rebuilt tag cannot change the
  foundation underneath a build. Dependabot watches the Dockerfile, which is
  what keeps that pin from freezing the base at an unpatched version.
- **Dependabot proposes every bump as a pull request**, and CI builds the image
  and boots it before the bump is believed. Bounds are only tolerable because
  the updates still arrive.
- **A test fails on any tracked file containing a key-shaped string**, so a
  pasted secret is caught at commit time.
- **`requirements.txt` and `requirements-dev.txt` are held to identical bounds**
  on shared packages. Testing against versions the container will not install
  is how both failures stayed hidden.

What is not in place: there is no lock file, so two builds a month apart can
install different patch versions. For a demo that is the right trade — the
alternative is a file that must be regenerated on every change — but it does
mean a build is reproducible only to the precision of the bounds.

## Reporting something

Open an issue, or email the address in `setup.py`. This is a learning project
with no users to protect, so there is no embargo process — a public issue is
fine.

## What this deployment does and does not defend against

Worth being explicit, since it is a public endpoint:

- **Quota exhaustion** is limited, not prevented. `RATE_LIMIT_PER_MINUTE` caps
  answers across all visitors, and `MAX_QUESTION_CHARS` caps the prompt, but the
  limit is global rather than per visitor and anyone can spend it.
- **No authentication.** Anyone who can reach the URL can ask questions.
- **Prompt injection is not mitigated.** The retrieved context comes from one
  trusted PDF, so there is no untrusted content in the prompt — but a question
  can still steer the model, and nothing validates that an answer came from the
  retrieved passages.
- **Output is treated as untrusted** by the front end, which renders replies
  with `.text()` and runs under a CSP that forbids inline script.
