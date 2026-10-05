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
