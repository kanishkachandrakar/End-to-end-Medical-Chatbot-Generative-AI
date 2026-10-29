---
title: Medical Chatbot
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# End-to-End Medical Chatbot (Generative AI)

A retrieval-augmented medical question-answering chatbot. It indexes a medical
reference book into a Pinecone vector store and answers user questions through a
Flask web UI, grounding every reply in passages retrieved from the book rather
than relying on the LLM's memory alone.

[![CI](https://github.com/kanishkachandrakar/End-to-end-Medical-Chatbot-Generative-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/kanishkachandrakar/End-to-end-Medical-Chatbot-Generative-AI/actions/workflows/ci.yml)
![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Python](https://img.shields.io/badge/python-3.10-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

## Features

- **Grounded answers** – responses are generated from the top-3 most relevant
  chunks of the source PDF, not from the model's general knowledge.
- **Medical-only scope** – the system prompt instructs the model to answer only
  medical terms and to decline anything else.
- **Concise, definition-style output** – answers are capped at three sentences
  and written as definitions.
- **Reasoning stripped from output** – `<think>…</think>` blocks emitted by the
  DeepSeek reasoning model are removed before the reply reaches the UI.
- **Simple chat UI** – a single-page jQuery front end that talks to a Flask
  endpoint.

## How it works

```
Data/Medical_book.pdf
        │  PyPDFLoader + DirectoryLoader
        ▼
   page documents
        │  RecursiveCharacterTextSplitter (chunk_size=500, overlap=20)
        ▼
   text chunks ──► all-MiniLM-L6-v2 embeddings (384-dim) ──► Pinecone index "medicalbot"
                                                                       │
 user question ──► embed ──► similarity search (k=3) ◄─────────────────┘
        │
        ▼
 ChatGroq (deepseek-r1-distill-qwen-32b, temperature=0)
   system prompt + retrieved context + question
        │
        ▼
 answer (with <think> tags stripped) ──► Flask /get ──► chat UI
```

1. **Ingest** – the PDF is loaded page by page and split into overlapping
   500-character chunks (`src/helper.py`).
2. **Embed & store** – each chunk is embedded with
   `sentence-transformers/all-MiniLM-L6-v2` and upserted into a serverless
   Pinecone index.
3. **Retrieve** – at query time the question is embedded the same way and the
   three nearest chunks are pulled back.
4. **Generate** – the chunks are stuffed into the system prompt
   (`src/prompt.py`) and sent to the Groq-hosted LLM via a LangChain
   `create_retrieval_chain`.

## Tech stack

| Layer | Choice |
|---|---|
| Web framework | Flask |
| Orchestration | `langchain-classic` (`create_retrieval_chain`, `create_stuff_documents_chain` — moved there in langchain 1.0) |
| LLM | `deepseek-r1-distill-qwen-32b` via [Groq](https://groq.com) (`langchain-groq`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` via `langchain-huggingface` |
| Vector store | Pinecone serverless (AWS `us-east-1`), cosine metric, `langchain-pinecone` |
| PDF parsing | `pypdf` through `PyPDFLoader` (`langchain-community`, the one sunset package left — see `src/loaders.py`) |
| Front end | Jinja template + jQuery + Bootstrap-style CSS |
| Config | `python-dotenv` (`.env`), validated at import by `src/config.py` |

## Project structure

```
.
├── app.py               # Flask app: builds the RAG chain and serves the chat UI
├── Dockerfile           # Container used for deployment
├── DEPLOY.md            # Hugging Face Spaces walkthrough
├── store_index.py       # Creates the Pinecone index (see "Build the vector index")
├── template.py          # One-off scaffold script that created the initial file layout
├── setup.py             # Makes `src/` installable (`pip install -e .`)
├── requirements.txt
├── .env.example         # Names of the environment variables the app expects
├── Data/
│   └── Medical_book.pdf # Source document that gets indexed
├── gunicorn.conf.py     # How the container serves it: bind, workers, timeouts
├── scripts/
│   ├── deploy_space.sh  # Pushes the runtime files at HEAD to a Space
│   └── check_no_keys.py # Pre-commit hook: refuses a file holding an API key
├── tests/               # pytest suite; no keys or network required
├── src/
│   ├── assets.py        # Cache-busting for the static files
│   ├── chunking.py      # Splitting pages into chunks
│   ├── config.py        # Settings, overridable by environment variable
│   ├── env.py           # Required keys, and a check on their shape
│   ├── errors.py        # Failure -> user-facing message and status
│   ├── headers.py       # Security headers, including the CSP
│   ├── helper.py        # The embedding model
│   ├── loaders.py       # Reading the source PDFs
│   ├── messages.py      # Everything a visitor can be shown
│   ├── middleware.py    # Request id, headers, plain-text errors
│   ├── prompt.py        # System prompt for the LLM
│   ├── ratelimit.py     # Token bucket guarding the Groq quota
│   ├── text.py          # Answer cleaning and chunk ids
│   └── webapp.py        # Flask routes, built by create_app()
├── templates/
│   └── chat.html        # Chat page
├── static/
│   ├── chat.js          # The whole chat UI: sends questions, renders replies
│   ├── style.css
│   └── medicalbotpic.jpeg  # Bot avatar
└── research/
    └── trials.ipynb     # Notebook used to prototype the pipeline and upload chunks
```

## Prerequisites

- Python 3.10 (the project was developed in a conda env named `medibot`)
- A [Pinecone](https://www.pinecone.io/) account and API key
- A [Groq](https://console.groq.com/) API key
- ~1 GB of free disk for the sentence-transformers model, downloaded on first run

## Installation

```bash
git clone https://github.com/kanishkachandrakar/End-to-end-Medical-Chatbot-Generative-AI.git
cd End-to-end-Medical-Chatbot-Generative-AI

conda create -n medibot python=3.10 -y
conda activate medibot

pip install -r requirements.txt
```

`requirements.txt` ends with `-e .`, which installs the local `src` package in
editable mode so `from src.helper import …` works from anywhere in the repo.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

`make check` runs the lint and the suite exactly as CI does; `make help` lists
the rest.

Optionally, `pip install pre-commit && pre-commit install` runs the fast checks
before each commit.

The suite needs no API keys, no network and no embedding model — which is also
why CI does not install torch. Everything in `src/` is covered; the floor is
enforced at 95% in CI.

The whole suite is about twenty seconds, and nearly all of that is the files
marked `slow` — they pay for heavyweight imports and for real git work in
temporary repositories, the deploy script's tests among them. Deselecting those
leaves about four seconds, which is the number to work against while iterating:

```bash
make fast                      # or: python -m pytest -m "not slow"
```

A single file is a single file:

```bash
python -m pytest tests/test_webapp.py
```

Coverage is not collected by a bare `pytest`, and is deliberately not in
`addopts`: `--cov` takes an optional value, so sitting there it swallowed the
path and turned the command above into the whole suite measuring the wrong
file. `make test` reports it, and applies the same 95% floor CI does -- along
with `COVERAGE_CORE=sysmon` on Python 3.12 and later, which is the difference
between coverage costing one second and fifteen.

## Configuration

Copy the example file and fill in your keys:

```bash
cp .env.example .env
```

```dotenv
PINECONE_API_KEY=
GROQ_API_KEY=
```

`.env` is gitignored. Never commit real keys — both `app.py` and
`store_index.py` load them with `python-dotenv` at startup.

## Build the vector index

Run this once before starting the app:

```bash
python store_index.py
```

This loads and chunks `Data/Medical_book.pdf`, downloads the embedding model,
creates a serverless Pinecone index called `medicalbot` (384 dimensions, cosine
metric, AWS `us-east-1`) and upserts every chunk into it.

The script is safe to re-run. Vector ids are derived from the chunk text, so a
second run overwrites rather than duplicates.

| Flag | What it does |
|---|---|
| `--dry-run` | Load and chunk the PDFs, report the counts, touch nothing. Needs no API key. |
| `--limit N` | Run the whole pipeline over the first N chunks — a cheap end-to-end check before a full rebuild. |
| `--recreate` | Delete the index first, then rebuild. Needed to clear vectors written under ids this script no longer generates. Asks you to type the index name. |
| `--yes` | Skip that confirmation. Required when running without a terminal. |

After upserting, the script reads the index back and waits up to a minute for
the vector count to match what it sent — Pinecone acknowledges an upsert before
the vectors are queryable, so the count lags. A short count after the wait is
reported, not treated as a failure.

```bash
python store_index.py --dry-run          # what would a rebuild produce?
python store_index.py --limit 50         # does the whole pipeline work?
python store_index.py --recreate         # clean rebuild (asks to confirm)
```

## Run the chatbot

```bash
conda activate medibot
python app.py
```

Then open <http://localhost:8080>. The development server binds loopback only.
It holds your real Pinecone and Groq keys and `POST /get` needs no credentials,
so binding every interface would share your Groq budget with whatever network
you are on. Set `HOST=0.0.0.0` if you actually want to reach it from another
machine — the container is unaffected either way, since gunicorn binds
`0.0.0.0` because that is the only useful thing to do inside one.

Debug mode is off unless you ask for it — set `FLASK_DEBUG=1` in `.env` while
working locally to get the auto-reloader.

The UI posts each message as a form field `msg` to `POST /get` and renders the
plain-text reply, so you can also hit the endpoint directly. It is POST-only:
answering spends a Groq call and the shared rate limit, which is not something
a crawler or a link prefetch should be able to trigger.

```bash
curl -X POST http://localhost:8080/get -d "msg=What is hypertension?"
```

Every response carries `X-Request-Id` and `X-App-Revision`, so a reply kept in
a bug report can be traced to a request and a build.

`GET /livez` answers `alive` whenever the process is serving — that is what the
container's health check uses. `GET /healthz` answers without spending an LLM
call, and reports the running revision, the vector count, and whether the
service is degraded — the chat page
asks it on load and shows a banner if answers would not be grounded. See
[DEPLOY.md](DEPLOY.md) for the states it can return.

The resolved configuration is logged in one line at startup, so a deploy
reading the wrong index or retrieving the wrong number of passages is visible
without comparing environment variables by hand.

## Before this is useful to anyone else

Two things are outstanding and neither can be done from the code:

1. **Rotate the Pinecone and Groq keys.** Both are in this repository's public
   history. Everything in the working tree reads them from `.env`, which is
   gitignored, but the old values remain reachable in the history and should be
   treated as disclosed. [SECURITY.md](SECURITY.md) has the links.
2. **Rebuild the index.** The live `medicalbot` index holds six copies of every
   chunk, written by the notebook's upsert cell under random ids before the
   indexer derived them from the chunk text. Because duplicates embed to
   identical vectors, a top-3 search returns the same passage three times — so
   the model gets a third of the context it is configured for.

   ```bash
   python store_index.py --dry-run     # what a rebuild would produce
   python store_index.py --recreate    # delete and rebuild; asks to confirm
   ```

## Deploying

`Dockerfile` serves the app through gunicorn and reads `$PORT`, so it runs on
Hugging Face Spaces, Cloud Run, Fly or a plain VM:

```bash
docker build -t medibot .
docker run -p 8080:8080 -e PORT=8080 --env-file .env medibot
```

See [DEPLOY.md](DEPLOY.md) for the Hugging Face Spaces walkthrough, including
the two things to do first — rotating the API keys that leaked into this
repository's history, and rebuilding the index to clear duplicate chunks.

## Notes & troubleshooting

- **First start is slow** – `download_hugging_face_embeddings()` pulls
  `all-MiniLM-L6-v2` from HuggingFace on the first run; later runs use the cache.
- **`Missing required environment variable(s)` on startup** – `.env` is absent
  or a key is blank. Copy `.env.example` and fill both values in.
- **Empty or off-topic answers** – ask `/healthz`. `degraded … index-empty`
  means the index has no vectors, so nothing is being retrieved and the model
  is answering from its own knowledge; rebuild with
  `python store_index.py --recreate`.
- **`degraded … chain-unavailable` from `/healthz`** – the app started but the
  retrieval chain did not build: usually a wrong or missing key, or Pinecone
  unreachable. The traceback is in the log. The app stays up on purpose so that
  this is reportable rather than a restart loop.
- **The page loads but every question fails** – check `/healthz` first; if that
  is `ok`, the failure is per-request, and the reply quotes a `ref` that appears
  in the log next to the traceback.
- **Questions are refused with a 429** – the demo answers at most
  `RATE_LIMIT_PER_MINUTE` a minute across all visitors, and the reply says when
  to retry. `X-RateLimit-Remaining` on a successful answer shows what is left,
  so a script can pace itself instead of being refused. Raise the limit, or set
  it to `0` to remove it, if you are running this privately.
- **Index dimension errors** – `store_index.py` refuses up front if the
  existing index's dimension does not match the embedding model, and names
  both. Changing `EMBED_MODEL` means rebuilding with `--recreate`.
- **Changing the model** – edit the `ChatGroq(...)` call in `app.py`. If you
  switch to a model that doesn't emit `<think>` tags the regex cleanup in
  `chat()` is harmless.
- **Changing the source document** – drop any `*.pdf` into `Data/`;
  `load_pdf()` globs the whole directory. Rebuild the index afterwards.

## Known limitations

Worth knowing before judging the answers:

- **Retrieval quality is bounded by one book.** Everything comes from the 1999
  Gale Encyclopedia of Medicine, so the coverage and the vintage are whatever
  that book has. Nothing in the pipeline checks a claim against anything newer.
- **No conversation memory.** Each question is answered independently; the
  retriever never sees what you asked before, so follow-ups like "and in
  children?" retrieve nothing useful.
- **Chunks are 500 characters with 20 of overlap**, which splits mid-sentence
  often enough that a retrieved passage can begin or end mid-clause. The model
  sees exactly that.
- **One global rate limit**, not one per visitor — see `src/ratelimit.py` for
  why. A busy visitor can make others wait.
- **Reasoning tokens are paid for and discarded.** The model emits a `<think>`
  block that is stripped before display, so part of every response's latency and
  token cost never reaches the user.
- **Not medical advice**, and nothing enforces that beyond the prompt asking for
  it. Treat it as a search tool over one reference book.

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) covers the setup, what each CI check is for,
and the handful of design choices that look odd until you know why.

## Security

API keys were committed to this repository's history and should be treated as
public. [SECURITY.md](SECURITY.md) explains what to rotate, how secrets are
handled now, and what a public deployment of this does and does not defend
against.

## Changelog

[CHANGELOG.md](CHANGELOG.md) records the behavioural changes, including several
bugs that made earlier versions of this project answer without retrieving
anything.

## License

MIT — see [LICENSE](LICENSE).
