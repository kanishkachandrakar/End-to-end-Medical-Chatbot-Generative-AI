# End-to-End Medical Chatbot (Generative AI)

A retrieval-augmented medical question-answering chatbot. It indexes a medical
reference book into a Pinecone vector store and answers user questions through a
Flask web UI, grounding every reply in passages retrieved from the book rather
than relying on the LLM's memory alone.

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
| Orchestration | LangChain (`create_retrieval_chain`, `create_stuff_documents_chain`) |
| LLM | `deepseek-r1-distill-qwen-32b` via [Groq](https://groq.com) (`langchain_groq`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| Vector store | Pinecone serverless (AWS `us-east-1`), cosine metric |
| PDF parsing | `pypdf` through LangChain's `PyPDFLoader` |
| Front end | Jinja template + jQuery + Bootstrap-style CSS |
| Config | `python-dotenv` (`.env`) |

## Project structure

```
.
├── app.py               # Flask app: builds the RAG chain and serves the chat UI
├── store_index.py       # Creates the Pinecone index (see "Build the vector index")
├── template.py          # One-off scaffold script that created the initial file layout
├── setup.py             # Makes `src/` installable (`pip install -e .`)
├── requirements.txt
├── .env.example         # Names of the environment variables the app expects
├── Data/
│   └── Medical_book.pdf # Source document that gets indexed
├── src/
│   ├── helper.py        # PDF loading, chunking, embedding model
│   └── prompt.py        # System prompt for the LLM
├── templates/
│   └── chat.html        # Chat page
├── static/
│   └── style.css
├── research/
│   └── trials.ipynb     # Notebook used to prototype the pipeline and upload chunks
└── medicalbotpic.jpeg   # Bot avatar
```

## Prerequisites

- Python 3.10 (the project was developed in a conda env named `medibot`)
- A [Pinecone](https://www.pinecone.io/) account and API key
- A [Groq](https://console.groq.com/) API key
- An OpenAI API key (read at startup by `app.py`; the LLM itself runs on Groq)
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

## Configuration

Copy the example file and fill in your keys:

```bash
cp .env.example .env
```

```dotenv
PINECONE_API_KEY=
OPENAI_API_KEY=
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
and creates a serverless Pinecone index called `medicalbot`
(384 dimensions, cosine metric, AWS `us-east-1`).

> **Note:** `store_index.py` currently only *creates* the index. The step that
> embeds the chunks and uploads them (`PineconeVectorStore.from_documents(...)`)
> lives in `research/trials.ipynb`. Run that cell after the index exists, or
> the app will retrieve nothing. Re-running `store_index.py` against an existing
> index raises a Pinecone `409 Conflict`.

## Run the chatbot

```bash
conda activate medibot
python app.py
```

Then open <http://localhost:8080>. The server binds to `0.0.0.0:8080` with
Flask debug mode on, so it is reachable from other machines on your network.

The UI posts each message as a form field `msg` to `POST /get` and renders the
plain-text reply, so you can also hit the endpoint directly:

```bash
curl -X POST http://localhost:8080/get -d "msg=What is hypertension?"
```

## Notes & troubleshooting

- **First start is slow** – `download_hugging_face_embeddings()` pulls
  `all-MiniLM-L6-v2` from HuggingFace on the first run; later runs use the cache.
- **`TypeError: str expected, not NoneType` on startup** – one of the keys in
  `.env` is missing. `app.py` writes `PINECONE_API_KEY` and `OPENAI_API_KEY`
  straight into `os.environ` and will fail if either is unset.
- **Empty or off-topic answers** – confirm the `medicalbot` index actually
  contains vectors (see the note under *Build the vector index*).
- **Changing the model** – edit the `ChatGroq(...)` call in `app.py`. If you
  switch to a model that doesn't emit `<think>` tags the regex cleanup in
  `chat()` is harmless.
- **Changing the source document** – drop any `*.pdf` into `Data/`;
  `load_pdf()` globs the whole directory. Rebuild the index afterwards.

## License

MIT — see [LICENSE](LICENSE).
