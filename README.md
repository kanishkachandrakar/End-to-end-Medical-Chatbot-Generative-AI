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
