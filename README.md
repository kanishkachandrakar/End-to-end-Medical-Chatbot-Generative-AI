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
