# End-to-End Medical Chatbot (Generative AI)

A retrieval-augmented medical question-answering chatbot. It indexes a medical
reference book into a Pinecone vector store and answers user questions through a
Flask web UI, grounding every reply in passages retrieved from the book rather
than relying on the LLM's memory alone.

![Python](https://img.shields.io/badge/python-3.10-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
