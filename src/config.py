"""Tunable settings, overridable through the environment.

Secrets are deliberately not read here -- the API keys are handled where
they are used, so importing this module never requires a configured .env.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Pinecone index. EMBED_DIM must match the output size of EMBED_MODEL,
# otherwise Pinecone rejects the upserts from store_index.py.
INDEX_NAME = os.environ.get("PINECONE_INDEX", "medicalbot")
PINECONE_CLOUD = os.environ.get("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.environ.get("PINECONE_REGION", "us-east-1")

# Embeddings.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "384"))

# Chunking.
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "20"))

# Retrieval and generation.
TOP_K = int(os.environ.get("TOP_K", "3"))
MAX_QUESTION_CHARS = int(os.environ.get("MAX_QUESTION_CHARS", "500"))
GROQ_MODEL = os.environ.get("GROQ_MODEL", "deepseek-r1-distill-qwen-32b")

# Web server.
PORT = int(os.environ.get("PORT", "8080"))
