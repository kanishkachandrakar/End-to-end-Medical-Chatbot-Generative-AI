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
# Questions answered per minute across the whole app; 0 disables the limit.
RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "30"))
GROQ_MODEL = os.environ.get("GROQ_MODEL", "deepseek-r1-distill-qwen-32b")
GROQ_TIMEOUT = float(os.environ.get("GROQ_TIMEOUT", "60"))

# Web server.
PORT = int(os.environ.get("PORT", "8080"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()


def _validate() -> None:
    """Reject settings that would fail later and further from the cause.

    Every one of these produces a confusing symptom rather than an error:
    an overlap at or above the chunk size makes RecursiveCharacterTextSplitter
    loop, a TOP_K of zero retrieves nothing so answers are silently ungrounded,
    and a dimension mismatch is only reported by Pinecone at upsert time.
    """
    problems = []
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        problems.append(
            f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) must be smaller than "
            f"CHUNK_SIZE ({CHUNK_SIZE})"
        )
    if CHUNK_SIZE <= 0:
        problems.append(f"CHUNK_SIZE ({CHUNK_SIZE}) must be positive")
    if CHUNK_OVERLAP < 0:
        problems.append(f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) cannot be negative")
    if TOP_K < 1:
        problems.append(f"TOP_K ({TOP_K}) must be at least 1")
    if EMBED_DIM < 1:
        problems.append(f"EMBED_DIM ({EMBED_DIM}) must be at least 1")
    if MAX_QUESTION_CHARS < 1:
        problems.append(f"MAX_QUESTION_CHARS ({MAX_QUESTION_CHARS}) must be at least 1")
    if GROQ_TIMEOUT <= 0:
        problems.append(f"GROQ_TIMEOUT ({GROQ_TIMEOUT}) must be positive")
    if not 1 <= PORT <= 65535:
        problems.append(f"PORT ({PORT}) is outside 1-65535")

    if problems:
        raise ValueError("Invalid configuration: " + "; ".join(problems))


_validate()
