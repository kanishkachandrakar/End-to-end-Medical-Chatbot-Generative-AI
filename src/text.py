"""Pure text helpers.

Deliberately free of imports from the rest of the project and from LangChain:
this module can be imported -- and therefore tested -- without loading the
embedding model or reaching Pinecone, which importing app.py or store_index.py
both do.
"""

import hashlib
import re
from collections.abc import Iterable

# deepseek-r1 wraps its chain of thought in <think>...</think>. It is useful in
# a log and should never reach the user.
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_reasoning(answer: str) -> str:
    """Remove any <think> blocks from a model reply and tidy the whitespace."""
    return THINK_BLOCK.sub("", answer).strip()


def chunk_ids(chunks: Iterable) -> list[str]:
    """Derive a stable vector id for each chunk from its text.

    Hashing the content rather than generating a uuid makes re-indexing an
    overwrite instead of an append. Byte-identical chunks collapse onto one id,
    which is wanted: duplicates embed to the same vector, so keeping several
    only wastes space and lets one passage fill a top-k result on its own.
    """
    return [
        hashlib.sha1(chunk.page_content.encode("utf-8")).hexdigest()
        for chunk in chunks
    ]
