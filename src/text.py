"""Pure text helpers.

Deliberately free of imports from the rest of the project and from LangChain:
this module can be imported -- and therefore tested -- without loading the
embedding model or reaching Pinecone, which importing app.py or store_index.py
both do.
"""

import re

# deepseek-r1 wraps its chain of thought in <think>...</think>. It is useful in
# a log and should never reach the user.
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_reasoning(answer: str) -> str:
    """Remove any <think> blocks from a model reply and tidy the whitespace."""
    return THINK_BLOCK.sub("", answer).strip()
