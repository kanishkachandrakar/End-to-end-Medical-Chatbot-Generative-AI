"""Pure text helpers.

Deliberately free of imports from the rest of the project and from LangChain:
this module can be imported -- and therefore tested -- without loading the
embedding model or reaching Pinecone, which importing app.py or store_index.py
both do.
"""

import hashlib
import re
from collections.abc import Iterable, Iterator, Sequence
from typing import TypeVar

# deepseek-r1 wraps its chain of thought in <think>...</think>. It is useful in
# a log and should never reach the user.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)

# An unterminated block, which is what a reply cut short produces: the model
# hits the token ceiling or the 60s timeout mid-reasoning, so the opening tag
# arrives and the closing one never does.
_UNTERMINATED_THINK = re.compile(r"<think>.*\Z", re.DOTALL)

# A closing tag with no opener, from a reply whose beginning was lost. Whatever
# precedes it is the tail of the reasoning.
_ORPHAN_CLOSE = re.compile(r"\A.*?</think>", re.DOTALL)

T = TypeVar("T")


def strip_reasoning(answer: str) -> str:
    """Remove the model's chain of thought from a reply, closed or not.

    Pairs first, then the two unbalanced cases, which are what a truncated
    reply produces -- the model reaches its token ceiling or the 60s timeout
    partway through reasoning, so the opening tag arrives and the closing one
    never does. Those used to be left alone, on the stated grounds that a lone
    opening tag must not swallow the answer.

    The trade is the other way round. An answer after an unclosed <think> is
    not a thing this model produces: the tag is unclosed precisely because
    generation stopped, so there is nothing after it but reasoning. Leaving it
    alone published that reasoning to a visitor, tag and all, in a module whose
    first comment says it should never reach the user. Stripping it leaves an
    empty string, which the route already answers with NO_ANSWER.

    The cost is a reply that quotes "<think>" on purpose, which for a chatbot
    grounded in a medical textbook is a trade worth making.
    """
    answer = _THINK_BLOCK.sub("", answer)
    answer = _UNTERMINATED_THINK.sub("", answer)
    answer = _ORPHAN_CLOSE.sub("", answer)
    return answer.strip()


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


def batched(items: Sequence[T], size: int) -> Iterator[list[T]]:
    """Yield consecutive slices of ``items`` of at most ``size`` entries.

    Used to upsert the book a few hundred chunks at a time rather than in one
    call that embeds everything into memory and reports no progress.
    """
    if size < 1:
        raise ValueError("size must be at least 1")
    for start in range(0, len(items), size):
        yield list(items[start : start + size])
