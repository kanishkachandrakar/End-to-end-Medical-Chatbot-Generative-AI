"""The system prompt is interpolated by ChatPromptTemplate at request time."""

import re

from src.prompt import system_prompt


def test_declares_the_context_placeholder():
    """create_stuff_documents_chain fills {context}; without it nothing is cited."""
    assert "{context}" in system_prompt


def test_context_is_the_only_placeholder():
    """A stray brace becomes an unknown template variable and raises at runtime.

    ChatPromptTemplate parses the prompt as an f-string-style template, so a
    literal '{' anywhere else -- in a unit, a range, an example -- fails the
    request rather than the import, which makes it an expensive typo.
    """
    assert re.findall(r"\{([^}]*)\}", system_prompt) == ["context"]


def test_braces_are_balanced():
    assert system_prompt.count("{") == system_prompt.count("}") == 1


def test_is_a_single_nonempty_string():
    """It is written as adjacent string literals, which silently concatenate."""
    assert isinstance(system_prompt, str)
    assert len(system_prompt.strip()) > 100


def test_words_are_not_run_together():
    """Adjacent literals need a trailing space, or 'conciseWrite' sneaks in."""
    assert not re.search(r"[a-z][A-Z]", system_prompt.replace("{context}", ""))
