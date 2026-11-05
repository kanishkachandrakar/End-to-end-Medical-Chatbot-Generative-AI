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


def test_every_module_in_src_explains_why_it_exists():
    """Each docstring here records a decision, which is the convention.

    src/prompt.py had none -- the one module with no behaviour in it and the
    most influence over what a visitor reads, so the clauses in it looked
    arbitrary rather than chosen. src/__init__.py is exempt: it is empty on
    purpose, which is what makes src/ a package.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    missing = [
        path.name
        for path in sorted((root / "src").glob("*.py"))
        if path.name != "__init__.py"
        and not ast.get_docstring(ast.parse(path.read_text()))
    ]
    assert not missing, f"no module docstring: {missing}"


def test_the_prompt_and_the_page_agree_about_what_the_bot_does():
    """The page promises a lookup of medical terms; the prompt has to ask for it.

    Two statements of the same scope, in two files, shown to the same person
    -- one as a subtitle under the name and one to the model. If they drift,
    the page is advertising something the prompt does not request.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    page = (root / "templates" / "chat.html").read_text()
    assert "medical terms" in page, "the page no longer says what it looks up"
    assert "medical term" in system_prompt, "the prompt no longer asks for it"
