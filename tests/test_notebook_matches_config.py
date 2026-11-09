"""The notebook prototypes the pipeline, and hardcodes what config.py decides.

research/trials.ipynb is where this project's pipeline was worked out, and it
is still the thing someone opens to understand it. It names the model, the
embedding model, the index and the dimension as literals, because a notebook
has no settings module -- so the two agree only as long as nobody changes
src/config.py.

This notebook has drifted before, twice: it kept importing the sunset
langchain-community embeddings after the application had moved, and it kept
importing pinecone.grpc after store_index.py had stopped. Both were found by
a check that was added after the fact. This is the same check for the values.
"""

import json
import pathlib

import pytest

from src import config

ROOT = pathlib.Path(__file__).resolve().parent.parent
NOTEBOOK = json.loads((ROOT / "research" / "trials.ipynb").read_text())
SOURCE = "".join("".join(cell.get("source", [])) for cell in NOTEBOOK["cells"])


def test_the_notebook_source_was_read():
    """Every check below looks for a string in it; an empty read passes them."""
    assert len(SOURCE) > 1000
    assert "PineconeVectorStore" in SOURCE


@pytest.mark.parametrize(
    "setting",
    ["GROQ_MODEL", "EMBED_MODEL", "INDEX_NAME", "EMBED_DIM", "TOP_K"],
)
def test_the_notebook_uses_the_configured_value(setting):
    """A literal that disagrees with config.py prototypes a different system.

    The comparison is one-directional on purpose: the notebook is allowed to
    not mention a setting, but if it mentions one it has to be the current
    value. The name is reported rather than the diff, because the fix is
    always to look at the cell.
    """
    import re

    value = getattr(config, setting)
    if isinstance(value, str):
        # The quoted literal, not the bare value. Searching for the value
        # alone made this pass when the notebook said "medicalbot-v2",
        # because the configured name is a substring of the changed one.
        quoted = (f'"{value}"', f"'{value}'")
        assert any(form in SOURCE for form in quoted), (
            f"{setting}={value!r} is not in the notebook as a literal"
        )
        return
    # Numbers appear as arguments, so a word boundary keeps 384 from matching
    # inside 3840.
    assert re.search(rf"\b{value}\b", SOURCE), (
        f"{setting}={value} is not in the notebook"
    )


def test_a_changed_default_would_be_noticed():
    """The check above is only useful if the notebook really pins these.

    Asserting the current literals are present means a config change makes it
    fail -- which is the point -- so this records that the notebook is
    expected to be edited alongside, not that the test is wrong.
    """
    assert config.INDEX_NAME in SOURCE
    assert config.GROQ_MODEL in SOURCE


def test_the_notebooks_table_points_at_the_right_files():
    """Its header table maps each prototype function to the maintained one.

    `load_pdf` was listed under src/helper.py and has lived in
    src/loaders.py since that module was split out to keep the sunset
    langchain-community import away from the web app. So the one document
    whose entire purpose is sending a reader to the maintained code sent
    them to the wrong file, with a link to match.
    """
    import re

    rows = re.findall(
        r"\|\s*`(\w+)`\s*\|\s*\[`(src/\w+\.py)`\]\((\.\./src/\w+\.py)\)\s*\|", SOURCE
    )
    assert len(rows) >= 3, rows

    for function, named, link in rows:
        assert link == f"../{named}", (function, named, link)
        path = ROOT / named
        assert path.exists(), (function, named)
        assert re.search(rf"^def {function}\b", path.read_text(), re.MULTILINE), (
            f"the table sends a reader to {named} for {function}, "
            "which is not defined there"
        )


def test_every_function_the_notebook_claims_to_duplicate_exists_once():
    """A row for a function defined in two modules would be ambiguous."""
    import re

    for function in re.findall(r"\|\s*`(\w+)`\s*\|\s*\[`src/", SOURCE):
        homes = [
            path.name
            for path in sorted((ROOT / "src").glob("*.py"))
            if re.search(rf"^def {function}\b", path.read_text(), re.MULTILINE)
        ]
        assert len(homes) == 1, (function, homes)
