"""Bad settings should fail at import, not three steps later."""

import importlib

import pytest

import src.config


def _reload_with(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(src.config, "load_dotenv", lambda *a, **k: None, raising=False)
    return importlib.reload(src.config)


@pytest.fixture(autouse=True)
def _restore():
    yield
    importlib.reload(src.config)


def test_the_shipped_defaults_are_valid():
    """Whatever else changes, the committed defaults must pass."""
    importlib.reload(src.config)._validate()


def test_overlap_at_or_above_chunk_size_is_rejected(monkeypatch):
    """RecursiveCharacterTextSplitter loops forever on this, with no error."""
    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        _reload_with(monkeypatch, CHUNK_SIZE="100", CHUNK_OVERLAP="100")


def test_overlap_above_chunk_size_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        _reload_with(monkeypatch, CHUNK_SIZE="100", CHUNK_OVERLAP="500")


def test_a_zero_top_k_is_rejected(monkeypatch):
    """k=0 retrieves nothing, so every answer is ungrounded but looks fine."""
    with pytest.raises(ValueError, match="TOP_K"):
        _reload_with(monkeypatch, TOP_K="0")


def test_a_negative_chunk_size_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="CHUNK_SIZE"):
        _reload_with(monkeypatch, CHUNK_SIZE="-1", CHUNK_OVERLAP="-5")


def test_a_zero_dimension_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="EMBED_DIM"):
        _reload_with(monkeypatch, EMBED_DIM="0")


def test_a_zero_question_limit_is_rejected(monkeypatch):
    """It would reject every question, including valid ones."""
    with pytest.raises(ValueError, match="MAX_QUESTION_CHARS"):
        _reload_with(monkeypatch, MAX_QUESTION_CHARS="0")


def test_a_zero_timeout_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="GROQ_TIMEOUT"):
        _reload_with(monkeypatch, GROQ_TIMEOUT="0")


def test_an_out_of_range_port_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="PORT"):
        _reload_with(monkeypatch, PORT="70000")


def test_the_message_lists_every_problem_at_once(monkeypatch):
    """One run should not mean one fix at a time."""
    with pytest.raises(ValueError) as caught:
        _reload_with(monkeypatch, TOP_K="0", EMBED_DIM="0", GROQ_TIMEOUT="-1")
    message = str(caught.value)
    assert "TOP_K" in message and "EMBED_DIM" in message and "GROQ_TIMEOUT" in message


def test_valid_overrides_still_load(monkeypatch):
    config = _reload_with(monkeypatch, TOP_K="5", CHUNK_SIZE="800", CHUNK_OVERLAP="100")
    assert (config.TOP_K, config.CHUNK_SIZE, config.CHUNK_OVERLAP) == (5, 800, 100)
