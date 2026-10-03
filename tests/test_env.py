"""The startup check that decides whether the app can run at all."""

import pytest

from src.env import require_env


def test_returns_the_values_when_present(monkeypatch):
    monkeypatch.setenv("ALPHA", "a")
    monkeypatch.setenv("BETA", "b")
    assert require_env("ALPHA", "BETA") == {"ALPHA": "a", "BETA": "b"}


def test_raises_naming_the_missing_variable(monkeypatch):
    monkeypatch.delenv("ALPHA", raising=False)
    with pytest.raises(RuntimeError, match="ALPHA"):
        require_env("ALPHA")


def test_names_every_missing_variable_at_once(monkeypatch):
    """A deploy should not need one restart per missing secret."""
    monkeypatch.delenv("ALPHA", raising=False)
    monkeypatch.delenv("BETA", raising=False)
    with pytest.raises(RuntimeError) as caught:
        require_env("ALPHA", "BETA")
    assert "ALPHA" in str(caught.value) and "BETA" in str(caught.value)


def test_an_empty_value_counts_as_missing(monkeypatch):
    """Copying .env.example leaves 'PINECONE_API_KEY=' set but useless."""
    monkeypatch.setenv("ALPHA", "")
    with pytest.raises(RuntimeError, match="ALPHA"):
        require_env("ALPHA")


def test_whitespace_only_counts_as_missing(monkeypatch):
    """A pasted secret that came through as a newline is not a key."""
    monkeypatch.setenv("ALPHA", "   ")
    with pytest.raises(RuntimeError, match="ALPHA"):
        require_env("ALPHA")


def test_the_message_says_what_to_do(monkeypatch):
    monkeypatch.delenv("ALPHA", raising=False)
    with pytest.raises(RuntimeError, match=r"\.env\.example"):
        require_env("ALPHA")


def test_a_present_variable_is_not_reported(monkeypatch):
    monkeypatch.setenv("ALPHA", "a")
    monkeypatch.delenv("BETA", raising=False)
    with pytest.raises(RuntimeError) as caught:
        require_env("ALPHA", "BETA")
    assert "ALPHA" not in str(caught.value)
