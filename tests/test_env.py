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


class TestKeyShapes:
    """Catching a swapped pair of keys, which otherwise fails per-request."""

    def test_correct_keys_draw_no_complaint(self):
        from src.env import warn_on_suspicious_keys

        assert warn_on_suspicious_keys(
            {"PINECONE_API_KEY": "pcsk_abc", "GROQ_API_KEY": "gsk_abc"}
        ) == []

    def test_a_swap_is_named_as_a_swap(self):
        from src.env import warn_on_suspicious_keys

        complaints = warn_on_suspicious_keys(
            {"PINECONE_API_KEY": "gsk_abc", "GROQ_API_KEY": "pcsk_abc"}
        )
        assert len(complaints) == 2
        assert all("swapped" in c for c in complaints)

    def test_an_unrecognisable_value_is_reported_as_possibly_truncated(self):
        from src.env import warn_on_suspicious_keys

        complaints = warn_on_suspicious_keys({"PINECONE_API_KEY": "abc123"})
        assert len(complaints) == 1
        assert "truncated" in complaints[0]
        assert "pcsk_" in complaints[0]

    def test_an_absent_key_is_not_complained_about(self):
        """require_env already reports that, and more usefully."""
        from src.env import warn_on_suspicious_keys

        assert warn_on_suspicious_keys({}) == []
        assert warn_on_suspicious_keys({"GROQ_API_KEY": ""}) == []

    def test_an_unknown_variable_is_ignored(self):
        from src.env import warn_on_suspicious_keys

        assert warn_on_suspicious_keys({"OPENAI_API_KEY": "sk-abc"}) == []

    def test_it_only_warns_and_never_raises(self):
        """A provider changing its key format must not stop the app starting."""
        from src.env import warn_on_suspicious_keys

        assert isinstance(warn_on_suspicious_keys({"GROQ_API_KEY": "nonsense"}), list)
