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


def test_a_negative_rate_limit_is_rejected(monkeypatch):
    """0 disables the limit on purpose; -1 would disable it by accident."""
    with pytest.raises(ValueError, match="RATE_LIMIT_PER_MINUTE"):
        _reload_with(monkeypatch, RATE_LIMIT_PER_MINUTE="-1")


def test_a_zero_rate_limit_is_allowed(monkeypatch):
    config = _reload_with(monkeypatch, RATE_LIMIT_PER_MINUTE="0")
    assert config.RATE_LIMIT_PER_MINUTE == 0


def test_a_body_ceiling_below_the_question_limit_is_rejected(monkeypatch):
    """Otherwise a question inside the documented limit is refused as too big."""
    with pytest.raises(ValueError, match="MAX_CONTENT_BYTES"):
        _reload_with(monkeypatch, MAX_CONTENT_BYTES="100", MAX_QUESTION_CHARS="500")


def test_an_equal_body_ceiling_is_also_rejected(monkeypatch):
    """Form encoding means the body is always larger than the question."""
    with pytest.raises(ValueError, match="MAX_CONTENT_BYTES"):
        _reload_with(monkeypatch, MAX_CONTENT_BYTES="500", MAX_QUESTION_CHARS="500")


def test_a_ceiling_that_only_clears_ascii_is_rejected(monkeypatch):
    """The check used to be `>`, which this passes and should not.

    501 bytes holds a 500-character question only while every character is
    an unreserved ASCII one. A single accented letter costs two bytes and
    six once urlencoded, so the question is refused for being oversized
    while being exactly the documented length.
    """
    with pytest.raises(ValueError, match="MAX_CONTENT_BYTES"):
        _reload_with(monkeypatch, MAX_CONTENT_BYTES="501", MAX_QUESTION_CHARS="500")


def test_the_worst_case_question_fits_the_ceiling_it_demands(monkeypatch):
    """The arithmetic in config.py, checked against urlencode rather than
    against itself."""
    import urllib.parse

    import src.config as config

    for character in ("a", "?", "é", "医", "\U0001f600"):
        body = urllib.parse.urlencode(
            {"msg": character * config.MAX_QUESTION_CHARS}
        ).encode()
        assert len(body) <= config.MAX_CONTENT_BYTES, (character, len(body))
        needed = (
            config.MAX_QUESTION_CHARS * config._WORST_BYTES_PER_CHAR
            + config._FORM_FIELD_OVERHEAD
        )
        assert len(body) <= needed, (character, len(body), needed)


def test_the_shipped_defaults_satisfy_the_cross_checks():
    import src.config as config

    assert config.MAX_CONTENT_BYTES >= (
        config.MAX_QUESTION_CHARS * config._WORST_BYTES_PER_CHAR
        + config._FORM_FIELD_OVERHEAD
    )
    assert config.RATE_LIMIT_PER_MINUTE >= 0


@pytest.mark.parametrize("empty", ["", "   ", "\t"])
def test_an_empty_host_is_rejected(monkeypatch, empty):
    """HOST="" binds every interface, which is the opposite of the default.

    Flask and Werkzeug treat an empty host as 0.0.0.0, so a HOST that got
    through as blank -- an unset shell variable expanded into a .env line,
    a trailing `HOST=` -- would quietly undo the loopback default and serve
    the network, with the real keys loaded and /get needing none.

    This rule was added with the HOST setting and without a test, in a
    function where the other nine rules all have one. Coverage was pointing
    at the line.
    """
    with pytest.raises(ValueError, match="HOST"):
        _reload_with(monkeypatch, HOST=empty)


def test_a_real_host_is_accepted(monkeypatch):
    """So the check above is rejecting blankness rather than the variable."""
    for host in ("127.0.0.1", "0.0.0.0", "localhost", "::1"):
        assert _reload_with(monkeypatch, HOST=host).HOST == host


def test_every_setting_validate_rejects_has_a_test_that_provokes_it():
    """The HOST rule went untested because nothing connected the two lists.

    Each rule in _validate() names the setting it is about, and each
    rejection test says which message it expects with `match=`. So the
    settings named in the one have to appear in the other.

    The limit: this catches a rule about a setting no test provokes, which
    is the case that happened. It does not catch a second rule about a
    setting already covered -- comparing counts would, and the first
    version of this test did exactly that and caught nothing, because the
    count it compared against already had slack in it.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent
    config = (root / "src" / "config.py").read_text()
    body = config.split("def _validate()")[1].split("\ndef ")[0]

    # The setting each message is about: the first uppercase name inside the
    # problems.append(...) that the rule raises.
    rejected = set()
    for append in re.findall(r"problems\.append\((.*?)\)\n", body, re.DOTALL):
        names = re.findall(r"\b([A-Z][A-Z_]{3,})\b", append)
        if names:
            rejected.add(names[0])

    tests = (root / "tests" / "test_config_validation.py").read_text()
    provoked = set(re.findall(r'pytest\.raises\(ValueError, match="([^"]+)"', tests))

    assert rejected, "no rules found in _validate"
    unprovoked = sorted(
        name
        for name in rejected
        if not any(name in expected for expected in provoked)
    )
    assert not unprovoked, (
        f"_validate rejects these and nothing provokes it: {unprovoked}"
    )
