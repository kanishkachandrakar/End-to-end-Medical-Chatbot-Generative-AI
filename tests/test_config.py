"""src/config.py holds the defaults the whole app is calibrated against."""

import importlib

import pytest

import src.config


@pytest.fixture
def reload_config(monkeypatch):
    """Reload config with a patched environment, then restore the real module."""

    def _reload(**env):
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        # load_dotenv would otherwise put the developer's .env back on top
        monkeypatch.setattr(src.config, "load_dotenv", lambda *a, **k: None, raising=False)
        return importlib.reload(src.config)

    yield _reload
    importlib.reload(src.config)


def test_defaults_match_the_deployed_index():
    """These four are not free choices -- the live Pinecone index has them."""
    assert src.config.INDEX_NAME == "medicalbot"
    assert src.config.EMBED_DIM == 384
    assert src.config.EMBED_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
    assert src.config.PINECONE_REGION == "us-east-1"


def test_chunking_defaults():
    assert src.config.CHUNK_SIZE == 500
    assert src.config.CHUNK_OVERLAP == 20
    assert src.config.CHUNK_OVERLAP < src.config.CHUNK_SIZE


def test_retrieval_and_serving_defaults():
    assert src.config.TOP_K == 3
    assert src.config.MAX_QUESTION_CHARS == 500
    assert src.config.PORT == 8080
    assert src.config.GROQ_TIMEOUT == 60.0
    assert src.config.LOG_LEVEL == "INFO"


def test_numeric_settings_are_not_strings():
    """os.environ.get returns str; forgetting the int() breaks arithmetic."""
    for value in (
        src.config.EMBED_DIM,
        src.config.CHUNK_SIZE,
        src.config.CHUNK_OVERLAP,
        src.config.TOP_K,
        src.config.MAX_QUESTION_CHARS,
        src.config.PORT,
    ):
        assert isinstance(value, int)
    assert isinstance(src.config.GROQ_TIMEOUT, float)


def test_environment_overrides_are_applied(reload_config):
    config = reload_config(PINECONE_INDEX="other", TOP_K="7", GROQ_TIMEOUT="12.5")
    assert config.INDEX_NAME == "other"
    assert config.TOP_K == 7
    assert config.GROQ_TIMEOUT == 12.5


def test_log_level_is_upper_cased(reload_config):
    """logging accepts 'DEBUG', not 'debug'."""
    assert reload_config(LOG_LEVEL="debug").LOG_LEVEL == "DEBUG"


def test_no_secrets_are_read_here(reload_config):
    """Importing config must not require keys; only the callers need them."""
    config = reload_config()
    assert not hasattr(config, "PINECONE_API_KEY")
    assert not hasattr(config, "GROQ_API_KEY")
