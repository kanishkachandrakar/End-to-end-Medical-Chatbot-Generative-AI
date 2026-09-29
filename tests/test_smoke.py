"""The suite must stay importable without the heavy runtime dependencies."""


def test_pure_helpers_import_without_langchain():
    """src.text is the module the other tests build on; it must stay light."""
    import src.text

    assert hasattr(src.text, "strip_reasoning")
    assert hasattr(src.text, "chunk_ids")
