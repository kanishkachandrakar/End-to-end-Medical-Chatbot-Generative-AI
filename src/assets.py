"""Cache-busting for the static files.

One concern, three parts that have to agree: a cache lifetime long enough to be
worth having, URLs that change when a file does, and a key that does not change
when the file has not. Keeping them together means the lifetime is never set
without the fingerprinting that makes it safe.
"""

import functools
import hashlib
import pathlib

from flask import Flask

# A year. Only safe because the URLs below carry a content hash.
CACHE_SECONDS = 31_536_000


@functools.lru_cache(maxsize=32)
def digest(path: str) -> str:
    """Short content hash of a file, or "" if it cannot be read."""
    try:
        return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:12]
    except OSError:
        return ""


def static_version(app: Flask, filename: str) -> str:
    """A cache key for a static file, derived from its contents.

    Flask's url_for does not fingerprint static files, so a long cache lifetime
    would otherwise leave a visitor holding a stale chat.js.

    Content, not modification time: an mtime is a property of the checkout
    rather than the file. A deploy copies the files and gives them new mtimes,
    so every release would invalidate assets that had not changed -- and two
    replicas built separately would advertise different URLs for identical
    bytes. A hash is the same everywhere the file is.
    """
    if not app.static_folder:
        return ""
    return digest(str(pathlib.Path(app.static_folder, filename)))


def install(app: Flask) -> None:
    """Set the cache lifetime and make url_for fingerprint static URLs."""
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = CACHE_SECONDS

    @app.url_defaults
    def _fingerprint_static(endpoint: str, values: dict) -> None:
        if endpoint == "static" and "filename" in values:
            version = static_version(app, values["filename"])
            if version:
                values["v"] = version
