"""Cache-busting for the static files.

One concern, three parts that have to agree: a cache lifetime long enough to be
worth having, URLs that change when a file does, and a key that does not change
when the file has not. Keeping them together means the lifetime is never set
without the fingerprinting that makes it safe.
"""

import functools
import hashlib
import pathlib

from flask import Flask, Response, request

# A year. Only safe because the URLs below carry a content hash.
_CACHE_SECONDS = 31_536_000


@functools.lru_cache(maxsize=32)
def _digest_of_readable(path: str) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:12]


def digest(path: str) -> str:
    """Short content hash of a file, or "" if it cannot be read.

    Only a success is memoised. A failure is not a property of the file --
    it is a file being replaced mid-deploy, or a storage layer blinking --
    and caching one would de-fingerprint that asset for the lifetime of the
    process, long after reading it started working again.
    """
    try:
        return _digest_of_readable(path)
    except OSError:
        return ""


# Kept on the public name: callers -- and two tests -- clear the cache
# through digest(), and splitting the memoised half out should not move that.
digest.cache_clear = _digest_of_readable.cache_clear  # type: ignore[attr-defined]


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
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = _CACHE_SECONDS

    @app.url_defaults
    def _fingerprint_static(endpoint: str, values: dict) -> None:
        if endpoint == "static" and "filename" in values:
            version = static_version(app, values["filename"])
            if version:
                values["v"] = version

    @app.after_request
    def _only_cache_what_carries_a_hash(response: Response) -> Response:
        """A year is safe for a URL carrying a content hash, and only then.

        Three ways to be served a static file with no hash in the URL, and
        all three were getting the year from SEND_FILE_MAX_AGE_DEFAULT:
        /favicon.ico, which is a route rather than the static endpoint and
        so is never fingerprinted; a request someone typed or kept; and a
        file whose digest could not be read, where static_version returns ""
        and url_for emits a bare URL. In each case a replacement cannot
        reach anyone holding the old copy, for a year, with no way to
        invalidate it -- which is the exact failure the fingerprinting
        exists to prevent.

        no-cache rather than no-store: the file can still be held, it just
        has to be revalidated, which costs a conditional request and
        usually answers 304.
        """
        if "max-age" not in response.headers.get("Cache-Control", ""):
            return response
        if not request.args.get("v"):
            response.headers["Cache-Control"] = "no-cache"
        return response
