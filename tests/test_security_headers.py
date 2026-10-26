"""Headers on every response, and a policy that matches what the page loads."""

import pathlib
import re

import pytest

from src.headers import CSP, SECURITY_HEADERS
from src.webapp import create_app
from tests.stubs import StubChain

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAGE_SOURCES = (
    (ROOT / "templates" / "chat.html").read_text()
    + (ROOT / "static" / "chat.js").read_text()
)


@pytest.fixture
def client():
    return create_app(StubChain()).test_client()


@pytest.mark.parametrize("header", sorted(SECURITY_HEADERS))
def test_the_header_is_on_the_page(client, header):
    assert client.get("/").headers.get(header) == SECURITY_HEADERS[header]


@pytest.mark.parametrize("header", sorted(SECURITY_HEADERS))
def test_the_header_is_on_the_answer_too(client, header):
    """An answer is the response most worth protecting, not the page."""
    response = client.post("/get", data={"msg": "q"})
    assert response.headers.get(header) == SECURITY_HEADERS[header]


def test_errors_also_carry_the_headers(client):
    assert client.post("/get", data={"msg": ""}).headers.get("X-Content-Type-Options")


def test_every_origin_the_page_loads_is_allowed_by_the_policy():
    """A policy that blocks the stylesheet is worse than no policy at all."""
    origins = set(re.findall(r"https://[a-z0-9.-]+", PAGE_SOURCES))
    for origin in origins:
        assert origin in CSP, f"{origin} is loaded by the page but absent from the CSP"


def test_the_policy_allows_no_origin_the_page_does_not_use():
    """Unused permissions are permissions granted to an attacker."""
    allowed = set(re.findall(r"https://[a-z0-9.-]+", CSP))
    used = set(re.findall(r"https://[a-z0-9.-]+", PAGE_SOURCES))
    assert allowed <= used, f"CSP permits unused origins: {allowed - used}"


# Source expressions that are not origins, so the https:// regex above never
# saw them. Each is a way for an injected tag to fetch or run something, and
# none of them is needed by a page that loads four files from three CDNs.
FORBIDDEN_SOURCES = frozenset(
    {
        "data:",
        "blob:",
        "filesystem:",
        "mediastream:",
        "*",
        "'unsafe-inline'",
        "'unsafe-eval'",
    }
)


def _csp_tokens():
    """(directive, source) for every source expression in the policy."""
    pairs = []
    for directive in CSP.split(";"):
        words = directive.split()
        if not words:
            continue
        pairs.extend((words[0], source) for source in words[1:])
    return pairs


def test_the_directives_were_parsed():
    """Everything below iterates them, so finding none would pass silently."""
    pairs = _csp_tokens()
    assert len(pairs) > 10, pairs
    assert ("img-src", "https://i.ibb.co") in pairs


def test_the_policy_permits_no_non_origin_source_either():
    """`img-src ... data:` sat in the policy with nothing using it.

    Both checks on unused origins matched `https://[a-z0-9.-]+`, so a
    scheme-only source was outside what either could see -- a complete
    sounding pair with a hole the exact width of the thing sitting in it.
    Nothing here loads a data: URI: not the template, chat.js or style.css,
    and not bootstrap 4.1.3 or Font Awesome 5.5.0, whose stylesheets were
    fetched and searched for `url(data:`.

    The policy is tokenised rather than searched. The first version of this
    test asked whether each source appeared anywhere in the page's text and
    skipped the ones that did, which excused `data:` immediately: chat.js
    contains `data: { msg: question }`, an object key. Putting `data:` back
    into the CSP did not fail it.
    """
    permitted = {
        (directive, source)
        for directive, source in _csp_tokens()
        if source in FORBIDDEN_SOURCES
    }
    assert not permitted, f"CSP permits sources nothing uses: {sorted(permitted)}"


def test_inline_script_is_not_permitted():
    """The whole reason chat.js was extracted from the template."""
    assert "unsafe-inline" not in CSP.split("style-src")[0]
    assert "unsafe-eval" not in CSP


def test_the_page_cannot_be_framed():
    assert "frame-ancestors 'none'" in CSP
    assert SECURITY_HEADERS["X-Frame-Options"] == "DENY"


def test_the_default_is_restrictive():
    assert CSP.startswith("default-src 'self'")


def test_plugins_and_frames_are_denied():
    """An injected <object> or <iframe> should have nowhere to point."""
    assert "object-src 'none'" in CSP
    assert "frame-src 'none'" in CSP


def test_device_permissions_are_denied():
    policy = SECURITY_HEADERS["Permissions-Policy"]
    for feature in ("camera", "microphone", "geolocation"):
        assert f"{feature}=()" in policy


def test_the_page_is_cross_origin_isolated():
    assert SECURITY_HEADERS["Cross-Origin-Opener-Policy"] == "same-origin"
