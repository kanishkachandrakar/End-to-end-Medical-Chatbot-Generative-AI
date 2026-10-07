"""Headers on every response, and a policy that matches what the page loads."""

import pathlib
import re

import pytest

from src.headers import CSP, SECURITY_HEADERS
from src.webapp import create_app

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAGE_SOURCES = (
    (ROOT / "templates" / "chat.html").read_text()
    + (ROOT / "static" / "chat.js").read_text()
)


class StubChain:
    def invoke(self, payload):
        return {"answer": "ok"}


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


def test_the_policy_still_permits_only_what_the_page_uses():
    """The directives added here must not smuggle in a new origin."""
    import re

    allowed = set(re.findall(r"https://[a-z0-9.-]+", CSP))
    used = set(re.findall(r"https://[a-z0-9.-]+", PAGE_SOURCES))
    assert allowed == used
