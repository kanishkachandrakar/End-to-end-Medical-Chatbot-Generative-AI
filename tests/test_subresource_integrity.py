"""The page loads four files from three CDNs, and trusts them.

The CSP has to permit code.jquery.com, stackpath.bootstrapcdn.com and
use.fontawesome.com, because that is where the page's script and stylesheets
come from -- so the policy cannot be what stops a swapped file running in the
page. Subresource integrity is: the browser hashes what arrives and refuses it
if the hash disagrees.

All three tags carry one today. Nothing required them to, and the failure if
one is dropped is silent in exactly the wrong direction -- the page keeps
working, and keeps working if the file is replaced.

The hashes were checked against the live CDNs when this was written: jQuery
3.3.1 (sha256), bootstrap 4.1.3 (sha384) and Font Awesome 5.5.0 (sha384) all
matched. That check is not repeated here, because the suite is designed to pass
with no network -- a test that fetches from three CDNs fails on a train.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEMPLATE = (ROOT / "templates" / "chat.html").read_text()

# Any tag pulling something over https: <script src>, <link href>.
_EXTERNAL = re.compile(r"<(script|link)\b[^>]*\b(?:src|href)=\"(https://[^\"]+)\"[^>]*>")

# Hash algorithms a browser accepts for integrity. sha1 and md5 are not here
# and are not accepted; a tag carrying one is a tag with no integrity.
_ACCEPTED = ("sha256-", "sha384-", "sha512-")


def _external_tags():
    """(tag name, url, full tag) for everything the page fetches over https."""
    return [
        (match.group(1), match.group(2), match.group(0))
        for match in _EXTERNAL.finditer(TEMPLATE)
    ]


def test_the_external_tags_were_found():
    """Every assertion below iterates them, so none would fail on an empty list."""
    tags = _external_tags()
    assert len(tags) == 3, [url for _, url, _ in tags]
    assert {url.split("/")[2] for _, url, _ in tags} == {
        "code.jquery.com",
        "stackpath.bootstrapcdn.com",
        "use.fontawesome.com",
    }


def test_every_external_subresource_declares_its_hash():
    for name, url, tag in _external_tags():
        assert "integrity=" in tag, f"<{name}> from {url} has no integrity"


def test_every_hash_uses_an_algorithm_a_browser_accepts():
    """An unrecognised algorithm is ignored, so the tag is unprotected."""
    for _name, url, tag in _external_tags():
        declared = re.search(r"integrity=\"([^\"]+)\"", tag)
        assert declared, url
        assert declared.group(1).startswith(_ACCEPTED), (url, declared.group(1))


def test_every_external_subresource_is_fetched_anonymously():
    """Integrity is only enforced on a request made with CORS.

    Without crossorigin the response is opaque, the browser cannot read it to
    hash it, and the integrity attribute is skipped -- so a tag with a hash and
    no crossorigin looks protected and is not.
    """
    for name, url, tag in _external_tags():
        assert 'crossorigin="anonymous"' in tag, f"<{name}> from {url}"


def test_local_assets_are_not_given_hashes():
    """They are fingerprinted by content in the URL instead.

    A hash in the template would be a second copy of the same fact, pinned at
    whatever the file said when someone last pasted it.
    """
    for line in TEMPLATE.splitlines():
        if "url_for('static'" in line:
            assert "integrity=" not in line, line.strip()
