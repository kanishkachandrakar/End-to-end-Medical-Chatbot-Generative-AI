"""The request path, driven through Flask's test client with a stub chain."""

import pathlib

import pytest

from src.config import MAX_QUESTION_CHARS
from src.errors import RATE_LIMITED, UNAVAILABLE
from src.ratelimit import TokenBucket
from src.webapp import NO_ANSWER, NO_QUESTION, TOO_BUSY, create_app


class StubChain:
    """Records what it was asked and returns whatever it was told to."""

    def __init__(self, answer="Acne is a skin condition.", raises=None):
        self.answer = answer
        self.raises = raises
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        if self.raises is not None:
            raise self.raises
        return {"answer": self.answer}


@pytest.fixture
def client_for():
    def _build(chain):
        app = create_app(chain)
        app.config.update(TESTING=True)
        return app.test_client()

    return _build


def test_index_serves_the_chat_page(client_for):
    response = client_for(StubChain()).get("/")
    assert response.status_code == 200
    assert b'id="messageArea"' in response.data


def test_healthz_is_plain_ok(client_for):
    response = client_for(StubChain()).get("/healthz")
    assert response.status_code == 200
    assert response.data.decode().startswith("ok")
    assert response.mimetype == "text/plain"


def test_healthz_does_not_touch_the_chain(client_for):
    """An uptime monitor must not spend a Groq call every thirty seconds."""
    chain = StubChain()
    client_for(chain).get("/healthz")
    assert chain.calls == []


def test_a_question_is_answered(client_for):
    chain = StubChain()
    response = client_for(chain).post("/get", data={"msg": "What is acne?"})
    assert response.status_code == 200
    assert response.data == b"Acne is a skin condition."
    assert chain.calls == [{"input": "What is acne?"}]


def test_reasoning_is_stripped_before_the_answer_is_sent(client_for):
    chain = StubChain(answer="<think>weighing it up</think>Final answer.")
    response = client_for(chain).post("/get", data={"msg": "q"})
    assert response.data == b"Final answer."


def test_the_question_is_trimmed_before_use(client_for):
    chain = StubChain()
    client_for(chain).post("/get", data={"msg": "  padded  "})
    assert chain.calls == [{"input": "padded"}]


@pytest.mark.parametrize("msg", ["", "   "])
def test_an_empty_question_is_rejected_without_calling_the_chain(client_for, msg):
    chain = StubChain()
    response = client_for(chain).post("/get", data={"msg": msg})
    assert response.status_code == 400
    assert response.data.decode() == NO_QUESTION
    assert chain.calls == []


def test_a_missing_field_is_rejected(client_for):
    """request.values would raise a KeyError if this were read directly."""
    response = client_for(StubChain()).post("/get", data={})
    assert response.status_code == 400


def test_get_is_rejected(client_for):
    """Answering spends money and rate limit, so it is not a safe method."""
    chain = StubChain()
    response = client_for(chain).get("/get?msg=hello")
    assert response.status_code == 405
    assert chain.calls == [], "a crawler or prefetch must not reach the model"


def test_a_query_string_on_a_post_is_ignored(client_for):
    """Reading request.form, not request.values, keeps the question out of URLs."""
    chain = StubChain()
    response = client_for(chain).post("/get?msg=from-url", data={"msg": "from-body"})
    assert response.status_code == 200
    assert chain.calls == [{"input": "from-body"}]


def test_an_over_long_question_is_rejected(client_for):
    chain = StubChain()
    oversized = "x" * (MAX_QUESTION_CHARS + 1)
    response = client_for(chain).post("/get", data={"msg": oversized})
    assert response.status_code == 413
    assert str(MAX_QUESTION_CHARS) in response.data.decode()
    assert chain.calls == [], "an oversized prompt must not reach the model"


def test_a_question_exactly_at_the_limit_is_allowed(client_for):
    chain = StubChain()
    response = client_for(chain).post("/get", data={"msg": "x" * MAX_QUESTION_CHARS})
    assert response.status_code == 200
    assert chain.calls


def test_an_empty_answer_falls_back_to_a_sentence(client_for):
    """A reply that is only reasoning would otherwise render an empty bubble."""
    chain = StubChain(answer="<think>all reasoning</think>")
    response = client_for(chain).post("/get", data={"msg": "q"})
    assert response.data.decode() == NO_ANSWER


def test_a_response_without_an_answer_key_does_not_500(client_for):
    class Shapeless:
        calls = []

        def invoke(self, payload):
            return {}

    response = client_for(Shapeless()).post("/get", data={"msg": "q"})
    assert response.status_code == 200
    assert response.data.decode() == NO_ANSWER


def test_a_chain_failure_becomes_a_readable_reply(client_for):
    chain = StubChain(raises=ValueError("pinecone is down"))
    response = client_for(chain).post("/get", data={"msg": "q"})
    assert response.status_code == 502
    assert response.data.decode().startswith(UNAVAILABLE)
    assert b"Traceback" not in response.data


def test_a_rate_limit_is_passed_through_as_429(client_for):
    class RateLimited(Exception):
        status_code = 429

    chain = StubChain(raises=RateLimited())
    response = client_for(chain).post("/get", data={"msg": "q"})
    assert response.status_code == 429
    assert response.data.decode().startswith(RATE_LIMITED)


def test_every_reply_is_plain_text(client_for):
    """The browser must not be told to parse any of these as HTML."""
    client = client_for(StubChain())
    for response in (
        client.get("/healthz"),
        client.post("/get", data={"msg": "q"}),
        client.post("/get", data={"msg": ""}),
    ):
        assert response.mimetype == "text/plain", response.status_code


def test_the_limiter_refuses_a_question_once_the_budget_is_spent(client_for):
    chain = StubChain()
    app = create_app(chain, limiter=TokenBucket(capacity=1, rate=0.0001))
    client = app.test_client()

    assert client.post("/get", data={"msg": "first"}).status_code == 200
    refused = client.post("/get", data={"msg": "second"})
    assert refused.status_code == 429
    assert refused.data.decode() == TOO_BUSY
    assert len(chain.calls) == 1, "the refused question must not reach the model"


def test_a_rejected_question_does_not_spend_budget(client_for):
    """An empty question is free, so a typo cannot exhaust the demo's quota."""
    chain = StubChain()
    app = create_app(chain, limiter=TokenBucket(capacity=1, rate=0.0001))
    client = app.test_client()

    assert client.post("/get", data={"msg": ""}).status_code == 400
    assert client.post("/get", data={"msg": "real question"}).status_code == 200


def test_healthz_is_never_rate_limited(client_for):
    chain = StubChain()
    app = create_app(chain, limiter=TokenBucket(capacity=1, rate=0.0001))
    client = app.test_client()
    client.post("/get", data={"msg": "spend it"})
    assert client.get("/healthz").status_code == 200


def test_the_input_carries_the_server_side_limit(client_for):
    """Without this the browser lets you type a question the server refuses."""
    page = client_for(StubChain()).get("/")
    assert f'maxlength="{MAX_QUESTION_CHARS}"'.encode() in page.data


def test_every_response_carries_a_request_id(client_for):
    client = client_for(StubChain())
    for response in (client.get("/"), client.post("/get", data={"msg": "q"})):
        assert len(response.headers["X-Request-Id"]) == 8


def test_each_request_gets_a_different_id(client_for):
    client = client_for(StubChain())
    first = client.post("/get", data={"msg": "a"}).headers["X-Request-Id"]
    second = client.post("/get", data={"msg": "b"}).headers["X-Request-Id"]
    assert first != second


def test_a_failure_reply_quotes_the_request_id(client_for):
    """So a user can report 'ref ab12cd34' and it can be found in the log."""
    chain = StubChain(raises=ValueError("boom"))
    response = client_for(chain).post("/get", data={"msg": "q"})
    reference = response.headers["X-Request-Id"]
    assert f"(ref {reference})" in response.data.decode()


def test_a_successful_answer_is_not_cluttered_with_the_id(client_for):
    """It is in the header either way; the chat bubble should stay clean."""
    response = client_for(StubChain()).post("/get", data={"msg": "q"})
    assert "ref" not in response.data.decode()


def test_robots_txt_disallows_everything(client_for):
    response = client_for(StubChain()).get("/robots.txt")
    assert response.status_code == 200
    assert response.mimetype == "text/plain"
    assert "User-agent: *" in response.data.decode()
    assert "Disallow: /" in response.data.decode()


def test_robots_txt_does_not_touch_the_chain(client_for):
    chain = StubChain()
    client_for(chain).get("/robots.txt")
    assert chain.calls == []


def test_the_time_taken_is_logged(client_for, caplog):
    """A slow demo is unexplainable without this; it is most of the latency."""
    with caplog.at_level("INFO"):
        client_for(StubChain()).post("/get", data={"msg": "q"})
    assert any("answered in" in record.getMessage() for record in caplog.records)


def test_a_failure_logs_how_long_it_took_to_fail(client_for, caplog):
    chain = StubChain(raises=ValueError("boom"))
    with caplog.at_level("ERROR"):
        client_for(chain).post("/get", data={"msg": "q"})
    assert any("failed after" in record.getMessage() for record in caplog.records)


def test_an_unknown_path_answers_in_plain_text(client_for):
    """Flask's default 404 is an HTML document; the UI would paste it verbatim."""
    response = client_for(StubChain()).get("/no-such-page")
    assert response.status_code == 404
    assert response.mimetype == "text/plain"
    assert b"<html" not in response.data.lower()


def test_a_wrong_method_answers_in_plain_text(client_for):
    response = client_for(StubChain()).get("/get?msg=hi")
    assert response.status_code == 405
    assert response.mimetype == "text/plain"
    assert b"<html" not in response.data.lower()


def test_error_pages_keep_the_security_headers(client_for):
    response = client_for(StubChain()).get("/no-such-page")
    assert response.headers.get("X-Content-Type-Options") == "nosniff"


def test_healthz_names_the_revision(client_for, monkeypatch):
    """So a deploy can be confirmed without opening the Space's log."""
    import importlib

    import src.webapp as webapp

    monkeypatch.setenv("APP_REVISION", "deadbee")
    reloaded = importlib.reload(webapp)
    try:
        client = reloaded.create_app(StubChain()).test_client()
        assert client.get("/healthz").data == b"ok deadbee"
    finally:
        monkeypatch.delenv("APP_REVISION", raising=False)
        importlib.reload(webapp)


def test_healthz_still_says_ok_without_a_revision(client_for):
    body = client_for(StubChain()).get("/healthz").data.decode()
    assert body.startswith("ok ")


def test_an_enormous_body_is_refused(client_for):
    """MAX_QUESTION_CHARS only applies after parsing; this refuses it earlier."""
    from src.config import MAX_CONTENT_BYTES

    chain = StubChain()
    response = client_for(chain).post(
        "/get", data={"msg": "x" * (MAX_CONTENT_BYTES + 1024)}
    )
    assert response.status_code == 413
    assert chain.calls == []


def test_the_body_ceiling_is_well_above_the_question_limit(client_for):
    """It is a backstop, not the limit a user should ever meet."""
    from src.config import MAX_CONTENT_BYTES, MAX_QUESTION_CHARS

    assert MAX_CONTENT_BYTES > MAX_QUESTION_CHARS * 10


def test_static_urls_are_fingerprinted(client_for):
    """Needed before a long cache lifetime is safe to set."""
    page = client_for(StubChain()).get("/").data.decode()
    assert "chat.js?v=" in page
    assert "style.css?v=" in page


def test_the_fingerprint_changes_when_the_file_does(tmp_path, monkeypatch):
    """Otherwise an edit would never reach a visitor who has the old copy."""
    import os

    from src.webapp import _static_version, create_app

    app = create_app(StubChain())
    asset = pathlib.Path(app.static_folder) / "chat.js"
    before = _static_version(app, "chat.js")
    os.utime(asset, (asset.stat().st_atime, asset.stat().st_mtime + 60))
    try:
        assert _static_version(app, "chat.js") != before
    finally:
        os.utime(asset, (asset.stat().st_atime, asset.stat().st_mtime - 60))


def test_a_missing_static_file_does_not_break_the_page(client_for):
    """url_for is called during render; an OSError there would be a 500."""
    from src.webapp import _static_version, create_app

    app = create_app(StubChain())
    assert _static_version(app, "no-such-file.js") == ""


def test_static_assets_are_sent_with_a_long_cache_lifetime(client_for):
    response = client_for(StubChain()).get("/static/chat.js")
    assert response.status_code == 200
    assert "max-age=" in response.headers.get("Cache-Control", "")
