"""The request path, driven through Flask's test client with a stub chain."""

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
    assert response.data == b"ok"
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
