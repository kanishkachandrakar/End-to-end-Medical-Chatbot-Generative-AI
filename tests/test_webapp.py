"""The request path, driven through Flask's test client with a stub chain."""

import pytest

from src.config import MAX_QUESTION_CHARS
from src.errors import RATE_LIMITED, UNAVAILABLE
from src.webapp import NO_ANSWER, NO_QUESTION, create_app


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


def test_get_works_as_well_as_post(client_for):
    """The route allows both verbs, so both must read the parameter."""
    chain = StubChain()
    response = client_for(chain).get("/get?msg=hello")
    assert response.status_code == 200
    assert chain.calls == [{"input": "hello"}]


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
    assert response.data.decode() == UNAVAILABLE
    assert b"Traceback" not in response.data


def test_a_rate_limit_is_passed_through_as_429(client_for):
    class RateLimited(Exception):
        status_code = 429

    chain = StubChain(raises=RateLimited())
    response = client_for(chain).post("/get", data={"msg": "q"})
    assert response.status_code == 429
    assert response.data.decode() == RATE_LIMITED


def test_every_reply_is_plain_text(client_for):
    """The browser must not be told to parse any of these as HTML."""
    client = client_for(StubChain())
    for response in (
        client.get("/healthz"),
        client.post("/get", data={"msg": "q"}),
        client.post("/get", data={"msg": ""}),
    ):
        assert response.mimetype == "text/plain", response.status_code
