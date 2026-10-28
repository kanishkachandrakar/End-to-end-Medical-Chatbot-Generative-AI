"""The states /healthz reports, and everything that has to recognise them.

Three places have to agree: src/webapp.py emits the state, app.py decides when,
static/chat.js shows a banner for it, and DEPLOY.md tells a reader what it
means. A new state that only two of them know about is a state the page says
nothing about -- which is how the degraded case went unnoticed on the page for
as long as it did.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
WEBAPP = (ROOT / "src" / "webapp.py").read_text()
APP = (ROOT / "app.py").read_text()
SCRIPT = (ROOT / "static" / "chat.js").read_text()
DEPLOY = (ROOT / "DEPLOY.md").read_text()

# Emitted as a literal in the /healthz reply, or assigned to `unavailable`.
LITERAL = re.compile(r'degraded \{APP_REVISION\} ([a-z-]+)')
ASSIGNED = re.compile(r'unavailable = "([a-z-]+)"')


def _states():
    states = set(LITERAL.findall(WEBAPP)) | set(ASSIGNED.findall(APP))
    return states


def test_the_states_were_found():
    """Everything below compares against this set; empty would pass them all."""
    assert len(_states()) >= 2, _states()


def test_every_state_is_recognised_by_the_page():
    missing = {state for state in _states() if state not in SCRIPT}
    assert not missing, (
        f"/healthz can report {missing} and static/chat.js does not mention them, "
        "so the page would say nothing about it"
    )


def test_every_state_is_documented_for_whoever_deploys_it():
    missing = {state for state in _states() if state not in DEPLOY}
    assert not missing, f"undocumented in DEPLOY.md: {missing}"


def test_the_page_invents_no_state_of_its_own():
    """A banner for a state the app cannot emit is dead code that reads live."""
    mentioned = {
        state
        for state in re.findall(r'indexOf\("([a-z-]+)"\)', SCRIPT)
        if "-" in state
    }
    assert mentioned <= _states(), f"chat.js checks for {mentioned - _states()}"


def test_liveness_is_separate_from_readiness():
    """A restart cannot fix either state, so /livez must not report them."""
    livez = WEBAPP[WEBAPP.index('@app.route("/livez")') :]
    livez = livez[: livez.index('@app.route("/healthz")')]
    for state in _states():
        assert state not in livez
    assert "index_size" not in livez and "unavailable" not in livez


def test_the_states_are_log_and_url_safe():
    """They go into a plain-text body and a log line; keep them boring."""
    for state in _states():
        assert re.fullmatch(r"[a-z][a-z-]*[a-z]", state), state


def test_every_degraded_state_answers_a_failure_status():
    """chat.js puts all of its banner logic in .fail(), which relies on this.

    /healthz answers 200 only when there is nothing to report, so the front
    end reads the reason off a non-2xx response. A degraded state that
    answered 2xx would take the banner away silently -- the page would look
    fine and answer nothing -- so the contract is asserted here rather than
    left implied by a jQuery callback.
    """
    from src.webapp import create_app
    from tests.stubs import StubChain

    for state in (
        {"index_size": 0},
        {"unavailable": "chain-unavailable"},
        {"unavailable": "chain-unavailable", "index_size": 0},
    ):
        response = create_app(StubChain(), **state).test_client().get("/healthz")
        assert response.status_code >= 400, (state, response.status_code)
        assert "degraded" in response.data.decode(), state


def test_a_healthy_reply_names_no_degraded_reason():
    """The removed .done() branch searched a 2xx body for "index-empty"."""
    from src.webapp import create_app
    from tests.stubs import StubChain

    for state in ({"index_size": 5860}, {"index_size": None}):
        response = create_app(StubChain(), **state).test_client().get("/healthz")
        assert response.status_code == 200, state
        body = response.data.decode()
        assert "index-empty" not in body, body
        assert "chain-unavailable" not in body, body
        assert body.startswith("ok "), body


def test_the_script_has_no_done_handler_for_healthz():
    """Because a 2xx /healthz has nothing a banner could report.

    The branch that was there searched a 200 body for "index-empty", which a
    200 body cannot contain -- a healthy reply is "ok <revision> vectors=<n>".
    It carried its own copy of the banner wording, so the text existed twice
    and only one of the two could ever be shown.
    """
    healthz_call = SCRIPT.split('url: "/healthz"')[1].split('$("#messageArea")')[0]
    assert ".done(" not in healthz_call, healthz_call[:200]
