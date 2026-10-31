"""Tests for the local page. A fake model or the practice model. No API key, no network."""

import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

import agent
import app
from test_agent import CPU, FakeClient, text_reply, tool_reply


@pytest.fixture(autouse=True)
def reset_page():
    app.pending_action = None
    app.last_view = None
    yield
    app.pending_action = None
    app.last_view = None


def test_practice_mode_answers_from_the_runbooks_without_a_key(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    def boom():
        raise AssertionError("the real client must not be built when no key is set")

    monkeypatch.setattr(agent, "make_client", boom)
    view = app.ask("web server CPU is at 95 percent")
    assert view["mode"] == "practice"
    assert view["error"] is None
    assert view["stop_reason"] == agent.ANSWERED
    assert view["answer"].split("\n")[0].startswith("1.")
    assert view["sources"]
    assert "High CPU" in view["sources"][0]["section"]
    assert "swap" not in view["answer"].lower()  # a weak neighbor is not turned into a step
    assert "does not know" not in view["stop_plain"].lower()
    html = app.render(view)
    assert "Why it stopped" in html
    assert view["stop_plain"] in html
    assert "Stop reason: ANSWERED" in html
    assert "Sources" in html
    assert "step by step" in html
    assert "XAI_API_KEY" not in view["answer"]


def test_an_unknown_question_says_it_does_not_know(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setattr(agent.retrieve, "search", lambda q: [])
    view = app.ask("how do I bake a sourdough pancake", client=app.DemoClient())
    assert view["stop_reason"] == agent.NO_RELEVANT_RUNBOOK
    assert view["answer"].startswith("I don't know")
    assert view["sources"] == []
    assert "will not guess" in view["stop_plain"]
    html = app.render(view)
    assert "No sources" in html
    assert "Approve and simulate" not in html


def test_only_weak_matches_are_an_i_dont_know_on_the_page(monkeypatch):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: [
        {"source": "memory_leaks.txt", "section": "High swap usage", "text": "See swap.", "score": 0.08},
    ])
    view = app.ask("the box is swapping", client=app.DemoClient())
    assert view["stop_reason"] == agent.WEAK_MATCH_ONLY
    assert view["answer"].startswith("I don't know")
    assert view["sources"] == []
    assert "too weak" in view["stop_plain"]


def test_a_restart_waits_for_a_click_and_does_not_run(monkeypatch, capsys):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: [CPU])
    view = app.ask("restart the process 'runaway-worker'", client=app.DemoClient())
    out = capsys.readouterr().out
    assert view["stop_reason"] == agent.ACTION_REQUESTED
    assert view["pending_action"] == "restart the process 'runaway-worker'"
    assert "Nothing has run" in view["answer"]
    assert "[simulated]" not in out
    html = app.render(view)
    assert "Approve and simulate" in html
    assert "Do not run" in html
    assert "This action has not run" in html
    assert "SIGTERM" not in html


def test_do_not_run_denies_and_approve_only_simulates(monkeypatch, capsys):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: [CPU])
    view = app.ask("restart the runaway worker", client=app.DemoClient())
    action = view["pending_action"]
    assert action == "restart the process 'runaway-worker'"
    refused = app.decide("no", action)
    assert refused["decision"] == "You said no. Nothing was run."
    assert "[simulated]" not in capsys.readouterr().out
    assert app.pending_action is None

    view = app.ask("restart the process 'runaway-worker'", client=app.DemoClient())
    approved = app.decide("yes", view["pending_action"])
    assert approved["decision_ok"] is True
    assert "Simulated only" in approved["decision"]
    assert "No real process was touched" in approved["decision"]
    assert "pid 1842" in approved["decision"]
    assert "waiting for you" not in approved["answer"].lower()
    html = app.render(approved)
    assert "Approve and simulate" not in html  # the button is gone after the click
    assert "Nothing on a real machine was changed" in html


def test_a_word_other_than_yes_does_not_approve(monkeypatch, capsys):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: [CPU])
    view = app.ask("restart the process 'runaway-worker'", client=app.DemoClient())
    refused = app.decide("true", view["pending_action"])
    assert "Nothing was run" in refused["decision"]
    assert "[simulated]" not in capsys.readouterr().out


def test_an_old_approve_click_cannot_run_a_different_action(monkeypatch):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: [CPU])
    view = app.ask("restart the process 'runaway-worker'", client=app.DemoClient())
    forged = app.decide("yes", "restart the process 'sshd'")
    assert forged["error"]["name"] == "STALE_APPROVAL"
    assert app.pending_action == view["pending_action"]  # the real one is still waiting
    # the forged action did not replace it, and sshd was not restarted
    stale = app.decide("yes", "restart the process 'sshd'")
    assert stale["error"]["name"] == "STALE_APPROVAL"


def test_live_mode_uses_the_real_client_and_hides_the_key(monkeypatch):
    secret = "sk-live-test-secret-do-not-leak"
    monkeypatch.setenv("XAI_API_KEY", secret)
    monkeypatch.setattr(agent.retrieve, "search", lambda q: [CPU])
    client = FakeClient(
        tool_reply("search_runbooks", {"query": "cpu"}),
        text_reply("1. Restart the worker that is using the CPU."),
    )
    monkeypatch.setattr(agent, "make_client", lambda: client)
    view = app.ask("cpu high")
    html = app.render(view)
    assert view["mode"] == "live"
    assert secret not in html
    assert "Live mode" in html
    assert client.calls
    assert view["stop_reason"] == agent.ANSWERED
    assert "Restart the worker" in view["answer"]


def test_a_key_that_leaks_into_an_error_is_hidden(monkeypatch):
    secret = "sk-live-test-secret-do-not-leak"
    monkeypatch.setenv("XAI_API_KEY", secret)
    shown = app.public_error(agent.AgentError("MODEL_ERROR", f"failed with {secret}"))
    assert secret not in shown["message"]
    assert "[key hidden]" in shown["message"]


def test_runbook_text_is_shown_as_text_not_html(monkeypatch):
    evil = {
        "source": "x.txt",
        "section": "<script>alert(1)</script>",
        "text": "<b>boom</b> Restart the worker.",
        "score": 0.9,
    }
    monkeypatch.setattr(agent.retrieve, "search", lambda q: [evil])
    view = app.ask("cpu", client=app.DemoClient())
    html = app.render(view)
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html
    assert "<b>boom</b>" not in html
    assert "&lt;b&gt;boom&lt;/b&gt;" in html


def test_the_page_reports_max_steps_in_plain_words(monkeypatch):
    monkeypatch.setattr(agent, "MAX_STEPS", 1)
    monkeypatch.setattr(agent.retrieve, "search", lambda q: [CPU])
    view = app.ask("cpu is high", client=app.DemoClient())
    assert view["stop_reason"] == agent.MAX_STEPS_REACHED
    assert "every step" in view["stop_plain"]
    html = app.render(view)
    assert "Stop reason: MAX_STEPS_REACHED" in html
    assert view["stop_plain"] in html
    assert "What the copilot did" in html


def test_an_empty_question_is_a_named_error_on_the_page():
    view = app.ask("   ", client=app.DemoClient())
    assert view["error"]["name"] == "EMPTY_REQUEST"
    html = app.render(view)
    assert "EMPTY_REQUEST" in html
    assert "Approve and simulate" not in html


def _post(port, path, fields):
    data = urllib.parse.urlencode(fields).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data)
    with urllib.request.urlopen(request) as response:
        return response.status, response.read().decode()


def test_the_http_page_requires_a_click_before_the_simulation(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setattr(agent, "make_client", lambda: (_ for _ in ()).throw(AssertionError("no real client")))
    monkeypatch.setattr(agent.retrieve, "search", lambda q: [CPU])
    server = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as home:
            page = home.read().decode()
        assert home.status == 200
        assert "Ask the copilot" in page
        assert "SIGTERM" not in page
        assert "Approve and simulate" not in page

        status, asked = _post(port, "/ask", {"question": "restart the process 'runaway-worker'"})
        assert status == 200
        assert "This action has not run" in asked
        assert "Approve and simulate" in asked
        assert "SIGTERM" not in asked
        assert "Stop reason: ACTION_REQUESTED" in asked

        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/decide")
            raise AssertionError("GET /decide should not succeed")
        except urllib.error.HTTPError as err:
            assert err.code == 404
        assert "[simulated]" not in asked

        status, refused = _post(port, "/decide", {
            "choice": "no",
            "action": "restart the process 'runaway-worker'",
        })
        assert status == 200
        assert "You said no. Nothing was run." in refused
        assert "SIGTERM" not in refused

        _post(port, "/ask", {"question": "restart the process 'runaway-worker'"})
        status, approved = _post(port, "/decide", {
            "choice": "yes",
            "action": "restart the process 'runaway-worker'",
        })
        assert status == 200
        assert "Simulated only" in approved
        assert "No real process was touched" in approved
        assert "pid 2107" in approved

        status, again = _post(port, "/decide", {
            "choice": "yes",
            "action": "restart the process 'runaway-worker'",
        })
        assert "STALE_APPROVAL" in again
    finally:
        server.shutdown()
        server.server_close()
