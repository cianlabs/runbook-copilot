"""Space handlers. Practice model only. No API key, no Gradio server, no network."""

import agent
import app
import space.app as space_app


def _session():
    return space_app.new_session()


def test_practice_answer_cites_a_runbook(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    session = _session()
    error, stop_plain, stop_reason, answer, sources, _trace, pending, decision, triage, session = (
        space_app.on_ask("web server CPU is at 95 percent", session)
    )
    assert error == ""
    assert stop_reason == agent.ANSWERED
    assert answer.split("\n")[0].startswith("1.")
    assert "High CPU" in sources
    assert "sample_runbook.txt" in sources
    assert pending == ""
    assert decision == ""
    assert triage == ""
    assert "will not guess" not in stop_plain
    assert "XAI_API_KEY" not in answer


def test_unknown_question_says_i_dont_know(monkeypatch):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: [])
    error, _plain, stop_reason, answer, sources, _trace, pending, _decision, _triage, _session_out = (
        space_app.on_ask("how do I bake a sourdough pancake", _session())
    )
    assert error == ""
    assert stop_reason == agent.NO_RELEVANT_RUNBOOK
    assert answer.startswith("I don't know")
    assert sources.startswith("No sources")
    assert pending == ""


def test_restart_waits_and_only_the_session_action_can_be_approved(monkeypatch, capsys):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: [])
    session = _session()
    _fields = space_app.on_ask("restart the runaway worker", session)
    pending = _fields[6]
    assert pending == "restart the process 'runaway-worker'"
    assert session.pending_action == pending
    assert "[simulated]" not in capsys.readouterr().out

    # A different session cannot approve this visitor's action.
    other = _session()
    stale = space_app.on_decide("yes", other)
    assert "STALE_APPROVAL" in stale[0]
    assert "[simulated]" not in capsys.readouterr().out
    assert session.pending_action == pending

    refused = space_app.on_decide("no", session)
    assert "Nothing was run" in refused[7]
    assert session.pending_action is None
    assert "[simulated]" not in capsys.readouterr().out

    session = _session()
    space_app.on_ask("restart the process 'runaway-worker'", session)
    approved = space_app.on_decide("yes", session)
    assert "Simulated only" in approved[7]
    assert "No real process was touched" in approved[7]
    assert session.pending_action is None


def test_two_sessions_do_not_share_a_waiting_action(monkeypatch):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: [])
    first = _session()
    second = _session()
    space_app.on_ask("restart the process 'runaway-worker'", first)
    space_app.on_ask("web server CPU is at 95 percent", second)
    assert first.pending_action == "restart the process 'runaway-worker'"
    assert second.pending_action is None
    assert app.pending_action is None


def test_practice_sample_triage_does_not_run_an_action():
    session = _session()
    session.pending_action = "restart the process 'runaway-worker'"
    fields = space_app.on_triage("", "", "", "", "", session, sample=True)
    triage = fields[8]
    assert "not confirmed" in triage
    assert "No action was run" in triage
    assert session.pending_action is None
    assert fields[6] == ""


def test_mode_banner_does_not_include_a_key(monkeypatch):
    secret = "sk-live-test-secret-do-not-leak"
    monkeypatch.setenv("XAI_API_KEY", secret)
    text = space_app.mode_banner()
    assert "Live mode" in text
    assert secret not in text
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    assert "Practice mode" in space_app.mode_banner()


def test_build_is_skipped_when_gradio_is_missing(monkeypatch):
    monkeypatch.setattr(space_app, "gr", None)
    try:
        space_app.build()
    except RuntimeError as err:
        assert "Gradio is not installed" in str(err)
    else:
        raise AssertionError("build() should refuse to start without Gradio")
