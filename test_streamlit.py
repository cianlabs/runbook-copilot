"""Streamlit handlers. Practice model only. No API key, no Streamlit server, no network."""

import os

import agent
import app
import streamlit_app


def _store():
    return {}


def test_practice_answer_cites_a_runbook(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    store = _store()
    session = streamlit_app.ensure_session(store)
    error, stop_plain, stop_reason, answer, sources, _trace, pending, decision, triage, _out = (
        streamlit_app.on_ask("web server CPU is at 95 percent", session)
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
    error, _plain, stop_reason, answer, sources, _trace, pending, _decision, _triage, _out = (
        streamlit_app.on_ask("how do I bake a sourdough pancake", streamlit_app.ensure_session(_store()))
    )
    assert error == ""
    assert stop_reason == agent.NO_RELEVANT_RUNBOOK
    assert answer.startswith("I don't know")
    assert sources.startswith("No sources")
    assert pending == ""


def test_restart_waits_and_only_this_session_can_be_approved(monkeypatch, capsys):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: [])
    session = streamlit_app.ensure_session(_store())
    fields = streamlit_app.on_ask("restart the runaway worker", session)
    assert fields[6] == "restart the process 'runaway-worker'"
    assert session.pending_action == fields[6]
    assert "[simulated]" not in capsys.readouterr().out

    other = streamlit_app.ensure_session(_store())
    stale = streamlit_app.on_approve(other)
    assert "STALE_APPROVAL" in stale[0]
    assert "[simulated]" not in capsys.readouterr().out
    assert session.pending_action == fields[6]

    refused = streamlit_app.on_refuse(session)
    assert "Nothing was run" in refused[7]
    assert session.pending_action is None
    assert "[simulated]" not in capsys.readouterr().out

    session = streamlit_app.ensure_session(_store())
    streamlit_app.on_ask("restart the process 'runaway-worker'", session)
    approved = streamlit_app.on_approve(session)
    assert "Simulated only" in approved[7]
    assert "No real process was touched" in approved[7]
    assert session.pending_action is None
    assert approved[6] == ""


def test_two_sessions_do_not_share_a_waiting_action(monkeypatch):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: [])
    first = streamlit_app.ensure_session(_store())
    second = streamlit_app.ensure_session(_store())
    streamlit_app.on_ask("restart the process 'runaway-worker'", first)
    streamlit_app.on_ask("web server CPU is at 95 percent", second)
    assert first.pending_action == "restart the process 'runaway-worker'"
    assert second.pending_action is None
    assert app.pending_action is None


def test_practice_sample_triage_does_not_run_an_action():
    session = streamlit_app.ensure_session(_store())
    session.pending_action = "restart the process 'runaway-worker'"
    fields = streamlit_app.on_triage("", "", "", "", "", session, sample=True)
    assert "not confirmed" in fields[8]
    assert "No action was run" in fields[8]
    assert session.pending_action is None
    assert fields[6] == ""


def test_secret_turns_on_live_mode_without_showing_the_key(monkeypatch):
    secret = "sk-live-test-secret-do-not-leak"
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    streamlit_app.apply_streamlit_secret({"XAI_API_KEY": secret})
    banner = streamlit_app.mode_banner()
    assert os.environ.get("XAI_API_KEY") == secret
    assert "Live mode" in banner
    assert secret not in banner
    assert secret not in streamlit_app.mode_banner()


def test_missing_secret_stays_in_practice_mode(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    class Boom(dict):
        def __contains__(self, key):
            raise RuntimeError("no secrets file")

    streamlit_app.apply_streamlit_secret(Boom())
    streamlit_app.apply_streamlit_secret({})
    assert os.environ.get("XAI_API_KEY", "") == ""
    assert "Practice mode" in streamlit_app.mode_banner()


def test_an_existing_environment_key_is_not_replaced(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "already-set")
    streamlit_app.apply_streamlit_secret({"XAI_API_KEY": "sk-other-secret"})
    assert os.environ["XAI_API_KEY"] == "already-set"
    assert "sk-other-secret" not in streamlit_app.mode_banner()


def test_main_refuses_to_start_without_streamlit(monkeypatch):
    monkeypatch.setattr(streamlit_app, "st", None)
    try:
        streamlit_app.main()
    except RuntimeError as err:
        assert "Streamlit is not installed" in str(err)
    else:
        raise AssertionError("main() should refuse to start without Streamlit")
