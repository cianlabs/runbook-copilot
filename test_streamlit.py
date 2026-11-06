"""Streamlit handlers. Practice model only. No API key, no Streamlit server, no network."""

import os

import pytest

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


def test_approve_fills_triage_widgets_and_writes_lessons(monkeypatch, capsys):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: [])
    store = _store()
    session = streamlit_app.ensure_session(store)
    streamlit_app.on_ask("restart the runaway worker", session)
    fields = streamlit_app.approve_and_fill(store, session)
    assert "Simulated only" in fields[7]
    assert "No real process was touched" in fields[7]
    assert "not confirmed" in fields[8]
    assert "metric-cpu" in fields[8]
    assert session.pending_action is None
    assert fields[6] == ""
    for key in (
        "triage_incident_id",
        "triage_symptom",
        "triage_impact",
        "triage_timeline",
        "triage_evidence",
    ):
        assert str(store.get(key) or "").strip(), key
    assert store["triage_incident_id"] == "INC-PRACTICE-CPU"
    assert "95" in store["triage_symptom"]
    assert "metric-cpu" in store["triage_evidence"]
    assert "sim-restart" in store["triage_evidence"]
    lessons = session.last_view["lessons"]
    assert "Practice lessons learned" in lessons
    assert "engineering" in lessons.lower()
    assert "not confirmed" in lessons.lower()
    assert "fake restart done" not in lessons.lower()
    assert capsys.readouterr().out.count("Sending SIGTERM") == 1


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


def test_the_visible_form_fills_in_after_approve(monkeypatch):
    """The Streamlit widgets themselves, not only the handler return value."""
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    # A secret test may have copied a key into the environment. Stay in practice mode.
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    app = AppTest.from_file("streamlit_app.py", default_timeout=30)
    app.run()
    app.selectbox[0].select("restart the runaway worker").run()
    app.button[0].click().run()
    assert not app.exception
    approve = next(button for button in app.button if button.label == "Approve and simulate")
    approve.click().run()
    assert not app.exception
    values = {field.label: field.value or "" for field in list(app.text_input) + list(app.text_area)}
    assert values["Incident id"] == "INC-PRACTICE-CPU"
    assert "95 percent" in values["Symptom"]
    assert values["Impact"].strip()
    assert "runaway-worker" in values["Timeline, one note per line"]
    assert "metric-cpu" in values["Evidence, one per line: id | source | observation"]
    assert "sim-restart" in values["Evidence, one per line: id | source | observation"]
    shown = "\n".join(block.value or "" for block in app.text)
    assert "Practice lessons learned" in shown
    assert "not confirmed" in shown.lower()
    assert "fake restart done" not in shown.lower()
    assert all(button.label != "Approve and simulate" for button in app.button)


def test_main_refuses_to_start_without_streamlit(monkeypatch):
    monkeypatch.setattr(streamlit_app, "st", None)
    try:
        streamlit_app.main()
    except RuntimeError as err:
        assert "Streamlit is not installed" in str(err)
    else:
        raise AssertionError("main() should refuse to start without Streamlit")
