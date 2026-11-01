"""Offline checks for incident triage in the agent, the graph, and the page.

No network, no API key, and no real model. The sample incident is fake.
"""

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

import agent
import app
import graph_agent
import incident_triage
from test_agent import FakeClient, make_asker, tool_reply


@pytest.fixture(autouse=True)
def reset_page():
    app.pending_action = None
    app.last_view = None
    yield
    app.pending_action = None
    app.last_view = None


def hypothesis(result, title):
    matches = [item for item in result["hypotheses"] if item["title"] == title]
    assert len(matches) == 1
    return matches[0]


def section_after(page, start, until):
    begin = page.index(start)
    end = page.index(until, begin)
    return page[begin:end]


def test_sample_triage_links_evidence_and_runs_nothing(capsys):
    answer = agent.run_incident_triage(agent.SAMPLE_INCIDENT)
    out = capsys.readouterr().out
    triage = agent.last_trace.triage
    worker = hypothesis(triage, "Reconstruction worker failure")
    saturation = hypothesis(triage, "Processing resource saturation")
    storage = hypothesis(triage, "Storage or image-read failure")
    assert worker["supporting_evidence_ids"] == ["log-1"]
    assert saturation["supporting_evidence_ids"] == ["metric-1"]
    assert storage["supporting_evidence_ids"] == []
    assert worker["status"] == "unverified"
    assert triage["confirmed_root_cause"] is None
    assert triage["action_mode"] == "simulation_only"
    assert "Supporting evidence: log-1." in answer
    assert "log-1 from worker.log: worker exception during reconstruction" in answer
    assert "Supporting evidence: metric-1." in answer
    assert "Supporting evidence: none." in answer
    assert "Confirmed root cause: not confirmed." in answer
    assert "No action was run." in answer
    assert agent.last_trace.stop_reason == agent.INCIDENT_TRIAGED
    assert agent.last_trace.sources == []
    assert "[simulated]" not in out
    assert "needs your approval" not in out
    assert "Practice triage: no model was called." in out
    log = agent.LOG_FILE.read_text()
    assert "triage_incident" in log
    assert "log-1" in log


def test_the_tool_path_matches_on_the_loop_and_the_graph(capsys):
    payload = {
        "incident_id": "INC-1",
        "symptom": "Scan reconstruction unavailable",
        "impact": "One practice customer blocked",
        "evidence": [{"id": "log-1", "source": "worker.log", "observation": "worker exception"}],
        "approved": True,
        "action": "restart the process 'runaway-worker'",
    }

    def client():
        return FakeClient(tool_reply("triage_incident", payload))

    ask = make_asker("y")
    agent_answer = agent.run_agent("please triage this incident", client=client(), ask=ask)
    agent_triage = agent.last_trace.triage
    graph_answer = graph_agent.run_graph("please triage this incident", client=client(), ask=ask)
    out = capsys.readouterr().out
    assert ask.prompts == []
    assert agent.last_trace.stop_reason == graph_agent.last_trace.stop_reason == agent.INCIDENT_TRIAGED
    assert graph_answer == agent_answer
    assert agent_triage["hypotheses"][0]["supporting_evidence_ids"] == ["log-1"]
    assert graph_agent.last_trace.triage["hypotheses"][0]["supporting_evidence_ids"] == ["log-1"]
    assert graph_agent.last_trace.node_path == ["PLAN", "TRIAGE"]
    assert graph_agent.last_trace.steps == 1
    assert "[simulated]" not in out
    assert "needs your approval" not in out
    assert "not confirmed" in agent_answer.lower()


def test_triage_sample_flag_does_not_call_a_model(monkeypatch, capsys):
    def boom():
        raise AssertionError("the model client must not be built")

    monkeypatch.setattr(agent, "make_client", boom)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    answer = agent.main(["--triage-sample"])
    out = capsys.readouterr().out
    assert "Stop reason: INCIDENT_TRIAGED" in out
    assert "log-1" in answer
    assert "metric-1" in answer
    assert "[simulated]" not in out


def test_triage_flag_reads_json_and_rejects_bad_json(capsys):
    raw = json.dumps({
        "incident_id": "INC-9",
        "symptom": "Queue delay",
        "impact": "Practice only",
        "evidence": [{"id": "metric-9", "source": "queue.metrics", "observation": "queue latency"}],
    })
    answer = agent.main(["--triage", raw])
    assert "metric-9" in answer
    assert hypothesis(agent.last_trace.triage, "Processing resource saturation")["supporting_evidence_ids"] == ["metric-9"]

    agent.main(["--triage", "not-json"])
    out = capsys.readouterr().out
    assert "INCIDENT_INVALID" in out
    assert "[simulated]" not in out


def test_triage_flags_do_not_run_the_forced_action(monkeypatch, capsys):
    monkeypatch.setattr(agent, "make_client", lambda: (_ for _ in ()).throw(AssertionError("no model")))
    agent.main(["--triage-sample", "--force-action"])
    out = capsys.readouterr().out
    assert "INCIDENT_INVALID" in out
    assert "[simulated]" not in out
    assert "needs your approval" not in out


def test_a_write_style_next_step_is_not_run(monkeypatch, capsys):
    monkeypatch.setattr(
        incident_triage,
        "PATTERNS",
        (("Forced restart idea", ("worker",), "restart the process 'runaway-worker'"),),
    )
    data = {
        "incident_id": "INC-1",
        "symptom": "Worker noise",
        "impact": "Practice only",
        "evidence": [{"id": "log-1", "source": "worker.log", "observation": "worker exception"}],
    }
    answer = agent.run_incident_triage(data)
    out = capsys.readouterr().out
    assert "was not run" in answer
    assert "was not queued for approval" in answer
    assert "[simulated]" not in out
    assert "needs your approval" not in out
    assert agent.last_trace.stop_reason == agent.INCIDENT_TRIAGED


def test_invalid_evidence_is_a_named_error_and_does_not_run():
    data = {
        "incident_id": "INC-1",
        "symptom": "Worker noise",
        "impact": "Practice only",
        "evidence": [
            {"id": "log-1", "source": "worker.log", "observation": "worker exception"},
            {"id": "log-1", "source": "worker.log", "observation": "worker exception again"},
        ],
    }
    with pytest.raises(agent.AgentError) as caught:
        agent.run_incident_triage(data)
    assert caught.value.name == "INCIDENT_INVALID"
    assert agent.last_trace is None


def test_triage_does_not_bypass_the_step_cap(monkeypatch):
    monkeypatch.setattr(agent, "MAX_STEPS", 0)
    client = FakeClient(tool_reply("triage_incident", agent.SAMPLE_INCIDENT))
    answer = agent.run_agent("please triage this incident", client=client)
    assert agent.last_trace.stop_reason == agent.MAX_STEPS_REACHED
    assert client.calls == []
    assert "log-1" not in answer

    monkeypatch.setattr(graph_agent, "MAX_STEPS", 0)
    client = FakeClient(tool_reply("triage_incident", agent.SAMPLE_INCIDENT))
    graph_agent.run_graph("please triage this incident", client=client)
    assert graph_agent.last_trace.stop_reason == agent.MAX_STEPS_REACHED
    assert client.calls == []
    assert graph_agent.last_trace.node_path[-1] == "IDK"


def test_a_weak_search_still_stops_before_triage(monkeypatch):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: [
        {"source": "memory_leaks.txt", "section": "High swap usage", "text": "See swap.", "score": 0.08},
    ])
    client = FakeClient(
        tool_reply("search_runbooks", {"query": "swap"}),
        tool_reply("triage_incident", agent.SAMPLE_INCIDENT),
    )
    answer = agent.run_agent("the box is swapping", client=client)
    assert agent.last_trace.stop_reason == agent.WEAK_MATCH_ONLY
    assert "INC-SAMPLE-1" not in answer
    assert len(client.calls) == 1


def test_the_page_links_evidence_from_the_form_without_a_model(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("triage must not search or build a model client")

    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setattr(agent, "make_client", boom)
    monkeypatch.setattr(agent.retrieve, "search", boom)
    view = app.triage_request({
        "incident_id": ["INC-1"],
        "symptom": ["Scan reconstruction unavailable"],
        "impact": ["One practice customer blocked"],
        "timeline": ["10:00 acquisition complete"],
        "evidence": ["log-1 | worker.log | worker exception\nmetric-1 | queue.metrics | queue latency"],
    })
    assert view["error"] is None
    assert view["pending_action"] is None
    assert view["stop_reason"] == agent.INCIDENT_TRIAGED
    assert view["triage"]["confirmed_root_cause"] is None
    page = app.render(view)
    worker = section_after(page, "Reconstruction worker failure", "Processing resource saturation")
    storage = section_after(page, "Storage or image-read failure", "We are investigating")
    assert "log-1" in worker
    assert "metric-1" not in worker
    assert "worker exception" in worker
    assert "Supporting evidence: none" in storage
    assert "Confirmed root cause: not confirmed." in page
    assert "No action was run." in page
    assert "Approve and simulate" not in page
    assert "SIGTERM" not in page


def test_triage_text_is_shown_as_text_not_html():
    view = app.triage_request({
        "incident_id": ["INC-1"],
        "symptom": ["<script>alert(1)</script>"],
        "impact": ["<b>blocked</b>"],
        "timeline": [""],
        "evidence": ["log-1 | worker.log | <img src=x onerror=alert(1)> worker exception"],
    })
    page = app.render(view)
    assert view["stop_reason"] == agent.INCIDENT_TRIAGED
    assert "<script>alert" not in page
    assert "<b>blocked</b>" not in page
    assert "<img src" not in page
    assert "&lt;script&gt;" in page
    assert "&lt;b&gt;blocked&lt;/b&gt;" in page
    assert "&lt;img src" in page


def test_a_bad_evidence_line_is_an_error_and_does_not_approve():
    view = app.triage_request({
        "incident_id": ["INC-1"],
        "symptom": ["Worker noise"],
        "impact": ["Practice only"],
        "evidence": ["only-one-field"],
    })
    assert view["error"]["name"] == "INCIDENT_INVALID"
    assert view["pending_action"] is None
    page = app.render(view)
    assert "INCIDENT_INVALID" in page
    assert "Approve and simulate" not in page


def _post(port, path, fields):
    data = urllib.parse.urlencode(fields).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data)
    with urllib.request.urlopen(request) as response:
        return response.status, response.read().decode()


def test_the_http_triage_route_links_evidence_and_does_not_run(monkeypatch, capsys):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setattr(agent, "make_client", lambda: (_ for _ in ()).throw(AssertionError("no real client")))
    monkeypatch.setattr(agent.retrieve, "search", lambda q: (_ for _ in ()).throw(AssertionError("no search")))
    server = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as home:
            page = home.read().decode()
        assert "Triage the practice sample" in page
        assert "Triage this incident" in page
        assert "Approve and simulate" not in page

        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/triage")
            raise AssertionError("GET /triage should not succeed")
        except urllib.error.HTTPError as err:
            assert err.code == 404

        status, triaged = _post(port, "/triage", {"sample": "1"})
        assert status == 200
        assert "Stop reason: INCIDENT_TRIAGED" in triaged
        assert "Reconstruction worker failure" in triaged
        assert "log-1" in triaged
        assert "metric-1" in triaged
        assert "Confirmed root cause: not confirmed." in triaged
        assert "No action was run." in triaged
        assert "Approve and simulate" not in triaged
        assert "SIGTERM" not in triaged
        assert "[simulated]" not in capsys.readouterr().out

        status, manual = _post(port, "/triage", {
            "incident_id": "INC-2",
            "symptom": "Scan reconstruction unavailable",
            "impact": "Practice only",
            "evidence": "log-2 | worker.log | worker exception",
        })
        assert status == 200
        worker = section_after(manual, "Reconstruction worker failure", "Storage or image-read failure")
        assert "log-2" in worker
        assert "Approve and simulate" not in manual
    finally:
        server.shutdown()
        server.server_close()


def test_triage_clears_a_waiting_restart_so_an_old_click_cannot_run(monkeypatch, capsys):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: [{
        "source": "rb.txt",
        "section": "High CPU on web servers",
        "text": "Restart the cpu worker.",
        "score": 0.4,
    }])
    view = app.ask("restart the process 'runaway-worker'", client=app.DemoClient())
    waiting = view["pending_action"]
    triaged = app.triage_request({"sample": ["1"]})
    assert triaged["pending_action"] is None
    assert app.pending_action is None
    stale = app.decide("yes", waiting)
    assert stale["error"]["name"] == "STALE_APPROVAL"
    assert "[simulated]" not in capsys.readouterr().out
