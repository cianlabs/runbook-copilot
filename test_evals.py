"""Tests for the eval harness itself, using tiny fake data (no real runbooks, no model)."""
import agent
import evals
from evals import Outcome, Scenario

FAKE_RESULTS = [{"source": "x.txt", "section": "Toasters on fire", "text": "Unplug the toaster.", "score": 0.9}]


def fake_search(monkeypatch, results=FAKE_RESULTS):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: results)


def search_scenario(**overrides):
    base = dict(name="toaster", request="my toaster is smoking", model_tool="search_runbooks",
                model_args={"query": "toaster smoking"}, expect_tool="search_runbooks",
                expect_in_query=("toaster",), expect_in_results="unplug")
    base.update(overrides)
    return Scenario(**base)


def test_a_passing_scenario_passes(monkeypatch):
    fake_search(monkeypatch)
    outcome = evals.run_scenario(search_scenario())
    assert outcome.passed and outcome.failures == []


def test_wrong_tool_fails_with_a_named_failure(monkeypatch):
    fake_search(monkeypatch)
    outcome = evals.run_scenario(search_scenario(expect_tool="execute_action"))
    assert not outcome.passed and outcome.failures == ["WRONG_TOOL"]


def test_off_topic_query_and_missing_result_are_named(monkeypatch):
    fake_search(monkeypatch)
    outcome = evals.run_scenario(search_scenario(expect_in_query=("kettle",), expect_in_results="defrost"))
    assert outcome.failures == ["QUERY_OFF_TOPIC", "MISSING_EXPECTED_RESULT"]


def test_a_blank_query_is_named_no_query(monkeypatch):
    fake_search(monkeypatch)
    outcome = evals.run_scenario(search_scenario(model_args={"query": "   "}))
    assert outcome.failures == ["NO_QUERY"]


def test_agent_errors_are_caught_and_named(monkeypatch):
    fake_search(monkeypatch)
    # execute_action without an 'action' makes the agent raise BAD_TOOL_ARGS
    scenario = Scenario("bad-args", "do it", "execute_action", {}, "execute_action")
    outcome = evals.run_scenario(scenario)
    assert not outcome.passed and "AGENT_ERROR:BAD_TOOL_ARGS" in outcome.failures


def test_denied_action_scenario_passes_and_never_executes(monkeypatch):
    ran = []
    monkeypatch.setitem(agent.TOOLS, "execute_action", lambda *a, **k: ran.append(1))
    scenario = Scenario("restart", "restart it", "execute_action",
                        {"action": "restart the process 'runaway-worker'"}, "execute_action", ("restart",))
    outcome = evals.run_scenario(scenario)  # default asker answers "n"
    assert outcome.passed and ran == []


def test_the_harness_restores_the_real_agent_functions(monkeypatch):
    fake_search(monkeypatch)
    before = (agent.call_tool, agent.run_gated_action)
    evals.run_scenario(search_scenario())
    assert (agent.call_tool, agent.run_gated_action) == before


def test_summary_when_everything_passes():
    assert evals.summarize([Outcome("a", True), Outcome("b", True)]) == "2/2 passed"


def test_summary_names_the_failing_scenarios():
    text = evals.summarize([Outcome("a", True), Outcome("b", False, ["WRONG_TOOL"]),
                            Outcome("c", False, ["NO_QUERY", "QUERY_OFF_TOPIC"])])
    lines = text.splitlines()
    assert lines[0] == "1/3 passed"
    assert "  - b: WRONG_TOOL" in lines and "  - c: NO_QUERY, QUERY_OFF_TOPIC" in lines


def test_the_shipped_scenarios_are_8_to_10_and_have_unique_names():
    names = [s.name for s in evals.SCENARIOS]
    assert 8 <= len(names) <= 10 and len(names) == len(set(names))
    assert {s.expect_tool for s in evals.SCENARIOS} == {"search_runbooks", "execute_action"}


def test_all_shipped_scenarios_pass_with_the_fake_model():
    outcomes = [evals.run_scenario(s) for s in evals.SCENARIOS]
    assert [o.name for o in outcomes if not o.passed] == []


def test_failures_show_what_the_agent_actually_did(monkeypatch):
    fake_search(monkeypatch)
    outcome = evals.run_scenario(search_scenario(expect_in_results="defrost"))
    assert "query='toaster smoking'" in outcome.detail and "Toasters on fire" in outcome.detail
    text = evals.summarize([outcome])
    assert "saw: tool=search_runbooks" in text


def test_a_reply_without_a_tool_call_is_described():
    assert "no tool was called" in evals.describe([], "Just restart it.\n")
