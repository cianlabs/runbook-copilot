"""Tests for the box-and-arrow agent.

Each box is tested on its own. Each stop reason is tested as a full walk.
The practice-model tests run the same questions through agent.py and
graph_agent.py and expect the same stop reason.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent
import app
import evals
import graph_agent

CPU = {
    "source": "rb.txt",
    "section": "High CPU on web servers",
    "text": "Restart the offending worker.",
    "score": 0.4,
}
WEAK = {
    "source": "memory_leaks.txt",
    "section": "High swap usage",
    "text": "See which process is using swap.",
    "score": 0.08,
}
RESTART = "restart the process 'runaway-worker'"


def tool_reply(name, arguments, call_id="call_1"):
    call = SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))
    return SimpleNamespace(content=None, tool_calls=[call])


def text_reply(text):
    return SimpleNamespace(content=text, tool_calls=None)


class FakeClient:
    """Scripted model. Records every call and never touches the network."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=self.replies.pop(0))])


class ForeverClient:
    """A model that searches on every turn and would never stop by itself."""

    def __init__(self):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        call = tool_reply("search_runbooks", {"query": f"again {len(self.calls)}"})
        return SimpleNamespace(choices=[SimpleNamespace(message=call)])


def make_asker(*answers):
    prompts = []
    queue = list(answers)

    def ask(prompt=""):
        prompts.append(prompt)
        return queue.pop(0)

    ask.prompts = prompts
    return ask


def patch_search(monkeypatch, results):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: results)


def cited_sources(answer):
    block = answer.split("Sources:", 1)[1]
    return [line[2:] for line in block.splitlines() if line.startswith("- ")]


def mermaid_block(text):
    start = text.index("```mermaid")
    end = text.index("```", start + len("```mermaid"))
    return text[start:end].strip()


# ---- The picture and the six boxes ----------------------------------------

def test_the_diagram_matches_in_the_module_the_readme_and_the_docs():
    module = mermaid_block(graph_agent.__doc__)
    readme = mermaid_block(Path("README.md").read_text())
    guide = mermaid_block(Path("docs/graph.md").read_text())
    assert module == readme == guide
    for name in graph_agent.NODE_NAMES:
        assert name in module


def test_the_graph_has_exactly_the_six_boxes():
    assert set(graph_agent.NODES) == set(graph_agent.NODE_NAMES)
    for name in graph_agent.NODE_NAMES:
        assert callable(graph_agent.NODES[name])


def test_max_steps_is_four_like_the_loop():
    assert graph_agent.MAX_STEPS == 4
    assert agent.MAX_STEPS == 4


def test_only_plan_and_a_fresh_answer_spend_a_step():
    state = graph_agent.new_state("cpu", client=None)
    assert graph_agent.spends_a_step("PLAN", state) is True
    assert graph_agent.spends_a_step("SEARCH", state) is False
    assert graph_agent.spends_a_step("CHECK", state) is False
    assert graph_agent.spends_a_step("IDK", state) is False
    assert graph_agent.spends_a_step("ACTION_GATE", state) is False
    assert graph_agent.spends_a_step("ANSWER", state) is True
    state.has_draft = True
    assert graph_agent.spends_a_step("ANSWER", state) is False


# ---- PLAN ------------------------------------------------------------------

def test_plan_sends_a_question_to_search():
    client = FakeClient(tool_reply("search_runbooks", {"query": "high cpu"}))
    state = graph_agent.new_state("web server CPU is high", client=client)
    state.steps = 1
    assert graph_agent.plan(state) == "SEARCH"
    assert state.arguments == {"query": "high cpu"}
    assert state.has_draft is False
    assert "I should search the runbooks for 'high cpu'." in state.thoughts


def test_plan_sends_an_action_request_to_the_gate():
    # The model also sent approved=True. PLAN must ignore that.
    client = FakeClient(tool_reply("execute_action", {"action": RESTART, "approved": True}))
    state = graph_agent.new_state("restart it", client=client)
    state.steps = 1
    assert graph_agent.plan(state) == "ACTION_GATE"
    assert state.action == RESTART


def test_plan_sends_a_written_answer_to_check():
    client = FakeClient(text_reply("1. Restart the database."))
    state = graph_agent.new_state("database down", client=client)
    state.steps = 1
    assert graph_agent.plan(state) == "CHECK"
    assert state.has_draft is True
    assert state.draft_answer == "1. Restart the database."


def test_plan_rejects_tool_arguments_that_are_not_json():
    call = SimpleNamespace(id="x", function=SimpleNamespace(name="search_runbooks", arguments="not json"))
    client = FakeClient(SimpleNamespace(content=None, tool_calls=[call]))
    state = graph_agent.new_state("cpu", client=client)
    with pytest.raises(agent.AgentError) as caught:
        graph_agent.plan(state)
    assert caught.value.name == "BAD_TOOL_ARGS"


# ---- SEARCH ----------------------------------------------------------------

def test_search_calls_the_existing_tool_and_goes_to_check(monkeypatch):
    seen = []
    monkeypatch.setattr(agent.retrieve, "search", lambda q: seen.append(q) or [CPU])
    state = graph_agent.new_state("cpu", client=None)
    state.arguments = {"query": "runaway worker cpu"}
    state.steps = 1
    assert graph_agent.search(state) == "CHECK"
    assert seen == ["runaway worker cpu"]
    assert state.results == [CPU]


def test_search_with_no_query_is_a_named_failure():
    state = graph_agent.new_state("cpu", client=None)
    state.arguments = {}
    with pytest.raises(agent.AgentError) as caught:
        graph_agent.search(state)
    assert caught.value.name == "BAD_TOOL_ARGS"


# ---- CHECK -----------------------------------------------------------------

def test_check_stops_when_nothing_matches():
    state = graph_agent.new_state("cake", client=None)
    state.results = []
    assert graph_agent.check(state) == "IDK"
    assert state.stop_reason == agent.NO_RELEVANT_RUNBOOK
    assert state.answer_text.startswith("I don't know")
    assert "No relevant runbook sections" in state.answer_text


def test_check_stops_when_every_score_is_under_the_line():
    state = graph_agent.new_state("swap", client=None)
    state.results = [WEAK, dict(WEAK, section="Disk space almost full", score=0.06)]
    assert graph_agent.check(state) == "IDK"
    assert state.stop_reason == agent.WEAK_MATCH_ONLY
    assert "too weak" in state.answer_text


def test_check_treats_a_score_on_the_line_as_strong_enough():
    state = graph_agent.new_state("cpu", client=None)
    state.results = [dict(CPU, score=agent.CITE_MIN_SCORE)]
    assert graph_agent.check(state) == "ANSWER"
    assert state.sources[0]["section"] == "High CPU on web servers"


def test_check_sends_strong_sections_to_answer_and_remembers_them():
    reply = tool_reply("search_runbooks", {"query": "cpu"}, "abc")
    state = graph_agent.new_state("cpu", client=None)
    state.results = [CPU]
    state.reply = reply
    state.call = reply.tool_calls[0]
    assert graph_agent.check(state) == "ANSWER"
    assert state.sources == [{
        "source": "rb.txt",
        "section": "High CPU on web servers",
        "score": 0.4,
        "text": "Restart the offending worker.",
    }]
    assert state.messages[-1]["role"] == "tool"
    assert state.messages[-1]["tool_call_id"] == "abc"
    assert "High CPU on web servers" in state.messages[-1]["content"]


def test_check_sends_an_answer_that_cites_nothing_to_idk():
    state = graph_agent.new_state("cpu", client=None)
    state.has_draft = True
    state.draft_answer = "1. Restart the database."
    state.sources = [{"source": "m.txt", "section": "High swap usage", "score": 0.05}]
    assert graph_agent.check(state) == "IDK"
    assert state.stop_reason == agent.UNGROUNDED_ANSWER
    assert "did not cite a runbook section" in state.answer_text


def test_check_allows_a_short_request_for_more_detail():
    state = graph_agent.new_state("hi", client=None)
    state.has_draft = True
    state.draft_answer = "Please give me more detail."
    assert graph_agent.check(state) == "ANSWER"
    assert state.stop_reason == ""


def test_check_sends_a_cited_answer_on_to_be_saved():
    state = graph_agent.new_state("cpu", client=None)
    state.has_draft = True
    state.draft_answer = "Restart the worker."
    state.sources = [{"source": "rb.txt", "section": "High CPU on web servers", "score": 0.4}]
    assert graph_agent.check(state) == "ANSWER"


def test_check_does_not_let_a_later_empty_search_wipe_a_strong_section():
    state = graph_agent.new_state("cpu", client=None)
    state.sources = [{"source": "rb.txt", "section": "High CPU on web servers", "score": 0.4}]
    state.results = []
    reply = tool_reply("search_runbooks", {"query": "pancakes"}, "c2")
    state.reply = reply
    state.call = reply.tool_calls[0]
    assert graph_agent.check(state) == "ANSWER"
    assert state.sources[0]["section"] == "High CPU on web servers"


# ---- ANSWER ----------------------------------------------------------------

def test_answer_asks_the_model_to_write_then_goes_back_to_check():
    client = FakeClient(text_reply("Restart the worker.\n\nSources:\n- Made up (nope.txt)"))
    state = graph_agent.new_state("cpu", client=client)
    state.steps = 2
    graph_agent.last_trace = None
    assert graph_agent.answer(state) == "CHECK"
    assert state.has_draft is True
    # The model's own Sources block is dropped. Only CHECK may add one.
    assert state.draft_answer == "Restart the worker."
    assert "Made up" not in state.draft_answer
    assert graph_agent.last_trace is None
    assert len(client.calls) == 1


def test_answer_saves_a_grounded_reply_and_stops():
    state = graph_agent.new_state("cpu", client=None)
    state.has_draft = True
    state.draft_answer = "Restart the worker."
    state.sources = [{"source": "rb.txt", "section": "High CPU on web servers", "score": 0.4}]
    state.steps = 2
    state.node_path = ["PLAN", "SEARCH", "CHECK", "ANSWER", "CHECK", "ANSWER"]
    assert graph_agent.answer(state) is None
    assert state.return_value.startswith("Restart the worker.")
    assert cited_sources(state.return_value) == ["High CPU on web servers (rb.txt)"]
    assert graph_agent.last_trace.stop_reason == agent.ANSWERED
    assert graph_agent.last_trace.node_path[-1] == "ANSWER"
    assert graph_agent.last_trace.steps == 2


def test_answer_can_search_again_when_the_model_asks():
    client = FakeClient(tool_reply("search_runbooks", {"query": "disk"}))
    state = graph_agent.new_state("cpu and disk", client=client)
    state.steps = 2
    assert graph_agent.answer(state) == "SEARCH"
    assert state.has_draft is False
    assert state.arguments == {"query": "disk"}


# ---- IDK -------------------------------------------------------------------

def test_idk_ends_with_the_reason_check_chose():
    state = graph_agent.new_state("cake", client=None)
    state.stop_reason = agent.NO_RELEVANT_RUNBOOK
    state.answer_text = "I don't know. No relevant runbook sections were found, so I will not guess."
    state.steps = 1
    state.node_path = ["PLAN", "SEARCH", "CHECK", "IDK"]
    assert graph_agent.idk(state) is None
    assert state.return_value.startswith("I don't know")
    assert "Sources:" not in state.return_value
    assert graph_agent.last_trace.stop_reason == agent.NO_RELEVANT_RUNBOOK
    assert graph_agent.last_trace.node_path == ["PLAN", "SEARCH", "CHECK", "IDK"]


# ---- ACTION_GATE -----------------------------------------------------------

def test_action_gate_can_wait_without_asking_or_running(monkeypatch):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: [CPU])
    ask = make_asker("y")
    state = graph_agent.new_state("fix", client=None, ask=ask, defer_approval=True)
    state.action = RESTART
    state.steps = 1
    state.node_path = ["PLAN", "ACTION_GATE"]
    assert graph_agent.action_gate(state) is None
    assert isinstance(state.return_value, agent.PendingAction)
    assert state.return_value.action == RESTART
    assert ask.prompts == []
    assert graph_agent.last_trace.stop_reason == agent.ACTION_REQUESTED
    assert graph_agent.last_trace.node_path == ["PLAN", "ACTION_GATE"]


def test_action_gate_denies_anything_but_yes(monkeypatch, capsys):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: [CPU])
    state = graph_agent.new_state("fix", client=None, ask=make_asker("n"))
    state.action = RESTART
    state.steps = 1
    state.node_path = ["PLAN", "ACTION_GATE"]
    assert graph_agent.action_gate(state) is None
    assert state.return_value == "denied"
    assert "[simulated]" not in capsys.readouterr().out
    assert graph_agent.last_trace.stop_reason == agent.ACTION_REQUESTED


def test_action_gate_yes_runs_the_same_simulation(monkeypatch, capsys):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: [CPU])
    state = graph_agent.new_state("fix", client=None, ask=make_asker("y"))
    state.action = RESTART
    state.steps = 1
    state.node_path = ["PLAN", "ACTION_GATE"]
    assert graph_agent.action_gate(state) is None
    result = state.return_value
    assert isinstance(result, agent.ActionResult)
    assert result.ok
    assert result.action == "restart process runaway-worker"
    assert "pid 1842" in capsys.readouterr().out


# ---- Full walks: one terminal path each -----------------------------------

def test_answered_path_matches_the_loop(monkeypatch):
    patch_search(monkeypatch, [CPU])

    def client():
        return FakeClient(
            tool_reply("search_runbooks", {"query": "cpu"}),
            text_reply("Restart the worker."),
        )

    ask = lambda prompt="": "n"
    agent_answer = agent.run_agent("cpu high", client=client(), ask=ask)
    agent_reason = agent.last_trace.stop_reason
    agent_sources = list(agent.last_trace.sources)
    graph_answer = graph_agent.run_graph("cpu high", client=client(), ask=ask)
    assert graph_agent.last_trace.stop_reason == agent_reason == agent.ANSWERED
    assert graph_answer == agent_answer
    assert graph_agent.last_trace.sources == agent_sources
    assert graph_agent.last_trace.steps == 2
    assert graph_agent.last_trace.node_path == [
        "PLAN", "SEARCH", "CHECK", "ANSWER", "CHECK", "ANSWER",
    ]
    assert cited_sources(graph_answer) == ["High CPU on web servers (rb.txt)"]


def test_nothing_relevant_matches_the_loop(monkeypatch, capsys):
    patch_search(monkeypatch, [])

    def client():
        return FakeClient(
            tool_reply("search_runbooks", {"query": "pancake recipe"}),
            text_reply("SHOULD NOT BE USED"),
        )

    ask = lambda prompt="": "n"
    agent_answer = agent.run_agent("how do I bake a cake", client=client(), ask=ask)
    agent_reason = agent.last_trace.stop_reason
    graph_answer = graph_agent.run_graph("how do I bake a cake", client=client(), ask=ask)
    out = capsys.readouterr().out
    assert graph_agent.last_trace.stop_reason == agent_reason == agent.NO_RELEVANT_RUNBOOK
    assert graph_answer == agent_answer
    assert graph_answer.startswith("I don't know")
    assert "SHOULD NOT BE USED" not in graph_answer
    assert "Sources:" not in graph_answer
    assert graph_agent.last_trace.node_path == ["PLAN", "SEARCH", "CHECK", "IDK"]
    assert graph_agent.last_trace.steps == 1
    assert "Stop reason: NO_RELEVANT_RUNBOOK" in out


def test_weak_matches_only_matches_the_loop(monkeypatch):
    patch_search(monkeypatch, [WEAK])

    def client():
        return FakeClient(
            tool_reply("search_runbooks", {"query": "memory"}),
            text_reply("SHOULD NOT BE USED"),
        )

    ask = lambda prompt="": "n"
    agent_answer = agent.run_agent("the box is swapping", client=client(), ask=ask)
    graph_answer = graph_agent.run_graph("the box is swapping", client=client(), ask=ask)
    assert graph_agent.last_trace.stop_reason == agent.last_trace.stop_reason == agent.WEAK_MATCH_ONLY
    assert graph_answer == agent_answer
    assert "too weak" in graph_answer
    assert graph_agent.last_trace.sources == []
    assert graph_agent.last_trace.node_path[-1] == "IDK"


def test_ungrounded_answer_matches_the_loop(capsys):
    def client():
        return FakeClient(text_reply("1. Restart the database.\n2. Delete every log."))

    ask = lambda prompt="": "n"
    agent_answer = agent.run_agent("the database is down", client=client(), ask=ask)
    graph_answer = graph_agent.run_graph("the database is down", client=client(), ask=ask)
    out = capsys.readouterr().out
    assert graph_agent.last_trace.stop_reason == agent.last_trace.stop_reason == agent.UNGROUNDED_ANSWER
    assert graph_answer == agent_answer
    assert "Restart the database" not in graph_answer
    assert "Restart the database" not in out
    assert graph_agent.last_trace.node_path == ["PLAN", "CHECK", "IDK"]
    assert graph_agent.last_trace.steps == 1


def test_a_request_for_more_detail_is_answered_with_no_sources():
    client = FakeClient(text_reply("Please give me more detail."))
    answer = graph_agent.run_graph("hi", client=client)
    assert answer == "Please give me more detail."
    assert graph_agent.last_trace.stop_reason == agent.ANSWERED
    assert graph_agent.last_trace.sources == []
    assert graph_agent.last_trace.steps == 1
    assert graph_agent.last_trace.node_path == ["PLAN", "CHECK", "ANSWER"]
    assert "Sources:" not in answer


def test_action_requested_matches_the_loop(monkeypatch):
    patch_search(monkeypatch, [CPU])

    def client():
        return FakeClient(tool_reply("execute_action", {"action": RESTART, "approved": True}))

    ask = lambda prompt="": "n"
    agent_result = agent.run_agent("fix cpu", client=client(), ask=ask)
    agent_reason = agent.last_trace.stop_reason
    graph_result = graph_agent.run_graph("fix cpu", client=client(), ask=ask)
    assert graph_result == agent_result == "denied"
    assert graph_agent.last_trace.stop_reason == agent_reason == agent.ACTION_REQUESTED
    assert graph_agent.last_trace.node_path == ["PLAN", "ACTION_GATE"]
    assert graph_agent.last_trace.steps == 1


def test_a_deferred_action_waits_and_does_not_run(monkeypatch, capsys):
    patch_search(monkeypatch, [CPU])
    ask = make_asker("y")
    client = FakeClient(tool_reply("execute_action", {"action": RESTART}))
    result = graph_agent.run_graph("fix cpu", client=client, ask=ask, defer_approval=True)
    out = capsys.readouterr().out
    assert isinstance(result, agent.PendingAction)
    assert result.action == RESTART
    assert ask.prompts == []
    assert "[simulated]" not in out
    assert "Waiting for a person" in out
    assert graph_agent.last_trace.stop_reason == agent.ACTION_REQUESTED


def test_yes_on_the_full_walk_simulates_the_restart(monkeypatch, capsys):
    patch_search(monkeypatch, [CPU])
    client = FakeClient(tool_reply("execute_action", {"action": RESTART}))
    result = graph_agent.run_graph("fix cpu", client=client, ask=make_asker("yes"))
    assert isinstance(result, agent.ActionResult) and result.ok
    assert "CPU normalized to 12%" in capsys.readouterr().out
    assert graph_agent.last_trace.stop_reason == agent.ACTION_REQUESTED


def test_an_unknown_action_is_still_a_named_error(monkeypatch, capsys):
    patch_search(monkeypatch, [CPU])
    client = FakeClient(tool_reply("execute_action", {"action": "delete the database"}))
    with pytest.raises(agent.AgentError) as caught:
        graph_agent.run_graph("fix", client=client, ask=make_asker("y"))
    assert caught.value.name == "UNKNOWN_ACTION"
    assert "[simulated]" not in capsys.readouterr().out
    assert graph_agent.last_trace.stop_reason == agent.ACTION_REQUESTED


def test_a_search_with_no_model_thought_still_says_what_it_needs(monkeypatch, capsys):
    patch_search(monkeypatch, [CPU])
    client = FakeClient(
        tool_reply("search_runbooks", {"query": "cpu"}),
        text_reply("Restart the worker."),
    )
    graph_agent.run_graph("cpu high", client=client)
    out = capsys.readouterr().out
    assert "Thinking: I should search the runbooks for 'cpu'." in out
    assert out.index("Thinking:") < out.index("Model chose:")
    assert "Nodes visited: PLAN -> SEARCH -> CHECK -> ANSWER -> CHECK -> ANSWER" in out


def test_reworded_searches_match_the_loop(monkeypatch):
    """Changing the words twice, and finding nothing, stops the same way in both.

    The answer says it could not find this, and it stops before the step limit.
    """
    stray = {
        "source": "etcd.txt",
        "section": "Apiserver writes fail",
        "text": "The database is large.",
        "score": 0.18,
    }

    def search(query):
        if query == "postgres vacuum tuning large table":
            return [stray]
        return []

    monkeypatch.setattr(agent.retrieve, "search", search)

    def client():
        return FakeClient(
            tool_reply("search_runbooks", {"query": "postgres vacuum tuning large table"}, "c1"),
            tool_reply("search_runbooks", {"query": "postgres vacuum"}, "c2"),
            tool_reply("search_runbooks", {"query": "vacuum and postgres"}, "c3"),
            text_reply("SHOULD NOT BE USED"),
        )

    ask = lambda prompt="": "n"
    question = "How do I tune Postgres vacuum for a large table"
    agent_answer = agent.run_agent(question, client=client(), ask=ask)
    agent_reason = agent.last_trace.stop_reason
    agent_steps = agent.last_trace.steps
    graph_answer = graph_agent.run_graph(question, client=client(), ask=ask)
    assert "could not find this in the runbooks" in agent_answer
    assert graph_answer == agent_answer
    assert "SHOULD NOT BE USED" not in graph_answer
    assert "Sources:" not in graph_answer
    assert graph_agent.last_trace.stop_reason == agent_reason == agent.NO_RELEVANT_RUNBOOK
    assert graph_agent.last_trace.steps == agent_steps == 3
    assert agent_steps <= graph_agent.MAX_STEPS
    assert graph_agent.last_trace.node_path[-1] == "IDK"
    assert graph_agent.last_trace.sources == []


def test_a_fake_action_request_matches_the_loop(monkeypatch, capsys):
    """Both copies propose the action and run nothing until a person agrees."""
    seen = []
    monkeypatch.setattr(agent.retrieve, "search", lambda q: seen.append(q) or [])

    def client():
        return FakeClient(tool_reply(
            "execute_action",
            {"action": "restart the process 'web-worker'", "approved": True},
        ))

    ask = make_asker("y")
    question = "Restart the stuck web worker"
    agent_result = agent.run_agent(question, client=client(), ask=ask, defer_approval=True)
    agent_reason = agent.last_trace.stop_reason
    graph_result = graph_agent.run_graph(question, client=client(), ask=make_asker("y"), defer_approval=True)
    out = capsys.readouterr().out
    assert isinstance(agent_result, agent.PendingAction)
    assert isinstance(graph_result, agent.PendingAction)
    assert graph_result.action == agent_result.action == "restart the process 'web-worker'"
    assert graph_agent.last_trace.stop_reason == agent_reason == agent.ACTION_REQUESTED
    assert graph_agent.last_trace.node_path == ["PLAN", "ACTION_GATE"]
    assert seen == []
    assert "[simulated]" not in out


def test_an_answerable_practice_question_matches_the_loop():
    """A question the runbooks cover still gets the same practice-model reply."""
    question = "The web server CPU is at 95 percent and pages are slow"
    ask = lambda prompt="": "n"
    agent_answer = agent.run_agent(question, client=app.DemoClient(), ask=ask, defer_approval=True)
    agent_reason = agent.last_trace.stop_reason
    graph_answer = graph_agent.run_graph(question, client=app.DemoClient(), ask=ask, defer_approval=True)
    assert graph_answer == agent_answer
    assert graph_agent.last_trace.stop_reason == agent_reason == agent.ANSWERED
    assert graph_answer.startswith("1. Symptom: web servers show CPU above ninety percent")
    assert "High CPU on web servers (sample_runbook.txt)" in graph_answer


def test_step_limit_with_nothing_read_matches_the_loop(monkeypatch, capsys):
    """No strong section at the step cap: plain cannot-find words, same stop name."""
    monkeypatch.setattr(agent, "MAX_STEPS", 0)
    monkeypatch.setattr(graph_agent, "MAX_STEPS", 0)
    question = "How do I tune Postgres vacuum for a large table"
    ask = lambda prompt="": "n"
    agent_answer = agent.run_agent(question, client=FakeClient(text_reply("NO")), ask=ask)
    agent_reason = agent.last_trace.stop_reason
    graph_answer = graph_agent.run_graph(question, client=FakeClient(text_reply("NO")), ask=ask)
    out = capsys.readouterr().out
    assert agent_answer == "I could not find this in the runbooks."
    assert graph_answer == agent_answer
    assert graph_agent.last_trace.stop_reason == agent_reason == agent.MAX_STEPS_REACHED
    assert "Stop reason: MAX_STEPS_REACHED" in out
    assert graph_agent.last_trace.node_path[-1] == "IDK"


def test_an_empty_later_search_keeps_the_section_already_read(monkeypatch):
    def search(query):
        if query == "cpu":
            return [CPU]
        return []

    monkeypatch.setattr(agent.retrieve, "search", search)
    client = FakeClient(
        tool_reply("search_runbooks", {"query": "cpu"}, "c1"),
        tool_reply("search_runbooks", {"query": "pancakes"}, "c2"),
        text_reply("Restart the worker."),
    )
    answer = graph_agent.run_graph("cpu high", client=client)
    assert answer.startswith("Restart the worker.")
    assert "I don't know" not in answer
    assert graph_agent.last_trace.stop_reason == agent.ANSWERED
    assert cited_sources(answer) == ["High CPU on web servers (rb.txt)"]
    assert graph_agent.last_trace.steps == 3
    assert graph_agent.last_trace.node_path.count("SEARCH") == 2


# ---- The step limit --------------------------------------------------------

def test_a_model_that_searches_forever_stops_at_four(monkeypatch, capsys):
    patch_search(monkeypatch, [CPU])
    client = ForeverClient()
    answer = graph_agent.run_graph("cpu high", client=client)
    out = capsys.readouterr().out
    assert len(client.calls) == graph_agent.MAX_STEPS == 4
    assert graph_agent.last_trace.stop_reason == agent.MAX_STEPS_REACHED
    assert graph_agent.last_trace.steps == 4
    assert graph_agent.last_trace.node_path[0] == "PLAN"
    assert graph_agent.last_trace.node_path[-1] == "IDK"
    assert graph_agent.last_trace.node_path.count("SEARCH") == 4
    assert graph_agent.last_trace.node_path.count("PLAN") == 1
    assert "maximum number of steps (4)" in answer
    assert "I will not guess" in answer
    assert "Step 5" not in out
    assert cited_sources(answer) == ["High CPU on web servers (rb.txt)"]


def test_forever_search_matches_the_loop_stop_reason(monkeypatch):
    patch_search(monkeypatch, [CPU])
    ask = lambda prompt="": "n"
    agent.run_agent("cpu high", client=ForeverClient(), ask=ask)
    agent_reason = agent.last_trace.stop_reason
    agent_steps = agent.last_trace.steps
    graph_agent.run_graph("cpu high", client=ForeverClient(), ask=ask)
    assert graph_agent.last_trace.stop_reason == agent_reason == agent.MAX_STEPS_REACHED
    assert graph_agent.last_trace.steps == agent_steps == 4


def test_an_answer_on_the_last_allowed_step_is_kept(monkeypatch):
    monkeypatch.setattr(graph_agent, "MAX_STEPS", 2)
    patch_search(monkeypatch, [CPU])
    client = FakeClient(
        tool_reply("search_runbooks", {"query": "cpu"}),
        text_reply("Restart the worker."),
    )
    answer = graph_agent.run_graph("cpu high", client=client)
    assert answer.startswith("Restart the worker.")
    assert graph_agent.last_trace.stop_reason == agent.ANSWERED
    assert graph_agent.last_trace.steps == 2
    assert "maximum number of steps" not in answer
    assert len(client.calls) == 2


def test_a_lower_cap_stops_a_forever_model_early(monkeypatch, capsys):
    monkeypatch.setattr(graph_agent, "MAX_STEPS", 2)
    patch_search(monkeypatch, [CPU])
    client = ForeverClient()
    answer = graph_agent.run_graph("cpu high", client=client)
    out = capsys.readouterr().out
    assert len(client.calls) == 2
    assert graph_agent.last_trace.steps == 2
    assert graph_agent.last_trace.stop_reason == agent.MAX_STEPS_REACHED
    assert "maximum number of steps (2)" in answer
    assert "Step 3" not in out


def test_one_step_stops_before_the_answer_is_written(monkeypatch):
    monkeypatch.setattr(graph_agent, "MAX_STEPS", 1)
    patch_search(monkeypatch, [CPU])
    client = FakeClient(
        tool_reply("search_runbooks", {"query": "cpu"}),
        text_reply("SHOULD NOT BE USED"),
    )
    answer = graph_agent.run_graph("cpu high", client=client)
    assert len(client.calls) == 1
    assert "SHOULD NOT BE USED" not in answer
    assert graph_agent.last_trace.stop_reason == agent.MAX_STEPS_REACHED
    assert graph_agent.last_trace.node_path == ["PLAN", "SEARCH", "CHECK", "IDK"]
    assert cited_sources(answer) == ["High CPU on web servers (rb.txt)"]


# ---- Named failures, same names as agent.py --------------------------------

def test_empty_request_is_a_named_failure():
    graph_agent.last_trace = "stale"
    with pytest.raises(agent.AgentError) as caught:
        graph_agent.run_graph("   ", client=FakeClient())
    assert caught.value.name == "EMPTY_REQUEST"
    assert graph_agent.last_trace is None


def test_missing_key_is_a_named_failure(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    with pytest.raises(agent.AgentError) as caught:
        graph_agent.run_graph("cpu high")
    assert caught.value.name == "MISSING_KEY"
    assert graph_agent.last_trace is None


def test_rejected_key_is_a_named_failure():
    class AuthenticationError(Exception):
        pass

    def boom(**kwargs):
        raise AuthenticationError("nope")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=boom)))
    with pytest.raises(agent.AgentError) as caught:
        graph_agent.run_graph("cpu", client=client)
    assert caught.value.name == "BAD_KEY"


def test_an_unknown_tool_is_a_named_failure():
    client = FakeClient(tool_reply("delete_everything", {"target": "moon"}))
    with pytest.raises(agent.AgentError) as caught:
        graph_agent.run_graph("cpu", client=client)
    assert caught.value.name == "UNKNOWN_TOOL"
    assert graph_agent.last_trace is None


def test_a_missing_action_string_is_a_named_failure():
    client = FakeClient(tool_reply("execute_action", {}))
    with pytest.raises(agent.AgentError) as caught:
        graph_agent.run_graph("fix", client=client, ask=make_asker("y"))
    assert caught.value.name == "BAD_TOOL_ARGS"


def test_main_uses_the_graph(monkeypatch, capsys):
    patch_search(monkeypatch, [CPU])
    client = FakeClient(
        tool_reply("search_runbooks", {"query": "web server cpu"}),
        text_reply("Restart the worker."),
    )
    monkeypatch.setattr(agent, "make_client", lambda: client)
    result = graph_agent.main(["web", "server", "cpu"], ask=make_asker())
    out = capsys.readouterr().out
    assert result.startswith("Restart the worker.")
    assert "Stop reason: ANSWERED" in out
    assert "Nodes visited:" in out
    assert client.calls[0]["messages"][1]["content"] == "web server cpu"


# ---- Practice model: the eval questions, both files ------------------------

@pytest.mark.parametrize("scenario", evals.SCENARIOS, ids=[s.name for s in evals.SCENARIOS])
def test_practice_model_stop_reason_matches_the_loop(scenario):
    """Same question, same practice model, same stop reason.

    The practice model is the built-in one from the page. It needs no API key.
    """
    ask = lambda prompt="": "n"
    agent.run_agent(
        scenario.request, client=app.DemoClient(), ask=ask, defer_approval=True,
    )
    agent_reason = agent.last_trace.stop_reason
    graph_agent.run_graph(
        scenario.request, client=app.DemoClient(), ask=ask, defer_approval=True,
    )
    assert graph_agent.last_trace.stop_reason == agent_reason
    assert agent_reason in {
        agent.ANSWERED,
        agent.NO_RELEVANT_RUNBOOK,
        agent.WEAK_MATCH_ONLY,
        agent.UNGROUNDED_ANSWER,
        agent.MAX_STEPS_REACHED,
        agent.ACTION_REQUESTED,
    }
    assert graph_agent.last_trace.node_path
    assert graph_agent.last_trace.node_path[0] == "PLAN"


def test_the_runbook_match_check_agrees_with_the_loop(monkeypatch):
    """A similar problem is refused in both. An exact one is answered in both.

    An action request stays an action request. The graph uses the same check.
    """
    similar = {
        "source": "sample_runbook.txt",
        "section": "Disk space almost full",
        "text": "The server disk is full. Delete old logs on the server.",
        "score": 0.42,
    }
    exact = {
        "source": "sample_runbook.txt",
        "section": "High CPU on web servers",
        "text": "Restart the worker that is using the CPU on the web server.",
        "score": 0.5,
    }

    def search(query):
        if "phone" in query:
            return [similar]
        return [exact]

    monkeypatch.setattr(agent.retrieve, "search", search)
    ask = lambda prompt="": "n"

    def pair(question, replies):
        agent_answer = agent.run_agent(question, client=FakeClient(*replies), ask=ask, defer_approval=True)
        agent_reason = agent.last_trace.stop_reason
        graph_answer = graph_agent.run_graph(
            question, client=FakeClient(*replies), ask=ask, defer_approval=True,
        )
        return agent_answer, agent_reason, graph_answer

    tool = tool_reply("search_runbooks", {"query": "phone storage"})
    words = text_reply("SHOULD NOT BE USED: Delete old logs on the server.")
    agent_answer, agent_reason, graph_answer = pair(
        "My phone storage is full and I cannot save pictures", (tool, words),
    )
    assert agent_reason == agent.NO_RELEVANT_RUNBOOK
    assert graph_answer == agent_answer
    assert "SHOULD NOT BE USED" not in graph_answer
    assert graph_agent.last_trace.stop_reason == agent_reason
    assert graph_agent.last_trace.node_path[-1] == "IDK"
    assert graph_agent.last_trace.sources == []

    tool = tool_reply("search_runbooks", {"query": "web server cpu"})
    words = text_reply("Restart the worker that is using the CPU.")
    agent_answer, agent_reason, graph_answer = pair("The web server CPU is high", (tool, words))
    assert agent_reason == agent.ANSWERED
    assert graph_answer == agent_answer
    assert graph_answer.startswith("Restart the worker that is using the CPU.")
    assert graph_agent.last_trace.stop_reason == agent_reason

    action = tool_reply("execute_action", {"action": "restart the process 'web-worker'"})
    agent_result = agent.run_agent(
        "Restart the worker on my laptop", client=FakeClient(action), ask=ask, defer_approval=True,
    )
    agent_reason = agent.last_trace.stop_reason
    graph_result = graph_agent.run_graph(
        "Restart the worker on my laptop", client=FakeClient(action), ask=ask, defer_approval=True,
    )
    assert isinstance(agent_result, agent.PendingAction)
    assert isinstance(graph_result, agent.PendingAction)
    assert graph_result.action == agent_result.action
    assert graph_agent.last_trace.stop_reason == agent_reason == agent.ACTION_REQUESTED
    assert graph_agent.last_trace.node_path == ["PLAN", "ACTION_GATE"]
