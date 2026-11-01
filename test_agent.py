import json
from datetime import datetime
from types import SimpleNamespace

import pytest

import agent

RESULTS = [
    {"source": "rb.txt", "section": "High CPU on web servers", "text": "Restart the offending worker.", "score": 0.4},
    {"source": "rb.txt", "section": "Disk space almost full", "text": "Rotate the logs.", "score": 0.1},
]


def tool_reply(name, arguments, call_id="call_1"):
    call = SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))
    return SimpleNamespace(content=None, tool_calls=[call])


def text_reply(text):
    return SimpleNamespace(content=text, tool_calls=None)


class FakeClient:
    """Stands in for the xAI client. Returns scripted replies and records every call."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return SimpleNamespace(choices=[SimpleNamespace(message=self.replies.pop(0))])


def test_model_is_called_with_the_request_and_both_tools(monkeypatch):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: RESULTS)
    client = FakeClient(tool_reply("search_runbooks", {"query": "cpu"}), text_reply("done"))
    agent.run_agent("web server CPU is high", client=client)
    first = client.calls[0]
    assert first["messages"][-1] == {"role": "user", "content": "web server CPU is high"}
    names = [t["function"]["name"] for t in first["tools"]]
    assert names == ["search_runbooks", "execute_action"]
    assert first["model"] == "grok-build-0.1"


def test_search_uses_the_models_chosen_query(monkeypatch):
    seen = []
    monkeypatch.setattr(agent.retrieve, "search", lambda q: seen.append(q) or RESULTS)
    client = FakeClient(tool_reply("search_runbooks", {"query": "runaway worker cpu"}), text_reply("done"))
    agent.run_agent("my site is slow", client=client)
    assert seen == ["runaway worker cpu"]  # not the user's own words


def test_results_are_returned_to_the_model_and_final_answer_is_used(monkeypatch):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: RESULTS)
    client = FakeClient(tool_reply("search_runbooks", {"query": "cpu"}, "abc"), text_reply("Restart the worker."))
    answer = agent.run_agent("cpu high", client=client)
    assert answer.startswith("Restart the worker.")  # the model's sentence, plus the sources the loop adds
    second = client.calls[1]["messages"]
    tool_msg = second[-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "abc"
    assert "High CPU on web servers" in tool_msg["content"]
    assert second[-2]["tool_calls"][0]["function"]["name"] == "search_runbooks"


def test_results_are_printed_with_section_labels_and_scores(monkeypatch, capsys):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: RESULTS)
    client = FakeClient(tool_reply("search_runbooks", {"query": "cpu"}), text_reply("ok"))
    agent.run_agent("cpu high", client=client)
    out = capsys.readouterr().out
    assert "High CPU on web servers" in out and "0.400" in out
    assert "Answer: ok" in out


def make_asker(*answers):
    """A fake input(): returns the scripted answers and records every prompt shown."""
    prompts = []
    queue = list(answers)

    def ask(prompt):
        prompts.append(prompt)
        return queue.pop(0)

    ask.prompts = prompts
    return ask


RESTART = "restart the process 'runaway-worker'"


def action_client(action=RESTART):
    return FakeClient(tool_reply("execute_action", {"action": action}))


def test_approval_prompt_shows_the_action_summary(monkeypatch, capsys):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: RESULTS)
    ask = make_asker("n")
    agent.run_agent("fix cpu", client=action_client(), ask=ask)
    out = capsys.readouterr().out
    assert "needs your approval" in out
    assert "execute_action" in out
    assert "runaway-worker" in out
    assert "High CPU on web servers" in out  # runbook context
    assert ask.prompts == ["Approve this action? (y/n) "]


@pytest.mark.parametrize("answer", ["y", "yes", "Y", " YES "])
def test_yes_allows_the_action(monkeypatch, capsys, answer):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: RESULTS)
    result = agent.run_agent("fix cpu", client=action_client(), ask=make_asker(answer))
    assert result.status == "success"
    assert result.action == "restart process runaway-worker"
    assert "Result: success" in capsys.readouterr().out


def test_n_denies_the_action(monkeypatch, capsys):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: RESULTS)
    result = agent.run_agent("fix cpu", client=action_client(), ask=make_asker("n"))
    assert result == "denied"
    out = capsys.readouterr().out
    assert "Action denied. Nothing was run." in out
    assert "[simulated]" not in out


@pytest.mark.parametrize("answer", ["", "   ", "maybe", "sure", "ok", "yep", "no"])
def test_empty_or_invalid_answer_denies_the_action(monkeypatch, capsys, answer):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: RESULTS)
    result = agent.run_agent("fix cpu", client=action_client(), ask=make_asker(answer))
    assert result == "denied"
    assert "[simulated]" not in capsys.readouterr().out


def test_model_cannot_approve_its_own_action(monkeypatch):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: RESULTS)
    client = FakeClient(tool_reply("execute_action", {"action": "restart", "approved": True}))
    ask = make_asker("n")
    result = agent.run_agent("fix cpu", client=client, ask=ask)
    assert result == "denied"
    assert len(ask.prompts) == 1  # the human was still asked


def test_execute_action_without_approval_still_raises():
    with pytest.raises(agent.AgentError) as e:
        agent.call_tool("execute_action", action="restart", approved="yes")
    assert e.value.name == "NOT_IMPLEMENTED_APPROVAL"


def test_execute_action_without_an_action_is_a_named_failure():
    client = FakeClient(tool_reply("execute_action", {}))
    with pytest.raises(agent.AgentError) as e:
        agent.run_agent("fix cpu", client=client, ask=make_asker("y"))
    assert e.value.name == "BAD_TOOL_ARGS"


def test_execute_action_placeholder_requires_approval():
    with pytest.raises(agent.AgentError) as e:
        agent.call_tool("execute_action", action="restart worker")
    assert "approval" in str(e.value)


def test_missing_key_is_a_named_failure(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    with pytest.raises(agent.AgentError) as e:
        agent.run_agent("cpu high")
    assert e.value.name == "MISSING_KEY"


def test_model_answering_without_a_tool_is_returned():
    client = FakeClient(text_reply("Please give me more detail."))
    assert agent.run_agent("hi", client=client) == "Please give me more detail."


def test_bad_tool_arguments_are_a_named_failure():
    call = SimpleNamespace(id="x", function=SimpleNamespace(name="search_runbooks", arguments="not json"))
    client = FakeClient(SimpleNamespace(content=None, tool_calls=[call]))
    with pytest.raises(agent.AgentError) as e:
        agent.run_agent("cpu", client=client)
    assert e.value.name == "BAD_TOOL_ARGS"


def test_rejected_key_is_a_named_failure():
    class AuthenticationError(Exception):
        pass

    def boom(**kwargs):
        raise AuthenticationError("nope")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=boom)))
    with pytest.raises(agent.AgentError) as e:
        agent.run_agent("cpu", client=client)
    assert e.value.name == "BAD_KEY"


def test_empty_request_is_a_named_failure():
    with pytest.raises(agent.AgentError) as e:
        agent.run_agent("  ", client=FakeClient())
    assert e.value.name == "EMPTY_REQUEST"


def test_unknown_tool_is_a_named_failure():
    with pytest.raises(agent.AgentError) as e:
        agent.call_tool("delete_everything")
    assert e.value.name == "UNKNOWN_TOOL"


# ---- --force-action: skip the model, go straight to the approval gate --------

@pytest.fixture
def no_model(monkeypatch):
    def boom():
        raise AssertionError("the model must not be used with --force-action")

    monkeypatch.setattr(agent, "make_client", boom)


@pytest.mark.parametrize("flag", ["--force-action", "-f"])
def test_force_flag_shows_the_approval_prompt_without_the_model(no_model, capsys, flag):
    ask = make_asker("n")
    agent.main([flag], ask=ask)
    out = capsys.readouterr().out
    assert "Forced action (model skipped)" in out
    assert "needs your approval" in out
    assert "Tool: execute_action" in out
    assert "runaway-worker" in out
    assert "High CPU on web servers" in out  # real runbook context, found by search
    assert ask.prompts == ["Approve this action? (y/n) "]


@pytest.mark.parametrize("answer", ["y", "yes", "YES"])
def test_force_flag_y_approves_and_runs_the_simulation(no_model, capsys, answer):
    result = agent.main(["--force-action"], ask=make_asker(answer))
    assert result.status == "success"
    assert result.details["process"] == "runaway-worker"
    assert "CPU normalized to 12%" in capsys.readouterr().out


@pytest.mark.parametrize("answer", ["n", "no", "", "maybe"])
def test_force_flag_n_or_anything_else_denies(no_model, capsys, answer):
    result = agent.main(["-f"], ask=make_asker(answer))
    assert result == "denied"
    out = capsys.readouterr().out
    assert "Action denied. Nothing was run." in out
    assert "[simulated]" not in out


def test_force_flag_ignores_other_words_on_the_command_line(no_model, capsys):
    result = agent.main(["-f", "restart", "the", "web", "server"], ask=make_asker("n"))
    assert result == "denied"
    assert "runaway-worker" in capsys.readouterr().out


def test_without_the_flag_the_model_path_works_as_before(monkeypatch, capsys):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: RESULTS)
    client = FakeClient(tool_reply("search_runbooks", {"query": "web server cpu"}), text_reply("Restart the worker."))
    monkeypatch.setattr(agent, "make_client", lambda: client)
    result = agent.main(["web", "server", "cpu", "is", "high"], ask=make_asker())
    out = capsys.readouterr().out
    assert result.startswith("Restart the worker.")
    assert "Model chose: search_runbooks" in out
    assert "Forced action" not in out
    assert len(client.calls) == 2  # the model was used: one to pick the tool, one to answer
    assert client.calls[0]["messages"][1]["content"] == "web server cpu is high"


def test_without_the_flag_a_model_requested_action_still_hits_the_gate(monkeypatch, capsys):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: RESULTS)
    monkeypatch.setattr(agent, "make_client", lambda: action_client())
    result = agent.main(["fix", "cpu"], ask=make_asker("n"))
    assert result == "denied"
    assert "Forced action" not in capsys.readouterr().out


def test_without_the_flag_missing_key_is_still_a_named_failure(monkeypatch, capsys):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    agent.main(["fix", "cpu"], ask=make_asker())
    assert "MISSING_KEY" in capsys.readouterr().out


# ---- Simulated execution ------------------------------------------------------

def test_approved_restart_prints_the_simulated_steps_and_returns_success(monkeypatch, capsys):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: RESULTS)
    result = agent.run_agent("fix cpu", client=action_client(), ask=make_asker("y"))
    out = capsys.readouterr().out
    assert "Sending SIGTERM to process runaway-worker (pid 1842)" in out
    assert "Waiting 5 seconds" in out
    assert "runaway-worker stopped" in out
    assert "Replacement started with pid 2107" in out
    assert "CPU normalized to 12%" in out
    assert "Result: success - restart process runaway-worker" in out
    assert isinstance(result, agent.ActionResult)
    assert result.ok and result.status == "success" and result.error == ""
    assert result.action == "restart process runaway-worker"
    assert result.details == {
        "process": "runaway-worker", "old_pid": 1842, "new_pid": 2107, "cpu_percent": 12, "simulated": True,
    }


def test_simulation_does_not_really_wait_or_touch_processes(monkeypatch):
    import os
    import time

    monkeypatch.setattr(time, "sleep", lambda s: pytest.fail("simulation must not sleep"))
    monkeypatch.setattr(os, "kill", lambda *a: pytest.fail("simulation must not signal a process"))
    agent.call_tool("execute_action", action=RESTART, approved=True)


def test_other_process_names_get_a_steady_simulation(capsys):
    first = agent.call_tool("execute_action", action="restart the service 'billing-api'", approved=True)
    second = agent.call_tool("execute_action", action="Restart service billing-api", approved=True)
    assert first.status == "success"
    assert first.details == second.details  # same name, same made-up numbers
    assert first.details["new_pid"] == first.details["old_pid"] + 265
    assert "billing-api" in capsys.readouterr().out


@pytest.mark.parametrize("action", [
    "restart the process 'runaway-worker' on the web server (runbook: High CPU on web servers)",
    "please restart worker runaway-worker",
    "restart runaway-worker",
])
def test_known_restart_wordings_are_understood(action):
    assert agent.parse_action(action) == ("restart", "runaway-worker")


@pytest.mark.parametrize("action", [
    "restart the offending worker",
    "delete the database",
    "roll back the release",
    "",
])
def test_unknown_action_is_a_named_error_and_prints_no_steps(action, capsys):
    with pytest.raises(agent.AgentError) as e:
        agent.call_tool("execute_action", action=action, approved=True)
    assert e.value.name == "UNKNOWN_ACTION"
    assert "UNKNOWN_ACTION" in str(e.value)
    assert "[simulated]" not in capsys.readouterr().out


def test_unknown_action_approved_by_the_human_is_still_a_named_error(monkeypatch, capsys):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: RESULTS)
    with pytest.raises(agent.AgentError) as e:
        agent.run_agent("fix", client=action_client("delete the database"), ask=make_asker("y"))
    assert e.value.name == "UNKNOWN_ACTION"
    assert "[simulated]" not in capsys.readouterr().out


def test_main_prints_the_unknown_action_error(monkeypatch, capsys):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: RESULTS)
    monkeypatch.setattr(agent, "make_client", lambda: action_client("delete the database"))
    agent.main(["fix"], ask=make_asker("y"))
    assert "UNKNOWN_ACTION" in capsys.readouterr().out


def test_protected_process_is_a_failure_result(capsys):
    result = agent.call_tool("execute_action", action="restart the process 'sshd'", approved=True)
    assert result.status == "failure" and not result.ok
    assert result.error == "PROTECTED_PROCESS"
    assert "SIGTERM" not in capsys.readouterr().out


def test_denied_action_never_reaches_execution(monkeypatch, capsys):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: RESULTS)
    reached = []
    monkeypatch.setitem(agent.TOOLS, "execute_action", lambda **kw: reached.append(kw))
    for answer in ["n", "", "maybe"]:
        result = agent.run_agent("fix cpu", client=action_client(), ask=make_asker(answer))
        assert result == "denied"
    assert reached == []
    assert "[simulated]" not in capsys.readouterr().out


def test_forced_action_denied_never_reaches_execution(monkeypatch):
    reached = []
    monkeypatch.setitem(agent.TOOLS, "execute_action", lambda **kw: reached.append(kw))
    assert agent.run_forced_action(ask=make_asker("n")) == "denied"
    assert reached == []


def test_the_forced_sample_action_is_a_known_action():
    assert agent.parse_action(agent.SAMPLE_ACTION) == ("restart", "runaway-worker")


# ---- Tool-call log ---------------------------------------------------------

def read_log():
    return agent.LOG_FILE.read_text().splitlines()


def test_a_search_call_is_logged_with_timestamp_args_and_top_result(monkeypatch):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: RESULTS)
    client = FakeClient(tool_reply("search_runbooks", {"query": "cpu"}), text_reply("done"))
    agent.run_agent("cpu high", client=client)
    lines = read_log()
    assert len(lines) == 1
    stamp, tool, args, summary = lines[0].split(" | ")
    assert datetime.fromisoformat(stamp)  # a real ISO timestamp
    assert tool == "search_runbooks"
    assert args == 'args={"query": "cpu"}'
    assert 'top="High CPU on web servers" score=0.400' in summary


def test_denied_and_approved_actions_are_logged():
    agent.run_gated_action("restart the process 'runaway-worker'", ask=make_asker("n"))
    agent.run_gated_action("restart the process 'runaway-worker'", ask=make_asker("y"))
    denied, approved = read_log()
    assert denied.endswith("| denied") and "execute_action" in denied
    assert "runaway-worker" in approved and "approved, result=" in approved


def test_the_log_is_appended_not_overwritten(monkeypatch):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: RESULTS)
    agent.call_tool("search_runbooks", query="one")
    agent.call_tool("search_runbooks", query="two")
    lines = read_log()
    assert len(lines) == 2 and '"one"' in lines[0] and '"two"' in lines[1]


def test_a_named_error_is_logged():
    with pytest.raises(agent.AgentError):
        agent.call_tool("launch_missiles", target="moon")
    assert "error=UNKNOWN_TOOL" in read_log()[0]


def test_a_log_problem_never_breaks_the_agent(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(agent, "LOG_FILE", tmp_path / "no_such_folder" / "agent.log")
    monkeypatch.setattr(agent.retrieve, "search", lambda q: RESULTS)
    assert agent.call_tool("search_runbooks", query="cpu") == RESULTS
    assert "LOG_WRITE_FAILED" in capsys.readouterr().err


# ---- The multi-step loop: think, search, read, repeat or stop ----------------

CPU = {"source": "rb.txt", "section": "High CPU on web servers", "text": "Restart the offending worker.", "score": 0.4}
DISK = {"source": "rb.txt", "section": "Disk space almost full", "text": "Rotate the logs.", "score": 0.2}


def cited_sources(answer):
    """The section lines under the Sources block the loop added to the answer."""
    block = answer.split("Sources:", 1)[1]
    return [line[2:] for line in block.splitlines() if line.startswith("- ")]


def test_the_loop_searches_reads_and_searches_again_then_stops(monkeypatch, capsys):
    queries = []

    def search(query):
        queries.append(query)
        if query == "cpu":
            return [CPU, dict(CPU, text="A second chunk of the same section.")]  # same section twice
        return [DISK, CPU]  # CPU was already read

    monkeypatch.setattr(agent.retrieve, "search", search)
    first = tool_reply("search_runbooks", {"query": "cpu"}, "c1")
    first.content = "The problem sounds like high CPU. I need that runbook."
    second = tool_reply("search_runbooks", {"query": "disk"}, "c2")
    second.content = "I have the CPU section. I still want the disk section."
    client = FakeClient(first, second, text_reply("Restart the worker, then rotate the logs."))
    answer = agent.run_agent("cpu is high and the disk is filling up", client=client)
    out = capsys.readouterr().out

    assert queries == ["cpu", "disk"]  # it repeated the search, then stopped
    assert len(client.calls) == 3
    # the model reads the previous results before the next step
    assert "High CPU on web servers" in client.calls[1]["messages"][-1]["content"]
    assert "Disk space almost full" in client.calls[2]["messages"][-1]["content"]
    think1 = out.index("The problem sounds like high CPU")
    chose1 = out.index("Model chose:")
    read1 = out.index("Reading 2 result(s)")
    think2 = out.index("I have the CPU section")
    chose2 = out.index("Model chose:", chose1 + 1)
    stop = out.index("Stop reason: ANSWERED")
    assert think1 < chose1 < read1 < think2 < chose2 < stop
    assert agent.last_trace.stop_reason == agent.ANSWERED
    assert agent.last_trace.steps == 3
    assert agent.last_trace.sources == [
        {"source": "rb.txt", "section": "High CPU on web servers"},
        {"source": "rb.txt", "section": "Disk space almost full"},
    ]


def test_the_answer_cites_only_the_sections_that_were_read(monkeypatch):
    def search(query):
        if query == "cpu":
            return [CPU]
        return [DISK]

    monkeypatch.setattr(agent.retrieve, "search", search)
    client = FakeClient(
        tool_reply("search_runbooks", {"query": "cpu"}, "c1"),
        tool_reply("search_runbooks", {"query": "disk"}, "c2"),
        text_reply("Restart the worker. Certificate expiry is a different problem."),
    )
    answer = agent.run_agent("cpu and disk", client=client)
    assert cited_sources(answer) == [
        "High CPU on web servers (rb.txt)",
        "Disk space almost full (rb.txt)",
    ]
    assert cited_sources(agent.last_trace.answer) == cited_sources(answer)
    # named in the model's prose, but never returned by search, so not a source
    assert "Certificate expiry" not in cited_sources(answer)
    assert "Certificate expiry" in answer.split("Sources:", 1)[0]


def test_a_search_with_no_model_thought_still_says_what_it_needs(monkeypatch, capsys):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: [CPU])
    client = FakeClient(tool_reply("search_runbooks", {"query": "cpu"}), text_reply("Restart the worker."))
    agent.run_agent("cpu high", client=client)
    out = capsys.readouterr().out
    assert "Thinking: I should search the runbooks for 'cpu'." in out
    assert out.index("Thinking:") < out.index("Model chose:")


def test_answering_without_a_search_is_answered_and_cites_nothing():
    client = FakeClient(text_reply("Please give me more detail."))
    answer = agent.run_agent("hi", client=client)
    assert answer == "Please give me more detail."
    assert agent.last_trace.stop_reason == agent.ANSWERED
    assert agent.last_trace.sources == [] and agent.last_trace.steps == 1
    assert "Sources:" not in answer


def test_no_relevant_runbook_says_i_dont_know_and_does_not_guess(monkeypatch, capsys):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: [])
    client = FakeClient(
        tool_reply("search_runbooks", {"query": "pancake recipe"}),
        text_reply("SHOULD NOT BE USED: bake at 350."),
    )
    answer = agent.run_agent("how do I bake a cake", client=client)
    out = capsys.readouterr().out
    assert answer.startswith("I don't know")
    assert "guess" in answer
    assert "SHOULD NOT BE USED" not in answer
    assert len(client.calls) == 1  # the model was not asked to invent an answer
    assert agent.last_trace.stop_reason == agent.NO_RELEVANT_RUNBOOK
    assert agent.last_trace.sources == []
    assert "Stop reason: NO_RELEVANT_RUNBOOK" in out
    assert "No relevant runbook sections found." in out


def test_an_empty_later_search_does_not_wipe_sections_already_read(monkeypatch):
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
    answer = agent.run_agent("cpu high", client=client)
    assert answer.startswith("Restart the worker.")
    assert "I don't know" not in answer
    assert agent.last_trace.stop_reason == agent.ANSWERED
    assert cited_sources(answer) == ["High CPU on web servers (rb.txt)"]


def test_max_steps_reached_stops_and_cites_what_was_read(monkeypatch, capsys):
    monkeypatch.setattr(agent, "MAX_STEPS", 2)
    queries = []

    def search(query):
        queries.append(query)
        return [CPU] if query == "cpu" else [DISK]

    monkeypatch.setattr(agent.retrieve, "search", search)
    client = FakeClient(
        tool_reply("search_runbooks", {"query": "cpu"}, "c1"),
        tool_reply("search_runbooks", {"query": "disk"}, "c2"),
        text_reply("SHOULD NOT BE USED"),
    )
    answer = agent.run_agent("cpu and disk", client=client)
    out = capsys.readouterr().out
    assert queries == ["cpu", "disk"]
    assert len(client.calls) == 2  # no extra model call after the limit
    assert "SHOULD NOT BE USED" not in answer
    assert agent.last_trace.stop_reason == agent.MAX_STEPS_REACHED
    assert agent.last_trace.steps == 2
    assert "Stop reason: MAX_STEPS_REACHED" in out
    assert "Step 3" not in out
    assert cited_sources(answer) == [
        "High CPU on web servers (rb.txt)",
        "Disk space almost full (rb.txt)",
    ]


def test_an_answer_on_the_last_allowed_step_is_answered_not_max_steps(monkeypatch):
    monkeypatch.setattr(agent, "MAX_STEPS", 2)
    monkeypatch.setattr(agent.retrieve, "search", lambda q: [CPU])
    client = FakeClient(
        tool_reply("search_runbooks", {"query": "cpu"}),
        text_reply("Restart the worker."),
    )
    answer = agent.run_agent("cpu high", client=client)
    assert answer.startswith("Restart the worker.")
    assert agent.last_trace.stop_reason == agent.ANSWERED
    assert agent.last_trace.steps == 2
    assert "maximum number of steps" not in answer


def test_a_relevant_section_is_cited_and_a_weak_one_is_not(monkeypatch):
    """Score cutoff: 0.494 is cited, 0.085 is read but not listed in Sources."""
    strong = {
        "source": "sample_runbook.txt", "section": "High CPU on web servers",
        "text": "Restart the offending worker.", "score": 0.494,
    }
    weak = {
        "source": "memory_leaks.txt", "section": "High swap usage",
        "text": "See which process is using swap.", "score": 0.085,
    }
    monkeypatch.setattr(agent.retrieve, "search", lambda q: [strong, weak])
    client = FakeClient(
        tool_reply("search_runbooks", {"query": "web server cpu"}),
        text_reply("The web server is too busy. Restart the worker that is using the CPU."),
    )
    answer = agent.run_agent("web server CPU is at 95 percent", client=client)
    assert cited_sources(answer) == ["High CPU on web servers (sample_runbook.txt)"]
    assert "High swap usage" not in answer.split("Sources:", 1)[1]
    assert agent.last_trace.sources == [
        {"source": "sample_runbook.txt", "section": "High CPU on web servers"},
    ]
    # the weak section was still handed to the model; it is just not a citation
    assert "High swap usage" in client.calls[1]["messages"][-1]["content"]


def test_a_weak_section_is_cited_when_the_answer_names_it(monkeypatch):
    strong = {
        "source": "sample_runbook.txt", "section": "High CPU on web servers",
        "text": "Restart the offending worker.", "score": 0.494,
    }
    weak = {
        "source": "memory_leaks.txt", "section": "High swap usage",
        "text": "See which process is using swap.", "score": 0.085,
    }
    monkeypatch.setattr(agent.retrieve, "search", lambda q: [strong, weak])
    client = FakeClient(
        tool_reply("search_runbooks", {"query": "cpu"}),
        text_reply("Restart the worker. Also check High swap usage, because the box is swapping."),
    )
    answer = agent.run_agent("cpu and memory", client=client)
    assert cited_sources(answer) == [
        "High CPU on web servers (sample_runbook.txt)",
        "High swap usage (memory_leaks.txt)",
    ]


def test_a_section_named_in_the_answer_but_never_read_is_not_cited(monkeypatch):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: [CPU])
    client = FakeClient(
        tool_reply("search_runbooks", {"query": "cpu"}),
        text_reply("Restart the worker. This is not the Certificate expiry section."),
    )
    answer = agent.run_agent("cpu", client=client)
    assert cited_sources(answer) == ["High CPU on web servers (rb.txt)"]
    assert "Certificate expiry" not in cited_sources(answer)


def test_only_weak_matches_do_not_become_an_answer(monkeypatch, capsys):
    """Scores under CITE_MIN_SCORE are not enough to answer or to cite.

    Week 3 kept the highest weak section so Sources was never empty. Week 4
    treats that as a guess: the model is not asked to write steps, and the
    weak titles are not listed as sources. They are still printed, so you
    can see why the guardrail fired.
    """
    weak = {
        "source": "memory_leaks.txt", "section": "High swap usage",
        "text": "See which process is using swap.", "score": 0.08,
    }
    weaker = {
        "source": "rb.txt", "section": "Disk space almost full",
        "text": "Rotate the logs.", "score": 0.06,
    }
    monkeypatch.setattr(agent.retrieve, "search", lambda q: [weak, weaker])
    client = FakeClient(
        tool_reply("search_runbooks", {"query": "memory"}),
        text_reply("SHOULD NOT BE USED: Free some memory on the box."),
    )
    answer = agent.run_agent("the box is swapping", client=client)
    out = capsys.readouterr().out
    assert answer.startswith("I don't know")
    assert "too weak" in answer
    assert "Sources:" not in answer
    assert "SHOULD NOT BE USED" not in answer
    assert "SHOULD NOT BE USED" not in out
    assert "High swap usage" in out  # shown while debugging, not cited
    assert agent.last_trace.sources == []
    assert agent.last_trace.stop_reason == agent.WEAK_MATCH_ONLY
    assert len(client.calls) == 1


def test_nothing_relevant_has_no_citations_and_does_not_guess(monkeypatch):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: [])
    client = FakeClient(
        tool_reply("search_runbooks", {"query": "pancake recipe"}),
        text_reply("SHOULD NOT BE USED: bake at 350."),
    )
    answer = agent.run_agent("how do I bake a cake", client=client)
    assert "Sources:" not in answer
    assert agent.last_trace.sources == []
    assert agent.last_trace.stop_reason == agent.NO_RELEVANT_RUNBOOK
    assert answer.startswith("I don't know")
    assert "SHOULD NOT BE USED" not in answer
    assert len(client.calls) == 1


def test_the_system_prompt_asks_for_a_short_plain_language_answer():
    prompt = agent.SYSTEM_PROMPT
    lowered = prompt.lower()
    assert "plain language" in lowered
    assert "numbered steps" in lowered
    assert "do not paste the runbook text back verbatim" in lowered
    assert "Do not invent sections." in prompt
    assert "say I don't know" in prompt
    assert "nothing relevant" in lowered
    # After one empty or weak search, do not keep changing the words.
    assert "do not search again with different words" in lowered
    assert "runbooks do not cover this" in lowered
    # A request to do something is proposed at once, for a person to approve.
    assert "restart, delete, or roll back" in lowered
    assert "do not search first unless you cannot tell what to act on" in lowered


def test_numbered_steps_without_a_read_section_are_not_shown(capsys):
    client = FakeClient(text_reply("1. Restart the database.\n2. Delete every log."))
    answer = agent.run_agent("the database is down", client=client)
    out = capsys.readouterr().out
    assert answer.startswith("I don't know")
    assert "have not read a runbook section" in answer
    assert "Restart the database" not in answer
    assert "Delete every log" not in answer
    assert "Restart the database" not in out
    assert "Sources:" not in answer
    assert agent.last_trace.stop_reason == agent.UNGROUNDED_ANSWER
    assert agent.last_trace.sources == []
    assert agent.last_trace.steps == 1


def test_a_sources_list_written_by_the_model_is_replaced(monkeypatch):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: [CPU])
    client = FakeClient(
        tool_reply("search_runbooks", {"query": "cpu"}),
        text_reply("Restart the worker.\n\nSources:\n- Certificate expiry (made-up.txt)"),
    )
    answer = agent.run_agent("cpu", client=client)
    assert "Certificate expiry" not in answer
    assert "made-up.txt" not in answer
    assert cited_sources(answer) == ["High CPU on web servers (rb.txt)"]
    assert agent.last_trace.stop_reason == agent.ANSWERED


def test_a_score_on_the_citation_line_is_strong_enough_to_answer(monkeypatch):
    hit = {
        "source": "rb.txt", "section": "High CPU on web servers",
        "text": "Restart the offending worker.", "score": agent.CITE_MIN_SCORE,
    }
    monkeypatch.setattr(agent.retrieve, "search", lambda q: [hit])
    client = FakeClient(
        tool_reply("search_runbooks", {"query": "cpu"}),
        text_reply("Restart the worker."),
    )
    answer = agent.run_agent("cpu high", client=client)
    assert answer.startswith("Restart the worker.")
    assert agent.last_trace.stop_reason == agent.ANSWERED
    assert cited_sources(answer) == ["High CPU on web servers (rb.txt)"]


def test_a_later_weak_search_does_not_erase_a_strong_section(monkeypatch):
    weak = {
        "source": "memory_leaks.txt", "section": "High swap usage",
        "text": "See which process is using swap.", "score": 0.08,
    }

    def search(query):
        if query == "cpu":
            return [CPU]
        return [weak]

    monkeypatch.setattr(agent.retrieve, "search", search)
    client = FakeClient(
        tool_reply("search_runbooks", {"query": "cpu"}, "c1"),
        tool_reply("search_runbooks", {"query": "swap"}, "c2"),
        text_reply("Restart the worker."),
    )
    answer = agent.run_agent("cpu high", client=client)
    assert answer.startswith("Restart the worker.")
    assert "I don't know" not in answer
    assert agent.last_trace.stop_reason == agent.ANSWERED
    assert cited_sources(answer) == ["High CPU on web servers (rb.txt)"]


def test_the_step_budget_refuses_another_call_after_the_limit():
    budget = agent.ModelCallBudget(2)
    assert budget.allow() is True
    assert budget.allow() is True
    assert budget.allow() is False
    assert budget.used == 2  # the refused call is not counted past the limit


def test_max_steps_is_an_explicit_limit_of_four():
    assert agent.MAX_STEPS == 4


class ForeverClient:
    """A model that would call search on every turn, forever."""

    def __init__(self):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        call = tool_reply("search_runbooks", {"query": f"again {len(self.calls)}"})
        return SimpleNamespace(choices=[SimpleNamespace(message=call)])


def test_a_model_that_loops_forever_cannot_exceed_max_steps(monkeypatch, capsys):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: [CPU])
    client = ForeverClient()
    answer = agent.run_agent("cpu high", client=client)
    out = capsys.readouterr().out
    assert len(client.calls) == agent.MAX_STEPS == 4
    assert agent.last_trace.stop_reason == agent.MAX_STEPS_REACHED
    assert agent.last_trace.steps == 4
    assert "maximum number of steps (4)" in answer
    assert "I will not guess" in answer
    assert "Step 5" not in out
    assert "Stop reason: MAX_STEPS_REACHED" in out
    assert cited_sources(answer) == ["High CPU on web servers (rb.txt)"]


def test_a_forever_model_stops_at_a_lower_cap_too(monkeypatch, capsys):
    monkeypatch.setattr(agent, "MAX_STEPS", 3)
    monkeypatch.setattr(agent.retrieve, "search", lambda q: [CPU])
    client = ForeverClient()
    answer = agent.run_agent("cpu high", client=client)
    out = capsys.readouterr().out
    assert len(client.calls) == 3
    assert agent.last_trace.steps == 3
    assert agent.last_trace.stop_reason == agent.MAX_STEPS_REACHED
    assert "maximum number of steps (3)" in answer
    assert "Step 4" not in out


def test_the_page_can_hold_an_action_without_running_it(monkeypatch, capsys):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: [CPU])
    ask = make_asker("y")  # would approve if the terminal gate ran
    result = agent.run_agent("fix cpu", client=action_client(), ask=ask, defer_approval=True)
    out = capsys.readouterr().out
    assert isinstance(result, agent.PendingAction)
    assert result.action == RESTART
    assert ask.prompts == []
    assert "[simulated]" not in out
    assert "needs your approval" not in out
    assert "Waiting for a person" in out
    assert agent.last_trace.stop_reason == agent.ACTION_REQUESTED


def test_settle_action_runs_only_when_approved_is_exactly_true(capsys):
    assert agent.settle_action(RESTART, "yes") == "denied"
    assert "[simulated]" not in capsys.readouterr().out  # the word "yes" is not approval
    result = agent.settle_action(RESTART, True)
    assert result.ok
    assert "pid 1842" in capsys.readouterr().out


def test_reworded_searches_stop_with_a_cannot_find_answer(monkeypatch):
    """A model that keeps changing the words should not use every step.

    The first search looks like a hit. The next two find nothing. That is
    the same miss twice in a row, so the loop says it could not find this.
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
    client = FakeClient(
        tool_reply("search_runbooks", {"query": "postgres vacuum tuning large table"}, "c1"),
        tool_reply("search_runbooks", {"query": "postgres vacuum"}, "c2"),
        tool_reply("search_runbooks", {"query": "vacuum and postgres"}, "c3"),
        text_reply("SHOULD NOT BE USED"),
    )
    answer = agent.run_agent("How do I tune Postgres vacuum for a large table", client=client)
    assert "could not find this in the runbooks" in answer
    assert answer.startswith("I don't know")
    assert "SHOULD NOT BE USED" not in answer
    assert "Sources:" not in answer
    assert len(client.calls) == 3
    assert len(client.calls) <= agent.MAX_STEPS
    assert agent.last_trace.stop_reason == agent.NO_RELEVANT_RUNBOOK
    assert agent.last_trace.steps == 3
    assert agent.last_trace.sources == []


def test_two_weak_searches_in_a_row_stop_without_using_every_step(monkeypatch):
    """Two weak results in a row are the same kind of miss. Stop there."""
    strong = {
        "source": "rb.txt", "section": "High CPU on web servers",
        "text": "Restart the worker.", "score": 0.4,
    }
    weak = {
        "source": "memory_leaks.txt", "section": "High swap usage",
        "text": "See swap.", "score": 0.08,
    }

    def search(query):
        if query == "cpu":
            return [strong]
        return [weak]

    monkeypatch.setattr(agent.retrieve, "search", search)
    client = FakeClient(
        tool_reply("search_runbooks", {"query": "cpu"}, "c1"),
        tool_reply("search_runbooks", {"query": "swap"}, "c2"),
        tool_reply("search_runbooks", {"query": "memory"}, "c3"),
        text_reply("SHOULD NOT BE USED"),
    )
    answer = agent.run_agent("cpu then memory", client=client)
    assert "could not find this in the runbooks" in answer
    assert "too weak" in answer
    assert "SHOULD NOT BE USED" not in answer
    assert "Sources:" not in answer
    assert len(client.calls) == 3
    assert agent.last_trace.stop_reason == agent.WEAK_MATCH_ONLY
    assert agent.last_trace.sources == []


def test_a_fake_action_request_proposes_the_action(monkeypatch, capsys):
    """A request to do something is proposed. Nothing runs until a person agrees.

    The model's own approved flag is ignored. The loop waits.
    """
    seen = []
    monkeypatch.setattr(agent.retrieve, "search", lambda q: seen.append(q) or [])
    client = FakeClient(tool_reply(
        "execute_action",
        {"action": "restart the process 'web-worker'", "approved": True},
    ))
    result = agent.run_agent(
        "Restart the stuck web worker", client=client, ask=make_asker("y"), defer_approval=True,
    )
    out = capsys.readouterr().out
    assert isinstance(result, agent.PendingAction)
    assert result.action == "restart the process 'web-worker'"
    assert agent.last_trace.stop_reason == agent.ACTION_REQUESTED
    assert seen == []
    assert len(client.calls) == 1
    assert "[simulated]" not in out
    assert "Waiting for a person" in out


def test_an_answerable_question_is_unchanged_on_the_practice_model():
    """The built-in practice model still answers a real runbook question.

    No API key is used. This is the same reply as before the search-loop change.
    """
    import app

    question = "The web server CPU is at 95 percent and pages are slow"
    answer = agent.run_agent(question, client=app.DemoClient(), defer_approval=True)
    assert agent.last_trace.stop_reason == agent.ANSWERED
    assert agent.last_trace.steps == 2
    assert answer == (
        "1. Symptom: web servers show CPU above ninety percent and response times climb.\n"
        "\n"
        "Sources:\n"
        "- High CPU on web servers (sample_runbook.txt)"
    )


def test_step_limit_with_nothing_read_says_it_could_not_find_this(monkeypatch, capsys):
    """Hitting the step cap with no strong section still names MAX_STEPS_REACHED.

    The words say the runbooks do not cover this, instead of an empty answer.
    """
    monkeypatch.setattr(agent, "MAX_STEPS", 0)
    client = FakeClient(text_reply("SHOULD NOT BE USED"))
    answer = agent.run_agent("How do I tune Postgres vacuum for a large table", client=client)
    out = capsys.readouterr().out
    assert answer == "I could not find this in the runbooks."
    assert "SHOULD NOT BE USED" not in answer
    assert agent.last_trace.stop_reason == agent.MAX_STEPS_REACHED
    assert agent.last_trace.sources == []
    assert len(client.calls) == 0
    assert "Stop reason: MAX_STEPS_REACHED" in out


def test_an_action_after_a_search_still_stops_at_the_approval_gate(monkeypatch, capsys):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: [CPU])
    client = FakeClient(
        tool_reply("search_runbooks", {"query": "cpu"}, "c1"),
        tool_reply("execute_action", {"action": RESTART}, "c2"),
    )
    result = agent.run_agent("cpu is high, restart the worker", client=client, ask=make_asker("n"))
    out = capsys.readouterr().out
    assert result == "denied"
    assert agent.last_trace.stop_reason == agent.ACTION_REQUESTED
    assert agent.last_trace.sources == [{"source": "rb.txt", "section": "High CPU on web servers"}]
    assert "Stop reason: ACTION_REQUESTED" in out
    assert "needs your approval" in out
    assert "[simulated]" not in out
