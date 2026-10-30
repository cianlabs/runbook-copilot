import json
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
    assert answer == "Restart the worker."
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


def action_client(action="restart the offending worker"):
    return FakeClient(tool_reply("execute_action", {"action": action}))


def test_approval_prompt_shows_the_action_summary(monkeypatch, capsys):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: RESULTS)
    ask = make_asker("n")
    agent.run_agent("fix cpu", client=action_client(), ask=ask)
    out = capsys.readouterr().out
    assert "needs your approval" in out
    assert "execute_action" in out
    assert "restart the offending worker" in out
    assert "High CPU on web servers" in out  # runbook context
    assert ask.prompts == ["Approve this action? (y/n) "]


@pytest.mark.parametrize("answer", ["y", "yes", "Y", " YES "])
def test_yes_allows_the_action(monkeypatch, capsys, answer):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: RESULTS)
    result = agent.run_agent("fix cpu", client=action_client(), ask=make_asker(answer))
    assert "[STUB] Approved and executed: restart the offending worker" in result
    assert "Approved and executed" in capsys.readouterr().out


def test_n_denies_the_action(monkeypatch, capsys):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: RESULTS)
    result = agent.run_agent("fix cpu", client=action_client(), ask=make_asker("n"))
    assert result == "denied"
    out = capsys.readouterr().out
    assert "Action denied. Nothing was run." in out
    assert "Approved and executed" not in out


@pytest.mark.parametrize("answer", ["", "   ", "maybe", "sure", "ok", "yep", "no"])
def test_empty_or_invalid_answer_denies_the_action(monkeypatch, capsys, answer):
    monkeypatch.setattr(agent.retrieve, "search", lambda q: RESULTS)
    result = agent.run_agent("fix cpu", client=action_client(), ask=make_asker(answer))
    assert result == "denied"
    assert "Approved and executed" not in capsys.readouterr().out


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
def test_force_flag_y_approves_and_runs_the_stub(no_model, capsys, answer):
    result = agent.main(["--force-action"], ask=make_asker(answer))
    assert "[STUB] Approved and executed: restart the process 'runaway-worker'" in result
    assert "nothing real was changed" in capsys.readouterr().out


@pytest.mark.parametrize("answer", ["n", "no", "", "maybe"])
def test_force_flag_n_or_anything_else_denies(no_model, capsys, answer):
    result = agent.main(["-f"], ask=make_asker(answer))
    assert result == "denied"
    out = capsys.readouterr().out
    assert "Action denied. Nothing was run." in out
    assert "Approved and executed" not in out


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
    assert result == "Restart the worker."
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
