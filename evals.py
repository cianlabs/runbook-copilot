"""Eval harness for runbook-copilot. Runs scenarios through the real agent loop
with a FAKE model, so it needs no API key and makes no network calls.

What it checks per scenario (each failure has a name, so it is easy to root-cause):
  AGENT_ERROR          the agent loop raised one of its named errors
  WRONG_TOOL           the agent ran a different tool than expected (or none)
  NO_QUERY             the tool call had no usable query/action text
  QUERY_OFF_TOPIC      the query did not mention any of the expected words
  MISSING_EXPECTED_RESULT  search results did not contain the expected section/keyword

Run it:  .venv/bin/python evals.py          (fake model)
         .venv/bin/python evals.py --real   (real Grok; needs XAI_API_KEY, small cost)
         add --only=name1,name2 to run just some scenarios
"""
import contextlib
import io
import json
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace

import agent


@dataclass
class Scenario:
    name: str
    request: str                 # what the user types
    model_tool: str              # the tool call the FAKE model will make
    model_args: dict             # ...and its arguments
    expect_tool: str             # "search_runbooks" or "execute_action"
    expect_in_query: tuple = ()  # the query/action should mention at least one of these
    expect_in_results: str = ""  # optional: must appear in a result's section or text


@dataclass
class Outcome:
    name: str
    passed: bool
    failures: list = field(default_factory=list)  # named failure classes
    detail: str = ""  # what the agent actually did, shown for failures


# ---- A fake model: returns scripted replies, never touches the network ------

class FakeClient:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=self.replies.pop(0))])


def _tool_reply(name, arguments):
    call = SimpleNamespace(id="call_1", function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))
    return SimpleNamespace(content=None, tool_calls=[call])


def _text_reply(text):
    return SimpleNamespace(content=text, tool_calls=None)


# ---- Running one scenario ----------------------------------------------------

def run_scenario(scenario, ask=lambda prompt="": "n", client=None):
    """Run one scenario through agent.run_agent and score it. Never raises.
    Pass a real client to let the real model choose the tool; default is the scripted fake."""
    calls = []  # (tool name, argument text, results)
    real_call_tool, real_gated = agent.call_tool, agent.run_gated_action

    def spy_call_tool(name, **arguments):
        results = real_call_tool(name, **arguments)
        calls.append((name, arguments.get("query"), results))
        return results

    def spy_gated(action, ask=input):
        calls.append(("execute_action", action, None))
        return real_gated(action, ask=ask)

    client = client or FakeClient(
        _tool_reply(scenario.model_tool, scenario.model_args),
        _text_reply("(fake final answer)"),
    )
    agent.call_tool, agent.run_gated_action = spy_call_tool, spy_gated
    real_log, agent.LOG_FILE = agent.LOG_FILE, None  # eval runs stay out of agent.log
    failures = []
    printed = io.StringIO()
    try:
        with contextlib.redirect_stdout(io.StringIO()) as printed:  # keep the summary readable
            agent.run_agent(scenario.request, client=client, ask=ask)
    except agent.AgentError as e:
        failures.append(f"AGENT_ERROR:{e.name}")
    finally:
        agent.call_tool, agent.run_gated_action = real_call_tool, real_gated
        agent.LOG_FILE = real_log

    failures += score(scenario, calls)
    return Outcome(scenario.name, not failures, failures, describe(calls, printed.getvalue()))


def describe(calls, printed):
    """One plain line about what the agent did: the tool, its query, and the sections it found."""
    if not calls:
        said = " ".join(printed.split())[:200]
        return f"no tool was called. The model said: {said!r}"
    tool, text, results = calls[0]
    line = f"tool={tool}, {'action' if tool == 'execute_action' else 'query'}={text!r}"
    if results is not None:
        line += ", top sections=" + "; ".join(r.get("section", "?") for r in results[:3]) if results else ", no results"
    return line


def score(scenario, calls):
    """Pure scoring: compare what the agent did with what the scenario expected."""
    if not calls or calls[0][0] != scenario.expect_tool:
        return ["WRONG_TOOL"]
    _tool, text, results = calls[0]
    if not isinstance(text, str) or not text.strip():
        return ["NO_QUERY"]
    failures = []
    lowered = text.lower()
    if scenario.expect_in_query and not any(w.lower() in lowered for w in scenario.expect_in_query):
        failures.append("QUERY_OFF_TOPIC")
    if scenario.expect_in_results:
        wanted = scenario.expect_in_results.lower()
        haystack = " ".join(f"{r.get('section', '')} {r.get('text', '')}" for r in (results or [])).lower()
        if wanted not in haystack:
            failures.append("MISSING_EXPECTED_RESULT")
    return failures


# ---- Summary -----------------------------------------------------------------

def summarize(outcomes):
    passed = sum(1 for o in outcomes if o.passed)
    lines = [f"{passed}/{len(outcomes)} passed"]
    failed = [o for o in outcomes if not o.passed]
    if failed:
        lines.append("Failing scenarios:")
        for o in failed:
            lines.append(f"  - {o.name}: {', '.join(o.failures)}")
            if o.detail:
                lines.append(f"      saw: {o.detail}")
    return "\n".join(lines)


# ---- The scenarios -----------------------------------------------------------

def search(name, request, query, expect_query, expect_result):
    return Scenario(name, request, "search_runbooks", {"query": query}, "search_runbooks",
                    expect_query, expect_result)


SCENARIOS = [
    search("high-cpu", "web server CPU is at 95 percent", "high CPU web server", ("cpu",), "High CPU"),
    search("disk-full", "the disk is almost full", "disk space full", ("disk",), "disk"),
    search("dns-failure", "DNS name not resolved", "DNS name not resolved", ("dns",), "dns"),
    search("memory-leak", "memory keeps growing until the app dies", "memory leak growing", ("memory",), "memory"),
    search("deploy-rollback", "my deployment failed, what now", "deployment failed rollback", ("deploy",), "deploy"),
    search("coredns-timeout", "CoreDNS shows i/o timeout", "CoreDNS i/o timeout", ("coredns", "timeout"), "coredns"),
    search("etcd-space", "etcd says database space exceeded", "etcd database space exceeded", ("etcd",), "etcd"),
    search("yum-lock", "yum fails, lockfile held by another process", "yum lockfile held", ("yum", "lock"), "lock"),
    Scenario("restart-runaway-worker", "restart the runaway worker", "execute_action",
             {"action": "restart the process 'runaway-worker'"}, "execute_action", ("restart", "runaway-worker")),
    Scenario("restart-web-worker", "restart the web worker, it is stuck", "execute_action",
             {"action": "restart the process 'web-worker'"}, "execute_action", ("restart", "web-worker")),
]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    client = None
    if "--real" in argv:  # the real model chooses the tools; needs XAI_API_KEY, makes real API calls
        try:
            client = agent.make_client()
        except agent.AgentError as e:
            print(e)
            return 2
        print("Running against the REAL model (real API calls).\n")
    only = next((a.split("=", 1)[1].split(",") for a in argv if a.startswith("--only=")), None)
    scenarios = [s for s in SCENARIOS if only is None or s.name in only]
    outcomes = [run_scenario(s, client=client) for s in scenarios]
    for o in outcomes:
        print(f"{'PASS' if o.passed else 'FAIL'}  {o.name}" + ("" if o.passed else f"  ({', '.join(o.failures)})"))
    print()
    print(summarize(outcomes))
    return 0 if all(o.passed for o in outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
