"""runbook-copilot agent loop.

A language model reads the user's request and decides which tool to call.
It works in steps: think about what it needs, call search_runbooks, read the
sections, then search again or answer. The answer names the runbook sections
that search actually returned. Every run prints a stop reason and keeps the
same record on last_trace, so you can see why the loop ended.

The model can ask for execute_action, but only a human can allow it: the agent
shows the proposed action and asks y/n. Anything other than y or yes denies it.
execute_action then SIMULATES the action: it parses the action text, prints
realistic steps and returns an ActionResult. It never signals a real process,
and it refuses to run unless the approval gate passed.

Stop reasons (printed as "Stop reason: <name>", also on last_trace.stop_reason):
  ANSWERED              - the model had enough information and wrote an answer
  NO_RELEVANT_RUNBOOK   - search found nothing relevant, so the answer is "I don't know"
  MAX_STEPS_REACHED     - the loop hit MAX_STEPS before an answer and stopped
  ACTION_REQUESTED      - the model asked for execute_action; the approval gate ran

Testing the approval gate without the model:
  python agent.py --force-action   (or -f) skips the model and goes straight to
  the approval prompt with a sample action.

Named failures:
  MISSING_KEY               - XAI_API_KEY is not set in this Terminal tab
  BAD_KEY                   - xAI rejected the key
  MODEL_ERROR               - the model call failed for another reason
  BAD_TOOL_ARGS             - the model's tool call had unusable arguments
  EMPTY_REQUEST             - the user request was empty
  UNKNOWN_TOOL              - the model asked for a tool that does not exist
  NOT_IMPLEMENTED_APPROVAL  - execute_action was called without human approval
  UNKNOWN_ACTION            - the approved action did not match any known action pattern
"""

import json
import os
import re
import sys
import zlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import retrieve

MODEL = "grok-build-0.1"
XAI_BASE_URL = "https://api.x.ai/v1"
MAX_STEPS = 4  # model turns one question may take before the loop stops

# Why the loop stopped. Always printed, and always stored on last_trace.
ANSWERED = "ANSWERED"  # enough information; the model wrote an answer
NO_RELEVANT_RUNBOOK = "NO_RELEVANT_RUNBOOK"  # nothing matched; do not guess
MAX_STEPS_REACHED = "MAX_STEPS_REACHED"  # hit MAX_STEPS before an answer
ACTION_REQUESTED = "ACTION_REQUESTED"  # execute_action; the approval gate decides

SYSTEM_PROMPT = (
    "You are a runbook assistant. Work in steps. "
    "On each step, first think about what you still need to know and write that thought in the message. "
    "Then call search_runbooks with a short, focused query. "
    "Read the sections that come back. If they are not enough, call search_runbooks again with a better query. "
    "When you have enough to answer, stop calling tools and write the answer in plain text, from those sections only. "
    "Do not invent runbook sections. If a search returns nothing relevant, say that you do not know. Do not guess. "
    "If you think a step should be carried out, call execute_action; a human must approve it first. "
    "Write the action as: restart the process '<name>' (for example 'runaway-worker')."
)


class AgentError(Exception):
    """A named, human-readable failure. The first word of the message is the class."""

    def __init__(self, name, message):
        self.name = name
        super().__init__(f"{name}: {message}")


# ---- Tool-call log ---------------------------------------------------------
# One line per tool call, appended to agent.log next to this file:
#   2026-10-29T18:10:00-07:00 | search_runbooks | args={"query": "cpu"} | top="High CPU" score=0.400
#   2026-10-29T18:10:09-07:00 | execute_action | args={"action": "restart ..."} | denied
# Only tool names, arguments and short result summaries are logged, never keys.
# Set LOG_FILE to None to turn logging off.

LOG_FILE = Path(__file__).resolve().parent / "agent.log"


def log_tool_call(tool, arguments, summary):
    """Append one readable line to LOG_FILE. A logging problem never breaks the agent."""
    if LOG_FILE is None:
        return
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    line = f"{stamp} | {tool} | args={json.dumps(arguments, default=str)} | {' '.join(str(summary).split())}\n"
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
    except OSError as e:
        print(f"LOG_WRITE_FAILED: could not write {LOG_FILE} ({type(e).__name__}). Continuing without a log.", file=sys.stderr)


def summarize_search(results):
    if not results:
        return "no results"
    top = results[0]
    return f'top="{top["section"]}" score={top["score"]:.3f} ({len(results)} results)'


# ---- Tools -----------------------------------------------------------------

def search_runbooks(query):
    """Search the runbook knowledge base. Read-only and safe to run any time."""
    return retrieve.search(query)  # looked up at call time so tests can swap it


@dataclass
class ActionResult:
    """What an approved action did. status is "success" or "failure"."""

    status: str
    action: str  # what was done, in plain words
    details: dict = field(default_factory=dict)
    error: str = ""  # a named failure class when status is "failure"

    @property
    def ok(self):
        return self.status == "success"


# Read the action text. Only "restart" is known so far. The name must be given
# after process/worker/service, or be a hyphenated name such as runaway-worker.
RESTART_PATTERNS = [
    re.compile(r"^\s*(?:please\s+)?restart\s+(?:the\s+)?(?:process|worker|service)\s+['\"]?([A-Za-z0-9][\w.-]*)", re.I),
    re.compile(r"^\s*(?:please\s+)?restart\s+(?:the\s+)?['\"]?([A-Za-z0-9]\w*[-_.][\w.-]*)", re.I),
]
PROTECTED_PROCESSES = {"init", "systemd", "sshd"}  # the simulation refuses these
# Made-up process table: name -> (old pid, new pid, CPU percent after restart).
KNOWN_PROCESSES = {"runaway-worker": (1842, 2107, 12)}


def parse_action(action):
    """Return ("restart", process_name), or raise UNKNOWN_ACTION."""
    for pattern in RESTART_PATTERNS:
        m = pattern.match(action or "")
        if m:
            return "restart", m.group(1)
    raise AgentError(
        "UNKNOWN_ACTION",
        f"I do not know how to do: {action!r}. Known action: restart the process '<name>'.",
    )


def simulate_restart(name):
    """Print a realistic restart, step by step. Nothing real is touched."""
    if name.lower() in PROTECTED_PROCESSES:
        print(f"Refusing to stop protected process {name}.")
        return ActionResult(
            "failure", f"restart process {name}", {"process": name, "reason": "protected process"},
            error="PROTECTED_PROCESS",
        )
    if name in KNOWN_PROCESSES:
        old_pid, new_pid, cpu = KNOWN_PROCESSES[name]
    else:  # a steady made-up pid and CPU so the same name always looks the same
        h = zlib.crc32(name.encode())
        old_pid, new_pid, cpu = 1000 + h % 8000, 1000 + h % 8000 + 265, 8 + h % 13
    print(f"[simulated] Sending SIGTERM to process {name} (pid {old_pid})...")
    print("[simulated] Waiting 5 seconds...")  # printed only, nothing actually waits
    print(f"[simulated] Process {name} stopped.")
    print(f"[simulated] Starting replacement {name}...")
    print(f"[simulated] Replacement started with pid {new_pid}.")
    print(f"[simulated] CPU normalized to {cpu}%.")
    return ActionResult(
        "success", f"restart process {name}",
        {"process": name, "old_pid": old_pid, "new_pid": new_pid, "cpu_percent": cpu, "simulated": True},
    )


def execute_action(action, approved=False):
    """SIMULATED. Only runs after a human approved this exact action.

    Reads the action text, prints what a real run would do, and returns an
    ActionResult. No real process is ever touched. `approved` is set by the
    approval gate in run_agent, never by the model.
    """
    if approved is not True:
        raise AgentError(
            "NOT_IMPLEMENTED_APPROVAL",
            "execute_action requires human approval before running anything.",
        )
    kind, name = parse_action(action)  # raises UNKNOWN_ACTION
    return simulate_restart(name)


# The registry: tool name -> function.
TOOLS = {
    "search_runbooks": search_runbooks,
    "execute_action": execute_action,
}

# What the model is told about the tools.
TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "search_runbooks",
            "description": (
                "Look things up in the runbooks. Use this when the user asks a question, describes a problem, "
                "or wants information: 'what does the runbook say about high CPU', 'why is the disk full', "
                "'how do I roll back a bad release', 'DNS name not resolved', 'deployment failed'. "
                "It only reads and changes nothing. Returns the most relevant sections with scores."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Short search words."}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_action",
            "description": (
                "Do something. Use this when the user gives a clear command to carry out: 'restart the web server', "
                "'kill the runaway process', 'restart the web worker', 'stop runaway-worker'. Do not search first for "
                "a clear command. Write the action as: restart the process '<name>' (for example "
                "'runaway-worker'; for 'the web server' use 'web-server'). A human must approve it before it runs, "
                "and for now it is only simulated. Do NOT use it to answer questions or to look things up."
            ),
            "parameters": {
                "type": "object",
                "properties": {"action": {"type": "string", "description": "The exact action to run."}},
                "required": ["action"],
            },
        },
    },
]


def call_tool(name, **arguments):
    if name not in TOOLS:
        log_tool_call(name, arguments, "error=UNKNOWN_TOOL")
        raise AgentError("UNKNOWN_TOOL", f"no tool named '{name}'. Known tools: {', '.join(TOOLS)}.")
    try:
        result = TOOLS[name](**arguments)
    except TypeError:
        log_tool_call(name, arguments, "error=BAD_TOOL_ARGS")
        raise AgentError("BAD_TOOL_ARGS", f"the model gave the wrong arguments for {name}: {arguments}.")
    except AgentError as e:
        log_tool_call(name, arguments, f"error={e.name}")
        raise
    if name == "search_runbooks":  # execute_action is logged by run_gated_action (approved or denied)
        log_tool_call(name, arguments, summarize_search(result))
    return result


# ---- The model -------------------------------------------------------------

def make_client():
    key = os.environ.get("XAI_API_KEY", "").strip()
    if not key:
        raise AgentError("MISSING_KEY", "XAI_API_KEY is not set in this Terminal tab.")
    from openai import OpenAI  # imported here so tests do not need the package

    return OpenAI(api_key=key, base_url=XAI_BASE_URL, timeout=60, max_retries=0)


def ask_model(client, messages, tools=None):
    """One model call. Returns the reply message. Failures become named errors."""
    kwargs = {"model": MODEL, "messages": messages}
    if tools:
        kwargs["tools"] = tools
    try:
        return client.chat.completions.create(**kwargs).choices[0].message
    except Exception as e:  # classify by name so we do not depend on the package here
        if type(e).__name__ == "AuthenticationError":
            raise AgentError("BAD_KEY", "xAI rejected the key. Check XAI_API_KEY.")
        raise AgentError("MODEL_ERROR", f"the model call failed ({type(e).__name__}).")


# ---- Human approval --------------------------------------------------------

def approval_gate(action, ask=input):
    """Show the proposed action with runbook context, then ask a human y/n.

    Returns True only for "y" or "yes" (any case). Everything else denies.
    """
    print("\n--- Proposed action needs your approval ---")
    print("Tool: execute_action")
    print(f"Arguments: {{'action': {action!r}}}")
    try:
        context = retrieve.search(action)  # read-only lookup, for context only
    except retrieve.RetrieveError:
        context = []
    if context:
        print("Related runbook sections:")
        for r in context:
            print(f"  - [{r['score']:.3f}] {r['section']}: {r['text'][:120]}")
    else:
        print("Related runbook sections: none found.")
    answer = ask("Approve this action? (y/n) ")
    return answer.strip().lower() in ("y", "yes")


def print_results(results):
    if not results:
        print("No relevant runbook sections found.")
        return
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['score']:.3f}] {r['section']} ({r['source']})")
        print(f"   {r['text']}\n")


# ---- Running an action behind the gate -------------------------------------

def run_gated_action(action, ask=input):
    """Ask a human, then run execute_action only if they said yes."""
    if not approval_gate(action, ask=ask):
        print("Action denied. Nothing was run.")
        log_tool_call("execute_action", {"action": action}, "denied")
        return "denied"
    # Only the action text comes from outside. approved=True is ours.
    try:
        outcome = call_tool("execute_action", action=action, approved=True)
    except AgentError:  # call_tool already logged the named error
        raise
    log_tool_call("execute_action", {"action": action},
                  f"approved, result={outcome.status}" + (f" error={outcome.error}" if outcome.error else ""))
    line = f"Result: {outcome.status} - {outcome.action}"
    print(line + (f" ({outcome.error})" if outcome.error else ""))
    return outcome


# A made-up action for trying the approval gate without the model.
SAMPLE_ACTION = (
    "restart the process 'runaway-worker' on the web server "
    "(runbook: High CPU on web servers)"
)
FORCE_FLAGS = ("--force-action", "-f")


def run_forced_action(ask=input):
    """Skip the model: go straight to the approval gate with a sample action."""
    arguments = {"action": SAMPLE_ACTION}
    print(f"Forced action (model skipped): execute_action({arguments})")
    return run_gated_action(SAMPLE_ACTION, ask=ask)


# ---- The loop --------------------------------------------------------------

@dataclass
class LoopTrace:
    """What one question run did. stop_reason is always set.

    sources lists the runbook sections search actually returned, in the order
    they were first read. thoughts are what the model (or the search fallback)
    said it needed before each tool call.
    """

    stop_reason: str
    answer: str
    sources: list
    steps: int
    thoughts: list = field(default_factory=list)


# Set when run_agent stops normally. Cleared at the start of each run, so a
# named AgentError leaves None here instead of the previous question.
last_trace = None


def thought_for(name, arguments, model_thought):
    """What this step is looking for.

    The model's own words win. A search with no thought still gets a plain
    sentence, so the "think" step is visible even with a scripted model.
    """
    thought = (model_thought or "").strip()
    if thought or name != "search_runbooks":
        return thought
    query = arguments.get("query") if isinstance(arguments, dict) else ""
    return f"I should search the runbooks for {query!r}."


def remember_sources(sources, results):
    """Keep each runbook section that was actually read, once, in read order."""
    seen = {(item["source"], item["section"]) for item in sources}
    for result in results or []:
        key = (result.get("source"), result.get("section"))
        if not key[1] or key in seen:
            continue
        seen.add(key)
        sources.append({"source": key[0], "section": key[1]})


def show_answer(answer, sources):
    """The answer plus a Sources block naming only the sections that were read."""
    text = (answer or "").rstrip()
    if not sources:
        return text
    lines = [text, "", "Sources:"] if text else ["Sources:"]
    for item in sources:
        lines.append(f"- {item['section']} ({item['source']})")
    return "\n".join(lines)


def record_stop(stop_reason, answer, sources, steps, thoughts):
    """Remember why the loop stopped, and print that name. Returns the answer text."""
    global last_trace
    shown = show_answer(answer, sources)
    last_trace = LoopTrace(
        stop_reason, shown, [dict(item) for item in sources], steps, list(thoughts),
    )
    print(f"Stop reason: {stop_reason}")
    return shown


def append_tool_result(messages, reply, call, results):
    """Hand one tool result back so the model can read it on the next step."""
    messages.append({
        "role": "assistant",
        "content": reply.content,
        "tool_calls": [{
            "id": call.id,
            "type": "function",
            "function": {"name": call.function.name, "arguments": call.function.arguments},
        }],
    })
    messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(results)})


def run_agent(request, client=None, ask=input):
    """Think, search, read, then repeat or stop.

    The model's own sentence is kept. When search ran, the returned answer
    also lists the sections that were read. execute_action still goes through
    the approval gate and is not run from this loop. last_trace always holds
    the stop reason when this function returns normally.
    """
    global last_trace
    last_trace = None
    if not request or not request.strip():
        raise AgentError("EMPTY_REQUEST", "type a question or problem, e.g. 'web server CPU is high'.")
    client = client or make_client()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": request},
    ]
    sources = []
    thoughts = []

    for step in range(1, MAX_STEPS + 1):
        # 1) ask the model what it needs, or whether it can answer now
        print(f"Step {step} of {MAX_STEPS}")
        reply = ask_model(client, messages, tools=TOOL_SPECS)

        # No tool call: the model has enough information (or chose not to search).
        if not getattr(reply, "tool_calls", None):
            shown = record_stop(ANSWERED, reply.content or "", sources, step, thoughts)
            if sources:
                print(f"Answer: {shown}")
                return shown
            print(reply.content)  # same wording as before, when nothing was read
            return reply.content

        # 2) read which tool it chose. One tool call per step; the first one is used.
        call = reply.tool_calls[0]
        name = call.function.name
        try:
            arguments = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            raise AgentError(
                "BAD_TOOL_ARGS",
                f"the arguments for {name} were not valid JSON: {call.function.arguments!r}.",
            )
        thought = thought_for(name, arguments, reply.content)
        if thought:
            thoughts.append(thought)
            print(f"Thinking: {thought}")
        print(f"Model chose: {name}({arguments})")

        # execute_action is never run here. The human approval gate decides.
        if name == "execute_action":
            action = arguments.get("action") if isinstance(arguments, dict) else None
            if not isinstance(action, str) or not action.strip():
                raise AgentError("BAD_TOOL_ARGS", f"execute_action needs an 'action' string: {arguments}.")
            record_stop(ACTION_REQUESTED, action, sources, step, thoughts)
            return run_gated_action(action, ask=ask)

        # 3) run the tool and read what came back
        results = call_tool(name, **arguments)
        if results:
            print(f"Reading {len(results)} result(s) from the runbooks.")
        print_results(results)

        # Nothing relevant anywhere in this run: say so, and do not let the model guess.
        if name == "search_runbooks" and not results and not sources:
            answer = "I don't know. No relevant runbook sections were found, so I will not guess."
            shown = record_stop(NO_RELEVANT_RUNBOOK, answer, sources, step, thoughts)
            print(f"Answer: {shown}")
            return shown

        if name == "search_runbooks":
            remember_sources(sources, results)
        append_tool_result(messages, reply, call, results)
        # 4) loop: the next step can search again, answer, or ask for an action

    # Every allowed step used a tool. Stop instead of calling the model forever.
    answer = (
        f"I stopped because I reached the maximum number of steps ({MAX_STEPS}). "
        "I will not guess beyond the runbook sections I already read."
    )
    shown = record_stop(MAX_STEPS_REACHED, answer, sources, MAX_STEPS, thoughts)
    print(f"Answer: {shown}")
    return shown


def main(argv=None, ask=input):
    args = list(sys.argv[1:] if argv is None else argv)
    force = any(a in FORCE_FLAGS for a in args)
    args = [a for a in args if a not in FORCE_FLAGS]
    try:
        if force:
            return run_forced_action(ask=ask)
        text = " ".join(args) or ask("What's the problem? ")
        return run_agent(text, ask=ask)
    except (AgentError, retrieve.RetrieveError) as e:
        print(e)


if __name__ == "__main__":
    main()
