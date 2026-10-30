"""runbook-copilot agent loop (scaffold).

A language model reads the user's request and decides which tool to call.
The loop runs search_runbooks for it, gives the results back, and prints the
model's final answer. The model can ask for execute_action, but only a human
can allow it: the agent shows the proposed action and asks y/n. Anything other
than y or yes denies it. execute_action itself is still a stub that only
prints, and it refuses to run unless the approval gate passed.

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
"""

import json
import os
import sys

import retrieve

MODEL = "grok-build-0.1"
XAI_BASE_URL = "https://api.x.ai/v1"

SYSTEM_PROMPT = (
    "You are a runbook assistant. To answer a problem, call search_runbooks with a "
    "short, focused query. Answer only from what the search returns. If you think a "
    "step should be carried out, call execute_action; a human must approve it first."
)


class AgentError(Exception):
    """A named, human-readable failure. The first word of the message is the class."""

    def __init__(self, name, message):
        self.name = name
        super().__init__(f"{name}: {message}")


# ---- Tools -----------------------------------------------------------------

def search_runbooks(query):
    """Search the runbook knowledge base. Read-only and safe to run any time."""
    return retrieve.search(query)  # looked up at call time so tests can swap it


def execute_action(action, approved=False):
    """STUB. Only runs after a human approved this exact action.

    When built for real, this will change real systems (restart a worker,
    roll back a deploy, ...). For now it only reports what it would do.
    `approved` is set by the approval gate in run_agent, never by the model.
    """
    if approved is not True:
        raise AgentError(
            "NOT_IMPLEMENTED_APPROVAL",
            "execute_action requires human approval before running anything.",
        )
    return f"[STUB] Approved and executed: {action} (nothing real was changed)"


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
            "description": "Search the runbook knowledge base. Returns the most relevant sections with scores.",
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
            "description": "Carry out an action on a real system. A human must approve it before it runs.",
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
        raise AgentError("UNKNOWN_TOOL", f"no tool named '{name}'. Known tools: {', '.join(TOOLS)}.")
    try:
        return TOOLS[name](**arguments)
    except TypeError:
        raise AgentError("BAD_TOOL_ARGS", f"the model gave the wrong arguments for {name}: {arguments}.")


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
        return "denied"
    # Only the action text comes from outside. approved=True is ours.
    outcome = call_tool("execute_action", action=action, approved=True)
    print(outcome)
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

def run_agent(request, client=None, ask=input):
    """The model picks a tool; we run it; the model writes the final answer."""
    if not request or not request.strip():
        raise AgentError("EMPTY_REQUEST", "type a question or problem, e.g. 'web server CPU is high'.")
    client = client or make_client()

    # 1) send the request and the tool descriptions to the model
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": request},
    ]
    reply = ask_model(client, messages, tools=TOOL_SPECS)

    # The model answered without using a tool.
    if not reply.tool_calls:
        print(reply.content)
        return reply.content

    # 2) read which tool it chose and with what arguments
    call = reply.tool_calls[0]
    name = call.function.name
    try:
        arguments = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError:
        raise AgentError("BAD_TOOL_ARGS", f"the arguments for {name} were not valid JSON: {call.function.arguments!r}.")
    print(f"Model chose: {name}({arguments})")

    # 3) run it. execute_action goes through the human approval gate first.
    if name == "execute_action":
        action = arguments.get("action") if isinstance(arguments, dict) else None
        if not isinstance(action, str) or not action.strip():
            raise AgentError("BAD_TOOL_ARGS", f"execute_action needs an 'action' string: {arguments}.")
        return run_gated_action(action, ask=ask)
    results = call_tool(name, **arguments)
    print_results(results)

    # 4) give the results back to the model for a final answer
    messages.append({
        "role": "assistant",
        "content": reply.content,
        "tool_calls": [{
            "id": call.id,
            "type": "function",
            "function": {"name": name, "arguments": call.function.arguments},
        }],
    })
    messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(results)})
    final = ask_model(client, messages)
    print(f"Answer: {final.content}")
    return final.content


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
