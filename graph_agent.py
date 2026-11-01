"""The agent loop, drawn as boxes and arrows.

`agent.py` walks in a circle: think, search, read, then do that again or stop.
This file is the same work as a picture. Each box does one small job and
returns the name of the next box. A walker follows those names until a box
says "stop". People call this kind of picture a state machine.

The page still uses `agent.py`. This file is a second copy of the same rules
so the two can be compared. It uses the same tools, the same stop reasons, and
the same named errors. LangGraph can come later. It is not used here.

```mermaid
flowchart TD
    startBox([Start]) --> PLAN
    PLAN -->|Question| SEARCH
    PLAN -->|Action request| ACTION_GATE
    PLAN -->|Wrote an answer| CHECK
    SEARCH --> CHECK
    CHECK -->|Nothing matched| IDK
    CHECK -->|Scores under 0.15| IDK
    CHECK -->|Answer cites nothing| IDK
    CHECK -->|Asking for more detail| ANSWER
    CHECK -->|Strong sections| ANSWER
    CHECK -->|Citations look good| ANSWER
    ANSWER -->|Search again| SEARCH
    ANSWER -->|Check the citations| CHECK
    ANSWER -->|Save the reply| answered([Stop: ANSWERED])
    IDK --> idkStop([Stop: I do not know])
    ACTION_GATE --> actionStop([Stop: ACTION_REQUESTED])
```

What each box does:

- PLAN: read the request (or the sections just found) and decide. A question
  goes to SEARCH. An action request goes to ACTION_GATE. Words with no tool
  call go to CHECK.
- SEARCH: call the existing `search_runbooks` tool.
- CHECK: the guardrails. Nothing matched is NO_RELEVANT_RUNBOOK. Every score
  under 0.15 is WEAK_MATCH_ONLY. An answer that cites nothing is
  UNGROUNDED_ANSWER. A real match goes on to ANSWER. Two searches in a row
  that are both empty, or both weak, also go to IDK. That stops a model
  that only changes the words.
- ANSWER: write the reply from the sections that were read, then list sources.
- IDK: stop with the reason CHECK chose, or with MAX_STEPS_REACHED.
- ACTION_GATE: the same human y/n question as today. Stop with ACTION_REQUESTED.

The walker counts model steps, the same idea as `MAX_STEPS` in `agent.py`.
Four is the most. A fifth try to call the model goes to IDK with
MAX_STEPS_REACHED. Only PLAN spends a step every time. ANSWER spends a step
only when it still needs the model to write. SEARCH, CHECK, IDK, and
ACTION_GATE are free.

`run_graph(request, ...)` takes the same arguments as `run_agent` and returns
the same kind of result. The extra piece is `last_trace.node_path`: the list
of boxes the walker visited, in order.

On a normal question the path is PLAN, SEARCH, CHECK, ANSWER, CHECK, ANSWER.
The first ANSWER asks the model for words. CHECK makes sure those words cite
a section that was really read. The second ANSWER saves the reply and stops.

Stop reasons (same names as agent.py):
  ANSWERED, NO_RELEVANT_RUNBOOK, WEAK_MATCH_ONLY, UNGROUNDED_ANSWER,
  MAX_STEPS_REACHED, ACTION_REQUESTED.

Named failures are the ones `agent.py` already raises: MISSING_KEY, BAD_KEY,
MODEL_ERROR, BAD_TOOL_ARGS, EMPTY_REQUEST, UNKNOWN_TOOL,
NOT_IMPLEMENTED_APPROVAL, and UNKNOWN_ACTION.
"""

import json
import sys
from dataclasses import dataclass, field

import agent
import retrieve

# Same cap as agent.py. The walker reads this on every question, so a test
# can lower it. A run never calls the model more times than this.
MAX_STEPS = 4

# Set at the end of a normal run. Cleared at the start, so a named error
# leaves None here instead of the previous question.
last_trace = None

# The six boxes. The walker only follows these names.
NODE_NAMES = ("PLAN", "SEARCH", "CHECK", "ANSWER", "IDK", "ACTION_GATE")


@dataclass
class GraphTrace:
    """What one graph run did.

    Same fields as the loop's trace, plus node_path: the boxes visited, in
    order. `steps` counts model calls, not boxes.
    """

    stop_reason: str
    answer: str
    sources: list
    steps: int
    thoughts: list
    events: list
    node_path: list


@dataclass
class GraphState:
    """The sheet of paper the boxes pass along. One question, one sheet."""

    request: str
    client: object = None
    ask: object = input
    defer_approval: bool = False
    messages: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    thoughts: list = field(default_factory=list)
    events: list = field(default_factory=list)
    results: list = field(default_factory=list)
    arguments: dict = field(default_factory=dict)
    reply: object = None
    call: object = None
    action: str = ""
    draft_answer: str = ""
    has_draft: bool = False  # True once the model has written an answer
    # "strong", "weak", or "none" for the search before this one. "" means none yet.
    last_search_quality: str = ""
    stop_reason: str = ""
    answer_text: str = ""
    steps: int = 0
    node_path: list = field(default_factory=list)
    return_value: object = None


def new_state(request, client, ask=input, defer_approval=False):
    """A blank sheet for one question, with the same opening messages as the loop."""
    return GraphState(
        request=request,
        client=client,
        ask=ask,
        defer_approval=defer_approval,
        messages=[
            {"role": "system", "content": agent.SYSTEM_PROMPT},
            {"role": "user", "content": request},
        ],
    )


def save_trace(state, stop_reason, answer_text):
    """Remember why we stopped, and print that name. Returns the answer text."""
    global last_trace
    cited = agent.citations_for(state.sources, answer_text)
    shown = agent.show_answer(answer_text, cited)
    events = list(state.events)
    plain = agent.STOP_REASON_PLAIN.get(stop_reason, "")
    agent.note_event(events, state.steps, "Stopped", f"{stop_reason}. {plain}".strip())
    last_trace = GraphTrace(
        stop_reason=stop_reason,
        answer=shown,
        sources=[dict(item) for item in cited],
        steps=state.steps,
        thoughts=list(state.thoughts),
        events=events,
        node_path=list(state.node_path),
    )
    print(f"Stop reason: {stop_reason}")
    if plain:
        print(plain)
    print("Nodes visited: " + " -> ".join(state.node_path))
    return shown


def stop_at_limit(state):
    """The counter is full. Stop instead of calling the model again.

    If no strong section was read, say we could not find this in the
    runbooks. The stop reason stays MAX_STEPS_REACHED either way.
    """
    answer, sources = agent.step_limit_answer(state.sources, MAX_STEPS)
    state.stop_reason = agent.MAX_STEPS_REACHED
    state.answer_text = answer
    state.sources = sources
    if not state.node_path or state.node_path[-1] != "IDK":
        state.node_path.append("IDK")
    idk(state)
    return state.return_value


def spends_a_step(node, state):
    """True when this box is about to call the model.

    PLAN always asks the model. ANSWER asks only when it does not have words
    yet. Saving a reply that CHECK already approved does not cost a step.
    """
    if node == "PLAN":
        return True
    if node == "ANSWER" and not state.has_draft:
        return True
    return False


def choose_next(state, reply):
    """Read one model reply and return the name of the next box.

    A tool call is a question (SEARCH) or an action request (ACTION_GATE).
    Words, and no tool call, are an answer for CHECK to look at.
    """
    if not getattr(reply, "tool_calls", None):
        state.has_draft = True
        state.draft_answer = agent.strip_model_sources(getattr(reply, "content", None) or "")
        return "CHECK"

    # One tool call per step. If the model sends several, the first one is used.
    call = reply.tool_calls[0]
    name = call.function.name
    try:
        arguments = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError:
        raise agent.AgentError(
            "BAD_TOOL_ARGS",
            f"the arguments for {name} were not valid JSON: {call.function.arguments!r}.",
        )
    if not isinstance(arguments, dict):
        raise agent.AgentError(
            "BAD_TOOL_ARGS",
            f"the model gave the wrong arguments for {name}: {arguments}.",
        )

    thought = agent.thought_for(name, arguments, getattr(reply, "content", None))
    if thought:
        state.thoughts.append(thought)
        print(f"Thinking: {thought}")
        agent.note_event(state.events, state.steps, "Thinking", thought)
    print(f"Model chose: {name}({arguments})")
    agent.note_event(state.events, state.steps, "Tool", agent.tool_event_detail(name, arguments))
    state.reply = reply
    state.call = call
    state.arguments = arguments

    if name == "execute_action":
        action = arguments.get("action")
        if not isinstance(action, str) or not action.strip():
            raise agent.AgentError(
                "BAD_TOOL_ARGS",
                f"execute_action needs an 'action' string: {arguments}.",
            )
        # The model's own "approved" flag is ignored. Only a person can approve.
        state.action = action
        return "ACTION_GATE"
    if name == "search_runbooks":
        return "SEARCH"
    # Not a tool we know. call_tool raises UNKNOWN_TOOL, same as the loop.
    agent.call_tool(name, **arguments)
    raise agent.AgentError("UNKNOWN_TOOL", f"no tool named '{name}'. Known tools: {', '.join(agent.TOOLS)}.")


# ---- The boxes -------------------------------------------------------------

def plan(state):
    """Decide whether this is a question, an action request, or an answer."""
    print(f"Step {state.steps} of {MAX_STEPS}")
    reply = agent.ask_model(state.client, state.messages, tools=agent.TOOL_SPECS)
    return choose_next(state, reply)


def search(state):
    """Call the existing search_runbooks tool and hold onto the sections."""
    state.results = agent.call_tool("search_runbooks", **state.arguments)
    if state.results:
        print(f"Reading {len(state.results)} result(s) from the runbooks.")
    agent.print_results(state.results)
    agent.note_event(state.events, state.steps, "Read", agent.describe_results(state.results))
    return "CHECK"


def check(state):
    """Apply the same guardrails as the loop. Return IDK or ANSWER.

    Nothing matched -> NO_RELEVANT_RUNBOOK.
    Every score under 0.15, and nothing stronger was read earlier -> WEAK_MATCH_ONLY.
    An answer that cites nothing -> UNGROUNDED_ANSWER.
    A short "please give me more detail" is not a guess. It may be shown.
    Two searches in a row that are both empty, or both only weak, also stop.
    That is the same "I don't know" idea. It does not depend on the model
    deciding to stop.
    """
    if state.has_draft:
        if not state.sources and agent.reply_is_only_asking_for_detail(state.draft_answer):
            return "ANSWER"
        cited = agent.citations_for(state.sources, state.draft_answer)
        if not cited:
            state.stop_reason = agent.UNGROUNDED_ANSWER
            if state.sources:
                state.answer_text = (
                    "I don't know. The answer did not cite a runbook section, so I will not guess."
                )
            else:
                # Same words as agent.py when nothing has been read yet.
                state.answer_text = "I don't know. I have not read a runbook section, so I will not guess."
            return "IDK"
        return "ANSWER"

    # Just searched. Do not let a weak or empty result become an answer
    # when nothing stronger was read earlier. This is the first weak search.
    if not agent.has_strong_support(state.results) and not agent.has_strong_support(state.sources):
        if not state.results and not state.sources:
            state.stop_reason = agent.NO_RELEVANT_RUNBOOK
            state.answer_text = "I don't know. No relevant runbook sections were found, so I will not guess."
        else:
            state.stop_reason = agent.WEAK_MATCH_ONLY
            state.answer_text = "I don't know. The only runbook matches were too weak to trust, so I will not guess."
        return "IDK"

    # The model searched again, and this result is just as weak as the last
    # one. Stop. A stray earlier hit is not listed as a source.
    reason, text, quality = agent.repeated_search_stop(state.last_search_quality, state.results)
    state.last_search_quality = quality
    if reason:
        state.stop_reason = reason
        state.answer_text = text
        state.sources = []
        return "IDK"

    agent.remember_sources(state.sources, state.results)
    if state.reply is not None and state.call is not None:
        agent.append_tool_result(state.messages, state.reply, state.call, state.results)
    return "ANSWER"


def answer(state):
    """Produce the grounded answer and its sources.

    The first visit asks the model to write. CHECK then looks at citations.
    The second visit saves the reply and stops. Returning None means stop.
    """
    if not state.has_draft:
        print(f"Step {state.steps} of {MAX_STEPS}")
        reply = agent.ask_model(state.client, state.messages, tools=agent.TOOL_SPECS)
        return choose_next(state, reply)

    text = state.draft_answer
    shown = save_trace(state, agent.ANSWERED, text)
    if state.sources:
        print(f"Answer: {shown}")
        state.return_value = shown
    else:
        print(text)  # a short request for more detail, same as the loop
        state.return_value = text
    return None


def idk(state):
    """Stop with the reason already stored on the state. Returning None means stop."""
    shown = save_trace(state, state.stop_reason, state.answer_text)
    print(f"Answer: {shown}")
    state.return_value = shown
    return None


def action_gate(state):
    """Ask a person, then stop with ACTION_REQUESTED. The model cannot approve itself."""
    save_trace(state, agent.ACTION_REQUESTED, state.action)
    if state.defer_approval:
        # The caller shows the action and waits. Nothing runs here.
        print("Action not run. Waiting for a person to approve it.")
        state.return_value = agent.PendingAction(state.action)
        return None
    state.return_value = agent.run_gated_action(state.action, ask=state.ask)
    return None


NODES = {
    "PLAN": plan,
    "SEARCH": search,
    "CHECK": check,
    "ANSWER": answer,
    "IDK": idk,
    "ACTION_GATE": action_gate,
}


# ---- The walker ------------------------------------------------------------

def walk(state):
    """Start at PLAN and follow the arrows until a box returns None.

    Before PLAN or a writing ANSWER, the counter is checked. When it is
    already at MAX_STEPS, the walker goes to IDK and does not call the model.
    """
    node = "PLAN"
    visits = 0
    while node:
        visits += 1
        # A bug that loops forever should still stop. Normal runs stay well under this.
        if visits > MAX_STEPS * 6:
            return stop_at_limit(state)
        if spends_a_step(node, state):
            if state.steps >= MAX_STEPS:
                return stop_at_limit(state)
            state.steps += 1
        state.node_path.append(node)
        node = NODES[node](state)
    return state.return_value


def run_graph(request, client=None, ask=input, defer_approval=False):
    """Run one request through the boxes.

    Same arguments and same kind of return value as `agent.run_agent`.
    After a normal return, `last_trace` holds the stop reason and `node_path`.
    A named AgentError leaves `last_trace` as None.
    """
    global last_trace
    last_trace = None
    if not request or not request.strip():
        raise agent.AgentError(
            "EMPTY_REQUEST",
            "type a question or problem, e.g. 'web server CPU is high'.",
        )
    client = client or agent.make_client()
    state = new_state(request, client, ask=ask, defer_approval=defer_approval)
    return walk(state)


def main(argv=None, ask=input):
    """Ask one question from the command line. The page still uses agent.py."""
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        text = " ".join(args) or ask("What's the problem? ")
        return run_graph(text, ask=ask)
    except (agent.AgentError, retrieve.RetrieveError) as e:
        print(e)


if __name__ == "__main__":
    main()
