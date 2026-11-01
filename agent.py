"""runbook-copilot agent loop.

A language model reads the user's request and decides which tool to call.
It works in steps: think about what it needs, call search_runbooks, read the
sections, then search again or answer. The answer is a short plain-language
reply. Its Sources list names the read sections the answer relies on (see
the citation rule on CITE_MIN_SCORE), never a section search did not return.
Every run prints a stop reason and keeps the same record on last_trace, so
you can see why the loop ended.

The model can ask for execute_action, but only a human can allow it: the agent
shows the proposed action and asks y/n. Anything other than y or yes denies it.
execute_action then SIMULATES the action: it parses the action text, prints
realistic steps and returns an ActionResult. It never signals a real process,
and it refuses to run unless the approval gate passed.

Stop reasons (printed as "Stop reason: <name>", also on last_trace.stop_reason):
  ANSWERED              - the model had enough information and wrote an answer
  NO_RELEVANT_RUNBOOK   - search found nothing relevant, so the answer is "I don't know"
  WEAK_MATCH_ONLY       - search found only weak matches, so the answer is "I don't know"
  UNGROUNDED_ANSWER     - the model tried to give steps before reading any section
  MAX_STEPS_REACHED     - the loop hit MAX_STEPS before an answer and stopped
  ACTION_REQUESTED      - the model asked for execute_action; a person still has to approve it

WEAK_MATCH_ONLY and UNGROUNDED_ANSWER are the same idea as NO_RELEVANT_RUNBOOK:
the runbooks do not support an answer, so the loop says "I don't know" and does
not guess. They have their own names so a log shows which guardrail fired.

A runbook that only looks similar is also NO_RELEVANT_RUNBOOK. Before the
answer is kept, the section has to be about the same problem. See
runbook_match_failure. That check does not call the model.

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
# Hard cap on model turns for one question. The loop reads this every run
# (tests can change it) and checks it before every model call. A model that
# keeps asking for tools forever still cannot take more turns than this.
MAX_STEPS = 4
# A read section at or above this score is strong enough to answer from and
# to cite. Search itself keeps weaker hits (retrieve.MIN_SCORE is 0.05).
# 0.15 sits above that tail (a stray "High swap usage" hit scored 0.085)
# and below a real match. Scores under this line are too weak to answer from
# when they are the only thing search returned.
CITE_MIN_SCORE = 0.15
# A section has to score at least this high before its words can support an
# answer. This is the same line as CITE_MIN_SCORE. A weaker hit can still be
# printed. It is not enough to count as the same problem.
MATCH_MIN_SCORE = CITE_MIN_SCORE
# How many distinctive words from the question have to show up in that
# section. One is enough: "cpu" in a CPU runbook, or "yum" in a yum runbook.
# Zero means the runbook never actually named the problem.
DISTINCTIVE_WORD_MIN = 1
# Shorter than this is skipped. "a" and "to" do not identify a problem.
MIN_WORD_LENGTH = 3

# Why the loop stopped. Always printed, and always stored on last_trace.
ANSWERED = "ANSWERED"  # enough information; the model wrote an answer
NO_RELEVANT_RUNBOOK = "NO_RELEVANT_RUNBOOK"  # nothing matched; do not guess
WEAK_MATCH_ONLY = "WEAK_MATCH_ONLY"  # only weak scores; do not guess
UNGROUNDED_ANSWER = "UNGROUNDED_ANSWER"  # steps were written before any section was read
MAX_STEPS_REACHED = "MAX_STEPS_REACHED"  # hit MAX_STEPS before an answer
ACTION_REQUESTED = "ACTION_REQUESTED"  # execute_action; a person still has to approve

# Plain sentences for the page and the terminal. The code name stays on the
# "Stop reason:" line so logs are easy to search.
STOP_REASON_PLAIN = {
    ANSWERED: "The copilot answered from the runbook sections it read.",
    NO_RELEVANT_RUNBOOK: "The copilot does not know. Nothing in the runbooks matched, so it will not guess.",
    WEAK_MATCH_ONLY: "The copilot does not know. The only matches were too weak to trust, so it will not guess.",
    UNGROUNDED_ANSWER: "The copilot does not know. It tried to give steps before reading a runbook section, so it will not guess.",
    MAX_STEPS_REACHED: "The copilot stopped because it used every step it is allowed. It will not keep going or guess.",
    ACTION_REQUESTED: "The copilot wants to do something. Nothing has run. It waits for you to approve or refuse.",
}

SYSTEM_PROMPT = (
    "You are a runbook assistant. Work in steps. "
    "On each step, first think about what you still need to know and write that thought in the message. "
    "Then call search_runbooks with a short, focused query. "
    "Read the sections that come back. "
    "If a search returns nothing, or only weak matches, do not search again with different words. "
    "Say the runbooks do not cover this. "
    "If the sections are on the right topic but miss one detail, you may search one more time. "
    "When you have enough to answer, stop calling tools. "
    "Write a short reply in plain language: a few numbered steps the person can follow. "
    "Ground it only in the sections you read, and say it in your own words. "
    "Do not paste the runbook text back verbatim, and do not put that text under a bold title. "
    "Do not invent sections. "
    "If nothing relevant was found, say I don't know. Do not guess. "
    "Before you answer, check that the runbook is about the same problem and the same kind of system as the question. "
    "A server is not a laptop, a phone, or a browser. "
    "The tool named in the question has to be the tool in the runbook. "
    "The symptom has to be the same symptom. "
    "If the runbook is only similar, say the runbooks do not cover this. "
    "In one short sentence, say what the closest runbook covers. "
    "Do not give its steps as the answer. "
    "If you have not read a section yet, do not write numbered steps and do not invent a Sources list. "
    "Ask for more detail, or say I don't know. "
    "If the person asks you to restart, delete, or roll back something, call execute_action right away "
    "so a human can approve it. Do not search first unless you cannot tell what to act on. "
    "Write a restart as: restart the process '<name>' (for example 'runaway-worker')."
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
                "It only reads and changes nothing. Returns the most relevant sections with scores. "
                "If the results are empty or only weak matches, do not call this again with different words."
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
                "Do something. Use this when the user gives a clear command to carry out: 'restart the web worker', "
                "'delete old logs', 'roll back the release'. Call this right away. Do not search first unless the "
                "target is unclear. Write a restart as: restart the process '<name>' (for example "
                "'runaway-worker'). A human must approve it before it runs, and for now it is only simulated. "
                "Do NOT use it to answer questions or to look things up."
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

def settle_action(action, approved):
    """Run the simulation only when approved is exactly True. Anything else denies.

    The terminal gate and the page's Approve button both end up here.
    approved=True is set by our code after a person agrees, never by the model.
    """
    if approved is not True:
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


def run_gated_action(action, ask=input):
    """Ask a human, then run execute_action only if they said yes."""
    return settle_action(action, approval_gate(action, ask=ask))


@dataclass
class PendingAction:
    """An action the model asked for. Nothing has run yet.

    The page returns this instead of asking y/n in the terminal. The action
    runs only after a person clicks Approve, which calls settle_action.
    """

    action: str


class ModelCallBudget:
    """Counts model calls and refuses any call past the limit.

    run_agent asks allow() before every model call. When allow() is False the
    loop stops with MAX_STEPS_REACHED and does not call the model. A model
    that would answer forever still cannot take more than `limit` turns.
    """

    def __init__(self, limit):
        self.limit = limit
        self.used = 0

    def allow(self):
        """True if another model call is still allowed. That call is then counted."""
        if self.used >= self.limit:
            return False
        self.used += 1
        return True


def max_steps_message(limit):
    """What we tell the person when the hard cap stops the loop."""
    return (
        f"I stopped because I reached the maximum number of steps ({limit}). "
        "I will not guess beyond the runbook sections I already read. "
        "Ask a narrower question if you want another try."
    )


def search_quality(results):
    """How useful this one search was: "strong", "weak", or "none".

    "strong" means some score is at least CITE_MIN_SCORE. "weak" means every
    score is under that line. "none" means the search returned no rows.
    """
    if has_strong_support(results):
        return "strong"
    if not results:
        return "none"
    return "weak"


def repeated_search_stop(previous_quality, results):
    """Stop when the model searches again and the result is just as weak.

    previous_quality is "" before any search, then "strong", "weak", or "none".
    Two "none" results in a row, or two "weak" results in a row, means the
    model is only changing the words. Stop with the same "I don't know" idea
    as an empty search. A strong hit resets the count. The first weak or
    empty search does not stop here, so a single search still behaves as before.

    Returns (stop_reason, answer, quality). stop_reason is None when the
    loop should keep going.
    """
    quality = search_quality(results)
    if quality == "strong" or previous_quality != quality:
        return None, None, quality
    if quality == "weak":
        answer = (
            "I don't know. I could not find this in the runbooks. "
            "The only matches were too weak to trust, so I will not guess."
        )
        return WEAK_MATCH_ONLY, answer, quality
    answer = "I don't know. I could not find this in the runbooks, so I will not guess."
    return NO_RELEVANT_RUNBOOK, answer, quality


def step_limit_answer(sources, limit):
    """Words to show when the step cap stops the loop, and which sections to cite.

    A strong section keeps the older sentence, so a real runbook can still
    be cited. If nothing strong was read, there is no answer to give. Say
    that plainly. The caller still records the stop reason MAX_STEPS_REACHED.
    """
    if has_strong_support(sources):
        return max_steps_message(limit), sources
    return "I could not find this in the runbooks.", []


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

    sources is the Sources list: sections search returned that the citation
    rule kept, in the order they were first read. thoughts are what the model
    (or the search fallback) said it needed before each tool call.
    """

    stop_reason: str
    answer: str
    sources: list
    steps: int
    thoughts: list = field(default_factory=list)
    events: list = field(default_factory=list)  # step-by-step lines for the page


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
    """Keep each runbook section that was actually read, once, in read order.

    The best score for that section is kept. A later weaker chunk of the same
    section does not pull it under the citation cutoff.
    """
    by_key = {(item["source"], item["section"]): item for item in sources}
    for result in results or []:
        section = result.get("section")
        if not section:
            continue
        key = (result.get("source"), section)
        try:
            score = float(result.get("score") or 0)
        except (TypeError, ValueError):
            score = 0.0
        existing = by_key.get(key)
        # The text is kept so a later check can see whether this section is
        # actually about the question. The Sources list does not print it.
        extra = (result.get("text") or "").strip()
        if existing is None:
            item = {"source": key[0], "section": section, "score": score, "text": extra}
            sources.append(item)
            by_key[key] = item
        else:
            if score > existing["score"]:
                existing["score"] = score
            if extra and extra not in existing.get("text", ""):
                existing["text"] = (existing.get("text", "") + " " + extra).strip()


def answer_names_section(answer, section):
    """True when the reply text contains this section's title."""
    title = (section or "").strip()
    if len(title) < 3 or not answer:
        return False
    return title.lower() in answer.lower()


def best_score(results):
    """Highest score in a search result list. Missing or broken scores count as 0."""
    best = 0.0
    for result in results or []:
        try:
            score = float(result.get("score") or 0)
        except (TypeError, ValueError):
            score = 0.0
        if score > best:
            best = score
    return best


def has_strong_support(results):
    """True when some section scores at least CITE_MIN_SCORE.

    That is the line between "the runbooks support an answer" and "these hits
    are too weak to trust". A weak hit can still be printed so you can see it.
    """
    return best_score(results) >= CITE_MIN_SCORE


# Everyday words. They show up in almost any sentence, so they do not prove
# the runbook is about this question. "user", "set", and "large" are here
# because many runbooks use them and they still name no specific problem.
COMMON_WORDS = frozenset({
    "the", "and", "for", "are", "was", "were", "you", "your", "our", "their",
    "how", "what", "when", "where", "who", "why", "not", "can", "has", "have",
    "had", "but", "its", "from", "with", "that", "this", "they", "them", "his",
    "her", "she", "him", "been", "being", "will", "would", "could", "should",
    "may", "might", "into", "over", "after", "before", "about", "than", "then",
    "also", "just", "only", "other", "some", "any", "all", "each", "more",
    "most", "such", "same", "very", "too", "out", "off", "down", "because",
    "while", "during", "without", "within", "again", "still", "there", "here",
    "please", "something", "anything", "another", "even", "though", "does",
    "did", "doing", "get", "gets", "got", "getting", "make", "makes", "made",
    "let", "per", "via",
    "user", "users", "set", "sets", "large", "table", "tables",
    "say", "says", "said", "use", "uses", "used", "using",
    "high", "slow", "new", "old",
})

# A question that says one of these is probably about a person's own device,
# not a server. The runbook text has to contain that same word, or we do not
# answer from it. "wifi" is the spelling we use for "wi-fi" once the hyphen
# is removed.
PERSONAL_SCOPE_WORDS = frozenset({
    "laptop", "iphone", "phone", "browser", "home", "wifi",
})


def words_in(text):
    """Whole words, in lower case. "wi-fi" and "wifi" count as the same word."""
    raw = (text or "").lower().replace("wi-fi", "wifi").replace("wi‑fi", "wifi")
    return {
        word for word in re.findall(r"[a-z0-9]+", raw)
        if len(word) >= MIN_WORD_LENGTH
    }


def distinctive_words(text):
    """Words that could show the runbook is about this question.

    Common words are left out. What remains should be the tool, the symptom,
    or the kind of system the person named.
    """
    return {word for word in words_in(text) if word not in COMMON_WORDS}


def section_words(item):
    """Words in one read section. The title counts, because it names the problem."""
    title = item.get("section") or ""
    body = item.get("text") or ""
    return words_in(f"{title} {body}")


def strong_sections(sources):
    """Sections whose score is high enough to answer from."""
    kept = []
    for item in sources or []:
        try:
            score = float(item.get("score") or 0)
        except (TypeError, ValueError):
            score = 0.0
        if score >= MATCH_MIN_SCORE:
            kept.append(item)
    return kept


def unmatched_runbook_answer(reason):
    """The same 'I don't know' sentences the other guardrails already use."""
    if reason == WEAK_MATCH_ONLY:
        return "I don't know. The only runbook matches were too weak to trust, so I will not guess."
    return "I don't know. I could not find this in the runbooks, so I will not guess."


def runbook_match_failure(question, sources):
    """Return a stop reason when these sections should not be the answer.

    None means this check does not block the answer. The match is good
    enough, or there is no strong section and the other guardrails decide.
    A strong section has to share at least DISTINCTIVE_WORD_MIN distinctive
    words with the question. A personal-device word (laptop, phone, and so
    on) blocks the answer unless that same word is in the runbook.

    This is a word check. It can be wrong. It does not call a model.
    """
    strong = strong_sections(sources)
    if not strong:
        # No section scored high enough for this check. If every score is
        # also under the citation line, the other guardrails decide.
        # If a score clears the citation line but not MATCH_MIN_SCORE, the
        # hit is too weak to trust.
        if sources and best_score(sources) >= CITE_MIN_SCORE:
            return WEAK_MATCH_ONLY
        return None
    runbook_words = set()
    for item in strong:
        runbook_words |= section_words(item)
    # "laptop" in the question, and no "laptop" in the runbook: different system.
    personal = words_in(question) & PERSONAL_SCOPE_WORDS
    if personal and not personal <= runbook_words:
        return NO_RELEVANT_RUNBOOK
    overlap = distinctive_words(question) & runbook_words
    if len(overlap) < DISTINCTIVE_WORD_MIN:
        return NO_RELEVANT_RUNBOOK
    return None


def reply_is_only_asking_for_detail(text):
    """True when the model is asking for more information, not giving steps.

    A short question, or a short request such as "Please give me more detail.",
    can be shown even when no section has been read. Numbered steps, a Sources
    list, or any other reply is a guess. The loop replaces that guess.
    """
    raw = (text or "").strip()
    if not raw or len(raw) > 120:
        return False
    if re.search(r"\d+\s*[\.\)]\s+\S", raw):
        return False
    lowered = raw.lower()
    if "sources:" in lowered:
        return False
    if raw.endswith("?"):
        return True
    asking = ("more detail", "more information", "tell me more", "clarify")
    return any(phrase in lowered for phrase in asking)


def strip_model_sources(text):
    """Drop a Sources block the model wrote. Only the citation rule may add one."""
    if not text:
        return ""
    kept = []
    for line in text.splitlines():
        if line.strip().lower() in {"sources:", "sources"}:
            break
        kept.append(line)
    return "\n".join(kept).rstrip()


def citations_for(sources, answer):
    """Which read sections belong in Sources.

    A section can be cited only if search returned it. It is cited when its
    best score is at least CITE_MIN_SCORE, or when the answer names its title
    (the reply drew from it even though the score was weak). A weak section
    the answer does not name is left off. If nothing qualifies, Sources is
    empty: the loop does not invent a citation and does not keep a weak
    section just to have one. Read order is preserved.
    """
    read = [item for item in sources if item.get("section")]
    chosen = [
        item for item in read
        if item.get("score", 0) >= CITE_MIN_SCORE or answer_names_section(answer, item["section"])
    ]
    return [{"source": item["source"], "section": item["section"]} for item in chosen]


def show_answer(answer, sources):
    """The answer plus a Sources block naming the cited sections."""
    text = (answer or "").rstrip()
    if not sources:
        return text
    lines = [text, "", "Sources:"] if text else ["Sources:"]
    for item in sources:
        lines.append(f"- {item['section']} ({item['source']})")
    return "\n".join(lines)


def note_event(events, step, label, detail):
    """One line of the step-by-step trace. The page shows these in order."""
    events.append({"step": step, "label": label, "detail": detail})


def describe_results(results):
    """A short trace line for what search just returned."""
    if not results:
        return "No relevant sections."
    top = results[0]
    try:
        score = float(top.get("score") or 0)
    except (TypeError, ValueError):
        score = 0.0
    return f"{len(results)} section(s). Top: {top.get('section')} ({score:.3f})."


def record_stop(stop_reason, answer, sources, steps, thoughts, events=None):
    """Remember why the loop stopped, and print that name. Returns the answer text."""
    global last_trace
    cited = citations_for(sources, answer)
    shown = show_answer(answer, cited)
    trace_events = list(events or [])
    plain = STOP_REASON_PLAIN.get(stop_reason, "")
    note_event(trace_events, steps, "Stopped", f"{stop_reason}. {plain}".strip())
    last_trace = LoopTrace(
        stop_reason, shown, [dict(item) for item in cited], steps, list(thoughts), trace_events,
    )
    print(f"Stop reason: {stop_reason}")
    if plain:
        print(plain)
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


def tool_event_detail(name, arguments):
    """A plain trace line for the tool the model just picked."""
    if name == "search_runbooks":
        query = arguments.get("query") if isinstance(arguments, dict) else ""
        return f"search_runbooks for {query!r}."
    if name == "execute_action":
        action = arguments.get("action") if isinstance(arguments, dict) else ""
        return f"execute_action: {action}"
    return f"{name}({arguments})"


def run_agent(request, client=None, ask=input, defer_approval=False):
    """Think, search, read, then repeat or stop.

    The model's own sentence is kept when it is grounded in a section that
    was actually read. When search ran, the returned answer also lists the
    sections the citation rule kept. execute_action still goes through a
    person: the terminal asks y/n, unless defer_approval is True, in which
    case the loop returns a PendingAction and runs nothing. last_trace always
    holds the stop reason when this function returns normally.

    The model is never called more than MAX_STEPS times. The budget is checked
    before every call, so a model that keeps requesting tools still stops.
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
    events = []
    # "strong", "weak", or "none" for the search just before this one.
    # "" means we have not searched yet.
    previous_quality = ""
    # Built here, from the current MAX_STEPS, so a test can lower the cap.
    budget = ModelCallBudget(MAX_STEPS)

    while True:
        # Hard stop. This runs before every model call. Do not call ask_model
        # anywhere else in this loop.
        if not budget.allow():
            # No strong section means the model never reached an answer.
            # Say so in plain words. The stop reason name stays MAX_STEPS_REACHED.
            answer, kept = step_limit_answer(sources, MAX_STEPS)
            shown = record_stop(MAX_STEPS_REACHED, answer, kept, budget.used, thoughts, events)
            print(f"Answer: {shown}")
            return shown

        step = budget.used
        # 1) ask the model what it needs, or whether it can answer now
        print(f"Step {step} of {MAX_STEPS}")
        reply = ask_model(client, messages, tools=TOOL_SPECS)

        # No tool call: answer only if a section was read, or this is a short
        # request for more detail. Anything else would be a guess.
        if not getattr(reply, "tool_calls", None):
            text = strip_model_sources(reply.content or "")
            if not sources and not reply_is_only_asking_for_detail(text):
                answer = "I don't know. I have not read a runbook section, so I will not guess."
                shown = record_stop(UNGROUNDED_ANSWER, answer, sources, step, thoughts, events)
                print(f"Answer: {shown}")
                return shown
            # The model wrote an answer. Keep it only when a strong section is
            # about this question. A similar runbook is not good enough.
            if sources and not reply_is_only_asking_for_detail(text):
                reason = runbook_match_failure(request, sources)
                if reason:
                    answer = unmatched_runbook_answer(reason)
                    shown = record_stop(reason, answer, [], step, thoughts, events)
                    print(f"Answer: {shown}")
                    return shown
            shown = record_stop(ANSWERED, text, sources, step, thoughts, events)
            if sources:
                print(f"Answer: {shown}")
                return shown
            print(text)  # a clarification, same wording as before, when nothing was read
            return text

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
            note_event(events, step, "Thinking", thought)
        print(f"Model chose: {name}({arguments})")
        note_event(events, step, "Tool", tool_event_detail(name, arguments))

        # execute_action is never run here. A person decides.
        if name == "execute_action":
            action = arguments.get("action") if isinstance(arguments, dict) else None
            if not isinstance(action, str) or not action.strip():
                raise AgentError("BAD_TOOL_ARGS", f"execute_action needs an 'action' string: {arguments}.")
            record_stop(ACTION_REQUESTED, action, sources, step, thoughts, events)
            if defer_approval:
                # The page shows the action and waits for a button. Nothing runs.
                print("Action not run. Waiting for a person to approve it.")
                return PendingAction(action)
            return run_gated_action(action, ask=ask)

        # 3) run the tool and read what came back
        results = call_tool(name, **arguments)
        if results:
            print(f"Reading {len(results)} result(s) from the runbooks.")
        print_results(results)
        note_event(events, step, "Read", describe_results(results))

        # Nothing strong enough to answer from, and nothing stronger was read
        # earlier: say so, and do not let the model guess. This is the first
        # weak or empty search. A later empty search does not come through
        # here, because sources already holds a strong section.
        if name == "search_runbooks" and not has_strong_support(results) and not has_strong_support(sources):
            if not results and not sources:
                answer = "I don't know. No relevant runbook sections were found, so I will not guess."
                reason = NO_RELEVANT_RUNBOOK
            else:
                answer = "I don't know. The only runbook matches were too weak to trust, so I will not guess."
                reason = WEAK_MATCH_ONLY
            shown = record_stop(reason, answer, sources, step, thoughts, events)
            print(f"Answer: {shown}")
            return shown

        # The model searched again, and this result is just as weak as the
        # last one. Stop. Do not keep changing the words. A stray earlier
        # hit is not listed as a source.
        if name == "search_runbooks":
            reason, answer, previous_quality = repeated_search_stop(previous_quality, results)
            if reason:
                shown = record_stop(reason, answer, [], step, thoughts, events)
                print(f"Answer: {shown}")
                return shown
            remember_sources(sources, results)
        append_tool_result(messages, reply, call, results)
        # 4) loop: the next step can search again, answer, or ask for an action


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
