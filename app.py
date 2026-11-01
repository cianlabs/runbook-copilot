"""A small local page for the runbook copilot.

Run it with one command:

    python3 app.py

Then open http://127.0.0.1:8000

This file uses the Python standard library for the page (no Gradio, no
Streamlit, no Flask). The copilot logic stays in agent.py. The page only
shows what run_agent already decided and asks a person to click before any
action.

When XAI_API_KEY is set in the terminal, the page uses the real model.
The key is never printed and never written into the page. When the key is
not set, a practice model answers from the runbooks so the page can be
tried with no key and no network call to the model.

The practice model is not as careful a writer as the real one. It quotes
one sentence from each matching section. A question that is itself a
restart command proposes that restart and waits for a button.
"""

import copy
import html
import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import agent
import retrieve

HOST = "127.0.0.1"
PORT = 8000

# The action the page is waiting on, if any. Cleared when a new question
# is asked and when a button is clicked, so an old Approve click cannot
# run a different action later.
pending_action = None
last_view = None


def current_mode():
    """live when a key is set, practice otherwise. The key itself is not returned."""
    if os.environ.get("XAI_API_KEY", "").strip():
        return "live"
    return "practice"


def public_error(err):
    """A named error safe to show. The API key is stripped out if it ever appears."""
    message = str(err)
    key = os.environ.get("XAI_API_KEY", "").strip()
    if key and key in message:
        message = message.replace(key, "[key hidden]")
    name = getattr(err, "name", type(err).__name__)
    return {"name": name, "message": message}


def empty_view(question=""):
    return {
        "question": question,
        "mode": current_mode(),
        "error": None,
        "decision": None,
        "decision_ok": None,
        "answer": "",
        "sources": [],
        "stop_reason": "",
        "stop_plain": "",
        "events": [],
        "pending_action": None,
        "related": [],
        "triage": None,
        "triage_form": None,
    }


def error_view(question, err):
    view = empty_view(question)
    view["error"] = public_error(err)
    return view


def client_for_page():
    """The real model when a key is set, otherwise the practice model.

    make_client reads XAI_API_KEY itself. This function does not print it.
    """
    if current_mode() == "live":
        return agent.make_client()
    return DemoClient()


def _terminal_ask_is_a_bug(prompt):
    """The page must never block on a terminal prompt. A button is the gate."""
    raise agent.AgentError(
        "APPROVAL_NEEDS_A_CLICK",
        "The page tried to ask for approval in the terminal. Use the Approve button.",
    )


def related_sections(action):
    """Runbook lines to show next to a proposed action. Read-only."""
    try:
        return retrieve.search(action)
    except retrieve.RetrieveError:
        return []


def view_from_result(question, result):
    """Turn one run_agent result into the fields the page draws."""
    view = empty_view(question)
    trace = agent.last_trace
    if trace is not None:
        view["answer"] = agent.strip_model_sources(trace.answer)
        view["sources"] = list(trace.sources)
        view["stop_reason"] = trace.stop_reason
        view["stop_plain"] = agent.STOP_REASON_PLAIN.get(trace.stop_reason, trace.stop_reason)
        view["events"] = list(trace.events)
        view["triage"] = trace.triage
    if isinstance(result, agent.PendingAction):
        # Do not show the action as if it already happened.
        view["answer"] = "The copilot proposed an action. It is waiting for you. Nothing has run."
        view["pending_action"] = result.action
        view["related"] = related_sections(result.action)
    return view


def ask(question, client=None):
    """Run one question. An action is held for a button; it is not run here."""
    global pending_action, last_view
    pending_action = None
    try:
        if client is None:
            client = client_for_page()
        result = agent.run_agent(
            question,
            client=client,
            ask=_terminal_ask_is_a_bug,
            defer_approval=True,
        )
    except (agent.AgentError, retrieve.RetrieveError) as err:
        last_view = error_view(question, err)
        return last_view
    if isinstance(result, agent.PendingAction):
        pending_action = result.action
    last_view = view_from_result(question, result)
    return last_view


def _form_value(fields, name):
    values = fields.get(name) or [""]
    return values[0] if values else ""


def parse_evidence_lines(text):
    """One observation per line: id | source | observation | optional timestamp."""
    rows = []
    for line_number, raw in enumerate((text or "").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 3 or not parts[0] or not parts[1] or not parts[2]:
            raise agent.AgentError(
                "INCIDENT_INVALID",
                f"Evidence line {line_number} needs id | source | observation.",
            )
        if len(parts) == 3:
            observation = parts[2]
            timestamp = ""
        else:
            observation = " | ".join(parts[2:-1]).strip()
            timestamp = parts[-1]
            if not observation:
                raise agent.AgentError(
                    "INCIDENT_INVALID",
                    f"Evidence line {line_number} needs id | source | observation.",
                )
        row = {"id": parts[0], "source": parts[1], "observation": observation}
        if timestamp:
            row["timestamp"] = timestamp
        rows.append(row)
    return rows


def evidence_lines(evidence):
    lines = []
    for row in evidence or []:
        line = f"{row.get('id', '')} | {row.get('source', '')} | {row.get('observation', '')}"
        if row.get("timestamp"):
            line += f" | {row['timestamp']}"
        lines.append(line)
    return "\n".join(lines)


def incident_from_form(fields):
    """Build an incident dict from the triage form. The sample is fake data."""
    sample = _form_value(fields, "sample").strip().lower()
    if sample in ("1", "yes", "true", "sample"):
        return copy.deepcopy(agent.SAMPLE_INCIDENT)
    timeline = [line.strip() for line in _form_value(fields, "timeline").splitlines() if line.strip()]
    return {
        "incident_id": _form_value(fields, "incident_id").strip(),
        "symptom": _form_value(fields, "symptom").strip(),
        "impact": _form_value(fields, "impact").strip(),
        "timeline": timeline,
        "evidence": parse_evidence_lines(_form_value(fields, "evidence")),
    }


def triage_form_from_incident(incident):
    return {
        "incident_id": incident.get("incident_id", ""),
        "symptom": incident.get("symptom", ""),
        "impact": incident.get("impact", ""),
        "timeline": "\n".join(incident.get("timeline") or []),
        "evidence": evidence_lines(incident.get("evidence")),
    }


def triage_form_from_fields(fields):
    return {
        "incident_id": _form_value(fields, "incident_id"),
        "symptom": _form_value(fields, "symptom"),
        "impact": _form_value(fields, "impact"),
        "timeline": _form_value(fields, "timeline"),
        "evidence": _form_value(fields, "evidence"),
    }


def triage_request(fields):
    """Run one incident triage. No model and no action. Clears any waiting approval."""
    global pending_action, last_view
    pending_action = None
    try:
        data = incident_from_form(fields)
        answer = agent.run_incident_triage(data)
    except (agent.AgentError, retrieve.RetrieveError) as err:
        last_view = error_view("", err)
        last_view["triage_form"] = triage_form_from_fields(fields)
        return last_view
    view = view_from_result("", answer)
    view["triage_form"] = triage_form_from_incident(data)
    view["pending_action"] = None
    last_view = view
    return view


def describe_outcome(outcome):
    """Plain words for an approve or refuse click. No real process was touched."""
    if outcome == "denied":
        return "You said no. Nothing was run."
    if isinstance(outcome, agent.ActionResult) and outcome.ok:
        details = outcome.details
        return (
            f"Simulated only. {outcome.action}. "
            f"The practice restart stopped pid {details.get('old_pid')} "
            f"and started pid {details.get('new_pid')}. "
            f"CPU would show about {details.get('cpu_percent')}%. "
            "No real process was touched."
        )
    if isinstance(outcome, agent.ActionResult):
        reason = outcome.error or "it did not succeed"
        return f"The simulation refused ({reason}). {outcome.action}. No real process was touched."
    return "Nothing was run."


def decide(choice, action):
    """The Approve / Do not run buttons. Only the waiting action can be approved."""
    global pending_action, last_view
    expected = pending_action
    view = dict(last_view or empty_view())
    if not expected or action != expected:
        # Leave the real waiting action in place. A wrong click must not
        # approve something else, and it must not throw away the real one.
        view["error"] = public_error(agent.AgentError(
            "STALE_APPROVAL",
            "That is not the action waiting for approval. Use the button on the latest result.",
        ))
        last_view = view
        return view
    pending_action = None
    view["pending_action"] = None
    view["related"] = []
    approved = str(choice or "").strip().lower() in ("y", "yes")
    try:
        outcome = agent.settle_action(expected, approved)
    except agent.AgentError as err:
        view["error"] = public_error(err)
        last_view = view
        return view
    view["decision"] = describe_outcome(outcome)
    view["decision_ok"] = isinstance(outcome, agent.ActionResult) and outcome.ok
    # The question stopped at ACTION_REQUESTED, which said it was still waiting.
    # After the click, say what the person just decided so the page does not
    # keep saying "nothing has run" next to a finished simulation.
    if outcome == "denied":
        view["answer"] = "You refused the action. Nothing was run."
        view["stop_plain"] = "You refused the action. Nothing was run."
    elif view["decision_ok"]:
        view["answer"] = "The simulated restart finished. Nothing on a real machine was changed."
        view["stop_plain"] = "You approved the action. Only a simulation ran. No real process was touched."
    else:
        view["answer"] = "The simulation did not do the action. Nothing on a real machine was changed."
        view["stop_plain"] = view["answer"]
    last_view = view
    return view


# ---- Practice model --------------------------------------------------------
# Used only when XAI_API_KEY is unset. It searches, then writes numbered
# steps from the sections that search returned. It does not invent sections.

def _text_message(text):
    return SimpleNamespace(content=text, tool_calls=None)


def _tool_message(name, arguments):
    call = SimpleNamespace(
        id="demo-call",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )
    return SimpleNamespace(content=None, tool_calls=[call])


def _first_sentence(text):
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return ""
    period = cleaned.find(". ")
    sentence = cleaned if period == -1 else cleaned[: period + 1]
    if len(sentence) > 240:
        sentence = sentence[:237].rstrip() + "..."
    return sentence


def answer_from_search(content):
    """Numbered steps using only the sections in this tool result."""
    try:
        results = json.loads(content or "[]")
    except json.JSONDecodeError:
        return "I don't know. I could not read the search results, so I will not guess."
    if not isinstance(results, list) or not results:
        return "I don't know. No relevant runbook sections were found, so I will not guess."
    lines = []
    for item in results:
        if not isinstance(item, dict):
            continue
        # A weak tail can sit next to a real match. Do not turn that tail into a step.
        try:
            score = float(item.get("score") or 0)
        except (TypeError, ValueError):
            score = 0.0
        if score < agent.CITE_MIN_SCORE:
            continue
        sentence = _first_sentence(item.get("text") or "")
        if sentence:
            lines.append(f"{len(lines) + 1}. {sentence}")
    if not lines:
        return "I don't know. No relevant runbook sections were found, so I will not guess."
    return "\n".join(lines)


def demo_restart_action(text):
    """An action string when the question is a restart command, otherwise None.

    "restart the runaway worker" counts. "what does the runbook say about
    restarting" does not: that question should search.
    """
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return None
    try:
        agent.parse_action(cleaned)
        return cleaned
    except agent.AgentError:
        pass
    if not re.match(r"(?i)^(?:please\s+)?restart\b", cleaned):
        return None
    filler = {"the", "a", "an", "please", "my"}
    tokens = re.findall(r"[A-Za-z0-9][\w.-]*", cleaned)
    lowered = [token.lower() for token in tokens]
    if "restart" not in lowered:
        return None
    after = tokens[lowered.index("restart") + 1 :]
    name_words = [word for word in after if word.lower() not in filler]
    if not name_words:
        return None
    name = "-".join(name_words[:3])
    return f"restart the process '{name}'"


class DemoClient:
    """Stands in for the xAI client. Same shape, no network."""

    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        messages = kwargs.get("messages") or []
        last = messages[-1] if messages else {}
        if last.get("role") == "tool":
            message = _text_message(answer_from_search(last.get("content") or ""))
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])
        user = ""
        for message in messages:
            if message.get("role") == "user":
                user = message.get("content") or ""
        action = demo_restart_action(user)
        if action:
            message = _tool_message("execute_action", {"action": action})
        else:
            query = " ".join(user.split())[:200] or "runbook"
            message = _tool_message("search_runbooks", {"query": query})
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


# ---- HTML ------------------------------------------------------------------

PAGE_CSS = """
body { margin: 0; background: #f4f6fb; color: #1b2333; font-family: Georgia, "Iowan Old Style", serif; }
main { max-width: 740px; margin: 0 auto; padding: 32px 20px 64px; }
h1 { font-size: 34px; margin: 0 0 8px; letter-spacing: -0.03em; }
h2 { font-size: 22px; margin: 28px 0 8px; }
p, li, label { font-size: 18px; line-height: 1.45; }
.banner { background: #fff; border: 1px solid #d5dbe8; border-radius: 12px; padding: 14px 16px; }
.banner strong { font-family: "Segoe UI", sans-serif; font-size: 13px; letter-spacing: 0.08em; text-transform: uppercase; color: #5c6b86; }
form { margin-top: 18px; }
textarea { width: 100%; box-sizing: border-box; font: 18px/1.4 Georgia, serif; padding: 12px; border-radius: 10px; border: 1px solid #c5cedf; }
button { font: 600 16px/1 "Segoe UI", sans-serif; border-radius: 999px; border: 0; padding: 12px 18px; margin: 12px 8px 0 0; cursor: pointer; }
.ask { background: #1f6feb; color: white; }
.yes { background: #1a7f37; color: white; }
.no { background: #fff; color: #a40e26; border: 1px solid #a40e26; }
.card { background: #fff; border: 1px solid #d5dbe8; border-radius: 12px; padding: 16px 18px; margin-top: 14px; }
.answer { white-space: pre-wrap; }
.plain { font-size: 20px; }
.code { font-family: ui-monospace, monospace; font-size: 14px; color: #5c6b86; }
.warn { background: #fff8e6; border-color: #e6c200; }
.bad { background: #fff1f0; border-color: #e5484d; }
.good { background: #eefbf3; border-color: #1a7f37; }
.trace { list-style: none; padding: 0; margin: 0; }
.trace li { padding: 8px 0; border-top: 1px solid #e6ebf5; }
.trace li:first-child { border-top: 0; }
.hypothesis { border-top: 1px solid #e6ebf5; padding-top: 10px; margin-top: 10px; }
label.field { display: block; margin-top: 10px; }
"""


def _esc(value):
    return html.escape("" if value is None else str(value), quote=True)


def _mode_banner(mode):
    if mode == "live":
        title = "Live mode"
        body = (
            "The real model is answering. Your API key stays in the terminal "
            "and is not shown on this page."
        )
    else:
        title = "Practice mode"
        body = (
            "No API key is set, so a built-in practice model is answering. "
            "It only reads your runbooks and does not call the model network. "
            "To use the real model, set XAI_API_KEY in the terminal and start this page again. "
            "The key is never shown here."
        )
    return f'<div class="banner"><strong>{title}</strong><p>{body}</p></div>'


def _triage_form(view):
    form = view.get("triage_form") or {}
    return (
        "<h2>Triage an incident</h2>"
        "<p>Type a practice incident, or use the sample. "
        "This form does not call a model and does not use a network, even in live mode. "
        "It only ranks hypotheses from the evidence you type. "
        "It does not confirm a root cause, and it does not run a check or a restart. "
        "A restart still has to be asked in the box above, and it still waits for Approve.</p>"
        "<form method=\"post\" action=\"/triage\">"
        "<label class=\"field\" for=\"incident_id\">Incident id</label>"
        f"<textarea id=\"incident_id\" name=\"incident_id\" rows=\"1\">{_esc(form.get('incident_id'))}</textarea>"
        "<label class=\"field\" for=\"symptom\">Symptom</label>"
        f"<textarea id=\"symptom\" name=\"symptom\" rows=\"2\">{_esc(form.get('symptom'))}</textarea>"
        "<label class=\"field\" for=\"impact\">Impact</label>"
        f"<textarea id=\"impact\" name=\"impact\" rows=\"2\">{_esc(form.get('impact'))}</textarea>"
        "<label class=\"field\" for=\"timeline\">Timeline, one note per line</label>"
        f"<textarea id=\"timeline\" name=\"timeline\" rows=\"3\">{_esc(form.get('timeline'))}</textarea>"
        "<label class=\"field\" for=\"evidence\">Evidence, one per line: id | source | observation</label>"
        f"<textarea id=\"evidence\" name=\"evidence\" rows=\"4\">{_esc(form.get('evidence'))}</textarea>"
        "<button class=\"ask\" type=\"submit\">Triage this incident</button>"
        "</form>"
        "<form method=\"post\" action=\"/triage\">"
        "<input type=\"hidden\" name=\"sample\" value=\"1\">"
        "<button class=\"ask\" type=\"submit\">Triage the practice sample</button>"
        "</form>"
    )


def _triage_card(triage):
    """Hypotheses with the evidence ids that matched. User text is escaped."""
    if not triage:
        return ""
    incident = triage.get("incident") or {}
    evidence = {row.get("id"): row for row in (incident.get("evidence") or []) if isinstance(row, dict)}
    parts = [
        "<section class=\"card\"><h2>Incident triage</h2>",
        "<p>Confirmed root cause: not confirmed.</p>",
        f"<p>Action mode: {_esc(triage.get('action_mode'))}. No action was run.</p>",
        (
            f"<p>Incident {_esc(incident.get('incident_id'))}. "
            f"Symptom: {_esc(incident.get('symptom'))}. "
            f"Impact: {_esc(incident.get('impact'))}.</p>"
        ),
    ]
    for hypothesis in triage.get("hypotheses") or []:
        ids = hypothesis.get("supporting_evidence_ids") or []
        linked = ", ".join(str(item) for item in ids) if ids else "none"
        step = hypothesis.get("next_diagnostic_step") or ""
        if agent.step_is_write(step):
            step_line = (
                f"Next check was not run: {_esc(step)} "
                "This would change something. It was not queued for approval."
            )
        else:
            step_line = f"Next check, not run: {_esc(step)}"
        parts.append("<article class=\"hypothesis\">")
        parts.append(f"<h3>{_esc(hypothesis.get('title'))}</h3>")
        parts.append(f"<p>Status: {_esc(hypothesis.get('status'))}.</p>")
        parts.append(f"<p>Supporting evidence: {_esc(linked)}.</p>")
        if ids:
            parts.append("<ul>")
            for evidence_id in ids:
                row = evidence.get(evidence_id) or {}
                parts.append(
                    "<li><span class=\"evidence-id\">"
                    f"{_esc(evidence_id)}</span> "
                    f"({_esc(row.get('source'))}): {_esc(row.get('observation'))}</li>"
                )
            parts.append("</ul>")
        missing = hypothesis.get("missing_evidence") or []
        if missing:
            parts.append("<p>Missing: " + _esc("; ".join(str(item) for item in missing)) + ".</p>")
        parts.append(f"<p>{step_line}</p>")
        parts.append("</article>")
    parts.append(f"<p>{_esc(triage.get('customer_update'))}</p>")
    parts.append("</section>")
    return "".join(parts)


def _question_form(question):
    return (
        "<form method=\"post\" action=\"/ask\">"
        "<label for=\"question\">What is going on?</label><br>"
        f"<textarea id=\"question\" name=\"question\" rows=\"4\" required>{_esc(question)}</textarea><br>"
        "<button class=\"ask\" type=\"submit\">Ask the copilot</button>"
        "</form>"
    )


def render(view=None):
    """The whole page. Runbook text is escaped so it shows as text, not HTML."""
    view = view or empty_view()
    parts = [
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        "<title>Runbook copilot</title>",
        f"<style>{PAGE_CSS}</style></head><body><main>",
        "<h1>Runbook copilot</h1>",
        "<p>Ask in ordinary words. You will see the steps, where they came from, "
        "why the copilot stopped, and what it did along the way. "
        "If it wants to do something, nothing runs until you click Approve. "
        "That click only runs a simulation.</p>",
        _mode_banner(view.get("mode") or current_mode()),
        _question_form(view.get("question") or ""),
        _triage_form(view),
    ]
    error = view.get("error")
    if error:
        parts.append(
            "<section class=\"card bad\"><h2>Something needs attention</h2>"
            f"<p class=\"code\">{_esc(error.get('name'))}</p>"
            f"<p>{_esc(error.get('message'))}</p></section>"
        )
    if view.get("stop_reason"):
        parts.append(
            "<section class=\"card\"><h2>Why it stopped</h2>"
            f"<p class=\"plain\">{_esc(view.get('stop_plain'))}</p>"
            f"<p class=\"code\">Stop reason: {_esc(view.get('stop_reason'))}</p></section>"
        )
    if view.get("answer"):
        parts.append(
            "<section class=\"card\"><h2>What to do</h2>"
            f"<p class=\"answer\">{_esc(view.get('answer'))}</p></section>"
        )
    if view.get("stop_reason"):
        sources = view.get("sources") or []
        items = "".join(
            f"<li>{_esc(item.get('section'))} ({_esc(item.get('source'))})</li>" for item in sources
        )
        if not items and view.get("triage"):
            items = "<li>No runbook sections. This triage used only the evidence you supplied.</li>"
        elif not items:
            items = "<li>No sources. The copilot is not citing a section.</li>"
        parts.append(f"<section class=\"card\"><h2>Sources</h2><ul>{items}</ul></section>")
    triage_html = _triage_card(view.get("triage"))
    if triage_html:
        parts.append(triage_html)
    pending = view.get("pending_action")
    if pending:
        related = view.get("related") or []
        related_html = ""
        if related:
            rows = "".join(
                f"<li>{_esc(item.get('section'))}: {_esc((item.get('text') or '')[:160])}</li>"
                for item in related
            )
            related_html = f"<p>Related runbook sections:</p><ul>{rows}</ul>"
        else:
            related_html = "<p>Related runbook sections: none found.</p>"
        parts.append(
            "<section class=\"card warn\"><h2>This action has not run</h2>"
            f"<p class=\"answer\">{_esc(pending)}</p>{related_html}"
            "<p>Clicking Approve only simulates the restart. No real process is signaled. "
            "The other button runs nothing.</p>"
            "<form method=\"post\" action=\"/decide\">"
            f"<input type=\"hidden\" name=\"action\" value=\"{_esc(pending)}\">"
            "<button class=\"yes\" type=\"submit\" name=\"choice\" value=\"yes\">Approve and simulate</button>"
            "<button class=\"no\" type=\"submit\" name=\"choice\" value=\"no\">Do not run</button>"
            "</form></section>"
        )
    decision = view.get("decision")
    if decision:
        kind = "good" if view.get("decision_ok") else "bad"
        parts.append(
            f"<section class=\"card {kind}\"><h2>Your decision</h2>"
            f"<p>{_esc(decision)}</p></section>"
        )
    events = view.get("events") or []
    if events:
        rows = "".join(
            "<li><strong>"
            f"Step {_esc(event.get('step'))} · {_esc(event.get('label'))}"
            f"</strong><br>{_esc(event.get('detail'))}</li>"
            for event in events
        )
        parts.append(
            "<section class=\"card\"><h2>What the copilot did, step by step</h2>"
            f"<ol class=\"trace\">{rows}</ol></section>"
        )
    parts.append("</main></body></html>")
    return "".join(parts)


def render_message(text, status_note=""):
    view = empty_view()
    view["error"] = {"name": status_note or "NOT_FOUND", "message": text}
    return render(view)


class Handler(BaseHTTPRequestHandler):
    """One page. GET shows the form. POST /ask runs the copilot. POST /decide is the button."""

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(render(None), 200)
            return
        # GET must not approve or run an action, even on /decide.
        self._send(render_message("That page does not exist."), 404)

    def do_POST(self):
        path = urlparse(self.path).path
        fields = self._form()
        if path == "/ask":
            question = fields.get("question", [""])[0]
            self._send(render(ask(question)), 200)
            return
        if path == "/triage":
            self._send(render(triage_request(fields)), 200)
            return
        if path == "/decide":
            choice = fields.get("choice", [""])[0]
            action = fields.get("action", [""])[0]
            self._send(render(decide(choice, action)), 200)
            return
        self._send(render_message("That page does not exist."), 404)

    def _form(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        return parse_qs(raw, keep_blank_values=True)

    def _send(self, body, status):
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        # The default log is only the request line, but skip it anyway so a
        # question or a key never lands in the server log from this page.
        return


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    port = PORT
    if args:
        try:
            port = int(args[0])
        except ValueError:
            print("PORT_NOT_A_NUMBER: use python3 app.py or python3 app.py 8000")
            return 2
    try:
        server = ThreadingHTTPServer((HOST, port), Handler)
    except OSError as err:
        print(f"PORT_IN_USE: could not listen on {HOST}:{port} ({type(err).__name__}).")
        return 1
    print(f"Runbook copilot is running at http://{HOST}:{port}")
    if current_mode() == "practice":
        print("Practice mode: no XAI_API_KEY in this terminal, so the built-in practice model will answer.")
    else:
        print("Live mode: XAI_API_KEY is set. The key is not printed.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
