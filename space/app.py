"""Hugging Face Space (Gradio) for the runbook copilot.

The page logic stays in app.py. This file only draws it and keeps one
approval session per visitor. Practice mode is the default: if XAI_API_KEY
is unset, DemoClient answers from the bundled runbooks and does not call
the model. On a Space, put the key in a secret named XAI_API_KEY. Do not
write the key into this file or the README.

Run locally:

    python3 space/app.py

Then open the URL Gradio prints (usually http://127.0.0.1:7860).
"""

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app as local_app  # noqa: E402
import agent  # noqa: E402

try:
    import gradio as gr
except ImportError:  # tests and practice imports still work without the UI package
    gr = None


def new_session():
    """One visitor's waiting action. Not shared with anyone else."""
    return SimpleNamespace(pending_action=None, last_view=None)


def _ensure(session):
    if session is None or not hasattr(session, "pending_action"):
        return new_session()
    return session


def mode_banner():
    if local_app.current_mode() == "live":
        return (
            "**Live mode.** The real model is answering. "
            "The API key is a Space secret (or a local environment variable). "
            "It is not shown here."
        )
    return (
        "**Practice mode.** No API key is set, so a built-in practice model "
        "is answering from the sample runbooks. It does not call the model network. "
        "To use the real model, set the secret `XAI_API_KEY` on the Space "
        "(Settings, then Secrets) and restart. Do not paste the key into this page."
    )


def _error_text(view):
    error = view.get("error")
    if not error:
        return ""
    return f"{error.get('name')}: {error.get('message')}"


def _sources_text(view):
    if not view.get("stop_reason"):
        return ""
    sources = view.get("sources") or []
    if not sources and view.get("triage"):
        return "No runbook sections. This triage used only the evidence you supplied."
    if not sources:
        return "No sources. The copilot is not citing a section."
    lines = []
    for item in sources:
        lines.append(f"- {item.get('section')} ({item.get('source')})")
    return "\n".join(lines)


def _trace_text(view):
    lines = []
    for event in view.get("events") or []:
        lines.append(
            f"Step {event.get('step')} · {event.get('label')}: {event.get('detail')}"
        )
    return "\n".join(lines)


def _triage_text(view):
    triage = view.get("triage")
    if not triage:
        return ""
    incident = triage.get("incident") or {}
    evidence = {
        row.get("id"): row
        for row in (incident.get("evidence") or [])
        if isinstance(row, dict)
    }
    parts = [
        "Confirmed root cause: not confirmed.",
        f"Action mode: {triage.get('action_mode')}. No action was run.",
        (
            f"Incident {incident.get('incident_id')}. "
            f"Symptom: {incident.get('symptom')}. "
            f"Impact: {incident.get('impact')}."
        ),
    ]
    for hypothesis in triage.get("hypotheses") or []:
        ids = hypothesis.get("supporting_evidence_ids") or []
        linked = ", ".join(str(item) for item in ids) if ids else "none"
        step = hypothesis.get("next_diagnostic_step") or ""
        if agent.step_is_write(step):
            step_line = (
                f"Next check was not run: {step} "
                "This would change something. It was not queued for approval."
            )
        else:
            step_line = f"Next check, not run: {step}"
        parts.append(hypothesis.get("title") or "")
        parts.append(f"Status: {hypothesis.get('status')}.")
        parts.append(f"Supporting evidence: {linked}.")
        for evidence_id in ids:
            row = evidence.get(evidence_id) or {}
            parts.append(
                f"- {evidence_id} ({row.get('source')}): {row.get('observation')}"
            )
        missing = hypothesis.get("missing_evidence") or []
        if missing:
            parts.append("Missing: " + "; ".join(str(item) for item in missing) + ".")
        parts.append(step_line)
    parts.append(triage.get("customer_update") or "")
    return "\n".join(parts)


def present(view, session):
    """Plain text for the Space. Runbook text is not passed through HTML."""
    pending = view.get("pending_action") or ""
    return (
        _error_text(view),
        view.get("stop_plain") or "",
        view.get("stop_reason") or "",
        view.get("answer") or "",
        _sources_text(view),
        _trace_text(view),
        pending,
        view.get("decision") or "",
        _triage_text(view),
        session,
    )


def on_ask(question, session):
    session = _ensure(session)
    view = local_app.ask(question, session=session)
    return present(view, session)


def on_decide(choice, session):
    """Approve only the action stored for this visitor. A text box cannot change it."""
    session = _ensure(session)
    action = session.pending_action or ""
    view = local_app.decide(choice, action, session=session)
    return present(view, session)


def on_approve(session):
    return on_decide("yes", session)


def on_refuse(session):
    return on_decide("no", session)


def on_triage_form(incident_id, symptom, impact, timeline, evidence, session):
    return on_triage(incident_id, symptom, impact, timeline, evidence, session, sample=False)


def on_triage_sample(incident_id, symptom, impact, timeline, evidence, session):
    return on_triage(incident_id, symptom, impact, timeline, evidence, session, sample=True)


def on_triage(incident_id, symptom, impact, timeline, evidence, session, sample=False):
    session = _ensure(session)
    fields = {
        "incident_id": [incident_id or ""],
        "symptom": [symptom or ""],
        "impact": [impact or ""],
        "timeline": [timeline or ""],
        "evidence": [evidence or ""],
    }
    if sample:
        fields["sample"] = ["1"]
    view = local_app.triage_request(fields, session=session)
    return present(view, session)


def build():
    if gr is None:
        raise RuntimeError(
            "Gradio is not installed. For a local Space preview: pip install gradio"
        )
    with gr.Blocks(title="Runbook copilot") as demo:
        session = gr.State(new_session)
        gr.Markdown(
            "# Runbook copilot\n"
            "Ask in ordinary words. You will see the steps, where they came from, "
            "and why the copilot stopped. If it wants to do something, nothing runs "
            "until you click **Approve and simulate**. That click only runs a simulation. "
            "The runbooks in this demo are public samples, not customer data."
        )
        gr.Markdown(mode_banner())
        question = gr.Textbox(
            label="What is going on?",
            lines=4,
            placeholder="web server CPU is at 95 percent",
        )
        ask_btn = gr.Button("Ask the copilot")
        gr.Examples(
            examples=[
                ["web server CPU is at 95 percent"],
                ["how do I bake a sourdough pancake"],
                ["restart the runaway worker"],
            ],
            inputs=[question],
            label="Try a sample question",
        )
        error = gr.Textbox(label="Something needs attention", lines=2)
        stop_plain = gr.Textbox(label="Why it stopped")
        stop_reason = gr.Textbox(label="Stop reason")
        answer = gr.Textbox(label="What to do", lines=8)
        sources = gr.Textbox(label="Sources", lines=4)
        trace = gr.Textbox(label="What the copilot did, step by step", lines=8)
        pending = gr.Textbox(
            label="This action has not run",
            lines=2,
            interactive=False,
        )
        gr.Markdown(
            "Clicking **Approve and simulate** only simulates the restart. "
            "No real process is signaled. **Do not run** runs nothing. "
            "The button uses the action this session is waiting on. "
            "Editing the box does not change what can be approved."
        )
        with gr.Row():
            yes_btn = gr.Button("Approve and simulate")
            no_btn = gr.Button("Do not run")
        decision = gr.Textbox(label="Your decision", lines=3)
        gr.Markdown(
            "## Triage an incident\n"
            "Type a practice incident, or use the sample. "
            "This form does not call a model and does not use a network, even in live mode. "
            "It only ranks hypotheses from the evidence you type. "
            "It does not confirm a root cause, and it does not run a check or a restart. "
            "A restart still has to be asked above, and it still waits for Approve."
        )
        incident_id = gr.Textbox(label="Incident id")
        symptom = gr.Textbox(label="Symptom", lines=2)
        impact = gr.Textbox(label="Impact", lines=2)
        timeline = gr.Textbox(label="Timeline, one note per line", lines=3)
        evidence = gr.Textbox(
            label="Evidence, one per line: id | source | observation",
            lines=4,
        )
        with gr.Row():
            triage_btn = gr.Button("Triage this incident")
            sample_btn = gr.Button("Triage the practice sample")
        triage_out = gr.Textbox(label="Incident triage", lines=12)

        outputs = [
            error,
            stop_plain,
            stop_reason,
            answer,
            sources,
            trace,
            pending,
            decision,
            triage_out,
            session,
        ]
        ask_btn.click(on_ask, [question, session], outputs)
        yes_btn.click(on_approve, [session], outputs)
        no_btn.click(on_refuse, [session], outputs)
        triage_btn.click(
            on_triage_form,
            [incident_id, symptom, impact, timeline, evidence, session],
            outputs,
        )
        sample_btn.click(
            on_triage_sample,
            [incident_id, symptom, impact, timeline, evidence, session],
            outputs,
        )
    return demo


def main():
    demo = build()
    demo.launch()


if __name__ == "__main__":
    main()
