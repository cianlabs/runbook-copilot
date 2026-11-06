"""Streamlit page for the runbook copilot.

Community Cloud looks for this file at the repository root. The copilot
logic stays in app.py. space/app.py already turns one visitor session into
plain text. This file draws that text and keeps one approval session in
the browser (st.session_state), so two visitors do not share a waiting action.

Practice mode is the default. If XAI_API_KEY is unset, DemoClient answers
from the bundled runbooks and does not call the model. On Streamlit Cloud,
put the key in the app Secrets as XAI_API_KEY. This file copies that secret
into the process environment so agent.make_client can read it. The value is
never printed and never written into the page.

Run locally:

    streamlit run streamlit_app.py
"""

import os

import app as local_app
import space.app as space_app

try:
    import streamlit as st
except ImportError:  # pytest stays green when Streamlit is not installed
    st = None


def apply_streamlit_secret(secrets=None):
    """Copy XAI_API_KEY from Streamlit secrets into the environment.

    An existing environment variable is left alone, so a local export wins.
    A missing secrets file is practice mode. The key is not returned.
    """
    if os.environ.get("XAI_API_KEY", "").strip():
        return
    if secrets is None:
        if st is None:
            return
        try:
            secrets = st.secrets
        except Exception:
            return
    try:
        if "XAI_API_KEY" not in secrets:
            return
        raw = secrets["XAI_API_KEY"]
    except Exception:
        return
    text = str(raw).strip() if raw is not None else ""
    if text:
        os.environ["XAI_API_KEY"] = text


def mode_banner():
    """Which model is answering. The banner never includes the key."""
    if local_app.current_mode() == "live":
        return (
            "**Live mode.** The real model is answering. "
            "The API key comes from Streamlit secrets or the environment. "
            "It is not shown here."
        )
    return (
        "**Practice mode.** No API key is set, so a built-in practice model "
        "is answering from the sample runbooks. It does not call the model network. "
        "To use the real model, add `XAI_API_KEY` under Secrets on Streamlit Community Cloud "
        "and reboot the app. Do not paste the key into this page."
    )


def ensure_session(store):
    """One browser's waiting action. `store` is st.session_state or a dict."""
    session = store.get("copilot_session")
    if session is None or not hasattr(session, "pending_action"):
        session = space_app.new_session()
        store["copilot_session"] = session
    return session


def on_ask(question, session):
    return space_app.on_ask(question, session)


def on_approve(session):
    return space_app.on_approve(session)


def on_refuse(session):
    return space_app.on_refuse(session)


def on_triage(incident_id, symptom, impact, timeline, evidence, session, sample=False):
    return space_app.on_triage(
        incident_id, symptom, impact, timeline, evidence, session, sample=sample
    )


def _related_text(session):
    view = getattr(session, "last_view", None) or {}
    if not view.get("pending_action"):
        return ""
    related = view.get("related") or []
    if not related:
        return "Related runbook sections: none found."
    lines = ["Related runbook sections:"]
    for item in related:
        snippet = (item.get("text") or "").replace("\n", " ")[:160]
        lines.append(f"- {item.get('section')}: {snippet}")
    return "\n".join(lines)


TRIAGE_WIDGET_KEYS = {
    "incident_id": "triage_incident_id",
    "symptom": "triage_symptom",
    "impact": "triage_impact",
    "timeline": "triage_timeline",
    "evidence": "triage_evidence",
}


def sync_triage_form(store, session):
    """Copy a filled practice form into the widgets. Empty forms are left alone.

    Call this before the widgets are created on the next run. Streamlit shows
    the value stored under each key.
    """
    form = (getattr(session, "last_view", None) or {}).get("triage_form") or {}
    if not any(str(form.get(name) or "").strip() for name in TRIAGE_WIDGET_KEYS):
        return
    for name, key in TRIAGE_WIDGET_KEYS.items():
        store[key] = form.get(name) or ""


def draw(fields, session):
    """Show the last result. Runbook text is plain text, not HTML."""
    error, stop_plain, stop_reason, answer, sources, trace, pending, decision, triage, _session = fields
    if error:
        st.error(error)
    if stop_plain or stop_reason:
        st.subheader("Why it stopped")
        if stop_plain:
            st.write(stop_plain)
        if stop_reason:
            st.caption(f"Stop reason: {stop_reason}")
    if answer:
        st.subheader("What to do")
        st.text(answer)
    if stop_reason:
        st.subheader("Sources")
        st.text(sources or "No sources. The copilot is not citing a section.")
    if trace:
        st.subheader("What the copilot did, step by step")
        st.text(trace)
    if pending:
        st.subheader("This action has not run")
        st.text(pending)
        related = _related_text(session)
        if related:
            st.text(related)
        st.caption(
            "Approve and simulate only pretends to restart a process. "
            "No real process is signaled. Do not run does nothing. "
            "The button uses the action this browser session is waiting on."
        )
        yes, no = st.columns(2)
        if yes.button("Approve and simulate", type="primary"):
            st.session_state.last_fields = on_approve(session)
            sync_triage_form(st.session_state, session)
            st.rerun()
        if no.button("Do not run"):
            st.session_state.last_fields = on_refuse(session)
            st.rerun()
    if decision:
        st.subheader("Your decision")
        st.write(decision)
    if triage:
        st.subheader("Incident triage")
        st.text(triage)


def main():
    if st is None:
        raise RuntimeError(
            "Streamlit is not installed. Install it with: pip install -r requirements.txt"
        )
    apply_streamlit_secret()
    st.set_page_config(page_title="Runbook copilot", layout="centered")
    session = ensure_session(st.session_state)
    st.title("Runbook copilot")
    st.write(
        "Ask in ordinary words. You will see the steps, where they came from, "
        "and why the copilot stopped. If it wants to do something, nothing runs "
        "until you click **Approve and simulate**. That click only runs a simulation. "
        "In practice mode it then fills the triage form from that CPU restart and "
        "adds a lessons note. Nothing on a real machine changes. "
        "The runbooks in this demo are public samples, not customer data."
    )
    st.markdown(mode_banner())

    question = st.text_area("What is going on?", height=120, placeholder="web server CPU is at 95 percent")
    samples = (
        "web server CPU is at 95 percent",
        "how do I bake a sourdough pancake",
        "restart the runaway worker",
    )
    picked = st.selectbox("Try a sample question", ("(type your own)",) + samples)
    if st.button("Ask the copilot"):
        asked = question
        if picked != "(type your own)":
            asked = picked
        st.session_state.last_fields = on_ask(asked, session)
        st.rerun()

    fields = st.session_state.get("last_fields")
    if fields:
        draw(fields, session)

    st.divider()
    st.subheader("Triage an incident")
    st.write(
        "Type a practice incident, or use the sample. "
        "After a simulated CPU restart, practice mode fills these fields from that restart. "
        "This form does not call a model and does not use a network, even in live mode. "
        "It only ranks hypotheses from the evidence you type. "
        "It does not confirm a root cause, and it does not run a check or a restart. "
        "A restart still has to be asked above, and it still waits for Approve."
    )
    incident_id = st.text_input("Incident id", key="triage_incident_id")
    symptom = st.text_area("Symptom", height=68, key="triage_symptom")
    impact = st.text_area("Impact", height=68, key="triage_impact")
    timeline = st.text_area("Timeline, one note per line", height=80, key="triage_timeline")
    evidence = st.text_area(
        "Evidence, one per line: id | source | observation",
        height=100,
        key="triage_evidence",
    )
    triage_col, sample_col = st.columns(2)
    if triage_col.button("Triage this incident"):
        st.session_state.last_fields = on_triage(
            incident_id, symptom, impact, timeline, evidence, session, sample=False
        )
        st.rerun()
    if sample_col.button("Triage the practice sample"):
        st.session_state.last_fields = on_triage(
            incident_id, symptom, impact, timeline, evidence, session, sample=True
        )
        st.rerun()

    lessons = (getattr(session, "last_view", None) or {}).get("lessons") or ""
    if lessons:
        st.subheader("Lessons learned")
        st.caption(
            "For a product or engineering reader. Practice only. "
            "This note does not run a check, a restart, or a roll back."
        )
        st.text(lessons)


if __name__ == "__main__":
    main()
