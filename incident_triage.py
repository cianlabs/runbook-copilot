"""Deterministic, read-only incident triage. No LLM or action execution."""
from dataclasses import dataclass, asdict, field
from typing import Any
import json
import re

@dataclass
class Evidence:
    id: str
    source: str
    observation: str
    timestamp: str = ""

@dataclass
class Incident:
    incident_id: str
    symptom: str
    impact: str
    timeline: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)

@dataclass
class Hypothesis:
    title: str
    rationale: str
    supporting_evidence_ids: list[str]
    missing_evidence: list[str]
    next_diagnostic_step: str
    status: str = "unverified"

PATTERNS = (
    ("Reconstruction worker failure", ("worker", "exception", "crash", "restart", "oom"), "Inspect worker errors and restart history."),
    ("Storage or image-read failure", ("storage", "image", "tif", "read", "permission", "object"), "Inspect object availability, read permissions and decoder errors."),
    ("Processing resource saturation", ("cpu", "memory", "queue", "io", "latency", "capacity"), "Inspect resource and queue metrics for the incident window."),
)

def validate_incident(data: dict[str, Any]) -> Incident:
    if not isinstance(data, dict):
        raise ValueError("incident must be an object")
    for key in ("incident_id", "symptom", "impact"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    timeline = data.get("timeline", [])
    if not isinstance(timeline, list) or any(not isinstance(x, str) for x in timeline):
        raise ValueError("timeline must be a list of strings")
    raw = data.get("evidence", [])
    if not isinstance(raw, list):
        raise ValueError("evidence must be a list")
    evidence = []
    seen = set()
    for row in raw:
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k].strip() for k in ("id", "source", "observation")):
            raise ValueError("evidence requires nonempty id, source and observation")
        if row["id"] in seen:
            raise ValueError("duplicate evidence id")
        seen.add(row["id"])
        stamp = row.get("timestamp", "")
        if not isinstance(stamp, str):
            raise ValueError("timestamp must be a string")
        evidence.append(Evidence(row["id"], row["source"], row["observation"], stamp))
    return Incident(data["incident_id"], data["symptom"], data["impact"], timeline, evidence)

def triage(data: dict[str, Any]) -> dict[str, Any]:
    incident = validate_incident(data)
    hypotheses = []
    for title, keywords, step in PATTERNS:
        matches = [e.id for e in incident.evidence if any(re.search(r"\b" + re.escape(k) + r"\b", e.observation.lower()) for k in keywords)]
        hypotheses.append(Hypothesis(
            title=title,
            rationale=("Evidence mentions a related signal; causation is not established." if matches else "Possible investigation path; no supporting evidence supplied."),
            supporting_evidence_ids=matches,
            missing_evidence=[] if matches else ["Relevant logs or metrics for this hypothesis"],
            next_diagnostic_step=step,
        ))
    hypotheses.sort(key=lambda h: -len(h.supporting_evidence_ids))
    return {
        "incident": asdict(incident),
        "hypotheses": [asdict(h) for h in hypotheses],
        "confirmed_root_cause": None,
        "action_mode": "simulation_only",
        "customer_update": f"We are investigating {incident.symptom}. Impact reported: {incident.impact}. We have not confirmed a root cause.",
        "engineering_escalation": {
            "incident_id": incident.incident_id,
            "observed_symptom": incident.symptom,
            "reported_impact": incident.impact,
            "timeline": incident.timeline,
            "evidence_ids": [e.id for e in incident.evidence],
            "next_steps": [h.next_diagnostic_step for h in hypotheses],
        },
    }

if __name__ == "__main__":
    import sys
    print(json.dumps(triage(json.load(sys.stdin)), indent=2))
