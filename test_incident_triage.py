import unittest
from incident_triage import triage, validate_incident

BASE = {"incident_id": "INC-1", "symptom": "Scan reconstruction unavailable", "impact": "One customer blocked", "timeline": ["10:00 acquisition complete"], "evidence": []}

class IncidentTriageTests(unittest.TestCase):
    def test_three_unverified_hypotheses_without_evidence(self):
        result = triage(BASE)
        self.assertEqual(len(result["hypotheses"]), 3)
        self.assertTrue(all(h["status"] == "unverified" and not h["supporting_evidence_ids"] for h in result["hypotheses"]))
        self.assertIsNone(result["confirmed_root_cause"])
        self.assertEqual(result["action_mode"], "simulation_only")

    def test_evidence_linked_by_id(self):
        data = {**BASE, "evidence": [{"id": "log-1", "source": "worker.log", "observation": "worker exception"}]}
        result = triage(data)
        self.assertEqual(result["hypotheses"][0]["title"], "Reconstruction worker failure")
        self.assertEqual(result["hypotheses"][0]["supporting_evidence_ids"], ["log-1"])

    def test_duplicate_evidence_rejected(self):
        data = {**BASE, "evidence": [{"id": "a", "source": "log", "observation": "cpu"}, {"id": "a", "source": "log", "observation": "io"}]}
        with self.assertRaises(ValueError):
            validate_incident(data)

    def test_invalid_incident_rejected(self):
        with self.assertRaises(ValueError):
            triage({"incident_id": "x", "symptom": "", "impact": "blocked"})

    def test_no_action_execution(self):
        self.assertNotIn("execute_action", triage(BASE))
        self.assertIn("not confirmed", triage(BASE)["customer_update"].lower())

if __name__ == "__main__":
    unittest.main()
