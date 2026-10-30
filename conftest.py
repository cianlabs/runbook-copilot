import pytest

import agent


@pytest.fixture(autouse=True)
def private_agent_log(tmp_path, monkeypatch):
    """Every test logs to its own temp file, never to the real agent.log."""
    monkeypatch.setattr(agent, "LOG_FILE", tmp_path / "agent.log")
