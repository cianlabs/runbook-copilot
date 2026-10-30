# runbook-copilot

**Goal:** an agent that searches a runbook knowledge base (RAG) and takes actions via tool calls, with a human approval step before anything runs.

## Status

Working end to end, with a stub for the action itself. `execute_action` only prints what it would do and never changes a real system.

- Search: the `search_runbooks` tool uses `retrieve.py`, which splits each runbook at its `# ` headers first, then cuts each section into chunks (about 400 characters, 80 overlap) so a chunk never mixes sections. It ranks chunks with TF-IDF and returns the top 3 with scores.
- Agent loop: `agent.py` sends the request to the model (Grok, model `grok-build-0.1`, through the xAI API). The model picks a tool, the code runs it, and the results go back to the model for the final answer.
- Approval gate: the agent shows the proposed action and related runbook sections, then asks `Approve this action? (y/n)`. Only `y` or `yes` lets it through. The model cannot approve its own action.

## How it works

1. Search: find the relevant runbook sections for the request.
2. Propose: the model may ask for `execute_action`.
3. Approve: a human types y or n.
4. Execute: only approved actions run (stub for now).

## Sample runbook

`runbooks/sample_runbook.txt` has four sections: Database connection errors, High CPU on web servers, Disk space almost full, and Certificate expiry. Add more `.txt` files to `runbooks/` to grow the knowledge base.

## Setup

```
python3 -m venv .venv
.venv/bin/pip install openai scikit-learn pytest
```

Set `XAI_API_KEY` in your Terminal tab first. Never put the key in a file.

## Usage

```
.venv/bin/python agent.py "web server CPU is at 95 percent"
.venv/bin/python agent.py --force-action
```

`--force-action` (or `-f`) skips the model and goes straight to the approval prompt with a sample action, so you can try the gate without an API key.

## Tests

```
.venv/bin/python -m pytest -q
```

52 tests pass. They use fake data and a fake model, so they need no key and no network.

## Named failures

Errors have names that say what went wrong:

- `retrieve.py`: NO_RUNBOOKS, EMPTY_QUERY, BAD_SETTINGS
- `agent.py`: MISSING_KEY, BAD_KEY, MODEL_ERROR, BAD_TOOL_ARGS, EMPTY_REQUEST, UNKNOWN_TOOL, NOT_IMPLEMENTED_APPROVAL

## Limits

- Search uses word matching without stemming, so "servers" does not match "server". A move to embeddings is a possible upgrade.
- The runbooks are a small sample.
