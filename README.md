# runbook-copilot

**Goal:** an agent that searches a runbook knowledge base (RAG) and takes actions via tool calls, with a human approval step before anything runs.

## Status

Working end to end, with a stub for the action itself. `execute_action` only prints what it would do and never changes a real system.

- Search: the `search_runbooks` tool uses `retrieve.py`, which splits each runbook at its `# ` headers first, then cuts each section into chunks (about 400 characters, 80 overlap) so a chunk never mixes sections. It ranks chunks with TF-IDF and returns the top 3 with scores.
- Agent loop: `agent.py` asks the model (Grok, model `grok-build-0.1`, through the xAI API) one step at a time. The model thinks, calls `search_runbooks`, reads the sections, and can search again. It then writes a short plain-language answer and cites the sections that answer relies on, or stops for a named reason. See "The agent loop" below.
- Approval gate: the agent shows the proposed action and related runbook sections, then asks `Approve this action? (y/n)`. Only `y` or `yes` lets it through. The model cannot approve its own action.

## How it works

1. Loop: think, search the runbooks, read the sections, and repeat until a stop reason. Details are in "The agent loop" below.
2. Propose: the model may ask for `execute_action`.
3. Approve: a human types y or n.
4. Execute: only approved actions run (stub for now).

## The agent loop

Asking a question runs a short loop. `run_agent` in `agent.py` repeats a few steps, then stops for a reason you can read.

![The repeating cycle: ask, think, search, read, then repeat or stop](images/agent-loop.png)

1. **Think.** The model writes what it still needs. If it stays quiet, the loop says which search it is about to run, so this step is never invisible.
2. **Search.** It calls the `search_runbooks` tool. That tool is `retrieve.search`, the same search the rest of the project uses.
3. **Read.** The sections that come back are printed and handed to the model. Each section is remembered once, even if two chunks share a title. A weak hit is still shown here. It is not a citation yet.
4. **Repeat or stop.** The model may search again with a better query. When it can answer, it stops calling tools and writes a short reply in plain language: a few numbered steps, in its own words, using only the sections it read. It should not paste the runbook text back under a bold title. The prompt still says do not invent sections, and say "I don't know" when nothing relevant was found. If it asks to do something, `execute_action` still stops at the approval question (`y` or `yes` only). `--force-action` still skips the model and goes straight to that question.

![The approval gate: only y or yes runs the simulated action](images/approval-gate.png)

The printed answer ends with a `Sources:` list. That list is built when the loop stops, from sections `search_runbooks` actually returned on this run. The same list is on `agent.last_trace.sources`.

**Citation rule.** A section is cited when its best score is at least `CITE_MIN_SCORE` in `agent.py` (0.15), or when the answer text names that section's title. A weaker neighbor (for example "High swap usage" at 0.085 beside a CPU section at 0.49) is left off, unless the answer names it. If every read section is under 0.15 and the answer names none of them, Sources still lists the highest-scoring one, so a run that read something never drops every citation. A title search did not return is never listed. Search can still print the weak hit; Sources is the shorter list.

![When each stop reason happens, including an empty later search](images/stop-reasons.png)

Every `run_agent` call that finishes normally prints one line, `Stop reason: NAME`. That name is how you tell why it stopped. It is also stored on `agent.last_trace.stop_reason`.

- `ANSWERED` — the model had enough information and wrote an answer. This is the normal stop.
- `NO_RELEVANT_RUNBOOK` — the first search found nothing useful, and nothing had been read on an earlier step. The answer says "I don't know" and there is no Sources list. The loop writes that itself and stops. An empty search on a later step does not do this: sections already read stay, and the loop keeps going.
- `MAX_STEPS_REACHED` — the loop used every step allowed by `MAX_STEPS` in `agent.py` (4) and still had no answer. It cites the sections the citation rule kept and stops, so the loop ends even if the model keeps asking to search.
- `ACTION_REQUESTED` — the model asked for `execute_action`. The approval gate runs next, the same way as before.

A run looks roughly like this:

```
Step 1 of 4
Thinking: I should search the runbooks for 'high cpu'.
Model chose: search_runbooks({'query': 'high cpu'})
Reading 1 result(s) from the runbooks.
1. [0.420] High CPU on web servers (sample_runbook.txt)
   ...
Step 2 of 4
Stop reason: ANSWERED
Answer: 1. Find the worker that is using the CPU.
2. Restart that worker.

Sources:
- High CPU on web servers (sample_runbook.txt)
```

The pictures above are drawn by `make_diagrams.py` (Pillow). Run `python make_diagrams.py` to regenerate the PNGs in `images/`.

The tests in `test_agent.py` drive this loop with a fake model. They need no API key and make no network calls. To try it yourself with the real model, set `XAI_API_KEY` in the terminal (never in a file) and run:

```
.venv/bin/python agent.py "web server CPU is at 95 percent"
```

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

124 tests pass. They use fake data and a fake model, so they need no key and no network. Run the scenarios with `.venv/bin/python evals.py` (add `--real` only when you want the live model).

## Named failures

Errors have names that say what went wrong:

- `retrieve.py`: NO_RUNBOOKS, EMPTY_QUERY, BAD_SETTINGS
- `agent.py`: MISSING_KEY, BAD_KEY, MODEL_ERROR, BAD_TOOL_ARGS, EMPTY_REQUEST, UNKNOWN_TOOL, NOT_IMPLEMENTED_APPROVAL, UNKNOWN_ACTION
- Stop reasons in `agent.py` are not crashes. They are ANSWERED, NO_RELEVANT_RUNBOOK, MAX_STEPS_REACHED, and ACTION_REQUESTED. See "The agent loop".

## Limits

- Search folds simple word endings, so "deployment" matches "deploy". A later upgrade could switch this to embeddings.
- The runbooks are a small sample.
