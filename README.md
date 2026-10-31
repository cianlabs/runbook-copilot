# runbook-copilot

**Goal:** an agent that searches a runbook knowledge base (RAG) and takes actions via tool calls, with a human approval step before anything runs.

## Status

Working end to end, with a stub for the action itself. `execute_action` only prints what it would do and never changes a real system.

- Search: the `search_runbooks` tool uses `retrieve.py`, which splits each runbook at its `# ` headers first, then cuts each section into chunks (about 400 characters, 80 overlap) so a chunk never mixes sections. It ranks chunks with TF-IDF and returns the top 3 with scores.
- Agent loop: `agent.py` asks the model (Grok, model `grok-build-0.1`, through the xAI API) one step at a time. The model thinks, calls `search_runbooks`, reads the sections, and can search again. It then writes a short plain-language answer and cites the sections that answer relies on, or stops for a named reason. See "The agent loop" below.
- Approval gate: the agent shows the proposed action and related runbook sections, then asks `Approve this action? (y/n)`. Only `y` or `yes` lets it through. The model cannot approve its own action. The local page uses the same rule: nothing runs until you click Approve, and that click only simulates the action.
- Page: `app.py` is a small local web page so you can ask a question without the terminal. It shows the numbered steps, the Sources, the stop reason in plain words, and the step-by-step trace. See "The page" below.

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

**Citation rule.** A section is cited when its best score is at least `CITE_MIN_SCORE` in `agent.py` (0.15), or when the answer text names that section's title. A weaker neighbor (for example "High swap usage" at 0.085 beside a CPU section at 0.49) is left off, unless the answer names it. If every read section is under 0.15, the loop does not answer at all: that stop is `WEAK_MATCH_ONLY`, and Sources stays empty. A title search did not return is never listed. A Sources block written by the model is removed and replaced by this rule. Search can still print a weak hit; Sources is the shorter list.

![When each stop reason happens, including an empty later search](images/stop-reasons.png)

Every `run_agent` call that finishes normally prints one line, `Stop reason: NAME`. That name is how you tell why it stopped. It is also stored on `agent.last_trace.stop_reason`.

- `ANSWERED` — the model had enough information and wrote an answer. This is the normal stop.
- `NO_RELEVANT_RUNBOOK` — the first search found nothing useful, and nothing had been read on an earlier step. The answer says "I don't know" and there is no Sources list. The loop writes that itself and stops. An empty search on a later step does not do this: sections already read stay, and the loop keeps going.
- `WEAK_MATCH_ONLY` — search returned sections, but every score was under `CITE_MIN_SCORE` (0.15) and nothing stronger was read earlier. That is not enough to answer from. The loop says "I don't know", cites nothing, and does not ask the model to invent steps. The weak hits are still printed so you can see why. A later weak search does not erase a strong section that was already read.
- `UNGROUNDED_ANSWER` — the model wrote an answer before reading any section (numbered steps, or anything that is not a short request for more detail). Those words are dropped. The loop says "I don't know" and cites nothing. A short "Please give me more detail." is still shown, because that is a question, not a guess.
- `MAX_STEPS_REACHED` — the loop used every step allowed by `MAX_STEPS` in `agent.py` (4) and still had no answer. It cites the sections the citation rule kept and stops. See "The step cap" below.
- `ACTION_REQUESTED` — the model asked for `execute_action`. The approval gate runs next, the same way as before. On the page, the action waits for a button instead.

`WEAK_MATCH_ONLY` and `UNGROUNDED_ANSWER` are the same idea as `NO_RELEVANT_RUNBOOK`: do not guess. They have their own names so a log shows which guardrail fired.

![I-don't-know stops, the hard step cap, and the local page](images/guardrails.png)

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
The copilot answered from the runbook sections it read.
Answer: 1. Find the worker that is using the CPU.
2. Restart that worker.

Sources:
- High CPU on web servers (sample_runbook.txt)
```

The pictures above are drawn by `make_diagrams.py` (Pillow). Run `python make_diagrams.py` to regenerate the PNGs in `images/`. No secrets are drawn into them.

## The step cap

`MAX_STEPS` in `agent.py` is 4. That is the only limit, and the loop reads it at the start of each question.

Before every model call the loop asks a small counter, `ModelCallBudget`, whether another call is still allowed. When the count is already 4, the loop does not call the model. It stops with `MAX_STEPS_REACHED` and a message that says it will not guess past the sections it already read. A model that would search forever still stops at step 4. Tests cover that with a fake model that never stops asking for `search_runbooks`.

An answer on the last allowed step is still `ANSWERED`. The cap stops extra calls. It does not throw away a real answer.

## The page

The page is `app.py`. It uses Python's built-in web server. Gradio, Streamlit, or Flask would mean another install for one form and two buttons. This page adds no packages, starts with one command, and the HTML sits in the same file so it is easy to read.

```
.venv/bin/python app.py
```

Then open http://127.0.0.1:8000

The page listens only on your own computer. It shows:

- the numbered steps
- the Sources list (or a line that says there are no sources)
- why it stopped, in a plain sentence, plus the code name such as `ANSWERED`
- the step-by-step trace (what it thought, which tool it called, what it read)

If the copilot wants to restart something, the page shows the action and two buttons: **Approve and simulate** and **Do not run**. Asking the question does not run it. Opening the page does not run it. Only the Approve button does, and that path is the same simulation as the terminal (`settle_action` in `agent.py`). No real process is signaled.

**Practice mode and live mode.** If `XAI_API_KEY` is not set, the page uses a built-in practice model (`DemoClient` in `app.py`). It searches the runbooks and quotes one sentence from each matching section. It does not call the model network, so you can try the page with no key. If the key is set, the page uses the real model through `agent.make_client`. The key is read from the terminal. It is not printed and it is not written into the page. Set the key in the same terminal, then start `app.py` again.

The tests in `test_agent.py` and `test_app.py` drive the loop and the page with a fake model. They need no API key and make no network calls. To try the terminal yourself with the real model, set `XAI_API_KEY` in the terminal (never in a file) and run:

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
.venv/bin/python app.py
.venv/bin/python agent.py "web server CPU is at 95 percent"
.venv/bin/python agent.py --force-action
```

`app.py` opens the page at http://127.0.0.1:8000. It works with no API key (practice mode). `--force-action` (or `-f`) skips the model and goes straight to the approval prompt with a sample action, so you can try the gate in the terminal without an API key.

## Tests

```
.venv/bin/python -m pytest -q
```

147 tests pass. They use fake data and a fake model, so they need no key and no network. `test_app.py` also starts the page on a random local port and clicks through ask, refuse, and approve. Run the scenarios with `.venv/bin/python evals.py` (add `--real` only when you want the live model).

## Named failures

Errors have names that say what went wrong:

- `retrieve.py`: NO_RUNBOOKS, EMPTY_QUERY, BAD_SETTINGS
- `agent.py`: MISSING_KEY, BAD_KEY, MODEL_ERROR, BAD_TOOL_ARGS, EMPTY_REQUEST, UNKNOWN_TOOL, NOT_IMPLEMENTED_APPROVAL, UNKNOWN_ACTION
- `app.py`: STALE_APPROVAL (the Approve click did not match the action that is waiting), APPROVAL_NEEDS_A_CLICK (the page tried to ask in the terminal instead of using the button), PORT_IN_USE, PORT_NOT_A_NUMBER
- Stop reasons in `agent.py` are not crashes. They are ANSWERED, NO_RELEVANT_RUNBOOK, WEAK_MATCH_ONLY, UNGROUNDED_ANSWER, MAX_STEPS_REACHED, and ACTION_REQUESTED. See "The agent loop".

## Known limits

- Search folds simple word endings, so "deployment" matches "deploy". A later upgrade could switch this to embeddings.
- The runbooks are a small sample.
- The page is local only (`127.0.0.1`, port 8000). It is not a hosted site.
- Practice mode quotes one sentence from each matching section. The real model is asked to paraphrase. A practice-mode question that starts with "restart" proposes that restart and does not search first. A question that only mentions restarting, such as "what does the runbook say about restarting", still searches.
- A short request for more detail, with no steps, is still shown when nothing has been read. Any other reply in that spot is `UNGROUNDED_ANSWER`.
- When a section was actually read, the loop does not check that every sentence of the answer is supported by that section. It does refuse to invent citations: a Sources list written by the model is removed and replaced by the citation rule.
- Actions are still a simulation. The only known action is a restart.
