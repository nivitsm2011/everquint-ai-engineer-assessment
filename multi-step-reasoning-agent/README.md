# Multi-Step Reasoning Agent with Self-Checking

## 1. Overview

A small Python agent that solves short word problems (time, arithmetic, scheduling, simple logic).
It does not trust a single LLM answer. It plans, solves with Python tools doing the maths, then has a
verifier re-solve the problem independently. If any check fails, it retries with feedback.

```
Question
   |
   v
Planner  (LLM: writes a 3-6 step plan, does not solve)
   |
   v
Executor (LLM follows the plan; every calculation is re-run by Python tools)
   |
   v
Verifier (5 checks, including an independent re-solve by the LLM)
   |
   +-- all checks pass -----> final JSON (status "success")
   |
   +-- a check fails -------> feedback goes back to the Planner, retry (up to MAX_RETRIES)
   |
   +-- retries used up -----> final JSON (status "failed", answer marked "Unverified")
```

The LLM is swappable (Anthropic, Gemini or a built-in mock) behind one small interface.

## 2. Project structure

```
agent.py                 CLI entry point (interactive loop or --question)
requirements.txt         Python dependencies (not version-pinned)
.env.example             Template for settings (copy to .env)
.gitignore               Ignores .env, .venv, __pycache__, logs/*.jsonl
reasoner/__init__.py     Exposes solve(question) for notebooks and scripts
reasoner/config.py       Loads .env: provider, model names, API keys, MAX_RETRIES
reasoner/utils.py        extract_json(): pulls a JSON object out of an LLM reply
reasoner/llm_client.py   LLMClient interface; Anthropic, Gemini and Mock clients; get_llm_client()
reasoner/tools.py        Deterministic tools: safe_calc, time_diff, add_duration, slots_that_fit...
reasoner/prompts.py      ALL prompts (the only file containing prompt text)
reasoner/planner.py      Planner: question -> plan
reasoner/executor.py     Executor: plan -> steps + answer, then re-runs every step with tools
reasoner/verifier.py     Verifier: the 5 checks and the feedback text
reasoner/orchestrator.py ReasoningAgent: plan/execute/verify loop, retries, final JSON, trace log
tests/test_tools.py      Unit tests for the tools
tests/test_cases.py      The 13 end-to-end questions (8 easy, 5 tricky)
tests/test_agent.py      pytest tests of the whole agent (mock provider)
tests/run_tests.py       Runs the 13 questions (or chosen ones with --ids) and writes the reports below
logs/example_runs.md     Report of the real-model run, with the run history (see section 9)
logs/example_runs_mock.md Report of the full mock run (13/13)
logs/live_notes.md       Hand-written run history that is included in the real-model report
logs/q13_initial_failure.json  The first, failed real-model result for Q13 (kept as evidence, see section 8)
logs/trace.jsonl         Full trace of every question (git-ignored, created on first run)
logs/test_results_*.jsonl Raw per-case results (git-ignored)
```

## 3. Setup and how to run

Requires Python 3.10+ (developed on 3.13). Commands are for Windows PowerShell.
This project lives in the `multi-step-reasoning-agent/` folder of the assessment repository, so run everything
from inside that folder (the file paths in this README are relative to it).

```powershell
cd multi-step-reasoning-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Choose the provider in `.env`:

| `LLM_PROVIDER` | Needs | Notes |
|---|---|---|
| `mock` (default) | nothing | Canned answers for the 13 test questions only. Any other question returns a "failed" result. |
| `gemini` | `GEMINI_API_KEY` | Free keys are limited to 20 requests/day per model (see section 8). |
| `anthropic` | `ANTHROPIC_API_KEY` | Paid. |

If a real provider is selected but its key is missing, the agent prints a warning and falls back to the mock.

Run the CLI:

```powershell
python agent.py                                   # interactive: asks "Ask a question (or 'exit'):"
python agent.py --question "What is 15% of 240?"  # answer once and exit
python agent.py --debug --question "..."          # also prints the path of logs/trace.jsonl
```

The JSON answer goes to stdout; the "LLM provider: ..." line and the `--debug` path go to stderr.

Use it from Python or a notebook:

```python
from reasoner import solve
print(solve("A train leaves at 22:45 and arrives at 01:15 the next day. How long is the journey?"))
```

Run the tests:

```powershell
pytest -v                    # 13 unit and end-to-end tests (always uses the mock)
python tests/run_tests.py    # runs all 13 questions with the provider in .env; writes the reports
python tests/run_tests.py --ids 3 4 5   # run only some questions; results are merged into the report
python tests/run_tests.py --report-only # rebuild the report from saved results, no API calls
```

With a real provider the report is `logs/example_runs.md`; with the mock it is `logs/example_runs_mock.md`.
To run on a specific model without editing `.env`: `$env:GEMINI_MODEL="gemini-3.1-flash-lite"; python tests/run_tests.py`.

## 4. Where prompts live

All prompt text is in `reasoner/prompts.py`, as module-level constants. Each prompt states the exact JSON
output format and includes worked examples (train journey, Alice's apples, meeting slots).

| Constant | Purpose |
|---|---|
| `PLANNER_SYSTEM_PROMPT` | Role and rules for the planner; output `problem_type`, `quantities`, `plan`, `expected_answer_format` |
| `PLANNER_USER_TEMPLATE` | Fills in the question and, on a retry, the verifier feedback |
| `PLANNER_FEEDBACK_PREFIX` | Wording placed in front of the feedback text on a retry |
| `EXECUTOR_SYSTEM_PROMPT` | Follow the plan exactly, name a tool for every calculation, tool argument formats, `solvable` rules |
| `EXECUTOR_USER_TEMPLATE` | Fills in the question and the plan |
| `VERIFIER_SYSTEM_PROMPT` | Solve independently first, then compare; output verdict and issues |
| `VERIFIER_USER_TEMPLATE` | Fills in the question and the proposed solution |
| `_CANONICAL_RULES` (shared) | The machine-comparable answer format used by the executor and the verifier |

## 5. Output schema

`solve()` and the CLI always return exactly these four top-level keys:

```
answer                       str   short human answer
status                       str   "success" or "failed"
reasoning_visible_to_user    str   short explanation (never the full chain of thought)
metadata.plan                str   numbered plan steps joined into one string
metadata.checks              list  the five check results from the final attempt
metadata.retries             int   retries actually used (0 if the first attempt passed)
```

Real output from the Gemini run (`gemini-3.5-flash`) for "If a train leaves at 14:30 and arrives at 18:05, how long is the journey?":

```json
{
  "answer": "3 hours and 35 minutes (215 minutes)",
  "status": "success",
  "reasoning_visible_to_user": "The train leaves at 14:30 and arrives at 18:05, making the journey 215 minutes long (3 hours and 35 minutes).",
  "metadata": {
    "plan": "1. Parse departure (14:30) and arrival (18:05) times 2. Convert both times into total minutes past midnight 3. Subtract departure minutes from arrival minutes to find duration in minutes 4. Convert duration into hours and minutes 5. Format the final answer",
    "checks": [
      {"check_name": "output_format", "passed": true, "details": "All required keys present"},
      {"check_name": "tool_consistency", "passed": true, "details": "1 tool call(s) succeeded; ok"},
      {"check_name": "constraints", "passed": true, "details": "No negative values, valid times, answer present"},
      {"check_name": "independent_resolve", "passed": true, "details": "Independent answer '215' vs executor answer '215'"},
      {"check_name": "llm_verdict", "passed": true, "details": "Verdict: approve"}
    ],
    "retries": 0
  }
}
```

The full chain of thought (every attempt's plan, executor steps, tool results and verifier output) is
not shown to the user. It is appended, with a timestamp, to `logs/trace.jsonl` as one JSON line per question.

## 6. Verification design

After each attempt the verifier runs five checks. The attempt passes only if all five pass.

1. **output_format**: the executor output has all required keys (`solvable`, `steps`, `final_answer`, `canonical_answer`, `short_explanation`).
2. **tool_consistency**: every tool call succeeded. Steps where the tool result differed from the LLM's claimed result are noted (the executor has already replaced the claim with the tool value and marked it `corrected_by_tool`).
3. **constraints**: no negative tool results, valid HH:MM times in tool arguments, and a non-empty `canonical_answer`.
4. **independent_resolve**: the LLM solves the question again from scratch. Its `independent_canonical_answer` must equal the executor's `canonical_answer` after normalising (lowercase, no spaces, numbers compared as floats, list items sorted).
5. **llm_verdict**: the verifier's own verdict must be `approve`.

Special case: if the executor says `solvable: false` and the independent re-solve also says `UNSOLVABLE`, the checks are marked passed and the result is `status: "failed"` with answer "Cannot be solved".

Retry logic:
- `MAX_RETRIES` (default 2 in `.env.example`) means up to 3 attempts in total.
- On a failed attempt, the failed checks and the verifier's issues are turned into a feedback text and passed back to the planner for the next attempt.
- An error inside an attempt (invalid JSON, a failed LLM call) is recorded as a failed check (`attempt_error`) and the loop continues.
- If all attempts fail, the result is `status: "failed"` with the best attempt's answer prefixed `Unverified: `.
- Separately, each API call retries twice on errors. Rate-limit (HTTP 429) errors wait 20 s, then 40 s; other errors wait 1 s.

## 7. Assumptions

- Overnight times wrap to the next day: 22:45 to 01:15 is 150 minutes.
- An exact-fit slot counts as fitting: a 30-minute slot fits a 30-minute meeting.
- Impossible states (for example negative counts) return `status: "failed"` and the answer "Cannot be solved".
- Arithmetic is done in Python (`reasoner/tools.py`, using `ast`, never `eval`), not by the LLM.
- Temperature is 0 for the Anthropic and Gemini clients.
- Times are 24-hour HH:MM. "A dozen" is 12.

## 8. Prompt documentation

### Why the prompts are designed this way

- **Strict JSON output.** Every role must reply with one JSON object in a fixed format, so the code can parse it (`extract_json` also strips markdown fences and surrounding text).
- **Canonical answers for machine comparison.** The executor and verifier both produce a `canonical_answer` with the same rules (plain number, integer minutes, HH:MM, sorted comma-separated lists, or `UNSOLVABLE`). This lets code compare two answers exactly instead of comparing free text.
- **Tool delegation for arithmetic.** The executor must name a tool for every calculation, and the program re-runs it. A wrong claimed number is replaced by the tool's number.
- **Independent re-solve.** The verifier is told to solve the question first, before looking at the proposed answer, to avoid simply agreeing with it (confirmation bias).

### What did not work well (honest notes)

- **Slow model.** `gemini-3.8-flash` took 60 to 120 seconds for a single call, even a trivial one. Lowering its thinking level did not help. Switching to `gemini-3.5-flash` fixed it (about 4 s for a trivial call). A retired model (`gemini-2.5-flash`) also returned 404 for this account.
- **Retries could not fix rate limits.** The first retry logic waited 1 second, which is useless against HTTP 429. I added the 20 s / 40 s wait for rate-limit errors. It did not help with the free tier's daily cap of 20 requests per model. A 13-question run needs 39+ calls, so I split the real-model run into batches on different models (section 9).
- **The verifier prompt had a gap, found by the real model (Q13).** On the first real run the impossible question "Alice has 5 apples and gives away 8" failed. The executor correctly said UNSOLVABLE, but the verifier re-solved it independently and answered `-3`, because only the executor prompt knew that an impossible state means UNSOLVABLE. The verifier rejected the correct answer, and the retry feedback then pushed the executor to the wrong answer `-3` (the result is saved in `logs/q13_initial_failure.json`). The mock could never have shown this, since its answers are scripted. Fix: a general rule and a worked example on a different question in `VERIFIER_SYSTEM_PROMPT`. Q13 then passed with 0 retries. Lesson: an independent verifier needs the same problem rules as the solver, and feedback from a wrong verifier can make a correct solver worse.
- **Mixed prompt versions.** Q1 to Q12 ran before that fix and could not be re-run because of the free-tier quota, so the real-model table mixes two prompt versions (the change only adds the UNSOLVABLE rule and one example). They also ran on three different Gemini models, two of them "lite" models.
- **No other real-model failures seen.** Apart from Q13, the real runs showed no JSON parsing errors and no canonical-format mismatches, so I have no evidence of problems such as wrong mental arithmetic or answers that could not be compared. Those design choices are reasoning about likely failure modes, not fixes for failures I saw.
- **The retry demo is mock-only.** On the real model, bat-and-ball (Q11) was answered correctly at the first attempt (0 retries). The scenario where the verifier catches a wrong first answer (0.10 instead of 0.05) and the agent retries is demonstrated only with the mock, whose wrong first answer is scripted. The only real retries I saw were the failed Q13 attempts described above.
- **Known weaknesses of the checks.** The constraints check flags any negative tool result, so a legitimate negative intermediate value would fail it. In the "unsolvable" case every check is forced to passed, including `output_format`.
- **Mock limits.** The mock only knows the 13 test questions.

### What I would improve with more time

- Majority voting / self-consistency: sample several solutions and take the most common verified answer.
- An HTTP endpoint (for example FastAPI) around `solve()`.
- A larger and more varied test set, with the full run repeated in one pass, on one model and the final prompts, on a paid or higher-quota key.
- A cost and latency tracker per question.
- Unit-aware parsing (hours, minutes, currencies, percentages) instead of relying on the prompt and the canonical rules.
- Pin dependency versions in `requirements.txt`.

## 9. Example run logs

Full reports:
- [`logs/example_runs.md`](logs/example_runs.md): real models (Gemini, three different models in four batches). Read its "Run history" first.
- [`logs/example_runs_mock.md`](logs/example_runs_mock.md): all 13 questions with the mock provider.

The examples below are abridged (the five checks are summarised; `reasoning_visible_to_user` is shortened).

**Easy: Q1, real model (Gemini), 0 retries**

```json
{
  "answer": "3 hours and 35 minutes (215 minutes)",
  "status": "success",
  "reasoning_visible_to_user": "The train leaves at 14:30 and arrives at 18:05, making the journey 215 minutes long ...",
  "metadata": { "checks": "5/5 passed", "retries": 0 }
}
```

**Bat and ball: Q11, MOCK provider (scripted wrong first answer), 1 retry**

Attempt 1: the executor answered `0.10`. `independent_resolve` failed ("Independent answer '0.05' vs executor answer '0.10'") and `llm_verdict` failed, so the verifier's issues went back to the planner. Attempt 2:

```json
{
  "answer": "0.05",
  "status": "success",
  "reasoning_visible_to_user": "If the ball costs 0.05 the bat costs 1.05, together 1.10 and 1.00 apart.",
  "metadata": { "checks": "5/5 passed", "retries": 1 }
}
```

**Unsolvable: Q13, real model (Gemini `gemini-3.5-flash-lite`), after the verifier prompt fix, 0 retries**

```json
{
  "answer": "Cannot be solved",
  "status": "failed",
  "reasoning_visible_to_user": "Alice starts with 5 apples but cannot give away 8, resulting in a negative count of apples.",
  "metadata": { "checks": "5/5 passed (both sides agree: UNSOLVABLE)", "retries": 0 }
}
```

`status` is `"failed"` here by design: an impossible question has no valid answer. The first real run of this
question, before the fix, ended as `Unverified: -3 apples` (see section 8).

## 10. Test results summary

| Test run | Provider | Result |
|---|---|---|
| `pytest -v` (13 tests: 8 tool tests, 5 agent tests) | mock | 13 passed |
| `python tests/run_tests.py`, all 13 questions | mock | 13/13 passed, 1 retry (Q11) |
| `python tests/run_tests.py`, all 13 questions | Gemini (3 models, 4 batches, free tier) | 13/13 passed, 0 retries. **Q13 failed on its first real run** (verifier gap, see section 8); the 13/13 counts its re-run after the fix. Q1 to Q12 ran before the fix. |

| Real-model batch | Cases | Model | Result |
|---|---|---|---|
| 1 | Q1, Q2 | `gemini-3.5-flash` | 2/2 passed |
| 2 | Q3 to Q8 | `gemini-3.1-flash-lite` | 6/6 passed |
| 3 | Q9 to Q13 | `gemini-3.5-flash-lite` | 4/5 passed (Q13 failed) |
| 4 | Q13 re-run, fixed verifier prompt | `gemini-3.5-flash-lite` | 1/1 passed |

The mock run checks the pipeline logic (planner, executor, tools, verifier, retry loop). Its answers are
scripted, so it does not measure real-model accuracy. The real-model run is a single pass per question on
free-tier models, not a repeated benchmark, so it shows the agent works on these 13 questions but says
little about how often it would fail on others.
