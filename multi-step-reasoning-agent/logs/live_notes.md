### Run history (read this before trusting the table)

| Batch | Cases | Model | Prompts |
|---|---|---|---|
| 1 | Q1, Q2 | `gemini-3.5-flash` | original |
| 2 | Q3-Q8 | `gemini-3.1-flash-lite` | original |
| 3 | Q9-Q13 | `gemini-3.5-flash-lite` | original |
| 4 | Q13 only (re-run) | `gemini-3.5-flash-lite` | verifier prompt tightened |

**Q13 failed on its first real-model run** (see `logs/q13_initial_failure.json`). The executor correctly
said UNSOLVABLE, but the verifier's independent re-solve answered `-3`, because the verifier prompt did not
say that an impossible state (giving away more apples than you have) means UNSOLVABLE. The verifier rejected
the correct answer, and the retry feedback then pushed the executor to the wrong answer `-3`.
Fix: a general rule plus a worked example (a different question) in `VERIFIER_SYSTEM_PROMPT`. Q13 was then re-run
and passed with 0 retries, and that is the result shown in the table.

**Caveat:** Q1-Q12 ran before the verifier prompt change and were not re-run (free-tier quota), so the table mixes two
prompt versions. The change only adds an UNSOLVABLE rule and one example, but this is not a single clean run.
Different models were used per batch, and the lite models are weaker than a full model.
