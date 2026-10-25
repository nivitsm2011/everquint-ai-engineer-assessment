# Example runs - REAL MODEL

> Provider: Google Gemini, free tier. Each free model allows only 20 requests per day and one
> question needs 3 or more calls, so the 13 questions were run in batches on different models.
> The **Model** column shows which model produced each result. "NOT EVALUATED" means the API
> rejected every call (no model answer exists), so it is not a model failure.
>
> The full 13-case run on the mock provider is in [`example_runs_mock.md`](example_runs_mock.md).
> The mock is canned, so it proves the pipeline logic, not the model's accuracy.

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

**13/13 evaluated cases passed, 0 retries triggered**

| # | Category | Model | Status | Retries |
|---|---|---|---|---|
| Q1 | easy | gemini (gemini-3.5-flash) | PASS | 0 |
| Q2 | easy | gemini (gemini-3.5-flash) | PASS | 0 |
| Q3 | easy | gemini (gemini-3.1-flash-lite) | PASS | 0 |
| Q4 | easy | gemini (gemini-3.1-flash-lite) | PASS | 0 |
| Q5 | easy | gemini (gemini-3.1-flash-lite) | PASS | 0 |
| Q6 | easy | gemini (gemini-3.1-flash-lite) | PASS | 0 |
| Q7 | easy | gemini (gemini-3.1-flash-lite) | PASS | 0 |
| Q8 | easy | gemini (gemini-3.1-flash-lite) | PASS | 0 |
| Q9 | tricky | gemini (gemini-3.5-flash-lite) | PASS | 0 |
| Q10 | tricky | gemini (gemini-3.5-flash-lite) | PASS | 0 |
| Q11 | tricky | gemini (gemini-3.5-flash-lite) | PASS | 0 |
| Q12 | tricky | gemini (gemini-3.5-flash-lite) | PASS | 0 |
| Q13 | tricky | gemini (gemini-3.5-flash-lite) | PASS | 0 |

## Q1 (easy): PASS

**Question:** If a train leaves at 14:30 and arrives at 18:05, how long is the journey?

- Model: gemini (gemini-3.5-flash)
- Expected: `215` (success)
- Got: `215` (success)
- Verifier passed: True
- Retries used: 0

```json
{
  "answer": "3 hours and 35 minutes (215 minutes)",
  "status": "success",
  "reasoning_visible_to_user": "The train leaves at 14:30 and arrives at 18:05, making the journey 215 minutes long (3 hours and 35 minutes).",
  "metadata": {
    "plan": "1. Parse departure (14:30) and arrival (18:05) times 2. Convert both times into total minutes past midnight 3. Subtract departure minutes from arrival minutes to find duration in minutes 4. Convert duration into hours and minutes 5. Format the final answer",
    "checks": [
      {
        "check_name": "output_format",
        "passed": true,
        "details": "All required keys present"
      },
      {
        "check_name": "tool_consistency",
        "passed": true,
        "details": "1 tool call(s) succeeded; ok"
      },
      {
        "check_name": "constraints",
        "passed": true,
        "details": "No negative values, valid times, answer present"
      },
      {
        "check_name": "independent_resolve",
        "passed": true,
        "details": "Independent answer '215' vs executor answer '215'"
      },
      {
        "check_name": "llm_verdict",
        "passed": true,
        "details": "Verdict: approve"
      }
    ],
    "retries": 0
  }
}
```

## Q2 (easy): PASS

**Question:** Alice has 3 red apples and twice as many green apples as red. How many apples does she have in total?

- Model: gemini (gemini-3.5-flash)
- Expected: `9` (success)
- Got: `9` (success)
- Verifier passed: True
- Retries used: 0

```json
{
  "answer": "9 apples",
  "status": "success",
  "reasoning_visible_to_user": "Alice has 6 green apples (twice 3), for a total of 3 + 6 = 9 apples.",
  "metadata": {
    "plan": "1. Extract the number of red apples (3) and green multiplier (2) 2. Compute the number of green apples by multiplying red apples by 2 3. Sum the red and green apples to find the total count 4. Format the final total as a plain number",
    "checks": [
      {
        "check_name": "output_format",
        "passed": true,
        "details": "All required keys present"
      },
      {
        "check_name": "tool_consistency",
        "passed": true,
        "details": "2 tool call(s) succeeded; ok"
      },
      {
        "check_name": "constraints",
        "passed": true,
        "details": "No negative values, valid times, answer present"
      },
      {
        "check_name": "independent_resolve",
        "passed": true,
        "details": "Independent answer '9' vs executor answer '9'"
      },
      {
        "check_name": "llm_verdict",
        "passed": true,
        "details": "Verdict: approve"
      }
    ],
    "retries": 0
  }
}
```

## Q3 (easy): PASS

**Question:** A meeting needs 60 minutes. There are free slots: 09:00-09:30, 09:45-10:30, 11:00-12:00. Which slots can fit the meeting?

- Model: gemini (gemini-3.1-flash-lite)
- Expected: `11:00-12:00` (success)
- Got: `11:00-12:00` (success)
- Verifier passed: True
- Retries used: 0

```json
{
  "answer": "Only 11:00-12:00 fits",
  "status": "success",
  "reasoning_visible_to_user": "Only the 11:00-12:00 slot is 60 minutes long; the others are shorter.",
  "metadata": {
    "plan": "1. Parse the required duration of 60 minutes 2. Calculate the duration of each provided time slot 3. Compare each slot duration to the required 60 minutes 4. Identify which slots have a duration of at least 60 minutes 5. Format the result as a list of fitting slots",
    "checks": [
      {
        "check_name": "output_format",
        "passed": true,
        "details": "All required keys present"
      },
      {
        "check_name": "tool_consistency",
        "passed": true,
        "details": "1 tool call(s) succeeded; ok"
      },
      {
        "check_name": "constraints",
        "passed": true,
        "details": "No negative values, valid times, answer present"
      },
      {
        "check_name": "independent_resolve",
        "passed": true,
        "details": "Independent answer '11:00-12:00' vs executor answer '11:00-12:00'"
      },
      {
        "check_name": "llm_verdict",
        "passed": true,
        "details": "Verdict: approve"
      }
    ],
    "retries": 0
  }
}
```

## Q4 (easy): PASS

**Question:** What is 15% of 240?

- Model: gemini (gemini-3.1-flash-lite)
- Expected: `36` (success)
- Got: `36` (success)
- Verifier passed: True
- Retries used: 0

```json
{
  "answer": "36",
  "status": "success",
  "reasoning_visible_to_user": "15% of 240 is calculated by multiplying 240 by 0.15, which equals 36.",
  "metadata": {
    "plan": "1. Convert the percentage 15% to a decimal (0.15) 2. Multiply the decimal by the base number 240 3. Format the result as a plain number",
    "checks": [
      {
        "check_name": "output_format",
        "passed": true,
        "details": "All required keys present"
      },
      {
        "check_name": "tool_consistency",
        "passed": true,
        "details": "1 tool call(s) succeeded; ok"
      },
      {
        "check_name": "constraints",
        "passed": true,
        "details": "No negative values, valid times, answer present"
      },
      {
        "check_name": "independent_resolve",
        "passed": true,
        "details": "Independent answer '36' vs executor answer '36'"
      },
      {
        "check_name": "llm_verdict",
        "passed": true,
        "details": "Verdict: approve"
      }
    ],
    "retries": 0
  }
}
```

## Q5 (easy): PASS

**Question:** A book has 12 chapters with 25 pages each. How many pages in total?

- Model: gemini (gemini-3.1-flash-lite)
- Expected: `300` (success)
- Got: `300` (success)
- Verifier passed: True
- Retries used: 0

```json
{
  "answer": "300 pages",
  "status": "success",
  "reasoning_visible_to_user": "The book has 12 chapters with 25 pages each, resulting in 300 pages in total.",
  "metadata": {
    "plan": "1. Identify the number of chapters and pages per chapter 2. Multiply the number of chapters by the pages per chapter 3. Format the result as a plain number",
    "checks": [
      {
        "check_name": "output_format",
        "passed": true,
        "details": "All required keys present"
      },
      {
        "check_name": "tool_consistency",
        "passed": true,
        "details": "1 tool call(s) succeeded; ok"
      },
      {
        "check_name": "constraints",
        "passed": true,
        "details": "No negative values, valid times, answer present"
      },
      {
        "check_name": "independent_resolve",
        "passed": true,
        "details": "Independent answer '300' vs executor answer '300'"
      },
      {
        "check_name": "llm_verdict",
        "passed": true,
        "details": "Verdict: approve"
      }
    ],
    "retries": 0
  }
}
```

## Q6 (easy): PASS

**Question:** A bus leaves at 08:15 and the trip takes 2 hours 50 minutes. What time does it arrive?

- Model: gemini (gemini-3.1-flash-lite)
- Expected: `11:05` (success)
- Got: `11:05` (success)
- Verifier passed: True
- Retries used: 0

```json
{
  "answer": "11:05",
  "status": "success",
  "reasoning_visible_to_user": "Adding 2 hours and 50 minutes to 08:15 results in an arrival time of 11:05.",
  "metadata": {
    "plan": "1. Parse the departure time as hours and minutes 2. Add the duration hours to the departure hours 3. Add the duration minutes to the departure minutes 4. Adjust the time if the total minutes exceed 60 5. Format the final time as HH:MM",
    "checks": [
      {
        "check_name": "output_format",
        "passed": true,
        "details": "All required keys present"
      },
      {
        "check_name": "tool_consistency",
        "passed": true,
        "details": "1 tool call(s) succeeded; ok"
      },
      {
        "check_name": "constraints",
        "passed": true,
        "details": "No negative values, valid times, answer present"
      },
      {
        "check_name": "independent_resolve",
        "passed": true,
        "details": "Independent answer '11:05' vs executor answer '11:05'"
      },
      {
        "check_name": "llm_verdict",
        "passed": true,
        "details": "Verdict: approve"
      }
    ],
    "retries": 0
  }
}
```

## Q7 (easy): PASS

**Question:** Tom had 50 rupees, spent 18 on lunch and 7 on tea. How much is left?

- Model: gemini (gemini-3.1-flash-lite)
- Expected: `25` (success)
- Got: `25` (success)
- Verifier passed: True
- Retries used: 0

```json
{
  "answer": "25 rupees",
  "status": "success",
  "reasoning_visible_to_user": "Tom spent 25 rupees in total (18 for lunch and 7 for tea), leaving him with 25 rupees from his original 50.",
  "metadata": {
    "plan": "1. Identify the initial amount and the costs incurred 2. Calculate the total spent by adding lunch and tea costs 3. Subtract the total spent from the initial amount 4. Format the result as a plain number",
    "checks": [
      {
        "check_name": "output_format",
        "passed": true,
        "details": "All required keys present"
      },
      {
        "check_name": "tool_consistency",
        "passed": true,
        "details": "2 tool call(s) succeeded; ok"
      },
      {
        "check_name": "constraints",
        "passed": true,
        "details": "No negative values, valid times, answer present"
      },
      {
        "check_name": "independent_resolve",
        "passed": true,
        "details": "Independent answer '25' vs executor answer '25'"
      },
      {
        "check_name": "llm_verdict",
        "passed": true,
        "details": "Verdict: approve"
      }
    ],
    "retries": 0
  }
}
```

## Q8 (easy): PASS

**Question:** What is the average of 12, 18 and 30?

- Model: gemini (gemini-3.1-flash-lite)
- Expected: `20` (success)
- Got: `20` (success)
- Verifier passed: True
- Retries used: 0

```json
{
  "answer": "20",
  "status": "success",
  "reasoning_visible_to_user": "The sum of 12, 18, and 30 is 60, and dividing by 3 gives an average of 20.",
  "metadata": {
    "plan": "1. Identify the numbers to be averaged 2. Calculate the sum of the numbers 3. Divide the sum by the count of numbers 4. Format the result as a plain number",
    "checks": [
      {
        "check_name": "output_format",
        "passed": true,
        "details": "All required keys present"
      },
      {
        "check_name": "tool_consistency",
        "passed": true,
        "details": "2 tool call(s) succeeded; ok"
      },
      {
        "check_name": "constraints",
        "passed": true,
        "details": "No negative values, valid times, answer present"
      },
      {
        "check_name": "independent_resolve",
        "passed": true,
        "details": "Independent answer '20' vs executor answer '20'"
      },
      {
        "check_name": "llm_verdict",
        "passed": true,
        "details": "Verdict: approve"
      }
    ],
    "retries": 0
  }
}
```

## Q9 (tricky): PASS

**Question:** A train leaves at 22:45 and arrives at 01:15 the next day. How long is the journey?

- Model: gemini (gemini-3.5-flash-lite)
- Expected: `150` (success)
- Got: `150` (success)
- Verifier passed: True
- Retries used: 0

```json
{
  "answer": "150 minutes (2 hours and 30 minutes)",
  "status": "success",
  "reasoning_visible_to_user": "The journey takes 75 minutes from 22:45 to midnight, plus another 75 minutes from midnight to 01:15, for a total of 150 minutes.",
  "metadata": {
    "plan": "1. Parse the departure time as 22:45 on day 1 2. Parse the arrival time as 01:15 on day 2 3. Calculate the time remaining from departure until midnight 4. Add the time from midnight to the arrival time 5. Format the total journey duration",
    "checks": [
      {
        "check_name": "output_format",
        "passed": true,
        "details": "All required keys present"
      },
      {
        "check_name": "tool_consistency",
        "passed": true,
        "details": "3 tool call(s) succeeded; ok"
      },
      {
        "check_name": "constraints",
        "passed": true,
        "details": "No negative values, valid times, answer present"
      },
      {
        "check_name": "independent_resolve",
        "passed": true,
        "details": "Independent answer '150' vs executor answer '150'"
      },
      {
        "check_name": "llm_verdict",
        "passed": true,
        "details": "Verdict: approve"
      }
    ],
    "retries": 0
  }
}
```

## Q10 (tricky): PASS

**Question:** A meeting needs 30 minutes. Free slots: 09:00-09:30, 10:00-10:20, 13:15-13:45. Which slots fit?

- Model: gemini (gemini-3.5-flash-lite)
- Expected: `09:00-09:30,13:15-13:45` (success)
- Got: `09:00-09:30,13:15-13:45` (success)
- Verifier passed: True
- Retries used: 0

```json
{
  "answer": "09:00-09:30, 13:15-13:45",
  "status": "success",
  "reasoning_visible_to_user": "The slots 09:00-09:30 and 13:15-13:45 are each at least 30 minutes long and fit the meeting.",
  "metadata": {
    "plan": "1. Parse the required meeting duration of 30 minutes and the list of free slots 2. Calculate the duration in minutes for each free slot 3. Filter the slots that are greater than or equal to 30 minutes 4. Format the fitting slots as a comma-separated list of time ranges",
    "checks": [
      {
        "check_name": "output_format",
        "passed": true,
        "details": "All required keys present"
      },
      {
        "check_name": "tool_consistency",
        "passed": true,
        "details": "1 tool call(s) succeeded; ok"
      },
      {
        "check_name": "constraints",
        "passed": true,
        "details": "No negative values, valid times, answer present"
      },
      {
        "check_name": "independent_resolve",
        "passed": true,
        "details": "Independent answer '09:00-09:30,13:15-13:45' vs executor answer '09:00-09:30,13:15-13:45'"
      },
      {
        "check_name": "llm_verdict",
        "passed": true,
        "details": "Verdict: approve"
      }
    ],
    "retries": 0
  }
}
```

## Q11 (tricky): PASS

**Question:** A bat and a ball cost 1.10 in total. The bat costs 1.00 more than the ball. How much does the ball cost?

- Model: gemini (gemini-3.5-flash-lite)
- Expected: `0.05` (success)
- Got: `0.05` (success)
- Verifier passed: True
- Retries used: 0

```json
{
  "answer": "0.05",
  "status": "success",
  "reasoning_visible_to_user": "If the bat costs 1.00 more than the ball, and they total 1.10, the ball must cost 0.05 and the bat 1.05.",
  "metadata": {
    "plan": "1. Set up equations where bat + ball = 1.10 and bat = ball + 1.00 2. Substitute the second equation into the first: (ball + 1.00) + ball = 1.10 3. Simplify to 2 * ball + 1.00 = 1.10 4. Solve for ball by subtracting 1.00 and dividing by 2 5. Format the answer as a decimal number",
    "checks": [
      {
        "check_name": "output_format",
        "passed": true,
        "details": "All required keys present"
      },
      {
        "check_name": "tool_consistency",
        "passed": true,
        "details": "2 tool call(s) succeeded; ok"
      },
      {
        "check_name": "constraints",
        "passed": true,
        "details": "No negative values, valid times, answer present"
      },
      {
        "check_name": "independent_resolve",
        "passed": true,
        "details": "Independent answer '0.05' vs executor answer '0.05'"
      },
      {
        "check_name": "llm_verdict",
        "passed": true,
        "details": "Verdict: approve"
      }
    ],
    "retries": 0
  }
}
```

## Q12 (tricky): PASS

**Question:** A recipe needs 3 eggs per cake. How many eggs are needed for 2 dozen cakes?

- Model: gemini (gemini-3.5-flash-lite)
- Expected: `72` (success)
- Got: `72` (success)
- Verifier passed: True
- Retries used: 0

```json
{
  "answer": "72",
  "status": "success",
  "reasoning_visible_to_user": "2 dozen cakes is 24 cakes, and at 3 eggs per cake, you need 24 * 3 = 72 eggs in total.",
  "metadata": {
    "plan": "1. Extract eggs per cake = 3 and cakes in dozens = 2 2. Convert 2 dozen cakes into a total number of cakes (2 * 12) 3. Multiply total cakes by eggs per cake to find the total eggs needed 4. Format the answer as a plain number",
    "checks": [
      {
        "check_name": "output_format",
        "passed": true,
        "details": "All required keys present"
      },
      {
        "check_name": "tool_consistency",
        "passed": true,
        "details": "2 tool call(s) succeeded; ok"
      },
      {
        "check_name": "constraints",
        "passed": true,
        "details": "No negative values, valid times, answer present"
      },
      {
        "check_name": "independent_resolve",
        "passed": true,
        "details": "Independent answer '72' vs executor answer '72'"
      },
      {
        "check_name": "llm_verdict",
        "passed": true,
        "details": "Verdict: approve"
      }
    ],
    "retries": 0
  }
}
```

## Q13 (tricky): PASS

**Question:** Alice has 5 apples and gives away 8. How many apples does she have left?

- Model: gemini (gemini-3.5-flash-lite)
- Expected: `UNSOLVABLE` (failed)
- Got: `UNSOLVABLE` (failed)
- Verifier passed: True
- Retries used: 0

```json
{
  "answer": "Cannot be solved",
  "status": "failed",
  "reasoning_visible_to_user": "Alice starts with 5 apples but cannot give away 8, resulting in a negative count of apples.",
  "metadata": {
    "plan": "1. Extract initial apples = 5 and given away = 8 2. Compute final_apples = initial_apples - given_away 3. Format the answer as a plain number",
    "checks": [
      {
        "check_name": "output_format",
        "passed": true,
        "details": "All required keys present (both sides agree: UNSOLVABLE)"
      },
      {
        "check_name": "tool_consistency",
        "passed": true,
        "details": "1 tool call(s) succeeded; ok (both sides agree: UNSOLVABLE)"
      },
      {
        "check_name": "constraints",
        "passed": true,
        "details": "step 1: negative result -3.0 (both sides agree: UNSOLVABLE)"
      },
      {
        "check_name": "independent_resolve",
        "passed": true,
        "details": "Independent answer 'UNSOLVABLE' vs executor answer 'UNSOLVABLE' (both sides agree: UNSOLVABLE)"
      },
      {
        "check_name": "llm_verdict",
        "passed": true,
        "details": "Verdict: approve (both sides agree: UNSOLVABLE)"
      }
    ],
    "retries": 0
  }
}
```
