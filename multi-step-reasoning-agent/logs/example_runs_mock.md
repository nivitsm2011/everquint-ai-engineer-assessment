# Example runs - MOCK PROVIDER (canned responses)

> These results come from the mock LLM, whose answers are canned. They verify the pipeline
> (planner, executor, tools, verifier, retry loop), not the accuracy of a real model.
> Real-model results are in [`example_runs.md`](example_runs.md).

**13/13 evaluated cases passed, 1 retries triggered**

| # | Category | Model | Status | Retries |
|---|---|---|---|---|
| Q1 | easy | mock | PASS | 0 |
| Q2 | easy | mock | PASS | 0 |
| Q3 | easy | mock | PASS | 0 |
| Q4 | easy | mock | PASS | 0 |
| Q5 | easy | mock | PASS | 0 |
| Q6 | easy | mock | PASS | 0 |
| Q7 | easy | mock | PASS | 0 |
| Q8 | easy | mock | PASS | 0 |
| Q9 | tricky | mock | PASS | 0 |
| Q10 | tricky | mock | PASS | 0 |
| Q11 | tricky | mock | PASS | 1 |
| Q12 | tricky | mock | PASS | 0 |
| Q13 | tricky | mock | PASS | 0 |

## Q1 (easy): PASS

**Question:** If a train leaves at 14:30 and arrives at 18:05, how long is the journey?

- Model: mock
- Expected: `215` (success)
- Got: `215` (success)
- Verifier passed: True
- Retries used: 0

```json
{
  "answer": "3 h 35 min (215 minutes)",
  "status": "success",
  "reasoning_visible_to_user": "The train leaves at 14:30 and arrives at 18:05, so the journey takes 3 h 35 min.",
  "metadata": {
    "plan": "1. Parse the departure and arrival times 2. Compute the minutes between them 3. Format the answer as integer minutes",
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

- Model: mock
- Expected: `9` (success)
- Got: `9` (success)
- Verifier passed: True
- Retries used: 0

```json
{
  "answer": "9 apples",
  "status": "success",
  "reasoning_visible_to_user": "Alice has 6 green apples (twice 3), so 3 + 6 = 9 apples in total.",
  "metadata": {
    "plan": "1. Extract red = 3 2. Compute green = red * 2 3. Compute total = red + green 4. Format as a plain number",
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

- Model: mock
- Expected: `11:00-12:00` (success)
- Got: `11:00-12:00` (success)
- Verifier passed: True
- Retries used: 0

```json
{
  "answer": "Only 11:00-12:00 fits",
  "status": "success",
  "reasoning_visible_to_user": "Only the 11:00-12:00 slot (60 minutes) is long enough for the meeting.",
  "metadata": {
    "plan": "1. Parse the slots and the needed duration 2. Keep slots of at least 60 minutes 3. Format as a sorted list",
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

- Model: mock
- Expected: `36` (success)
- Got: `36` (success)
- Verifier passed: True
- Retries used: 0

```json
{
  "answer": "36",
  "status": "success",
  "reasoning_visible_to_user": "15% of 240 is 240 x 0.15 = 36.",
  "metadata": {
    "plan": "1. Extract 15% and 240 2. Compute 240 * 15 / 100 3. Format as a plain number",
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

- Model: mock
- Expected: `300` (success)
- Got: `300` (success)
- Verifier passed: True
- Retries used: 0

```json
{
  "answer": "300 pages",
  "status": "success",
  "reasoning_visible_to_user": "12 chapters of 25 pages each make 300 pages.",
  "metadata": {
    "plan": "1. Extract chapters and pages per chapter 2. Multiply them 3. Format as a plain number",
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

- Model: mock
- Expected: `11:05` (success)
- Got: `11:05` (success)
- Verifier passed: True
- Retries used: 0

```json
{
  "answer": "11:05",
  "status": "success",
  "reasoning_visible_to_user": "Leaving at 08:15 and travelling 2 h 50 min, the bus arrives at 11:05.",
  "metadata": {
    "plan": "1. Convert the trip length to minutes 2. Add it to the departure time 3. Format as HH:MM",
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

- Model: mock
- Expected: `25` (success)
- Got: `25` (success)
- Verifier passed: True
- Retries used: 0

```json
{
  "answer": "25 rupees",
  "status": "success",
  "reasoning_visible_to_user": "Tom spent 18 + 7 = 25 rupees, so 25 of his 50 rupees are left.",
  "metadata": {
    "plan": "1. Extract the amounts 2. Subtract both expenses 3. Format as a plain number",
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

- Model: mock
- Expected: `20` (success)
- Got: `20` (success)
- Verifier passed: True
- Retries used: 0

```json
{
  "answer": "20",
  "status": "success",
  "reasoning_visible_to_user": "The numbers add up to 60, and 60 divided by 3 is 20.",
  "metadata": {
    "plan": "1. List the numbers 2. Add them and divide by the count 3. Format as a plain number",
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

- Model: mock
- Expected: `150` (success)
- Got: `150` (success)
- Verifier passed: True
- Retries used: 0

```json
{
  "answer": "2 h 30 min (150 minutes)",
  "status": "success",
  "reasoning_visible_to_user": "The journey crosses midnight: 75 minutes to midnight plus 75 after, 2 h 30 min.",
  "metadata": {
    "plan": "1. Parse the times 2. Compute the minutes, allowing for passing midnight 3. Format as integer minutes",
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

- Model: mock
- Expected: `09:00-09:30,13:15-13:45` (success)
- Got: `09:00-09:30,13:15-13:45` (success)
- Verifier passed: True
- Retries used: 0

```json
{
  "answer": "09:00-09:30 and 13:15-13:45 fit",
  "status": "success",
  "reasoning_visible_to_user": "Two slots are exactly 30 minutes long, which is enough; 10:00-10:20 is too short.",
  "metadata": {
    "plan": "1. Parse the slots and the needed duration 2. Keep slots of at least 30 minutes (exact fit counts) 3. Format as a sorted list",
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

- Model: mock
- Expected: `0.05` (success)
- Got: `0.05` (success)
- Verifier passed: True
- Retries used: 1

```json
{
  "answer": "0.05",
  "status": "success",
  "reasoning_visible_to_user": "If the ball costs 0.05 the bat costs 1.05, together 1.10 and 1.00 apart.",
  "metadata": {
    "plan": "1. Let ball = x and bat = x + 1.00 2. Set x + (x + 1.00) = 1.10 and solve for x 3. Check bat - ball = 1.00 4. Format as a plain number",
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
        "details": "Independent answer '0.05' vs executor answer '0.05'"
      },
      {
        "check_name": "llm_verdict",
        "passed": true,
        "details": "Verdict: approve"
      }
    ],
    "retries": 1
  }
}
```

## Q12 (tricky): PASS

**Question:** A recipe needs 3 eggs per cake. How many eggs are needed for 2 dozen cakes?

- Model: mock
- Expected: `72` (success)
- Got: `72` (success)
- Verifier passed: True
- Retries used: 0

```json
{
  "answer": "72 eggs",
  "status": "success",
  "reasoning_visible_to_user": "2 dozen is 24 cakes, and at 3 eggs each that needs 72 eggs.",
  "metadata": {
    "plan": "1. Convert 2 dozen to 24 cakes 2. Multiply by 3 eggs per cake 3. Format as a plain number",
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

- Model: mock
- Expected: `UNSOLVABLE` (failed)
- Got: `UNSOLVABLE` (failed)
- Verifier passed: True
- Retries used: 0

```json
{
  "answer": "Cannot be solved",
  "status": "failed",
  "reasoning_visible_to_user": "Alice cannot give away 8 apples when she only has 5.",
  "metadata": {
    "plan": "1. Extract the quantities 2. Check Alice can give away that many 3. Report the result",
    "checks": [
      {
        "check_name": "output_format",
        "passed": true,
        "details": "All required keys present (both sides agree: UNSOLVABLE)"
      },
      {
        "check_name": "tool_consistency",
        "passed": true,
        "details": "0 tool call(s) succeeded; ok (both sides agree: UNSOLVABLE)"
      },
      {
        "check_name": "constraints",
        "passed": true,
        "details": "No negative values, valid times, answer present (both sides agree: UNSOLVABLE)"
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
