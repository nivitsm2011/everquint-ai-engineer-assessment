# Max Profit Optimization

## Problem Statement

Determine the maximum earnings achievable within a given time limit
by constructing Theatre (T), Pub (P), and Commercial Park (C) buildings.

## Building Details

| Building            | Construction Time | Earnings per Time Unit |
| ------------------- | ----------------: | ---------------------: |
| Theatre (T)         |                 5 |                 $1,500 |
| Pub (P)             |                 4 |                 $1,000 |
| Commercial Park (C) |                10 |                 $2,000 |

## Approach

The solution uses recursion and memoization to find an optimal
sequence of buildings.

For each possible building:

1. Check whether it can be completed within the remaining time.
2. Calculate its earnings for the time remaining after construction.
3. Recursively calculate the best possible earnings from the remaining time.
4. Compare the total earnings and retain the best sequence.

Memoization stores previously calculated results to avoid
repeating the same computations.

## Output

The program returns:

- Maximum achievable earnings
- Number of each building constructed
- One optimal construction sequence

If multiple sequences achieve the same maximum earnings, the
program returns one of them.

## Requirements

- Python 3.x
- Standard Python library (`functools`)

## How to Run

Run the following command in a terminal:

python max_profit.py

## Example Test Cases

| Time Unit | Expected Maximum Earnings |
| --------: | ------------------------: |
|         7 |                    $3,000 |
|         8 |                    $4,500 |
|        13 |                   $16,500 |

## Additional Testing

The implementation can also be tested with:

- Zero and small time limits
- Time limits near building construction times
- Larger time limits

## Author

Niveditha
