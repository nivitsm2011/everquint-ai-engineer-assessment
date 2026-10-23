"""
Max Profit Problem
------------------
Find the maximum earnings by constructing Theatres, Pubs,
and Commercial Parks sequentially.

Assumption:
Each completed building earns money for every remaining
time unit after its construction is completed.
"""

from functools import lru_cache


# --------------------------------------------------
# 1. BUILDING CONFIGURATION
# --------------------------------------------------

BUILDINGS = {
    "T": {
        "name": "Theatre",
        "construction_time": 5,
        "earning_rate": 1500,
    },
    "P": {
        "name": "Pub",
        "construction_time": 4,
        "earning_rate": 1000,
    },
    "C": {
        "name": "Commercial Park",
        "construction_time": 10,
        "earning_rate": 2000,
    },
}


# --------------------------------------------------
# 2. INPUT VALIDATION
# --------------------------------------------------

def validate_time(n):
    """Ensure that n is a non-negative integer."""

    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError(
            "Time must be a non-negative integer."
        )


# --------------------------------------------------
# 3. OPTIMAL SOLUTION: RECURSION + MEMOIZATION
# --------------------------------------------------

@lru_cache(maxsize=None)
def best_profit(time_left):
    """
    Return the maximum earnings and an optimal building sequence.

    Each building is constructed sequentially.
    A completed building earns money during all remaining
    time units.

    Returns:
        (maximum_earnings, building_sequence)
    """

    max_earnings = 0
    best_sequence = ()

    for code, details in BUILDINGS.items():

        construction_time = details["construction_time"]
        earning_rate = details["earning_rate"]

        # Skip buildings that cannot be completed.
        if construction_time > time_left:
            continue

        # Remaining time after construction.
        remaining = time_left - construction_time

        # Earnings from this building.
        current_earnings = remaining * earning_rate

        # Best possible earnings from subsequent buildings.
        future_earnings, future_sequence = best_profit(remaining)

        total_earnings = current_earnings + future_earnings

        # Keep the best solution found.
        if total_earnings > max_earnings:
            max_earnings = total_earnings
            best_sequence = (code,) + future_sequence

    return max_earnings, best_sequence


# --------------------------------------------------
# 4. PUBLIC SOLUTION FUNCTION
# --------------------------------------------------

def solve(n):
    """
    Solve the Max Profit problem.

    Args:
        n: Available construction time.

    Returns:
        earnings: Maximum possible earnings.
        counts: Number of each building constructed.
        sequence: An optimal construction sequence.
    """

    validate_time(n)

    earnings, sequence = best_profit(n)

    counts = {
        "T": sequence.count("T"),
        "P": sequence.count("P"),
        "C": sequence.count("C"),
    }

    return earnings, counts, sequence


# --------------------------------------------------
# 5. INDEPENDENT EARNINGS CALCULATION
# --------------------------------------------------

def calculate_sequence_earnings(n, sequence):
    """
    Calculate earnings directly from a proposed building sequence.

    This independently checks whether the reported earnings
    agree with the construction order and available time.
    """

    validate_time(n)

    remaining = n
    total_earnings = 0

    for code in sequence:

        if code not in BUILDINGS:
            raise ValueError(f"Unknown building code: {code}")

        details = BUILDINGS[code]
        construction_time = details["construction_time"]
        earning_rate = details["earning_rate"]

        if construction_time > remaining:
            raise ValueError(
                f"Building {code} cannot be completed "
                f"with {remaining} time units remaining."
            )

        remaining -= construction_time

        total_earnings += remaining * earning_rate

    return total_earnings


# --------------------------------------------------
# 6. INDEPENDENT BOTTOM-UP DYNAMIC PROGRAMMING
# --------------------------------------------------

def verify_profit(n):
    """
    Calculate maximum earnings using bottom-up dynamic programming.

    This implementation does not use the recursive solver.
    """

    validate_time(n)

    dp = [0] * (n + 1)

    for time in range(1, n + 1):

        best = 0

        for details in BUILDINGS.values():

            construction_time = details["construction_time"]
            earning_rate = details["earning_rate"]

            if construction_time <= time:

                remaining = time - construction_time

                earnings = (
                    remaining * earning_rate
                    + dp[remaining]
                )

                best = max(best, earnings)

        dp[time] = best

    return dp[n]


# --------------------------------------------------
# 7. RESULT DISPLAY
# --------------------------------------------------

def display_result(n):
    """Print the solution in a clear, assignment-friendly format."""

    earnings, counts, sequence = solve(n)

    verified_earnings = verify_profit(n)

    sequence_earnings = calculate_sequence_earnings(n, sequence)

    verified = (
        earnings == verified_earnings
        and earnings == sequence_earnings
    )

    print(f"\nAvailable time: {n} units")
    print(f"Maximum earnings: ${earnings:,}")
    print(
        f"T: {counts['T']}, "
        f"P: {counts['P']}, "
        f"C: {counts['C']}"
    )

    print(
        "Construction sequence: "
        + (" -> ".join(sequence) if sequence else "None")
    )

    print(
        "Verification: "
        + ("PASSED" if verified else "FAILED")
    )


# --------------------------------------------------
# 8. PROGRAM ENTRY POINT
# --------------------------------------------------

def main():
    """Run the official examples and additional demonstrations."""

    official_tests = {
        7: 3000,
        8: 4500,
        13: 16500,
    }

    print("=" * 55)
    print("MAX PROFIT PROBLEM")
    print("=" * 55)

    all_passed = True

    print("\nOFFICIAL EXAMPLES")

    for n, expected in official_tests.items():

        earnings, counts, sequence = solve(n)

        verified_earnings = verify_profit(n)

        sequence_earnings = calculate_sequence_earnings(
            n, sequence
        )

        passed = (
            earnings == expected
            and earnings == verified_earnings
            and earnings == sequence_earnings
        )

        if not passed:
            all_passed = False

        print(f"\nTime: {n}")
        print(f"Expected earnings: ${expected:,}")
        print(f"Calculated earnings: ${earnings:,}")
        print(
            f"T: {counts['T']}, "
            f"P: {counts['P']}, "
            f"C: {counts['C']}"
        )
        print(f"Status: {'PASS' if passed else 'FAIL'}")

    print("\nADDITIONAL EXAMPLES")

    for n in [0, 4, 5, 6, 9, 10, 11, 14, 19, 20, 25, 30]:
        display_result(n)

    print("\n" + "=" * 55)
    print(
        "OVERALL RESULT: "
        + (
            "ALL OFFICIAL TESTS PASSED"
            if all_passed
            else "SOME TESTS FAILED"
        )
    )
    print("=" * 55)


if __name__ == "__main__":
    main()