import unittest

from max_profit import (
    solve,
    verify_profit,
    calculate_sequence_earnings,
)


class TestMaxProfit(unittest.TestCase):

    # ---------------------------------------------
    # 1. Official examples
    # ---------------------------------------------

    def test_time_7(self):
        earnings, counts, sequence = solve(7)

        self.assertEqual(earnings, 3000)
        self.assertEqual(counts["T"], 1)
        self.assertEqual(counts["P"], 0)
        self.assertEqual(counts["C"], 0)

    def test_time_8(self):
        earnings, counts, sequence = solve(8)

        self.assertEqual(earnings, 4500)
        self.assertEqual(counts["T"], 1)
        self.assertEqual(counts["P"], 0)
        self.assertEqual(counts["C"], 0)

    def test_time_13(self):
        earnings, counts, sequence = solve(13)

        self.assertEqual(earnings, 16500)
        self.assertEqual(counts["T"], 2)
        self.assertEqual(counts["P"], 0)
        self.assertEqual(counts["C"], 0)

    # ---------------------------------------------
    # 2. Edge cases
    # ---------------------------------------------

    def test_zero_time(self):
        earnings, counts, sequence = solve(0)

        self.assertEqual(earnings, 0)
        self.assertEqual(
            counts,
            {"T": 0, "P": 0, "C": 0},
        )
        self.assertEqual(sequence, ())

    def test_time_four(self):
        earnings, counts, sequence = solve(4)

        self.assertEqual(earnings, 0)
        self.assertEqual(counts["T"], 0)
        self.assertEqual(counts["P"], 0)
        self.assertEqual(counts["C"], 0)

    def test_time_ten(self):
        earnings, counts, sequence = solve(10)

        # Theatre earns $7,500.
        # Pub constructed afterwards earns $1,000.
        # Total earnings = $8,500.
        self.assertEqual(earnings, 8500)
        self.assertEqual(counts["T"], 1)
        self.assertEqual(counts["P"], 1)
        self.assertEqual(counts["C"], 0)

    # ---------------------------------------------
    # 3. Independent verification
    # ---------------------------------------------

    def test_recursive_and_bottom_up_agree(self):
        for n in range(101):
            with self.subTest(time=n):
                earnings, _, _ = solve(n)
                verified_earnings = verify_profit(n)

                self.assertEqual(
                    earnings,
                    verified_earnings,
                )

    def test_sequence_earnings_match_solution(self):
        for n in range(101):
            with self.subTest(time=n):
                earnings, _, sequence = solve(n)

                calculated_earnings = calculate_sequence_earnings(n, sequence)

                self.assertEqual(
                    earnings,
                    calculated_earnings,
                )

    # ---------------------------------------------
    # 4. Invalid input tests
    # ---------------------------------------------

    def test_negative_time(self):
        with self.assertRaises(ValueError):
            solve(-1)

    def test_float_input(self):
        with self.assertRaises(ValueError):
            solve(2.5)

    def test_string_input(self):
        with self.assertRaises(ValueError):
            solve("10")

    def test_boolean_input(self):
        with self.assertRaises(ValueError):
            solve(True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
