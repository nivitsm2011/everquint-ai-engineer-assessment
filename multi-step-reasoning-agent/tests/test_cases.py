"""The 13 end-to-end test questions (8 easy, 5 tricky)."""


def _case(id: int, category: str, question: str, expected: str, status: str = "success") -> dict:
    return {
        "id": id,
        "category": category,
        "question": question,
        "expected_canonical": expected,
        "expected_status": status,
    }


TEST_CASES = [
    # ---- easy ----
    _case(1, "easy", "If a train leaves at 14:30 and arrives at 18:05, how long is the journey?", "215"),
    _case(2, "easy", "Alice has 3 red apples and twice as many green apples as red. How many apples does she have in total?", "9"),
    _case(3, "easy", "A meeting needs 60 minutes. There are free slots: 09:00-09:30, 09:45-10:30, 11:00-12:00. Which slots can fit the meeting?", "11:00-12:00"),
    _case(4, "easy", "What is 15% of 240?", "36"),
    _case(5, "easy", "A book has 12 chapters with 25 pages each. How many pages in total?", "300"),
    _case(6, "easy", "A bus leaves at 08:15 and the trip takes 2 hours 50 minutes. What time does it arrive?", "11:05"),
    _case(7, "easy", "Tom had 50 rupees, spent 18 on lunch and 7 on tea. How much is left?", "25"),
    _case(8, "easy", "What is the average of 12, 18 and 30?", "20"),
    # ---- tricky ----
    # overnight edge case
    _case(9, "tricky", "A train leaves at 22:45 and arrives at 01:15 the next day. How long is the journey?", "150"),
    # exact-fit boundary
    _case(10, "tricky", "A meeting needs 30 minutes. Free slots: 09:00-09:30, 10:00-10:20, 13:15-13:45. Which slots fit?", "09:00-09:30,13:15-13:45"),
    # classic naive-reasoning trap
    _case(11, "tricky", "A bat and a ball cost 1.10 in total. The bat costs 1.00 more than the ball. How much does the ball cost?", "0.05"),
    # hidden unit (dozen)
    _case(12, "tricky", "A recipe needs 3 eggs per cake. How many eggs are needed for 2 dozen cakes?", "72"),
    # impossible constraint
    _case(13, "tricky", "Alice has 5 apples and gives away 8. How many apples does she have left?", "UNSOLVABLE", "failed"),
]
