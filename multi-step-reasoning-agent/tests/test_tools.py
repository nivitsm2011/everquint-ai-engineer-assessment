import pytest

from reasoner.tools import (
    add_duration,
    format_minutes,
    run_tool,
    safe_calc,
    slots_that_fit,
    time_diff,
)


def test_time_diff_same_day():
    assert time_diff("14:30", "18:05") == 215


def test_time_diff_overnight():
    assert time_diff("22:45", "01:15") == 150


def test_slots_exact_fit_counts():
    slots = [["09:00", "10:00"], ["10:00", "10:59"], ["13:00", "15:00"]]
    # 60 minutes needed: the exact-fit slot and the longer one fit, the 59 min one doesn't.
    assert slots_that_fit(slots, 60) == [["09:00", "10:00"], ["13:00", "15:00"]]


def test_safe_calc_basic():
    assert safe_calc("2 * (3 + 4) - 10 / 4") == 11.5


def test_safe_calc_rejects_code():
    with pytest.raises(ValueError):
        safe_calc("__import__('os')")


def test_add_duration_wraps_past_midnight():
    assert add_duration("23:30", 90) == "01:00"


def test_format_minutes():
    assert format_minutes(215) == "3 h 35 min"


def test_run_tool_success_and_failure():
    assert run_tool("calc", {"expression": "1+2"}) == {
        "ok": True,
        "result": 3.0,
        "error": None,
    }
    bad = run_tool("time_diff", {"start": "25:00", "end": "01:00"})
    assert bad["ok"] is False and "Hours" in bad["error"]
    assert run_tool("nope", {})["ok"] is False
