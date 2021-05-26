import pytest

from timer import Timer


def test_start_stop_get_normal():
    timer = Timer()
    timer.start_timer("test")
    timer.stop_timer("test")
    assert timer.get_duration("test") > 0


def test_start_stop_get_reset():
    timer = Timer()
    timer.start_timer("test")
    timer.stop_timer("test")
    t1 = timer.get_duration("test")
    timer.start_timer("test")
    timer.stop_timer("test")
    t2 = timer.get_duration("test")
    assert t2 > 0
    assert t2 < t1 + t2


def test_timer_not_existing():
    timer = Timer()
    t1 = timer.get_duration("test")
    assert t1 is None


def test_timer_two_timers():
    timer = Timer()
    timer.start_timer("test1")
    timer.start_timer("test2")
    timer.stop_timer("test2")
    timer.stop_timer("test1")
    t1 = timer.get_duration("test1")
    t2 = timer.get_duration("test2")
    assert t1 > 0 and t2 > 0
    assert t1 > t2
