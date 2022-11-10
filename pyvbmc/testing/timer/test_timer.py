from time import sleep

from pyvbmc.timer import Timer


def test_start_stop_get_normal():
    timer = Timer()
    timer.start_timer("test")
    sleep(0.2)
    timer.stop_timer("test")
    assert timer.get_duration("test") >= 0.2


def test_start_stop_get_reset():
    timer = Timer()
    timer.start_timer("test")
    timer.stop_timer("test")
    timer1 = timer.get_duration("test")
    timer.start_timer("test")
    sleep(0.2)
    timer.stop_timer("test")
    timer2 = timer.get_duration("test")
    assert timer2 >= 0.2
    assert timer2 <= timer1 + timer2


def test_timer_not_existing():
    timer = Timer()
    timer1 = timer.get_duration("test")
    assert timer1 is None


def test_timer_two_timers():
    timer = Timer()
    timer.start_timer("testimer1")
    sleep(0.2)
    timer.start_timer("testimer2")
    timer.stop_timer("testimer2")
    timer.stop_timer("testimer1")
    timer1 = timer.get_duration("testimer1")
    timer2 = timer.get_duration("testimer2")
    assert timer1 >= 0.2 and timer2 >= 0
    assert timer1 >= timer2 + 0.2


def test_timer_cumulative():
    timer = Timer()
    timer.start_timer("a")
    sleep(0.1)
    timer.stop_timer("a")
    timer.start_timer("a")
    sleep(0.1)
    timer.stop_timer("a")
    a = timer.get_duration("a")
    assert a > 0.19


def test__str__and__repr__():
    timer = Timer()
    timer.start_timer("foo")
    sleep(0.1)
    timer.stop_timer("foo")
    assert "'foo'" in timer.__str__()
    assert "'foo'" in timer.__repr__()
