"""Tests for the old-release reminder at the start of a new run.

The suite's conftest keeps the reminder inert; the fixture here resets it
and points its state file into the test's temporary directory. The dates,
the installed version, the environment and the interactivity are injected,
so that no test depends on the calendar, the installation or the terminal.
"""

import datetime
import importlib.metadata
import json
import os
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyvbmc import VBMC, _release
from pyvbmc.calibration import _cache
from pyvbmc.calibration.profile import default_profile
from pyvbmc.vbmc import _release_reminder, _runtime_tips
from pyvbmc.vbmc._tip_catalog import TIPS

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
# The state path of the package, taken before any fixture replaces it.
DEFAULT_STATE_PATH = _release_reminder._default_state_path

RELEASED = datetime.date(2026, 3, 15)
RELEASE_DATE = RELEASED.isoformat()
INSTALLED = "1.5.0"
ELIGIBLE_DAY = RELEASED + datetime.timedelta(days=500)
NOTE = "Note: PyVBMC"
LAST = "This is the last reminder for PyVBMC"
CALIBRATION_REMINDER = "PyVBMC is using standard performance settings."

_RELEASE_HEADING = re.compile(
    r"^## \[([0-9]+\.[0-9]+\.[0-9]+)\] - ([0-9]{4}-[0-9]{2}-[0-9]{2})\s*$",
    re.MULTILINE,
)


@pytest.fixture(autouse=True)
def reminder(_inert_release_reminder, monkeypatch, tmp_path):
    """Opt back into the reminder, with a state file of the test's own."""
    inherited = SimpleNamespace(
        shown=_release_reminder._SHOWN_THIS_SESSION,
        path=_release_reminder._default_state_path(),
    )
    state_path = tmp_path / "cache" / _release_reminder.STATE_FILE_NAME
    monkeypatch.setattr(
        _release_reminder, "_default_state_path", lambda: state_path
    )
    for name in _release_reminder.OPT_OUT_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    _release_reminder._reset_release_reminder_state()
    _runtime_tips._reset_runtime_tip_state(rng=random.Random(7))
    _cache._clear_process_state()
    yield SimpleNamespace(state_path=state_path, inherited=inherited)
    _runtime_tips._reset_runtime_tip_state(rng=random.Random(7))
    _cache._clear_process_state()


@pytest.fixture
def eligible(monkeypatch):
    """Make the defaults of the reminder describe an eligible start."""
    monkeypatch.setattr(_release, "RELEASE_DATE", RELEASE_DATE)
    monkeypatch.setattr(_release_reminder, "_today", lambda: ELIGIBLE_DAY)
    monkeypatch.setattr(
        _release_reminder, "_installed_version", lambda: INSTALLED
    )
    monkeypatch.setattr(_release_reminder, "_is_interactive", lambda: True)


def _consider(state_path, **kwargs):
    """Consider the reminder at an eligible start, amended by *kwargs*."""
    arguments = {
        "display": "iter",
        "enabled": True,
        "slot_taken": False,
        "today": ELIGIBLE_DAY,
        "installed": INSTALLED,
        "release_date": RELEASE_DATE,
        "interactive": True,
        "environ": {},
        "state_path": state_path,
    }
    arguments.update(kwargs)
    return _release_reminder.consider_release_reminder(**arguments)


def _day(days):
    return ELIGIBLE_DAY + datetime.timedelta(days=days)


def _new_session():
    _release_reminder._reset_release_reminder_state()


def _log_density(x):
    x = np.asarray(x).reshape(-1)
    return float(-0.5 * np.sum(x**2))


def _make_vbmc(*, seed=3, **options):
    opts = {"performance_calibration": "off"}
    opts.update(options)
    return VBMC(
        _log_density,
        np.zeros((1, 2)),
        np.full((1, 2), -5.0),
        np.full((1, 2), 5.0),
        np.full((1, 2), -2.0),
        np.full((1, 2), 2.0),
        options=opts,
        seed=seed,
    )


def _stop_after_startup(monkeypatch, vbmc):
    def stop():
        raise RuntimeError("startup complete")

    monkeypatch.setattr(vbmc, "_log_column_headers", stop)
    with pytest.raises(RuntimeError, match="startup complete"):
        vbmc.optimize()


def _mock_cache_miss(monkeypatch):
    profile = default_profile(
        source="default", status="cache_miss", provenance={"reason": "missing"}
    )
    monkeypatch.setattr(_cache, "resolve_cached_profile", lambda: profile)
    monkeypatch.setattr(
        _cache, "machine_identity", lambda: {"release_reminder_test": True}
    )


def test_suite_keeps_the_reminder_inert_and_out_of_the_user_cache(
    reminder, tmp_path_factory
):
    assert reminder.inherited.shown is True
    assert reminder.inherited.path.name == "update_reminder.json"
    assert reminder.inherited.path.is_relative_to(
        tmp_path_factory.getbasetemp()
    )


def test_state_file_lies_in_the_user_cache_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("PYVBMC_CACHE_DIR", str(tmp_path / "cache"))
    assert DEFAULT_STATE_PATH() == tmp_path / "cache" / "update_reminder.json"
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize(
    "days, printed",
    [(0, False), (100, False), (365, False), (366, True), (1000, True)],
)
def test_prints_only_for_a_release_older_than_the_threshold(
    reminder, capsys, days, printed
):
    today = RELEASED + datetime.timedelta(days=days)
    assert _consider(reminder.state_path, today=today) is printed
    assert (NOTE in capsys.readouterr().out) is printed
    assert reminder.state_path.exists() is printed


@pytest.mark.parametrize(
    "released, today, printed",
    [
        # A year that holds 29 February has 366 days: the anniversary
        # itself is not "more than a year" after the release.
        ("2027-03-01", datetime.date(2028, 3, 1), False),
        ("2027-03-01", datetime.date(2028, 3, 2), True),
        ("2028-02-29", datetime.date(2029, 2, 28), False),
        ("2028-02-29", datetime.date(2029, 3, 1), True),
    ],
)
def test_threshold_is_the_first_anniversary(
    reminder, capsys, released, today, printed
):
    assert (
        _consider(reminder.state_path, release_date=released, today=today)
        is printed
    )
    assert (NOTE in capsys.readouterr().out) is printed


@pytest.mark.parametrize(
    "released, today, age",
    [
        ("2026-03-15", datetime.date(2027, 3, 16), "more than a year ago"),
        ("2026-03-15", datetime.date(2028, 3, 15), "more than a year ago"),
        ("2026-03-15", datetime.date(2028, 3, 16), "more than 2 years ago"),
        ("2026-03-15", datetime.date(2028, 9, 15), "more than 2 years ago"),
        ("2026-03-15", datetime.date(2031, 3, 16), "more than 5 years ago"),
        ("2028-02-29", datetime.date(2030, 2, 28), "more than a year ago"),
        ("2028-02-29", datetime.date(2030, 3, 1), "more than 2 years ago"),
    ],
)
def test_line_names_the_version_and_the_age(
    reminder, capsys, released, today, age
):
    assert _consider(reminder.state_path, release_date=released, today=today)
    assert capsys.readouterr().out == (
        f"Note: PyVBMC 1.5.0 was released {age}. Run "
        "pyvbmc.check_for_updates() to see whether a newer version is "
        "available.\n"
        "https://pypi.org/project/pyvbmc/\n"
    )


def test_only_the_third_showing_is_the_last(reminder, capsys):
    outputs = []
    for days in (0, 90, 180):
        _new_session()
        assert _consider(reminder.state_path, today=_day(days))
        outputs.append(capsys.readouterr().out)

    assert [LAST in output for output in outputs] == [False, False, True]
    last_line = outputs[2].splitlines()[0]
    assert last_line.endswith(
        "to see whether a newer version is available. "
        "This is the last reminder for PyVBMC 1.5.0."
    )


def test_prints_once_per_session_and_again_after_a_reset(reminder, capsys):
    path = reminder.state_path
    assert _consider(path)
    assert not _consider(path, today=_day(90))
    assert not _consider(path, today=_day(90), installed="1.5.1")
    assert capsys.readouterr().out.count(NOTE) == 1

    _new_session()
    assert _consider(path, today=_day(90))


def test_cap_three_showings_per_version_at_least_ninety_days_apart(
    reminder, capsys
):
    path = reminder.state_path

    def start(days, installed=INSTALLED):
        _new_session()
        return _consider(path, today=_day(days), installed=installed)

    assert start(0)
    assert not start(1)
    assert not start(89)
    assert start(90)
    assert not start(179)
    assert start(180)
    assert not start(270)
    assert not start(2000)
    assert start(2000, installed="1.5.1")

    assert capsys.readouterr().out.count(NOTE) == 4
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "1.5.0": [
            _day(0).isoformat(),
            _day(90).isoformat(),
            _day(180).isoformat(),
        ],
        "1.5.1": [_day(2000).isoformat()],
    }
    assert [item.name for item in path.parent.iterdir()] == [path.name]


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b"not json", id="not-json"),
        pytest.param(b"\xff\xfe", id="not-utf-8"),
        pytest.param(b"[]", id="not-a-mapping"),
        pytest.param(b'{"1.5.0": "2027-07-28"}', id="not-a-list"),
        pytest.param(b'{"1.5.0": [20270728]}', id="not-a-string"),
        pytest.param(b'{"1.5.0": ["yesterday"]}', id="not-a-date"),
        pytest.param(b'{"1.5.0": ["2027-02-30"]}', id="invalid-date"),
        pytest.param(b'{"1.5.0.dev1": []}', id="not-a-release"),
        pytest.param(
            b"{}" + b" " * _release_reminder._MAX_STATE_BYTES, id="oversized"
        ),
    ],
)
def test_malformed_state_starts_afresh(reminder, capsys, content):
    path = reminder.state_path
    path.parent.mkdir(parents=True)
    path.write_bytes(content)

    assert _consider(path)
    assert json.loads(path.read_text("utf-8")) == {
        INSTALLED: [ELIGIBLE_DAY.isoformat()]
    }
    _new_session()
    assert not _consider(path, today=_day(1))

    output = capsys.readouterr().out
    assert output.count(NOTE) == 1
    assert LAST not in output


def test_unreadable_state_falls_back_to_once_per_session(reminder, capsys):
    path = reminder.state_path
    path.mkdir(parents=True)

    assert _consider(path)
    assert not _consider(path, today=_day(90))
    _new_session()
    assert _consider(path, today=_day(1))

    assert capsys.readouterr().out.count(NOTE) == 2
    assert path.is_dir()


def test_unknown_state_path_falls_back_to_once_per_session(
    reminder, eligible, monkeypatch, capsys
):
    def unavailable():
        raise RuntimeError("no cache directory")

    monkeypatch.setattr(_release_reminder, "_default_state_path", unavailable)
    consider = _release_reminder.consider_release_reminder

    assert consider(display="iter", enabled=True, slot_taken=False)
    assert not consider(display="iter", enabled=True, slot_taken=False)
    assert capsys.readouterr().out.count(NOTE) == 1


def test_unwritable_state_falls_back_to_once_per_session(
    reminder, monkeypatch, capsys
):
    def refuse(source, destination):
        raise PermissionError("read-only cache")

    monkeypatch.setattr(os, "replace", refuse)
    path = reminder.state_path

    assert _consider(path)
    assert not path.exists()
    assert list(path.parent.iterdir()) == []
    assert not _consider(path, today=_day(90))
    _new_session()
    assert _consider(path, today=_day(1))
    assert capsys.readouterr().out.count(NOTE) == 2


def test_write_goes_through_a_temporary_file_and_a_rename(
    reminder, monkeypatch
):
    real_replace = os.replace
    calls = []

    def spy(source, destination):
        source, destination = Path(source), Path(destination)
        calls.append((source, destination, source.read_text("utf-8")))
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", spy)
    path = reminder.state_path

    assert _consider(path)
    [(source, destination, content)] = calls
    assert destination == path
    assert source.parent == path.parent and source != path
    assert not source.exists()
    assert json.loads(content) == {INSTALLED: [ELIGIBLE_DAY.isoformat()]}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"installed": "1.5.0.dev3"},
        {"installed": "1.5.1.dev2+g1234abcd"},
        {"installed": "1.5.0+local"},
        {"installed": "1.5.0rc1"},
        {"installed": "1.5"},
        {"release_date": "not a date"},
        {"release_date": "2026-02-30"},
        {"release_date": "20260315"},
        {"today": RELEASED - datetime.timedelta(days=1)},
        {"today": datetime.date(1970, 1, 1)},
        {"interactive": False},
        {"environ": {"CI": "true"}},
        {"environ": {"PYVBMC_NO_UPDATE_REMINDER": "1"}},
        {"environ": {"NO_UPDATE_NOTIFIER": "1"}},
        {"display": "off"},
        {"display": False},
        {"enabled": False},
        {"slot_taken": True},
    ],
)
def test_ineligible_start_prints_writes_and_uses_up_nothing(
    reminder, capsys, kwargs
):
    path = reminder.state_path
    assert not _consider(path, **kwargs)
    assert capsys.readouterr().out == ""
    assert not path.parent.exists()

    assert _consider(path)


@pytest.mark.parametrize("value", ["", "0", "false", "False", " FALSE "])
def test_unset_opt_out_values_do_not_silence(reminder, value):
    environ = {name: value for name in _release_reminder.OPT_OUT_VARIABLES}
    assert _consider(reminder.state_path, environ=environ)


def test_defaults_read_the_release_the_metadata_and_the_environment(
    reminder, eligible, monkeypatch, capsys
):
    def consider():
        return _release_reminder.consider_release_reminder(
            display="iter", enabled=True, slot_taken=False
        )

    monkeypatch.setattr(_release, "RELEASE_DATE", None)
    assert not consider()
    monkeypatch.setattr(_release, "RELEASE_DATE", RELEASE_DATE)
    monkeypatch.setattr(_release_reminder, "_installed_version", lambda: None)
    assert not consider()
    monkeypatch.setattr(
        _release_reminder, "_installed_version", lambda: INSTALLED
    )
    for name in _release_reminder.OPT_OUT_VARIABLES:
        monkeypatch.setenv(name, "1")
        assert not consider()
        monkeypatch.delenv(name)
    monkeypatch.setattr(_release_reminder, "_is_interactive", lambda: False)
    assert not consider()
    assert capsys.readouterr().out == ""
    assert not reminder.state_path.parent.exists()

    monkeypatch.setattr(_release_reminder, "_is_interactive", lambda: True)
    assert consider()
    assert NOTE in capsys.readouterr().out
    assert json.loads(reminder.state_path.read_text(encoding="utf-8")) == {
        INSTALLED: [ELIGIBLE_DAY.isoformat()]
    }


def test_installed_version_reads_the_package_metadata(monkeypatch):
    assert _release_reminder._installed_version() == (
        importlib.metadata.version("pyvbmc")
    )

    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(_release_reminder, "version", missing)
    assert _release_reminder._installed_version() is None


class _Stream:
    def __init__(self, tty):
        self.tty = tty

    def isatty(self):
        if isinstance(self.tty, Exception):
            raise self.tty
        return self.tty

    def write(self, text):
        return len(text)

    def flush(self):
        pass


def test_interactive_session_is_a_terminal_or_an_ipython_kernel(monkeypatch):
    is_interactive = _release_reminder._is_interactive
    monkeypatch.delitem(sys.modules, "IPython", raising=False)

    monkeypatch.setattr(sys, "stdout", _Stream(True))
    assert is_interactive()
    for stream in (
        _Stream(False),
        _Stream(ValueError("closed")),
        object(),
        None,
    ):
        monkeypatch.setattr(sys, "stdout", stream)
        assert not is_interactive()
    assert "IPython" not in sys.modules

    monkeypatch.setattr(sys, "stdout", _Stream(False))
    for shell, interactive in (
        (SimpleNamespace(kernel=object()), True),
        (SimpleNamespace(kernel=None), False),
        (SimpleNamespace(), False),
        (None, False),
    ):
        monkeypatch.setitem(
            sys.modules,
            "IPython",
            SimpleNamespace(get_ipython=lambda shell=shell: shell),
        )
        assert is_interactive() is interactive


def test_reminder_takes_the_slot_and_the_tip_waits_for_the_next_start(
    reminder, eligible, monkeypatch, capsys
):
    _stop_after_startup(monkeypatch, _make_vbmc())
    output = capsys.readouterr().out
    assert output.count(NOTE) == 1
    assert output.count("https://pypi.org/project/pyvbmc/") == 1
    assert "Tip:" not in output
    assert _runtime_tips._ELIGIBLE_STARTS == 0
    assert _runtime_tips._ORDER is None

    _stop_after_startup(monkeypatch, _make_vbmc())
    output = capsys.readouterr().out
    expected = list(TIPS)
    random.Random(7).shuffle(expected)
    assert NOTE not in output
    assert f"Tip: {expected[0].text}" in output
    assert _runtime_tips._ELIGIBLE_STARTS == 1


def test_calibration_reminder_takes_the_slot_before_the_release_reminder(
    reminder, eligible, monkeypatch, capsys
):
    _mock_cache_miss(monkeypatch)

    _stop_after_startup(
        monkeypatch, _make_vbmc(performance_calibration="cached")
    )
    output = capsys.readouterr().out
    assert CALIBRATION_REMINDER in output
    assert NOTE not in output
    assert "Tip:" not in output
    assert not reminder.state_path.parent.exists()

    _stop_after_startup(
        monkeypatch, _make_vbmc(performance_calibration="cached")
    )
    output = capsys.readouterr().out
    assert CALIBRATION_REMINDER not in output
    assert output.count(NOTE) == 1
    assert "Tip:" not in output


@pytest.mark.parametrize("options", [{"display": "off"}, {"show_tips": False}])
def test_run_options_silence_the_reminder_without_using_it_up(
    reminder, eligible, monkeypatch, capsys, options
):
    _stop_after_startup(monkeypatch, _make_vbmc(**options))
    assert NOTE not in capsys.readouterr().out
    assert not reminder.state_path.parent.exists()

    _stop_after_startup(monkeypatch, _make_vbmc())
    assert NOTE in capsys.readouterr().out


def test_resumed_and_continued_runs_print_neither(
    reminder, eligible, monkeypatch, capsys
):
    started = _make_vbmc()
    _stop_after_startup(monkeypatch, started)
    assert NOTE in capsys.readouterr().out

    # A later session, in which a new run would show the reminder again.
    monkeypatch.setattr(_release_reminder, "_today", lambda: _day(200))
    _new_session()
    _runtime_tips._reset_runtime_tip_state(rng=random.Random(7))
    _stop_after_startup(monkeypatch, started)
    resumed = _make_vbmc()
    # What VBMC.load restores for a run that has started.
    resumed._runtime_tip_handled = True
    _stop_after_startup(monkeypatch, resumed)
    output = capsys.readouterr().out
    assert NOTE not in output
    assert "Tip:" not in output

    _stop_after_startup(monkeypatch, _make_vbmc())
    assert NOTE in capsys.readouterr().out


def test_reminder_uses_stdout_and_stays_out_of_the_log_file(
    reminder, eligible, monkeypatch, capsys, tmp_path
):
    log_path = tmp_path / "vbmc.log"
    _stop_after_startup(monkeypatch, _make_vbmc(log_file_name=str(log_path)))

    assert NOTE in capsys.readouterr().out
    log = log_path.read_text(encoding="utf-8")
    assert NOTE not in log
    assert "pypi.org" not in log


def test_reminder_leaves_every_random_stream_alone(
    reminder, eligible, monkeypatch, capsys
):
    printed = _make_vbmc()
    quiet = _make_vbmc()
    numpy_state = np.random.get_state()
    stdlib_state = random.getstate()
    tip_state = _runtime_tips._RNG.getstate()

    _stop_after_startup(monkeypatch, printed)
    assert NOTE in capsys.readouterr().out
    assert _runtime_tips._RNG.getstate() == tip_state
    assert random.getstate() == stdlib_state
    after = np.random.get_state()
    assert numpy_state[0] == after[0]
    assert np.array_equal(numpy_state[1], after[1])
    assert numpy_state[2:] == after[2:]

    # The twin run starts with the session's showing used up.
    _stop_after_startup(monkeypatch, quiet)
    assert NOTE not in capsys.readouterr().out
    assert printed.rng.bit_generator.state == quiet.rng.bit_generator.state


def test_seeded_run_gives_the_same_results_with_and_without_the_reminder(
    reminder, eligible, capsys
):
    # Two runs capped at two iterations at D = 2, as the short runs of
    # test_vbmc_seed.py; the first shows the reminder, the second does not.
    vbmc_1 = _make_vbmc(seed=42, max_iter=2, do_final_boost=False)
    vp_1, results_1 = vbmc_1.optimize()
    output_1 = capsys.readouterr().out
    vbmc_2 = _make_vbmc(seed=42, max_iter=2, do_final_boost=False)
    vp_2, results_2 = vbmc_2.optimize()
    output_2 = capsys.readouterr().out

    assert output_1.count(NOTE) == 1
    assert NOTE not in output_2
    assert results_1["elbo"] == results_2["elbo"]
    assert results_1["elbo_sd"] == results_2["elbo_sd"]
    for name in ("w", "mu", "sigma", "lambd"):
        assert np.array_equal(getattr(vp_1, name), getattr(vp_2, name))
    logger_1, logger_2 = vbmc_1.function_logger, vbmc_2.function_logger
    assert np.array_equal(
        logger_1.X_orig[logger_1.X_flag], logger_2.X_orig[logger_2.X_flag]
    )
    assert np.array_equal(
        logger_1.y_orig[logger_1.X_flag], logger_2.y_orig[logger_2.X_flag]
    )
    assert vbmc_1.rng.bit_generator.state == vbmc_2.rng.bit_generator.state


def test_release_date_matches_the_changelog():
    changelog = REPOSITORY_ROOT / "CHANGELOG.md"
    if not changelog.is_file():
        pytest.skip("CHANGELOG.md is not beside the package")
    text = changelog.read_text(encoding="utf-8")
    if "changes to PyVBMC" not in text:
        pytest.skip("the CHANGELOG.md beside the package is not PyVBMC's")
    # The first section heading other than [Unreleased] is the latest
    # release, and it must be written in the form the date is read from.
    released = [
        line
        for line in text.splitlines()
        if line.startswith("## [") and not line.startswith("## [Unreleased]")
    ]

    if not released:
        assert _release.RELEASE_DATE is None
    else:
        heading = _RELEASE_HEADING.fullmatch(released[0].rstrip())
        assert heading is not None, released[0]
        assert _release.RELEASE_DATE == heading.group(2)
        assert _release_reminder._parse_date(_release.RELEASE_DATE)


class _BrokenStream:
    def write(self, text):
        raise BrokenPipeError(32, "Broken pipe")

    def flush(self):
        raise BrokenPipeError(32, "Broken pipe")


def test_failing_output_stops_neither_the_reminder_nor_the_tip(
    reminder, monkeypatch
):
    with monkeypatch.context() as patch:
        patch.setattr(sys, "stdout", _BrokenStream())
        assert not _consider(reminder.state_path)
        # The slot passes to the tip, which prints through the same stream.
        assert (
            _runtime_tips.consider_runtime_tip(
                display="iter",
                enabled=True,
                calibration_reminder_emitted=False,
                release_reminder_emitted=False,
            )
            is None
        )
    # The tip reached the emitter, which failed: nothing is marked seen.
    assert _runtime_tips._ELIGIBLE_STARTS == 1
    assert _runtime_tips._SEEN_IDS == set()
    assert not reminder.state_path.exists()
    # Nothing was used up: the next start prints.
    assert _consider(reminder.state_path)
