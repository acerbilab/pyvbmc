"""The old-release reminder at the start of a new run.

PyVBMC ships the date of its release (``pyvbmc._release.RELEASE_DATE``).
When a new run starts in an interactive session and the installed release is
more than a year old, one line says that a newer version may exist and how
to find out. The reminder opens no network connection and writes no file but
its state file.

The state file, ``update_reminder.json`` in PyVBMC's user cache directory,
maps each installed version to the dates of its showings, which caps the
reminder at three showings per version, at least 90 days apart. A file
whose content is malformed is started afresh. A process-local flag allows one
showing per Python session, and it is the only cap left when the file cannot
be read or written.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import sys
import tempfile
import threading
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from pyvbmc import _release
from pyvbmc._user_hints import emit_user_hint

REMINDER_TEMPLATE = (
    "Note: PyVBMC {version} was released {age}. Run "
    "pyvbmc.check_for_updates() to see whether a newer version is available."
)
LAST_REMINDER_TEMPLATE = "This is the last reminder for PyVBMC {version}."
AGE_ONE_YEAR = "more than a year ago"
AGE_YEARS_TEMPLATE = "more than {years} years ago"
PYPI_URL = "https://pypi.org/project/pyvbmc/"

THRESHOLD_DAYS = 365
SPACING_DAYS = 90
MAX_SHOWINGS = 3
STATE_FILE_NAME = "update_reminder.json"
OPT_OUT_VARIABLES = ("CI", "PYVBMC_NO_UPDATE_REMINDER", "NO_UPDATE_NOTIFIER")
# Values of an opt-out variable that leave it unset, as CI detection reads
# them.
UNSET_VALUES = frozenset({"", "0", "false"})

_RELEASE_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")
_ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_MAX_STATE_BYTES = 64 * 1024

_STATE_LOCK = threading.Lock()
_SHOWN_THIS_SESSION = False


def _today() -> datetime.date:
    """Return the date of the run from the clock."""
    return datetime.date.today()


def _installed_version() -> str | None:
    """Return the installed version of PyVBMC, or ``None`` if unknown."""
    try:
        return version("pyvbmc")
    except PackageNotFoundError:
        return None


def _is_interactive() -> bool:
    """Return whether output reaches a terminal or an IPython kernel.

    IPython is consulted only when the session has imported it already.
    """
    stdout = sys.stdout
    try:
        if stdout is not None and stdout.isatty():
            return True
    except (AttributeError, OSError, ValueError):
        pass
    get_ipython = getattr(sys.modules.get("IPython"), "get_ipython", None)
    if get_ipython is None:
        return False
    try:
        shell = get_ipython()
    except Exception:
        return False
    return getattr(shell, "kernel", None) is not None


def _default_state_path() -> Path:
    """Return the state file in PyVBMC's user cache directory."""
    from pyvbmc.calibration._cache import cache_root

    return cache_root() / STATE_FILE_NAME


def _parse_date(value: object) -> datetime.date | None:
    """Return the date of an ISO ``YYYY-MM-DD`` string, or ``None``."""
    if not isinstance(value, str) or not _ISO_DATE.fullmatch(value):
        return None
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        return None


def _age_phrase(released: datetime.date, today: datetime.date) -> str:
    """Return how long ago the release was, in whole years exceeded."""
    years = today.year - released.year
    if (today.month, today.day) <= (released.month, released.day):
        years -= 1
    if years >= 2:
        return AGE_YEARS_TEMPLATE.format(years=years)
    return AGE_ONE_YEAR


def _opted_out(environ: Mapping[str, str]) -> bool:
    """Return whether an opt-out variable is set to a value that counts."""
    return any(
        environ.get(name, "").strip().lower() not in UNSET_VALUES
        for name in OPT_OUT_VARIABLES
    )


def _read_state(path: Path | None) -> dict[str, list[str]] | None:
    """Return the recorded showings, or ``None`` when the file is unusable.

    A missing file holds no showings, and so does a file whose content is
    anything but a mapping from release versions to lists of ISO dates: the
    next write replaces it. A file that cannot be read is unusable.
    """
    if path is None:
        return None
    try:
        with open(path, "rb") as stream:
            payload = stream.read(_MAX_STATE_BYTES + 1)
    except FileNotFoundError:
        return {}
    except Exception:
        return None
    if len(payload) > _MAX_STATE_BYTES:
        return {}
    try:
        state = json.loads(payload.decode("utf-8"))
    except Exception:
        return {}
    if not isinstance(state, dict):
        return {}
    for key, dates in state.items():
        if (
            not _RELEASE_VERSION.fullmatch(key)
            or not isinstance(dates, list)
            or any(_parse_date(date) is None for date in dates)
        ):
            return {}
    return state


def _write_state(path: Path, state: Mapping[str, list[str]]) -> bool:
    """Replace the state file through a temporary file and a rename.

    Every error is swallowed; the return value says whether the file was
    replaced.
    """
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".update_reminder.", suffix=".tmp", dir=path.parent
        )
        with os.fdopen(
            descriptor, "w", encoding="utf-8", newline="\n"
        ) as stream:
            json.dump(state, stream, sort_keys=True)
        os.replace(temporary, path)
        temporary = None
        return True
    except Exception:
        return False
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def consider_release_reminder(
    *,
    display: bool | str,
    enabled: bool,
    slot_taken: bool,
    today: datetime.date | None = None,
    installed: str | None = None,
    release_date: str | None = None,
    interactive: bool | None = None,
    environ: Mapping[str, str] | None = None,
    state_path: str | os.PathLike[str] | None = None,
) -> bool:
    """Consider and possibly print the old-release reminder.

    The reminder prints when the installed version is a final release
    ``X.Y.Z`` released more than a year before the date of the run, in an
    interactive session without an opt-out environment variable, and when
    neither the session nor the state file's cap has used it up. A start
    at which it does not print changes nothing.

    Parameters
    ----------
    display : bool or str
        The run's ``display`` option: nothing prints when it is false or
        ``"off"``.
    enabled : bool
        The run's ``show_tips`` option, which silences the reminder too.
    slot_taken : bool
        Whether another hint, the calibration reminder, took the
        start-of-run slot.
    today : datetime.date, optional
        The date of the run. The default reads the clock.
    installed : str, optional
        The installed version. The default reads the package metadata.
    release_date : str, optional
        The ISO date of the installed release. The default is
        ``pyvbmc._release.RELEASE_DATE``.
    interactive : bool, optional
        Whether the session is interactive. The default checks whether
        standard output is a terminal or the code runs in an IPython kernel.
    environ : Mapping[str, str], optional
        The environment holding the opt-out variables. The default is
        ``os.environ``.
    state_path : str or os.PathLike, optional
        The state file that records the showings. The default is
        ``update_reminder.json`` in PyVBMC's user cache directory.

    Returns
    -------
    bool
        Whether the reminder printed.
    """
    global _SHOWN_THIS_SESSION

    if not display or display == "off" or not enabled or slot_taken:
        return False

    with _STATE_LOCK:
        if _SHOWN_THIS_SESSION:
            return False
        if environ is None:
            environ = os.environ
        if _opted_out(environ):
            return False
        if installed is None:
            installed = _installed_version()
        if not isinstance(installed, str) or not _RELEASE_VERSION.fullmatch(
            installed
        ):
            return False
        if release_date is None:
            release_date = _release.RELEASE_DATE
        released = _parse_date(release_date)
        if released is None:
            return False
        if today is None:
            today = _today()
        if today < released or (today - released).days <= THRESHOLD_DAYS:
            return False
        if interactive is None:
            interactive = _is_interactive()
        if not interactive:
            return False

        if state_path is None:
            try:
                path = _default_state_path()
            except Exception:
                path = None
        else:
            path = Path(state_path)
        state = _read_state(path)
        dates = [] if state is None else state.get(installed, [])
        if len(dates) >= MAX_SHOWINGS:
            return False
        if dates:
            last = max(_parse_date(date) for date in dates)
            if (today - last).days < SPACING_DAYS:
                return False

        message = REMINDER_TEMPLATE.format(
            version=installed, age=_age_phrase(released, today)
        )
        if len(dates) + 1 == MAX_SHOWINGS:
            message += " " + LAST_REMINDER_TEMPLATE.format(version=installed)
        if not emit_user_hint(message, display=display, urls=(PYPI_URL,)):
            return False
        _SHOWN_THIS_SESSION = True
        if state is not None:
            state[installed] = [*dates, today.isoformat()]
            _write_state(path, state)
        return True


def _reset_release_reminder_state() -> None:
    """Reset the process-local flag for isolated tests."""
    global _SHOWN_THIS_SESSION
    with _STATE_LOCK:
        _SHOWN_THIS_SESSION = False
