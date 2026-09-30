"""Fixtures shared by the whole shipped test suite."""

import pytest

from pyvbmc.vbmc import _release_reminder


@pytest.fixture(scope="session")
def _release_reminder_directory(tmp_path_factory):
    """A directory for the old-release reminder's state file."""
    return tmp_path_factory.mktemp("release_reminder")


@pytest.fixture(autouse=True)
def _inert_release_reminder(monkeypatch, _release_reminder_directory):
    """Keep the old-release reminder silent and out of the user's cache.

    The reminder counts as shown in this session, so that no test's output
    depends on the calendar: a year after a release, the reminder would
    otherwise print at the first run started in an interactive session and
    change the output of the test that started it. Its state file lies in
    pytest's temporary directory. ``test_release_reminder.py`` resets the
    flag to test the reminder.
    """
    monkeypatch.setattr(
        _release_reminder,
        "_default_state_path",
        lambda: _release_reminder_directory
        / _release_reminder.STATE_FILE_NAME,
    )
    monkeypatch.setattr(_release_reminder, "_SHOWN_THIS_SESSION", True)
