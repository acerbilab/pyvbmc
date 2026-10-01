"""Fixtures shared by the whole shipped test suite."""

import pytest

from pyvbmc.vbmc import _release_reminder


@pytest.fixture(scope="session")
def _release_reminder_directory(tmp_path_factory):
    """A directory for the old-release reminder's state file."""
    return tmp_path_factory.mktemp("release_reminder")


def _make_inert(patch, directory):
    patch.setattr(
        _release_reminder,
        "_default_state_path",
        lambda: directory / _release_reminder.STATE_FILE_NAME,
    )
    patch.setattr(_release_reminder, "_SHOWN_THIS_SESSION", True)


@pytest.fixture(scope="session", autouse=True)
def _inert_release_reminder_for_the_session(_release_reminder_directory):
    """Keep the reminder inert in fixtures shared by a module or session.

    Such a fixture is set up outside every test, so the fixture below, which
    acts per test, does not cover a run it starts.
    """
    with pytest.MonkeyPatch.context() as patch:
        _make_inert(patch, _release_reminder_directory)
        yield


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
    _make_inert(monkeypatch, _release_reminder_directory)
