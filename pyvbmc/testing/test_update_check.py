"""Tests of ``pyvbmc.check_for_updates``.

``urllib.request.urlopen`` is replaced in every test, so no test opens a
connection, and the installed version and installer are injected through
the metadata lookups of ``pyvbmc._update_check``.
"""

import http.client
import json
import socket
import subprocess
import sys
import textwrap
import urllib.error
import urllib.request
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import numpy as np
import pytest

import pyvbmc
from pyvbmc import _update_check
from pyvbmc._update_check import UpdateCheck, check_for_updates

URL = "https://pypi.org/pypi/pyvbmc/json"
PIP = "python -m pip install --upgrade pyvbmc"
CONDA = (
    "conda update --channel=conda-forge pyvbmc "
    "(the conda-forge package can follow PyPI by a few days)"
)
BOTH = (
    "python -m pip install --upgrade pyvbmc, or with conda: "
    "conda update --channel=conda-forge pyvbmc "
    "(the conda-forge package can follow PyPI by a few days)"
)


def _failure(reason):
    return (
        f"Could not reach PyPI ({reason}); "
        "see https://pypi.org/project/pyvbmc/.\n"
    )


def _file(yanked=False):
    return {"filename": "pyvbmc-x-py3-none-any.whl", "yanked": yanked}


def _releases(*versions):
    return {"info": {}, "releases": {v: [_file()] for v in versions}}


class _Response:
    def __init__(self, body, read_error=None, amounts=None):
        self.body = body
        self.read_error = read_error
        self.amounts = [] if amounts is None else amounts

    def read(self, amount=-1):
        self.amounts.append(amount)
        if self.read_error is not None:
            raise self.read_error
        return self.body if amount < 0 else self.body[:amount]

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class _FakePyPI:
    """Stand-in for ``urlopen``: replies with ``reply`` or raises ``error``."""

    def __init__(self):
        self.reply = _releases("1.0.4", "1.5.0")
        self.error = None
        self.read_error = None
        self.calls = []
        self.amounts = []

    def urlopen(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.error is not None:
            raise self.error
        body = self.reply
        if not isinstance(body, bytes):
            body = json.dumps(body).encode("utf-8")
        return _Response(body, self.read_error, self.amounts)


class _Distribution:
    def __init__(self, installer):
        self.installer = installer

    def read_text(self, filename):
        assert filename == "INSTALLER"
        return self.installer


@pytest.fixture(autouse=True)
def pypi(mocker):
    fake = _FakePyPI()
    mocker.patch("urllib.request.urlopen", side_effect=fake.urlopen)
    return fake


@pytest.fixture
def install(monkeypatch):
    """Inject the installed version (None: no metadata) and installer."""

    def _install(installed, installer="pip\n"):
        def fake_version(name):
            assert name == "pyvbmc"
            if installed is None:
                raise PackageNotFoundError(name)
            return installed

        def fake_distribution(name):
            assert name == "pyvbmc"
            if installer is PackageNotFoundError:
                raise PackageNotFoundError(name)
            return _Distribution(installer)

        monkeypatch.setattr(_update_check, "version", fake_version)
        monkeypatch.setattr(_update_check, "distribution", fake_distribution)

    return _install


def test_exported_from_the_package():
    assert pyvbmc.check_for_updates is check_for_updates
    assert UpdateCheck._fields == ("installed", "latest", "update_available")


def test_newer_release_available(install, pypi, capsys):
    install("1.0.4")
    pypi.reply = _releases("1.0.3", "1.0.4", "1.5.0")
    result = check_for_updates()
    assert result == UpdateCheck("1.0.4", "1.5.0", True)
    assert capsys.readouterr().out == (
        "PyVBMC 1.5.0 is available; you have 1.0.4. "
        "Update with: python -m pip install --upgrade pyvbmc\n"
    )


def test_latest_release_installed(install, pypi, capsys):
    install("1.5.0")
    pypi.reply = _releases("1.0.4", "1.5.0")
    result = check_for_updates()
    assert result == UpdateCheck("1.5.0", "1.5.0", False)
    assert capsys.readouterr().out == "PyVBMC 1.5.0 is the latest release.\n"


def test_installed_newer_than_pypi(install, pypi, capsys):
    install("1.6.0")
    pypi.reply = _releases("1.0.4", "1.5.0")
    result = check_for_updates()
    assert result == UpdateCheck("1.6.0", "1.5.0", False)
    assert capsys.readouterr().out == (
        "PyVBMC 1.6.0 is newer than the latest release on PyPI, 1.5.0.\n"
    )


@pytest.mark.parametrize(
    "installed",
    [
        "1.0.5.dev1001+g7bf916e5c",
        "1.5.0+g1234abc",
        "1.5.1.dev3",
        "1.5.1rc1",
        "1.5.0.post1",
        "1.5",
    ],
)
def test_development_install(install, pypi, capsys, installed):
    install(installed)
    pypi.reply = _releases("1.0.4", "1.5.0")
    result = check_for_updates()
    assert result == UpdateCheck(installed, "1.5.0", None)
    assert capsys.readouterr().out == (
        f"PyVBMC {installed} is a development version; "
        "the latest release is 1.5.0.\n"
    )


@pytest.mark.parametrize(
    "text, parsed",
    [
        ("1.5.0", (1, 5, 0)),
        ("1.10.0", (1, 10, 0)),
        ("1.5.0\n", None),
        (" 1.5.0", None),
        ("1.\N{ARABIC-INDIC DIGIT FIVE}.0", None),
        ("1.5.0.0", None),
        (None, None),
    ],
)
def test_only_x_y_z_is_a_release(text, parsed):
    assert _update_check._parse_release(text) == parsed


def test_installed_version_unknown(install, pypi, capsys):
    install(None)
    pypi.reply = _releases("1.0.4", "1.5.0")
    result = check_for_updates()
    assert result == UpdateCheck(None, "1.5.0", None)
    assert capsys.readouterr().out == (
        "PyVBMC's installed version is unknown; "
        "the latest release is 1.5.0.\n"
    )
    (request,), _ = pypi.calls[0]
    assert request.get_header("User-agent") == (
        "pyvbmc/unknown (check_for_updates)"
    )


def test_versions_compare_as_integers(install, pypi, capsys):
    install("1.9.0")
    pypi.reply = _releases("1.9.0", "1.10.0", "1.2.0")
    assert check_for_updates() == UpdateCheck("1.9.0", "1.10.0", True)
    assert "PyVBMC 1.10.0 is available" in capsys.readouterr().out


def test_releases_that_do_not_count_are_ignored(install, pypi, capsys):
    install("1.4.0")
    pypi.reply = {
        "releases": {
            "1.4.0": [_file()],
            # Partly yanked: one installable file, so it counts.
            "1.5.0": [_file(yanked=True), _file()],
            # Fully yanked, and without files: nothing to install.
            "1.6.0": [_file(yanked=True), _file(yanked=True)],
            "1.7.0": [],
            # Not of the form X.Y.Z.
            "2.0.0rc1": [_file()],
            "2.0.0a1": [_file()],
            "2.0.0b2": [_file()],
            "2.0.0.dev3": [_file()],
            "2.0.0.post1": [_file()],
            "2.0.0+local": [_file()],
            "2.0": [_file()],
        }
    }
    result = check_for_updates()
    assert result == UpdateCheck("1.4.0", "1.5.0", True)
    assert capsys.readouterr().out == (
        "PyVBMC 1.5.0 is available; you have 1.4.0. "
        "Update with: python -m pip install --upgrade pyvbmc\n"
    )


def test_malformed_release_entries_are_skipped(install, pypi, capsys):
    install("1.0.4")
    pypi.reply = {
        "releases": {
            # Not a list of files, and a file that is not a mapping.
            "1.6.0": "pyvbmc-1.6.0.tar.gz",
            "1.5.0": ["pyvbmc-1.5.0.tar.gz"],
            # A file without the yanked key is not yanked.
            "1.4.0": [{"filename": "pyvbmc-1.4.0.tar.gz"}],
        }
    }
    assert check_for_updates() == UpdateCheck("1.0.4", "1.4.0", True)
    assert "PyVBMC 1.4.0 is available" in capsys.readouterr().out


def test_reply_without_releases_gives_the_info_version(install, pypi, capsys):
    install("1.0.4")
    pypi.reply = {"info": {"version": "1.5.0"}}
    assert check_for_updates() == UpdateCheck("1.0.4", "1.5.0", True)
    assert "PyVBMC 1.5.0 is available" in capsys.readouterr().out


@pytest.mark.skipif(
    getattr(sys, "get_int_max_str_digits", lambda: 0)() == 0,
    reason="int() converts strings of any length on this interpreter",
)
def test_version_too_long_for_int_is_skipped(install, pypi, capsys):
    install("1.0.4")
    digits = sys.get_int_max_str_digits() + 1
    pypi.reply = {
        "releases": {"9" * digits + ".0.0": [_file()], "1.4.0": [_file()]}
    }
    assert check_for_updates() == UpdateCheck("1.0.4", "1.4.0", True)
    assert "PyVBMC 1.4.0 is available" in capsys.readouterr().out


def test_reply_is_read_up_to_its_bound(install, pypi):
    install("1.0.4")
    check_for_updates()
    assert pypi.amounts == [_update_check.MAX_REPLY_BYTES + 1]


def test_oversized_reply_is_unreadable(install, pypi, capsys, monkeypatch):
    install("1.0.4")
    # A valid reply of more bytes than the bound.
    pypi.reply = _releases("1.0.4", "1.5.0")
    body = json.dumps(pypi.reply).encode("utf-8")
    monkeypatch.setattr(_update_check, "MAX_REPLY_BYTES", len(body) - 1)
    assert check_for_updates() == UpdateCheck("1.0.4", None, None)
    assert capsys.readouterr().out == _failure("unreadable reply")


def test_fully_yanked_newest_release_is_not_offered(install, pypi, capsys):
    install("1.5.0")
    pypi.reply = {
        "releases": {
            "1.5.0": [_file()],
            "1.5.1": [_file(yanked=True)],
        }
    }
    assert check_for_updates() == UpdateCheck("1.5.0", "1.5.0", False)
    assert capsys.readouterr().out == "PyVBMC 1.5.0 is the latest release.\n"


@pytest.mark.parametrize(
    "error, read_error, reply, reason",
    [
        pytest.param(
            urllib.error.URLError(
                socket.gaierror(11001, "getaddrinfo failed")
            ),
            None,
            None,
            "getaddrinfo failed",
            id="URLError-name-resolution",
        ),
        pytest.param(
            urllib.error.URLError(
                ConnectionRefusedError(111, "Connection refused")
            ),
            None,
            None,
            "Connection refused",
            id="URLError-refused",
        ),
        pytest.param(
            urllib.error.URLError("Tunnel connection failed: 403 Forbidden"),
            None,
            None,
            "Tunnel connection failed: 403 Forbidden",
            id="URLError-text",
        ),
        pytest.param(
            urllib.error.HTTPError(URL, 503, "Service Unavailable", {}, None),
            None,
            None,
            "HTTP 503",
            id="HTTPError-503",
        ),
        pytest.param(
            urllib.error.HTTPError(URL, 404, "Not Found", {}, None),
            None,
            None,
            "HTTP 404",
            id="HTTPError-404",
        ),
        pytest.param(
            urllib.error.URLError(TimeoutError("timed out")),
            None,
            None,
            "timed out",
            id="timeout-connecting",
        ),
        pytest.param(
            socket.timeout("timed out"),
            None,
            None,
            "timed out",
            id="timeout-socket",
        ),
        pytest.param(
            None,
            TimeoutError("The read operation timed out"),
            None,
            "timed out",
            id="timeout-reading",
        ),
        pytest.param(
            None,
            ConnectionResetError(104, "Connection reset by peer"),
            None,
            "Connection reset by peer",
            id="reset-reading",
        ),
        pytest.param(
            None,
            http.client.IncompleteRead(b"{", 100),
            None,
            "unreadable reply",
            id="incomplete-read",
        ),
        pytest.param(
            None,
            None,
            b"<html>Service temporarily unavailable</html>",
            "unreadable reply",
            id="malformed-JSON",
        ),
        pytest.param(
            None, None, b"\xff\xfe\xfa", "unreadable reply", id="not-text"
        ),
        pytest.param(
            None,
            None,
            {"info": {"version": "2.0.0rc1"}},
            "no release found",
            id="no-releases-and-no-final-info-version",
        ),
        pytest.param(
            None, None, {"info": {}}, "unreadable reply", id="no-version"
        ),
        pytest.param(
            None,
            None,
            {"releases": ["1.5.0"]},
            "unreadable reply",
            id="releases-not-a-mapping",
        ),
        pytest.param(None, None, [], "unreadable reply", id="JSON-list"),
        pytest.param(
            None,
            None,
            {"releases": {"2.0.0rc1": [_file()], "1.9.0": []}},
            "no release found",
            id="no-final-release",
        ),
    ],
)
def test_failures_are_reported_not_raised(
    install, pypi, capsys, error, read_error, reply, reason
):
    install("1.0.4")
    pypi.error = error
    pypi.read_error = read_error
    if reply is not None:
        pypi.reply = reply
    result = check_for_updates()
    assert result == UpdateCheck("1.0.4", None, None)
    assert capsys.readouterr().out == _failure(reason)


@pytest.mark.parametrize(
    "installer, command",
    [
        ("pip\n", PIP),
        ("PIP \r\n", PIP),
        ("conda\n", CONDA),
        ("uv\n", BOTH),
        ("", BOTH),
        (None, BOTH),
        (PackageNotFoundError, BOTH),
    ],
    ids=[
        "pip",
        "pip-spelling",
        "conda",
        "other",
        "empty",
        "missing",
        "no-distribution",
    ],
)
def test_update_command_follows_installer(
    install, pypi, capsys, installer, command
):
    install("1.0.4", installer)
    pypi.reply = _releases("1.0.4", "1.5.0")
    assert check_for_updates() == UpdateCheck("1.0.4", "1.5.0", True)
    assert capsys.readouterr().out == (
        f"PyVBMC 1.5.0 is available; you have 1.0.4. Update with: {command}\n"
    )


@pytest.mark.parametrize(
    "timeout, expected",
    [
        (None, 5.0),
        (2.5, 2.5),
        (3, 3.0),
        (np.float64(0.5), 0.5),
        (3600, 3600.0),
    ],
)
def test_request(install, pypi, timeout, expected):
    install("1.0.4")
    if timeout is None:
        check_for_updates()
    else:
        check_for_updates(timeout=timeout)
    assert len(pypi.calls) == 1
    args, kwargs = pypi.calls[0]
    assert len(args) == 1
    assert kwargs == {"timeout": expected}
    assert type(kwargs["timeout"]) is float
    request = args[0]
    assert isinstance(request, urllib.request.Request)
    assert request.full_url == URL
    assert request.get_method() == "GET"
    assert request.data is None
    assert sorted(request.header_items()) == [
        ("Accept", "application/json"),
        ("User-agent", "pyvbmc/1.0.4 (check_for_updates)"),
    ]


@pytest.mark.parametrize(
    "timeout",
    [
        0,
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
        -float("inf"),
        True,
        False,
        "5",
        [5.0],
        1j,
        3600.5,
        1e9,
        10**400,
    ],
)
def test_invalid_timeout_raises(install, pypi, capsys, timeout):
    install("1.0.4")
    with pytest.raises(ValueError, match="timeout"):
        check_for_updates(timeout=timeout)
    assert pypi.calls == []
    assert capsys.readouterr().out == ""


def test_timeout_is_keyword_only():
    with pytest.raises(TypeError):
        check_for_updates(5.0)


def test_module_imports_no_networking_code():
    # Neither name is bound at module level ...
    namespace = vars(_update_check)
    assert "urllib" not in namespace
    assert "json" not in namespace
    # ... and loading the module alone, in a fresh interpreter, imports no
    # networking module that was not loaded before.
    script = textwrap.dedent(
        """
        import importlib.util
        import sys

        # What the module imports at load time, loaded before the snapshot:
        # importlib.metadata loads urllib.parse through email.utils.
        import importlib.metadata, math, numbers, re, typing

        before = set(sys.modules)
        path = sys.argv[1]
        spec = importlib.util.spec_from_file_location("_standalone", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["_standalone"] = module
        spec.loader.exec_module(module)
        assert callable(module.check_for_updates)
        loaded = set(sys.modules) - before
        networking = {
            "http.client", "json", "ssl", "urllib.error", "urllib.request"
        }
        print(sorted(loaded & networking))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(Path(_update_check.__file__))],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    assert completed.stdout.strip() == "[]"
