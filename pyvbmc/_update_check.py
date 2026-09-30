"""On-request check of PyPI for a newer release of PyVBMC.

`check_for_updates` is the only code in the package that opens a network
connection, and it runs only when the user calls it. The networking modules
are imported only when it runs, so that ``import pyvbmc`` imports none of
them.
"""

from __future__ import annotations

import math
import numbers
import re
from importlib.metadata import PackageNotFoundError, distribution, version
from typing import NamedTuple

PYPI_JSON_URL = "https://pypi.org/pypi/pyvbmc/json"

PIP_COMMAND = "python -m pip install --upgrade pyvbmc"
CONDA_COMMAND = "conda update --channel=conda-forge pyvbmc"
CONDA_NOTE = "(the conda-forge package can follow PyPI by a few days)"

NEWER_MESSAGE = (
    "PyVBMC {latest} is available; you have {installed}. "
    "Update with: {command}"
)
LATEST_MESSAGE = "PyVBMC {installed} is the latest release."
AHEAD_MESSAGE = (
    "PyVBMC {installed} is newer than the latest release on PyPI, {latest}."
)
DEVELOPMENT_MESSAGE = (
    "PyVBMC {installed} is a development version; "
    "the latest release is {latest}."
)
UNKNOWN_MESSAGE = (
    "PyVBMC's installed version is unknown; the latest release is {latest}."
)
FAILURE_MESSAGE = (
    "Could not reach PyPI ({reason}); see https://pypi.org/project/pyvbmc/."
)

# A release is a version of the form X.Y.Z and nothing else; [0-9] rather
# than \d, which also matches non-ASCII digits.
_FINAL_RELEASE = re.compile(r"([0-9]+)\.([0-9]+)\.([0-9]+)")


class UpdateCheck(NamedTuple):
    """Outcome of :func:`check_for_updates`.

    Attributes
    ----------
    installed : str or None
        The installed version of PyVBMC, or ``None`` when the package
        metadata cannot be found.
    latest : str or None
        The latest release of PyVBMC on PyPI, or ``None`` when PyPI could
        not be reached or its reply could not be read.
    update_available : bool or None
        Whether ``latest`` is newer than ``installed``. ``None`` when either
        is unknown, and for a development version, which is not compared
        with releases.
    """

    installed: str | None
    latest: str | None
    update_available: bool | None


def check_for_updates(*, timeout: float = 5.0) -> UpdateCheck:
    """Ask PyPI for the latest release of PyVBMC and say how to update.

    Sends one HTTPS GET request to ``https://pypi.org/pypi/pyvbmc/json``,
    with the ``User-Agent`` header ``pyvbmc/<installed version>
    (check_for_updates)``; the request carries nothing else about the
    installation or the user. This is the only network access in PyVBMC,
    and it happens only when this function is called. The proxy environment
    variables (``HTTPS_PROXY`` and its kin) are honored. Nothing is written
    to disk.

    Prints one message: that a newer release is available, with the command
    that installs it; that the installed version is the latest release; that
    the installed version is a development version, with the latest
    release; or that PyPI could not be reached, with the reason. The update
    command follows the installer recorded with the installed package:
    pip's, conda's (the conda-forge package can follow PyPI by a few days),
    or both when the installer is another or unknown.

    The latest release is the highest version of the form ``X.Y.Z`` on PyPI
    with at least one file that is not yanked, so that pre-releases,
    development releases and yanked releases are ignored. An installed
    version of any other form, such as a development install, is reported
    beside the latest release without being compared with it.

    Parameters
    ----------
    timeout : float, optional
        Seconds to wait for PyPI to connect and to reply. The default is
        ``5.0``.

    Returns
    -------
    UpdateCheck
        A named tuple of ``installed``, the installed version (``None`` when
        the package metadata cannot be found); ``latest``, the latest
        release on PyPI (``None`` when PyPI could not be reached or its
        reply could not be read); and ``update_available``, whether
        ``latest`` is newer than ``installed`` (``None`` when either is
        unknown or the installed version is a development version).

    Raises
    ------
    ValueError
        If `timeout` is not a finite positive number, or is a bool. A
        network, HTTP or parse failure raises nothing: the printed message
        gives its reason, and the returned tuple holds ``latest=None``.
    """
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, numbers.Real)
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError(
            "timeout must be a finite positive number of seconds, "
            f"got {timeout!r}"
        )

    installed = _installed_version()
    latest, reason = _fetch_latest_release(installed, float(timeout))
    if latest is None:
        print(FAILURE_MESSAGE.format(reason=reason))
        return UpdateCheck(installed, None, None)

    update_available = None
    installed_release = _parse_release(installed)
    if installed is None:
        message = UNKNOWN_MESSAGE.format(latest=latest)
    elif installed_release is None:
        message = DEVELOPMENT_MESSAGE.format(
            installed=installed, latest=latest
        )
    else:
        latest_release = _parse_release(latest)
        if installed_release < latest_release:
            update_available = True
            message = NEWER_MESSAGE.format(
                latest=latest,
                installed=installed,
                command=_update_command(_installer()),
            )
        elif installed_release == latest_release:
            update_available = False
            message = LATEST_MESSAGE.format(installed=installed)
        else:
            update_available = False
            message = AHEAD_MESSAGE.format(installed=installed, latest=latest)
    print(message)
    return UpdateCheck(installed, latest, update_available)


def _installed_version() -> str | None:
    try:
        return version("pyvbmc")
    except PackageNotFoundError:
        return None


def _installer() -> str | None:
    """Return the lowercased content of the distribution's INSTALLER."""
    try:
        text = distribution("pyvbmc").read_text("INSTALLER")
    except (PackageNotFoundError, OSError, ValueError):
        return None
    if text is None:
        return None
    return text.strip().lower() or None


def _update_command(installer: str | None) -> str:
    if installer == "pip":
        return PIP_COMMAND
    if installer == "conda":
        return f"{CONDA_COMMAND} {CONDA_NOTE}"
    return f"{PIP_COMMAND}, or with conda: {CONDA_COMMAND} {CONDA_NOTE}"


def _parse_release(text: str | None) -> tuple[int, int, int] | None:
    """Return a final X.Y.Z version as integers, or None for any other."""
    if not isinstance(text, str):
        return None
    match = _FINAL_RELEASE.fullmatch(text)
    if match is None:
        return None
    return tuple(int(part) for part in match.groups())


def _latest_release(releases: dict) -> str | None:
    """Return the highest final release with an installable file.

    A release whose file list is empty, or holds only yanked files, cannot
    be installed and is skipped.
    """
    best = None
    best_key = None
    for key, files in releases.items():
        parsed = _parse_release(key)
        if parsed is None or not isinstance(files, list):
            continue
        installable = any(
            isinstance(entry, dict) and not entry.get("yanked", False)
            for entry in files
        )
        if installable and (best is None or parsed > best):
            best, best_key = parsed, key
    return best_key


def _fetch_latest_release(
    installed: str | None, timeout: float
) -> tuple[str | None, str | None]:
    """Return PyPI's latest release, or None and the reason for failure."""
    import http.client
    import json
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        PYPI_JSON_URL,
        headers={
            "User-Agent": f"pyvbmc/{installed or 'unknown'} "
            "(check_for_updates)",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        return None, f"HTTP {error.code}"
    except urllib.error.URLError as error:
        return None, _describe_error(error.reason)
    except (OSError, ValueError) as error:
        return None, _describe_error(error)
    except http.client.HTTPException:
        return None, "unreadable reply"

    try:
        data = json.loads(body)
    except (ValueError, RecursionError):
        return None, "unreadable reply"
    releases = data.get("releases") if isinstance(data, dict) else None
    if not isinstance(releases, dict):
        return None, "unreadable reply"
    latest = _latest_release(releases)
    if latest is None:
        return None, "no release found"
    return latest, None


def _describe_error(error: object) -> str:
    """Return a few words on why the request failed."""
    if isinstance(error, TimeoutError):
        return "timed out"
    if isinstance(error, OSError) and error.strerror:
        return str(error.strerror)
    text = str(error).strip()
    return text or type(error).__name__
