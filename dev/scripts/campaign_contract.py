"""The part of the campaign contract that the Slurm-driven harnesses share.

``dev/plans/slurm-benchmark-support.md`` ("The campaign contract") defines
what a harness that the driver under ``dev/scripts/hpc/`` runs must do: the
subcommands ``prepare``, ``cases``, ``worker`` and ``verify``, and its
finishing steps. The pool (``svbmc_pool_run.py``), population
(``population_run.py``) and stacking (``svbmc_pool_stack.py``) harnesses
meet it through this module, which holds what they have in common:

- **the layout** of a campaign directory and of its case list (below);
- **the claim** of a case by the worker that runs it
  (:func:`acquire_claim`, :func:`release_claim`, :func:`claim_status`),
  with its staleness decided from the Slurm accounting
  (:func:`accounting_state`);
- **the identity**: the source identity that every worker compares with the
  manifest's (each tree's commit and clean state, the SHA-256 of named files
  and directories, the versions of the imported modules), where each tree
  and package was imported from, and the host part, recorded and never
  compared (:func:`identity`, :func:`source_differences`,
  :func:`host_part`, :func:`host_problems`);
- **the completion record** and its check (:func:`write_completion`,
  :func:`check_completion`), and the whole worker sequence that ends in it
  (:func:`run_worker`);
- **the environment check** against the pinned requirements file
  (:func:`environment_differences`), and the operator settings recorded in
  the manifest's ``site`` block (:func:`site_block`,
  :func:`site_differences`);
- **the reconciliation** of ``verify``'s states (:func:`reconcile`) and the
  finish's decision on its report (:func:`finish_decision`).

The layout, relative to the campaign directory ``out``::

    manifest.json                 the harness's ``prepare``
    cases.txt                     the case list; line i is case index i
    subsets/<name>.txt            a subset: "<index> <case>" per line
    records/<tag>.complete.json   completion records
    claims/<tag>                  claims; retired ones are
                                  claims/<tag>.stale.<job>_<task>
    <tag>.error.txt               the error file of a failed case
    slurm/                        the driver's job ids, logs, accounting
    tmp/                          TMPDIR of every process of the campaign

A case line is ``<tag> [<field> ...]``: its first field, up to the first
space, is the case's tag, unique in the campaign, and the rest is the
harness's own. A tag may hold ``/``, which puts the case's files, records
and claims in one subdirectory per group of cases (a condition, a
configuration), so that no directory holds thousands of files. The
harness writes its artifacts under ``out`` as it likes; the record, claim
and error paths above are fixed, because the driver's task script reads
the record path of a case from its line alone.

The worker's exit codes: 0 for a completed case (or one that already has a
record), 1 for a case that failed, :data:`EXIT_CLAIMED` (75) when a live
task holds the case's claim, :data:`EXIT_IDENTITY` (78) when the worker's
source identity differs from the manifest's, and 128 plus the signal's
number (143 for SIGTERM) when a signal stopped the run. The claim and the
identity refusals leave the case's files as they are.

A task that Slurm stops ends in one of two ways. At its time limit or on
``scancel`` it receives SIGTERM, and KillWait seconds later SIGKILL; the
worker catches SIGTERM (and SIGINT), removes the case's partial files and
its claim and exits, writing no error file, since a kill is no failure of
the run, and ``verify`` reports the case as ``missing``. A task killed
outright (SIGKILL, an out-of-memory kill, a node lost) leaves its partial
files and its claim, which the accounting then shows as ended: ``verify``
reports the case as ``interrupted``, which, like ``missing``, is
resubmitted, and the worker that takes over the stale claim removes the
partial files before it runs.

The module imports only the standard library at module level, so that a
harness can import it before it sets its thread variables and pins its
source trees. :func:`host_part` imports NumPy (so that its BLAS is loaded)
and ``threadpoolctl``, :func:`identity` imports the modules whose versions
it records, and the environment check imports ``packaging``. What a
platform lacks (the CPU affinity on Windows, for one) is recorded as null
there, and raises on Linux, where the campaigns run.

Run as a script, it offers the checks that the driver's shell scripts
call::

    python dev/scripts/campaign_contract.py check-env --requirements FILE
    python dev/scripts/campaign_contract.py check-site --manifest FILE
    python dev/scripts/campaign_contract.py check-cases --cases FILE \\
        [--subset FILE]
    python dev/scripts/campaign_contract.py finishing-steps --manifest FILE
    python dev/scripts/campaign_contract.py finish-check \\
        --verification FILE [--queued FILE] [--allow-missing] \\
        [--allow-running]
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

#: The version of this contract, recorded in every identity, claim,
#: completion record and verification report.
CONTRACT_VERSION = 1
#: Exit code of a worker that finds its case claimed by a live task
#: (``EX_TEMPFAIL``).
EXIT_CLAIMED = 75
#: Exit code of a worker whose source identity differs from the manifest's
#: (``EX_CONFIG``).
EXIT_IDENTITY = 78
#: The signals that stop the run of a case cleanly: Slurm's SIGTERM, at the
#: time limit and on ``scancel``, and the interactive SIGINT. A worker they
#: stop exits with 128 plus the signal's number.
STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT)

THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
#: The operator settings of the plan, recorded in the ``site`` block of the
#: manifest (``CONDA_SETUP`` and ``LOGIN_PROFILE`` are the driver's own).
SETTINGS = (
    "HARNESS",
    "CAMPAIGN_ENV",
    "NODE_FEATURE",
    "PARTITION",
    "LOGIN_SETUP",
    "CONDA_SETUP",
    "LOGIN_PROFILE",
    "PYVBMC_SOURCE",
    "PYVBMC_GPYREG_SOURCE",
    "BASELINE_DIR",
    "CASES_SUBSET",
    "THROTTLE",
    "TIME",
    "MEM",
    "ARRAY",
    "SBATCH_EXTRA",
)
#: The settings a campaign is bound to for its whole life: every later
#: submission and the finish must name the same ones. The others (the
#: partition, the limits, the subset, the array) may differ from one
#: submission to the next, and ``slurm/jobs.txt`` records each one's.
FIXED_SETTINGS = (
    "HARNESS",
    "CAMPAIGN_ENV",
    "NODE_FEATURE",
    "PYVBMC_SOURCE",
    "PYVBMC_GPYREG_SOURCE",
    "BASELINE_DIR",
)
PATH_SETTINGS = (
    "CAMPAIGN_ENV",
    "PYVBMC_SOURCE",
    "PYVBMC_GPYREG_SOURCE",
    "BASELINE_DIR",
)

#: The modules whose versions every source identity records, imported if
#: they are not yet, and those it records only where the process has
#: imported them (null otherwise).
REQUIRED_MODULES = ("numpy", "scipy", "cma")
OPTIONAL_MODULES = ("torch",)

#: The Slurm job states of a task that has ended; any other state, and no
#: answer, means that the task may still run.
ENDED_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "TIMEOUT",
    }
)
#: The states ``verify`` places a case in, in report order; ``stray`` counts
#: the files no case of the allocation owns.
STATUSES = (
    "verified",
    "verify_failed",
    "failed",
    "in_flight",
    "interrupted",
    "partial",
    "missing",
)
#: What makes ``verify`` exit non-zero: a record that fails its check, a
#: partial case (artifacts with neither a record nor a claim, which no path
#: of the contract leaves), a stray file. Failed, interrupted and missing
#: cases are reported and then rerun or left out.
FATAL = ("verify_failed", "partial", "stray")
#: The exit codes of ``finish-check``.
FINISH_FATAL, FINISH_MISSING, FINISH_IN_FLIGHT = 1, 3, 4

RECORDS, CLAIMS = "records", "claims"
RECORD_SUFFIX, ERROR_SUFFIX, STALE_INFIX = (
    ".complete.json",
    ".error.txt",
    ".stale.",
)
_COMPONENT = r"[A-Za-z0-9_+=-][A-Za-z0-9_.+=-]*"
TAG_PATTERN = re.compile(rf"{_COMPONENT}(?:/{_COMPONENT})*")
#: The characters a finishing step's arguments may hold, so that the
#: driver passes them through the shell as words.
STEP_ARGUMENT = re.compile(r"[A-Za-z0-9_.,=:+/@%-]+")
PYTHON_PIN = re.compile(r"#\s*python==([0-9]+(?:\.[0-9]+)*)\s*")
_RECORD_KEYS = (
    "contract",
    "tag",
    "case",
    "identity",
    "started",
    "finished",
    "elapsed_seconds",
    "artifacts",
)


class ContractError(RuntimeError):
    """A campaign directory, a record or an environment breaks the contract."""


class ClaimRefused(ContractError):
    """The case is claimed by a task that may still run."""


class IdentityError(ContractError):
    """The process has no identity, or one other than the campaign's."""


class HostError(ContractError):
    """A host fact the contract records is unavailable on Linux."""


class CompletionError(ContractError):
    """A completion record does not describe its case."""

    def __init__(self, tag, problems):
        self.tag = tag
        self.problems = list(problems)
        super().__init__(f"{tag}: " + "; ".join(self.problems))


class Interrupted(BaseException):
    """A signal of :data:`STOP_SIGNALS` stopped the run of a case.

    A ``BaseException``, like ``KeyboardInterrupt``, so that a harness's
    ``except Exception`` does not take it for a failure of the run.
    """

    def __init__(self, signum):
        self.signum = signum
        super().__init__(f"stopped by {signal.Signals(signum).name}")


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------


def now():
    """The local time, ISO 8601 with its UTC offset."""
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def write_json(path, value):
    """Write ``value`` as JSON through a temporary file and a rename.

    The temporary file is a dot-file beside the target with a unique name,
    so that a reader never sees a partial file and the stray scans of
    :func:`reconcile` never see the temporary one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_sha256(path):
    """The SHA-256 of a directory's content, independent of the platform.

    One line per file, ``<relative POSIX path> NUL <file SHA-256>``, in
    sorted order, hashed together; ``__pycache__`` directories and ``.pyc``
    files are left out. Any added, removed, renamed or changed file
    changes it.
    """
    root = Path(path)
    if not root.is_dir():
        raise IdentityError(f"{root} is not a directory")
    lines = []
    for directory, subdirectories, files in os.walk(root):
        subdirectories[:] = sorted(
            d for d in subdirectories if d != "__pycache__"
        )
        for name in files:
            if name.endswith(".pyc"):
                continue
            full = Path(directory) / name
            relative = full.relative_to(root).as_posix()
            lines.append(f"{relative}\0{sha256_file(full)}\n")
    digest = hashlib.sha256()
    for line in sorted(lines):
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def record_path(out, tag):
    """``<out>/records/<tag>.complete.json``."""
    return Path(out) / RECORDS / f"{tag}{RECORD_SUFFIX}"


def error_path(out, tag):
    """``<out>/<tag>.error.txt``."""
    return Path(out) / f"{tag}{ERROR_SUFFIX}"


def claim_path(out, tag):
    """``<out>/claims/<tag>``."""
    return Path(out) / CLAIMS / tag


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------


def case_tag(line):
    """The tag of a case line, its first field, checked."""
    if (
        not isinstance(line, str)
        or not line
        or any(c in line for c in "\r\n\t")
    ):
        raise ContractError(f"{line!r} is not a case line: one line, no tab")
    tag = line.split(" ", 1)[0]
    if not TAG_PATTERN.fullmatch(tag):
        raise ContractError(
            f"{line!r}: the tag {tag!r} must be letters, digits and _.+=- "
            "in components joined by /, none starting with a dot"
        )
    return tag


def read_cases(path):
    """The case lines of ``cases.txt``, checked: unique tags, LF endings."""
    text = Path(path).read_bytes().decode("utf-8")
    if "\r" in text:
        raise ContractError(f"{path} has carriage returns; lines end in LF")
    if text and not text.endswith("\n"):
        raise ContractError(f"{path} does not end in a newline")
    lines = text.splitlines()
    if not lines:
        raise ContractError(f"{path} holds no case")
    seen = {}
    for index, line in enumerate(lines, start=1):
        tag = case_tag(line)
        if tag in seen:
            raise ContractError(
                f"{path}: the tag {tag} is on lines {seen[tag]} and {index}"
            )
        seen[tag] = index
    return lines


def read_subset(path, cases):
    """The case indices of a subset file, checked against the full list.

    Each line is ``<index> <case>``, the index a line number of the full
    list and the case its text there; the indices rise strictly.
    """
    text = Path(path).read_bytes().decode("utf-8")
    if "\r" in text:
        raise ContractError(f"{path} has carriage returns; lines end in LF")
    indices = []
    for number, line in enumerate(text.splitlines(), start=1):
        index, _, case = line.partition(" ")
        if not index.isdigit() or not case:
            raise ContractError(f"{path}:{number}: not '<index> <case>'")
        index = int(index)
        if not 1 <= index <= len(cases) or cases[index - 1] != case:
            raise ContractError(
                f"{path}:{number}: case {index} of the full list is "
                f"not {case!r}"
            )
        if indices and index <= indices[-1]:
            raise ContractError(f"{path}:{number}: the indices must rise")
        indices.append(index)
    if not indices:
        raise ContractError(f"{path} names no case")
    return indices


def compress_indices(indices):
    """Sorted case indices as the shortest sbatch-style list, ``1-3,7``."""
    parts, run = [], None
    for index in sorted(set(int(i) for i in indices)):
        if run and index == run[1] + 1:
            run[1] = index
            continue
        if run:
            parts.append(
                str(run[0]) if run[0] == run[1] else f"{run[0]}-{run[1]}"
            )
        run = [index, index]
    if run:
        parts.append(str(run[0]) if run[0] == run[1] else f"{run[0]}-{run[1]}")
    return ",".join(parts)


# --------------------------------------------------------------------------
# Slurm
# --------------------------------------------------------------------------


def _command(args, timeout):
    """Run a command found on the PATH (``.cmd`` shims included on Windows)."""
    executable = shutil.which(args[0])
    if executable is None:
        raise FileNotFoundError(f"{args[0]} is not on the PATH")
    return subprocess.run(
        [executable, *args[1:]],
        capture_output=True,
        text=True,
        timeout=timeout,
        stdin=subprocess.DEVNULL,
    )


def slurm_task(environ=None):
    """The Slurm task this process runs in, or None outside Slurm.

    ``job`` is the array's job id and ``array_task`` the task's index in it
    for an array task (``SLURM_ARRAY_JOB_ID``, ``SLURM_ARRAY_TASK_ID``), so
    that ``<job>_<array_task>`` names the task in the accounting; for a job
    that is not an array, ``job`` is its id and ``array_task`` is None.
    ``restart_count`` is ``SLURM_RESTART_COUNT``, 0 before any requeue.
    """
    env = os.environ if environ is None else environ
    job_id = env.get("SLURM_JOB_ID")
    if not job_id:
        return None
    array_job = env.get("SLURM_ARRAY_JOB_ID") or None
    return {
        "job": array_job or job_id,
        "array_task": env.get("SLURM_ARRAY_TASK_ID") if array_job else None,
        "job_id": job_id,
        "restart_count": int(env.get("SLURM_RESTART_COUNT") or 0),
        "node": env.get("SLURMD_NODENAME"),
    }


def task_step(job, array_task=None):
    """``<job>_<array_task>``, or ``<job>`` for a job that is not an array."""
    if array_task is None or array_task == "":
        return str(job)
    return f"{job}_{array_task}"


def accounting_state(job, array_task=None, timeout=60):
    """Whether a Slurm task may still run, from the accounting.

    Runs ``sacct -n -X -P -j <job>_<task> -o State``. The task has ended when
    every line of the answer starts with one of :data:`ENDED_STATES`
    (``CANCELLED by <uid>`` counts as ``CANCELLED``). A task that has not
    ended (pending, running, requeued, suspended, or any state not listed),
    a query that fails and an empty answer all count as live, since none
    of them shows that the task is over.

    Returns ``{"live": bool, "state": str or None, "detail": str}``.
    """
    step = task_step(job, array_task)
    try:
        result = _command(
            ["sacct", "-n", "-X", "-P", "-j", step, "-o", "State"], timeout
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {
            "live": True,
            "state": None,
            "detail": f"sacct failed: {error}",
        }
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip()
        return {
            "live": True,
            "state": None,
            "detail": f"sacct exited {result.returncode}: {message}",
        }
    states = [
        line.split()[0].rstrip("+").upper()
        for line in result.stdout.splitlines()
        if line.split()
    ]
    if not states:
        return {
            "live": True,
            "state": None,
            "detail": f"sacct knows no {step}",
        }
    joined = ",".join(states)
    if all(state in ENDED_STATES for state in states):
        return {
            "live": False,
            "state": joined,
            "detail": f"{step} is {joined}",
        }
    return {"live": True, "state": joined, "detail": f"{step} is {joined}"}


# --------------------------------------------------------------------------
# Claims
# --------------------------------------------------------------------------


def new_claim(tag, task=None):
    """The claim this process makes: its task, host, pid, time and a token."""
    task = slurm_task() if task is None else task
    task = task or {}
    return {
        "contract": CONTRACT_VERSION,
        "tag": tag,
        "job": task.get("job"),
        "array_task": task.get("array_task"),
        "job_id": task.get("job_id"),
        "restart_count": task.get("restart_count"),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "started": now(),
        "token": uuid.uuid4().hex,
    }


def claim_owner(record):
    """The name of a claim's owner: its Slurm task, or ``<host>-<pid>``."""
    if record.get("job"):
        return task_step(record["job"], record.get("array_task"))
    name = f"{record.get('host')}-{record.get('pid')}"
    return re.sub(r"[^A-Za-z0-9_.+=-]", "_", name)


def _process_alive(pid):
    """Whether a process of this host runs with this pid."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = (
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        )
        kernel32.GetExitCodeProcess.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        )
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            # Access denied means that the process exists.
            return ctypes.get_last_error() == 5
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def claim_liveness(record, query=None):
    """Whether a claim's owner may still run: ``{"live", "state", "detail"}``.

    A claim made by a Slurm task is live unless the accounting shows that
    the task has ended (:func:`accounting_state`, or ``query`` with the
    same signature). A claim made outside Slurm names no task the
    accounting knows: it is live while its process runs on this host, and
    one made on another host is live until an operator removes it.
    """
    query = accounting_state if query is None else query
    if record.get("job"):
        return query(record["job"], record.get("array_task"))
    pid = record.get("pid")
    if record.get("host") == socket.gethostname():
        alive = _process_alive(pid)
        return {
            "live": alive,
            "state": None,
            "detail": f"process {pid} of this host is "
            + ("running" if alive else "gone"),
        }
    return {
        "live": True,
        "state": None,
        "detail": f"made outside Slurm on {record.get('host')}",
    }


def _write_temporary(directory, name, value):
    temporary = Path(directory) / f".{name}.{uuid.uuid4().hex}.tmp"
    with open(temporary, "x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return temporary


def _link_new(path, record):
    """Create ``path`` holding ``record`` unless it exists; True if created."""
    temporary = _write_temporary(path.parent, path.name, record)
    try:
        os.link(temporary, path)
        return True
    except FileExistsError:
        return False
    finally:
        temporary.unlink(missing_ok=True)


def _replace(path, record):
    temporary = _write_temporary(path.parent, path.name, record)
    os.replace(temporary, path)


def _retire(path, judged):
    """Rename the stale claim ``judged`` to ``<tag>.stale.<owner>``.

    The rename is a hard link that fails when the name exists, then the
    removal of the claim, so that of two workers that judged the same
    claim stale only one retires it. A worker whose link reached another
    claim than the one it judged (the claim changed between its read and
    its link) undoes the link and reports failure.
    """
    stale = path.with_name(f"{path.name}{STALE_INFIX}{claim_owner(judged)}")
    try:
        os.link(path, stale)
    except (FileExistsError, FileNotFoundError):
        return False
    try:
        retired = read_json(stale)
    except (OSError, ValueError):
        retired = None
    if retired != judged:
        stale.unlink(missing_ok=True)
        return False
    path.unlink(missing_ok=True)
    return True


def _same_task(existing, mine):
    return bool(mine.get("job")) and (
        existing.get("job"),
        existing.get("array_task"),
    ) == (mine.get("job"), mine.get("array_task"))


def _previous(existing, reason, state=None):
    return {
        "reason": reason,
        "owner": claim_owner(existing),
        "restart_count": existing.get("restart_count"),
        "host": existing.get("host"),
        "started": existing.get("started"),
        "slurm_state": state,
    }


def acquire_claim(out, tag, query=None, task=None, attempts=3):
    """Claim one case for this process, or raise :class:`ClaimRefused`.

    The claim ``claims/<tag>`` is written to a temporary file and
    hard-linked into place, so that it is never empty and the link fails
    when a claim exists. An existing claim that names this process's own
    job and array task (a requeue) is taken over, rewritten with the
    current restart count. Otherwise it is judged by
    :func:`claim_liveness`: a live claim refuses the case, and a stale one
    is renamed to ``claims/<tag>.stale.<job>_<task>`` before this process
    creates its own. Nothing but the claim files is touched.

    Returns the claim, a mapping with its ``path`` and ``token``, which
    :func:`release_claim` takes. ``query`` replaces
    :func:`accounting_state` and ``task`` :func:`slurm_task`, for tests.
    """
    path = claim_path(out, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    mine = new_claim(tag, task=task)
    for _ in range(attempts):
        if _link_new(path, mine):
            return {**mine, "path": str(path)}
        try:
            existing = read_json(path)
        except FileNotFoundError:
            continue  # released or retired meanwhile
        except (OSError, ValueError) as error:
            raise ClaimRefused(
                f"{path} exists and cannot be read ({error}); an operator "
                "removes it"
            ) from error
        if _same_task(existing, mine):
            mine["previous"] = _previous(existing, "requeue")
            _replace(path, mine)
            return {**mine, "path": str(path)}
        liveness = claim_liveness(existing, query=query)
        if liveness["live"]:
            raise ClaimRefused(
                f"claimed by {claim_owner(existing)} since "
                f"{existing.get('started')}, which may still run "
                f"({liveness['detail']})"
            )
        if not _retire(path, existing):
            raise ClaimRefused(
                f"another worker is replacing the stale claim of "
                f"{claim_owner(existing)}"
            )
        mine["previous"] = _previous(existing, "stale", liveness["state"])
    raise ClaimRefused(f"{path} changed {attempts} times while it was claimed")


def release_claim(claim):
    """Remove a claim this process holds; True if it was removed.

    A claim file that holds another token (a claim some other worker made
    after this one's was judged stale) is left in place.
    """
    path = Path(claim["path"])
    try:
        current = read_json(path)
    except (OSError, ValueError):
        return False
    if current.get("token") != claim["token"]:
        return False
    path.unlink(missing_ok=True)
    return True


def claim_status(out, tag, query=None):
    """The claim of one case: ``{"state": "absent" | "live" | "stale", ...}``.

    A claim that cannot be read counts as live, like a failed query.
    """
    path = claim_path(out, tag)
    try:
        record = read_json(path)
    except FileNotFoundError:
        return {"state": "absent"}
    except (OSError, ValueError) as error:
        return {
            "state": "live",
            "owner": None,
            "detail": f"unreadable: {error}",
        }
    liveness = claim_liveness(record, query=query)
    return {
        "state": "live" if liveness["live"] else "stale",
        "owner": claim_owner(record),
        "started": record.get("started"),
        "slurm_state": liveness["state"],
        "detail": liveness["detail"],
    }


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------


def git(path, *args):
    return subprocess.run(
        ["git", "-C", str(path), *args],
        capture_output=True,
        text=True,
        check=True,
        stdin=subprocess.DEVNULL,
    ).stdout.strip()


def tree_state(path):
    """``(source, where)`` of one source tree, which must be a git checkout.

    ``source`` is what the identity compares, the commit and whether the
    tree is clean (``git status --porcelain`` prints nothing, so no
    uncommitted or untracked change); ``where`` is its resolved path and
    the status lines of a dirty tree, recorded only. ``path`` must be the
    top of its checkout.
    """
    path = Path(path).resolve()
    try:
        top = git(path, "rev-parse", "--show-toplevel")
        commit = git(path, "rev-parse", "HEAD")
        status = git(path, "status", "--porcelain")
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", None) or error
        raise IdentityError(
            f"{path} is not a git checkout: {detail}"
        ) from error
    if Path(top).resolve() != path:
        raise IdentityError(f"{path} is not the top of its checkout {top}")
    return (
        {"commit": commit, "clean": not status},
        {"path": str(path), "dirty": status.splitlines()},
    )


def file_hashes(root, names):
    """``{name: SHA-256}`` of files and directories named relative to ``root``.

    A directory's key ends in ``/`` and its value is
    :func:`directory_sha256`.
    """
    root = Path(root)
    hashes = {}
    for name in names:
        relative = Path(name)
        if relative.is_absolute():
            raise IdentityError(f"{name} must be relative to {root}")
        path = root / relative
        if path.is_dir():
            hashes[f"{relative.as_posix().rstrip('/')}/"] = directory_sha256(
                path
            )
        elif path.is_file():
            hashes[relative.as_posix()] = sha256_file(path)
        else:
            raise IdentityError(f"{path} does not exist")
    return hashes


def module_version(module):
    version = getattr(module, "__version__", None)
    if not isinstance(version, str):
        raise IdentityError(f"{module.__name__} has no __version__ string")
    return version


def imported_versions(required=REQUIRED_MODULES, optional=OPTIONAL_MODULES):
    """The versions of Python and of the imported modules.

    Each version is the imported module's ``__version__``, never the
    installed distribution's metadata, which names another version when a
    tree is pinned by path. The ``required`` modules are imported if they
    are not yet; an ``optional`` one is recorded where the process has
    imported it, and is null otherwise.
    """
    versions = {"python": platform.python_version()}
    for name in required:
        versions[name] = module_version(importlib.import_module(name))
    for name in optional:
        module = sys.modules.get(name)
        versions[name] = None if module is None else module_version(module)
    return versions


def installed_version(name):
    """The version the installed distribution's metadata names, or None."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def module_origin(name, tree=None):
    """Where the package ``name`` is imported from; under ``tree`` if given."""
    try:
        module = importlib.import_module(name)
    except ImportError as error:
        raise IdentityError(f"{name} cannot be imported: {error}") from error
    location = Path(module.__file__).resolve().parent
    if tree is not None:
        tree = Path(tree).resolve()
        if location != tree and tree not in location.parents:
            raise IdentityError(
                f"{name} is imported from {location}, not from the tree {tree}"
            )
    return str(location)


def identity(
    trees,
    files=(),
    root=None,
    modules=None,
    required=REQUIRED_MODULES,
    optional=OPTIONAL_MODULES,
    host=True,
):
    """This process's identity: ``source``, ``imports`` and ``host``.

    Parameters
    ----------
    trees : mapping of str to path
        The source trees, each the top of a clean git checkout: ``harness``
        (the whole checkout the harness runs from), the package tree, gpyreg
        and, for the original S-VBMC arm, the baseline, under the names the
        harness chooses.
    files : sequence of str
        The files and directories whose SHA-256 the source identity holds
        (the harness modules, the targets module, ``dev/scripts/data``),
        relative to ``root``.
    root : path, optional
        The base of ``files``; ``trees["harness"]`` by default.
    modules : mapping of str to str or None, optional
        Packages whose import path is recorded, each mapped to the name of
        the tree it must be imported from (None: recorded only). Their
        installed metadata versions are recorded too, under
        ``imports.installed_metadata_versions``, labelled as such.
    required, optional : sequence of str
        See :func:`imported_versions`.
    host : bool
        Whether to add the host part (:func:`host_part`).

    Returns
    -------
    dict
        ``source`` is what every worker compares with the manifest's
        (:func:`source_differences`): the trees' commits and clean states,
        the file hashes and the imported versions. ``imports`` (each tree's
        path, each package's import path, the installed metadata versions)
        and ``host`` are recorded and never compared, so that any node may
        run any case and a campaign read elsewhere keeps its identity.
    """
    source_trees, tree_paths = {}, {}
    for name, path in trees.items():
        source_trees[name], tree_paths[name] = tree_state(path)
    base = Path(root) if root is not None else Path(trees["harness"])
    origins, installed = {}, {}
    for name, tree_name in (modules or {}).items():
        tree = None if tree_name is None else trees[tree_name]
        origins[name] = module_origin(name, tree)
        installed[name] = installed_version(name)
    record = {
        "contract": CONTRACT_VERSION,
        "source": {
            "trees": source_trees,
            "files": file_hashes(base, files),
            "versions": imported_versions(required, optional),
        },
        "imports": {
            "trees": tree_paths,
            "modules": origins,
            "installed_metadata_versions": installed,
        },
    }
    if host:
        record["host"] = host_part()
    return record


def _differences(actual, expected, prefix=""):
    keys = set(actual) | set(expected)
    found = []
    for key in sorted(keys, key=str):
        name = f"{prefix}{key}"
        a, e = actual.get(key), expected.get(key)
        if isinstance(a, dict) and isinstance(e, dict):
            found += _differences(a, e, f"{name}.")
        elif key not in actual or key not in expected or a != e:
            found.append(name)
    return found


def source_differences(actual, expected):
    """The dotted keys in which two identities' ``source`` parts differ."""
    return _differences(
        (actual or {}).get("source") or {},
        (expected or {}).get("source") or {},
    )


# --------------------------------------------------------------------------
# Host part
# --------------------------------------------------------------------------


def _unavailable(strict, what, error=None):
    if strict:
        detail = f": {error}" if error is not None else ""
        raise HostError(f"{what} is unavailable on this Linux host{detail}")
    return None


def cpu_model(strict):
    """The CPU's model name."""
    if sys.platform.startswith("linux"):
        try:
            text = Path("/proc/cpuinfo").read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError as error:
            return _unavailable(strict, "/proc/cpuinfo", error)
        for key in (
            "model name",
            "Model",
            "Hardware",
            "cpu model",
            "Processor",
        ):
            match = re.search(rf"^{key}\s*:\s*(.+)$", text, re.MULTILINE)
            if match:
                return match.group(1).strip()
        return _unavailable(strict, "the CPU model", "not in /proc/cpuinfo")
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                return str(
                    winreg.QueryValueEx(key, "ProcessorNameString")[0]
                ).strip()
        except OSError:
            pass
    if sys.platform == "darwin":
        try:
            result = _command(["sysctl", "-n", "machdep.cpu.brand_string"], 10)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or _unavailable(strict, "the CPU model")


def _features(text, key):
    match = re.search(rf"\b{key}=(\S*)", text)
    if match is None:
        return None
    value = match.group(1)
    return [] if value in ("", "(null)") else value.split(",")


def node_features(strict, environ=None):
    """The features of the Slurm node this process runs on; None outside Slurm.

    From ``scontrol show node <SLURMD_NODENAME> -o`` (the short hostname
    when the variable is unset): ``{"node", "available", "active"}``.
    """
    env = os.environ if environ is None else environ
    if not env.get("SLURM_JOB_ID"):
        return None
    node = env.get("SLURMD_NODENAME") or socket.gethostname().split(".")[0]
    try:
        result = _command(["scontrol", "show", "node", node, "-o"], 60)
    except (OSError, subprocess.SubprocessError) as error:
        return _unavailable(strict, f"scontrol show node {node}", error)
    if result.returncode != 0:
        return _unavailable(
            strict, f"scontrol show node {node}", result.stderr.strip()
        )
    available = _features(result.stdout, "AvailableFeatures")
    active = _features(result.stdout, "ActiveFeatures")
    if available is None and active is None:
        return _unavailable(
            strict, f"the features of node {node}", "scontrol names none"
        )
    return {"node": node, "available": available or [], "active": active or []}


def blas_libraries(strict):
    """The BLAS and OpenMP libraries and their threads (threadpoolctl)."""
    try:
        import numpy  # noqa: F401  (loads its BLAS, which threadpoolctl reads)
        from threadpoolctl import threadpool_info
    except ImportError as error:
        return _unavailable(strict, "threadpoolctl", error)
    keys = (
        "user_api",
        "internal_api",
        "prefix",
        "version",
        "num_threads",
        "threading_layer",
        "architecture",
        "filepath",
    )
    libraries = [
        {key: entry.get(key) for key in keys} for entry in threadpool_info()
    ]
    if strict and not any(entry["user_api"] == "blas" for entry in libraries):
        raise HostError(
            "threadpoolctl finds no BLAS library on this Linux host"
        )
    return libraries


def cpu_affinity(strict):
    """``{"cpus", "physical_cores"}`` of ``os.sched_getaffinity(0)``.

    ``physical_cores`` holds one ``<package>:<core>`` per physical core
    that the CPUs belong to, from ``/sys/devices/system/cpu``, so that a
    record read on another machine still shows whether its task ran on one
    physical core. None where the platform has no affinity call.
    """
    getter = getattr(os, "sched_getaffinity", None)
    if getter is None:
        return _unavailable(strict, "os.sched_getaffinity")
    cpus = sorted(getter(0))
    cores = set()
    for cpu in cpus:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            package = (topology / "physical_package_id").read_text().strip()
            core = (topology / "core_id").read_text().strip()
        except OSError as error:
            _unavailable(strict, f"the topology of CPU {cpu}", error)
            return {"cpus": cpus, "physical_cores": None}
        cores.add(f"{package}:{core}")
    return {"cpus": cpus, "physical_cores": sorted(cores)}


def slurm_ids(environ=None):
    env = os.environ if environ is None else environ
    names = {
        "job_id": "SLURM_JOB_ID",
        "array_job_id": "SLURM_ARRAY_JOB_ID",
        "array_task_id": "SLURM_ARRAY_TASK_ID",
        "restart_count": "SLURM_RESTART_COUNT",
        "node": "SLURMD_NODENAME",
        "partition": "SLURM_JOB_PARTITION",
        "cpus_per_task": "SLURM_CPUS_PER_TASK",
    }
    return {key: env.get(name) for key, name in names.items()}


def host_part(strict=None, environ=None):
    """Where this process runs: recorded in every identity, never compared.

    The hostname, the platform, the interpreter, the CPU model, the node's
    Slurm features, the BLAS libraries and their thread counts as
    ``threadpoolctl`` reports them, the CPU affinity with its physical
    cores, the thread variables and the Slurm job, array task and restart
    ids. ``strict`` (the default on Linux) raises :class:`HostError` where
    a fact is unavailable; elsewhere it is recorded as null.
    """
    strict = sys.platform.startswith("linux") if strict is None else strict
    env = os.environ if environ is None else environ
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "cpu_model": cpu_model(strict),
        "node_features": node_features(strict, env),
        "blas": blas_libraries(strict),
        "cpu_affinity": cpu_affinity(strict),
        "threads": {key: env.get(key) for key in THREAD_KEYS},
        "slurm": slurm_ids(env),
    }


def host_problems(host, node_feature):
    """Why a record's host part does not show the campaign's node and core.

    The node's features must include ``node_feature`` and the CPU affinity
    must lie on one physical core. Returns the problems, empty when there
    are none.
    """
    host = host or {}
    problems = []
    features = host.get("node_features")
    if not features:
        problems.append("the record holds no node features")
    elif node_feature not in set(features.get("available") or []) | set(
        features.get("active") or []
    ):
        problems.append(
            f"node {features.get('node')} lacks the feature {node_feature} "
            f"(it has {features.get('available')})"
        )
    affinity = host.get("cpu_affinity")
    if not affinity or affinity.get("physical_cores") is None:
        problems.append("the record holds no CPU affinity")
    elif len(affinity["physical_cores"]) != 1:
        problems.append(
            f"the CPU affinity {affinity.get('cpus')} spans "
            f"{len(affinity['physical_cores'])} physical cores"
        )
    return problems


# --------------------------------------------------------------------------
# Completion records
# --------------------------------------------------------------------------


def artifact_entries(out, paths):
    """``{relative POSIX path: {"sha256", "bytes"}}`` of a case's artifacts."""
    out = Path(out).resolve()
    entries = {}
    for path in paths:
        full = Path(path)
        full = (full if full.is_absolute() else out / full).resolve()
        try:
            relative = full.relative_to(out).as_posix()
        except ValueError as error:
            raise ContractError(f"{full} is outside {out}") from error
        entries[relative] = {
            "sha256": sha256_file(full),
            "bytes": full.stat().st_size,
        }
    if not entries:
        raise ContractError("a completion record needs at least one artifact")
    return entries


def write_completion(
    out, tag, case, artifacts, identity, started, elapsed_seconds, extra=None
):
    """Write ``records/<tag>.complete.json``, the last file a case writes.

    It holds the case line, the SHA-256 and size of every artifact (paths
    relative to ``out``), the start and finish times, the elapsed seconds
    and the process's identity with its host part; ``extra`` adds the
    harness's own fields, which may not reuse the contract's.
    """
    record = {
        "contract": CONTRACT_VERSION,
        "tag": tag,
        "case": case,
        "identity": identity,
        "started": time.strftime(
            "%Y-%m-%dT%H:%M:%S%z", time.localtime(started)
        ),
        "finished": now(),
        "elapsed_seconds": float(elapsed_seconds),
        "artifacts": artifact_entries(out, artifacts),
    }
    clash = sorted(set(extra or {}) & set(_RECORD_KEYS))
    if clash:
        raise ContractError(f"the harness's record fields reuse {clash}")
    record.update(extra or {})
    write_json(record_path(out, tag), record)
    return record


def check_completion(out, tag, expected, required=(), node_feature=None):
    """Re-check one completion record; raise :class:`CompletionError`.

    The record must name its tag, hold a source identity equal to
    ``expected``'s (the manifest's), list every ``required`` artifact, and
    every artifact it lists must exist with its recorded SHA-256. With
    ``node_feature``, its host part must also pass :func:`host_problems`.
    """
    path = record_path(out, tag)
    try:
        record = read_json(path)
    except (OSError, ValueError) as error:
        raise CompletionError(
            tag, [f"{path} cannot be read: {error}"]
        ) from error
    problems = []
    if record.get("tag") != tag:
        problems.append(f"the record belongs to {record.get('tag')}")
    differing = source_differences(record.get("identity"), expected)
    if differing:
        problems.append(f"the recorded source identity differs in {differing}")
    artifacts = record.get("artifacts") or {}
    absent = [name for name in required if name not in artifacts]
    if absent:
        problems.append(f"the record lists no {absent}")
    for name, entry in sorted(artifacts.items()):
        full = Path(out) / name
        if not full.is_file():
            problems.append(f"{name} is missing")
        elif sha256_file(full) != entry.get("sha256"):
            problems.append(f"{name} differs from its recorded SHA-256")
    if node_feature is not None:
        problems += host_problems(
            (record.get("identity") or {}).get("host"), node_feature
        )
    if problems:
        raise CompletionError(tag, problems)
    return record


def _say(message):
    print(message, flush=True)


def _remove(out, paths):
    removed = []
    for path in paths:
        path = Path(path)
        path = path if path.is_absolute() else Path(out) / path
        if path.exists():
            path.unlink()
            removed.append(str(path))
    return removed


def _raise_interrupted(signum, frame):
    raise Interrupted(signum)


def _handle_stop_signals(handler):
    """Set ``handler`` for :data:`STOP_SIGNALS`; return the previous ones.

    Python sets signal handlers in the main thread alone; elsewhere this
    does nothing and returns None.
    """
    if threading.current_thread() is not threading.main_thread():
        return None
    return {signum: signal.signal(signum, handler) for signum in STOP_SIGNALS}


def _restore_stop_signals(previous):
    for signum, handler in (previous or {}).items():
        signal.signal(signum, signal.SIG_DFL if handler is None else handler)


def run_worker(out, case, expected, identity, run, partial_files, query=None):
    """The contract's worker sequence for one case; returns the exit code.

    Parameters
    ----------
    out : path
        The campaign directory.
    case : str
        The case's line of ``cases.txt``, as ``worker --case`` receives it.
    expected : dict
        The manifest's identity.
    identity : callable
        ``identity()`` returns this process's identity with its host part
        (:func:`identity`).
    run : callable
        ``run(identity)`` runs the case, writes its artifacts under ``out``
        and returns ``(artifacts, extra)``: the artifact paths and the
        harness's own fields for the completion record (a mapping or None).
    partial_files : callable
        ``partial_files()`` returns every artifact path the case may have
        written (absolute, or relative to ``out``). Those that exist are
        removed before the run, so that a rerun never mixes its files with
        an interrupted attempt's, and after a failure or a stop.
    query : callable, optional
        Replaces :func:`accounting_state` in the claim.

    Returns
    -------
    int
        0 when the case has a completion record, at once if it had one;
        :data:`EXIT_IDENTITY` when the identity cannot be established or
        differs from ``expected``'s source; :data:`EXIT_CLAIMED` when the
        case is claimed by a live task or the claim cannot be made; 1 when
        the case failed; 128 plus the signal's number when a signal of
        :data:`STOP_SIGNALS` stopped it. The two refusals touch none of the
        case's files. From the claim to its release, the stop signals raise
        :class:`Interrupted` (the previous handlers are restored after).
        A failure removes the case's partial files, writes
        ``<tag>.error.txt`` with the traceback and removes the claim; a
        stop removes the partial files and the claim and writes no error
        file, so that ``verify`` reports the case as missing, unless the
        completion record was already written, which leaves the case
        complete; a success writes the completion record, removes an
        earlier attempt's error file and then the claim. While the
        clean-up of a failure or a stop runs, and while the claim is
        released, further stop signals are ignored.
    """
    out = Path(out)
    tag = case_tag(case)
    record = record_path(out, tag)
    if record.exists():
        _say(f"{tag}: already has a completion record; nothing to do")
        return 0
    try:
        actual = identity()
    except Exception:
        _say(f"{tag}: refused, the identity cannot be established:")
        _say(traceback.format_exc())
        return EXIT_IDENTITY
    differing = source_differences(actual, expected)
    if differing:
        _say(
            f"{tag}: refused, the source identity differs from the "
            f"manifest's in {differing}"
        )
        return EXIT_IDENTITY
    try:
        claim = acquire_claim(out, tag, query=query)
    except Exception as error:
        _say(f"{tag}: refused, {error}; the case's files are left as they are")
        return EXIT_CLAIMED
    handlers = _handle_stop_signals(_raise_interrupted)
    try:
        if record.exists():
            _say(f"{tag}: completed by another task; nothing to do")
            return 0
        previous = claim.get("previous")
        if previous:
            _say(
                f"{tag}: took over the {previous['reason']} claim of "
                f"{previous['owner']}"
            )
        started = time.time()
        try:
            removed = _remove(out, partial_files())
            if removed:
                _say(
                    f"{tag}: removed {len(removed)} files of an earlier "
                    "attempt before the run"
                )
            artifacts, extra = run(actual)
            write_completion(
                out,
                tag,
                case,
                artifacts,
                actual,
                started,
                time.time() - started,
                extra,
            )
            error_path(out, tag).unlink(missing_ok=True)
            _say(f"{tag}: complete in {time.time() - started:.1f} s")
            # No stop signal may interrupt what follows a complete case.
            _handle_stop_signals(signal.SIG_IGN)
        except Interrupted as stop:
            _handle_stop_signals(signal.SIG_IGN)
            name = signal.Signals(stop.signum).name
            if record.exists():
                _say(f"{tag}: {name} came after the completion record")
                return 128 + stop.signum
            try:
                removed = _remove(out, partial_files())
            except Exception:
                removed = []
                _say(
                    "removing the partial files failed:\n"
                    + traceback.format_exc()
                )
            _say(
                f"{tag}: stopped by {name}; removed {len(removed)} partial "
                "files and the claim, and left the case to be resubmitted"
            )
            return 128 + stop.signum
        except Exception as error:
            _handle_stop_signals(signal.SIG_IGN)
            text = traceback.format_exc()
            try:
                removed = _remove(out, partial_files())
            except Exception:
                removed = []
                text += (
                    "\nremoving the partial files failed:\n"
                    + traceback.format_exc()
                )
            path = error_path(out, tag)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                f"{tag}: {type(error).__name__}: {error}\n{text}",
                encoding="utf-8",
            )
            _say(text)
            _say(
                f"{tag}: failed; removed {len(removed)} partial files, "
                f"see {path}"
            )
            return 1
        return 0
    finally:
        _handle_stop_signals(signal.SIG_IGN)
        release_claim(claim)
        _restore_stop_signals(handlers)


# --------------------------------------------------------------------------
# Environment and settings
# --------------------------------------------------------------------------


def canonical_name(name):
    """A distribution name normalized as PEP 503 does."""
    return re.sub(r"[-_.]+", "-", name).lower()


def read_requirements(path):
    """The pins of a requirements file: ``{"python": str, "packages": {...}}``.

    Every requirement must pin one version with ``==``; a requirement whose
    environment marker is false here is skipped; index options
    (``--index-url``, ``--extra-index-url``, ``--find-links``) are pip's and
    are skipped. The comment ``# python==X.Y[.Z]`` pins the interpreter,
    which the build installs from conda-forge.
    """
    from packaging.requirements import InvalidRequirement, Requirement

    python, packages = None, {}
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for number, raw in enumerate(lines, start=1):
        line = raw.strip()
        pin = PYTHON_PIN.fullmatch(line)
        if pin:
            python = pin.group(1)
            continue
        line = re.split(r"\s+#", line, maxsplit=1)[0].strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            option = line.split()[0].split("=")[0]
            if option in (
                "--index-url",
                "-i",
                "--extra-index-url",
                "--find-links",
                "-f",
            ):
                continue
            raise ContractError(
                f"{path}:{number}: unsupported option {line!r}"
            )
        try:
            requirement = Requirement(line)
        except InvalidRequirement as error:
            raise ContractError(f"{path}:{number}: {error}") from error
        if (
            requirement.marker is not None
            and not requirement.marker.evaluate()
        ):
            continue
        specifiers = list(requirement.specifier)
        if (
            len(specifiers) != 1
            or specifiers[0].operator != "=="
            or "*" in specifiers[0].version
        ):
            raise ContractError(
                f"{path}:{number}: {line!r} is not pinned with =="
            )
        name = canonical_name(requirement.name)
        if name in packages:
            raise ContractError(f"{path}:{number}: {name} is pinned twice")
        packages[name] = specifiers[0].version
    return {"python": python, "packages": packages}


def installed_distributions():
    """Every distribution the interpreter sees, with how it was installed."""
    found = []
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if not name:
            continue
        direct = None
        text = dist.read_text("direct_url.json")
        if text:
            try:
                direct = json.loads(text)
            except ValueError:
                direct = {"url": "unreadable direct_url.json"}
        found.append(
            {
                "name": name,
                "version": dist.version,
                "installer": (dist.read_text("INSTALLER") or "")
                .strip()
                .lower(),
                "direct_url": direct,
                "location": str(dist.locate_file("")),
            }
        )
    return found


def _from_path(direct):
    """Whether a distribution was installed from a path, editable or not."""
    if not direct:
        return False
    return "dir_info" in direct or str(direct.get("url", "")).startswith(
        "file:"
    )


def _version_matches(installed, pinned):
    from packaging.specifiers import SpecifierSet
    from packaging.version import InvalidVersion, Version

    try:
        return SpecifierSet(f"=={pinned}").contains(
            Version(installed), prereleases=True
        )
    except InvalidVersion:
        return installed == pinned


def environment_differences(
    requirements, distributions=None, python_version=None
):
    """How the environment differs from a requirements file, if it does.

    Every pinned package must be installed once at its pinned version
    (PEP 440's ``==``, under which a pin without a local label matches an
    installed version with one), and every installed package must be
    pinned, except those installed from a path or editable (the source
    trees, such as PyVBMC and gpyreg) and those conda installed (the
    interpreter's own ``pip``, ``setuptools`` and ``wheel``). The
    interpreter must match the ``# python==`` pin to its precision.
    """
    pins = read_requirements(requirements)
    distributions = (
        installed_distributions() if distributions is None else distributions
    )
    python_version = (
        platform.python_version() if python_version is None else python_version
    )
    problems = []
    if pins["python"]:
        wanted = pins["python"].split(".")
        if python_version.split(".")[: len(wanted)] != wanted:
            problems.append(
                f"Python {python_version} runs, {pins['python']} is pinned"
            )
    by_name = {}
    for dist in distributions:
        by_name.setdefault(canonical_name(dist["name"]), []).append(dist)
    for name in sorted(by_name):
        found = by_name[name]
        copies = sorted(
            {(dist["location"], dist["version"]) for dist in found}
        )
        if len(copies) > 1:
            problems.append(
                f"{name} is installed {len(copies)} times: "
                + ", ".join(
                    f"{version} in {where}" for where, version in copies
                )
            )
            continue
        dist = found[0]
        pinned = pins["packages"].get(name)
        from_path = _from_path(dist["direct_url"])
        if pinned is not None:
            if from_path:
                problems.append(f"{name} is pinned but installed from a path")
            elif not _version_matches(dist["version"], pinned):
                problems.append(
                    f"{name} {dist['version']} is installed, "
                    f"{pinned} is pinned"
                )
        elif not from_path and dist["installer"] != "conda":
            problems.append(
                f"{name} {dist['version']} is installed but not pinned"
            )
    for name, version in sorted(pins["packages"].items()):
        if name not in by_name:
            problems.append(f"{name}=={version} is pinned but not installed")
    return problems


def pip_freeze():
    """``pip freeze --all`` of this interpreter, one line per distribution."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        raise ContractError(f"pip freeze failed: {result.stderr.strip()}")
    return result.stdout.splitlines()


def site_block(environ=None):
    """The operator settings as ``prepare`` records them (unset: null)."""
    env = os.environ if environ is None else environ
    return {name: env.get(name) or None for name in SETTINGS}


def site_differences(site, environ=None):
    """The fixed settings that differ between the environment and ``site``."""
    env = os.environ if environ is None else environ
    problems = []
    for name in FIXED_SETTINGS:
        recorded, current = site.get(name) or None, env.get(name) or None
        if name in PATH_SETTINGS and recorded and current:
            same = os.path.realpath(recorded) == os.path.realpath(current)
        else:
            same = recorded == current
        if not same:
            problems.append(
                f"{name} is {current!r} here but {recorded!r} in the manifest"
            )
    return problems


def finishing_steps(manifest):
    """The harness's finishing steps from its manifest, each a list of words.

    ``manifest["finishing_steps"]`` lists them in order, each the arguments
    of one harness invocation (the subcommand first), to which the finish
    appends ``--out DIR``. Every word must match :data:`STEP_ARGUMENT`.
    """
    steps = manifest.get("finishing_steps", [])
    if not isinstance(steps, list):
        raise ContractError("finishing_steps is not a list")
    for step in steps:
        if (
            not isinstance(step, list)
            or not step
            or not all(
                isinstance(w, str) and STEP_ARGUMENT.fullmatch(w) for w in step
            )
            or "--out" in step
        ):
            raise ContractError(
                f"the finishing step {step!r} breaks the contract"
            )
    return steps


# --------------------------------------------------------------------------
# Reconciliation
# --------------------------------------------------------------------------


def error_reason(path):
    """The first line of an error file, which names the exception."""
    try:
        lines = (
            Path(path)
            .read_text(encoding="utf-8", errors="replace")
            .splitlines()
        )
    except OSError as error:
        return f"unreadable: {error}"
    return lines[0] if lines else ""


def _listing(directory):
    try:
        return sorted(os.scandir(directory), key=lambda entry: entry.name)
    except FileNotFoundError:
        return []


def stray_files(out, tags):
    """Records, claims and error files of the directory that no case owns.

    Only the directories the allocation's tags live in are listed (and the
    top of ``records/`` and ``claims/``, for subdirectories no tag names),
    so that nothing walks a campaign directory. Dot-files are temporary
    files and are ignored; a retired claim ``<tag>.stale.<owner>`` belongs
    to its tag.
    """
    out = Path(out)
    tags = set(tags)
    groups = {str(Path(tag).parent.as_posix()) for tag in tags}
    groups = {"" if group == "." else group for group in groups}
    top_groups = {group.split("/", 1)[0] for group in groups if group}
    stray = []
    for area in (RECORDS, CLAIMS):
        for entry in _listing(out / area):
            if entry.is_dir() and entry.name not in top_groups:
                stray.append(f"{area}/{entry.name}/")
    for group in sorted(groups):
        prefix = f"{group}/" if group else ""
        for entry in _listing(out / RECORDS / group):
            name = entry.name
            if entry.is_file() and not name.startswith("."):
                tag = prefix + name[: -len(RECORD_SUFFIX)]
                if not name.endswith(RECORD_SUFFIX) or tag not in tags:
                    stray.append(f"{RECORDS}/{prefix}{name}")
        for entry in _listing(out / CLAIMS / group):
            name = entry.name
            if entry.is_file() and not name.startswith("."):
                tag = prefix + name.split(STALE_INFIX, 1)[0]
                if tag not in tags:
                    stray.append(f"{CLAIMS}/{prefix}{name}")
        for entry in _listing(out / group if group else out):
            name = entry.name
            if entry.is_file() and name.endswith(ERROR_SUFFIX):
                if prefix + name[: -len(ERROR_SUFFIX)] not in tags:
                    stray.append(f"{prefix}{name}")
    return stray


def reconcile(out, cases, check, partial_files, stray=(), query=None):
    """Place every case of the allocation, and report the directory's strays.

    Parameters
    ----------
    out : path
        The campaign directory.
    cases : sequence of str
        The case lines in allocation order; case ``i`` is line ``i``.
    check : callable
        ``check(tag)`` re-checks a case that has a completion record, the
        harness's own checks included (:func:`check_completion` among
        them), and returns the fields it adds to the case's report (a
        mapping or None); any exception fails the case's verification.
    partial_files : callable
        ``partial_files(tag)`` returns the artifact paths, relative to
        ``out``, that a case without a completion record left behind.
    stray : sequence of str
        The artifact files, relative to ``out``, that no case owns (the
        harness knows their names); the stray records, claims and error
        files are added here (:func:`stray_files`).
    query : callable, optional
        Replaces :func:`accounting_state` for the claims; each task is
        asked once.

    Returns
    -------
    dict
        The report: ``counts`` per state and ``stray``, ``cases`` (each
        with its ``index``, ``tag``, ``case`` and ``status`` and what
        explains it), ``stray`` and ``exit_code``. A case is
        ``verified`` or ``verify_failed`` when it has a record, whatever
        else it left; otherwise ``in_flight`` when its claim is live,
        which takes precedence over partial files and an error file. A
        case whose artifacts lie without a record is ``interrupted`` when
        a stale claim remains beside them (its task was killed outright,
        so that it could clean up nothing; a resubmission removes them)
        and ``partial`` when no claim does, which no path of the contract
        leaves. Otherwise a case is ``failed`` (an error file) or
        ``missing``. Each stale claim's details go with its case.
        ``exit_code`` is 1 when a state of :data:`FATAL` occurs.
    """
    query = accounting_state if query is None else query
    answers = {}

    def ask(job, array_task):
        key = (job, array_task)
        if key not in answers:
            answers[key] = query(job, array_task)
        return answers[key]

    out = Path(out)
    entries, tags = [], []
    for index, line in enumerate(cases, start=1):
        tag = case_tag(line)
        tags.append(tag)
        entry = {"index": index, "tag": tag, "case": line}
        if record_path(out, tag).exists():
            try:
                entry.update(check(tag) or {})
                entry["status"] = "verified"
            except Exception as error:
                entry["status"] = "verify_failed"
                entry["error"] = f"{type(error).__name__}: {error}"
        else:
            claim = claim_status(out, tag, query=ask)
            if claim["state"] == "live":
                entry.update(status="in_flight", claim=claim)
            else:
                files = [Path(f).as_posix() for f in partial_files(tag)]
                if files and claim["state"] == "stale":
                    entry.update(status="interrupted", files=files)
                elif files:
                    entry.update(status="partial", files=files)
                elif error_path(out, tag).exists():
                    entry.update(
                        status="failed",
                        reason=error_reason(error_path(out, tag)),
                    )
                else:
                    entry["status"] = "missing"
                if claim["state"] == "stale":
                    entry["claim"] = claim
        entries.append(entry)
    strays = sorted(set(stray) | set(stray_files(out, tags)))
    counts = {status: 0 for status in STATUSES}
    for entry in entries:
        counts[entry["status"]] += 1
    counts["stray"] = len(strays)
    return {
        "contract": CONTRACT_VERSION,
        "generated": now(),
        "counts": counts,
        "cases": entries,
        "stray": strays,
        "exit_code": 1 if any(counts[key] for key in FATAL) else 0,
    }


def finish_decision(
    report, queued=(), allow_missing=False, allow_running=False
):
    """What the finish does with a verification report: ``(code, lines)``.

    ``queued`` holds the case indices whose tasks the queue still holds;
    a missing or interrupted case among them is counted as ``queued``, in
    flight. The code is :data:`FINISH_FATAL` for a failed check, a partial
    case or a stray file; :data:`FINISH_IN_FLIGHT` for a case in flight or
    queued without ``allow_running``; :data:`FINISH_MISSING` for a missing
    or interrupted case without ``allow_missing``, the two resubmitted
    alike; and 0 to go on. ``lines`` is the summary, with the indices of
    every group but the verified one, and the indices to resubmit.
    """
    queued = {int(i) for i in queued}
    groups = {status: [] for status in (*STATUSES, "queued")}
    for case in report["cases"]:
        status = case["status"]
        if status in ("missing", "interrupted") and case["index"] in queued:
            status = "queued"
        groups[status].append(case["index"])
    stray = list(report.get("stray", []))
    lines = [
        ", ".join(
            f"{status.replace('_', ' ')} {len(groups[status])}"
            for status in (
                "verified",
                "verify_failed",
                "failed",
                "in_flight",
                "queued",
                "interrupted",
                "partial",
                "missing",
            )
        )
        + f", stray {len(stray)}"
    ]
    for status in (
        "verify_failed",
        "partial",
        "failed",
        "in_flight",
        "queued",
        "interrupted",
        "missing",
    ):
        if groups[status]:
            lines.append(
                f"{status} indices: {compress_indices(groups[status])}"
            )
    for name in stray:
        lines.append(f"stray: {name}")
    in_flight = len(groups["in_flight"]) + len(groups["queued"])
    if groups["verify_failed"] or groups["partial"] or stray:
        lines.append(
            "verification failed; fix or remove the reported cases first"
        )
        return FINISH_FATAL, lines
    if in_flight and not allow_running:
        lines.append(
            f"{in_flight} cases are in flight; wait for them, or pass "
            "--allow-running for a look at the campaign as it stands"
        )
        return FINISH_IN_FLIGHT, lines
    resubmit = sorted(groups["missing"] + groups["interrupted"])
    if resubmit and not allow_missing:
        lines.append(
            f"{len(groups['missing'])} cases are missing (their tasks never "
            "ran, or stopped and cleaned up) and "
            f"{len(groups['interrupted'])} were interrupted (their tasks "
            "were killed outright); resubmit them with "
            f"ARRAY={compress_indices(resubmit)} campaign_submit.sh, "
            "raising TIME or MEM where the accounting (slurm/sacct.txt) "
            "shows that a limit stopped them, or pass --allow-missing"
        )
        return FINISH_MISSING, lines
    return 0, lines


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def _cmd_check_env(args):
    problems = environment_differences(args.requirements)
    if problems:
        print(
            f"the environment differs from {args.requirements}:",
            file=sys.stderr,
        )
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    pins = read_requirements(args.requirements)
    print(
        f"the environment matches {args.requirements} "
        f"({len(pins['packages'])} pins, Python {platform.python_version()})"
    )
    return 0


def _cmd_check_site(args):
    manifest = read_json(args.manifest)
    if not isinstance(manifest.get("site"), dict):
        print(f"{args.manifest} holds no site block", file=sys.stderr)
        return 1
    problems = site_differences(manifest["site"])
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


def _cmd_check_cases(args):
    try:
        cases = read_cases(args.cases)
        indices = read_subset(args.subset, cases) if args.subset else None
    except (ContractError, UnicodeDecodeError) as error:
        print(error, file=sys.stderr)
        return 1
    for index in indices or ():
        print(index)
    return 0


def _cmd_finishing_steps(args):
    try:
        steps = finishing_steps(read_json(args.manifest))
    except ContractError as error:
        print(error, file=sys.stderr)
        return 1
    for step in steps:
        print(" ".join(step))
    return 0


def _cmd_finish_check(args):
    report = read_json(args.verification)
    queued = []
    if args.queued and Path(args.queued).exists():
        queued = [
            int(line.split()[0])
            for line in Path(args.queued)
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
    code, lines = finish_decision(
        report, queued, args.allow_missing, args.allow_running
    )
    for line in lines:
        print(line, file=sys.stderr)
    queued = set(queued)
    in_flight = sum(
        1
        for case in report["cases"]
        if case["status"] == "in_flight"
        or (
            case["status"] in ("missing", "interrupted")
            and case["index"] in queued
        )
    )
    print(in_flight)
    return code


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    check_env = sub.add_parser(
        "check-env", help="compare the environment with a requirements file"
    )
    check_env.add_argument("--requirements", type=Path, required=True)
    check_site = sub.add_parser(
        "check-site",
        help="compare the fixed operator settings with the manifest's "
        "site block",
    )
    check_site.add_argument("--manifest", type=Path, required=True)
    check_cases = sub.add_parser(
        "check-cases",
        help="check a case list (and a subset of it, whose indices it prints)",
    )
    check_cases.add_argument("--cases", type=Path, required=True)
    check_cases.add_argument("--subset", type=Path)
    steps = sub.add_parser(
        "finishing-steps",
        help="print the manifest's finishing steps, one per line",
    )
    steps.add_argument("--manifest", type=Path, required=True)
    finish = sub.add_parser(
        "finish-check",
        help="summarize a verification report and decide whether the finish "
        "goes on; prints the number of cases in flight",
    )
    finish.add_argument("--verification", type=Path, required=True)
    finish.add_argument(
        "--queued",
        type=Path,
        help="a file of '<case index> <state>' lines, the tasks still queued",
    )
    finish.add_argument("--allow-missing", action="store_true")
    finish.add_argument("--allow-running", action="store_true")
    return parser.parse_args(argv)


COMMANDS = {
    "check-env": _cmd_check_env,
    "check-site": _cmd_check_site,
    "check-cases": _cmd_check_cases,
    "finishing-steps": _cmd_finishing_steps,
    "finish-check": _cmd_finish_check,
}


def main(argv=None):
    # The driver reads these outputs in bash, which takes a carriage return
    # for part of a word.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(newline="\n")
        except AttributeError:
            pass
    args = parse_args(argv)
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
