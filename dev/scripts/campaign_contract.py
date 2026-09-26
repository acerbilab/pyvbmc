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
  finish's decision on its report (:func:`finish_decision`);
- **the finish's view of Slurm**: which recorded jobs the queue or the
  accounting shows may still run (:func:`queue_state`,
  :func:`accounting_problems`), and the wait for a step job
  (:func:`wait_job`);
- **the tracked copies** of a finished campaign, which its harness declares
  in the manifest (:func:`tracked_copies`), redacted for the repository
  (:func:`redact`), and read back where the analysis needs the files the
  records hash (:func:`source_sha256`, :func:`source_name`).

The layout, relative to the campaign directory ``out``::

    manifest.json                 the harness's ``prepare``
    cases.txt                     the case list; line i is case index i
    subsets/<name>.txt            a subset: "<index> <case>" per line
    records/<tag>.complete.json   completion records
    claims/<tag>                  claims; retired ones are
                                  claims/<tag>.stale.<owner>.<key>
    claims/<tag>.error.txt        an earlier attempt's error file, which
                                  the worker of a later one set aside
    <tag>.error.txt               the error file of a failed case
    slurm/                        the driver's job ids, logs, accounting
    tmp/                          TMPDIR of every process of the campaign

A retired claim's ``<owner>`` is the Slurm task that made it,
``<job>_<task>`` (``<host>-<pid>`` outside Slurm), and its ``<key>`` the
first eight hex digits of its token (:func:`retired_claim_path`).

A case line is ``<tag> [<field> ...]``: its first field, up to the first
space, is the case's tag, unique in the campaign, and the rest is the
harness's own. A tag may hold ``/``, which puts the case's files, records
and claims in one subdirectory per group of cases (a condition, a
configuration), so that no directory holds thousands of files. The
harness writes its artifacts under ``out`` as it likes; the record, claim
and error paths above are fixed, because the driver's task script reads
the record path of a case from its line alone.

The worker's exit codes: 0 for a completed case (or one that already has a
record), 1 for a case that failed, :data:`EXIT_USAGE` (64) when the
harness finds the case line or the directory is not its own (a line
outside the allocation, a directory that is not its campaign),
:data:`EXIT_CLAIMED` (75) when a live task holds the case's claim,
:data:`EXIT_IDENTITY` (78) when the worker's identity cannot be
established (a source tree that git cannot read, a host fact such as the
node's features unavailable) or its source part differs from the
manifest's, and 128 plus the signal's number (143 for SIGTERM) when a
signal stopped the run. The usage, claim and identity refusals leave the
case's files as they are.

A task that Slurm stops ends in one of two ways. At its time limit or on
``scancel`` it receives SIGTERM, and KillWait seconds later SIGKILL; the
worker catches SIGTERM (and SIGINT), removes the case's partial files and
its claim and exits, writing no error file, since a kill is no failure of
the run, and ``verify`` reports the case as ``missing``, as it does a case
that never ran. A task killed outright (SIGKILL, an out-of-memory kill, a
node lost) leaves its claim, which the accounting then shows as ended, and
whatever partial files it had written, none for a harness that writes its
artifacts at the end of a run: ``verify`` reports the case as
``interrupted``, which, like ``missing``, is resubmitted, and the worker
that takes over the stale claim removes the partial files before it runs.

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
    python dev/scripts/campaign_contract.py queue-check --slurm DIR
    python dev/scripts/campaign_contract.py archive-check --slurm DIR
    python dev/scripts/campaign_contract.py wait-job --job ID [--poll S]

and the redaction, which ``hpc/campaign_redact.sh`` runs after the finish,
and its check of files it did not write::

    python dev/scripts/campaign_contract.py redact --campaign DIR \\
        --out DIR [--path NAME=PATH ...]
    python dev/scripts/campaign_contract.py redact --campaign DIR \\
        --check FILE [FILE ...] [--path NAME=PATH ...]
"""

import argparse
import fnmatch
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import Counter
from pathlib import Path

#: The version of this contract, recorded in every identity, claim,
#: completion record and verification report.
CONTRACT_VERSION = 1
#: Exit code of a worker whose case line or campaign directory is not its
#: harness's (a line outside the allocation, a directory that is not its
#: campaign), a refusal of the harness's own that touches nothing
#: (``EX_USAGE``).
EXIT_USAGE = 64
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
#: The seconds :func:`node_features` waits before each repetition of a
#: failed ``scontrol show node``.
SCONTROL_WAITS = (2, 5, 10)
#: A Slurm node feature, as ``NODE_FEATURE`` names the campaign's family:
#: one name, no feature expression.
FEATURE_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")
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
#: How :func:`redact` treats each setting's value. The paths
#: (:data:`PATH_SETTINGS` and ``LOGIN_PROFILE``) are replaced by the
#: setting's name, ``$PYVBMC_GPYREG_SOURCE/gpyreg``; a partition is
#: ``$PARTITION``; the commands and sbatch arguments may appear nowhere,
#: whole or any word of them that holds a ``/``. The others may appear:
#: the harness is a path of this repository, the node feature is the family
#: every host of the campaign's Slurm jobs is named by, the subset is a
#: name of the harness, and the rest are numbers.
REDACTED_PATH_SETTINGS = (*PATH_SETTINGS, "LOGIN_PROFILE")
COMMAND_SETTINGS = ("LOGIN_SETUP", "CONDA_SETUP", "SBATCH_EXTRA")
PUBLIC_SETTINGS = (
    "HARNESS",
    "NODE_FEATURE",
    "CASES_SUBSET",
    "THROTTLE",
    "TIME",
    "MEM",
    "ARRAY",
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
#: The line of a requirements file that pins the interpreter, from its first
#: column; ``campaign_env.sh`` reads it with the same pattern.
PYTHON_PIN = re.compile(r"# *python==([0-9]+(?:\.[0-9]+)*) *")
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


def earlier_error_path(out, tag):
    """``<out>/claims/<tag>.error.txt``: the error file of an earlier
    attempt, which the worker of a later one sets aside when it starts."""
    return Path(out) / CLAIMS / f"{tag}{ERROR_SUFFIX}"


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


def claim_key(record):
    """Eight hex digits that tell one claim from another of the same owner.

    The first eight of the claim's token, or, for a claim that holds no
    token (one not made by :func:`new_claim`), of the SHA-256 of its JSON.
    """
    token = record.get("token")
    if isinstance(token, str) and re.fullmatch(r"[0-9a-f]{8,}", token):
        return token[:8]
    text = json.dumps(record, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def retired_claim_path(path, record):
    """``claims/<tag>.stale.<owner>.<key>``, the name the claim at ``path``
    holding ``record`` is retired to (:func:`claim_owner`,
    :func:`claim_key`)."""
    path = Path(path)
    return path.with_name(
        f"{path.name}{STALE_INFIX}{claim_owner(record)}.{claim_key(record)}"
    )


def retired_claims(out, tag):
    """The retired claims of one case, sorted by name."""
    path = claim_path(out, tag)
    prefix = f"{path.name}{STALE_INFIX}"
    return sorted(
        Path(entry.path)
        for entry in _listing(path.parent)
        if entry.is_file() and entry.name.startswith(prefix)
    )


def _retire(path, judged):
    """Rename the stale claim ``judged`` to its retired name.

    The retired name (:func:`retired_claim_path`) holds the claim's owner
    and key, so that no earlier retirement leaves it taken. The rename is
    a hard link that fails when the name exists, then the removal of the
    claim, so that of two workers that judged the same claim stale only
    one retires it. A worker whose link reached another claim than the one
    it judged (the claim changed between its read and its link) undoes the
    link and reports failure. When the retired name exists already and
    holds the judged claim, a retirement of this very claim stopped
    between its link and its removal (its worker was killed, or a network
    filesystem retried the link), and it is finished here
    (:func:`_finish_retirement`).
    """
    stale = retired_claim_path(path, judged)
    try:
        os.link(path, stale)
    except FileNotFoundError:
        return False
    except FileExistsError:
        return _finish_retirement(path, stale, judged)
    try:
        retired = read_json(stale)
    except (OSError, ValueError):
        retired = None
    if retired != judged:
        stale.unlink(missing_ok=True)
        return False
    path.unlink(missing_ok=True)
    return True


def _finish_retirement(path, stale, judged):
    """Remove the claim at ``path`` whose retired copy ``stale`` exists.

    Only when ``stale`` holds the judged claim, and only the claim file
    that holds it: the claim is first renamed to a name of this worker's
    own, which no other worker can reach, and a claim found there that is
    not the judged one (another worker's, made after the retirement) is
    linked back into place. Returns whether the judged claim is gone.
    """
    try:
        retired = read_json(stale)
    except (OSError, ValueError):
        return False
    if retired != judged:
        return False
    mine = path.with_name(f".{path.name}.{uuid.uuid4().hex}.retiring")
    try:
        os.rename(path, mine)
    except FileNotFoundError:
        return True
    try:
        taken = read_json(mine)
    except (OSError, ValueError):
        taken = None
    if taken != judged:
        try:
            os.link(mine, path)
        except FileExistsError:
            pass
        mine.unlink(missing_ok=True)
        return False
    mine.unlink(missing_ok=True)
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
    is renamed to ``claims/<tag>.stale.<owner>.<key>``
    (:func:`retired_claim_path`) before this process creates its own.
    Nothing but the claim files is touched.

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


def git(path, *args, strip=True):
    """The output of ``git -C path args``, stripped unless ``strip`` is False.

    Porcelain output keeps its leading spaces with ``strip=False``: the
    first column of ``git status --porcelain`` is the index's status, a
    space when the index holds no change.
    """
    output = subprocess.run(
        ["git", "-C", str(path), *args],
        capture_output=True,
        text=True,
        check=True,
        stdin=subprocess.DEVNULL,
    ).stdout
    return output.strip() if strip else output


def tree_state(path):
    """``(source, where)`` of one source tree, which must be a git checkout.

    ``source`` is what the identity compares, the commit and whether the
    tree is clean (``git status --porcelain`` prints nothing, so no
    uncommitted or untracked change); ``where`` is its resolved path and
    the status lines of a dirty tree as git prints them, recorded only.
    ``path`` must be the top of its checkout.
    """
    path = Path(path).resolve()
    try:
        top = git(path, "rev-parse", "--show-toplevel")
        commit = git(path, "rev-parse", "HEAD")
        status = git(path, "status", "--porcelain", strip=False)
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


def enclosing_checkout(path):
    """The top of the git checkout that holds ``path``, or None.

    The nearest of ``path`` and its parents that holds a ``.git`` entry, a
    directory in a clone and a file in a worktree or a submodule, so that
    a checkout nested inside another is told apart from it.
    """
    path = Path(path).resolve()
    for folder in (path, *path.parents):
        if (folder / ".git").exists():
            return folder
    return None


def module_origin(name, tree=None):
    """Where the package ``name`` is imported from; from ``tree`` if given.

    With ``tree``, the package must lie inside it, and the checkout that
    holds the package (:func:`enclosing_checkout`) must be the tree itself,
    not a checkout of another commit nested inside it, such as a worktree
    under its ignored ``dev/scripts/runs/``.
    """
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
        checkout = enclosing_checkout(location)
        if checkout != tree:
            raise IdentityError(
                f"{name} is imported from {location}, which lies in the "
                f"checkout {checkout} inside the tree {tree}, not in the tree "
                "itself"
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


def node_features(strict, environ=None, waits=None):
    """The features of the Slurm node this process runs on; None outside Slurm.

    From ``scontrol show node <SLURMD_NODENAME> -o`` (the short hostname
    when the variable is unset): ``{"node", "available", "active"}``. A
    query that fails is repeated after each of the ``waits``, in seconds
    (:data:`SCONTROL_WAITS` by default), so that a controller busy for a
    moment does not cost a task its case.
    """
    env = os.environ if environ is None else environ
    if not env.get("SLURM_JOB_ID"):
        return None
    node = env.get("SLURMD_NODENAME") or socket.gethostname().split(".")[0]
    waits = SCONTROL_WAITS if waits is None else waits
    failure = None
    for wait in (None, *waits):
        if wait is not None:
            time.sleep(wait)
        try:
            result = _command(["scontrol", "show", "node", node, "-o"], 60)
        except (OSError, subprocess.SubprocessError) as error:
            failure = error
            continue
        if result.returncode == 0:
            break
        failure = (result.stderr or result.stdout).strip()
    else:
        return _unavailable(
            strict,
            f"scontrol show node {node}",
            f"{failure} ({len(waits) + 1} attempts)",
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


def cpu_list(text):
    """The CPUs of a kernel CPU list such as ``0-3,8``, sorted."""
    cpus = set()
    for part in text.strip().split(","):
        if not part:
            continue
        first, _, last = part.partition("-")
        cpus.update(range(int(first), int(last or first) + 1))
    return sorted(cpus)


def job_cpuset(environ=None, proc_cgroup="/proc/self/cgroup", root=None):
    """The CPUs of the cgroup cpuset of this process's Slurm job, or None.

    The cgroup is the process's own (``/proc/self/cgroup``), cut at its
    ``job_<SLURM_JOB_ID>`` component where it has one, so that it is the
    whole job's; its cpuset is ``cpuset.cpus.effective`` under cgroup v2
    (``/sys/fs/cgroup``) and ``cpuset.effective_cpus`` or ``cpuset.cpus``
    under the cpuset controller of cgroup v1. It is where Slurm confines
    the job to the CPUs it allocated, so that it shows which hardware
    threads of a core are the job's, although a task bound to one of them
    has only that one in its affinity. Recorded where it can be read, and
    null elsewhere, on any platform.
    """
    env = os.environ if environ is None else environ
    root = Path("/sys/fs/cgroup" if root is None else root)
    try:
        lines = Path(proc_cgroup).read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    job = env.get("SLURM_JOB_ID")
    for line in lines:
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        _, controllers, path = fields
        if controllers == "":
            bases = [root, root / "unified"]
            names = ["cpuset.cpus.effective"]
        elif "cpuset" in controllers.split(","):
            bases = [root / "cpuset", root / controllers]
            names = ["cpuset.effective_cpus", "cpuset.cpus"]
        else:
            continue
        parts = [part for part in path.split("/") if part]
        if job and f"job_{job}" in parts:
            parts = parts[: parts.index(f"job_{job}") + 1]
        for base in bases:
            for name in names:
                try:
                    text = (base.joinpath(*parts) / name).read_text()
                except OSError:
                    continue
                try:
                    return cpu_list(text) or None
                except ValueError:
                    return None
    return None


def cpu_affinity(
    strict,
    getter=None,
    topology="/sys/devices/system/cpu",
    environ=None,
    cpuset=None,
):
    """The CPU affinity of this process, with the cores it lies on.

    ``cpus`` is ``os.sched_getaffinity(0)`` (``getter(0)`` in its place);
    ``physical_cores`` holds one ``<package>:<core>`` per physical core
    that the CPUs belong to, and ``core_threads`` maps each of them to the
    hardware threads of that core (``topology/core_cpus_list``, or
    ``thread_siblings_list`` on an older kernel), all from ``topology``,
    ``/sys/devices/system/cpu``; ``job_cpuset`` is :func:`job_cpuset`
    (``cpuset()`` in its place). A record read on another machine thus
    still shows whether its task had one physical core to itself
    (:func:`host_problems`). None where the platform has no affinity call.
    """
    getter = (
        getattr(os, "sched_getaffinity", None) if getter is None else getter
    )
    if getter is None:
        return _unavailable(strict, "os.sched_getaffinity")
    cpus = sorted(getter(0))
    cores, threads = set(), {}
    for cpu in cpus:
        folder = Path(topology) / f"cpu{cpu}" / "topology"
        try:
            package = (folder / "physical_package_id").read_text().strip()
            core = (folder / "core_id").read_text().strip()
            name = f"{package}:{core}"
            if name not in threads:
                siblings = folder / "core_cpus_list"
                if not siblings.is_file():
                    siblings = folder / "thread_siblings_list"
                threads[name] = cpu_list(siblings.read_text())
        except (OSError, ValueError) as error:
            _unavailable(strict, f"the topology of CPU {cpu}", error)
            return {
                "cpus": cpus,
                "physical_cores": None,
                "core_threads": None,
                "job_cpuset": None,
            }
        cores.add(name)
    return {
        "cpus": cpus,
        "physical_cores": sorted(cores),
        "core_threads": dict(sorted(threads.items())),
        "job_cpuset": (job_cpuset(environ) if cpuset is None else cpuset()),
    }


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
        "cpu_affinity": cpu_affinity(strict, environ=env),
        "threads": {key: env.get(key) for key in THREAD_KEYS},
        "slurm": slurm_ids(env),
    }


def host_problems(host, node_feature):
    """Why a record's host part does not show the campaign's node and core.

    The node's features must include ``node_feature``, and the task must
    have had one physical core to itself: its CPU affinity lies on one
    physical core, and the affinity or the job's cpuset holds exactly the
    hardware threads of that core, so that no other job could run on the
    core's other threads. Where the core has one thread (no SMT), the
    affinity is that thread. Returns the problems, empty when there are
    none.
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
    else:
        [core] = affinity["physical_cores"]
        threads = (affinity.get("core_threads") or {}).get(core)
        cpus = affinity.get("cpus") or []
        cpuset = affinity.get("job_cpuset")
        if not threads:
            problems.append(
                f"the record holds no hardware threads of core {core}"
            )
        elif set(cpus) != set(threads) and set(cpuset or ()) != set(threads):
            job = (
                "the record holds no cpuset of the job"
                if cpuset is None
                else f"the job's cpuset {cpuset} is not those threads either"
            )
            problems.append(
                f"the CPU affinity {cpus} is not the hardware threads "
                f"{threads} of core {core}, and {job}: another job may have "
                "run on the core"
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


def _say_safely(message):
    """:func:`_say`, where an output that cannot be written costs nothing."""
    try:
        _say(message)
    except (OSError, ValueError):
        pass


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
        :data:`STOP_SIGNALS` stopped it, even after its completion record.
        The two refusals touch none of the case's files. From the claim to
        its release, the stop signals raise
        :class:`Interrupted` (the previous handlers are restored after).
        Before the run, the case's partial files are removed and an
        earlier attempt's ``<tag>.error.txt`` is set aside as
        ``claims/<tag>.error.txt`` (:func:`earlier_error_path`). A failure
        removes the partial files, writes ``<tag>.error.txt`` with the
        traceback and removes the set-aside file and the claim; a stop
        removes the partial files and the claim, writes no error file and
        keeps the set-aside one, so that ``verify`` reports the case as
        missing, with the earlier attempt's error; a success writes the
        completion record, removes both error files and then the claim.
        Whatever happens once the completion record is written, a stop
        signal or an exception (the removal of an error file, a message
        whose output fails), leaves the case complete with every file it
        wrote. While the clean-up of a failure or a stop runs, and while
        the claim is released, further stop signals are ignored.
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
        earlier = earlier_error_path(out, tag)
        try:
            removed = _remove(out, partial_files())
            if removed:
                _say(
                    f"{tag}: removed {len(removed)} files of an earlier "
                    "attempt before the run"
                )
            if error_path(out, tag).exists():
                earlier.parent.mkdir(parents=True, exist_ok=True)
                os.replace(error_path(out, tag), earlier)
                _say(
                    f"{tag}: set the error file of an earlier attempt aside "
                    f"as {earlier}"
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
            # No stop signal may interrupt what follows a complete case.
            _handle_stop_signals(signal.SIG_IGN)
            error_path(out, tag).unlink(missing_ok=True)
            earlier.unlink(missing_ok=True)
            _say(f"{tag}: complete in {time.time() - started:.1f} s")
        except Interrupted as stop:
            _handle_stop_signals(signal.SIG_IGN)
            name = signal.Signals(stop.signum).name
            if record.exists():
                _say_safely(f"{tag}: {name} came after the completion record")
                return 128 + stop.signum
            try:
                removed = _remove(out, partial_files())
            except Exception:
                removed = []
                _say_safely(
                    "removing the partial files failed:\n"
                    + traceback.format_exc()
                )
            _say_safely(
                f"{tag}: stopped by {name}; removed {len(removed)} partial "
                "files and the claim, and left the case to be resubmitted"
            )
            return 128 + stop.signum
        except Exception as error:
            _handle_stop_signals(signal.SIG_IGN)
            text = traceback.format_exc()
            if record.exists():
                # The case is complete: what failed came after its record
                # (the removal of an earlier error file, a message), and
                # every file of the case stays.
                _say_safely(
                    f"{tag}: complete, then {type(error).__name__}: {error}; "
                    "the case keeps its files\n" + text
                )
                return 0
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
            earlier.unlink(missing_ok=True)
            _say_safely(text)
            _say_safely(
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
    are skipped. The comment ``# python==X.Y[.Z]``, a line of its own from
    the first column (:data:`PYTHON_PIN`), pins the interpreter, which the
    build installs from conda-forge; a file may hold one such line.
    """
    from packaging.requirements import InvalidRequirement, Requirement

    python, packages = None, {}
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for number, raw in enumerate(lines, start=1):
        pin = PYTHON_PIN.fullmatch(raw)
        if pin:
            if python is not None:
                raise ContractError(
                    f"{path}:{number}: a second '# python==' line"
                )
            python = pin.group(1)
            continue
        line = raw.strip()
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


def check_feature(value):
    """``value`` if it is one node feature (:data:`FEATURE_PATTERN`)."""
    if not isinstance(value, str) or not FEATURE_PATTERN.fullmatch(value):
        raise ContractError(
            f"NODE_FEATURE={value!r} is not one node feature: letters, "
            "digits and _.- alone"
        )
    return value


def site_block(environ=None):
    """The operator settings as ``prepare`` records them (unset: null).

    A ``NODE_FEATURE`` that is not one feature name raises
    :class:`ContractError` (:func:`check_feature`).
    """
    env = os.environ if environ is None else environ
    site = {name: env.get(name) or None for name in SETTINGS}
    if site["NODE_FEATURE"] is not None:
        check_feature(site["NODE_FEATURE"])
    return site


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
    files and are ignored; a retired claim ``<tag>.stale.<owner>.<key>``
    and a set-aside error file ``<tag>.error.txt`` under ``claims/`` belong
    to their tag.
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
                owners = {
                    prefix + name,
                    prefix + name.split(STALE_INFIX, 1)[0],
                }
                if name.endswith(ERROR_SUFFIX):
                    owners.add(prefix + name[: -len(ERROR_SUFFIX)])
                if not owners & tags:
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
        else it left. Without a record, a case is ``in_flight`` when its
        claim is live and ``interrupted`` when its claim is stale: its task
        was killed outright and could clean up nothing, and the case's
        ``files`` lists what it left (none, for a harness that writes its
        artifacts at the end of a run), which a resubmission removes; the
        claim's details, and an earlier attempt's error, go with it. A
        case without a record or a claim is ``partial`` when artifacts
        remain, which no path of the contract leaves, ``failed`` when its
        error file does, and ``missing`` otherwise: it never ran, or a
        stop signal ended it and it removed its files and its claim. An
        interrupted or missing case whose earlier attempt failed carries
        that attempt's reason as ``earlier_error``, from its error file or
        from the copy a later worker set aside (:func:`earlier_error_path`).
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
            error = error_path(out, tag)
            earlier = next(
                (
                    path
                    for path in (error, earlier_error_path(out, tag))
                    if path.exists()
                ),
                None,
            )
            if claim["state"] == "live":
                entry.update(status="in_flight", claim=claim)
            elif claim["state"] == "stale":
                files = [Path(f).as_posix() for f in partial_files(tag)]
                entry.update(status="interrupted", files=files, claim=claim)
                if earlier is not None:
                    entry["earlier_error"] = error_reason(earlier)
            else:
                files = [Path(f).as_posix() for f in partial_files(tag)]
                if files:
                    entry.update(status="partial", files=files)
                elif error.exists():
                    entry.update(status="failed", reason=error_reason(error))
                else:
                    entry["status"] = "missing"
                    if earlier is not None:
                        entry["earlier_error"] = error_reason(earlier)
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
            "ran, or a stop signal ended them and they cleaned up) and "
            f"{len(groups['interrupted'])} were interrupted (their tasks "
            "were killed outright and left their claims); resubmit them with "
            f"ARRAY={compress_indices(resubmit)} campaign_submit.sh, "
            "raising TIME or MEM where the accounting (slurm/sacct.txt) "
            "shows that a limit stopped them, or pass --allow-missing"
        )
        return FINISH_MISSING, lines
    return 0, lines


# --------------------------------------------------------------------------
# The queue and the accounting, for the finish
# --------------------------------------------------------------------------

#: The files of ``slurm/`` where the driver records its jobs: the array
#: submissions and the finish's step jobs.
JOB_FILES = ("jobs.txt", "steps.txt")
#: What ``squeue`` prints when it does not know a job id, which it does for
#: a job the controller has dropped as well as for one it never held.
SQUEUE_UNKNOWN_JOB = "Invalid job id"


def recorded_jobs(slurm_dir):
    """The jobs that ``slurm/jobs.txt`` and ``slurm/steps.txt`` record.

    Each line of those files starts with a job id (lines that do not, such
    as a step whose submission failed, are skipped), and a submission's
    first ``offset=<n>`` field is its index offset. Returns ``[{"job",
    "offset"}]``, one per job in the order first recorded, ``offset`` None
    for a job that names none (a step job).
    """
    jobs = {}
    for name in JOB_FILES:
        path = Path(slurm_dir) / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if not fields or not fields[0].isdigit():
                continue
            offset = next(
                (
                    int(f[len("offset=") :])
                    for f in fields[1:]
                    if f.startswith("offset=")
                    and f[len("offset=") :].isdigit()
                ),
                None,
            )
            entry = jobs.setdefault(
                fields[0], {"job": fields[0], "offset": None}
            )
            if entry["offset"] is None:
                entry["offset"] = offset
    return list(jobs.values())


def _state_word(text):
    """The state of a Slurm answer's field: ``CANCELLED by 5`` is
    ``CANCELLED``, ``RUNNING+`` is ``RUNNING``."""
    words = text.split()
    return words[0].rstrip("+").upper() if words else ""


def accounting_rows(slurm_dir):
    """The rows of ``slurm/sacct.txt`` by job: ``{job: [(JobID, state)]}``.

    None when the file is absent (the finish removes it when ``sacct``
    fails). A row belongs to the job its JobID starts with (``1001_3``,
    ``1001_3.batch``, ``1001_[4-9]``, ``1005.extern``).
    """
    path = Path(slurm_dir) / "sacct.txt"
    if not path.is_file():
        return None
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header = lines[0].split("|") if lines else []
    if "JobID" not in header or "State" not in header:
        return {}
    job_column, state_column = header.index("JobID"), header.index("State")
    rows = {}
    for line in lines[1:]:
        fields = line.split("|")
        if len(fields) <= max(job_column, state_column):
            continue
        match = re.match(r"\d+", fields[job_column])
        if match:
            rows.setdefault(match.group(0), []).append(
                (fields[job_column], _state_word(fields[state_column]))
            )
    return rows


def squeue_tasks(job, timeout=60):
    """What ``squeue -h -r -j <job> -o "%i %T"`` says of one job.

    Returns ``{"ok": bool, "tasks": [(task id, state)], "detail": str}``;
    ``ok`` is False when the query fails, which ``squeue`` does for a job
    the controller has dropped as for a controller it cannot reach.
    """
    try:
        result = _command(
            ["squeue", "-h", "-r", "-j", str(job), "-o", "%i %T"], timeout
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"ok": False, "tasks": [], "detail": f"squeue failed: {error}"}
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip()
        return {
            "ok": False,
            "tasks": [],
            "detail": f"squeue exited {result.returncode}: {message}",
        }
    tasks = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 2:
            tasks.append((fields[0], _state_word(" ".join(fields[1:]))))
    return {"ok": True, "tasks": tasks, "detail": ""}


def queue_state(slurm_dir, query=None):
    """Which recorded jobs may still run, from the queue and the accounting.

    Every job of :func:`recorded_jobs` is asked of ``squeue``
    (:func:`squeue_tasks`, or ``query`` with its signature). A task it
    lists in any state but one of :data:`ENDED_STATES` (``squeue`` lists
    jobs that ended moments ago) is ``live``. A job it cannot answer for
    is resolved by the accounting of ``slurm/sacct.txt``
    (:func:`accounting_rows`): ended when it holds rows of the job and all
    of them have ended, and ``unknown`` otherwise, since a failed query
    shows nothing.

    Returns ``{"live": [(job, task, state)], "cases": [(case index,
    state)], "unknown": [(job, detail)]}``, ``cases`` the live array tasks
    mapped to case indices by their submission's offset.
    """
    query = squeue_tasks if query is None else query
    accounting = accounting_rows(slurm_dir)
    live, cases, unknown = [], [], []
    for entry in recorded_jobs(slurm_dir):
        job = entry["job"]
        answer = query(job)
        if not answer["ok"]:
            rows = (accounting or {}).get(job) or []
            if not rows or any(state not in ENDED_STATES for _, state in rows):
                where = (
                    "no accounting (slurm/sacct.txt)"
                    if accounting is None
                    else "the accounting "
                    + (
                        "does not show its tasks ended"
                        if rows
                        else "holds no row of it"
                    )
                )
                unknown.append((job, f"{answer['detail']}; {where}"))
            continue
        for task, state in answer["tasks"]:
            if state in ENDED_STATES:
                continue
            live.append((job, task, state))
            head, _, index = task.partition("_")
            if entry["offset"] is not None and head == job and index.isdigit():
                cases.append((int(index) + entry["offset"], state))
    return {"live": live, "cases": cases, "unknown": unknown}


def accounting_problems(slurm_dir):
    """Why the accounting does not show every recorded task ended.

    Each row of ``slurm/sacct.txt`` of a recorded job whose state is not
    one of :data:`ENDED_STATES`, and the file's absence when any job is
    recorded. Empty when there is nothing to hold the archive back.
    """
    jobs = [entry["job"] for entry in recorded_jobs(slurm_dir)]
    if not jobs:
        return []
    rows = accounting_rows(slurm_dir)
    if rows is None:
        return [
            "the accounting of the recorded jobs (slurm/sacct.txt) is "
            "missing: sacct failed"
        ]
    return [
        f"{job_id} is {state} in slurm/sacct.txt"
        for job in jobs
        for job_id, state in rows.get(job, [])
        if state not in ENDED_STATES
    ]


def job_exit(job, timeout=60):
    """``(state, exit code)`` of a job from the accounting.

    Runs ``sacct -n -X -P -j <job> -o State,ExitCode``. The code is None
    while the job may still run (any state but one of
    :data:`ENDED_STATES`, a failed query, an empty answer). For an ended
    job it is the batch script's exit status, 128 plus the signal's number
    when a signal ended it, and 1 for a job that ended in any state but
    ``COMPLETED`` with a status of 0 (cancelled before it started, say).
    """
    try:
        result = _command(
            [
                "sacct",
                "-n",
                "-X",
                "-P",
                "-j",
                str(job),
                "-o",
                "State,ExitCode",
            ],
            timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if result.returncode != 0 or not lines:
        return None, None
    state_text, _, exit_text = lines[0].partition("|")
    state = _state_word(state_text)
    if state not in ENDED_STATES:
        return state, None
    status, _, signum = exit_text.strip().partition(":")
    try:
        status, signum = int(status or 0), int(signum or 0)
    except ValueError:
        status, signum = 1, 0
    code = 128 + signum if signum else status
    if code == 0 and state != "COMPLETED":
        code = 1
    return state, code


def wait_job(job, poll=30.0, say=None, query=None):
    """Wait until a job has ended in the accounting; return its exit code.

    ``query`` replaces :func:`job_exit`; each change of the job's state is
    reported with ``say`` (:func:`print` by default).
    """
    say = _say if say is None else say
    query = job_exit if query is None else query
    last = ()
    while True:
        state, code = query(job)
        if state != last:
            say(
                f"job {job}: {state or 'not in the accounting yet'}"
                + ("" if code is None else f", exit {code}")
            )
            last = state
        if code is not None:
            return code
        time.sleep(poll)


# --------------------------------------------------------------------------
# Tracked copies
# --------------------------------------------------------------------------

#: The files every campaign's tracked copies hold, beside those its harness
#: declares (:func:`tracked_copies`).
TRACKED_ALWAYS = ("manifest.json", "verification.json")
#: The record :func:`redact` writes beside the copies it makes.
REDACTION = "redaction.json"
#: The parts of a campaign directory that no tracked copy comes from: the
#: task logs and the Slurm accounting, the claims, and TMPDIR.
UNTRACKED = ("slurm", CLAIMS, "tmp")
#: What names, in the tracked copies, a host that ran none of the
#: campaign's Slurm jobs: the login node, where ``prepare``, the driver's
#: scripts and :func:`redact` run.
LOGIN_HOST = "login"
_GLOB = re.compile(r"[*?\[]")


def _tracked_path(path, what, glob_last=False):
    """Check one path or pattern of a declaration; return it."""
    if not isinstance(path, str) or not path:
        raise ContractError(f"{what}: {path!r} is not a path")
    parts = path.split("/")
    if (
        "\\" in path
        or path.startswith("/")
        or re.match(r"[A-Za-z]:", path)
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise ContractError(
            f"{what}: {path!r} is not a path relative to the campaign "
            "directory"
        )
    if parts[0] in UNTRACKED or path == REDACTION:
        raise ContractError(
            f"{what}: {path!r} lies in what the tracked copies leave in the "
            f"archive ({', '.join(UNTRACKED)}) or is the redaction's record"
        )
    if glob_last and any(_GLOB.search(part) for part in parts[:-1]):
        raise ContractError(
            f"{what}: {path!r} may hold a pattern in its last component alone"
        )
    return path


def tracked_copies(manifest):
    """The harness's declaration of its tracked copies, checked.

    ``manifest["tracked_copies"]`` names what of a finished campaign enters
    the repository (``dev/plans/slurm-benchmark-support.md``, "Records and
    hand-back"), beside the manifest and the verification report, which
    every campaign's copies hold (:data:`TRACKED_ALWAYS`):

    - ``files``: paths relative to the campaign directory, whose last
      component may be a glob pattern (``rescored/*.json``); a path
      without a pattern must exist in a finished campaign;
    - ``cases``: what each case that the verification report places as
      verified contributes: its completion record (``"record": true``) and
      those of its artifacts whose paths, as the record lists them, match
      one of the glob patterns of ``artifacts``.

    Nothing declared may lie under :data:`UNTRACKED` or be named
    :data:`REDACTION`.

    Returns
    -------
    dict
        ``{"files": [...], "cases": {"record": bool, "artifacts": [...]}}``.
    """
    spec = manifest.get("tracked_copies")
    if not isinstance(spec, dict):
        raise ContractError(
            "the manifest declares no tracked copies (tracked_copies); its "
            "harness names them at prepare"
        )
    unknown = sorted(set(spec) - {"files", "cases"})
    cases = spec.get("cases") or {}
    if unknown or not isinstance(cases, dict):
        raise ContractError(f"tracked_copies holds what it may not: {spec!r}")
    files = spec.get("files") or []
    artifacts = cases.get("artifacts") or []
    if (
        not isinstance(files, list)
        or not isinstance(artifacts, list)
        or set(cases) - {"record", "artifacts"}
        or not isinstance(cases.get("record", False), bool)
    ):
        raise ContractError(f"tracked_copies holds what it may not: {spec!r}")
    return {
        "files": [
            _tracked_path(p, "tracked_copies.files", glob_last=True)
            for p in files
        ],
        "cases": {
            "record": cases.get("record", False),
            "artifacts": [
                _tracked_path(p, "tracked_copies.cases.artifacts")
                for p in artifacts
            ],
        },
    }


def tracked_files(campaign, manifest, verification, records):
    """The files of a finished campaign that its tracked copies hold.

    ``records`` maps the tag of every case the verification report places
    as verified to its completion record. Returns the paths relative to the
    campaign directory, sorted; a declared path without a pattern that the
    directory does not hold raises :class:`ContractError`.
    """
    campaign = Path(campaign)
    spec = tracked_copies(manifest)
    found = set(TRACKED_ALWAYS)
    for pattern in spec["files"]:
        parent, _, name = pattern.rpartition("/")
        if _GLOB.search(name):
            for entry in _listing(campaign / parent if parent else campaign):
                if (
                    entry.is_file()
                    and not entry.name.startswith(".")
                    and fnmatch.fnmatchcase(entry.name, name)
                ):
                    found.add(
                        f"{parent}/{entry.name}" if parent else entry.name
                    )
        elif (campaign / pattern).is_file():
            found.add(pattern)
        else:
            raise ContractError(
                f"{campaign} holds no {pattern}, which its tracked copies "
                "declare; the finish writes it"
            )
    for case in verification.get("cases", []):
        if case.get("status") != "verified":
            continue
        tag = case["tag"]
        if spec["cases"]["record"]:
            found.add(f"{RECORDS}/{tag}{RECORD_SUFFIX}")
        for name in records[tag].get("artifacts") or {}:
            if any(
                fnmatch.fnmatchcase(name, pattern)
                for pattern in spec["cases"]["artifacts"]
            ):
                found.add(_tracked_path(name, f"an artifact of {tag}"))
    return sorted(found)


def read_redaction(directory):
    """The :data:`REDACTION` record of a directory of tracked copies; None
    for any other directory."""
    path = Path(directory) / REDACTION
    return read_json(path) if path.is_file() else None


def source_sha256(directory, relative):
    """The SHA-256 of a campaign's file as the campaign wrote it.

    In a campaign directory it is the file's own. In a directory of tracked
    copies that :func:`redact` wrote, the file must be the copy that
    :data:`REDACTION` records, and the one returned is the SHA-256 of the
    file it was made from, which the campaign's records and reports hash;
    a file the record does not list, or one changed since, raises
    :class:`ContractError`. A copy is the recorded one when its SHA-256 is
    the recorded one, or when it is once its CRLF line endings are read as
    LF, as git may check out a text file on Windows.
    """
    directory = Path(directory)
    redaction = read_redaction(directory)
    actual = sha256_file(directory / relative)
    if redaction is None:
        return actual
    entry = redaction["files"].get(Path(relative).as_posix())
    if entry is not None and entry["sha256"] != actual:
        data = (directory / relative).read_bytes()
        if b"\r\n" in data:
            actual = hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()
    if entry is None or entry["sha256"] != actual:
        raise ContractError(
            f"{directory / relative} is not the copy that "
            f"{directory / REDACTION} records"
        )
    return entry["source_sha256"]


def source_name(directory):
    """The name of the campaign directory that a directory is or copies."""
    redaction = read_redaction(directory)
    if redaction is None:
        return Path(directory).resolve().name
    return redaction["campaign"]


# --------------------------------------------------------------------------
# Redaction
# --------------------------------------------------------------------------

#: The lines of a campaign's logs that name a host: the two that
#: ``hpc/campaign_task.sbatch`` prints for a task and for a step, and the
#: one Slurm writes into a job's output when it stops the job.
LOG_HOST_LINES = (
    re.compile(r"^task \S+, case \d+: .*, on (\S+)\r?$", re.MULTILINE),
    re.compile(r"^step '.*' of .*: job \S+ on (\S+)\r?$", re.MULTILINE),
    re.compile(r"\*\*\* (?:JOB|STEP) \S+ ON (\S+) CANCELLED"),
)
#: The most occurrences of forbidden strings reported for one file.
MAX_LEAKS = 100
_HOST_WORD = "A-Za-z0-9"
#: The characters of a path component (``=`` aside, which in a copy mostly
#: joins a name to its value, ``X=/path``).
_PATH_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.+-"
)
#: The characters after a path that continue its last component, so that
#: the redaction does not replace the head of a longer name
#: (``/proj/envs`` for ``/proj/env``); a dot continues it when a character
#: of this class follows it (``/proj/env.old``), and ends a sentence
#: otherwise.
_PATH_AFTER = "A-Za-z0-9_+=-"
_HEX = frozenset("0123456789abcdef")
#: A run of this many lower-case hex digits or more is a digest or a token
#: (the SHA-256 of a file, a claim's token), whose characters are random:
#: the check does not read a username or a hostname into it.
DIGEST_LENGTH = 32
#: The directories an absolute path in a copy may lie in without a name,
#: those of the operating system: its programs and libraries, its
#: configuration, its kernel interfaces and its temporary directories, and
#: the conda of a container image.
SYSTEM_PREFIXES = (
    "/bin",
    "/dev",
    "/etc",
    "/lib",
    "/lib64",
    "/opt/conda",
    "/proc",
    "/sbin",
    "/sys",
    "/tmp",
    "/usr",
    "/var/tmp",
)
#: What replaces a partition in a field that holds one.
PARTITION_TOKEN = "$PARTITION"
#: The name of the directory that holds the campaign directory, where no
#: other name covers it.
CAMPAIGN_PARENT = "CAMPAIGN_PARENT"


def _trie(words):
    """A regular expression for any of ``words``, built as a trie, so that
    a search stays fast however many words there are; None for none."""
    root = {}
    for word in words:
        node = root
        for character in word:
            node = node.setdefault(character, {})
        node[None] = None

    def pattern(node):
        branches = [
            re.escape(character) + pattern(node[character])
            for character in sorted(c for c in node if c is not None)
        ]
        if not branches:
            return ""
        if len(branches) == 1 and None not in node:
            return branches[0]
        return "(?:" + "|".join(branches) + ")" + ("?" if None in node else "")

    return pattern(root) if root else None


def _bounded(words, word, flags=0):
    """``words`` matched where no character of the class ``word`` flanks
    them; None for no word."""
    body = _trie(words)
    if body is None:
        return None
    return re.compile(rf"(?<![{word}])(?:{body})(?![{word}])", flags)


def _paths_pattern(paths):
    """The directories ``paths``, the longest that fits first, each where
    the character after it does not continue its last component
    (:data:`_PATH_AFTER`); None for none."""
    body = _trie(paths)
    if body is None:
        return None
    return re.compile(rf"(?:{body})(?![{_PATH_AFTER}]|\.[{_PATH_AFTER}])")


def _continues_a_path(text, start):
    """Whether what starts at ``start`` of ``text`` continues a path.

    It does when the characters just before it are those of a path
    component (:data:`_PATH_CHARS`), as ``b`` in ``/a/b/proj/env`` or in
    ``b/proj/env`` is before ``/proj/env``, unless they are a flag such as
    the ``-L`` of ``-L/proj/env``. A path after ``=``, ``:``, a quote, a
    space or a separator (``file:///proj/env``) does not.
    """
    index = start
    while index > 0 and text[index - 1] in _PATH_CHARS:
        index -= 1
    run = text[index:start]
    return bool(run) and not re.fullmatch(r"-[A-Za-z]+", run)


def absolute_paths(text):
    """The absolute paths a text names: ``[(offset, path)]``.

    A POSIX path starts at a ``/`` followed by a character of a path
    component and does not continue a path (:func:`_continues_a_path`) or
    follow ``~`` or another separator, the ``file://`` of a URL aside; a
    Windows path starts at a drive letter that no letter, digit or
    separator precedes, then a colon and a separator. Each ends at a space,
    a quote, a colon or another character no path of a copy holds.
    """
    found = []
    end = 0
    for match in _PATH_START.finditer(text):
        start = match.start()
        if start < end:
            continue
        if text[start] == "/":
            before = text[start - 1] if start else ""
            if before in ("~", "\\") or (
                before == "/" and not text.endswith("file://", 0, start)
            ):
                continue
            if _continues_a_path(text, start):
                continue
            extent = _PATH_EXTENT.match(text, start + 1)
        else:
            extent = _PATH_EXTENT.match(text, start + 2)
        end = extent.end()
        found.append((start, text[start:end]))
    return found


#: Where an absolute path may start, and the characters that end one, in a
#: copy's text.
_PATH_START = re.compile(
    r"/(?=[A-Za-z0-9_.+-])|(?<![A-Za-z0-9/\\])[A-Za-z](?=:[\\/])"
)
_PATH_EXTENT = re.compile(r"[^\s\"'<>|;,()\[\]{}`*?:]*")


def _system_path(path):
    """Whether an absolute path lies in one of :data:`SYSTEM_PREFIXES`."""
    return any(
        path == prefix or path.startswith(f"{prefix}/")
        for prefix in SYSTEM_PREFIXES
    )


def _inside_digest(text, start, length):
    """Whether ``text[start:start + length]`` lies inside a run of at least
    :data:`DIGEST_LENGTH` lower-case hex digits."""
    end = start + length
    if any(character not in _HEX for character in text[start:end]):
        return False
    while start > 0 and text[start - 1] in _HEX:
        start -= 1
    while end < len(text) and text[end] in _HEX:
        end += 1
    return end - start >= DIGEST_LENGTH


def _is_root(path):
    """Whether a path is a filesystem's root (``/``, ``C:\\``)."""
    stripped = str(path).rstrip("/\\")
    return not stripped or re.fullmatch(r"[A-Za-z]:", stripped) is not None


def path_variants(value):
    """The forms a path may take in a campaign's records.

    As given, with ``~`` expanded, and as it resolves on this machine, each
    without a trailing separator, and on Windows with either separator,
    with the JSON-escaped form of each; a filesystem's root is none.
    """
    value = str(value)
    forms = {value, os.path.expanduser(value)}
    for form in list(forms):
        try:
            forms.add(os.path.realpath(form))
        except (OSError, ValueError):
            pass
    if sys.platform == "win32":
        forms |= {form.replace("\\", "/") for form in forms}
    forms = {form.rstrip("/\\") for form in forms if not _is_root(form)}
    forms |= {json.dumps(form)[1:-1] for form in forms}
    return {form for form in forms if len(form) > 1}


def operator_identity(environ=None, passwd=None):
    """The username and home of the account this process runs in.

    From the environment (``USER``, ``LOGNAME``, ``USERNAME``; ``HOME``,
    ``USERPROFILE``) and the password database (``passwd``, an entry of
    :mod:`pwd`, this process's own by default where the platform has one),
    never from a file of the campaign. Returns ``{"users": [...],
    "homes": [...]}``; a home that is a filesystem's root raises
    :class:`ContractError`, since the redaction would take every path for
    one under it.
    """
    env = os.environ if environ is None else environ
    users = {env.get(k) for k in ("USER", "LOGNAME", "USERNAME")}
    homes = {env.get(k) for k in ("HOME", "USERPROFILE")}
    if passwd is None:
        try:
            import pwd

            passwd = pwd.getpwuid(os.getuid())
        except (ImportError, KeyError, AttributeError):
            passwd = None
    if passwd is not None:
        users.add(passwd.pw_name)
        homes.add(passwd.pw_dir)
    users = sorted(u for u in users if u)
    homes = sorted(h for h in homes if h)
    rooted = [h for h in homes if _is_root(h)]
    if rooted:
        raise ContractError(
            f"the home directory {rooted[0]!r} is a filesystem's root"
        )
    if not users or not homes:
        raise ContractError(
            "this process names no username or no home directory (USER, "
            "HOME, the password database), which the redaction removes"
        )
    return {"users": users, "homes": homes}


def expand_hostlist(text):
    """The hosts of a Slurm host list: ``node[01-03,7],login1``.

    The accounting's ``None assigned`` and ``(null)``, of a job that never
    started, name none.
    """
    if text.strip() in ("", "(null)") or text.strip().startswith("None"):
        return []
    hosts = []
    depth, start = 0, 0
    items = []
    for index, character in enumerate(text):
        depth += {"[": 1, "]": -1}.get(character, 0)
        if character == "," and depth == 0:
            items.append(text[start:index])
            start = index + 1
    items.append(text[start:])
    for item in (i.strip() for i in items):
        match = re.match(r"([^\[]*)\[([^\]]*)\](.*)$", item)
        if not match:
            if item:
                hosts.append(item)
            continue
        prefix, ranges, rest = match.groups()
        for part in ranges.split(","):
            first, _, last = part.partition("-")
            if not first.isdigit() or (last and not last.isdigit()):
                raise ContractError(f"{text!r} is not a Slurm host list")
            for number in range(int(first), int(last or first) + 1):
                hosts += expand_hostlist(
                    f"{prefix}{number:0{len(first)}d}{rest}"
                )
    return hosts


def _dicts(value):
    """Every mapping in a JSON value, its own included."""
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _dicts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _dicts(item)


class Hosts:
    """The hosts a campaign names, and whether each ran a Slurm job of it."""

    def __init__(self):
        self.in_slurm = {}

    def add(self, name, in_slurm):
        """Add a hostname, and its short name if it is a domain name."""
        if not isinstance(name, str) or not name.strip():
            return
        names = {name.strip()}
        short = name.strip().split(".", 1)[0]
        if short and not re.fullmatch(r"[0-9.]+", name.strip()):
            names.add(short)
        for each in names:
            key = each.lower()
            self.in_slurm[key] = self.in_slurm.get(key, False) or in_slurm

    def add_documents(self, documents, node_feature):
        """The hosts that the JSON ``documents`` hold.

        A host part (a mapping with ``hostname`` and ``slurm`` or
        ``node_features``) ran in Slurm when it holds a job id or node
        features, which must then include ``node_feature``; any other
        mapping's ``hostname`` or ``host`` ran in Slurm when the mapping
        names a job. Raises :class:`ContractError` for a host whose recorded
        features lack the campaign's.
        """
        for document in documents:
            for mapping in _dicts(document):
                if isinstance(mapping.get("hostname"), str) and (
                    "slurm" in mapping or "node_features" in mapping
                ):
                    slurm = mapping.get("slurm") or {}
                    features = mapping.get("node_features") or {}
                    if features:
                        found = set(features.get("available") or []) | set(
                            features.get("active") or []
                        )
                        if node_feature not in found:
                            raise ContractError(
                                f"a record of {mapping['hostname']} lists the "
                                f"node features {sorted(found)}, not "
                                f"{node_feature}, the family its copies "
                                "would name it by"
                            )
                    in_slurm = bool(slurm.get("job_id") or features)
                    for name in (
                        mapping["hostname"],
                        features.get("node"),
                        slurm.get("node"),
                    ):
                        self.add(name, in_slurm)
                    continue
                for key in ("hostname", "host"):
                    if isinstance(mapping.get(key), str):
                        self.add(
                            mapping[key],
                            bool(mapping.get("job") or mapping.get("job_id")),
                        )

    def add_logs(self, slurm_dir):
        """The hosts of the task and step logs and of the accounting."""
        for entry in _listing(slurm_dir):
            if not entry.is_file() or not entry.name.endswith(".out"):
                continue
            text = Path(entry.path).read_text(
                encoding="utf-8", errors="replace"
            )
            for pattern in LOG_HOST_LINES:
                for match in pattern.finditer(text):
                    self.add(match.group(1), True)
        accounting = Path(slurm_dir) / "sacct.txt"
        if accounting.is_file():
            lines = accounting.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            header = lines[0].split("|") if lines else []
            if "NodeList" in header:
                column = header.index("NodeList")
                for line in lines[1:]:
                    fields = line.split("|")
                    if len(fields) > column:
                        for host in expand_hostlist(fields[column]):
                            self.add(host, True)

    def tokens(self, family):
        """``{lower-case hostname: what the copies name it by}``."""
        return {
            name: family if in_slurm else LOGIN_HOST
            for name, in_slurm in self.in_slurm.items()
        }


def _command_words(value):
    """The words of a command or of sbatch arguments that name a path."""
    try:
        words = shlex.split(value)
    except ValueError:
        words = value.split()
    found = set()
    for word in words:
        for part in (word, word.partition("=")[2]):
            part = part.strip(";&|()")
            if "/" in part and len(part) > 2:
                found.add(part)
    return found


def _partition_field(mapping, key, path):
    """Whether ``mapping[key]`` holds a partition: ``SLURM_JOB_PARTITION``,
    the ``partition`` of a host part's ``slurm`` block, or the
    ``PARTITION`` of a site block (a mapping that names ``NODE_FEATURE``)."""
    return (
        key == "SLURM_JOB_PARTITION"
        or (key == "partition" and path[-1:] == ("slurm",))
        or (key == "PARTITION" and "NODE_FEATURE" in mapping)
    )


class Redaction:
    """The rules that make one campaign's tracked copies, and their check.

    The replacements (:meth:`text`, :meth:`value`):

    - a path that starts with a named directory starts with its name
      instead, the longest directory that fits first; the directory is
      replaced where the character after it does not continue its last
      component (:data:`_PATH_AFTER`) and the characters before it do not
      continue a longer path (:func:`_continues_a_path`), so that
      ``X=/proj/env/lib`` and ``-L/proj/env/lib`` are replaced and neither
      ``/proj/envs`` nor ``/a/proj/env`` is corrupted;
    - a hostname, as a whole name in any letter case, is its family or
      :data:`LOGIN_HOST`;
    - in a JSON value, the string of a field that holds a partition
      (:func:`_partition_field`) is :data:`PARTITION_TOKEN`, whatever the
      partition's name.

    The check (:meth:`leaks`) does not rely on them: it searches each copy
    for every forbidden string as a plain substring (a hostname in any
    letter case), so that whatever a replacement left is refused. Before
    the search, the names the replacements write are blanked where they
    stand as whole words, so that a forbidden string inside one of them
    (a host named ``login``) is not taken for a leak, and an occurrence
    inside a run of :data:`DIGEST_LENGTH` hex digits or more is a digest's
    and not one. It also refuses every absolute path that no name covers
    and that lies outside :data:`SYSTEM_PREFIXES`.

    Parameters
    ----------
    paths : mapping of str to str
        Directories, each in every form it may take (:func:`path_variants`),
        and the name that replaces each: ``$NAME`` or ``~``.
    hosts : mapping of str to str
        Lower-case hostnames and the family or :data:`LOGIN_HOST` that
        replaces each.
    forbidden : mapping of str to list of (str, str)
        What may remain in no copy, each ``(string, what it is)``:
        ``paths`` (the named directories, the command settings and their
        words that name a path), ``users`` and ``hosts``.
    """

    def __init__(self, paths, hosts, forbidden):
        self.paths = dict(paths)
        self.hosts = dict(hosts)
        self.forbidden = {key: list(value) for key, value in forbidden.items()}
        self.counts = Counter()
        self._paths = _paths_pattern(self.paths)
        self._hosts = _bounded(self.hosts, _HOST_WORD, re.IGNORECASE)
        tokens = {
            token
            for token in (
                *self.paths.values(),
                *self.hosts.values(),
                PARTITION_TOKEN,
                "<redacted>",
            )
            if len(token) > 1
        }
        self._tokens = _bounded(tokens, "A-Za-z0-9_")

    def _host(self, match):
        token = self.hosts[match.group(0).lower()]
        self.counts["hosts"] += 1
        return token

    def _replace_paths(self, text):
        if self._paths is None:
            return text
        pieces, done, position = [], 0, 0
        while True:
            match = self._paths.search(text, position)
            if match is None:
                break
            if _continues_a_path(text, match.start()):
                position = match.start() + 1
                continue
            token = self.paths[match.group(0)]
            self.counts[token] += 1
            pieces += [text[done : match.start()], token]
            done = position = match.end()
        pieces.append(text[done:])
        return "".join(pieces)

    def text(self, text):
        """``text`` with its paths and then its hostnames replaced."""
        text = self._replace_paths(text)
        if self._hosts is not None:
            text = self._hosts.sub(self._host, text)
        return text

    def value(self, value, path=()):
        """A JSON value, every string and key of it redacted."""
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self.value(item, path) for item in value]
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                new = self.text(key)
                if new in result:
                    raise ContractError(
                        f"two keys of {'.'.join(path) or 'the document'} "
                        f"become {new!r}"
                    )
                if (
                    isinstance(item, str)
                    and item
                    and _partition_field(value, key, path)
                ):
                    self.counts[PARTITION_TOKEN] += 1
                    result[new] = PARTITION_TOKEN
                else:
                    result[new] = self.value(item, (*path, key))
            return result
        return value

    def leaks(self, text):
        """Where what may remain in no copy occurs in one file's text.

        Returns ``(line, string, what it is)`` for each occurrence, at most
        :data:`MAX_LEAKS`, in the order of the text.
        """
        masked = text
        if self._tokens is not None:
            masked = self._tokens.sub(
                lambda match: "\0" * len(match.group(0)), text
            )
        lowered = masked.lower()
        found = set()
        for kind, items in self.forbidden.items():
            for string, what in items:
                if not string:
                    continue
                haystack, needle = (
                    (lowered, string.lower())
                    if kind == "hosts"
                    else (masked, string)
                )
                index = haystack.find(needle)
                while index >= 0 and len(found) < MAX_LEAKS:
                    if not _inside_digest(masked, index, len(needle)):
                        line = text.count("\n", 0, index) + 1
                        found.add((line, string, what))
                    index = haystack.find(needle, index + 1)
        for offset, path in absolute_paths(text):
            if len(found) >= MAX_LEAKS:
                break
            if not _system_path(path):
                line = text.count("\n", 0, offset) + 1
                found.add((line, path, "an absolute path that no name covers"))
        return sorted(found)[:MAX_LEAKS]


def _json_layout(text):
    """``(indent, trailing newline)`` of a JSON text that ``json.dumps``
    wrote."""
    match = re.match(r"[\[{]\n( +)\S", text)
    return (len(match.group(1)) if match else None), text.endswith("\n")


def _pip_freeze(lines):
    """``pip freeze`` lines, each that names a path cut to its package's
    name; returns the lines and how many were cut."""
    kept, cut = [], 0
    for line in lines:
        name, at, reference = line.partition(" @ ")
        if at and reference.startswith("file:"):
            kept.append(f"{name} @ file:<redacted>")
            cut += 1
        elif line.startswith("-e ") and "://" not in line:
            kept.append("-e <redacted>")
            cut += 1
        else:
            kept.append(line)
    return kept, cut


def _archive(campaign):
    """The parts of the campaign's archive and their SHA-256, or None."""
    path = campaign.parent / f"{campaign.name}.tar.zst.sha256"
    if not path.is_file():
        return None
    parts = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) == 2:
            parts[fields[1].lstrip("*")] = fields[0]
    return {"sha256_file": path.name, "parts": parts}


def _inside(path, directory):
    path, directory = Path(path), Path(directory)
    return path == directory or directory in path.parents


def _covered(value, named):
    """Whether a directory lies in one of the ``named`` forms, or is one."""
    for form in path_variants(value):
        for prefix in named:
            if form == prefix or any(
                form.startswith(prefix + separator)
                for separator in ("/", "\\", "\\\\")
            ):
                return True
    return False


def _tree_paths(documents):
    """``{tree name: [path, ...]}`` of every identity the documents hold
    (``imports.trees.<name>.path``)."""
    found = {}
    for document in documents:
        for mapping in _dicts(document):
            imports = mapping.get("imports")
            trees = imports.get("trees") if isinstance(imports, dict) else None
            if not isinstance(trees, dict):
                continue
            for name, tree in trees.items():
                if isinstance(tree, dict) and isinstance(
                    tree.get("path"), str
                ):
                    if tree["path"]:
                        found.setdefault(name, []).append(tree["path"])
    return found


def tree_token(name):
    """The name a source tree's path takes in the copies: ``harness`` is
    ``$HARNESS_TREE``."""
    return "$" + re.sub(r"[^A-Z0-9]+", "_", name.upper()).strip("_") + "_TREE"


def redaction_rules(
    campaign, manifest, documents, operator, environ=None, paths=(), host=None
):
    """The :class:`Redaction` of one campaign.

    The named directories, each replaced by its name at the start of a path
    and forbidden in every form (:func:`path_variants`), in this order of
    precedence where two are one directory: the path settings of the site
    block and of this process (``$CAMPAIGN_ENV``, ``$PYVBMC_SOURCE``, ...,
    ``$LOGIN_PROFILE``), the ``paths`` given (``$NAME``), the operator's
    homes and ``~<username>`` (``~``); then, where none of those holds
    them, the path of every source tree the identities of ``documents``
    name (:func:`tree_token`, ``$HARNESS_TREE``) and the directory that
    holds the campaign directory (``$CAMPAIGN_PARENT``). The command
    settings, whole and each of their words that names a path, the
    usernames and every hostname are forbidden too.

    Parameters
    ----------
    campaign : path
        The campaign directory, whose ``slurm/`` holds the task logs and
        the accounting.
    manifest : dict
        Its manifest, whose ``site`` block names the operator settings.
    documents : sequence
        Every JSON document read from the campaign: the manifest, the
        verification report, every completion record, every tracked copy.
    operator : dict
        :func:`operator_identity`.
    environ : mapping, optional
        The settings of this process, which join the manifest's; this
        process's environment by default.
    paths : sequence of (str, str)
        Further directories to name, ``(NAME, PATH)``: their paths become
        ``$NAME``.
    host : str, optional
        The host that redacts, which names no Slurm job; this one's by
        default.
    """
    site = manifest.get("site")
    if not isinstance(site, dict) or not site.get("NODE_FEATURE"):
        raise ContractError(
            "the manifest's site block names no NODE_FEATURE, the node "
            "family that the tracked copies name the campaign's hosts by"
        )
    family = check_feature(site["NODE_FEATURE"])
    env = os.environ if environ is None else environ
    slurm_dir = Path(campaign) / "slurm"

    def values(name):
        return {v for v in (site.get(name), env.get(name)) if v}

    replace, forbidden = {}, {"paths": [], "users": [], "hosts": []}

    def name(value, token, what, first=True):
        for form in path_variants(value):
            if first:
                replace[form] = token
            else:
                replace.setdefault(form, token)
            forbidden["paths"].append((form, what))

    for setting in REDACTED_PATH_SETTINGS:
        for value in values(setting):
            name(value, f"${setting}", setting)
    reserved = {*SETTINGS, CAMPAIGN_PARENT}
    for key, value in paths:
        if (
            not re.fullmatch(r"[A-Z][A-Z0-9_]*", key)
            or key in reserved
            or key.endswith("_TREE")
        ):
            raise ContractError(
                f"--path {key}=...: the name is upper case, no operator "
                f"setting's, not {CAMPAIGN_PARENT} and not one ending in "
                "_TREE, which name the campaign's own directories"
            )
        name(value, f"${key}", f"--path {key}")
    for home in operator["homes"]:
        name(home, "~", "the home directory", first=False)
    for user in operator["users"]:
        replace.setdefault(f"~{user}", "~")
        forbidden["paths"].append((f"~{user}", "the home directory"))
    for tree, found in sorted(_tree_paths(documents).items()):
        for value in found:
            if not _covered(value, replace):
                name(value, tree_token(tree), f"the {tree} tree")
    parent = Path(campaign).resolve().parent
    if not _is_root(parent) and not _covered(parent, replace):
        name(parent, f"${CAMPAIGN_PARENT}", "the campaign's parent")
    for setting in COMMAND_SETTINGS:
        for value in values(setting):
            forbidden["paths"].append((value, setting))
            forbidden["paths"] += [(w, setting) for w in _command_words(value)]
    forbidden["users"] = [(u, "the username") for u in operator["users"]]

    hosts = Hosts()
    hosts.add_documents(documents, family)
    hosts.add_logs(slurm_dir)
    hosts.add(socket.gethostname() if host is None else host, False)
    tokens = hosts.tokens(family)
    forbidden["hosts"] = [(h, "a hostname") for h in tokens]
    return Redaction(replace, tokens, forbidden)


def _read_campaign(campaign):
    """What the redaction reads of a campaign directory.

    The manifest and the verification report, the completion record of
    every case the report places as verified, the claims of the allocated
    cases (for the hosts they name), and the tracked copies with their
    bytes, text and, for a ``.json`` file, JSON value. Raises
    :class:`ContractError` for any of them that is missing or unreadable.
    """
    campaign = Path(campaign)

    def document(path):
        try:
            return read_json(path)
        except FileNotFoundError as error:
            raise ContractError(f"{path} is missing") from error
        except (OSError, ValueError) as error:
            raise ContractError(f"{path} cannot be read: {error}") from error

    manifest = document(campaign / "manifest.json")
    verification_path = campaign / "verification.json"
    if not verification_path.is_file():
        raise ContractError(f"{campaign} holds no verification.json")
    verification = document(verification_path)
    cases = verification.get("cases", [])
    records, documents = {}, [manifest, verification]
    for case in cases:
        if case.get("status") == "verified":
            records[case["tag"]] = document(record_path(campaign, case["tag"]))
            documents.append(records[case["tag"]])
    tags = [case["tag"] for case in cases]
    for group in sorted({Path(t).parent.as_posix() for t in tags}):
        folder = campaign / CLAIMS / ("" if group == "." else group)
        for entry in _listing(folder):
            if entry.is_file() and not entry.name.startswith("."):
                try:
                    documents.append(read_json(entry.path))
                except (OSError, ValueError):
                    pass
    names = tracked_files(campaign, manifest, verification, records)
    sources = {}
    for name in names:
        data = (campaign / name).read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = None
        if text is None or "\0" in text:
            raise ContractError(
                f"{campaign / name} is not text; a tracked copy is JSON or "
                "text"
            )
        parsed = None
        if name.endswith(".json"):
            try:
                parsed = json.loads(text)
            except ValueError as error:
                raise ContractError(
                    f"{campaign / name} is not JSON: {error}"
                ) from error
        if parsed is not None and name not in TRACKED_ALWAYS:
            documents.append(parsed)
        sources[name] = (data, text, parsed)
    return {
        "manifest": manifest,
        "verification": verification,
        "records": records,
        "documents": documents,
        "names": names,
        "sources": sources,
    }


def _leak_lines(leaks, limit=40):
    """The refusal's lines for ``(file, line, string, what)`` leaks."""
    shown = [
        f"  {name}{f':{line}' if line else ''}: {what} {string!r}"
        for name, line, string, what in leaks[:limit]
    ]
    more = len(leaks) - len(shown)
    return "\n".join(shown) + (f"\n  and {more} more" if more else "")


def redact(
    campaign, out, operator=None, environ=None, paths=(), host=None, say=None
):
    """Write a finished campaign's tracked copies, redacted, into ``out``.

    The copies are the files its manifest declares (:func:`tracked_files`),
    at the same paths relative to ``out``, with :data:`REDACTION` beside
    them. The campaign's verification report must have passed and show no
    case in flight; its failed, interrupted and missing cases are listed in
    :data:`REDACTION`. In every copy (:func:`redaction_rules`,
    :class:`Redaction`):

    - every host that ran a Slurm job of the campaign (a record's host
      part, a task's or a step's log, the accounting) is named by the
      campaign's node family, the value of ``NODE_FEATURE``, and every
      other host (the login node where ``prepare`` and the driver ran, and
      the host that redacts) by :data:`LOGIN_HOST`;
    - a path under a named directory starts with its name: a path setting
      of the site block (``$PYVBMC_GPYREG_SOURCE/gpyreg``), a ``--path``,
      the operator's home (``~``), a source tree of the identities
      (``$HARNESS_TREE``) or the directory that holds the campaign
      (``$CAMPAIGN_PARENT``);
    - a field that holds a partition is :data:`PARTITION_TOKEN`;
    - the manifest keeps no ``site`` block, and its ``pip freeze`` lines
      that name a path keep their package's name alone;
    - the task logs, the accounting, the claims and the error files are
      not copied.

    Then every copy, :data:`REDACTION` included, is searched for each value
    of the site block that is a path or a command, every named directory,
    the operator's username and home directory, and every hostname, as
    plain substrings, and for any absolute path outside
    :data:`SYSTEM_PREFIXES` (:meth:`Redaction.leaks`); any of them raises
    :class:`ContractError` naming the file and the string, and nothing is
    written. The copies are written into a directory beside ``out`` and
    renamed into place, so ``out`` holds either all of them or none.

    Parameters
    ----------
    campaign : path
        The campaign directory.
    out : path
        Where the copies go; it must not exist, or be empty, and lie
        neither inside the campaign directory nor inside any source tree
        of its manifest's identity, since those stay clean until every
        campaign that uses them is done.
    operator : dict, optional
        :func:`operator_identity`, this process's by default.
    environ, paths, host
        See :func:`redaction_rules`.
    say : callable, optional
        Prints a progress line; :func:`print` by default.

    Returns
    -------
    dict
        The :data:`REDACTION` record written.
    """
    say = _say if say is None else say
    campaign = Path(campaign).resolve()
    out = Path(out).resolve()
    try:
        manifest = read_json(campaign / "manifest.json")
    except (OSError, ValueError) as error:
        raise ContractError(
            f"{campaign / 'manifest.json'} cannot be read: {error}"
        ) from error
    trees = (manifest.get("identity") or {}).get("imports", {}).get("trees")
    for where in [campaign] + [
        Path(t["path"]).resolve()
        for t in (trees or {}).values()
        if isinstance(t, dict) and t.get("path")
    ]:
        if _inside(out, where):
            raise ContractError(
                f"{out} lies inside {where}, which the campaign's tasks read "
                "or which must stay clean; write the copies into another "
                "checkout"
            )
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ContractError(f"{out} exists and is not an empty directory")
    verification_path = campaign / "verification.json"
    if not verification_path.is_file():
        raise ContractError(f"{campaign} holds no verification.json")
    try:
        verification = read_json(verification_path)
    except (OSError, ValueError) as error:
        raise ContractError(
            f"{verification_path} cannot be read: {error}"
        ) from error
    if verification.get("exit_code") != 0:
        raise ContractError(
            f"{verification_path} did not pass; the copies are made of a "
            "finished campaign"
        )
    statuses = Counter(
        case.get("status") for case in verification.get("cases", [])
    )
    if statuses["in_flight"]:
        raise ContractError(
            f"{verification_path} places {statuses['in_flight']} cases in "
            "flight; the copies are made of a finished campaign, so finish "
            "it again once they are done"
        )
    operator = operator_identity(environ) if operator is None else operator
    say(f"reading {campaign}")
    read = _read_campaign(campaign)
    names, sources = read["names"], read["sources"]
    say(f"redacting {len(names)} files")
    redaction = redaction_rules(
        campaign, manifest, read["documents"], operator, environ, paths, host
    )
    written, files, cut = {}, {}, 0
    for name in names:
        data, text, parsed = sources[name]
        if parsed is None:
            new_text = redaction.text(text)
            new = new_text.encode("utf-8")
        else:
            value = parsed
            if name == "manifest.json":
                value = {k: v for k, v in parsed.items() if k != "site"}
                if isinstance(value.get("pip_freeze"), list):
                    value["pip_freeze"], cut = _pip_freeze(value["pip_freeze"])
            value = redaction.value(value)
            if value == parsed:
                new, new_text = data, text
            else:
                indent, newline = _json_layout(text)
                new_text = json.dumps(value, indent=indent) + (
                    "\n" if newline else ""
                )
                new = new_text.encode("utf-8")
        written[name] = (new, new_text)
        files[name] = {
            "source_sha256": hashlib.sha256(data).hexdigest(),
            "sha256": hashlib.sha256(new).hexdigest(),
            "bytes": len(new),
        }
    record = {
        "contract": CONTRACT_VERSION,
        "campaign": campaign.name,
        "generated": now(),
        "node_family": manifest["site"]["NODE_FEATURE"],
        "rules": [
            "a host that ran a Slurm job of the campaign is named by the "
            "node family, NODE_FEATURE; any other host, "
            f"{LOGIN_HOST!r}",
            "a path under a path setting starts with its name "
            "($CAMPAIGN_ENV, $PYVBMC_SOURCE, ...), one under a --path "
            "directory with its name, one under the operator's home "
            "with ~, and one under a source tree or the directory that "
            "holds the campaign, where none of those holds them, with "
            f"$<TREE>_TREE or ${CAMPAIGN_PARENT}",
            f"a field that holds a partition is {PARTITION_TOKEN}",
            "the manifest keeps no site block, and its pip freeze lines "
            "that name a path keep their package's name alone",
            "each file's source_sha256 is that of the campaign's own file, "
            "which the records and reports hash",
        ],
        "replaced": dict(sorted(redaction.counts.items())),
        "pip_freeze_paths": cut,
        "cases_not_verified": {
            status: [
                case["tag"]
                for case in verification.get("cases", [])
                if case.get("status") == status
            ]
            for status in ("failed", "interrupted", "missing")
        },
        "archive": _archive(campaign),
        "files": files,
    }
    record_text = json.dumps(record, indent=2) + "\n"
    written[REDACTION] = (record_text.encode("utf-8"), record_text)
    leaks = [
        (name, line, string, what)
        for name, (_, text) in sorted(written.items())
        for line, string, what in redaction.leaks(text)
    ]
    if leaks:
        raise ContractError(
            "the redacted copies still hold what they may not, so none was "
            "written:\n" + _leak_lines(leaks)
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.parent / f".{out.name}.{uuid.uuid4().hex}.redacting"
    try:
        for name, (data, _) in written.items():
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        for path in staging.rglob("*"):
            if path.is_file():
                name = path.relative_to(staging).as_posix()
                if path.read_bytes() != written[name][0]:
                    raise ContractError(f"{path} is not what was written")
        if out.exists():
            out.rmdir()
        os.replace(staging, out)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    forbidden = sum(len(v) for v in redaction.forbidden.values())
    say(
        f"{len(names)} copies and {REDACTION} in {out}; replaced "
        + ", ".join(f"{k} {v}" for k, v in record["replaced"].items())
        + f"; none of {forbidden} forbidden strings remains"
    )
    return record


def check_files(
    campaign, files, operator=None, environ=None, paths=(), host=None
):
    """Search files that :func:`redact` did not write for a campaign's leaks.

    The same search as :func:`redact`'s (:meth:`Redaction.leaks`), with the
    campaign's forbidden strings and names (:func:`redaction_rules`), in
    files such as the hand-written README of the directory that holds the
    tracked copies. Returns ``(file, line, string, what it is)`` for each
    occurrence; none for files that hold nothing forbidden.
    """
    campaign = Path(campaign).resolve()
    read = _read_campaign(campaign)
    operator = operator_identity(environ) if operator is None else operator
    redaction = redaction_rules(
        campaign,
        read["manifest"],
        read["documents"],
        operator,
        environ,
        paths,
        host,
    )
    found = []
    for path in files:
        try:
            text = Path(path).read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise ContractError(f"{path} cannot be read as text: {error}")
        found += [
            (str(path), line, string, what)
            for line, string, what in redaction.leaks(text)
        ]
    return found


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


def _cmd_queue_check(args):
    state = queue_state(args.slurm)
    lines = [f"{index} {status}\n" for index, status in state["cases"]]
    (Path(args.slurm) / "queued.txt").write_text(
        "".join(lines), encoding="utf-8", newline="\n"
    )
    if state["live"]:
        print(
            "tasks of the recorded jobs are still queued or running:",
            file=sys.stderr,
        )
        for _, task, status in state["live"]:
            print(f"{task} {status}", file=sys.stderr)
    for job, detail in state["unknown"]:
        print(
            f"the queue does not answer for job {job}, so its tasks may "
            f"still run: {detail}",
            file=sys.stderr,
        )
    if state["live"]:
        print("queued")
    elif state["unknown"]:
        print("unknown")
    else:
        print("clear")
    return 0


def _cmd_archive_check(args):
    problems = accounting_problems(args.slurm)
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


def _cmd_wait_job(args):
    code = wait_job(args.job, poll=args.poll)
    return min(max(int(code), 0), 255)


def _cmd_redact(args):
    paths = []
    for item in args.path or ():
        name, equals, value = item.partition("=")
        if not equals or not value:
            print(f"--path {item!r} is not NAME=PATH", file=sys.stderr)
            return 1
        paths.append((name, value))
    try:
        if args.check:
            leaks = check_files(args.campaign, args.check, paths=paths)
            if leaks:
                print(
                    "refusing: the files hold what the tracked copies may "
                    "not:\n" + _leak_lines(leaks),
                    file=sys.stderr,
                )
                return 1
            print(
                f"none of {len(args.check)} files holds what the tracked "
                "copies may not"
            )
        else:
            redact(args.campaign, args.out, paths=paths)
    except (ContractError, OSError, ValueError) as error:
        print(f"refusing: {error}", file=sys.stderr)
        return 1
    return 0


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
    queue = sub.add_parser(
        "queue-check",
        help="ask the queue for every job of slurm/jobs.txt and "
        "slurm/steps.txt, write the queued case indices to "
        "slurm/queued.txt, and print clear, queued or unknown",
    )
    queue.add_argument("--slurm", type=Path, required=True)
    archive = sub.add_parser(
        "archive-check",
        help="refuse while slurm/sacct.txt shows a recorded task that has "
        "not ended, or is missing",
    )
    archive.add_argument("--slurm", type=Path, required=True)
    wait = sub.add_parser(
        "wait-job",
        help="wait until a job has ended in the accounting, and exit with "
        "its exit code",
    )
    wait.add_argument("--job", required=True)
    wait.add_argument(
        "--poll",
        type=float,
        default=30.0,
        help="seconds between two queries of the accounting (default 30)",
    )
    redaction = sub.add_parser(
        "redact",
        help="write a finished campaign's tracked copies, redacted, and "
        "check that no site detail, username, home or hostname remains; "
        "or, with --check, search other files for them",
    )
    redaction.add_argument("--campaign", type=Path, required=True)
    target = redaction.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--out",
        type=Path,
        help="the new or empty directory the copies go into, in the "
        "checkout that the hand-back is committed from; it may lie neither "
        "inside the campaign directory nor inside a source tree of the "
        "campaign, the harness checkout among them",
    )
    target.add_argument(
        "--check",
        type=Path,
        nargs="+",
        metavar="FILE",
        help="files that redact did not write, such as the README beside "
        "the tracked copies, to search as it searches the copies",
    )
    redaction.add_argument(
        "--path",
        action="append",
        metavar="NAME=PATH",
        help="a directory outside the home, such as a scratch area that "
        "holds the campaign, whose paths the copies write as $NAME/...; "
        "repeatable",
    )
    return parser.parse_args(argv)


COMMANDS = {
    "check-env": _cmd_check_env,
    "check-site": _cmd_check_site,
    "check-cases": _cmd_check_cases,
    "finishing-steps": _cmd_finishing_steps,
    "finish-check": _cmd_finish_check,
    "queue-check": _cmd_queue_check,
    "archive-check": _cmd_archive_check,
    "wait-job": _cmd_wait_job,
    "redact": _cmd_redact,
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
