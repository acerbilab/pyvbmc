"""Stub Slurm commands for the tests of the campaign contract and driver.

:func:`write_stubs` writes bash scripts named ``sbatch``, ``squeue``,
``sacct``, ``scontrol`` and ``zstd`` into a directory that the tests put
first on the PATH. Each reads its answers from, and records its calls in,
the state directory named by ``STUB_STATE``:

- ``sbatch`` takes the next job id from ``next_job`` (1001 first), records
  its arguments one per line in ``sbatch/<job>/args`` and the variables the
  driver passes in ``sbatch/<job>/env``, prints the job id, and with
  ``--wait`` runs the batch script in the foreground, as Slurm would with
  ``SLURM_JOB_ID`` and ``SLURMD_NODENAME`` set, exiting with its code. A
  file ``sbatch_fail`` makes it fail.
- ``squeue -j <job>`` prints ``squeue/<job>`` when it exists, and fails as
  for a job the controller has dropped when ``squeue/<job>.fail`` does.
- ``sacct ... -j <step> -o State`` prints ``sacct/<step>``, fails when
  ``sacct/<step>.fail`` exists and prints nothing otherwise; any other
  ``sacct`` call (the finish's accounting) is recorded in ``sacct_calls``.
- ``scontrol show config`` prints ``MaxArraySize = <max_array_size>`` (1001
  by default) unless ``scontrol_fail`` exists; ``scontrol show node <name>``
  prints the node with the features of ``features`` (``stubfeat`` by
  default).
- ``zstd ... -c`` copies its input, which is all the finish's archive needs.

On Windows the scripts run under Git Bash, and each also gets a ``.cmd``
shim, through which Python's ``subprocess`` reaches it. :func:`find_bash`
locates bash, failing rather than skipping where there is none, since the
tests must run wherever the campaigns do.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

SBATCH = r"""#!/bin/bash
set -u
state=${STUB_STATE:?}
n=$(( $(cat "$state/next_job" 2>/dev/null || echo 1000) + 1 ))
echo "$n" > "$state/next_job"
mkdir -p "$state/sbatch/$n"
printf '%s\n' "$@" > "$state/sbatch/$n/args"
env | grep -E '^(INDEX_OFFSET|CAMPAIGN_DIR|CAMPAIGN_STEP|HARNESS|NODE_FEATURE|PARTITION|TIME|MEM|THROTTLE|CASES_SUBSET)=' \
    | sort > "$state/sbatch/$n/env" || true
if [ -f "$state/sbatch_fail" ]; then
    echo "sbatch: error: the stub refuses" >&2
    exit 1
fi
wait=0
output=""
previous=""
for argument in "$@"; do
    case $argument in
        --wait) wait=1 ;;
        --output=*) output=${argument#--output=} ;;
    esac
    if [ "$previous" = --output ]; then
        output=$argument
    fi
    previous=$argument
done
script=${!#}
echo "$n"
if [ "$wait" = 1 ]; then
    output=${output//%j/$n}
    SLURM_JOB_ID=$n SLURMD_NODENAME=${STUB_NODE:-stubnode} \
        bash "$script" > "${output:-/dev/null}" 2>&1 < /dev/null
    exit $?
fi
"""

SQUEUE = r"""#!/bin/bash
set -u
state=${STUB_STATE:?}
job=""
previous=""
for argument in "$@"; do
    if [ "$previous" = -j ]; then
        job=$argument
    fi
    previous=$argument
done
echo "$*" >> "$state/squeue_calls"
if [ -f "$state/squeue/$job.fail" ]; then
    echo "slurm_load_jobs error: Invalid job id specified" >&2
    exit 1
fi
if [ -f "$state/squeue/$job" ]; then
    cat "$state/squeue/$job"
fi
"""

SACCT = r"""#!/bin/bash
set -u
state=${STUB_STATE:?}
job=""
previous=""
query=0
for argument in "$@"; do
    if [ "$previous" = -j ]; then
        job=$argument
    fi
    if [ "$previous" = -o ] && [ "$argument" = State ]; then
        query=1
    fi
    previous=$argument
done
if [ "$query" = 0 ]; then
    echo "$*" >> "$state/sacct_calls"
    echo "JobID|JobName|State"
    exit 0
fi
echo "$job" >> "$state/sacct_queries"
if [ -f "$state/sacct/$job.fail" ]; then
    echo "sacct: error: Problem talking to the database" >&2
    exit 1
fi
if [ -f "$state/sacct/$job" ]; then
    cat "$state/sacct/$job"
fi
"""

SCONTROL = r"""#!/bin/bash
set -u
state=${STUB_STATE:?}
if [ "${1:-}" = show ] && [ "${2:-}" = config ]; then
    if [ -f "$state/scontrol_fail" ]; then
        echo "slurm_load_ctl_conf error" >&2
        exit 1
    fi
    echo "Configuration data as of 2026-09-25T00:00:00"
    echo "MaxArraySize            = $(cat "$state/max_array_size" 2>/dev/null || echo 1001)"
    echo "MaxJobCount             = 10000"
    exit 0
fi
if [ "${1:-}" = show ] && [ "${2:-}" = node ]; then
    if [ -f "$state/scontrol_node_fail" ]; then
        echo "Node ${3:-} not found" >&2
        exit 1
    fi
    features=$(cat "$state/features" 2>/dev/null || echo stubfeat)
    echo "NodeName=${3:-} Arch=x86_64 CoresPerSocket=4 CPUAlloc=1" \
        "AvailableFeatures=$features ActiveFeatures=$features State=MIXED"
    exit 0
fi
echo "scontrol stub: unsupported $*" >&2
exit 1
"""

ZSTD = r"""#!/bin/bash
exec cat
"""

STUBS = {
    "sbatch": SBATCH,
    "squeue": SQUEUE,
    "sacct": SACCT,
    "scontrol": SCONTROL,
    "zstd": ZSTD,
}


def git_root():
    """The Git for Windows installation, from ``git --exec-path``."""
    exec_path = subprocess.run(
        ["git", "--exec-path"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return Path(exec_path).parents[2]


def find_bash():
    """The bash the driver's scripts run under: Git Bash on Windows."""
    if sys.platform == "win32":
        bash = git_root() / "usr" / "bin" / "bash.exe"
        if not bash.is_file():
            raise RuntimeError(f"Git Bash is not at {bash}")
        return str(bash)
    bash = shutil.which("bash")
    if bash is None:
        raise RuntimeError("bash is not on the PATH")
    return bash


def tool_path():
    """PATH entries that bash's coreutils need: Git's usr/bin on Windows."""
    if sys.platform == "win32":
        return [str(git_root() / "usr" / "bin")]
    return []


def write_script(path, text):
    """Write a bash script with LF line endings, executable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    path.chmod(0o755)
    return path


def write_stubs(directory, bash=None):
    """Write the stub commands into ``directory``; return it."""
    directory = Path(directory)
    bash = find_bash() if bash is None else bash
    for name, text in STUBS.items():
        script = write_script(directory / name, text)
        if sys.platform == "win32":
            (directory / f"{name}.cmd").write_bytes(
                f'@"{bash}" "{script.as_posix()}" %*\r\n'.encode("utf-8")
            )
    return directory


def stub_environment(stubs, state, base=None):
    """``base`` (this process's environment), the stubs first on the PATH."""
    environment = dict(os.environ if base is None else base)
    entries = [str(stubs), environment.get("PATH", ""), *tool_path()]
    environment["PATH"] = os.pathsep.join(entry for entry in entries if entry)
    environment["STUB_STATE"] = Path(state).as_posix()
    return environment
