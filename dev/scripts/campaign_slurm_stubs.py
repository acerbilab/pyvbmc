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

:class:`FakeSite` stands in for what a site and an operator leave in a
campaign's records, for the tests of the redaction of its tracked copies
(``campaign_contract.redact``).
"""

import json
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


class FakeSite:
    """The site details and the operator that a test plants in a campaign.

    Every value but the node family holds one of :data:`TOKENS`, which no
    file of the repository holds, so that :meth:`leaks` can search a
    redacted copy for them independently of how the redaction matches: the
    operator's username and home (under ``root``), a login host, compute
    nodes by their domain names, one node that only a task log names and
    two that only the accounting names, a partition, an environment and a
    gpyreg checkout outside the home, and the login, conda and sbatch
    settings. ``family``, the node feature, names every compute node in
    the copies, and may remain.
    """

    #: What no redacted copy may hold, in any letter case.
    TOKENS = (
        "fakeoperator",
        "fakelogin",
        "fakenode",
        "fakepartition",
        "fakeproject",
        "fakeproxy",
        "fakeconda",
    )

    def __init__(self, root):
        root = Path(root)
        self.user = "fakeoperator"
        self.home = root / "home" / self.user
        self.home.mkdir(parents=True, exist_ok=True)
        self.login = "fakelogin7.cluster.invalid"
        self.nodes = [f"fakenode{n}.cluster.invalid" for n in (17, 18)]
        self.log_node = "fakenode40"
        self.accounting = "fakenode[41-42]"
        self.family = "fakefamily"
        self.partition = "fakepartition"
        self.env = root / "fakeproject" / "env"
        self.gpyreg = root / "fakeproject" / "gpyreg"

    def site_block(self, harness):
        """The ``site`` block of a campaign of ``harness`` at this site."""
        import campaign_contract as contract

        site = {name: None for name in contract.SETTINGS}
        site.update(
            HARNESS=harness,
            CAMPAIGN_ENV=str(self.env),
            NODE_FEATURE=self.family,
            PARTITION=self.partition,
            LOGIN_SETUP="export https_proxy=http://fakeproxy.invalid:3128",
            CONDA_SETUP="module load fakeconda/24.1",
            PYVBMC_GPYREG_SOURCE=str(self.gpyreg),
            SBATCH_EXTRA="--comment=fakeproject/queue",
            THROTTLE="200",
            TIME="00:30:00",
            MEM="2G",
        )
        return site

    def operator(self):
        """The operator's identity, as
        ``campaign_contract.operator_identity`` gives it."""
        return {"users": [self.user], "homes": [str(self.home)]}

    def host(self, node=None, job="4242", task="1"):
        """A host part: the login node's, outside Slurm, or with ``node``
        that of the array task ``<job>_<task>`` on that node."""
        env = self.env.as_posix()
        part = {
            "hostname": self.login,
            "platform": "Linux-6.1-x86_64-with-glibc2.34",
            "executable": f"{env}/bin/python",
            "cpu_model": "Stand-in CPU",
            "node_features": None,
            "blas": [
                {
                    "user_api": "blas",
                    "internal_api": "openblas",
                    "num_threads": 1,
                    "filepath": f"{env}/lib/libopenblas.so",
                }
            ],
            "cpu_affinity": {"cpus": [3], "physical_cores": ["0:3"]},
            "threads": {"OMP_NUM_THREADS": "1"},
            "slurm": {
                "job_id": None,
                "array_job_id": None,
                "array_task_id": None,
                "restart_count": None,
                "node": None,
                "partition": None,
                "cpus_per_task": None,
            },
        }
        if node is not None:
            short = node.split(".")[0]
            part["hostname"] = node
            part["node_features"] = {
                "node": short,
                "available": [self.family, "avx2"],
                "active": [self.family, "avx2"],
            }
            part["slurm"] = {
                "job_id": f"{job}{int(task):03d}",
                "array_job_id": job,
                "array_task_id": str(task),
                "restart_count": "0",
                "node": short,
                "partition": self.partition,
                "cpus_per_task": "1",
            }
        return part

    def plant(self, identity, node=None, **task):
        """Give an identity this site's host part and paths; return it.

        Its host part becomes :meth:`host`'s, and the paths of its trees and
        imported packages lie under the home, gpyreg's at ``gpyreg``.
        """
        identity["host"] = self.host(node, **task)
        imports = identity.setdefault("imports", {})
        for name, tree in (imports.get("trees") or {}).items():
            tree["path"] = str(
                self.gpyreg if name == "gpyreg" else self.home / "src" / name
            )
        for name in imports.get("modules") or {}:
            root = self.gpyreg if name == "gpyreg" else self.home / "src"
            imports["modules"][name] = str(root / name)
        return identity

    def write_slurm(self, out, job="4242"):
        """The task logs, the accounting and a submission in ``out/slurm``.

        The logs name the nodes and :attr:`log_node`, the accounting a node
        and the host list :attr:`accounting`, the submission the partition.
        """
        slurm = Path(out) / "slurm"
        slurm.mkdir(parents=True, exist_ok=True)
        for index, node in enumerate([*self.nodes, self.log_node], start=1):
            (slurm / f"{job}_{index}.out").write_text(
                f"task {job}_{index}, case {index}: g/c{index} {index}, "
                f"on {node}\ng/c{index}: complete in 1.0 s\n",
                encoding="utf-8",
            )
        (slurm / f"verify_{job}9.out").write_text(
            f"step 'verify' of {self.home.as_posix()}/runs: job {job}9 on "
            f"{self.nodes[0]}\n",
            encoding="utf-8",
        )
        (slurm / "sacct.txt").write_text(
            "JobID|JobName|State|ExitCode|Elapsed|MaxRSS|AllocCPUS|NodeList\n"
            f"{job}_1|c|COMPLETED|0:0|00:01:00|100M|2|"
            f"{self.nodes[0].split('.')[0]}\n"
            f"{job}_2|c|COMPLETED|0:0|00:01:00|100M|2|{self.accounting}\n"
            f"{job}_3|c|PENDING|0:0|00:00:00||2|None assigned\n",
            encoding="utf-8",
        )
        with open(slurm / "jobs.txt", "a", encoding="utf-8") as stream:
            stream.write(
                f"{job} array=1-3 offset=0 subset=- throttle=200 "
                f"time=00:30:00 mem=2G partition={self.partition} "
                "2026-10-01T10:00:00\n"
            )

    def leaks(self, directory):
        """``(file, token)`` for each of :data:`TOKENS` a file under
        ``directory`` holds."""
        found = []
        for path in sorted(Path(directory).rglob("*")):
            if path.is_file():
                text = path.read_bytes().decode("utf-8", "replace").lower()
                name = path.relative_to(directory).as_posix()
                found += [(name, t) for t in self.TOKENS if t in text]
        return found

    def rewrite(self, path, change):
        """Apply ``change`` to the JSON document at ``path`` and rewrite it
        as ``campaign_contract.write_json`` writes."""
        path = Path(path)
        value = json.loads(path.read_text(encoding="utf-8"))
        change(value)
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        return value
