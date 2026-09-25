"""Tests of the Slurm driver, ``dev/scripts/hpc/campaign_*``.

Each test copies the driver's scripts, ``campaign_contract.py`` and the stub
harness ``campaign_stub_harness.py`` into a scratch git repository, and runs
the scripts there against the stub ``sbatch``, ``squeue``, ``sacct``,
``scontrol`` and ``zstd`` of ``campaign_slurm_stubs.py``, placed first on
the PATH, with a stand-in for the campaign's conda environment whose
``python`` is this interpreter and a requirements file that pins what this
interpreter has installed. The stub ``sbatch`` records every submission
and runs the batch jobs submitted with ``--wait``; the tests run the array
tasks themselves, with the variables Slurm would set. They run under bash,
Git Bash on Windows, and never skip.

Outside default pytest discovery; run it by path::

    python -m pytest dev/scripts/test_campaign_driver.py -vv
"""

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import campaign_contract as contract
import campaign_slurm_stubs as stubs
import pytest

HERE = Path(__file__).resolve().parent
HPC = HERE / "hpc"
SCRIPTS = (
    "campaign_env.sh",
    "campaign_submit.sh",
    "campaign_task.sbatch",
    "campaign_finish.sh",
)
HARNESS = "dev/scripts/campaign_stub_harness.py"
#: A conda stand-in: `conda activate PREFIX` puts PREFIX/bin first on the
#: PATH (resolved by bash, since a drive letter's colon would split it),
#: and every call is recorded.
CONDA_STUB = """conda() {
    echo "conda $*" >> "$STUB_STATE/conda_calls"
    if [ "${1:-}" = activate ]; then
        PATH="$(cd "$2/bin" && pwd):$PATH"
        CONDA_PREFIX=$2
        export PATH CONDA_PREFIX
    fi
}"""
PROFILE = """module() { echo "module $*" >> "$STUB_STATE/module_calls"; }
export STUB_PROFILE_READ=1
"""
#: Variables of the calling shell that the driver reads, and that must not
#: leak into a test.
LEAKS = {
    "PYTHONPATH",
    "BASH_ENV",
    "ENV",
    "CAMPAIGN_DIR",
    "CAMPAIGN_STEP",
    "INDEX_OFFSET",
    "REPO",
    "CONDA_PREFIX",
    "VERIFY_TIME",
    "VERIFY_MEM",
    "FINISH_TIME",
    "FINISH_MEM",
    "ARCHIVE_PART_SIZE",
    *contract.SETTINGS,
}


def posix(path):
    return Path(path).as_posix()


def git_environment(home):
    return {
        "HOME": str(home),
        "GIT_CONFIG_GLOBAL": str(Path(home) / ".gitconfig"),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@example.invalid",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@example.invalid",
    }


def base_environment():
    return {
        key: value
        for key, value in os.environ.items()
        if key.upper() not in LEAKS
        and not key.upper().startswith(("SLURM", "BASH_FUNC_"))
    }


def git(path, *args, home):
    environment = dict(base_environment(), **git_environment(home))
    return subprocess.run(
        ["git", "-C", str(path), *args],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def make_repo(path, files, home):
    path.mkdir(parents=True)
    for name, text in files.items():
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        (path / name).write_text(text, encoding="utf-8")
    git(path, "init", "-q", home=home)
    git(path, "add", "-A", home=home)
    git(path, "commit", "-q", "-m", "fixture", home=home)
    return path


def environment_pins(scripts):
    """A requirements file pinning what this interpreter has installed."""
    environment = base_environment()
    environment["PYTHONNOUSERSITE"] = "1"
    listing = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, campaign_contract as c; "
            "print(json.dumps(c.installed_distributions()))",
        ],
        cwd=scripts,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    pins = {}
    for dist in json.loads(listing.stdout):
        if (
            contract._from_path(dist["direct_url"])
            or dist["installer"] == "conda"
        ):
            continue
        pins[
            contract.canonical_name(dist["name"])
        ] = f"{dist['name']}=={dist['version']}"
    python = subprocess.run(
        [
            sys.executable,
            "-c",
            "import platform; print(platform.python_version())",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return f"# python=={python}\n" + "".join(
        f"{pins[n]}\n" for n in sorted(pins)
    )


@pytest.fixture(scope="module")
def bash():
    return stubs.find_bash()


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    """A committed scratch repository with the driver and the stub harness."""
    root = tmp_path_factory.mktemp("template")
    home = root / "home"
    home.mkdir()
    (home / ".gitconfig").write_text("", encoding="utf-8")
    repo = root / "repo"
    (repo / "dev" / "scripts" / "hpc").mkdir(parents=True)
    for name in SCRIPTS:
        shutil.copy2(HPC / name, repo / "dev" / "scripts" / "hpc" / name)
    for name in ("campaign_contract.py", "campaign_stub_harness.py"):
        shutil.copy2(HERE / name, repo / "dev" / "scripts" / name)
    (repo / ".gitignore").write_text("dev/scripts/runs/\n", encoding="utf-8")
    requirements = (
        repo / "dev" / "scripts" / "hpc" / "campaign_requirements.txt"
    )
    requirements.write_text(
        environment_pins(repo / "dev" / "scripts"), "utf-8"
    )
    git(repo, "init", "-q", home=home)
    git(repo, "add", "-A", home=home)
    git(repo, "commit", "-q", "-m", "driver", home=home)
    return root


class World:
    """A scratch checkout, environment, stub Slurm and campaigns directory."""

    def __init__(self, root, template, bash):
        self.root = root
        self.bash = bash
        self.repo = root / "repo"
        shutil.copytree(template / "repo", self.repo)
        self.home = root / "home"
        shutil.copytree(template / "home", self.home)
        self.campaigns = root / "campaigns"
        self.campaigns.mkdir()
        self.state = root / "state"
        self.state.mkdir()
        self.stubs = stubs.write_stubs(root / "bin", bash)
        self.prefix = root / "env"
        stubs.write_script(
            self.prefix / "bin" / "python",
            f'#!/bin/bash\nexec "{posix(sys.executable)}" "$@"\n',
        )
        self.profile = root / "profile.sh"
        self.profile.write_bytes(PROFILE.encode("utf-8"))
        environment = stubs.stub_environment(
            self.stubs, self.state, base_environment()
        )
        environment.update(git_environment(self.home))
        environment.update(
            HARNESS=HARNESS,
            CAMPAIGN_ENV=posix(self.prefix),
            NODE_FEATURE="stubfeat",
            LOGIN_PROFILE=posix(self.profile),
            CONDA_SETUP=CONDA_STUB,
            STUB_FAKE_AFFINITY="1",
        )
        self.environ = environment

    def git(self, *args, path=None):
        return git(path or self.repo, *args, home=self.home)

    def campaign(self, name="c1"):
        return self.campaigns / name

    def environment(self, env=None):
        environment = dict(self.environ)
        for key, value in (env or {}).items():
            if value is None:
                environment.pop(key, None)
            else:
                environment[key] = str(value)
        return environment

    def script(self, name):
        return posix(self.repo / "dev" / "scripts" / "hpc" / name)

    def run(self, script, *args, env=None):
        return subprocess.run(
            [self.bash, self.script(script), *(str(a) for a in args)],
            env=self.environment(env),
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=900,
        )

    def start(self, command, env, log):
        """A process left running, its output in the file ``log``."""
        stream = open(log, "w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            env=self.environment(env),
            stdout=stream,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
        stream.close()
        return process

    def submit(self, name="c1", *prepare, **env):
        return self.run(
            "campaign_submit.sh", posix(self.campaign(name)), *prepare, env=env
        )

    def finish(self, name="c1", *flags, **env):
        return self.run(
            "campaign_finish.sh", posix(self.campaign(name)), *flags, env=env
        )

    def task_variables(self, name, array_task, offset, job, restart, env):
        """What Slurm and the submitter give one array task."""
        variables = {
            "CAMPAIGN_DIR": posix(self.campaign(name)),
            "REPO": posix(self.repo),
            "INDEX_OFFSET": offset,
            "SLURM_JOB_ID": f"{job}{array_task:03d}",
            "SLURM_ARRAY_JOB_ID": job,
            "SLURM_ARRAY_TASK_ID": array_task,
            "SLURMD_NODENAME": "node1",
        }
        if restart is not None:
            variables["SLURM_RESTART_COUNT"] = restart
        variables.update(env)
        return variables

    def task(self, name, array_task, offset=0, job=1001, restart=None, **env):
        variables = self.task_variables(
            name, array_task, offset, job, restart, env
        )
        return self.run("campaign_task.sbatch", env=variables)

    def calls(self):
        """Every sbatch call: its job id, arguments and exported variables."""
        found = []
        for folder in sorted(
            (self.state / "sbatch").glob("*"), key=lambda p: int(p.name)
        ):
            env = {}
            for line in (folder / "env").read_text("utf-8").splitlines():
                key, _, value = line.partition("=")
                env[key] = value
            args = (folder / "args").read_text("utf-8").splitlines()
            found.append({"job": folder.name, "args": args, "env": env})
        return found

    def answer(self, step, text):
        (self.state / "sacct").mkdir(exist_ok=True)
        (self.state / "sacct" / step).write_text(text, encoding="utf-8")

    def queue(self, job, text):
        (self.state / "squeue").mkdir(exist_ok=True)
        (self.state / "squeue" / str(job)).write_text(text, encoding="utf-8")

    def claim(self, name, tag, job, array_task):
        record = contract.new_claim(
            tag, task={"job": job, "array_task": array_task}
        )
        contract.write_json(
            contract.claim_path(self.campaign(name), tag), record
        )
        return record


@pytest.fixture
def world(tmp_path, template, bash):
    return World(tmp_path, template, bash)


def value(args, flag):
    return args[args.index(flag) + 1]


def covered(calls):
    """The case indices a set of array submissions runs."""
    indices = set()
    for call in calls:
        spec = next(a for a in call["args"] if a.startswith("--array="))
        spec = spec[len("--array=") :].split("%")[0]
        offset = int(call["env"]["INDEX_OFFSET"])
        for item in spec.split(","):
            first, _, last = item.partition("-")
            indices |= {
                i + offset for i in range(int(first), int(last or first) + 1)
            }
    return indices


def arrays(calls):
    return [
        (
            next(a for a in c["args"] if a.startswith("--array=")),
            c["env"]["INDEX_OFFSET"],
        )
        for c in calls
    ]


def snapshot(directory):
    return {
        path.relative_to(directory).as_posix(): (
            path.read_bytes(),
            path.stat().st_mtime_ns,
        )
        for path in sorted(Path(directory).rglob("*"))
        if path.is_file()
    }


def ok(result):
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def refused(result, message):
    assert result.returncode == 1, result.stdout + result.stderr
    assert (
        "refusing" in result.stderr and message in result.stderr
    ), result.stderr
    return result


# --------------------------------------------------------------------------
# The scripts
# --------------------------------------------------------------------------


def test_every_script_parses(bash):
    for name in SCRIPTS:
        result = subprocess.run(
            [bash, "-n", posix(HPC / name)], capture_output=True, text=True
        )
        assert result.returncode == 0, f"{name}: {result.stderr}"


# --------------------------------------------------------------------------
# Submission
# --------------------------------------------------------------------------


def test_submit_prepares_once_and_submits_every_case(world):
    result = ok(world.submit("c1", "--cases", "12"))
    out = world.campaign()
    manifest = json.loads((out / "manifest.json").read_text("utf-8"))
    assert manifest["site"]["NODE_FEATURE"] == "stubfeat"
    assert manifest["site"]["HARNESS"] == HARNESS
    assert manifest["site"]["PARTITION"] is None
    assert manifest["pip_freeze"]
    lines = (out / "cases.txt").read_text("utf-8").splitlines()
    assert (
        len(lines) == 12
        and lines[0] == "g0/c001 1"
        and lines[11] == "g2/c012 12"
    )
    [call] = world.calls()
    args = call["args"]
    assert value(args, "-C") == "stubfeat"
    assert value(args, "-J") == "c1"
    for flag in (
        "--hint=nomultithread",
        "--cpus-per-task=1",
        "--mem=2G",
        "--time=00:30:00",
        "--array=1-12%200",
        "--export=ALL",
        "--parsable",
    ):
        assert flag in args
    assert "-p" not in args
    assert value(args, "--output").endswith("/slurm/%A_%a.out")
    assert args[-1].endswith("campaign_task.sbatch")
    assert call["env"]["INDEX_OFFSET"] == "0"
    assert call["env"]["HARNESS"] == HARNESS
    assert call["env"]["CAMPAIGN_DIR"].endswith("/c1")
    jobs = (out / "slurm" / "jobs.txt").read_text("utf-8").split()
    assert jobs[:4] == ["1001", "array=1-12", "offset=0", "subset=-"]
    assert "takes 1 chunk(s) of at most 1000 cases" in result.stdout
    assert (out / "tmp").is_dir()


def test_submit_passes_the_partition_and_limits(world):
    ok(
        world.submit(
            "c1",
            PARTITION="part1",
            TIME="02:00:00",
            MEM="3G",
            THROTTLE="7",
            SBATCH_EXTRA="--qos=test --comment=x",
        )
    )
    [call] = world.calls()
    args = call["args"]
    assert value(args, "-p") == "part1"
    assert "--time=02:00:00" in args and "--mem=3G" in args
    assert "--array=1-12%7" in args
    assert "--qos=test" in args and "--comment=x" in args
    assert "partition=part1" in (
        world.campaign() / "slurm" / "jobs.txt"
    ).read_text("utf-8")


def test_submit_splits_below_max_array_size(world):
    (world.state / "max_array_size").write_text("4", encoding="utf-8")
    result = ok(world.submit("c1", "--cases", "10"))
    assert arrays(world.calls()) == [
        ("--array=1-3%200", "0"),
        ("--array=1-3%200", "3"),
        ("--array=1-3%200", "6"),
        ("--array=1%200", "9"),
    ]
    assert covered(world.calls()) == set(range(1, 11))
    assert "takes 4 chunk(s) of at most 3 cases" in result.stdout
    offsets = [
        line.split()[2]
        for line in (world.campaign() / "slurm" / "jobs.txt")
        .read_text("utf-8")
        .splitlines()
    ]
    assert offsets == ["offset=0", "offset=3", "offset=6", "offset=9"]


def test_array_names_case_indices_across_chunks(world):
    (world.state / "max_array_size").write_text("4", encoding="utf-8")
    ok(world.submit("c1", "--cases", "12", ARRAY="1"))
    assert arrays(world.calls()) == [("--array=1%200", "0")]
    ok(world.submit("c1", ARRAY="2,5-7,10"))
    later = world.calls()[1:]
    assert arrays(later) == [
        ("--array=2%200", "0"),
        ("--array=2-3%200", "3"),
        ("--array=1%200", "6"),
        ("--array=1%200", "9"),
    ]
    assert covered(later) == {2, 5, 6, 7, 10}


@pytest.mark.parametrize(
    "array, message",
    [
        ("0", "outside the 12 cases"),
        ("13", "outside the 12 cases"),
        ("7-5", "empty range"),
        ("x", "not a list"),
        ("3,,4", "not a list"),
    ],
)
def test_array_refusals(world, array, message):
    ok(world.submit("c1", ARRAY="1"))
    refused(world.submit("c1", ARRAY=array), message)
    assert len(world.calls()) == 1


def test_a_subset_is_its_own_array(world):
    ok(
        world.submit(
            "c1", "--cases", "12", CASES_SUBSET="odd", TIME="04:00:00"
        )
    )
    out = world.campaign()
    assert (out / "subsets" / "odd.txt").read_text("utf-8").splitlines() == [
        f"{i} g{(i - 1) // 5}/c{i:03d} {i}" for i in range(1, 13, 2)
    ]
    [call] = world.calls()
    assert "--array=1,3,5,7,9,11%200" in call["args"]
    assert "--time=04:00:00" in call["args"]
    assert "subset=odd" in (out / "slurm" / "jobs.txt").read_text("utf-8")
    ok(world.submit("c1", CASES_SUBSET="odd", ARRAY="3,5"))
    assert covered(world.calls()[1:]) == {3, 5}
    refused(
        world.submit("c1", CASES_SUBSET="odd", ARRAY="3,4"),
        "not in the subset odd",
    )
    refused(world.submit("c1", CASES_SUBSET="nosuch"), "no subset nosuch")
    assert not (out / "subsets" / "nosuch.txt").exists()
    (out / "subsets" / "odd.txt").write_text("1 g0/c001 1\n", encoding="utf-8")
    refused(world.submit("c1", CASES_SUBSET="odd"), "no longer matches")
    ok(world.submit("c1", CASES_SUBSET="g1"))
    assert covered(world.calls()[-1:]) == {6, 7, 8, 9, 10}
    assert len(world.calls()) == 3


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["HARNESS", "CAMPAIGN_ENV", "NODE_FEATURE"])
def test_submit_refuses_a_missing_setting(world, name):
    refused(world.submit("c1", **{name: None}), f"{name} is not set")
    assert not world.campaign().exists() and not world.calls()


def test_submit_refuses_a_harness_outside_the_checkout(world):
    refused(
        world.submit("c1", HARNESS="../elsewhere.py"), "inside the checkout"
    )
    refused(world.submit("c1", HARNESS="dev/scripts/absent.py"), "no harness")


@pytest.mark.parametrize("change", ["untracked", "modified"])
def test_submit_refuses_a_dirty_checkout(world, change):
    if change == "untracked":
        (world.repo / "stray.txt").write_text("x", encoding="utf-8")
    else:
        with (world.repo / ".gitignore").open("a", encoding="utf-8") as stream:
            stream.write("*.tmp\n")
    refused(world.submit("c1"), "uncommitted or untracked")
    assert not world.campaign().exists() and not world.calls()


def test_submit_checks_every_source_tree(world):
    gpyreg = make_repo(
        world.root / "gpyreg", {"gpyreg/__init__.py": ""}, world.home
    )
    ok(world.submit("c1", PYVBMC_GPYREG_SOURCE=posix(gpyreg)))
    manifest = json.loads(
        (world.campaign() / "manifest.json").read_text("utf-8")
    )
    assert manifest["site"]["PYVBMC_GPYREG_SOURCE"] == posix(gpyreg)
    (gpyreg / "gpyreg" / "__init__.py").write_text(
        "changed = 1\n", encoding="utf-8"
    )
    refused(
        world.submit("c1", PYVBMC_GPYREG_SOURCE=posix(gpyreg)),
        "PYVBMC_GPYREG_SOURCE",
    )
    plain = world.root / "plain"
    plain.mkdir()
    refused(
        world.submit("c2", PYVBMC_SOURCE=posix(plain)), "not a git checkout"
    )
    refused(
        world.submit("c2", PYVBMC_SOURCE=posix(world.repo / "dev")),
        "not the top",
    )
    baseline = world.root / "baseline"
    make_repo(baseline / "source", {"svbmc.py": ""}, world.home)
    (baseline / "deps").mkdir()
    ok(world.submit("c3", BASELINE_DIR=posix(baseline)))
    refused(
        world.submit("c4", BASELINE_DIR=posix(plain)), "holds no git checkout"
    )
    assert len(world.calls()) == 2


def test_submit_refuses_an_environment_that_differs(world):
    requirements = (
        world.repo / "dev" / "scripts" / "hpc" / "campaign_requirements.txt"
    )
    text = requirements.read_text("utf-8")
    text = re.sub(r"^pytest==.*$", "pytest==0.0.1", text, flags=re.MULTILINE)
    requirements.write_bytes(text.encode("utf-8"))
    world.git("commit", "-q", "-am", "another pin")
    result = refused(world.submit("c1"), "differs from")
    assert "pytest" in result.stderr and "0.0.1 is pinned" in result.stderr
    assert not (world.campaign() / "manifest.json").exists()
    assert not world.calls()


def test_submit_refuses_to_revise_a_prepared_campaign(world):
    ok(world.submit("c1", "--cases", "12"))
    refused(world.submit("c1", "--cases", "5"), "takes no prepare flags")
    assert len(world.calls()) == 1


def test_submit_refuses_a_changed_case_list(world):
    ok(world.submit("c1", "--cases", "4"))
    cases = world.campaign() / "cases.txt"
    lines = cases.read_text("utf-8").splitlines()
    cases.write_bytes(
        ("\n".join([lines[1], lines[0], *lines[2:]]) + "\n").encode()
    )
    refused(world.submit("c1"), "no longer matches")
    assert not (world.campaign() / "cases.txt.new").exists()
    assert len(world.calls()) == 1


def test_submit_refuses_a_changed_fixed_setting(world):
    ok(world.submit("c1"))
    refused(world.submit("c1", NODE_FEATURE="otherfeat"), "NODE_FEATURE")
    other = world.root / "other_env"
    shutil.copytree(world.prefix, other)
    refused(world.submit("c1", CAMPAIGN_ENV=posix(other)), "CAMPAIGN_ENV")
    ok(world.submit("c1", TIME="01:00:00", PARTITION="long"))
    assert len(world.calls()) == 2


def test_submit_refuses_without_max_array_size(world):
    (world.state / "scontrol_fail").write_text("", encoding="utf-8")
    refused(world.submit("c1"), "MaxArraySize")
    assert not world.calls()


def test_submit_refuses_a_campaign_directory_the_checkout_tracks(world):
    inside = world.repo / "inside"
    result = world.run("campaign_submit.sh", posix(inside))
    refused(result, "inside the checkout")
    assert not inside.exists()
    ignored = world.repo / "dev" / "scripts" / "runs" / "c1"
    ok(world.run("campaign_submit.sh", posix(ignored)))
    assert (ignored / "manifest.json").exists()


# --------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------


def test_a_task_runs_its_case_and_then_exits_at_once(world):
    ok(world.submit("c1", "--cases", "12"))
    result = ok(world.task("c1", 2, offset=3))
    assert "case 5: g0/c005 5" in result.stdout
    out = world.campaign()
    record = json.loads(
        (out / "records" / "g0" / "c005.complete.json").read_text("utf-8")
    )
    assert record["case"] == "g0/c005 5" and set(record["artifacts"]) == {
        "g0/c005.out"
    }
    host = record["identity"]["host"]
    assert host["node_features"]["available"] == ["stubfeat"]
    assert (
        host["slurm"]["array_task_id"] == "2"
        and host["slurm"]["array_job_id"] == "1001"
    )
    assert host["threads"] == {key: "1" for key in contract.THREAD_KEYS}
    assert not (out / "claims" / "g0" / "c005").exists()
    # The second run never activates the environment.
    again = ok(
        world.task(
            "c1", 2, offset=3, CAMPAIGN_ENV=posix(world.root / "absent")
        )
    )
    assert "already has a completion record; nothing to do" in again.stdout


def test_a_task_beyond_the_list_fails(world):
    ok(world.submit("c1", "--cases", "3"))
    result = world.task("c1", 4)
    assert result.returncode == 2 and "no case on line 4" in result.stderr


def test_a_failing_case_leaves_its_error_file_alone(world):
    ok(world.submit("c1", "--cases", "6", "--fail", "4"))
    result = world.task("c1", 4)
    assert result.returncode == 1
    out = world.campaign()
    text = (out / "g0" / "c004.error.txt").read_text("utf-8")
    assert (
        "RuntimeError: the stub fails case 4 on purpose" in text
        and "Traceback" in text
    )
    assert not (out / "g0" / "c004.out").exists()
    assert not (out / "claims" / "g0" / "c004").exists()
    assert not (out / "records" / "g0" / "c004.complete.json").exists()


def test_a_live_claim_refuses_the_task_and_leaves_the_case_untouched(world):
    ok(world.submit("c1", "--cases", "6"))
    out = world.campaign()
    world.claim("c1", "g1/c006", "900", "1")
    (out / "g1").mkdir(exist_ok=True)
    (out / "g1" / "c006.out").write_text(
        "the other task's\n", encoding="utf-8"
    )
    (out / "g1" / "c006.error.txt").write_text(
        "an earlier attempt\n", encoding="utf-8"
    )
    world.answer("900_1", "RUNNING\n")
    before = snapshot(out)
    result = world.task("c1", 6)
    assert result.returncode == contract.EXIT_CLAIMED, (
        result.stdout + result.stderr
    )
    assert "900_1" in result.stdout and "left as they are" in result.stdout
    assert snapshot(out) == before
    assert "900_1" in (world.state / "sacct_queries").read_text("utf-8")


@pytest.mark.parametrize("how", ["fails", "answers nothing"])
def test_a_claim_the_accounting_cannot_judge_is_live(world, how):
    ok(world.submit("c1", "--cases", "6"))
    world.claim("c1", "g1/c006", "900", "1")
    if how == "fails":
        (world.state / "sacct").mkdir(exist_ok=True)
        (world.state / "sacct" / "900_1.fail").write_text("", encoding="utf-8")
    before = snapshot(world.campaign())
    assert world.task("c1", 6).returncode == contract.EXIT_CLAIMED
    assert snapshot(world.campaign()) == before


def test_a_task_takes_over_a_stale_claim(world):
    ok(world.submit("c1", "--cases", "7"))
    out = world.campaign()
    held = world.claim("c1", "g1/c007", "900", "2")
    world.answer("900_2", "TIMEOUT\n")
    result = ok(world.task("c1", 7))
    assert "took over the stale claim of 900_2" in result.stdout
    retired = json.loads(
        (out / "claims" / "g1" / "c007.stale.900_2").read_text("utf-8")
    )
    assert retired["token"] == held["token"]
    assert not (out / "claims" / "g1" / "c007").exists()
    assert (out / "records" / "g1" / "c007.complete.json").exists()


def test_a_requeued_task_takes_over_its_own_claim(world):
    ok(world.submit("c1", "--cases", "8"))
    world.claim("c1", "g1/c008", "1001", "8")
    result = ok(world.task("c1", 8, restart=1))
    assert "took over the requeue claim of 1001_8" in result.stdout
    queries = world.state / "sacct_queries"
    assert not queries.exists() or "1001_8" not in queries.read_text("utf-8")
    record = json.loads(
        (world.campaign() / "records" / "g1" / "c008.complete.json").read_text(
            "utf-8"
        )
    )
    assert record["identity"]["host"]["slurm"]["restart_count"] == "1"


def test_a_task_refuses_another_source_identity(world):
    ok(world.submit("c1", "--cases", "3"))
    harness = world.repo / HARNESS
    harness.write_bytes(harness.read_bytes() + b"# a later commit\n")
    world.git("commit", "-q", "-am", "a later commit")
    before = snapshot(world.campaign())
    result = world.task("c1", 3)
    assert result.returncode == contract.EXIT_IDENTITY
    assert "campaign_stub_harness.py" in result.stdout
    assert snapshot(world.campaign()) == before


# --------------------------------------------------------------------------
# The finish
# --------------------------------------------------------------------------


def run_tasks(world, name, indices):
    for index in indices:
        ok(world.task(name, index))


def test_finish_verifies_runs_the_steps_and_archives(world):
    ok(world.submit("c1", "--cases", "6"))
    run_tasks(world, "c1", range(1, 7))
    result = ok(world.finish("c1"))
    out = world.campaign()
    report = json.loads((out / "verification.json").read_text("utf-8"))
    assert report["counts"]["verified"] == 6 and report["exit_code"] == 0
    assert (
        json.loads((out / "summary.json").read_text("utf-8"))["counts"][
            "verified"
        ]
        == 6
    )
    verify, summarize = world.calls()[1:]
    for call, step in ((verify, "verify"), (summarize, "summarize")):
        assert call["env"]["CAMPAIGN_STEP"] == step
        assert (
            "--wait" in call["args"]
            and value(call["args"], "-C") == "stubfeat"
        )
        assert "--hint=nomultithread" in call["args"]
        assert value(call["args"], "--output").endswith(
            f"/slurm/{step}_%j.out"
        )
    assert "--time=01:00:00" in verify["args"] and "--mem=2G" in verify["args"]
    steps = (out / "slurm" / "steps.txt").read_text("utf-8").splitlines()
    assert [line.split()[1:3] for line in steps] == [
        ["step=verify", "rc=0"],
        ["step=summarize", "rc=0"],
    ]
    assert "-j 1001" in (world.state / "sacct_calls").read_text("utf-8")
    assert (out / "slurm" / "sacct.txt").exists()
    # The archive: parts and their SHA-256, which restore the directory.
    parts = sorted(world.campaigns.glob("c1.tar.zst.[0-9][0-9][0-9]"))
    assert [p.name for p in parts] == ["c1.tar.zst.000"]
    listed = (world.campaigns / "c1.tar.zst.sha256").read_text("utf-8").split()
    # `sha256sum` marks a file read in binary mode with a star, as it does
    # under Git Bash; `sha256sum -c` reads either form.
    listed[1] = listed[1].lstrip("*")
    assert listed == [
        hashlib.sha256(parts[0].read_bytes()).hexdigest(),
        "c1.tar.zst.000",
    ]
    assert listed[0] in result.stdout
    with tarfile.open(parts[0]) as archive:  # the stub zstd copies
        names = archive.getnames()
    assert (
        "c1/manifest.json" in names
        and "c1/records/g0/c001.complete.json" in names
    )


def test_finish_splits_the_archive_into_parts(world):
    ok(world.submit("c1", "--cases", "2"))
    run_tasks(world, "c1", (1, 2))
    ok(world.finish("c1", ARCHIVE_PART_SIZE="4K"))
    parts = sorted(world.campaigns.glob("c1.tar.zst.[0-9][0-9][0-9]"))
    assert len(parts) > 1
    joined = world.root / "joined.tar"
    joined.write_bytes(b"".join(p.read_bytes() for p in parts))
    with tarfile.open(joined) as archive:
        assert "c1/cases.txt" in archive.getnames()
    listed = (
        (world.campaigns / "c1.tar.zst.sha256").read_text("utf-8").splitlines()
    )
    assert len(listed) == len(parts)


def test_finish_waits_while_tasks_are_queued(world):
    ok(world.submit("c1", "--cases", "3"))
    world.queue(1001, "1001_3 RUNNING\n")
    result = world.finish("c1")
    assert result.returncode == 1
    assert (
        "still queued or running" in result.stderr
        and "1001_3 RUNNING" in result.stderr
    )
    assert len(world.calls()) == 1
    assert (world.campaign() / "slurm" / "queued.txt").read_text(
        "utf-8"
    ) == "3 RUNNING\n"


def test_finish_counts_cases_in_flight_apart_from_missing_ones(world):
    (world.state / "max_array_size").write_text("4", encoding="utf-8")
    ok(world.submit("c1", "--cases", "6"))
    for index, offset in ((1, 0), (2, 0), (3, 0), (1, 3)):
        ok(
            world.task(
                "c1", index, offset=offset, job=1001 if offset == 0 else 1002
            )
        )
    out = world.campaign()
    # Case 5 is running (its task holds a live claim), case 6 is still
    # queued: the tasks of the second chunk, with offset 3.
    world.claim("c1", "g0/c005", "1002", "2")
    world.answer("1002_2", "RUNNING\n")
    world.queue(1002, "1002_2 RUNNING\n1002_3 PENDING\n")
    assert world.finish("c1", "--no-archive").returncode == 1
    result = ok(world.finish("c1", "--allow-running", "--no-archive"))
    assert (out / "slurm" / "queued.txt").read_text(
        "utf-8"
    ) == "5 RUNNING\n6 PENDING\n"
    report = json.loads((out / "verification.json").read_text("utf-8"))
    assert (
        report["counts"]["in_flight"] == 1 and report["counts"]["missing"] == 1
    )
    assert (
        "in_flight indices: 5" in result.stderr
        and "queued indices: 6" in result.stderr
    )
    assert (out / "summary.json").exists()
    archive = world.finish("c1", "--allow-running")
    assert archive.returncode == 1 and "not archiving" in archive.stderr
    assert not list(world.campaigns.glob("c1.tar.zst*"))
    # The queue empties while case 5 still runs: in flight by its claim alone.
    shutil.rmtree(world.state / "squeue")
    assert (
        world.finish("c1", "--no-archive").returncode
        == contract.FINISH_IN_FLIGHT
    )
    # Case 5 ends; case 6 never ran and is missing.
    ok(world.task("c1", 2, offset=3, job=1002, restart=1))
    missing = world.finish("c1", "--allow-running", "--no-archive")
    assert missing.returncode == contract.FINISH_MISSING
    assert (
        "missing indices: 6" in missing.stderr and "ARRAY=6" in missing.stderr
    )
    ok(world.finish("c1", "--allow-missing", "--no-archive"))


def wait_for(path, process, log, timeout=120):
    """Wait until a worker left running has written ``path``."""
    deadline = time.monotonic() + timeout
    while not Path(path).exists():
        if process.poll() is not None:
            pytest.fail(
                f"the worker exited {process.returncode} before its case "
                f"was under way:\n{Path(log).read_text('utf-8')}"
            )
        if time.monotonic() > deadline:
            process.kill()
            pytest.fail("the worker never got under way")
        time.sleep(0.05)


def test_a_task_stopped_by_sigterm_leaves_a_missing_case(world):
    """Slurm's SIGTERM, at the time limit or on scancel: clean up and exit."""
    ok(world.submit("c1", "--cases", "2"))
    ok(world.task("c1", 1))
    out = world.campaign()
    if sys.platform == "win32":
        # No process can send another SIGTERM on Windows: the worker sends
        # it to itself, and it reaches the same handler.
        result = world.task("c1", 2, STUB_SIGNAL_SELF=1)
        code, output = result.returncode, result.stdout + result.stderr
    else:
        log = world.root / "task2.log"
        variables = world.task_variables(
            "c1", 2, 0, 1001, None, {"STUB_HOLD": 1}
        )
        process = world.start(
            [world.bash, world.script("campaign_task.sbatch")], variables, log
        )
        wait_for(out / "g0" / "c002.out", process, log)
        # The task script execs the worker, so that the signal reaches it
        # as Slurm's does.
        process.send_signal(signal.SIGTERM)
        code = process.wait(timeout=60)
        output = log.read_text("utf-8")
    assert code == 128 + signal.SIGTERM, output
    assert "stopped by SIGTERM" in output
    for path in (
        out / "g0" / "c002.out",
        out / "g0" / "c002.error.txt",
        out / "claims" / "g0" / "c002",
        out / "records" / "g0" / "c002.complete.json",
    ):
        assert not path.exists(), path
    result = world.finish("c1", "--no-archive")
    assert result.returncode == contract.FINISH_MISSING, result.stderr
    assert "missing indices: 2" in result.stderr
    assert "ARRAY=2" in result.stderr
    ok(world.task("c1", 2))
    ok(world.finish("c1", "--no-archive"))


def test_a_killed_task_is_interrupted_and_its_resubmission_completes(world):
    """SIGKILL (KillWait passed, an out-of-memory kill): files, claim stay."""
    ok(world.submit("c1", "--cases", "2"))
    ok(world.task("c1", 1))
    out = world.campaign()
    log = world.root / "worker2.log"
    variables = world.task_variables("c1", 2, 0, 1001, None, {"STUB_HOLD": 1})
    # The worker itself, so that the kill reaches the process that holds
    # the case on every platform (on Windows, Git Bash runs an exec'd
    # program as a child).
    command = [
        sys.executable,
        posix(world.repo / HARNESS),
        "worker",
        "--out",
        posix(out),
        "--case",
        "g0/c002 2",
    ]
    process = world.start(command, variables, log)
    wait_for(out / "g0" / "c002.out", process, log)
    process.kill()
    process.wait(timeout=60)
    assert (out / "g0" / "c002.out").exists()
    assert (out / "claims" / "g0" / "c002").exists()
    assert not (out / "g0" / "c002.error.txt").exists()
    world.answer("1001_2", "OUT_OF_MEMORY\n")
    result = world.finish("c1", "--no-archive")
    assert result.returncode == contract.FINISH_MISSING, result.stderr
    assert "interrupted indices: 2" in result.stderr
    assert "ARRAY=2" in result.stderr and "TIME or MEM" in result.stderr
    report = json.loads((out / "verification.json").read_text("utf-8"))
    assert report["exit_code"] == 0
    assert report["cases"][1]["status"] == "interrupted"
    # The resubmission, a task of another job, takes over the stale claim
    # and removes the killed attempt's file before it runs.
    (out / "g0" / "c002.out").write_text("the killed attempt's\n", "utf-8")
    result = ok(world.task("c1", 2, job=1002))
    assert "took over the stale claim of 1001_2" in result.stdout
    assert "removed 1 files of an earlier attempt" in result.stdout
    assert (out / "g0" / "c002.out").read_text("utf-8") == "case 2\n"
    assert (out / "claims" / "g0" / "c002.stale.1001_2").exists()
    ok(world.finish("c1", "--no-archive"))


def test_finish_stops_on_a_failed_verification(world):
    ok(world.submit("c1", "--cases", "3"))
    run_tasks(world, "c1", (1, 3))
    # An artifact with neither a record nor a claim, which no path of the
    # contract leaves: even --allow-missing does not pass it.
    (world.campaign() / "g0" / "c002.out").write_text(
        "partial\n", encoding="utf-8"
    )
    result = world.finish("c1", "--allow-missing")
    assert result.returncode == 1
    assert (
        "partial indices: 2" in result.stderr
        and "verify exited 1" in result.stderr
    )
    assert not (world.campaign() / "summary.json").exists()
    assert len(world.calls()) == 2


def test_finish_reports_failed_cases_and_goes_on(world):
    ok(world.submit("c1", "--cases", "3", "--fail", "2"))
    run_tasks(world, "c1", (1, 3))
    assert world.task("c1", 2).returncode == 1
    result = ok(world.finish("c1", "--no-archive"))
    assert "failed indices: 2" in result.stderr
    assert (world.campaign() / "summary.json").exists()


def test_finish_refusals(world):
    ok(world.submit("c1", "--cases", "2"))
    run_tasks(world, "c1", (1, 2))
    refused(world.finish("c1", NODE_FEATURE="otherfeat"), "NODE_FEATURE")
    (world.repo / "stray.txt").write_text("x", encoding="utf-8")
    refused(world.finish("c1"), "uncommitted or untracked")
    (world.repo / "stray.txt").unlink()
    assert world.finish("c2").returncode == 64
    assert world.finish("c1", "--bogus").returncode == 64
    assert len(world.calls()) == 1


# --------------------------------------------------------------------------
# The environment script
# --------------------------------------------------------------------------

SHOW = r"""set -euo pipefail
source "$REPO/dev/scripts/hpc/campaign_env.sh"
echo "python=$(command -v python)"
echo "threads=$OMP_NUM_THREADS,$OPENBLAS_NUM_THREADS,$MKL_NUM_THREADS"
echo "backend=$MPLBACKEND usersite=$PYTHONNOUSERSITE"
echo "tmpdir=${TMPDIR:-}"
echo "profile=${STUB_PROFILE_READ:-no} module=$(type -t module || echo none)"
echo "flags=$-"
echo "trees=${PYVBMC_GPYREG_SOURCE:-}"
"""


def show(world, **env):
    environment = dict(world.environ, REPO=posix(world.repo))
    for key, value in env.items():
        if value is None:
            environment.pop(key, None)
        else:
            environment[key] = str(value)
    return subprocess.run(
        [world.bash, "-c", SHOW],
        env=environment,
        capture_output=True,
        text=True,
    )


def shown(result):
    assert result.returncode == 0, result.stdout + result.stderr
    return dict(line.split("=", 1) for line in result.stdout.splitlines())


def same_file(bash, a, b):
    script = 'cd "$(dirname "$1")" && pwd -P; cd "$(dirname "$2")" && pwd -P'
    result = subprocess.run(
        [bash, "-c", script, "x", a, b], capture_output=True, text=True
    )
    first, second = result.stdout.splitlines()
    return first == second


def test_the_environment_script_activates_and_exports(world):
    campaign = world.campaign()
    values = shown(
        show(
            world,
            CAMPAIGN_DIR=posix(campaign),
            PYVBMC_GPYREG_SOURCE="/x/gpyreg",
        )
    )
    assert same_file(
        world.bash, values["python"], posix(world.prefix / "bin" / "python")
    )
    assert values["threads"] == "1,1,1"
    assert values["backend"] == "Agg usersite=1"
    assert values["tmpdir"].endswith("/c1/tmp") and (campaign / "tmp").is_dir()
    assert values["profile"] == "1 module=function"
    assert "e" in values["flags"] and "u" in values["flags"]
    assert values["trees"] == "/x/gpyreg"
    calls = (world.state / "conda_calls").read_text("utf-8").splitlines()
    assert calls[-1] == f"conda activate {posix(world.prefix)}"


def test_the_environment_script_activates_by_path_without_conda(world):
    path = os.pathsep.join(stubs.tool_path() or ["/usr/bin", "/bin"])
    result = show(world, CONDA_SETUP=None, PATH=path)
    values = shown(result)
    assert "conda is not defined" in result.stderr
    assert same_file(
        world.bash, values["python"], posix(world.prefix / "bin" / "python")
    )
    assert not (world.state / "conda_calls").exists()


def test_the_environment_script_refuses_a_python_outside_the_environment(
    world,
):
    empty = world.root / "empty_env"
    empty.mkdir()
    result = show(world, CAMPAIGN_ENV=posix(empty))
    assert result.returncode != 0 and "not the environment's" in result.stderr
    result = show(world, CONDA_SETUP="false")
    assert result.returncode != 0 and "CONDA_SETUP failed" in result.stderr
    result = show(world, CAMPAIGN_ENV=None)
    assert result.returncode != 0 and "CAMPAIGN_ENV" in result.stderr


BUILD_CONDA = r"""conda() {
    echo "conda $*" >> "$STUB_STATE/conda_calls"
    case ${1:-} in
        create)
            echo create >> "$STUB_STATE/order"
            cp -r "$STUB_STATE/built/." "$4"
            ;;
        activate)
            PATH="$(cd "$2/bin" && pwd):$PATH"
            export PATH
            ;;
    esac
}"""


def test_the_environment_build(world):
    prefix = world.root / "built_env"
    stubs.write_script(
        world.state / "built" / "bin" / "python",
        '#!/bin/bash\necho "$*" >> "$STUB_STATE/python_calls"\n',
    )
    env = {
        "CAMPAIGN_ENV": posix(prefix),
        "CONDA_SETUP": BUILD_CONDA,
        "LOGIN_SETUP": 'echo login >> "$STUB_STATE/order"',
    }
    result = world.run("campaign_env.sh", "build", env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (world.state / "order").read_text("utf-8").split() == [
        "login",
        "create",
    ]
    pin = contract.read_requirements(
        world.repo / "dev" / "scripts" / "hpc" / "campaign_requirements.txt"
    )["python"]
    create = (world.state / "conda_calls").read_text("utf-8").splitlines()[0]
    assert create == (
        f"conda create -y -p {posix(prefix)} --override-channels "
        "-c conda-forge "
        f"python={pin} zstd gh"
    )
    calls = (world.state / "python_calls").read_text("utf-8").splitlines()
    assert calls[0].startswith("-m pip install -r ")
    assert calls[0].endswith("dev/scripts/hpc/campaign_requirements.txt")
    assert calls[1] == "-m pip list --format=freeze --exclude-editable"
    assert "campaign_contract.py check-env --requirements" in calls[2]
    again = world.run("campaign_env.sh", "build", env=env)
    assert again.returncode == 1 and "exists" in again.stderr
    assert world.run("campaign_env.sh").returncode == 64
