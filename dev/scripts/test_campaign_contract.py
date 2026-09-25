"""Tests of ``campaign_contract.py``: the claim, the identity and its host
part, the completion record, the worker sequence, the environment check,
the reconciliation of ``verify``'s states, and the tracked copies with
their redaction, on a campaign that the contract's own functions write at
a stand-in site (``campaign_slurm_stubs.FakeSite``).

Outside default pytest discovery; run it by path::

    python -m pytest dev/scripts/test_campaign_contract.py -vv

The claims ask the stub Slurm commands of ``campaign_slurm_stubs.py``,
placed first on the PATH, or a query function given in their place. No
test skips: on Windows the stubs run under Git Bash, and what the platform
lacks (the CPU affinity) is checked to be recorded as null there and to
raise on Linux.
"""

import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tomllib
from pathlib import Path

import campaign_contract as contract
import campaign_slurm_stubs as stubs
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LINUX = sys.platform.startswith("linux")
TASK_A = {
    "job": "500",
    "array_task": "3",
    "job_id": "503",
    "restart_count": 0,
    "node": "n1",
}
TASK_B = {
    "job": "600",
    "array_task": "7",
    "job_id": "607",
    "restart_count": 0,
    "node": "n2",
}
TAG = "g0/c001"


def answer(live, state=None):
    """A query function that answers every task alike."""

    def query(job, array_task):
        return {"live": live, "state": state, "detail": f"stub says {state}"}

    return query


def never(job, array_task):
    raise AssertionError("the accounting must not be asked")


@pytest.fixture
def slurm(tmp_path, monkeypatch):
    """Stub Slurm commands first on the PATH; returns their state directory."""
    state = tmp_path / "state"
    state.mkdir()
    directory = stubs.write_stubs(tmp_path / "bin")
    environment = stubs.stub_environment(directory, state)
    monkeypatch.setenv("PATH", environment["PATH"])
    monkeypatch.setenv("STUB_STATE", environment["STUB_STATE"])
    return state


def set_answer(state, step, text=None, fail=False):
    folder = state / "sacct"
    folder.mkdir(exist_ok=True)
    if fail:
        (folder / f"{step}.fail").write_text("", encoding="utf-8")
    else:
        (folder / step).write_text(text, encoding="utf-8")


def snapshot(directory):
    """Every file under a directory with its bytes and modification time."""
    return {
        path.relative_to(directory).as_posix(): (
            path.read_bytes(),
            path.stat().st_mtime_ns,
        )
        for path in sorted(Path(directory).rglob("*"))
        if path.is_file()
    }


def git_env(tmp_path):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    (home / ".gitconfig").write_text("", encoding="utf-8")
    environment = dict(os.environ)
    environment.update(
        HOME=str(home),
        GIT_CONFIG_GLOBAL=str(home / ".gitconfig"),
        GIT_CONFIG_NOSYSTEM="1",
        GIT_AUTHOR_NAME="test",
        GIT_AUTHOR_EMAIL="test@example.invalid",
        GIT_COMMITTER_NAME="test",
        GIT_COMMITTER_EMAIL="test@example.invalid",
    )
    return environment


def make_repo(path, files, environment):
    path.mkdir(parents=True)
    for name, text in files.items():
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        (path / name).write_text(text, encoding="utf-8")
    for command in (
        ["git", "init", "-q"],
        ["git", "add", "-A"],
        ["git", "commit", "-q", "-m", "fixture"],
    ):
        subprocess.run(command, cwd=path, env=environment, check=True)
    return path


# --------------------------------------------------------------------------
# Claims
# --------------------------------------------------------------------------


def test_a_fresh_claim_holds_its_task(tmp_path):
    claim = contract.acquire_claim(tmp_path, TAG, query=never, task=TASK_A)
    path = tmp_path / "claims" / "g0" / "c001"
    assert Path(claim["path"]) == path
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["job"] == "500" and record["array_task"] == "3"
    assert record["restart_count"] == 0
    assert record["host"] and record["started"] and record["token"]
    assert record["token"] == claim["token"]
    # The temporary file the claim was linked from is gone.
    assert [p.name for p in path.parent.iterdir()] == ["c001"]


def test_a_live_claim_refuses_a_second_worker(tmp_path):
    contract.acquire_claim(tmp_path, TAG, query=never, task=TASK_A)
    before = snapshot(tmp_path / "claims")
    with pytest.raises(contract.ClaimRefused, match="500_3"):
        contract.acquire_claim(
            tmp_path, TAG, query=answer(True, "RUNNING"), task=TASK_B
        )
    assert snapshot(tmp_path / "claims") == before


@pytest.mark.parametrize(
    "state",
    [
        "COMPLETED",
        "FAILED",
        "CANCELLED by 1234",
        "TIMEOUT",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
    ],
)
def test_a_stale_claim_is_retired_and_replaced(tmp_path, slurm, state):
    held = contract.acquire_claim(tmp_path, TAG, query=never, task=TASK_A)
    set_answer(slurm, "500_3", f"{state}\n")
    claim = contract.acquire_claim(tmp_path, TAG, task=TASK_B)
    folder = tmp_path / "claims" / "g0"
    retired = json.loads((folder / "c001.stale.500_3").read_text("utf-8"))
    assert retired["token"] == held["token"]
    current = json.loads((folder / "c001").read_text("utf-8"))
    assert current["token"] == claim["token"] and current["job"] == "600"
    assert current["previous"]["reason"] == "stale"
    assert current["previous"]["owner"] == "500_3"
    assert current["previous"]["slurm_state"] == state.split()[0]
    # The query is exactly the plan's.
    assert (slurm / "sacct_queries").read_text("utf-8").split() == ["500_3"]


@pytest.mark.parametrize(
    "state", ["PENDING", "RUNNING", "REQUEUED", "SUSPENDED", "RESIZING"]
)
def test_a_task_that_has_not_ended_holds_its_claim(tmp_path, slurm, state):
    contract.acquire_claim(tmp_path, TAG, query=never, task=TASK_A)
    set_answer(slurm, "500_3", f"{state}\n")
    before = snapshot(tmp_path / "claims")
    with pytest.raises(contract.ClaimRefused, match=state):
        contract.acquire_claim(tmp_path, TAG, task=TASK_B)
    assert snapshot(tmp_path / "claims") == before


@pytest.mark.parametrize("how", ["fails", "answers nothing", "is missing"])
def test_no_answer_from_the_accounting_means_live(
    tmp_path, slurm, monkeypatch, how
):
    contract.acquire_claim(tmp_path, TAG, query=never, task=TASK_A)
    if how == "fails":
        set_answer(slurm, "500_3", fail=True)
    elif how == "is missing":
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.setenv("PATH", str(empty))
    before = snapshot(tmp_path / "claims")
    with pytest.raises(contract.ClaimRefused):
        contract.acquire_claim(tmp_path, TAG, task=TASK_B)
    assert snapshot(tmp_path / "claims") == before
    status = contract.accounting_state("500", "3")
    assert status["live"] and status["state"] is None


def test_a_requeued_task_takes_over_its_own_claim(tmp_path):
    held = contract.acquire_claim(tmp_path, TAG, query=never, task=TASK_A)
    requeued = dict(TASK_A, restart_count=1, node="n9")
    claim = contract.acquire_claim(tmp_path, TAG, query=never, task=requeued)
    record = json.loads(Path(claim["path"]).read_text("utf-8"))
    assert record["restart_count"] == 1
    assert record["token"] == claim["token"] != held["token"]
    assert record["previous"]["reason"] == "requeue"
    assert record["previous"]["restart_count"] == 0
    assert not list((tmp_path / "claims" / "g0").glob("*.stale.*"))


def test_of_two_workers_only_one_retires_a_stale_claim(tmp_path):
    contract.acquire_claim(tmp_path, TAG, query=never, task=TASK_A)
    folder = tmp_path / "claims" / "g0"
    # Another worker has linked the stale claim to its retired name and not
    # yet removed it.
    os.link(folder / "c001", folder / "c001.stale.500_3")
    before = snapshot(tmp_path / "claims")
    with pytest.raises(contract.ClaimRefused, match="another worker"):
        contract.acquire_claim(
            tmp_path, TAG, query=answer(False, "TIMEOUT"), task=TASK_B
        )
    assert snapshot(tmp_path / "claims") == before


def test_retiring_undoes_a_link_to_another_claim(tmp_path):
    contract.acquire_claim(tmp_path, TAG, query=never, task=TASK_A)
    path = tmp_path / "claims" / "g0" / "c001"
    judged = dict(json.loads(path.read_text("utf-8")), token="another")
    assert not contract._retire(path, judged)
    assert path.exists()
    assert not list(path.parent.glob("*.stale.*"))


def test_release_removes_only_the_holders_claim(tmp_path):
    claim = contract.acquire_claim(tmp_path, TAG, query=never, task=TASK_A)
    other = contract.new_claim(TAG, task=TASK_B)
    contract.write_json(claim["path"], other)
    assert not contract.release_claim(claim)
    assert Path(claim["path"]).exists()
    contract.write_json(claim["path"], {**claim, "path": None})
    assert contract.release_claim(claim)
    assert not Path(claim["path"]).exists()


def test_a_claim_made_outside_slurm(tmp_path):
    contract.acquire_claim(tmp_path, "local", query=never, task={})
    record = json.loads((tmp_path / "claims" / "local").read_text("utf-8"))
    assert record["job"] is None and record["pid"] == os.getpid()
    # Its process runs on this host: live.
    with pytest.raises(contract.ClaimRefused, match="running"):
        contract.acquire_claim(tmp_path, "local", query=never, task=TASK_B)
    # A process that has exited: stale.
    done = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"],
        capture_output=True,
        text=True,
        check=True,
    )
    contract.write_json(
        tmp_path / "claims" / "local", dict(record, pid=int(done.stdout))
    )
    claim = contract.acquire_claim(tmp_path, "local", query=never, task=TASK_B)
    assert claim["previous"]["reason"] == "stale"
    # A claim made on another host is live until an operator removes it.
    contract.write_json(
        tmp_path / "claims" / "local", dict(record, host="elsewhere")
    )
    with pytest.raises(contract.ClaimRefused, match="elsewhere"):
        contract.acquire_claim(tmp_path, "local", query=never, task=TASK_A)


def test_claim_status(tmp_path):
    assert contract.claim_status(tmp_path, TAG)["state"] == "absent"
    contract.acquire_claim(tmp_path, TAG, query=never, task=TASK_A)
    live = contract.claim_status(tmp_path, TAG, query=answer(True, "RUNNING"))
    assert live["state"] == "live" and live["owner"] == "500_3"
    stale = contract.claim_status(tmp_path, TAG, query=answer(False, "FAILED"))
    assert stale["state"] == "stale" and stale["slurm_state"] == "FAILED"
    (tmp_path / "claims" / "g0" / "c001").write_text("{", encoding="utf-8")
    assert contract.claim_status(tmp_path, TAG, query=never)["state"] == "live"


def test_accounting_state_parses_the_answer(slurm):
    set_answer(slurm, "700_1", "CANCELLED by 0\n")
    assert contract.accounting_state("700", "1")["live"] is False
    set_answer(slurm, "700_2", "COMPLETED\nRUNNING\n")
    assert contract.accounting_state("700", "2")["live"] is True
    set_answer(slurm, "700_3", "SPECIAL_EXIT\n")
    assert contract.accounting_state("700", "3")["live"] is True
    set_answer(slurm, "701", "TIMEOUT\n")
    assert contract.accounting_state("701")["state"] == "TIMEOUT"


def test_slurm_task_from_the_environment():
    assert contract.slurm_task({}) is None
    array = contract.slurm_task(
        {
            "SLURM_JOB_ID": "812",
            "SLURM_ARRAY_JOB_ID": "800",
            "SLURM_ARRAY_TASK_ID": "12",
            "SLURM_RESTART_COUNT": "2",
        }
    )
    assert (array["job"], array["array_task"], array["restart_count"]) == (
        "800",
        "12",
        2,
    )
    single = contract.slurm_task({"SLURM_JOB_ID": "900"})
    assert (single["job"], single["array_task"]) == ("900", None)
    assert contract.task_step("900") == "900"


# --------------------------------------------------------------------------
# The worker sequence
# --------------------------------------------------------------------------


def worker_identity(harness_root):
    return contract.identity({"harness": harness_root}, ["a.txt"], host=False)


@pytest.fixture
def repo(tmp_path):
    return make_repo(
        tmp_path / "repo",
        {"a.txt": "a\n", ".gitignore": "out/\n"},
        git_env(tmp_path),
    )


def test_run_worker_completes_a_case(repo, tmp_path):
    out = tmp_path / "out"
    expected = worker_identity(repo)
    contract.error_path(out, TAG).parent.mkdir(parents=True)
    contract.error_path(out, TAG).write_text("an earlier attempt\n", "utf-8")

    def run(identity):
        path = out / f"{TAG}.out"
        path.write_text("result\n", encoding="utf-8")
        return [path], {"value": 1}

    code = contract.run_worker(
        out,
        f"{TAG} 1",
        expected,
        lambda: expected,
        run,
        lambda: [],
        query=never,
    )
    assert code == 0
    record = contract.check_completion(out, TAG, expected, [f"{TAG}.out"])
    assert record["case"] == f"{TAG} 1" and record["value"] == 1
    assert record["elapsed_seconds"] >= 0
    assert set(record["artifacts"]) == {f"{TAG}.out"}
    assert not contract.error_path(out, TAG).exists()
    assert not contract.claim_path(out, TAG).exists()


def test_run_worker_exits_at_once_on_a_completed_case(tmp_path):
    contract.write_json(contract.record_path(tmp_path, TAG), {"tag": TAG})

    def refuse():
        raise AssertionError("the identity must not be computed")

    assert contract.run_worker(tmp_path, TAG, {}, refuse, None, None) == 0


def test_run_worker_refuses_another_identity(repo, tmp_path):
    out = tmp_path / "out"
    (out / "g0").mkdir(parents=True)
    (out / f"{TAG}.out").write_text("partial\n", encoding="utf-8")
    expected = worker_identity(repo)
    actual = json.loads(json.dumps(expected))
    actual["source"]["files"]["a.txt"] = "0" * 64
    before = snapshot(out)
    code = contract.run_worker(
        out, TAG, expected, lambda: actual, None, None, query=never
    )
    assert code == contract.EXIT_IDENTITY
    assert snapshot(out) == before

    def broken():
        raise RuntimeError("git is gone")

    assert (
        contract.run_worker(out, TAG, expected, broken, None, None)
        == contract.EXIT_IDENTITY
    )
    assert snapshot(out) == before


def test_run_worker_refusal_leaves_the_case_untouched(repo, tmp_path):
    out = tmp_path / "out"
    (out / "g0").mkdir(parents=True)
    (out / f"{TAG}.out").write_text("another worker's\n", encoding="utf-8")
    contract.error_path(out, TAG).write_text("earlier\n", encoding="utf-8")
    contract.acquire_claim(out, TAG, query=never, task=TASK_A)
    expected = worker_identity(repo)
    before = snapshot(out)

    def run(identity):
        raise AssertionError("a refused case must not run")

    code = contract.run_worker(
        out,
        TAG,
        expected,
        lambda: expected,
        run,
        lambda: [out / f"{TAG}.out"],
        query=answer(True, "RUNNING"),
    )
    assert code == contract.EXIT_CLAIMED
    assert snapshot(out) == before


def test_run_worker_failure_path(repo, tmp_path):
    out = tmp_path / "out"
    expected = worker_identity(repo)

    def run(identity):
        path = out / f"{TAG}.out"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("half\n", encoding="utf-8")
        raise ValueError("the case diverged")

    code = contract.run_worker(
        out,
        TAG,
        expected,
        lambda: expected,
        run,
        lambda: [f"{TAG}.out"],
        query=never,
    )
    assert code == 1
    assert not (out / f"{TAG}.out").exists()
    text = contract.error_path(out, TAG).read_text("utf-8")
    assert text.startswith(f"{TAG}: ValueError: the case diverged")
    assert "Traceback" in text
    assert not contract.claim_path(out, TAG).exists()
    assert not contract.record_path(out, TAG).exists()


def test_run_worker_leaves_a_case_another_task_completed(repo, tmp_path):
    out = tmp_path / "out"
    expected = worker_identity(repo)

    def identity():
        # The other task writes its record while this one starts.
        contract.write_json(contract.record_path(out, TAG), {"tag": TAG})
        return expected

    def run(identity):
        raise AssertionError("a completed case must not run again")

    assert contract.run_worker(out, TAG, expected, identity, run, None) == 0
    assert not contract.claim_path(out, TAG).exists()


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_a_stop_signal_cleans_up_the_case(repo, tmp_path, signum):
    """The signal is delivered to this process, through its handler."""
    out = tmp_path / "out"
    expected = worker_identity(repo)
    before = {s: signal.getsignal(s) for s in contract.STOP_SIGNALS}

    def run(identity):
        path = out / f"{TAG}.out"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("half\n", encoding="utf-8")
        signal.raise_signal(signum)
        raise AssertionError("the signal must stop the run")

    code = contract.run_worker(
        out,
        TAG,
        expected,
        lambda: expected,
        run,
        lambda: [f"{TAG}.out"],
        query=never,
    )
    assert code == 128 + signum
    assert not (out / f"{TAG}.out").exists()
    assert not contract.claim_path(out, TAG).exists()
    assert not contract.error_path(out, TAG).exists()
    assert not contract.record_path(out, TAG).exists()
    assert {s: signal.getsignal(s) for s in contract.STOP_SIGNALS} == before
    report = contract.reconcile(out, [TAG], None, lambda tag: [], query=never)
    assert report["cases"][0]["status"] == "missing"


def test_a_stop_signal_after_the_record_leaves_the_case_complete(
    repo, tmp_path, monkeypatch
):
    out = tmp_path / "out"
    expected = worker_identity(repo)
    write = contract.write_completion

    def write_then_signal(*args, **kwargs):
        record = write(*args, **kwargs)
        signal.raise_signal(signal.SIGTERM)
        return record

    monkeypatch.setattr(contract, "write_completion", write_then_signal)

    def run(identity):
        path = out / f"{TAG}.out"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("whole\n", encoding="utf-8")
        return [path], None

    code = contract.run_worker(
        out,
        TAG,
        expected,
        lambda: expected,
        run,
        lambda: [f"{TAG}.out"],
        query=never,
    )
    assert code == 128 + signal.SIGTERM
    contract.check_completion(out, TAG, expected, [f"{TAG}.out"])
    assert not contract.claim_path(out, TAG).exists()


def test_a_resubmission_takes_over_and_starts_from_clean(repo, tmp_path):
    """After a kill: the stale claim is taken over, the old files go first."""
    out = tmp_path / "out"
    expected = worker_identity(repo)
    (out / "g0").mkdir(parents=True)
    (out / f"{TAG}.out").write_text("the killed attempt's\n", "utf-8")
    (out / f"{TAG}.aux").write_text("the killed attempt's\n", "utf-8")
    claim_by(out, TAG, "800", "1")
    seen = []

    def run(identity):
        seen.append(sorted(p.name for p in (out / "g0").iterdir()))
        path = out / f"{TAG}.out"
        path.write_text("the resubmission's\n", encoding="utf-8")
        return [path], None

    code = contract.run_worker(
        out,
        TAG,
        expected,
        lambda: expected,
        run,
        lambda: [f"{TAG}.out", f"{TAG}.aux"],
        query=answer(False, "TIMEOUT"),
    )
    assert code == 0
    assert seen == [[]]
    record = contract.check_completion(out, TAG, expected, [f"{TAG}.out"])
    assert set(record["artifacts"]) == {f"{TAG}.out"}
    assert (out / f"{TAG}.out").read_text("utf-8") == "the resubmission's\n"
    assert (out / "claims" / "g0" / "c001.stale.800_1").exists()
    assert not contract.claim_path(out, TAG).exists()


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------


def test_tree_state(tmp_path):
    environment = git_env(tmp_path)
    tree = make_repo(tmp_path / "tree", {"x.py": "x = 1\n"}, environment)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tree, capture_output=True, text=True
    ).stdout.strip()
    source, where = contract.tree_state(tree)
    assert source == {"commit": commit, "clean": True}
    assert where["path"] == str(tree.resolve()) and where["dirty"] == []
    # The status lines as git prints them: the first column, a space here,
    # is the index's status, the second the working tree's.
    (tree / "x.py").write_text("x = 2\n", encoding="utf-8")
    source, where = contract.tree_state(tree)
    assert source["clean"] is False and where["dirty"] == [" M x.py"]
    (tree / "new.py").write_text("", encoding="utf-8")
    assert contract.tree_state(tree)[1]["dirty"] == [" M x.py", "?? new.py"]
    (tree / "x.py").write_text("x = 1\n", encoding="utf-8")
    source, where = contract.tree_state(tree)
    assert source["clean"] is False and where["dirty"] == ["?? new.py"]
    (tree / "sub").mkdir()
    with pytest.raises(contract.IdentityError, match="not the top"):
        contract.tree_state(tree / "sub")
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(contract.IdentityError, match="not a git checkout"):
        contract.tree_state(plain)


def test_file_and_directory_hashes(tmp_path):
    (tmp_path / "data" / "truths").mkdir(parents=True)
    (tmp_path / "data" / "a.npz").write_bytes(b"a")
    (tmp_path / "data" / "truths" / "b.json").write_bytes(b"b")
    (tmp_path / "data" / "__pycache__").mkdir()
    (tmp_path / "data" / "__pycache__" / "x.pyc").write_bytes(b"x")
    (tmp_path / "h.py").write_bytes(b"h")
    hashes = contract.file_hashes(tmp_path, ["h.py", "data"])
    assert set(hashes) == {"h.py", "data/"}
    assert hashes["h.py"] == contract.sha256_file(tmp_path / "h.py")
    first = hashes["data/"]
    (tmp_path / "data" / "__pycache__" / "x.pyc").write_bytes(b"y")
    assert contract.directory_sha256(tmp_path / "data") == first
    (tmp_path / "data" / "truths" / "b.json").write_bytes(b"c")
    second = contract.directory_sha256(tmp_path / "data")
    assert second != first
    (tmp_path / "data" / "truths" / "b.json").rename(
        tmp_path / "data" / "truths" / "c.json"
    )
    assert contract.directory_sha256(tmp_path / "data") not in (first, second)
    with pytest.raises(contract.IdentityError):
        contract.file_hashes(tmp_path, ["absent.py"])


def test_versions_come_from_the_imported_modules(monkeypatch):
    import cma

    monkeypatch.setattr(cma, "__version__", "0.0.imported")
    versions = contract.imported_versions(
        ("numpy", "scipy", "cma"), ("torch",)
    )
    assert versions["cma"] == "0.0.imported"
    assert contract.installed_version("cma") != "0.0.imported"
    assert versions["python"] == platform.python_version()
    fake = type(sys)("fake_optional_module")
    fake.__version__ = "2.14.0+cpu"
    monkeypatch.setitem(sys.modules, "fake_optional_module", fake)
    versions = contract.imported_versions(
        (), ("fake_optional_module", "absent_mod")
    )
    assert versions["fake_optional_module"] == "2.14.0+cpu"
    assert versions["absent_mod"] is None


def test_identity_records_trees_imports_and_metadata(tmp_path, monkeypatch):
    environment = git_env(tmp_path)
    tree = make_repo(
        tmp_path / "tree",
        {"fakepkg_campaign/__init__.py": "__version__ = '1'\n", "t.py": ""},
        environment,
    )
    other = make_repo(tmp_path / "other", {"o.py": ""}, environment)
    monkeypatch.syspath_prepend(str(tree))
    record = contract.identity(
        {"harness": tree, "gpyreg": other},
        ["t.py"],
        modules={"fakepkg_campaign": "harness"},
        required=(),
        optional=(),
        host=False,
    )
    assert set(record["source"]["trees"]) == {"harness", "gpyreg"}
    assert record["source"]["files"] == {
        "t.py": contract.sha256_file(tree / "t.py")
    }
    assert record["imports"]["modules"]["fakepkg_campaign"] == str(
        (tree / "fakepkg_campaign").resolve()
    )
    assert record["imports"]["installed_metadata_versions"] == {
        "fakepkg_campaign": None
    }
    assert "host" not in record
    with pytest.raises(contract.IdentityError, match="not from the tree"):
        contract.identity(
            {"harness": tree, "gpyreg": other},
            modules={"fakepkg_campaign": "gpyreg"},
            required=(),
            optional=(),
            host=False,
        )


def test_source_differences():
    base = {
        "source": {
            "trees": {"harness": {"commit": "a", "clean": True}},
            "files": {"h.py": "1"},
            "versions": {"numpy": "2.5.2", "torch": None},
        },
        "imports": {"trees": {"harness": {"path": "/x"}}},
        "host": {"hostname": "n1"},
    }
    other = json.loads(json.dumps(base))
    other["imports"]["trees"]["harness"]["path"] = "/y"
    other["host"]["hostname"] = "n2"
    assert contract.source_differences(other, base) == []
    other["source"]["trees"]["harness"]["commit"] = "b"
    other["source"]["versions"]["torch"] = "2.14.0+cpu"
    del other["source"]["files"]["h.py"]
    assert contract.source_differences(other, base) == [
        "files.h.py",
        "trees.harness.commit",
        "versions.torch",
    ]


# --------------------------------------------------------------------------
# Host part
# --------------------------------------------------------------------------


def test_host_part_outside_slurm(monkeypatch):
    for key in list(os.environ):
        if key.startswith("SLURM"):
            monkeypatch.delenv(key)
    host = contract.host_part(strict=False)
    assert host["hostname"] and host["cpu_model"]
    assert host["node_features"] is None
    assert set(host["threads"]) == set(contract.THREAD_KEYS)
    assert all(value is None for value in host["slurm"].values())
    assert isinstance(host["blas"], list)
    if LINUX:
        assert host["cpu_affinity"]["cpus"]
        assert host["cpu_affinity"]["physical_cores"]
    else:
        assert host["cpu_affinity"] is None


def test_host_part_is_strict_on_linux(monkeypatch):
    for key in list(os.environ):
        if key.startswith("SLURM"):
            monkeypatch.delenv(key)
    if LINUX:
        host = contract.host_part(strict=True)
        assert any(entry["user_api"] == "blas" for entry in host["blas"])
        assert host["cpu_affinity"]["physical_cores"]
    else:
        with pytest.raises(contract.HostError, match="sched_getaffinity"):
            contract.host_part(strict=True)


def test_node_features_from_scontrol(slurm, monkeypatch):
    environment = {"SLURM_JOB_ID": "5", "SLURMD_NODENAME": "node7"}
    (slurm / "features").write_text("amd,epyc\n", encoding="utf-8")
    features = contract.node_features(True, environment)
    assert features == {
        "node": "node7",
        "available": ["amd", "epyc"],
        "active": ["amd", "epyc"],
    }
    (slurm / "features").write_text("(null)\n", encoding="utf-8")
    assert contract.node_features(True, environment)["available"] == []
    (slurm / "scontrol_node_fail").write_text("", encoding="utf-8")
    assert contract.node_features(False, environment) is None
    with pytest.raises(contract.HostError, match="node7"):
        contract.node_features(True, environment)
    assert contract.node_features(True, {}) is None


def test_host_problems():
    good = {
        "node_features": {
            "node": "n",
            "available": ["amd", "x"],
            "active": [],
        },
        "cpu_affinity": {"cpus": [4, 68], "physical_cores": ["0:4"]},
    }
    assert contract.host_problems(good, "amd") == []
    assert (
        "lacks the feature intel" in contract.host_problems(good, "intel")[0]
    )
    two = dict(
        good, cpu_affinity={"cpus": [4, 5], "physical_cores": ["0:4", "0:5"]}
    )
    assert "2 physical cores" in contract.host_problems(two, "amd")[0]
    bare = {"node_features": None, "cpu_affinity": None}
    assert len(contract.host_problems(bare, "amd")) == 2


# --------------------------------------------------------------------------
# Completion records
# --------------------------------------------------------------------------


def identity_with_host(features=("stubfeat",), cores=("0:3",)):
    return {
        "source": {"trees": {"harness": {"commit": "a", "clean": True}}},
        "host": {
            "node_features": {
                "node": "n",
                "available": list(features),
                "active": [],
            },
            "cpu_affinity": {"cpus": [3], "physical_cores": list(cores)},
        },
    }


def test_completion_record_round_trip_and_checks(tmp_path):
    identity = identity_with_host()
    (tmp_path / "g0").mkdir()
    (tmp_path / "g0" / "c001.npz").write_bytes(b"npz")
    (tmp_path / "g0" / "c001.json").write_bytes(b"{}")
    record = contract.write_completion(
        tmp_path,
        TAG,
        f"{TAG} 1",
        [tmp_path / "g0" / "c001.npz", "g0/c001.json"],
        identity,
        started=0.0,
        elapsed_seconds=1.5,
        extra={"elbo": -1.0},
    )
    assert set(record["artifacts"]) == {"g0/c001.npz", "g0/c001.json"}
    assert record["artifacts"]["g0/c001.npz"]["bytes"] == 3
    checked = contract.check_completion(
        tmp_path, TAG, identity, ["g0/c001.npz"], node_feature="stubfeat"
    )
    assert checked["elbo"] == -1.0

    with pytest.raises(contract.CompletionError, match="lists no"):
        contract.check_completion(tmp_path, TAG, identity, ["g0/c001.pkl"])
    moved = json.loads(json.dumps(identity))
    moved["source"]["trees"]["harness"]["commit"] = "b"
    with pytest.raises(contract.CompletionError, match="trees.harness.commit"):
        contract.check_completion(tmp_path, TAG, moved)
    with pytest.raises(contract.CompletionError, match="feature amd"):
        contract.check_completion(tmp_path, TAG, identity, node_feature="amd")
    (tmp_path / "g0" / "c001.npz").write_bytes(b"changed")
    with pytest.raises(contract.CompletionError, match="SHA-256"):
        contract.check_completion(tmp_path, TAG, identity)
    (tmp_path / "g0" / "c001.npz").unlink()
    with pytest.raises(contract.CompletionError, match="missing"):
        contract.check_completion(tmp_path, TAG, identity)
    with pytest.raises(contract.CompletionError, match="belongs to"):
        record_elsewhere = contract.record_path(tmp_path, "g0/c002")
        record_elsewhere.write_text(
            contract.record_path(tmp_path, TAG).read_text("utf-8"), "utf-8"
        )
        contract.check_completion(tmp_path, "g0/c002", identity)


def test_completion_record_refusals(tmp_path):
    (tmp_path / "a.out").write_bytes(b"a")
    with pytest.raises(contract.ContractError, match="reuse"):
        contract.write_completion(
            tmp_path, "a", "a", ["a.out"], {}, 0.0, 1.0, {"artifacts": 1}
        )
    with pytest.raises(contract.ContractError, match="at least one"):
        contract.write_completion(tmp_path, "a", "a", [], {}, 0.0, 1.0)
    outside = tmp_path.parent / f"{tmp_path.name}_outside.out"
    outside.write_bytes(b"x")
    with pytest.raises(contract.ContractError, match="outside"):
        contract.write_completion(tmp_path, "a", "a", [outside], {}, 0.0, 1.0)


# --------------------------------------------------------------------------
# Environment and settings
# --------------------------------------------------------------------------


def dist(name, version, installer="pip", direct=None, location="/site"):
    return {
        "name": name,
        "version": version,
        "installer": installer,
        "direct_url": direct,
        "location": location,
    }


def write_requirements(tmp_path, text):
    path = tmp_path / "requirements.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_read_requirements(tmp_path):
    path = write_requirements(
        tmp_path,
        "# a comment\n# python==3.12\n\nNumPy==2.5.2  # inline\n"
        "--extra-index-url https://download.pytorch.org/whl/cpu\n"
        "torch==2.14.0+cpu\n"
        "pywin32==1; sys_platform == 'nonexistent'\n",
    )
    pins = contract.read_requirements(path)
    assert pins == {
        "python": "3.12",
        "packages": {"numpy": "2.5.2", "torch": "2.14.0+cpu"},
    }
    for text, message in (
        ("numpy>=2\n", "not pinned"),
        ("numpy==2.*\n", "not pinned"),
        ("numpy==2\nnumpy==3\n", "twice"),
        ("-r other.txt\n", "unsupported"),
    ):
        with pytest.raises(contract.ContractError, match=message):
            contract.read_requirements(write_requirements(tmp_path, text))


def test_environment_differences(tmp_path):
    path = write_requirements(
        tmp_path,
        "# python==3.12\nnumpy==2.5.2\ntorch==2.14.0+cpu\nscipy==1.18.1\n",
    )
    editable = {"url": "file:///src/pyvbmc", "dir_info": {"editable": True}}
    wheel = {"url": "file:///wheels/x.whl", "archive_info": {}}
    good = [
        dist("numpy", "2.5.2"),
        dist("torch", "2.14.0+cpu"),
        dist("SciPy", "1.18.1"),
        dist("PyVBMC", "1.5.0.dev3", direct=editable),
        dist("localthing", "0.1", direct=wheel),
        dist("pip", "25.2", installer="conda"),
    ]
    assert contract.environment_differences(path, good, "3.12.14") == []
    assert contract.environment_differences(path, good, "3.13.0") == [
        "Python 3.13.0 runs, 3.12 is pinned"
    ]
    changed = good[1:] + [dist("numpy", "2.5.3"), dist("extra", "1.0")]
    problems = contract.environment_differences(path, changed, "3.12.6")
    assert problems == [
        "extra 1.0 is installed but not pinned",
        "numpy 2.5.3 is installed, 2.5.2 is pinned",
    ]
    missing = [d for d in good if d["name"] != "SciPy"]
    assert contract.environment_differences(path, missing, "3.12.6") == [
        "scipy==1.18.1 is pinned but not installed"
    ]
    from_path = (
        good[:2] + [dist("scipy", "1.18.1", direct=editable)] + good[3:]
    )
    assert contract.environment_differences(path, from_path, "3.12.6") == [
        "scipy is pinned but installed from a path"
    ]
    twice = good + [dist("numpy", "2.4.0", location="/user-site")]
    assert (
        "numpy is installed 2 times"
        in contract.environment_differences(path, twice, "3.12.6")[0]
    )


def test_local_version_labels_follow_pep_440(tmp_path):
    public = write_requirements(tmp_path, "torch==2.14.0\n")
    assert (
        contract.environment_differences(
            public, [dist("torch", "2.14.0+cpu")], "3.12.6"
        )
        == []
    )
    local = write_requirements(tmp_path, "torch==2.14.0+cpu\n")
    assert contract.environment_differences(
        local, [dist("torch", "2.14.0")], "3.12.6"
    ) == ["torch 2.14.0 is installed, 2.14.0+cpu is pinned"]


def test_the_campaign_requirements_file():
    """The tracked file parses and pins PyVBMC's and gpyreg's dependencies."""
    path = HERE / "hpc" / "campaign_requirements.txt"
    pins = contract.read_requirements(path)
    assert pins["python"].split(".")[:2] == ["3", "12"]
    assert pins["packages"]["torch"] == "2.14.0+cpu"
    project = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    from packaging.requirements import Requirement

    needed = {
        contract.canonical_name(Requirement(line).name)
        for line in project["project"]["dependencies"]
        + project["project"]["optional-dependencies"]["test"]
    }
    needed -= {"gpyreg"}
    needed |= {"numdifftools", "psutil", "packaging", "torch"}
    assert needed <= set(pins["packages"])
    for line in project["project"]["dependencies"]:
        requirement = Requirement(line)
        name = contract.canonical_name(requirement.name)
        if name in pins["packages"]:
            assert requirement.specifier.contains(pins["packages"][name])
    text = path.read_text(encoding="utf-8")
    assert "--extra-index-url https://download.pytorch.org/whl/cpu" in text


def test_installed_distributions_describe_this_environment():
    found = contract.installed_distributions()
    names = {contract.canonical_name(d["name"]) for d in found}
    assert {"numpy", "packaging", "pytest"} <= names
    assert all(
        {"version", "installer", "direct_url", "location"} <= set(d)
        for d in found
    )


def test_site_block_and_differences(tmp_path):
    environment = {
        "HARNESS": "dev/scripts/x.py",
        "CAMPAIGN_ENV": str(tmp_path / "env"),
        "NODE_FEATURE": "amd",
        "TIME": "01:00:00",
        "PARTITION": "",
    }
    site = contract.site_block(environment)
    assert set(site) == set(contract.SETTINGS)
    assert site["PARTITION"] is None and site["TIME"] == "01:00:00"
    later = dict(environment, TIME="04:00:00", PARTITION="long")
    later["CAMPAIGN_ENV"] = str(tmp_path / "x" / ".." / "env")
    assert contract.site_differences(site, later) == []
    moved = dict(environment, NODE_FEATURE="intel")
    del moved["HARNESS"]
    assert contract.site_differences(site, moved) == [
        "HARNESS is None here but 'dev/scripts/x.py' in the manifest",
        "NODE_FEATURE is 'intel' here but 'amd' in the manifest",
    ]


def test_finishing_steps():
    assert contract.finishing_steps({}) == []
    steps = [["select"], ["summarize", "--target", "320"]]
    assert contract.finishing_steps({"finishing_steps": steps}) == steps
    for bad in ([[]], [["select;rm"]], [["select", "--out", "x"]], "select"):
        with pytest.raises(contract.ContractError):
            contract.finishing_steps({"finishing_steps": bad})


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------


def test_case_tags():
    assert contract.case_tag("gmm_D2_svbmc/gmm_D2_svbmc_seed1000 x 1") == (
        "gmm_D2_svbmc/gmm_D2_svbmc_seed1000"
    )
    assert contract.case_tag("multisensory_s1_D6_noise1.3_seed4") == (
        "multisensory_s1_D6_noise1.3_seed4"
    )
    for bad in (
        "",
        " x",
        "../x",
        "a/../b",
        ".hidden",
        "a//b",
        "/abs",
        "a\tb",
        "a b\r",
    ):
        with pytest.raises(contract.ContractError):
            contract.case_tag(bad)


def test_read_cases_and_subsets(tmp_path):
    cases = tmp_path / "cases.txt"
    cases.write_bytes(b"a 1\nb 2\nc 3\n")
    lines = contract.read_cases(cases)
    assert lines == ["a 1", "b 2", "c 3"]
    subset = tmp_path / "s.txt"
    subset.write_bytes(b"1 a 1\n3 c 3\n")
    assert contract.read_subset(subset, lines) == [1, 3]
    for text in (b"2 a 1\n", b"3 c 3\n1 a 1\n", b"x a 1\n", b""):
        subset.write_bytes(text)
        with pytest.raises(contract.ContractError):
            contract.read_subset(subset, lines)
    for text in (b"a 1\na 2\n", b"a 1\r\nb 2\r\n", b"a 1", b""):
        cases.write_bytes(text)
        with pytest.raises(contract.ContractError):
            contract.read_cases(cases)


def test_compress_indices():
    assert contract.compress_indices([]) == ""
    assert contract.compress_indices([3, 1, 2, 7, 9, 10, 10]) == "1-3,7,9-10"


# --------------------------------------------------------------------------
# Reconciliation and the finish's decision
# --------------------------------------------------------------------------


def claim_by(out, tag, job, task):
    contract.write_json(
        contract.claim_path(out, tag),
        contract.new_claim(tag, task={"job": job, "array_task": task}),
    )


def test_reconcile_places_every_case(tmp_path):
    out = tmp_path
    identity = identity_with_host()
    lines = [f"g{(i - 1) // 5}/c{i:03d} {i}" for i in range(1, 11)]
    tags = [line.split()[0] for line in lines]

    def artifact(tag, text="x"):
        path = out / f"{tag}.out"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def complete(tag):
        contract.write_completion(
            out, tag, tag, [artifact(tag)], identity, 0.0, 1.0
        )

    complete(tags[0])  # 1 verified
    complete(tags[1])  # 2 verify_failed: its artifact changed afterwards
    artifact(tags[1], "changed")
    contract.error_path(out, tags[2]).write_text("c003: E: x\nmore\n", "utf-8")
    claim_by(out, tags[3], "900", "4")  # 4 in flight, with partial files
    artifact(tags[3])
    contract.error_path(out, tags[3]).write_text("earlier\n", "utf-8")
    artifact(tags[4])  # 5 partial
    claim_by(out, tags[5], "901", "6")  # 6 interrupted: a stale claim alone
    # 7 missing
    artifact(tags[7])  # 8 partial, beside an earlier error file
    contract.error_path(out, tags[7]).write_text("earlier\n", "utf-8")
    complete(tags[8])  # 9 verified, although a stale claim remains
    claim_by(out, tags[8], "901", "9")
    claim_by(out, tags[9], "900", "10")  # 10 in flight
    # A retired claim of an allocated case, and temporary files: not strays.
    (out / "claims" / "g1" / "c006.stale.800_1").write_text("{}", "utf-8")
    (out / "claims" / "g1" / ".c006.abc.tmp").write_text("{}", "utf-8")
    (out / "records" / "g0" / ".c001.complete.json.x.tmp").write_text(
        "", "utf-8"
    )
    # Strays: a record, a claim and an error file of no case, and a group
    # the allocation does not name.
    (out / "records" / "g0" / "c099.complete.json").write_text("{}", "utf-8")
    (out / "claims" / "g0" / "c099").write_text("{}", "utf-8")
    (out / "g1" / "c099.error.txt").write_text("x", "utf-8")
    (out / "records" / "g7").mkdir()

    asked = []

    def query(job, array_task):
        asked.append((job, array_task))
        live = job == "900"
        return {
            "live": live,
            "state": "RUNNING" if live else "TIMEOUT",
            "detail": "",
        }

    def check(tag):
        contract.check_completion(
            out, tag, identity, [f"{tag}.out"], "stubfeat"
        )
        return {"checked": True}

    def partial(tag):
        return [f"{tag}.out"] if (out / f"{tag}.out").exists() else []

    report = contract.reconcile(
        out, lines, check, partial, stray=["g1/c099.out"], query=query
    )
    statuses = [case["status"] for case in report["cases"]]
    assert statuses == [
        "verified",
        "verify_failed",
        "failed",
        "in_flight",
        "partial",
        "interrupted",
        "missing",
        "partial",
        "verified",
        "in_flight",
    ]
    cases = report["cases"]
    assert cases[0]["checked"] is True
    assert "SHA-256" in cases[1]["error"]
    assert cases[2]["reason"] == "c003: E: x"
    assert cases[3]["claim"]["owner"] == "900_4"
    assert cases[4]["files"] == ["g0/c005.out"]
    assert cases[5]["claim"]["state"] == "stale" and cases[5]["files"] == []
    assert "claim" not in cases[6]
    assert [case["index"] for case in cases] == list(range(1, 11))
    assert report["stray"] == [
        "claims/g0/c099",
        "g1/c099.error.txt",
        "g1/c099.out",
        "records/g0/c099.complete.json",
        "records/g7/",
    ]
    assert report["counts"] == {
        "verified": 2,
        "verify_failed": 1,
        "failed": 1,
        "in_flight": 2,
        "interrupted": 1,
        "partial": 2,
        "missing": 1,
        "stray": 5,
    }
    assert report["exit_code"] == 1
    # One question per task, the record of case 9 asking none.
    assert sorted(asked) == [("900", "10"), ("900", "4"), ("901", "6")]


def test_reconcile_exits_zero_on_failed_and_missing_cases(tmp_path):
    lines = ["a 1", "b 2"]
    contract.error_path(tmp_path, "a").write_text("a: E\n", "utf-8")
    report = contract.reconcile(tmp_path, lines, None, lambda tag: [])
    assert report["counts"]["failed"] == 1 and report["counts"]["missing"] == 1
    assert report["exit_code"] == 0


def test_a_case_killed_outright_is_interrupted_and_resubmitted(tmp_path):
    """Files and a stale claim, as SIGKILL leaves them: not fatal."""
    out = tmp_path
    lines = ["g0/c001 1", "g0/c002 2", "g0/c003 3"]
    (out / "g0").mkdir()
    # Case 1: killed outright; its task has ended.
    (out / "g0" / "c001.out").write_text("half\n", encoding="utf-8")
    claim_by(out, "g0/c001", "800", "1")
    # Case 3: files with neither a record nor a claim, which no path of
    # the contract leaves.
    (out / "g0" / "c003.out").write_text("half\n", encoding="utf-8")

    def query(job, array_task):
        return {"live": False, "state": "OUT_OF_MEMORY", "detail": ""}

    def partial(tag):
        return [f"{tag}.out"] if (out / f"{tag}.out").exists() else []

    report = contract.reconcile(out, lines, None, partial, query=query)
    cases = report["cases"]
    assert [case["status"] for case in cases] == [
        "interrupted",
        "missing",
        "partial",
    ]
    assert cases[0]["files"] == ["g0/c001.out"]
    assert cases[0]["claim"]["slurm_state"] == "OUT_OF_MEMORY"
    assert report["exit_code"] == 1  # the partial case alone
    (out / "g0" / "c003.out").unlink()
    report = contract.reconcile(out, lines, None, partial, query=query)
    assert report["counts"]["interrupted"] == 1 and report["exit_code"] == 0
    code, lines_out = contract.finish_decision(report)
    assert code == contract.FINISH_MISSING
    assert "interrupted indices: 1" in lines_out
    assert "missing indices: 2-3" in lines_out
    assert "ARRAY=1-3" in lines_out[-1] and "TIME or MEM" in lines_out[-1]
    code, lines_out = contract.finish_decision(report, queued=[1])
    assert (
        code == contract.FINISH_IN_FLIGHT and "queued indices: 1" in lines_out
    )
    code, _ = contract.finish_decision(report, allow_missing=True)
    assert code == 0


def test_a_stale_claim_without_files_is_interrupted(tmp_path):
    """A harness that writes its artifacts at the end of a run, killed
    outright mid-run: a stale claim and nothing else."""
    out = tmp_path
    lines = ["g0/c001 1", "g0/c002 2", "g0/c003 3"]
    claim_by(out, "g0/c001", "800", "1")
    # Case 2 failed once, and its rerun was killed before it wrote a file.
    claim_by(out, "g0/c002", "800", "2")
    (out / "g0").mkdir()
    contract.error_path(out, "g0/c002").write_text("c002: E: x\n", "utf-8")
    # Case 3: files beside no claim, which stay fatal.
    (out / "g0" / "c003.out").write_text("half\n", encoding="utf-8")

    def partial(tag):
        return [f"{tag}.out"] if (out / f"{tag}.out").exists() else []

    report = contract.reconcile(
        out, lines, None, partial, query=answer(False, "TIMEOUT")
    )
    cases = report["cases"]
    assert [case["status"] for case in cases] == [
        "interrupted",
        "interrupted",
        "partial",
    ]
    assert cases[0]["files"] == [] and cases[0]["claim"]["owner"] == "800_1"
    assert cases[1]["earlier_error"] == "c002: E: x"
    assert report["exit_code"] == 1
    (out / "g0" / "c003.out").unlink()
    report = contract.reconcile(
        out, lines, None, partial, query=answer(False, "TIMEOUT")
    )
    assert report["exit_code"] == 0
    code, lines_out = contract.finish_decision(report)
    assert code == contract.FINISH_MISSING
    assert "interrupted indices: 1-2" in lines_out
    assert (
        "ARRAY=1-3" in lines_out[-1] and "left their claims" in lines_out[-1]
    )


def report_of(statuses):
    return {
        "cases": [
            {"index": index, "tag": f"t{index}", "status": status}
            for index, status in enumerate(statuses, start=1)
        ],
        "stray": [],
    }


def test_finish_counts_queued_and_in_flight_apart_from_missing():
    report = report_of(
        ["verified", "in_flight", "missing", "missing", "failed"]
    )
    code, lines = contract.finish_decision(report, queued=[4])
    assert code == contract.FINISH_IN_FLIGHT
    assert "in_flight indices: 2" in lines and "queued indices: 4" in lines
    assert "missing indices: 3" in lines and "failed indices: 5" in lines
    code, lines = contract.finish_decision(report, [4], allow_running=True)
    assert code == contract.FINISH_MISSING and "ARRAY=3" in lines[-1]
    assert contract.finish_decision(report, [3, 4], allow_running=True)[0] == 0
    assert (
        contract.finish_decision(
            report, [4], allow_missing=True, allow_running=True
        )[0]
        == 0
    )
    only_missing = report_of(["verified", "missing"])
    assert contract.finish_decision(only_missing)[0] == contract.FINISH_MISSING
    assert contract.finish_decision(only_missing, allow_missing=True)[0] == 0


def test_finish_stops_on_a_failed_verification():
    for statuses, stray in (
        (["verify_failed"], []),
        (["partial"], []),
        ([], ["x"]),
    ):
        report = report_of(statuses)
        report["stray"] = stray
        code, _ = contract.finish_decision(
            report, allow_missing=True, allow_running=True
        )
        assert code == contract.FINISH_FATAL


def run_cli(*args):
    return subprocess.run(
        [sys.executable, str(HERE / "campaign_contract.py"), *args],
        capture_output=True,
        text=True,
    )


def test_command_line(tmp_path):
    cases = tmp_path / "cases.txt"
    cases.write_bytes(b"a 1\nb 2\nc 3\n")
    subset = tmp_path / "s.txt"
    subset.write_bytes(b"1 a 1\n3 c 3\n")
    result = run_cli(
        "check-cases", "--cases", str(cases), "--subset", str(subset)
    )
    assert result.returncode == 0 and result.stdout.split() == ["1", "3"]
    subset.write_bytes(b"2 a 1\n")
    assert (
        run_cli(
            "check-cases", "--cases", str(cases), "--subset", str(subset)
        ).returncode
        == 1
    )

    manifest = tmp_path / "manifest.json"
    contract.write_json(
        manifest, {"finishing_steps": [["select"], ["summarize", "-v"]]}
    )
    result = subprocess.run(
        [
            sys.executable,
            str(HERE / "campaign_contract.py"),
            "finishing-steps",
            "--manifest",
            str(manifest),
        ],
        capture_output=True,
    )
    assert result.stdout == b"select\nsummarize -v\n"

    verification = tmp_path / "verification.json"
    contract.write_json(
        verification, report_of(["verified", "in_flight", "missing"])
    )
    queued = tmp_path / "queued.txt"
    queued.write_text("3 PENDING\n", encoding="utf-8")
    result = run_cli(
        "finish-check",
        "--verification",
        str(verification),
        "--queued",
        str(queued),
    )
    assert result.returncode == contract.FINISH_IN_FLIGHT
    assert result.stdout.strip() == "2"
    result = run_cli(
        "finish-check",
        "--verification",
        str(verification),
        "--queued",
        str(queued),
        "--allow-running",
    )
    assert result.returncode == 0 and result.stdout.strip() == "2"
    result = run_cli(
        "finish-check", "--verification", str(verification), "--allow-running"
    )
    assert result.returncode == contract.FINISH_MISSING
    assert "missing indices: 3" in result.stderr


# --------------------------------------------------------------------------
# Tracked copies and their redaction
# --------------------------------------------------------------------------

STUB_HARNESS = "dev/scripts/campaign_stub_harness.py"
LINES = ["g0/c001 1", "g0/c002 2", "g1/c003 3"]


def base_identity():
    """An identity whose source part every record of the campaign shares."""
    return {
        "contract": contract.CONTRACT_VERSION,
        "source": {
            "trees": {
                "harness": {"commit": "a" * 40, "clean": True},
                "gpyreg": {"commit": "b" * 40, "clean": True},
            },
            "files": {STUB_HARNESS: "c" * 64},
            "versions": {"python": "3.12.6", "numpy": "2.5.2"},
        },
        "imports": {
            "trees": {
                "harness": {"path": "", "dirty": []},
                "gpyreg": {"path": "", "dirty": []},
            },
            "modules": {"pyvbmc": "", "gpyreg": ""},
            "installed_metadata_versions": {"pyvbmc": None},
        },
    }


def finished_campaign(site, where=None):
    """A campaign finished at ``site``, written with the contract alone.

    Three cases of the stub harness's shape: two verified, run on the two
    nodes, and one interrupted, whose stale claim names the second node.
    Its tracked copies are the summary, a note under ``notes/``, each
    verified case's record and ``.out`` file. The site's details lie in
    the manifest (the site block, the login node's host part, the pip
    freeze), the records, a claim, the task logs and the accounting; the
    summary names the node that only a task log names and one that only
    the accounting names, and the note the login node, the host that
    redacts (``fakelogin9``) and the campaign's own path.
    """
    out = (site.home / "runs" / "c1") if where is None else where
    out.mkdir(parents=True)
    manifest = {
        "campaign": "stub",
        "cases": len(LINES),
        "identity": site.plant(base_identity()),
        "site": site.site_block(STUB_HARNESS),
        "pip_freeze": [
            "numpy==2.5.2",
            f"gpyreg @ file://{site.gpyreg.as_posix()}",
            f"-e {site.home.as_posix()}/src/pyvbmc",
        ],
        "finishing_steps": [["summarize"]],
        "tracked_copies": {
            "files": ["summary.json", "notes/*.md"],
            "cases": {"record": True, "artifacts": ["*.out"]},
        },
    }
    contract.write_json(out / "manifest.json", manifest)
    (out / "cases.txt").write_text("".join(f"{x}\n" for x in LINES))
    for index, line in enumerate(LINES[:2], start=1):
        tag = contract.case_tag(line)
        path = out / f"{tag}.out"
        path.parent.mkdir(parents=True, exist_ok=True)
        # The first case's file names the campaign's path; the second's
        # holds nothing to redact.
        path.write_text(
            f"case {index} in {out}\n" if index == 1 else "case 2\n",
            encoding="utf-8",
        )
        record = site.plant(
            base_identity(), node=site.nodes[index - 1], task=str(index)
        )
        contract.write_completion(
            out, tag, line, [path], record, 0.0, 1.0, {"index": index}
        )
    contract.write_json(
        contract.claim_path(out, "g1/c003"),
        {
            "job": "4242",
            "array_task": "3",
            "host": site.nodes[1],
            "pid": 7,
            "started": "2026-10-01T10:00:00+0300",
        },
    )
    report = contract.reconcile(
        out,
        LINES,
        lambda tag: None,
        lambda tag: [],
        query=answer(False, "OUT_OF_MEMORY"),
    )
    assert report["counts"]["verified"] == 2
    assert report["counts"]["interrupted"] == 1
    contract.write_json(out / "verification.json", report)
    contract.write_json(
        out / "summary.json",
        {
            "counts": report["counts"],
            "slowest": f"{site.log_node} and fakenode42",
        },
    )
    (out / "notes").mkdir()
    (out / "notes" / "prepared.md").write_text(
        f"Prepared on {site.login}, redacted on fakelogin9, in {out}.\n",
        encoding="utf-8",
    )
    site.write_slurm(out)
    (out.parent / "c1.tar.zst.sha256").write_text(
        f"{'d' * 64}  c1.tar.zst.000\n", encoding="utf-8"
    )
    return out


@pytest.fixture
def site(tmp_path):
    return stubs.FakeSite(tmp_path)


def redacted(site, out, target, **kwargs):
    """``redact`` with the site's operator, no settings of this process and
    ``fakelogin9`` as the host that redacts."""
    kwargs.setdefault("operator", site.operator())
    kwargs.setdefault("environ", {})
    kwargs.setdefault("host", "fakelogin9")
    return contract.redact(out, target, say=lambda message: None, **kwargs)


def test_the_setting_groups_cover_every_setting():
    groups = (
        contract.REDACTED_PATH_SETTINGS,
        contract.COMMAND_SETTINGS,
        contract.PUBLIC_SETTINGS,
        ("PARTITION",),
    )
    named = [name for group in groups for name in group]
    assert sorted(named) == sorted(contract.SETTINGS)


def test_tracked_copies_declaration():
    good = {
        "files": ["summary.md", "rescored/*.json"],
        "cases": {"record": True, "artifacts": ["*.json"]},
    }
    assert contract.tracked_copies({"tracked_copies": good}) == good
    assert contract.tracked_copies({"tracked_copies": {}}) == {
        "files": [],
        "cases": {"record": False, "artifacts": []},
    }
    with pytest.raises(contract.ContractError, match="declares no"):
        contract.tracked_copies({})
    for files in (
        ["/abs/summary.md"],
        ["../summary.md"],
        ["a//b"],
        ["a" + "\\" + "b.md"],
        ["C:/x.md"],
        ["slurm/sacct.txt"],
        ["claims/g0/c001"],
        ["tmp/x"],
        ["redaction.json"],
        ["*/summary.md"],
        [""],
        "summary.md",
    ):
        with pytest.raises(contract.ContractError):
            contract.tracked_copies({"tracked_copies": {"files": files}})
    for spec in (
        {"files": [], "other": 1},
        {"cases": {"record": "yes"}},
        {"cases": {"artifacts": ["slurm/*"]}},
        {"cases": {"records": True}},
    ):
        with pytest.raises(contract.ContractError):
            contract.tracked_copies({"tracked_copies": spec})


def test_operator_identity_from_the_environment_and_the_password_entry(
    tmp_path,
):
    from types import SimpleNamespace

    entry = SimpleNamespace(pw_name="fromdb", pw_dir=str(tmp_path / "db"))
    found = contract.operator_identity(
        {"USER": "fromenv", "HOME": str(tmp_path / "env")}, passwd=entry
    )
    assert found == {
        "users": ["fromdb", "fromenv"],
        "homes": sorted([str(tmp_path / "db"), str(tmp_path / "env")]),
    }
    with pytest.raises(contract.ContractError, match="root"):
        contract.operator_identity({"USER": "u", "HOME": "/"}, passwd=entry)
    nobody = SimpleNamespace(pw_name="", pw_dir="")
    with pytest.raises(contract.ContractError, match="no username"):
        contract.operator_identity({}, passwd=nobody)


def test_slurm_host_lists():
    assert contract.expand_hostlist("n[01-03,7],login1,a[1-2]b[3-4]") == [
        "n01",
        "n02",
        "n03",
        "n7",
        "login1",
        "a1b3",
        "a1b4",
        "a2b3",
        "a2b4",
    ]
    assert contract.expand_hostlist("None assigned") == []
    assert contract.expand_hostlist("(null)") == []
    with pytest.raises(contract.ContractError):
        contract.expand_hostlist("n[a-b]")


def test_redact_removes_every_site_detail(site, tmp_path):
    out = finished_campaign(site)
    target = tmp_path / "handback" / "tracked" / "c1"
    record = redacted(site, out, target)
    assert site.leaks(target) == []
    assert site.leaks(out)  # the campaign itself is left as it is
    names = sorted(
        p.relative_to(target).as_posix()
        for p in target.rglob("*")
        if p.is_file()
    )
    assert names == [
        "g0/c001.out",
        "g0/c002.out",
        "manifest.json",
        "notes/prepared.md",
        "records/g0/c001.complete.json",
        "records/g0/c002.complete.json",
        "redaction.json",
        "summary.json",
        "verification.json",
    ]
    manifest = contract.read_json(target / "manifest.json")
    assert "site" not in manifest
    assert manifest["pip_freeze"] == [
        "numpy==2.5.2",
        "gpyreg @ file:<redacted>",
        "-e <redacted>",
    ]
    assert manifest["identity"]["host"]["hostname"] == contract.LOGIN_HOST
    assert manifest["identity"]["host"]["executable"] == (
        "$CAMPAIGN_ENV/bin/python"
    )
    trees = manifest["identity"]["imports"]["trees"]
    assert trees["gpyreg"]["path"] == "$PYVBMC_GPYREG_SOURCE"
    assert trees["harness"]["path"].startswith("~")
    host = contract.read_json(target / "records/g0/c002.complete.json")[
        "identity"
    ]["host"]
    assert host["hostname"] == site.family
    assert host["node_features"]["node"] == site.family
    assert host["node_features"]["available"] == [site.family, "avx2"]
    assert host["slurm"]["node"] == site.family
    assert host["slurm"]["partition"] == "$PARTITION"
    assert host["slurm"]["array_job_id"] == "4242"  # job ids stay
    summary = contract.read_json(target / "summary.json")
    assert summary["slowest"] == f"{site.family} and {site.family}"
    note = (target / "notes" / "prepared.md").read_text(encoding="utf-8")
    assert note.startswith("Prepared on login, redacted on login, in ~")
    assert (target / "g0/c001.out").read_text().startswith("case 1 in ~")
    # The record of the redaction, and the SHA-256 chain through it.
    assert record == contract.read_json(target / contract.REDACTION)
    assert record["campaign"] == "c1"
    assert record["node_family"] == site.family
    assert record["archive"] == {
        "sha256_file": "c1.tar.zst.sha256",
        "parts": {"c1.tar.zst.000": "d" * 64},
    }
    assert set(record["files"]) == set(names) - {contract.REDACTION}
    for name, entry in record["files"].items():
        assert entry["source_sha256"] == contract.sha256_file(out / name)
        assert entry["sha256"] == contract.sha256_file(target / name)
        assert contract.source_sha256(target, name) == entry["source_sha256"]
        assert contract.source_sha256(out, name) == entry["source_sha256"]
    # A file with nothing to redact is copied as it is.
    assert (target / "g0/c002.out").read_bytes() == (
        out / "g0/c002.out"
    ).read_bytes()
    assert record["replaced"]["hosts"] > 0 and record["replaced"]["~"] > 0
    assert contract.source_name(target) == contract.source_name(out) == "c1"
    (target / "summary.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(contract.ContractError, match="not the copy"):
        contract.source_sha256(target, "summary.json")


def test_redact_refuses_what_survives_and_writes_nothing(site, tmp_path):
    out = finished_campaign(site)
    target = tmp_path / "handback" / "c1"
    for planted in (
        f"run by {site.user}",
        f"/scratch/{site.user}/c1",
        "export https_proxy=http://fakeproxy.invalid:3128",
        "fakeconda/24.1",
    ):
        site.rewrite(
            out / "summary.json", lambda value: value.update(note=planted)
        )
        with pytest.raises(contract.ContractError) as refusal:
            redacted(site, out, target)
        message = str(refusal.value)
        assert "summary.json" in message and "none was written" in message
        assert not target.parent.exists()
    # A partition in a field the redaction does not know is a JSON string
    # of its own, which it replaces; in a text it is refused.
    site.rewrite(
        out / "summary.json", lambda value: value.update(note=site.partition)
    )
    redacted(site, out, target)
    assert contract.read_json(target / "summary.json")["note"] == "$PARTITION"
    shutil.rmtree(target)
    with open(out / "notes" / "prepared.md", "a", encoding="utf-8") as note:
        note.write(f"Queued on {site.partition}.\n")
    with pytest.raises(contract.ContractError, match="a partition"):
        redacted(site, out, target)


def test_redact_names_a_directory_outside_the_home(site, tmp_path):
    scratch = tmp_path / "scratch" / site.user
    out = finished_campaign(site, where=scratch / "c1")
    target = tmp_path / "handback" / "c1"
    with pytest.raises(contract.ContractError, match="the username"):
        redacted(site, out, target)
    redacted(site, out, target, paths=[("SCRATCH", str(scratch))])
    assert site.leaks(target) == []
    assert (
        (target / "g0/c001.out").read_text().startswith("case 1 in $SCRATCH")
    )
    with pytest.raises(contract.ContractError, match="upper case"):
        redacted(site, out, tmp_path / "t2", paths=[("scratch", "/x")])


def test_redact_refusals(site, tmp_path):
    out = finished_campaign(site)
    target = tmp_path / "handback" / "c1"
    target.mkdir(parents=True)
    (target / "old.md").write_text("x")
    with pytest.raises(contract.ContractError, match="not an empty"):
        redacted(site, out, target)
    for inside in (out / "tracked", site.home / "src" / "harness" / "dev"):
        with pytest.raises(contract.ContractError, match="lies inside"):
            redacted(site, out, inside)

    def refuses(change, match, name="manifest.json"):
        before = (out / name).read_bytes()
        site.rewrite(out / name, change)
        with pytest.raises(contract.ContractError, match=match):
            redacted(site, out, tmp_path / "other")
        (out / name).write_bytes(before)
        assert not (tmp_path / "other").exists()

    refuses(lambda m: m.pop("tracked_copies"), "declares no tracked")
    refuses(lambda m: m["site"].pop("NODE_FEATURE"), "names no NODE_FEATURE")
    refuses(
        lambda m: m["tracked_copies"]["files"].append("absent.md"),
        "holds no absent.md",
    )
    refuses(
        lambda m: m.update(exit_code=1), "did not pass", "verification.json"
    )
    refuses(
        lambda r: r["identity"]["host"]["node_features"].update(
            available=["other"], active=["other"]
        ),
        "not fakefamily",
        "records/g0/c001.complete.json",
    )
    # A host named as the copies name the login node.
    refuses(
        lambda s: s.update(hostname=contract.LOGIN_HOST),
        "name hosts by 'login', which holds a hostname",
        "summary.json",
    )
    (out / "g0" / "c002.out").write_bytes(b"\x00\x01binary")
    with pytest.raises(contract.ContractError, match="is not text"):
        redacted(site, out, tmp_path / "other")
    (out / "verification.json").unlink()
    with pytest.raises(contract.ContractError, match="no verification"):
        redacted(site, out, tmp_path / "other")


def test_redact_from_the_command_line(site, tmp_path, monkeypatch, capsys):
    out = finished_campaign(site)
    for name in contract.SETTINGS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        contract, "operator_identity", lambda environ=None: site.operator()
    )
    # The host that redacts, which the campaign's note names.
    monkeypatch.setattr(contract.socket, "gethostname", lambda: "fakelogin9")
    target = tmp_path / "handback" / "c1"
    arguments = ["redact", "--campaign", str(out), "--out"]
    assert contract.main([*arguments, str(target)]) == 0
    assert site.leaks(target) == []
    assert "none of" in capsys.readouterr().out
    other = str(tmp_path / "t2")
    assert contract.main([*arguments, other, "--path", "X"]) == 1
    assert "not NAME=PATH" in capsys.readouterr().err
    site.rewrite(out / "summary.json", lambda v: v.update(by=site.user))
    assert contract.main([*arguments, other]) == 1
    assert "refusing: " in capsys.readouterr().err
    assert not (tmp_path / "t2").exists()
