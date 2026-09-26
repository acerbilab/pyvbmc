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

import errno
import json
import os
import platform
import re
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
    [path] = contract.retired_claims(tmp_path, TAG)
    assert path.name == f"c001.stale.500_3.{held['token'][:8]}"
    retired = json.loads(path.read_text("utf-8"))
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


def test_an_unfinished_retirement_is_finished(tmp_path):
    """The retired name holds the judged claim, which is still in place: a
    retirer was killed between its link and its removal, or a network
    filesystem retried the link. The next worker finishes it."""
    held = contract.acquire_claim(tmp_path, TAG, query=never, task=TASK_A)
    path = Path(held["path"])
    record = json.loads(path.read_text("utf-8"))
    os.link(path, contract.retired_claim_path(path, record))
    claim = contract.acquire_claim(
        tmp_path, TAG, query=answer(False, "TIMEOUT"), task=TASK_B
    )
    assert json.loads(path.read_text("utf-8"))["token"] == claim["token"]
    assert claim["previous"]["owner"] == "500_3"
    [retired] = contract.retired_claims(tmp_path, TAG)
    assert json.loads(retired.read_text("utf-8")) == record
    assert not [p for p in path.parent.iterdir() if p.name.startswith(".")]


def test_a_leftover_retired_name_does_not_block_the_case(tmp_path):
    """An earlier claim of the same task, retired before a requeue: its
    retired name holds another key, so the new claim retires beside it."""
    contract.acquire_claim(tmp_path, TAG, query=never, task=TASK_A)
    folder = tmp_path / "claims" / "g0"
    earlier = dict(contract.new_claim(TAG, task=TASK_A), token="0" * 32)
    contract.write_json(folder / "c001.stale.500_3.00000000", earlier)
    claim = contract.acquire_claim(
        tmp_path, TAG, query=answer(False, "TIMEOUT"), task=TASK_B
    )
    assert claim["previous"]["reason"] == "stale"
    assert len(contract.retired_claims(tmp_path, TAG)) == 2


def test_a_retired_name_that_holds_something_else_refuses(tmp_path):
    held = contract.acquire_claim(tmp_path, TAG, query=never, task=TASK_A)
    path = Path(held["path"])
    record = json.loads(path.read_text("utf-8"))
    contract.retired_claim_path(path, record).write_text("{", "utf-8")
    before = snapshot(tmp_path / "claims")
    with pytest.raises(contract.ClaimRefused, match="another worker"):
        contract.acquire_claim(
            tmp_path, TAG, query=answer(False, "TIMEOUT"), task=TASK_B
        )
    assert snapshot(tmp_path / "claims") == before


def test_finishing_a_retirement_leaves_a_later_claim_in_place(tmp_path):
    """Another worker finished the retirement and made its own claim: the
    claim in place is not the judged one, and stays."""
    held = contract.acquire_claim(tmp_path, TAG, query=never, task=TASK_A)
    path = Path(held["path"])
    judged = json.loads(path.read_text("utf-8"))
    stale = contract.retired_claim_path(path, judged)
    os.link(path, stale)
    later = contract.new_claim(TAG, task=TASK_B)
    contract.write_json(path, later)
    assert not contract._finish_retirement(path, stale, judged)
    assert json.loads(path.read_text("utf-8")) == later
    assert not [p for p in path.parent.iterdir() if p.name.startswith(".")]
    path.unlink()
    assert contract._finish_retirement(path, stale, judged)


def test_claim_keys():
    claim = contract.new_claim(TAG, task=TASK_A)
    assert contract.claim_key(claim) == claim["token"][:8]
    tokenless = {"job": "500", "array_task": "3", "host": "n1"}
    key = contract.claim_key(tokenless)
    assert re.fullmatch(r"[0-9a-f]{8}", key)
    assert contract.claim_key(dict(tokenless, host="n2")) != key
    path = contract.claim_path("out", TAG)
    assert contract.retired_claim_path(path, tokenless).name == (
        f"c001.stale.500_3.{key}"
    )


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


@pytest.mark.parametrize("where", ["after the record", "in a message"])
def test_a_failure_after_the_record_leaves_the_case_complete(
    repo, tmp_path, monkeypatch, where
):
    """EIO on the log's or the directory's filesystem once the record is
    written: the case keeps every file, and no error file is written."""
    out = tmp_path / "out"
    expected = worker_identity(repo)
    contract.error_path(out, TAG).parent.mkdir(parents=True)
    contract.error_path(out, TAG).write_text("an earlier attempt\n", "utf-8")
    if where == "after the record":
        write = contract.write_completion

        def write_then_fail(*args, **kwargs):
            write(*args, **kwargs)
            raise OSError(errno.EIO, "Input/output error")

        monkeypatch.setattr(contract, "write_completion", write_then_fail)
    else:
        say = contract._say

        def fail_on_completion(message):
            if "complete in" in message:
                raise OSError(errno.ENOSPC, "No space left on device")
            say(message)

        monkeypatch.setattr(contract, "_say", fail_on_completion)

    def run(identity):
        path = out / f"{TAG}.out"
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
    assert code == 0
    contract.check_completion(out, TAG, expected, [f"{TAG}.out"])
    assert (out / f"{TAG}.out").read_text("utf-8") == "whole\n"
    assert not contract.error_path(out, TAG).exists()
    assert not contract.claim_path(out, TAG).exists()
    report = contract.reconcile(out, [TAG], lambda tag: None, lambda tag: [])
    assert report["cases"][0]["status"] == "verified"


def test_a_stopped_rerun_of_a_failed_case_is_missing(repo, tmp_path):
    """The rerun sets the earlier error file aside when it starts; stopped,
    it leaves the case missing with the earlier error attached."""
    out = tmp_path / "out"
    expected = worker_identity(repo)
    error = contract.error_path(out, TAG)
    error.parent.mkdir(parents=True)
    error.write_text(f"{TAG}: ValueError: the first attempt\n", "utf-8")
    seen = []

    def stopped(identity):
        seen.append(error.exists())
        signal.raise_signal(signal.SIGTERM)

    code = contract.run_worker(
        out, TAG, expected, lambda: expected, stopped, lambda: [], never
    )
    assert code == 128 + signal.SIGTERM
    assert seen == [False]
    earlier = contract.earlier_error_path(out, TAG)
    assert earlier == out / "claims" / "g0" / "c001.error.txt"
    assert earlier.read_text("utf-8").startswith(f"{TAG}: ValueError")
    report = contract.reconcile(out, [TAG], None, lambda tag: [], query=never)
    [case] = report["cases"]
    assert case["status"] == "missing"
    assert case["earlier_error"] == f"{TAG}: ValueError: the first attempt"
    assert report["stray"] == [] and report["exit_code"] == 0
    # Killed outright on its next attempt: interrupted, with the error.
    claim_by(out, TAG, "800", "1")
    report = contract.reconcile(
        out, [TAG], None, lambda tag: [], query=answer(False, "TIMEOUT")
    )
    assert report["cases"][0]["status"] == "interrupted"
    assert report["cases"][0]["earlier_error"].endswith("the first attempt")
    contract.claim_path(out, TAG).unlink()

    # A failing attempt writes its own error and drops the set-aside one.
    def fails(identity):
        raise RuntimeError("the second attempt")

    assert (
        contract.run_worker(
            out, TAG, expected, lambda: expected, fails, lambda: [], never
        )
        == 1
    )
    assert not earlier.exists()
    assert "the second attempt" in contract.error_reason(error)
    report = contract.reconcile(out, [TAG], None, lambda tag: [], query=never)
    assert report["cases"][0]["status"] == "failed"
    # A stop, then a success: both error files go.
    contract.run_worker(
        out, TAG, expected, lambda: expected, stopped, lambda: [], never
    )
    assert earlier.exists() and not error.exists()

    def succeeds(identity):
        path = out / f"{TAG}.out"
        path.write_text("whole\n", encoding="utf-8")
        return [path], None

    assert (
        contract.run_worker(
            out, TAG, expected, lambda: expected, succeeds, lambda: [], never
        )
        == 0
    )
    assert not earlier.exists() and not error.exists()


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
    [retired] = contract.retired_claims(out, TAG)
    assert retired.name.startswith("c001.stale.800_1.")
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


@pytest.mark.parametrize("kind", ["clone", "worktree"])
def test_a_package_from_a_checkout_nested_in_the_tree_is_refused(
    tmp_path, monkeypatch, kind
):
    """A checkout of another commit inside the tree, as a frozen worktree
    under the harness checkout's ignored dev/scripts/runs/ is."""
    environment = git_env(tmp_path)
    tree = make_repo(
        tmp_path / "tree",
        {".gitignore": "runs/\n", "fakepkg_nested/__init__.py": ""},
        environment,
    )
    nested = tree / "runs" / "old"
    if kind == "clone":
        make_repo(
            nested,
            {"fakepkg_nested/__init__.py": "__version__ = 'old'\n"},
            environment,
        )
    else:
        subprocess.run(
            ["git", "worktree", "add", "-q", "--detach", str(nested)],
            cwd=tree,
            env=environment,
            check=True,
        )
    assert contract.enclosing_checkout(nested / "fakepkg_nested") == (
        nested.resolve()
    )
    monkeypatch.syspath_prepend(str(nested))
    with pytest.raises(contract.IdentityError, match="inside the tree"):
        contract.module_origin("fakepkg_nested", tree)
    sys.modules.pop("fakepkg_nested")
    monkeypatch.syspath_prepend(str(tree))
    assert contract.module_origin("fakepkg_nested", tree) == str(
        (tree / "fakepkg_nested").resolve()
    )
    sys.modules.pop("fakepkg_nested")


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
        affinity = host["cpu_affinity"]
        assert set(affinity["core_threads"]) == set(affinity["physical_cores"])
        for threads in affinity["core_threads"].values():
            assert threads and set(threads) & set(affinity["cpus"])
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
    assert contract.node_features(False, environment, waits=()) is None
    with pytest.raises(contract.HostError, match="node7.*1 attempts"):
        contract.node_features(True, environment, waits=())
    assert contract.node_features(True, {}) is None


def test_node_features_retries_a_failed_query(slurm, monkeypatch):
    """The controller fails twice, then answers: the third query counts."""
    environment = {"SLURM_JOB_ID": "5", "SLURMD_NODENAME": "node7"}
    slept = []
    monkeypatch.setattr(contract.time, "sleep", slept.append)
    (slurm / "scontrol_node_fail_count").write_text("2", encoding="utf-8")
    features = contract.node_features(True, environment)
    assert features["available"] == ["stubfeat"]
    assert slept == list(contract.SCONTROL_WAITS[:2])
    # Failing every time: each wait, then the refusal.
    (slurm / "scontrol_node_fail").write_text("", encoding="utf-8")
    slept.clear()
    with pytest.raises(contract.HostError, match="4 attempts"):
        contract.node_features(True, environment)
    assert slept == list(contract.SCONTROL_WAITS)


def affinity(cpus, threads, cpuset=None):
    """A recorded affinity on one core ``0:4`` whose threads are
    ``threads``."""
    return {
        "cpus": cpus,
        "physical_cores": ["0:4"],
        "core_threads": {"0:4": threads},
        "job_cpuset": cpuset,
    }


def test_host_problems():
    good = {
        "node_features": {
            "node": "n",
            "available": ["amd", "x"],
            "active": [],
        },
        "cpu_affinity": affinity([4, 68], [4, 68]),
    }
    assert contract.host_problems(good, "amd") == []
    assert (
        "lacks the feature intel" in contract.host_problems(good, "intel")[0]
    )
    two = dict(
        good,
        cpu_affinity={
            "cpus": [4, 5],
            "physical_cores": ["0:4", "0:5"],
            "core_threads": {"0:4": [4], "0:5": [5]},
            "job_cpuset": None,
        },
    )
    assert "2 physical cores" in contract.host_problems(two, "amd")[0]
    bare = {"node_features": None, "cpu_affinity": None}
    assert len(contract.host_problems(bare, "amd")) == 2
    # No SMT: the core's one thread.
    smt_off = dict(good, cpu_affinity=affinity([4], [4]))
    assert contract.host_problems(smt_off, "amd") == []
    # One hardware thread of a two-thread core: another job may hold the
    # other, unless the job's cpuset shows the whole core as the job's.
    [problem] = contract.host_problems(
        dict(good, cpu_affinity=affinity([4], [4, 68])), "amd"
    )
    assert "not the hardware threads [4, 68]" in problem
    assert "no cpuset of the job" in problem
    [problem] = contract.host_problems(
        dict(good, cpu_affinity=affinity([4], [4, 68], [4])), "amd"
    )
    assert "cpuset [4] is not those threads" in problem
    whole = dict(good, cpu_affinity=affinity([4], [4, 68], [4, 68]))
    assert contract.host_problems(whole, "amd") == []
    # A record written without the threads of its core.
    legacy = dict(good, cpu_affinity={"cpus": [4], "physical_cores": ["0:4"]})
    [problem] = contract.host_problems(legacy, "amd")
    assert "no hardware threads of core 0:4" in problem
    # The stubs' one-core affinity passes.
    stub = dict(good, cpu_affinity=stubs.ONE_CORE_AFFINITY)
    assert contract.host_problems(stub, "amd") == []


def fake_topology(root, cores, name="core_cpus_list"):
    """``/sys/devices/system/cpu`` with ``cores``: ``{(package, core):
    [cpu, ...]}``."""
    for (package, core), cpus in cores.items():
        for cpu in cpus:
            folder = root / f"cpu{cpu}" / "topology"
            folder.mkdir(parents=True)
            (folder / "physical_package_id").write_text(f"{package}\n")
            (folder / "core_id").write_text(f"{core}\n")
            text = ",".join(str(c) for c in cpus)
            (folder / name).write_text(f"{text}\n")
    return root


def test_cpu_affinity_from_the_topology(tmp_path):
    smt = fake_topology(tmp_path / "smt", {(0, 4): [4, 68], (0, 5): [5, 69]})
    found = contract.cpu_affinity(
        True, lambda pid: {4}, smt, cpuset=lambda: [4, 68]
    )
    assert found == {
        "cpus": [4],
        "physical_cores": ["0:4"],
        "core_threads": {"0:4": [4, 68]},
        "job_cpuset": [4, 68],
    }
    host = {"node_features": {"available": ["f"]}, "cpu_affinity": found}
    assert contract.host_problems(host, "f") == []
    both = contract.cpu_affinity(
        True, lambda pid: {4, 68}, smt, cpuset=lambda: None
    )
    assert both["core_threads"] == {"0:4": [4, 68]}
    assert contract.host_problems(dict(host, cpu_affinity=both), "f") == []
    lone = contract.cpu_affinity(
        True, lambda pid: {69}, smt, cpuset=lambda: None
    )
    assert lone["physical_cores"] == ["0:5"]
    assert contract.host_problems(dict(host, cpu_affinity=lone), "f")
    # No SMT, on a kernel that names the threads thread_siblings_list.
    flat = fake_topology(
        tmp_path / "flat", {(1, 0): [8], (1, 1): [9]}, "thread_siblings_list"
    )
    one = contract.cpu_affinity(
        True, lambda pid: {9}, flat, cpuset=lambda: None
    )
    assert one["core_threads"] == {"1:1": [9]}
    assert contract.host_problems(dict(host, cpu_affinity=one), "f") == []
    spread = contract.cpu_affinity(
        True, lambda pid: {8, 9}, flat, cpuset=lambda: None
    )
    assert spread["physical_cores"] == ["1:0", "1:1"]
    # A topology that cannot be read: recorded as unknown, or refused.
    with pytest.raises(contract.HostError, match="topology of CPU 3"):
        contract.cpu_affinity(True, lambda pid: {3}, flat)
    unknown = contract.cpu_affinity(False, lambda pid: {3}, flat)
    assert unknown["physical_cores"] is None


def test_the_job_cpuset_from_its_cgroup(tmp_path):
    job = {"SLURM_JOB_ID": "812"}
    proc = tmp_path / "cgroup"
    root = tmp_path / "fs"
    # cgroup v2: the task's cgroup lies under the job's, whose cpuset is
    # the job's allocation.
    proc.write_text(
        "0::/system.slice/slurmstepd.scope/job_812/step_batch/user/task_0\n"
    )
    job_dir = root / "system.slice" / "slurmstepd.scope" / "job_812"
    (job_dir / "step_batch" / "user" / "task_0").mkdir(parents=True)
    (job_dir / "cpuset.cpus.effective").write_text("4,68\n")
    assert contract.job_cpuset(job, proc, root) == [4, 68]
    # Without the job's component, the process's own cgroup.
    task = job_dir / "step_batch" / "user" / "task_0"
    (task / "cpuset.cpus.effective").write_text("4\n")
    assert contract.job_cpuset({"SLURM_JOB_ID": "9"}, proc, root) == [4]
    # cgroup v1: the cpuset controller's hierarchy.
    proc.write_text("7:cpuset:/slurm/uid_1/job_812/step_0\n4:memory:/x\n")
    v1 = root / "cpuset" / "slurm" / "uid_1" / "job_812"
    v1.mkdir(parents=True)
    (v1 / "cpuset.effective_cpus").write_text("10-11\n")
    assert contract.job_cpuset(job, proc, root) == [10, 11]
    # Nothing to read: null.
    assert contract.job_cpuset(job, tmp_path / "absent", root) is None
    proc.write_text("0::/elsewhere\n")
    assert contract.job_cpuset(job, proc, root) is None
    assert contract.cpu_list("0-2,7,9-10") == [0, 1, 2, 7, 9, 10]


# --------------------------------------------------------------------------
# Completion records
# --------------------------------------------------------------------------


def identity_with_host(features=("stubfeat",)):
    return {
        "source": {"trees": {"harness": {"commit": "a", "clean": True}}},
        "host": {
            "node_features": {
                "node": "n",
                "available": list(features),
                "active": [],
            },
            "cpu_affinity": dict(stubs.ONE_CORE_AFFINITY),
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
        ("# python==3.12\nnumpy==2\n#python==3.13\n", ":3: a second"),
    ):
        with pytest.raises(contract.ContractError, match=message):
            contract.read_requirements(write_requirements(tmp_path, text))
    # The pin is a line of its own from the first column, as the build's
    # sed reads it; an indented one, or one with more on its line, is a
    # comment.
    for text in ("  # python==3.12\n", "# python==3.12 or so\n"):
        path = write_requirements(tmp_path, text)
        assert contract.read_requirements(path)["python"] is None
    path = write_requirements(tmp_path, "#python==3.12.6  \n")
    assert contract.read_requirements(path)["python"] == "3.12.6"


def test_the_build_reads_the_python_pin_as_the_check_does(tmp_path):
    """``campaign_env.sh build`` takes the one ``# python==`` line that
    ``read_requirements`` takes, and refuses two."""
    script = (HERE / "hpc" / "campaign_env.sh").read_text(encoding="utf-8")
    command = re.search(r"_campaign_pins=\$\((sed .*?)\)\n", script, re.S)
    assert command, "the build's sed of the '# python==' line"
    bash = stubs.find_bash()
    for text, pins in (
        ("# python==3.12\nnumpy==2\n", ["3.12"]),
        ("#python==3.12.6  \n", ["3.12.6"]),
        ("  # python==3.12\n# python==3.12 or so\n", []),
        ("# python==3.12\n# python==3.13\n", ["3.12", "3.13"]),
    ):
        path = write_requirements(tmp_path, text)
        sed = command.group(1).replace(
            '"$CAMPAIGN_REQUIREMENTS"', f"'{path.as_posix()}'"
        )
        result = subprocess.run(
            [bash, "-c", sed], capture_output=True, text=True, check=True
        )
        assert result.stdout.split() == pins
        if len(pins) == 1:
            assert contract.read_requirements(path)["python"] == pins[0]
        elif pins:
            with pytest.raises(contract.ContractError, match="a second"):
                contract.read_requirements(path)


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
    # A node feature is one name, never an expression of several.
    for feature in ("amd&avx2", "amd|intel", "[amd]", "amd,x", "a b", "*"):
        with pytest.raises(contract.ContractError, match="one node feature"):
            contract.site_block(dict(environment, NODE_FEATURE=feature))
    assert contract.site_block(dict(environment, NODE_FEATURE="epyc_7.x-2"))


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
    # 7 missing: a rerun of a failed case, stopped, set its error aside
    contract.earlier_error_path(out, tags[6]).write_text(
        "c007: E: earlier\n", "utf-8"
    )
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
    (out / "claims" / "g0" / "c099.error.txt").write_text("x", "utf-8")
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
    assert cases[6]["earlier_error"] == "c007: E: earlier"
    assert "earlier_error" not in cases[0] and "earlier_error" not in cases[4]
    assert [case["index"] for case in cases] == list(range(1, 11))
    assert report["stray"] == [
        "claims/g0/c099",
        "claims/g0/c099.error.txt",
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
        "stray": 6,
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


# --------------------------------------------------------------------------
# The queue and the accounting, for the finish
# --------------------------------------------------------------------------


def slurm_records(out, jobs, steps=(), accounting=None):
    slurm = out / "slurm"
    slurm.mkdir(parents=True, exist_ok=True)
    (slurm / "jobs.txt").write_text("".join(f"{j}\n" for j in jobs), "utf-8")
    (slurm / "steps.txt").write_text("".join(f"{s}\n" for s in steps), "utf-8")
    if accounting is not None:
        (slurm / "sacct.txt").write_text(
            "JobID|JobName|State|ExitCode|NodeList\n"
            + "".join(f"{row}\n" for row in accounting),
            "utf-8",
        )
    return slurm


def test_recorded_jobs(tmp_path):
    slurm = slurm_records(
        tmp_path,
        [
            "1001 array=1-3 offset=0 subset=- throttle=200 partition=- "
            r"sbatch_extra=--comment=offset=9\ x 2026-10-01T10:00:00",
            "1002 array=1-2 offset=3 subset=odd throttle=200",
        ],
        [
            "1005 step=verify submitted 2026-10-01T11:00:00",
            "1005 step=verify rc=0 2026-10-01T11:05:00",
            "? step=summarize rc=1 2026-10-01T11:06:00",
        ],
    )
    assert contract.recorded_jobs(slurm) == [
        {"job": "1001", "offset": 0},
        {"job": "1002", "offset": 3},
        {"job": "1005", "offset": None},
    ]
    assert contract.recorded_jobs(tmp_path / "nowhere") == []


def test_the_queue_counts_only_tasks_that_have_not_ended(tmp_path, slurm):
    out = tmp_path / "c1"
    records = slurm_records(
        out,
        ["1001 array=1-4 offset=0", "1002 array=1-2 offset=4"],
        ["1005 step=verify submitted x"],
    )
    queue = slurm / "squeue"
    queue.mkdir()
    # squeue lists a task that ended moments ago beside the live ones.
    (queue / "1001").write_text(
        "1001_1 COMPLETED\n1001_2 RUNNING\n1001_3 COMPLETING\n"
        "1001_4 CANCELLED\n",
        "utf-8",
    )
    (queue / "1002").write_text("1002_2 PENDING\n", "utf-8")
    (queue / "1005").write_text("1005 CONFIGURING\n", "utf-8")
    state = contract.queue_state(records)
    assert state["live"] == [
        ("1001", "1001_2", "RUNNING"),
        ("1001", "1001_3", "COMPLETING"),
        ("1002", "1002_2", "PENDING"),
        ("1005", "1005", "CONFIGURING"),
    ]
    assert state["cases"] == [
        (2, "RUNNING"),
        (3, "COMPLETING"),
        (6, "PENDING"),
    ]
    assert state["unknown"] == []
    for name in ("1001", "1002", "1005"):
        (queue / name).unlink()
    assert contract.queue_state(records) == {
        "live": [],
        "cases": [],
        "unknown": [],
    }


def test_a_failed_queue_query_is_resolved_by_the_accounting(tmp_path, slurm):
    out = tmp_path / "c1"
    queue = slurm / "squeue"
    queue.mkdir()
    for job in ("1001", "1002", "1003"):
        (queue / f"{job}.fail").write_text("", "utf-8")
    jobs = ["1001 array=1-2 offset=0", "1002 array=1 offset=2", "1003 x"]
    # No accounting at all: nothing is known.
    records = slurm_records(out, jobs)
    state = contract.queue_state(records)
    assert [job for job, _ in state["unknown"]] == ["1001", "1002", "1003"]
    assert "Invalid job id" in state["unknown"][0][1]
    assert "no accounting" in state["unknown"][0][1]
    # The accounting shows 1001 ended, 1002 still running, and no 1003.
    records = slurm_records(
        out,
        jobs,
        accounting=[
            "1001_1|c1|COMPLETED|0:0|n1",
            "1001_1.batch|batch|COMPLETED|0:0|n1",
            "1001_2|c1|CANCELLED by 5|0:15|n1",
            "1002_1|c1|RUNNING|0:0|n2",
        ],
    )
    state = contract.queue_state(records)
    assert [job for job, _ in state["unknown"]] == ["1002", "1003"]
    assert "does not show its tasks ended" in state["unknown"][0][1]
    assert "holds no row" in state["unknown"][1][1]
    assert state["live"] == [] and state["cases"] == []


def test_the_archive_waits_for_the_accounting(tmp_path):
    out = tmp_path / "c1"
    assert contract.accounting_problems(slurm_records(out, [])) == []
    records = slurm_records(out, ["1001 array=1-3 offset=0"])
    [problem] = contract.accounting_problems(records)
    assert "is missing" in problem
    records = slurm_records(
        out,
        ["1001 array=1-3 offset=0"],
        ["1005 step=verify rc=0"],
        accounting=[
            "1001_1|c1|COMPLETED|0:0|n1",
            "1001_2|c1|RUNNING|0:0|n1",
            "1001_[3]|c1|PENDING|0:0|None assigned",
            "1005|c1_verify|COMPLETED|0:0|n2",
            "999|other|RUNNING|0:0|n3",
        ],
    )
    assert contract.accounting_problems(records) == [
        "1001_2 is RUNNING in slurm/sacct.txt",
        "1001_[3] is PENDING in slurm/sacct.txt",
    ]


@pytest.mark.parametrize(
    "state, exitcode, code",
    [
        ("COMPLETED", "0:0", 0),
        ("FAILED", "1:0", 1),
        ("FAILED", "3:0", 3),
        ("TIMEOUT", "0:15", 143),
        ("CANCELLED by 5", "0:0", 1),
        ("OUT_OF_MEMORY", "0:9", 137),
    ],
)
def test_the_exit_code_of_a_job(slurm, state, exitcode, code):
    set_answer(slurm, "1005", f"{state}\n")
    (slurm / "sacct" / "1005.exitcode").write_text(f"{exitcode}\n", "utf-8")
    assert contract.job_exit("1005") == (state.split()[0], code)


def test_waiting_for_a_job(slurm, monkeypatch):
    assert contract.job_exit("1006") == (None, None)
    set_answer(slurm, "1006", "PENDING\n")
    assert contract.job_exit("1006") == ("PENDING", None)
    set_answer(slurm, "1007", fail=True)
    assert contract.job_exit("1007") == (None, None)
    answers = iter(
        [(None, None), ("PENDING", None), ("RUNNING", None), ("FAILED", 1)]
    )
    said, slept = [], []
    monkeypatch.setattr(contract.time, "sleep", slept.append)
    code = contract.wait_job(
        "1006", poll=0.5, say=said.append, query=lambda job: next(answers)
    )
    assert code == 1 and slept == [0.5, 0.5, 0.5]
    assert said == [
        "job 1006: not in the accounting yet",
        "job 1006: PENDING",
        "job 1006: RUNNING",
        "job 1006: FAILED, exit 1",
    ]


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
    assert record["replaced"][contract.PARTITION_TOKEN] == 2
    assert record["cases_not_verified"] == {
        "failed": [],
        "interrupted": ["g1/c003"],
        "missing": [],
    }
    assert contract.source_name(target) == contract.source_name(out) == "c1"
    (target / "summary.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(contract.ContractError, match="not the copy"):
        contract.source_sha256(target, "summary.json")


def test_redact_refuses_what_survives_and_writes_nothing(site, tmp_path):
    out = finished_campaign(site)
    target = tmp_path / "handback" / "c1"
    env = site.env.as_posix()
    for planted, what in (
        (f"run by {site.user}", "the username"),
        (f"/scratch/{site.user}/c1", "the username"),
        ("export https_proxy=http://fakeproxy.invalid:3128", "LOGIN_SETUP"),
        ("fakeconda/24.1", "CONDA_SETUP"),
        # Where the replacement leaves a path, the check still finds it:
        # a longer directory and a path that continues another.
        (f"{env}s/pyvbmc2", "CAMPAIGN_ENV"),
        (f"/mnt{env}/lib", "CAMPAIGN_ENV"),
        (f"x{site.user}y", "the username"),
        ("on xfakenode17y", "a hostname"),
    ):
        site.rewrite(
            out / "summary.json", lambda value: value.update(note=planted)
        )
        with pytest.raises(contract.ContractError) as refusal:
            redacted(site, out, target)
        message = str(refusal.value)
        assert "summary.json" in message and "none was written" in message
        assert what in message, message
        assert not target.parent.exists()


def test_the_redaction_rewrites_every_form_of_a_name(site, tmp_path):
    """``KEY=/path``, a flag's path, a path in a sentence, ``~user``, a
    hostname in capitals or in a JSON key: each is replaced."""
    out = finished_campaign(site)
    env = site.env.as_posix()
    node = site.nodes[0]
    site.rewrite(
        out / "summary.json",
        lambda value: value.update(
            per_host={node: 1},
            upper=f"{site.nodes[1].split('.')[0].upper()} ran it",
            flags=f"X={env}/bin -L{env}/lib",
            sentence=f"The environment is {env}.",
            tilde=f"~{site.user}/runs/c1",
            digest="0123456789abcdef" * 4,
        ),
    )
    with open(out / "notes" / "prepared.md", "a", encoding="utf-8") as note:
        note.write(f"Run on {node.upper()}, from ~{site.user}/src.\n")
    target = tmp_path / "handback" / "c1"
    redacted(site, out, target)
    assert site.leaks(target) == []
    summary = contract.read_json(target / "summary.json")
    assert summary["per_host"] == {site.family: 1}
    assert summary["upper"] == f"{site.family} ran it"
    assert summary["flags"] == "X=$CAMPAIGN_ENV/bin -L$CAMPAIGN_ENV/lib"
    assert summary["sentence"] == "The environment is $CAMPAIGN_ENV."
    assert summary["tilde"] == "~/runs/c1"
    note = (target / "notes" / "prepared.md").read_text(encoding="utf-8")
    assert f"Run on {site.family}, from ~/src." in note


def test_a_digest_is_not_read_as_a_name(site, tmp_path):
    """A username or a hostname of hex letters alone, inside a SHA-256."""
    site.user = "fab"
    out = finished_campaign(site)
    site.rewrite(
        out / "summary.json",
        lambda value: value.update(digest="00fab0" + "1" * 58),
    )
    redacted(site, out, tmp_path / "handback" / "c1")
    site.rewrite(
        out / "summary.json", lambda value: value.update(digest="00fab0")
    )
    with pytest.raises(contract.ContractError, match="the username 'fab'"):
        redacted(site, out, tmp_path / "handback" / "c2")


def test_a_login_host_named_login(site, tmp_path):
    """The copies name the login node ``login``, which a login host's own
    name may be."""
    site.login = contract.LOGIN_HOST
    out = finished_campaign(site)
    target = tmp_path / "handback" / "c1"
    redacted(site, out, target)
    manifest = contract.read_json(target / "manifest.json")
    assert manifest["identity"]["host"]["hostname"] == contract.LOGIN_HOST
    note = (target / "notes" / "prepared.md").read_text(encoding="utf-8")
    assert note.startswith("Prepared on login, redacted on login")


def test_partitions_are_redacted_in_their_fields_alone(tmp_path):
    """A partition is not personal data: it is replaced where a field holds
    one, and a copy may hold its name elsewhere, even a common word."""
    site = stubs.FakeSite(tmp_path)
    site.partition = "test"
    out = finished_campaign(site)
    site.rewrite(
        out / "summary.json",
        lambda value: value.update(
            verdict="test",
            site={"NODE_FEATURE": site.family, "PARTITION": "test"},
            env={"SLURM_JOB_PARTITION": "test", "OTHER": "test"},
        ),
    )
    with open(out / "notes" / "prepared.md", "a", encoding="utf-8") as note:
        note.write("There is no equivalence test.\n")
    target = tmp_path / "handback" / "c1"
    record = redacted(site, out, target)
    summary = contract.read_json(target / "summary.json")
    assert summary["verdict"] == "test"
    assert summary["site"]["PARTITION"] == contract.PARTITION_TOKEN
    assert summary["env"] == {
        "SLURM_JOB_PARTITION": contract.PARTITION_TOKEN,
        "OTHER": "test",
    }
    host = contract.read_json(target / "records/g0/c001.complete.json")[
        "identity"
    ]["host"]
    assert host["slurm"]["partition"] == contract.PARTITION_TOKEN
    assert "no equivalence test" in (
        target / "notes" / "prepared.md"
    ).read_text("utf-8")
    assert record["replaced"][contract.PARTITION_TOKEN] == 4


def test_redact_names_a_directory_outside_the_home(site, tmp_path):
    scratch = tmp_path / "scratch" / site.user
    out = finished_campaign(site, where=scratch / "c1")
    # The directory that holds the campaign is named for it.
    target = tmp_path / "handback" / "c1"
    redacted(site, out, target)
    assert site.leaks(target) == []
    text = (target / "g0/c001.out").read_text()
    assert re.fullmatch(r"case 1 in \$CAMPAIGN_PARENT[/\\]c1\n", text)
    # A --path takes precedence, as it names what it holds.
    redacted(site, out, tmp_path / "t2", paths=[("SCRATCH", str(scratch))])
    text = (tmp_path / "t2" / "g0/c001.out").read_text()
    assert re.fullmatch(r"case 1 in \$SCRATCH[/\\]c1\n", text)
    # A --path above the username names none of it: refused.
    with pytest.raises(contract.ContractError, match="the username"):
        redacted(
            site,
            out,
            tmp_path / "t3",
            paths=[("SCRATCH", str(scratch.parent))],
        )
    for name in ("scratch", "CAMPAIGN_PARENT", "HARNESS_TREE", "PARTITION"):
        with pytest.raises(contract.ContractError, match="upper case"):
            redacted(site, out, tmp_path / "t4", paths=[(name, "/x")])


def test_redact_names_the_trees_and_refuses_other_absolute_paths(
    site, tmp_path
):
    out = finished_campaign(site)
    elsewhere = tmp_path / "elsewhere" / "pyvbmc"

    def move_harness(identity):
        identity["imports"]["trees"]["harness"]["path"] = elsewhere.as_posix()
        identity["imports"]["modules"]["pyvbmc"] = (
            elsewhere / "pyvbmc"
        ).as_posix()

    site.rewrite(out / "manifest.json", lambda m: move_harness(m["identity"]))
    for tag in ("g0/c001", "g0/c002"):
        site.rewrite(
            contract.record_path(out, tag),
            lambda r: move_harness(r["identity"]),
        )
    site.rewrite(
        out / "summary.json",
        lambda value: value.update(blas="/usr/lib64/libopenblas.so.0"),
    )
    target = tmp_path / "handback" / "c1"
    redacted(site, out, target)
    imports = contract.read_json(target / "manifest.json")["identity"][
        "imports"
    ]
    assert imports["trees"]["harness"]["path"] == "$HARNESS_TREE"
    assert imports["modules"]["pyvbmc"] == "$HARNESS_TREE/pyvbmc"
    assert imports["trees"]["gpyreg"]["path"] == "$PYVBMC_GPYREG_SOURCE"
    summary = contract.read_json(target / "summary.json")
    assert summary["blas"] == "/usr/lib64/libopenblas.so.0"
    # A path that no name covers, outside the system's directories.
    for planted in (
        "/opt/site/modules/x",
        "PATH=/usr/bin:/cluster/bin",
        r"D:\data",
    ):
        site.rewrite(
            out / "summary.json", lambda value: value.update(note=planted)
        )
        with pytest.raises(
            contract.ContractError, match="an absolute path that no name"
        ):
            redacted(site, out, tmp_path / "other")
        assert not (tmp_path / "other").exists()


def test_absolute_paths():
    text = (
        "X=/proj/env/a g0/c001 and/or 1/2 -L/usr/lib ~/x $CAMPAIGN_ENV/b "
        "file:///p/q https://h.org/w/c ./rel ../up (/opt/x) "
        r"PATH=/a/b:/c/d C:\Users\x "
        '"C:\\\\Users\\\\y"'
    )
    assert [path for _, path in contract.absolute_paths(text)] == [
        "/proj/env/a",
        "/usr/lib",
        "/p/q",
        "/opt/x",
        "/a/b",
        "/c/d",
        r"C:\Users\x",
        r"C:\\Users\\y",
    ]


def test_path_variants(tmp_path):
    real = tmp_path / "real"
    (real / "home").mkdir(parents=True)
    link = tmp_path / "link"
    if sys.platform == "win32":
        import _winapi

        _winapi.CreateJunction(str(real), str(link))
    else:
        os.symlink(real, link)
    forms = contract.path_variants(str(link / "home") + os.sep)
    assert str(link / "home") in forms  # without the trailing separator
    assert str((real / "home").resolve()) in forms  # resolved
    assert json.dumps(str((real / "home").resolve()))[1:-1] in forms
    windows = contract.path_variants("C:\\Users\\op\\")
    assert {"C:\\Users\\op", "C:\\\\Users\\\\op"} <= windows
    assert contract.path_variants("/") == set()


def test_source_sha256_reads_crlf_as_lf(site, tmp_path):
    out = finished_campaign(site)
    target = tmp_path / "handback" / "c1"
    record = redacted(site, out, target)
    for name, entry in record["files"].items():
        path = target / name
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
        assert contract.source_sha256(target, name) == entry["source_sha256"]
    path = target / "summary.json"
    path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n") + b" ")
    with pytest.raises(contract.ContractError, match="not the copy"):
        contract.source_sha256(target, "summary.json")


def test_the_copies_survive_a_round_trip_through_git(site, tmp_path):
    """Committed with the repository's .gitattributes and checked out with
    core.autocrlf=true, as on Windows: the release gate's copies keep their
    bytes, and copies elsewhere, converted to CRLF, still verify."""
    out = finished_campaign(site)
    environment = git_env(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    shutil.copy(ROOT / ".gitattributes", repo / ".gitattributes")
    gate = repo / "dev" / "experiments" / "release_gate_20261001" / "pools"
    record = redacted(site, out, gate)
    shutil.copytree(gate, repo / "elsewhere")
    for command in (
        ["git", "init", "-q"],
        ["git", "-c", "core.autocrlf=true", "add", "-A"],
        ["git", "commit", "-q", "-m", "copies"],
    ):
        subprocess.run(command, cwd=repo, env=environment, check=True)
    clone = tmp_path / "clone"
    subprocess.run(
        [
            "git",
            "clone",
            "-q",
            "-c",
            "core.autocrlf=true",
            str(repo),
            str(clone),
        ],
        env=environment,
        check=True,
    )
    copied = clone / "dev" / "experiments" / "release_gate_20261001" / "pools"
    converted = clone / "elsewhere"
    assert b"\r\n" in (converted / "summary.json").read_bytes()
    for name, entry in record["files"].items():
        assert (copied / name).read_bytes() == (gate / name).read_bytes()
        for directory in (copied, converted):
            assert (
                contract.source_sha256(directory, name)
                == entry["source_sha256"]
            )


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
    refuses(lambda m: m["site"].update(NODE_FEATURE="a&b"), "one node feature")
    refuses(
        lambda m: m["tracked_copies"]["files"].append("absent.md"),
        "holds no absent.md",
    )
    refuses(
        lambda m: m.update(exit_code=1), "did not pass", "verification.json"
    )

    def in_flight(report):
        report["cases"][2]["status"] = "in_flight"

    refuses(in_flight, "1 cases in flight", "verification.json")
    refuses(
        lambda r: r["identity"]["host"]["node_features"].update(
            available=["other"], active=["other"]
        ),
        "not fakefamily",
        "records/g0/c001.complete.json",
    )
    record = contract.record_path(out, "g0/c002")
    kept = record.read_bytes()
    record.unlink()
    with pytest.raises(contract.ContractError, match="c002.complete.json is"):
        redacted(site, out, tmp_path / "other")
    record.write_bytes(kept)
    summary = out / "summary.json"
    kept = summary.read_bytes()
    summary.write_text("{", encoding="utf-8")
    with pytest.raises(contract.ContractError, match="summary.json is not"):
        redacted(site, out, tmp_path / "other")
    summary.write_bytes(kept)
    (out / "g0" / "c002.out").write_bytes(b"\x00\x01binary")
    with pytest.raises(contract.ContractError, match="is not text"):
        redacted(site, out, tmp_path / "other")
    (out / "verification.json").unlink()
    with pytest.raises(contract.ContractError, match="no verification"):
        redacted(site, out, tmp_path / "other")


def test_the_redaction_records_the_cases_it_did_not_verify(site, tmp_path):
    out = finished_campaign(site)

    def ruled_on(report):
        report["cases"][1]["status"] = "failed"
        report["counts"]["verified"] -= 1
        report["counts"]["failed"] += 1

    site.rewrite(out / "verification.json", ruled_on)
    record = redacted(site, out, tmp_path / "handback" / "c1")
    assert record["cases_not_verified"] == {
        "failed": ["g0/c002"],
        "interrupted": ["g1/c003"],
        "missing": [],
    }
    assert "g0/c002.out" not in record["files"]


def test_the_check_of_files_redact_did_not_write(site, tmp_path):
    out = finished_campaign(site)
    readme = tmp_path / "handback" / "README.md"
    readme.parent.mkdir()
    readme.write_text(
        f"# The pools\n\nRun on {site.family} nodes, in ~/runs.\n", "utf-8"
    )
    arguments = dict(operator=site.operator(), environ={}, host="fakelogin9")
    assert contract.check_files(out, [readme], **arguments) == []
    with open(readme, "a", encoding="utf-8") as stream:
        stream.write(f"Logged in to {site.login} as {site.user}.\n")
        stream.write(f"The gpyreg is {site.gpyreg.as_posix()}.\n")
    leaks = contract.check_files(out, [readme], **arguments)
    assert {(line, what) for _, line, _, what in leaks} == {
        (4, "a hostname"),
        (4, "the username"),
        (5, "PYVBMC_GPYREG_SOURCE"),
        (5, "an absolute path that no name covers"),
    }
    assert all(name == str(readme) for name, *_ in leaks)


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
    # A file that does not parse: refused, with no traceback.
    manifest = (out / "manifest.json").read_bytes()
    (out / "manifest.json").write_text("{", encoding="utf-8")
    assert contract.main([*arguments, other]) == 1
    assert "refusing: " in capsys.readouterr().err
    (out / "manifest.json").write_bytes(manifest)
    # The check of other files.
    readme = tmp_path / "README.md"
    readme.write_text(f"By {site.user}.\n", encoding="utf-8")
    check = ["redact", "--campaign", str(out), "--check", str(readme)]
    assert contract.main(check) == 1
    assert "refusing: " in capsys.readouterr().err
    with pytest.raises(SystemExit):
        contract.main(["redact", "--campaign", str(out)])
