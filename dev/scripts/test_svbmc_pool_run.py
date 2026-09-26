"""Contracts of the S-VBMC pool harness, ``svbmc_pool_run.py``.

The artifact, resumption, selection (its walk over every seed, its
refusal of an unfinished seed below a selected run, and what makes it
stackable after a verification) and summary; the campaign contract
(the case list and its subsets, the worker's early exit and refusals, its
claim, its clean-up after a failure or a SIGTERM, the identity and host
fields, every state of ``verify``); the identity of a case run by the array
worker and by ``run``; the redaction of a pool's tracked copies; the
flat layout of the pools prepared before the contract; and one campaign
through the Slurm driver of ``dev/scripts/hpc/``, its redaction included.

One short campaign (``normal_D2``, at most three seeds, well under a
minute of inference) is generated once for the whole module through the
command line with ``run``, as a workstation pool is, and its stored runs
are reloaded, copied and re-verified. The tests of the worker's
bookkeeping run it in this process with the run itself replaced by a copy
of one of those stored runs; the tests that play a Slurm task set its
variables and put the stub Slurm commands of ``campaign_slurm_stubs.py``
first on the PATH.

The gpyreg checkout is the one ``PYVBMC_GPYREG_SOURCE`` names, which the
campaign's environment exports: every test skips when it is unset, and
fails when it names no gpyreg checkout. Outside default pytest discovery;
run it by path::

    PYVBMC_GPYREG_SOURCE=<gpyreg checkout> \\
        python -m pytest dev/scripts/test_svbmc_pool_run.py -vv
"""

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
from pathlib import Path

import campaign_contract as contract
import campaign_slurm_stubs as stubs
import numpy as np
import pytest
import svbmc_pool_run as runner

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRIPT = HERE / "svbmc_pool_run.py"
#: The gpyreg checkout every campaign of the module is generated against.
GPYREG_SOURCE = (
    Path(os.environ["PYVBMC_GPYREG_SOURCE"]).resolve()
    if os.environ.get("PYVBMC_GPYREG_SOURCE")
    else None
)
LABEL = "normal_D2"
SEED_START = 4000
TARGET, MAX_SEEDS = 2, 3
#: The checks `verify_run` runs against live objects, in order. The
#: hashes of the completion record are not among them: the record is
#: written after the artifact it describes.
LIVE_CHECKS = [
    "stats_keys",
    "recomputation_gate",
    "dtype_canary",
    "posterior",
    "evaluations",
    "gp_prediction",
]
POST_HOC_CHECKS = ["stats_keys", "recomputation_gate", "dtype_canary"]
#: The fields of a completion record that the contract writes; the rest
#: are the harness's.
CONTRACT_FIELDS = (
    "contract",
    "tag",
    "case",
    "identity",
    "started",
    "finished",
    "elapsed_seconds",
    "artifacts",
)
#: What says when a run was written and how long it and its evaluations
#: took: the sidecar's metadata fields, the logger's total and the array of
#: each evaluation's seconds. Two runs of one case differ in these, and in
#: the memory addresses that the representations of its objects carry,
#: alone.
TIMING_META = ("written", "wall_s", "target_eval_s")
TIMING_ARRAYS = ("logger/fun_eval_time",)
ADDRESS = re.compile(r" at 0x[0-9A-Fa-f]+")
#: The flag every campaign of the module is prepared with: the harness
#: and the suite are developed together, so the campaigns are generated
#: from whatever the checkout holds.
DIRTY = "--allow-dirty"

import svbmc_pool_io as pool_io  # noqa: E402

pytestmark = pytest.mark.skipif(
    GPYREG_SOURCE is None,
    reason="PYVBMC_GPYREG_SOURCE, the gpyreg checkout the pools are "
    "generated against, is unset",
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def tag_of(seed, label=LABEL):
    return runner.case_tag(label, seed)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def manifest_of(out):
    return read_json(Path(out) / "manifest.json")


def verification(out):
    """The report `verify` wrote in a campaign directory."""
    return read_json(Path(out) / "verification.json")


def reported(report, tag):
    """The one reported case of a tag."""
    (case,) = [c for c in report["cases"] if c["tag"] == tag]
    return case


def outside_campaign(key):
    """Whether a variable is Slurm's or an operator setting other than the
    gpyreg pin, which a test that is no campaign task must not see."""
    key = key.upper()
    return key.startswith("SLURM") or (
        key in contract.SETTINGS and key != "PYVBMC_GPYREG_SOURCE"
    )


def environment(**extra):
    """This process's environment outside any Slurm task and campaign, the
    campaign's thread settings and gpyreg pin, and ``extra``."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not outside_campaign(key)
    }
    env.update({k: "1" for k in runner.THREAD_KEYS})
    env["MPLBACKEND"] = "Agg"
    env["PYVBMC_GPYREG_SOURCE"] = str(GPYREG_SOURCE)
    env.update({key: str(value) for key, value in extra.items()})
    return env


def cli(*args, env=None):
    return subprocess.run(
        [sys.executable, "-u", str(SCRIPT), *(str(a) for a in args)],
        cwd=str(ROOT),
        env=environment() if env is None else env,
        capture_output=True,
        text=True,
    )


def prepare_args(out, target=TARGET, max_seeds=MAX_SEEDS, *extra):
    return [
        "prepare",
        "--out",
        str(out),
        "--suite",
        "smoke",
        "--only",
        LABEL,
        "--target",
        str(target),
        "--max-seeds",
        str(max_seeds),
        "--seed-start",
        str(SEED_START),
        "--gpyreg-source",
        str(GPYREG_SOURCE),
        *extra,
    ]


def prepare(out, target=TARGET, max_seeds=MAX_SEEDS, *extra):
    return cli(*prepare_args(out, target, max_seeds, DIRTY, *extra))


def ok(result):
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def completed_tags(out):
    return sorted(
        (
            f"{path.parent.name}/{path.name[: -len('.complete.json')]}"
            for path in (Path(out) / "records").glob("*/*.complete.json")
        ),
        key=runner.tag_seed,
    )


def copied(out, tmp_path, name="pool"):
    copy = tmp_path / name
    shutil.copytree(out, copy)
    (copy / "verification.json").unlink(missing_ok=True)
    return copy


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


def edit_manifest(out, change):
    manifest = manifest_of(out)
    change(manifest)
    contract.write_json(Path(out) / "manifest.json", manifest)
    return manifest


def plant(out, name, text="planted\n"):
    path = Path(out) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def claim_by(out, tag, job, array_task, restart_count=0):
    """A claim of ``tag`` made by the Slurm task ``<job>_<array_task>``."""
    record = contract.new_claim(
        tag,
        task={
            "job": job,
            "array_task": array_task,
            "restart_count": restart_count,
        },
    )
    contract.write_json(contract.claim_path(out, tag), record)
    return record


def retired_claims(out, tag, owner):
    """The retired claims of ``tag`` that the task ``owner``
    (``<job>_<task>``) held: ``claims/<tag>.stale.<owner>``, with or without
    a further ``.<key>`` after the owner."""
    claim = contract.claim_path(out, tag)
    name = f"{claim.name}.stale.{owner}"
    return [
        path
        for path in claim.parent.glob(f"{name}*")
        if path.name == name or path.name.startswith(f"{name}.")
    ]


def answer(state, step, text=None, fail=False):
    """What the stub ``sacct`` says of the task ``step``."""
    folder = state / "sacct"
    folder.mkdir(exist_ok=True)
    if fail:
        (folder / f"{step}.fail").write_text("", encoding="utf-8")
    else:
        (folder / step).write_text(text, encoding="utf-8")


def stub_environment(state, **extra):
    """:func:`environment` with the stub Slurm commands first on the PATH."""
    base = environment(**extra)
    return stubs.stub_environment(state.parent / "bin", state, base)


def as_task(monkeypatch, job, array_task, restart=0, node="node1"):
    """Make this process an array task of Slurm, as the task script's."""
    monkeypatch.setenv("SLURM_JOB_ID", f"{job}{int(array_task):03d}")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", str(job))
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", str(array_task))
    monkeypatch.setenv("SLURM_RESTART_COUNT", str(restart))
    monkeypatch.setenv("SLURMD_NODENAME", node)


def worker(out, line):
    """``worker --case LINE`` in this process."""
    return runner.main(["worker", "--out", str(out), "--case", line])


def replay(source_out, source_tag, seen=None):
    """A stand-in for ``run_case``: a stored run copied into place.

    The copy of ``source_tag``'s artifact from ``source_out`` becomes the
    case's, with its record fields under the case's label and seed. With
    ``seen``, it records the case's claim and files as the run finds them.
    """

    def run_case(out, manifest, tag, label, seed, save_vbmc=False):
        out = Path(out)
        if seen is not None:
            claim = contract.claim_path(out, tag)
            seen["claim"] = read_json(claim) if claim.exists() else None
            seen["files"] = runner.partial_artifacts(out, tag)
        target = out / tag
        target.parent.mkdir(parents=True, exist_ok=True)
        paths = []
        for suffix in pool_io.SUFFIXES:
            shutil.copyfile(
                f"{source_out / source_tag}{suffix}", f"{target}{suffix}"
            )
            paths.append(Path(f"{target}{suffix}"))
        record = read_json(pool_io.record_path(source_out, source_tag))
        fields = {k: v for k, v in record.items() if k not in CONTRACT_FIELDS}
        fields.update(label=label, seed=int(seed))
        return paths, fields

    return run_case


def must_not_run(*args, **kwargs):
    raise AssertionError("a refused case must not run")


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    """One generated pool directory, shared by every check below."""
    out = tmp_path_factory.mktemp("svbmc_pool")
    ok(prepare(out))
    result = ok(cli("run", "--out", str(out)))
    return out, result.stdout


@pytest.fixture
def no_slurm(monkeypatch):
    """This process outside any Slurm task, and without the operator
    settings a campaign's environment may export."""
    for key in list(os.environ):
        if outside_campaign(key):
            monkeypatch.delenv(key)


@pytest.fixture
def slurm(tmp_path, monkeypatch, no_slurm):
    """Stub Slurm commands first on this process's PATH; their state."""
    state = tmp_path / "slurm_state"
    state.mkdir()
    environment = stubs.stub_environment(
        stubs.write_stubs(tmp_path / "bin"), state
    )
    monkeypatch.setenv("PATH", environment["PATH"])
    monkeypatch.setenv("STUB_STATE", environment["STUB_STATE"])
    return state


@pytest.fixture
def fresh(tmp_path, no_slurm):
    """A campaign of eight seeds, prepared in this process.

    Its identity is this process's, so that the worker run here matches
    it whatever modules the process has imported.
    """
    out = tmp_path / "fresh"
    assert runner.main(prepare_args(out, 2, 8, DIRTY)) == 0
    return out


@pytest.fixture
def stored(campaign):
    """``(campaign directory, the tag of its first stored run)``."""
    out, _ = campaign
    return out, completed_tags(out)[0]


# --------------------------------------------------------------------------
# The artifact and the records
# --------------------------------------------------------------------------


def test_the_pinned_gpyreg_is_the_one_imported():
    import gpyreg

    assert (
        GPYREG_SOURCE / "gpyreg"
    ).is_dir(), f"PYVBMC_GPYREG_SOURCE={GPYREG_SOURCE} holds no gpyreg package"
    assert Path(gpyreg.__file__).resolve().parent == GPYREG_SOURCE / "gpyreg"


def test_manifest_records_the_allocation_identity_and_site(campaign):
    out, _ = campaign
    manifest = manifest_of(out)
    assert manifest["contract"] == contract.CONTRACT_VERSION
    assert manifest["allocation"] == [
        {
            "label": LABEL,
            "seed_start": SEED_START,
            "max_seeds": MAX_SEEDS,
            "target_filtered": TARGET,
        }
    ]
    assert manifest["options"] == runner.BASE_OPTIONS
    assert manifest["gpyreg_source"] == str(GPYREG_SOURCE)
    assert manifest["allow_dirty"] is True
    assert manifest["finishing_steps"] == [["select"], ["summarize"]]
    assert contract.finishing_steps(manifest) == [["select"], ["summarize"]]
    assert set(manifest["site"]) == set(contract.SETTINGS)
    assert manifest["pip_freeze"] and isinstance(manifest["pip_freeze"], list)
    identity = manifest["identity"]
    assert set(identity) == {"contract", "source", "imports", "host"}
    trees = identity["source"]["trees"]
    assert set(trees) == {"harness", "gpyreg"}
    assert trees["harness"]["commit"] == runner.git(ROOT, "rev-parse", "HEAD")
    assert trees["gpyreg"] == {
        "commit": runner.git(GPYREG_SOURCE, "rev-parse", "HEAD"),
        "clean": True,
    }
    files = identity["source"]["files"]
    assert set(files) == {
        "dev/scripts/svbmc_pool_run.py",
        "dev/scripts/svbmc_pool_io.py",
        "dev/scripts/benchmark_targets.py",
        "dev/scripts/campaign_contract.py",
        "dev/scripts/profile_run.py",
        "dev/scripts/data/",
    }
    assert files["dev/scripts/svbmc_pool_run.py"] == contract.sha256_file(
        SCRIPT
    )
    assert files["dev/scripts/data/"] == contract.directory_sha256(
        HERE / "data"
    )
    versions = identity["source"]["versions"]
    assert set(versions) == {"python", "numpy", "scipy", "cma", "torch"}
    assert versions["numpy"] == np.__version__
    modules = identity["imports"]["modules"]
    assert Path(modules["pyvbmc"]) == ROOT / "pyvbmc"
    assert Path(modules["gpyreg"]) == GPYREG_SOURCE / "gpyreg"
    assert Path(identity["imports"]["trees"]["gpyreg"]["path"]) == (
        GPYREG_SOURCE
    )
    assert set(identity["host"]) == {
        "hostname",
        "platform",
        "executable",
        "cpu_model",
        "node_features",
        "blas",
        "cpu_affinity",
        "threads",
        "slurm",
    }
    assert identity["host"]["threads"] == {k: "1" for k in runner.THREAD_KEYS}


def test_every_case_wrote_its_artifact_and_a_valid_record(campaign):
    out, stdout = campaign
    tags = completed_tags(out)
    assert tags, stdout
    assert len(tags) <= MAX_SEEDS
    manifest = manifest_of(out)
    for tag in tags:
        for suffix in pool_io.SUFFIXES:
            assert (out / f"{tag}{suffix}").exists()
        record = runner.check_record(out, tag, manifest["identity"])
        assert record["tag"] == record["case"] == tag
        assert set(record["artifacts"]) == {f"{tag}.npz", f"{tag}.json"}
        assert record["label"] == LABEL
        assert record["seed"] == runner.tag_seed(tag)
        assert not contract.source_differences(
            record["identity"], manifest["identity"]
        )
        assert record["identity"]["host"]["slurm"]["job_id"] is None
        assert set(record["verdict"]) == {
            "stable",
            "max_J_sjk",
            "s_max",
            "passes",
        }
        assert record["K"] > 0 and record["func_count"] > 0
        assert isinstance(record["success_flag"], bool)
        assert record["convergence_status"] in ("probable", "no")
        assert isinstance(record["message"], str) and record["message"]
        assert isinstance(record["r_index"], float)
        assert isinstance(record["iterations"], int)
        assert record["iterations"] > 0
        assert record["verification"]["checks"] == LIVE_CHECKS
        assert (out / f"{tag}.npz").stat().st_size < 10**6
        assert not contract.claim_path(out, tag).exists()


def test_stopping_rule_and_seeds(campaign):
    out, _ = campaign
    tags = completed_tags(out)
    seeds = [runner.tag_seed(t) for t in tags]
    assert seeds == list(range(SEED_START, SEED_START + len(seeds)))
    passing = sum(
        read_json(pool_io.record_path(out, tag))["verdict"]["passes"]
        for tag in tags
    )
    assert passing >= TARGET or len(tags) == MAX_SEEDS


def test_load_run_rebuilds_the_stored_posterior(stored):
    out, tag = stored
    state = pool_io.load_run(out / tag, rng=0)
    assert set(state) == set(pool_io.load_run.KEYS)
    sidecar = read_json(out / f"{tag}.json")
    assert sidecar["vp"]["stats"]["I_sk"] == "@@npz:vp/stats/I_sk"
    with np.load(out / f"{tag}.npz", allow_pickle=False) as stored_arrays:
        np.testing.assert_array_equal(
            state["vp"].stats["I_sk"], stored_arrays["vp/stats/I_sk"]
        )
        np.testing.assert_array_equal(state["gp"].X, stored_arrays["gp/X"])
    for key in pool_io.STATS_KEYS:
        assert key in state["vp"].stats
    meta = state["meta"]
    assert meta["label"] == LABEL and meta["target"] == "normal"
    assert meta["seed"] == runner.tag_seed(tag)
    assert set(meta["filter"]) == {"stable", "max_J_sjk", "s_max", "passes"}
    assert np.isfinite(meta["metrics"]["gskl"])
    assert meta["identity"]["host"]["gpyreg_import"].startswith(
        str(GPYREG_SOURCE)
    )
    assert state["vp"].rng is not None


def test_verify_run_passes_post_hoc(campaign):
    out, _ = campaign
    for tag in completed_tags(out):
        report = pool_io.verify_run(
            out / tag, record=contract.record_path(out, tag)
        )
        assert report["checks"] == POST_HOC_CHECKS + ["hashes"]
        assert report["differences"]["I_sk"] <= pool_io.TOL_STATS
        assert report["differences"]["J_sjk"] <= pool_io.TOL_STATS
        assert report["dtype_leaves"] > 0
        # Without the record's path, the flat layout's is looked for,
        # which a campaign of the contract does not have.
        assert pool_io.verify_run(out / tag)["checks"] == POST_HOC_CHECKS


def test_verify_run_compares_the_live_evaluations(stored):
    """The live checks, against a rebuilt run standing in for the live one.

    A second rebuild of the same artifact holds the same objects, so
    every live check passes on it; changing one recorded evaluation
    makes the comparison of the evaluations fail.
    """
    out, tag = stored
    state = pool_io.load_run(out / tag, rng=0)

    class Run:
        vp = state["vp"]
        function_logger = state["logger"]

        def get_gp(self, iteration):
            return state["gp"]

    results = {"best_iter": 0}
    report = pool_io.verify_run(out / tag, vbmc=Run(), results=results)
    assert report["checks"] == LIVE_CHECKS
    logger = state["logger"]
    logger.y_orig[np.flatnonzero(np.ravel(logger.X_flag))[0], 0] += 1.0
    with pytest.raises(RuntimeError, match="logger.y_orig differs"):
        pool_io.verify_run(out / tag, vbmc=Run(), results=results)


def test_verify_run_reads_the_hashes_of_either_record(stored, tmp_path):
    out, tag = stored
    copy = copied(out, tmp_path)
    path = contract.record_path(copy, tag)
    record = read_json(path)
    name = tag.split("/")[1]
    assert pool_io.recorded_hashes(record, name) == pool_io.artifact_hashes(
        copy / LABEL, name
    )
    flat = {"hashes": pool_io.artifact_hashes(copy / LABEL, name)}
    assert pool_io.recorded_hashes(flat, name) == flat["hashes"]
    record["artifacts"][f"{tag}.json"]["sha256"] = "0" * 64
    contract.write_json(path, record)
    with pytest.raises(
        RuntimeError, match=r"\.json differs from the recorded"
    ):
        pool_io.verify_run(copy / tag, record=path)


def test_second_run_skips_every_completed_case(campaign):
    out, _ = campaign
    before = {
        tag: pool_io.artifact_hashes(out / LABEL, tag.split("/")[1])
        for tag in completed_tags(out)
    }
    result = ok(cli("run", "--out", str(out)))
    assert "START" not in result.stdout
    for tag, hashes in before.items():
        assert f"SKIP {tag}" in result.stdout
        assert (
            pool_io.artifact_hashes(out / LABEL, tag.split("/")[1]) == hashes
        )


def test_summarize_reports_the_counts_and_the_convergence(campaign):
    out, _ = campaign
    ok(cli("summarize", "--out", str(out)))
    summary = read_json(out / "summary.json")
    (condition,) = summary["conditions"]
    tags = completed_tags(out)
    records = [read_json(pool_io.record_path(out, tag)) for tag in tags]
    assert condition["label"] == LABEL
    assert condition["completed"] == len(tags)
    assert condition["seeds_run"] == len(tags) + condition["failed"]
    assert condition["seed_cap"] == MAX_SEEDS
    assert condition["target_filtered"] == TARGET
    assert condition["filtered"] == sum(
        r["verdict"]["passes"] for r in records
    )
    assert 0.0 <= condition["pass_rate"] <= 1.0
    assert condition["wall_minutes"]["median"] > 0
    assert condition["success_flag"] == sum(r["success_flag"] for r in records)
    assert sum(condition["convergence_status"].values()) == len(tags)
    assert sum(condition["messages"].values()) == len(tags)
    assert condition["iterations"]["n"] == len(tags)
    assert condition["iterations"]["median"] == float(
        np.median([r["iterations"] for r in records])
    )
    assert condition["r_index"]["n"] == len(tags)
    assert summary["totals"]["filtered"] == condition["filtered"]
    assert summary["totals"]["success_flag"] == condition["success_flag"]
    text = (out / "summary.md").read_text(encoding="utf-8")
    assert LABEL in text and "converged" in text and "success" in text


# --------------------------------------------------------------------------
# cases
# --------------------------------------------------------------------------


def test_manifest_cases_follows_the_allocation_order():
    """Every seed of every condition's range, conditions in manifest order."""
    manifest = {
        "contract": 1,
        "allocation": [
            {"label": "second", "seed_start": 10, "max_seeds": 2},
            {"label": "first", "seed_start": 1000, "max_seeds": 3},
        ],
    }
    assert runner.manifest_cases(manifest) == [
        ("second", 10),
        ("second", 11),
        ("first", 1000),
        ("first", 1001),
        ("first", 1002),
    ]
    assert runner.case_lines(manifest)[:3] == [
        "second/second_seed10",
        "second/second_seed11",
        "first/first_seed1000",
    ]
    flat = dict(manifest)
    del flat["contract"]
    assert runner.case_lines(flat)[:2] == ["second 10", "second 11"]
    assert runner.allocated_case(manifest, "first/first_seed1001") == (
        "first",
        1001,
    )
    assert runner.allocated_case(manifest, "first/first_seed1003") is None


def test_cases_prints_one_tag_per_line(campaign, tmp_path):
    """The case index is the line number; the count goes to stderr."""
    out, _ = campaign
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "cases", "--out", str(out)],
        cwd=str(ROOT),
        env=environment(),
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert b"\r" not in result.stdout
    lines = result.stdout.decode("utf-8").splitlines()
    assert lines == [tag_of(SEED_START + i) for i in range(MAX_SEEDS)]
    assert f"{MAX_SEEDS} cases" in result.stderr.decode("utf-8")
    listing = tmp_path / "cases.txt"
    listing.write_bytes(result.stdout)
    assert contract.read_cases(listing) == lines
    listed = json.loads(
        cli("cases", "--out", str(out), "--format", "json").stdout
    )
    assert listed["count"] == MAX_SEEDS
    assert [
        (case["index"], case["line"], case["seed"]) for case in listed["cases"]
    ] == [
        (i + 1, tag_of(SEED_START + i), SEED_START + i)
        for i in range(MAX_SEEDS)
    ]


def test_the_release_allocation_has_2930_cases(tmp_path):
    """The release pools: 320 filtered runs per condition, seeds 1000-1349
    and 1000-1479 for the ring, expressed with the existing flags."""
    out = tmp_path / "release"
    ok(
        cli(
            "prepare",
            "--out",
            out,
            "--target",
            "320",
            "--max-seeds",
            "350",
            "--allocation",
            "ring_D2_noise3_svbmc=320/480",
            "--gpyreg-source",
            GPYREG_SOURCE,
            DIRTY,
        )
    )
    manifest = manifest_of(out)
    assert [e["label"] for e in manifest["allocation"]] == list(
        runner.POOL_LABELS
    )
    for entry in manifest["allocation"]:
        ring = entry["label"] == "ring_D2_noise3_svbmc"
        assert entry["target_filtered"] == 320
        assert entry["max_seeds"] == (480 if ring else 350)
        assert entry["seed_start"] == 1000
    result = ok(cli("cases", "--out", out))
    lines = result.stdout.splitlines()
    assert len(lines) == 2930
    listing = tmp_path / "cases.txt"
    listing.write_text(result.stdout, encoding="utf-8", newline="\n")
    assert contract.read_cases(listing) == lines
    assert (
        lines[0]
        == "multisensory_s1_D6_noise3_svbmc/multisensory_s1_D6_noise3_svbmc_seed1000"
    )
    assert (
        lines[-1]
        == "multisensory_s1_D6_svbmc/multisensory_s1_D6_svbmc_seed1349"
    )
    subset = ok(cli("cases", "--out", out, "--subset", "ring_D2_noise3_svbmc"))
    rows = subset.stdout.splitlines()
    assert len(rows) == 480
    subset_file = tmp_path / "ring.txt"
    subset_file.write_text(subset.stdout, encoding="utf-8", newline="\n")
    indices = contract.read_subset(subset_file, lines)
    assert indices == list(range(1401, 1881))
    assert rows[0] == "1401 ring_D2_noise3_svbmc/ring_D2_noise3_svbmc_seed1000"


def test_cases_refuses_an_unknown_subset(campaign):
    out, _ = campaign
    result = cli("cases", "--out", out, "--subset", "no_such_condition")
    assert result.returncode == 1
    assert "no subset 'no_such_condition'" in result.stderr
    assert result.stdout == ""


# --------------------------------------------------------------------------
# select, the stacking harness and the sweep
# --------------------------------------------------------------------------


def test_select_takes_the_lowest_seeds_that_pass(campaign, tmp_path):
    out, _ = campaign
    passing = [
        tag
        for tag in completed_tags(out)
        if read_json(pool_io.record_path(out, tag))["verdict"]["passes"]
    ]
    ok(cli("select", "--out", str(out)))
    selection = read_json(out / "selection.json")
    (condition,) = selection["conditions"]
    assert condition["label"] == LABEL
    assert condition["target_filtered"] == TARGET
    assert condition["runs"] == [
        {"tag": tag, "seed": runner.tag_seed(tag)} for tag in passing[:TARGET]
    ]
    assert condition["selected"] == len(condition["runs"])
    assert condition["shortfall"] == TARGET - condition["selected"]
    assert condition["seeds_scanned"] >= condition["selected"]
    # Over the scanned prefix, not over the condition: `summarize` owns
    # the pass rate of every completed case.
    assert "pass_rate" not in condition
    assert 0.0 < condition["pass_rate_scanned"] <= 1.0
    assert condition["pass_rate_scanned"] == (
        condition["selected"] / condition["seeds_scanned"]
    )
    assert selection["totals"]["selected"] == condition["selected"]
    assert condition["unfinished"] == []
    assert selection["identity"] == manifest_of(out)["identity"]
    assert selection["manifest_sha256"] == contract.sha256_file(
        out / "manifest.json"
    )
    markdown = (out / "selection.md").read_text(encoding="utf-8")
    assert markdown.startswith("# S-VBMC filtered pool")
    assert "pass rate over the scanned seeds" in markdown
    assert "stops as soon as the target is met" in markdown

    copy = copied(out, tmp_path, "one")
    ok(cli("select", "--out", str(copy), "--target", "1"))
    lowered = read_json(copy / "selection.json")
    assert lowered["target_override"] == 1
    assert [r["tag"] for r in lowered["conditions"][0]["runs"]] == passing[:1]


def synthetic_pool(out, states, target):
    """A pool directory whose cases hold what ``states`` names, seed by seed
    from ``SEED_START``: ``passes`` or ``filtered_out`` (a completion
    record with that verdict), ``failed`` (an error file), ``claimed`` (a
    claim alone, a case in flight or interrupted) or ``missing``
    (nothing). Only what ``select`` reads of a case is written."""
    contract.write_json(
        Path(out) / "manifest.json",
        {
            "campaign": "svbmc_pool",
            "contract": contract.CONTRACT_VERSION,
            "identity": {},
            "allocation": [
                {
                    "label": LABEL,
                    "seed_start": SEED_START,
                    "max_seeds": len(states),
                    "target_filtered": target,
                }
            ],
        },
    )
    for seed, state in enumerate(states, start=SEED_START):
        tag = tag_of(seed)
        if state in ("passes", "filtered_out"):
            contract.write_json(
                contract.record_path(out, tag),
                {"tag": tag, "verdict": {"passes": state == "passes"}},
            )
        elif state == "failed":
            plant(out, f"{tag}.error.txt", f"{tag}: ValueError: x\n")
        elif state == "claimed":
            claim_by(out, tag, "700", str(seed))
    return Path(out)


def select_in_process(out, capsys):
    code = runner.main(["select", "--out", str(out)])
    return code, capsys.readouterr()


@pytest.mark.parametrize("unfinished", ["missing", "claimed"])
def test_select_refuses_an_unfinished_seed_below_a_selected_run(
    tmp_path, capsys, unfinished
):
    """A missing or claimed seed would have come first had it finished, so
    the selection is not the pool, and nothing is written."""
    states = ["passes", unfinished, "filtered_out", "passes", "passes"]
    out = synthetic_pool(tmp_path / "pool", states, target=3)
    code, printed = select_in_process(out, capsys)
    assert code == 1
    assert f"seeds {SEED_START + 1} lie below the last selected seed" in (
        printed.err
    )
    assert "select refuses" in printed.err
    assert not (out / "selection.json").exists()
    assert not (out / "selection.md").exists()


def test_select_lists_the_unfinished_seeds_past_a_shortfall(tmp_path, capsys):
    """Short of its target, the walk reaches the end of the range; the
    unfinished seeds past the last selected run are listed, and the
    selection is written."""
    states = ["failed", "passes", "filtered_out", "missing", "claimed"]
    out = synthetic_pool(tmp_path / "pool", states, target=2)
    code, printed = select_in_process(out, capsys)
    assert code == 0, printed.err
    selection = read_json(out / "selection.json")
    (condition,) = selection["conditions"]
    assert condition["runs"] == [
        {"tag": tag_of(SEED_START + 1), "seed": SEED_START + 1}
    ]
    assert condition["shortfall"] == 1
    assert condition["unfinished"] == [SEED_START + 3, SEED_START + 4]
    assert condition["seeds_scanned"] == 3
    assert condition["failed_while_scanning"] == 1
    assert condition["pass_rate_scanned"] == 1 / 3
    assert selection["totals"]["unfinished"] == 2
    assert selection["verification_sha256"] is None
    markdown = (out / "selection.md").read_text(encoding="utf-8")
    assert f"- {LABEL}: seeds {SEED_START + 3}-{SEED_START + 4}" in markdown
    # Met at the fourth seed, the walk stops there and the rest is never
    # looked at.
    states = ["passes", "failed", "filtered_out", "passes", "missing"]
    out = synthetic_pool(tmp_path / "met", states, target=2)
    code, printed = select_in_process(out, capsys)
    assert code == 0, printed.err
    (condition,) = read_json(out / "selection.json")["conditions"]
    assert [run["seed"] for run in condition["runs"]] == [
        SEED_START,
        SEED_START + 3,
    ]
    assert condition["unfinished"] == []
    assert condition["seeds_scanned"] == 4


def test_a_selection_is_stackable_only_after_a_passing_verification(
    campaign, tmp_path
):
    """What the stacking comparison's `prepare` requires of a pool: a
    verification that passed, a selection made after it, and a selection
    that is the stopping rule on the verified cases."""
    out, _ = campaign
    copy = copied(out, tmp_path)
    ok(cli("select", "--out", copy))
    with pytest.raises(RuntimeError, match="holds no verification.json"):
        runner.stackable_selection(copy)
    ok(cli("verify", "--out", copy))
    with pytest.raises(RuntimeError, match="held no verification.json"):
        runner.stackable_selection(copy)
    ok(cli("select", "--out", copy))
    read = runner.stackable_selection(copy)
    assert read["verification_sha256"] == contract.sha256_file(
        copy / "verification.json"
    )
    assert read["verification_counts"] == verification(copy)["counts"]
    # A verification after the selection makes it stale.
    ok(cli("verify", "--out", copy))
    with pytest.raises(RuntimeError, match="another verification.json"):
        runner.stackable_selection(copy)
    ok(cli("select", "--out", copy))
    runner.stackable_selection(copy)
    # A selection that records no hashes is dated against the report.
    selection = read_json(copy / "selection.json")
    undated = {
        key: value
        for key, value in selection.items()
        if key not in ("manifest_sha256", "verification_sha256")
    }
    contract.write_json(
        copy / "selection.json", dict(undated, generated="2000-01-01 00:00:00")
    )
    with pytest.raises(RuntimeError, match="precedes"):
        runner.stackable_selection(copy)
    contract.write_json(
        copy / "selection.json", dict(undated, generated="2999-01-01 00:00:00")
    )
    runner.stackable_selection(copy)
    # A selection that leaves out a passing run is not the pool.
    runs = selection["conditions"][0]["runs"]
    assert runs
    edited = json.loads(json.dumps(selection))
    edited["conditions"][0]["runs"] = runs[1:]
    contract.write_json(copy / "selection.json", edited)
    with pytest.raises(
        RuntimeError, match="the stopping rule on the verified"
    ):
        runner.stackable_selection(copy)
    # A report that places a seed of the walk neither verified nor failed,
    # while the files the selection is made from are complete.
    report = verification(copy)
    report["cases"][0]["status"] = "missing"
    contract.write_json(copy / "verification.json", report)
    ok(cli("select", "--out", copy))
    with pytest.raises(RuntimeError, match="neither verified nor failed"):
        runner.stackable_selection(copy)
    # A report that did not pass.
    report = verification(copy)
    report["cases"][0]["status"] = "verified"
    report["counts"]["partial"], report["exit_code"] = 1, 1
    contract.write_json(copy / "verification.json", report)
    ok(cli("select", "--out", copy))
    with pytest.raises(RuntimeError, match="verification did not pass"):
        runner.stackable_selection(copy)


def test_a_selection_is_stacked_with_the_allocations_numbers(
    campaign, tmp_path
):
    """The stopping rule is replayed with the allocation's first seed, seed
    cap and filtered target, or with the target that `select --target`
    gave, which the check returns for the stacking to record."""
    out, _ = campaign
    copy = copied(out, tmp_path)
    ok(cli("verify", "--out", copy))
    ok(cli("select", "--out", copy, "--target", "1"))
    assert runner.stackable_selection(copy)["target_override"] == 1
    ok(cli("select", "--out", copy))
    assert runner.stackable_selection(copy)["target_override"] is None
    selection = read_json(copy / "selection.json")
    for key, value in (
        ("seed_start", SEED_START - 1),
        ("seed_cap", MAX_SEEDS + 1),
        ("target_filtered", TARGET + 1),
    ):
        edited = json.loads(json.dumps(selection))
        edited["conditions"][0][key] = value
        contract.write_json(copy / "selection.json", edited)
        with pytest.raises(
            RuntimeError, match=f"allocation's numbers \\({key} {value}, not"
        ):
            runner.stackable_selection(copy)
    edited = json.loads(json.dumps(selection))
    edited["conditions"][0]["label"] = "gmm_D2_svbmc"
    contract.write_json(copy / "selection.json", edited)
    with pytest.raises(RuntimeError, match="not a condition of the alloc"):
        runner.stackable_selection(copy)


def test_the_stack_harness_reads_the_selection(campaign, tmp_path):
    """The comparison's pool reader resolves the selected runs' files.

    The reader is ``svbmc_pool_stack.py``'s; it is checked here because
    this is where a pool directory is generated.
    """
    import svbmc_pool_stack as stack

    out, _ = campaign
    copy = copied(out, tmp_path, "selected")
    ok(cli("select", "--out", str(copy), "--target", "1"))
    selection = read_json(copy / "selection.json")
    chosen = [run["tag"] for run in selection["conditions"][0]["runs"]]
    conditions, identities, labels = stack.pool_conditions([copy])
    assert labels == [LABEL]
    assert [entry["name"] for entry in conditions[LABEL]] == chosen
    assert [Path(entry["path"]).resolve() for entry in conditions[LABEL]] == [
        (copy / tag).resolve() for tag in chosen
    ]
    assert identities[0]["selection"]["conditions"] == {
        LABEL: "selection.json"
    }
    # A selection whose runs the directory no longer holds names the
    # entry that is wrong, not only the file that is absent.
    (copy / f"{chosen[0]}.npz").unlink()
    with pytest.raises(RuntimeError, match=r"selection\.json"):
        stack.pool_conditions([copy])


def test_a_changed_artifact_stops_the_sweep(stored, tmp_path):
    out, tag = stored
    copy = copied(out, tmp_path, "tampered")
    path = contract.record_path(copy, tag)
    record = read_json(path)
    record["artifacts"][f"{tag}.npz"]["sha256"] = "0" * 64
    contract.write_json(path, record)
    result = cli("run", "--out", str(copy))
    assert result.returncode != 0
    assert f"{tag}.npz differs from its recorded SHA-256" in result.stderr
    assert f"START {tag}" not in result.stdout


def test_a_partial_artifact_stops_the_sweep(campaign, tmp_path):
    out, _ = campaign
    copy = copied(out, tmp_path, "partial")
    # The first seed of the condition, so the sweep reaches it whatever
    # the filters did: its record and sidecar are gone, its .npz is not.
    tag = tag_of(SEED_START)
    contract.record_path(copy, tag).unlink()
    (copy / f"{tag}.json").unlink()
    result = cli("run", "--out", str(copy))
    assert result.returncode != 0
    assert "incomplete prior attempt" in result.stderr


def test_case_state_as_the_sweep_sees_it(stored, tmp_path, no_slurm):
    out, done = stored
    copy = copied(out, tmp_path)
    expected = manifest_of(copy)["identity"]
    assert runner.case_state(copy, done, expected)[0] == "done"
    assert runner.case_state(copy, tag_of(4005), expected) == ("new", None)
    plant(copy, f"{tag_of(4005)}.error.txt")
    assert runner.case_state(copy, tag_of(4005), expected) == ("failed", None)
    plant(copy, f"{tag_of(4006)}.npz")
    assert runner.case_state(copy, tag_of(4006), expected) == (
        "partial",
        [f"{tag_of(4006)}.npz"],
    )
    # A claim made outside Slurm is live while its process runs here.
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    claim = contract.new_claim(tag_of(4006), task={})
    claim["pid"] = gone.pid
    contract.write_json(contract.claim_path(copy, tag_of(4006)), claim)
    assert runner.case_state(copy, tag_of(4006), expected) == (
        "interrupted",
        [f"{tag_of(4006)}.npz"],
    )
    # A task killed before it wrote any file leaves its stale claim alone.
    (copy / f"{tag_of(4006)}.npz").unlink()
    assert runner.case_state(copy, tag_of(4006), expected) == (
        "interrupted",
        [],
    )
    claim["pid"] = os.getpid()
    contract.write_json(contract.claim_path(copy, tag_of(4006)), claim)
    assert runner.case_state(copy, tag_of(4006), expected)[0] == "in_flight"


def test_a_stale_log_is_not_a_partial_artifact(tmp_path):
    tag = tag_of(SEED_START)
    plant(tmp_path, f"{tag}.log", "interrupted\n")
    assert runner.partial_artifacts(tmp_path, tag) == []
    plant(tmp_path, f"{tag}.npz", "")
    assert runner.partial_artifacts(tmp_path, tag) == [f"{tag}.npz"]


def test_a_worker_that_died_leaves_a_failed_case(tmp_path):
    """What `run` makes of a worker that died without its error file: the
    tail of its log as the error file, no partial file, and no claim of
    its own; a claim some other process made stays."""
    tag = tag_of(SEED_START)
    log = plant(
        tmp_path, f"{tag}.log", "".join(f"line {i}\n" for i in range(60))
    )
    plant(tmp_path, f"{tag}.npz")
    claim = contract.new_claim(tag, task={})
    claim["pid"] = 424242
    contract.write_json(contract.claim_path(tmp_path, tag), claim)
    runner.died_worker(tmp_path, tag, -9, log, 424242)
    error = contract.error_path(tmp_path, tag).read_text(encoding="utf-8")
    assert error.startswith(f"{tag}: worker exited with code -9")
    assert "line 59" in error and "line 19\n" not in error
    assert runner.partial_artifacts(tmp_path, tag) == []
    assert not contract.claim_path(tmp_path, tag).exists()
    contract.write_json(contract.claim_path(tmp_path, tag), claim)
    runner.died_worker(tmp_path, tag, -9, log, 1)
    assert contract.claim_path(tmp_path, tag).exists()


# --------------------------------------------------------------------------
# prepare
# --------------------------------------------------------------------------


def test_prepare_requires_the_gpyreg_source(tmp_path):
    out = tmp_path / "no_source"
    result = cli(
        "prepare", "--out", out, "--suite", "smoke", "--only", LABEL, DIRTY
    )
    assert result.returncode == 2
    assert "--gpyreg-source" in result.stderr
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    result = cli(
        *prepare_args(out, TARGET, MAX_SEEDS, DIRTY),
        env=environment(PYVBMC_GPYREG_SOURCE=elsewhere),
    )
    assert result.returncode != 0
    assert "is not PYVBMC_GPYREG_SOURCE" in result.stderr
    assert not (out / "manifest.json").exists()


def test_prepare_refuses_a_structural_change(campaign, tmp_path):
    out, _ = campaign
    copy = copied(out, tmp_path)
    # Everything as prepared, but not the recorded relaxation.
    result = cli(*prepare_args(copy))
    assert result.returncode != 0
    assert "allow_dirty" in result.stderr
    edit_manifest(
        copy,
        lambda m: m["identity"]["source"]["files"].update(
            {"dev/scripts/benchmark_targets.py": "0" * 64}
        ),
    )
    result = prepare(copy)
    assert result.returncode != 0
    assert "identity (files.dev/scripts/benchmark_targets.py)" in result.stderr


def test_revised_history_guards_a_revision(campaign):
    out, _ = campaign
    previous = manifest_of(out)
    lowered = [
        dict(entry, target_filtered=1) for entry in previous["allocation"]
    ]
    history = runner.revised_history(out, previous, lowered)
    assert history[-1]["allocation"] == previous["allocation"]
    assert history[-1]["revised"]
    with pytest.raises(RuntimeError, match="cannot be dropped"):
        runner.revised_history(out, previous, [])
    moved = [
        dict(entry, seed_start=entry["seed_start"] + 1)
        for entry in previous["allocation"]
    ]
    with pytest.raises(RuntimeError, match="first seed"):
        runner.revised_history(out, previous, moved)


def test_revised_allocation_and_pilot_seeds(campaign, tmp_path):
    """A revised target, then a pilot sweep that ignores it.

    The copied campaign already holds its runs, so the sweep reruns
    nothing: the filtered target is lowered to one, and ``--pilot-seeds
    2`` still walks two seeds of the condition.
    """
    out, _ = campaign
    copy = copied(out, tmp_path, "revised")
    revision = ok(prepare(copy, target=1))
    assert "allocation revised" in revision.stdout
    before = manifest_of(out)
    manifest = manifest_of(copy)
    assert manifest["allocation"][0]["target_filtered"] == 1
    assert manifest["allocation_history"][-1] == {
        "allocation": before["allocation"],
        "revised": manifest["allocation_history"][-1]["revised"],
    }
    for key in (
        "campaign",
        "contract",
        "suite",
        "options",
        "identity",
        "site",
        "pip_freeze",
        "finishing_steps",
        "created",
    ):
        assert manifest[key] == before[key]
    # Prepared again as it is, nothing changes.
    ok(prepare(copy, target=1))
    assert manifest_of(copy) == manifest

    result = ok(cli("run", "--out", str(copy), "--pilot-seeds", "2"))
    assert "START" not in result.stdout
    finished = read_json(copy / "finished.json")
    condition = finished["conditions"][LABEL]
    assert condition["seeds_run"] == 2
    assert condition["seeds"] == [SEED_START, SEED_START + 1]


def bogus_case(out, label="no_such_target_D2", seed=1):
    """Allocate a condition whose target the suite does not define."""

    def add(manifest):
        manifest["allocation"].append(
            {
                "label": label,
                "seed_start": seed,
                "max_seeds": 1,
                "target_filtered": 1,
            }
        )

    edit_manifest(out, add)
    return label, tag_of(seed, label)


def test_a_failed_case_is_skipped_and_counted(tmp_path):
    """A case that failed earlier costs its seed and does not fail a sweep."""
    out = tmp_path / "failures"
    ok(prepare(out))
    planted = [tag_of(SEED_START + i) for i in range(2)]
    for tag in planted:
        plant(out, f"{tag}.error.txt", "exit code 1\nplanted failure\n")
    result = ok(cli("run", "--out", str(out), "--pilot-seeds", "2"))
    for tag in planted:
        assert f"SKIP {tag} (failed earlier)" in result.stdout
    assert "START" not in result.stdout
    finished = read_json(out / "finished.json")
    assert finished["failed"] == planted
    assert finished["failed_this_sweep"] == []
    condition = finished["conditions"][LABEL]
    assert condition["seeds_run"] == 2 and condition["failed"] == 2

    # A case that fails in this sweep does fail it: an allocation entry
    # naming a target the suite does not define makes its worker exit.
    _, tag = bogus_case(out)
    result = cli("run", "--out", str(out), "--pilot-seeds", "1")
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"FAILED {tag}" in result.stdout
    # The worker wrote the record of its own failure; the supervisor kept
    # it rather than replacing it with the exit code and the log tail.
    error = (out / f"{tag}.error.txt").read_text(encoding="utf-8")
    assert error.startswith(f"{tag}: ValueError")
    finished = read_json(out / "finished.json")
    assert finished["failed_this_sweep"] == [tag]
    assert not contract.claim_path(out, tag).exists()


def test_a_failing_worker_records_the_case_and_exits_non_zero(tmp_path):
    """`<tag>.error.txt` is the whole of what a failed case leaves.

    The worker removes the files an earlier attempt had begun before it
    runs, writes the error file with the traceback, writes no completion
    record, releases its claim and exits 1, which is how Slurm sees the
    failure. `select` and `summarize` then count the case as failed.
    """
    out = tmp_path / "array"
    ok(prepare(out))
    label, tag = bogus_case(out)
    for suffix in runner.ARTIFACT_SUFFIXES:
        plant(out, f"{tag}{suffix}", "partial")

    result = cli("worker", "--out", str(out), "--case", tag)
    assert result.returncode == 1, result.stdout + result.stderr
    error = (out / f"{tag}.error.txt").read_text(encoding="utf-8")
    assert error.startswith(f"{tag}: ValueError")
    assert label in error and "Traceback" in error
    assert not pool_io.record_path(out, tag).exists()
    assert runner.partial_artifacts(out, tag) == []
    assert not contract.claim_path(out, tag).exists()

    ok(cli("select", "--out", str(out)))
    selection = read_json(out / "selection.json")
    (failed,) = [c for c in selection["conditions"] if c["label"] == label]
    assert failed["failed_while_scanning"] == 1
    assert failed["selected"] == 0 and failed["shortfall"] == 1

    ok(cli("summarize", "--out", str(out)))
    summary = read_json(out / "summary.json")
    (counted,) = [c for c in summary["conditions"] if c["label"] == label]
    assert counted["failed"] == 1 and counted["completed"] == 0
    assert counted["failures"][0]["tag"] == tag
    assert "ValueError" in counted["failures"][0]["reason"]
    ok(cli("verify", "--out", out))
    assert reported(verification(out), tag)["status"] == "failed"


def test_a_manifest_with_extra_keys_is_read(campaign, tmp_path):
    """Every command reads a manifest by the keys it needs."""
    out, _ = campaign
    copy = copied(out, tmp_path, "extra_keys")
    edit_manifest(
        copy,
        lambda m: m.update({"released": True, "released_by": "a reviewer"}),
    )
    ok(cli("cases", "--out", str(copy)))
    ok(cli("select", "--out", str(copy)))
    ok(cli("summarize", "--out", str(copy)))


def test_prepare_allocates_the_whole_pool_suite(tmp_path):
    """The campaign's approved allocation, with no allocation flag.

    Without `--only` the allocation is the eight pool conditions, and
    without allocation flags it is the sizes the PI approved: 100 filtered
    runs per noisy condition and 50 per noiseless control, seed caps 150,
    200 for the ring and 75 for the controls.
    """
    out = tmp_path / "pool"
    ok(cli("prepare", "--out", out, "--gpyreg-source", GPYREG_SOURCE, DIRTY))
    manifest = manifest_of(out)
    allocated = [entry["label"] for entry in manifest["allocation"]]
    assert allocated == list(runner.POOL_LABELS)
    assert manifest["suite"] == "svbmc_pool"
    sizes = {
        entry["label"]: (entry["target_filtered"], entry["max_seeds"])
        for entry in manifest["allocation"]
    }
    assert sizes == {
        "multisensory_s1_D6_noise3_svbmc": (100, 150),
        "multisensory_s1_D6_noise1.3_svbmc": (100, 150),
        "rosenbrock_D2_noise3_svbmc": (100, 150),
        "gmm_D2_noise3_svbmc": (100, 150),
        "ring_D2_noise3_svbmc": (100, 200),
        "student_D8_noise3_svbmc": (100, 150),
        "gmm_D2_svbmc": (50, 75),
        "multisensory_s1_D6_svbmc": (50, 75),
    }
    assert all(entry["seed_start"] == 1000 for entry in manifest["allocation"])


def test_only_allocates_a_subset(tmp_path):
    out = tmp_path / "subset"
    wanted = [runner.POOL_LABELS[4], runner.POOL_LABELS[1]]
    ok(
        cli(
            "prepare",
            "--out",
            out,
            "--only",
            ",".join(wanted),
            "--gpyreg-source",
            GPYREG_SOURCE,
            DIRTY,
        )
    )
    manifest = manifest_of(out)
    assert [e["label"] for e in manifest["allocation"]] == wanted


def test_dirty_trees_refuses_an_uncommitted_source():
    def record(harness, gpyreg):
        return {
            "source": {
                "trees": {
                    "harness": {"commit": "a", "clean": harness},
                    "gpyreg": {"commit": "b", "clean": gpyreg},
                }
            },
            "imports": {
                "trees": {
                    "harness": {"dirty": [" M pyvbmc/vbmc/vbmc.py"]},
                    "gpyreg": {"dirty": ["?? scratch.py"]},
                }
            },
        }

    assert runner.dirty_trees(record(True, True), False) == []
    with pytest.raises(RuntimeError, match="pyvbmc/vbmc/vbmc.py"):
        runner.dirty_trees(record(False, True), False)
    assert runner.dirty_trees(record(False, True), True) == ["harness"]
    with pytest.raises(RuntimeError, match="gpyreg checkout"):
        runner.dirty_trees(record(True, False), True)


def test_allocation_overrides_and_bounds():
    """The narrower the flag, the later it wins."""

    class Args:
        target = max_seeds = control_target = control_max_seeds = None
        seed_start = 1000
        allocation = []

    from benchmark_targets import suite_configs

    def allocate():
        return {
            entry["label"]: (entry["target_filtered"], entry["max_seeds"])
            for entry in runner.allocation(Args(), suite_configs("svbmc_pool"))
        }

    # Nothing asked for: the campaign's approved allocation, with the
    # ring's own seed cap.
    sizes = allocate()
    assert sizes["rosenbrock_D2_noise3_svbmc"] == (
        runner.DEFAULT_TARGET,
        runner.DEFAULT_MAX_SEEDS,
    )
    assert sizes["ring_D2_noise3_svbmc"] == (
        runner.DEFAULT_TARGET,
        runner.DEFAULT_SEED_CAPS["ring_D2_noise3_svbmc"],
    )
    assert sizes["gmm_D2_svbmc"] == (
        runner.DEFAULT_CONTROL_TARGET,
        runner.DEFAULT_CONTROL_MAX_SEEDS,
    )
    # `--target` and `--max-seeds` set every condition, the noiseless
    # control and the ring included, so a small campaign stays small.
    Args.target, Args.max_seeds = 2, 3
    assert set(allocate().values()) == {(2, 3)}
    # `--control-*` then narrows the noiseless conditions alone.
    Args.target, Args.max_seeds = 60, 90
    Args.control_target, Args.control_max_seeds = 30, 45
    sizes = allocate()
    assert sizes["gmm_D2_svbmc"] == (30, 45)
    assert sizes["rosenbrock_D2_noise3_svbmc"] == (60, 90)
    assert sizes["ring_D2_noise3_svbmc"] == (60, 90)
    # `--allocation` then names one condition.
    Args.allocation = ["ring_D2_noise3_svbmc=40/80"]
    sizes = allocate()
    assert sizes["ring_D2_noise3_svbmc"] == (40, 80)
    assert sizes["rosenbrock_D2_noise3_svbmc"] == (60, 90)
    Args.target, Args.max_seeds = 90, 60
    with pytest.raises(RuntimeError, match="exceeds the seed cap"):
        runner.allocation(Args(), suite_configs("svbmc_pool"))


@pytest.mark.parametrize(
    "item", ["ring_D2_noise3_svbmc=40", "=40/80", "ring_D2_noise3_svbmc=x/80"]
)
def test_allocation_rejects_a_malformed_argument(item):
    with pytest.raises(RuntimeError, match="LABEL=TARGET/MAXSEEDS"):
        runner.parse_override(item)


class _Posterior:
    def __init__(self, stable, max_J_sjk):
        self.stats = {
            "stable": stable,
            "J_sjk": np.array([[[0.1, max_J_sjk], [0.0, 0.2]]]),
        }


@pytest.mark.parametrize(
    "stable,max_J_sjk,passes",
    [(True, 4.9, True), (True, 5.1, False), (False, 1.0, False)],
)
def test_filter_verdict(stable, max_J_sjk, passes):
    verdict = pool_io.filter_verdict(_Posterior(stable, max_J_sjk))
    assert verdict["stable"] is stable
    assert verdict["max_J_sjk"] == max_J_sjk
    assert verdict["passes"] is passes


# --------------------------------------------------------------------------
# The worker: early exit, refusals, claims, clean-up
# --------------------------------------------------------------------------


def test_the_worker_exits_at_once_on_a_completed_case(stored, tmp_path):
    out, tag = stored
    copy = copied(out, tmp_path)
    before = snapshot(copy)
    result = ok(cli("worker", "--out", copy, "--case", tag))
    assert "already has a completion record; nothing to do" in result.stdout
    assert snapshot(copy) == before


@pytest.mark.parametrize(
    "line",
    ["nonsense", tag_of(SEED_START + MAX_SEEDS), tag_of(SEED_START) + " 1"],
)
def test_the_worker_refuses_a_line_outside_the_allocation(
    stored, tmp_path, line
):
    out, _ = stored
    copy = copied(out, tmp_path)
    before = snapshot(copy)
    result = cli("worker", "--out", copy, "--case", line)
    assert result.returncode == runner.EXIT_USAGE
    assert "is not a case of the allocation" in result.stderr
    assert snapshot(copy) == before


def test_the_worker_refuses_a_directory_that_is_not_its_campaign(
    stored, tmp_path, capsys
):
    """A directory without a readable manifest, or with another harness's,
    exits 64 and is left as it is."""
    _, tag = stored
    empty = tmp_path / "empty"
    empty.mkdir()
    assert worker(empty, tag) == runner.EXIT_USAGE
    assert "holds no readable manifest.json" in capsys.readouterr().err
    assert list(empty.iterdir()) == []
    other = tmp_path / "other"
    other.mkdir()
    contract.write_json(
        other / "manifest.json",
        {"harness": "population_run", "contract": contract.CONTRACT_VERSION},
    )
    assert worker(other, tag) == runner.EXIT_USAGE
    assert "is not a manifest of svbmc_pool" in capsys.readouterr().err
    assert sorted(path.name for path in other.iterdir()) == ["manifest.json"]


#: A ``sitecustomize`` under which ``importlib.metadata`` finds no installed
#: distribution of the names it holds, in every process whose
#: ``PYTHONPATH`` names its directory.
HIDDEN_METADATA = """\
import importlib.metadata as _metadata

_HIDDEN = {names!r}
_from_name = _metadata.Distribution.from_name.__func__


def _hiding_from_name(cls, name):
    if name.lower().replace("_", "-") in _HIDDEN:
        raise _metadata.PackageNotFoundError(name)
    return _from_name(cls, name)


_metadata.Distribution.from_name = classmethod(_hiding_from_name)
"""
#: What a process prints of the installed versions of these packages.
METADATA_PROBE = """
import importlib.metadata as metadata
for name in ("pyvbmc", "gpyreg", "numpy"):
    try:
        print(metadata.version(name))
    except metadata.PackageNotFoundError:
        print("absent")
"""


def without_metadata(directory, names=("pyvbmc", "gpyreg")):
    """``PYTHONPATH`` for processes in which no distribution of ``names`` is
    installed, as in the campaign environment, which imports PyVBMC and
    gpyreg from their source trees on ``sys.path``: ``directory`` holds a
    ``sitecustomize`` that hides them, ahead of this process's path."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "sitecustomize.py").write_text(
        HIDDEN_METADATA.format(names=set(names)), encoding="utf-8"
    )
    inherited = os.environ.get("PYTHONPATH")
    return os.pathsep.join(
        [str(directory), *([inherited] if inherited else [])]
    )


def hide_metadata(monkeypatch, *names):
    """Make ``importlib.metadata`` find no distribution of ``names`` in this
    process, for the duration of a test."""
    from importlib import metadata

    original = metadata.Distribution.from_name.__func__

    def from_name(cls, name):
        if name.lower().replace("_", "-") in names:
            raise metadata.PackageNotFoundError(name)
        return original(cls, name)

    monkeypatch.setattr(
        metadata.Distribution, "from_name", classmethod(from_name)
    )


def test_a_case_runs_where_the_packages_have_no_metadata(tmp_path):
    """The campaign environment installs neither PyVBMC nor gpyreg, and
    ``importlib.metadata`` finds no distribution of either: a case
    prepared, run and verified there completes, and its artifact and
    record give None for the versions the installed metadata would name."""
    env = environment(PYTHONPATH=without_metadata(tmp_path / "site"))
    probe = subprocess.run(
        [sys.executable, "-c", METADATA_PROBE],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.split()[:2] == ["absent", "absent"]
    assert probe.stdout.split()[2] == np.__version__
    out = tmp_path / "pool"
    ok(cli(*prepare_args(out, 1, 1, DIRTY), env=env))
    (line,) = ok(cli("cases", "--out", out, env=env)).stdout.splitlines()
    ok(cli("worker", "--out", out, "--case", line, env=env))
    tag = contract.case_tag(line)
    unnamed = {"pyvbmc": None, "gpyreg": None}
    meta = pool_io.load_run(out / tag)["meta"]
    assert meta["identity"]["host"]["installed_metadata_versions"] == unnamed
    record = read_json(pool_io.record_path(out, tag))
    assert record["identity"]["imports"]["installed_metadata_versions"] == (
        unnamed
    )
    ok(cli("verify", "--out", out, env=env))
    assert verification(out)["counts"]["verified"] == 1


def test_the_flat_identity_names_no_version_it_cannot_read(monkeypatch):
    """The flat identity, which the pool readers record and the stacking's
    original arm reports, reads four versions from the installed metadata;
    each is None where no distribution is installed."""
    hide_metadata(monkeypatch, "pyvbmc", "gpyreg", "numpy", "scipy")
    record = runner.identity(GPYREG_SOURCE)
    assert record["host"]["installed_metadata_versions"] == {
        "pyvbmc": None,
        "gpyreg": None,
    }
    assert record["source"]["numpy"] is None
    assert record["source"]["scipy"] is None


def test_a_fresh_claim_holds_the_case_while_it_runs(
    fresh, stored, slurm, monkeypatch
):
    """The identity and host fields of a case an array task ran."""
    source_out, source_tag = stored
    tag = tag_of(SEED_START)
    seen = {}
    monkeypatch.setattr(
        runner, "run_case", replay(source_out, source_tag, seen)
    )
    as_task(monkeypatch, 700, 1)
    assert worker(fresh, tag) == 0
    assert seen["claim"]["job"] == "700" and seen["claim"]["array_task"] == "1"
    assert seen["claim"]["restart_count"] == 0
    assert seen["files"] == []
    assert not contract.claim_path(fresh, tag).exists()
    manifest = manifest_of(fresh)
    record = runner.check_record(fresh, tag, manifest["identity"])
    assert record["identity"]["source"] == manifest["identity"]["source"]
    host = record["identity"]["host"]
    assert host["slurm"] == {
        "job_id": "700001",
        "array_job_id": "700",
        "array_task_id": "1",
        "restart_count": "0",
        "node": "node1",
        "partition": None,
        "cpus_per_task": None,
    }
    assert host["node_features"] == {
        "node": "node1",
        "available": ["stubfeat"],
        "active": ["stubfeat"],
    }
    assert host["threads"] == {k: "1" for k in runner.THREAD_KEYS}
    assert any(entry["user_api"] == "blas" for entry in host["blas"])
    if sys.platform.startswith("linux"):
        assert host["cpu_affinity"]["cpus"] and host["cpu_model"]
    else:
        assert host["cpu_affinity"] is None
    assert Path(record["identity"]["imports"]["modules"]["gpyreg"]) == (
        GPYREG_SOURCE / "gpyreg"
    )
    ok(cli("verify", "--out", fresh, env=stub_environment(slurm)))
    assert reported(verification(fresh), tag)["status"] == "verified"


def test_a_live_claim_refuses_the_case_and_leaves_its_files(
    fresh, slurm, monkeypatch
):
    tag = tag_of(SEED_START)
    plant(fresh, f"{tag}.npz", "the running task's\n")
    claim_by(fresh, tag, "600", "7")
    answer(slurm, "600_7", "RUNNING\n")
    before = snapshot(fresh)
    monkeypatch.setattr(runner, "run_case", must_not_run)
    as_task(monkeypatch, 700, 1)
    assert worker(fresh, tag) == contract.EXIT_CLAIMED
    assert snapshot(fresh) == before


@pytest.mark.parametrize("how", ["fails", "answers nothing"])
def test_a_claim_the_accounting_cannot_judge_is_live(
    fresh, slurm, monkeypatch, how
):
    tag = tag_of(SEED_START)
    plant(fresh, f"{tag}.npz", "a task's\n")
    claim_by(fresh, tag, "600", "7")
    if how == "fails":
        answer(slurm, "600_7", fail=True)
    before = snapshot(fresh)
    monkeypatch.setattr(runner, "run_case", must_not_run)
    as_task(monkeypatch, 700, 1)
    assert worker(fresh, tag) == contract.EXIT_CLAIMED
    assert snapshot(fresh) == before


def test_a_stale_claim_is_retired_and_the_case_starts_clean(
    fresh, stored, slurm, monkeypatch
):
    source_out, source_tag = stored
    tag = tag_of(SEED_START)
    for suffix in runner.ARTIFACT_SUFFIXES:
        plant(fresh, f"{tag}{suffix}", "the killed attempt's\n")
    claim_by(fresh, tag, "600", "7")
    answer(slurm, "600_7", "CANCELLED by 1000\n")
    seen = {}
    monkeypatch.setattr(
        runner, "run_case", replay(source_out, source_tag, seen)
    )
    as_task(monkeypatch, 700, 1)
    assert worker(fresh, tag) == 0
    assert seen["files"] == []
    assert seen["claim"]["job"] == "700"
    assert len(retired_claims(fresh, tag, "600_7")) == 1
    assert not contract.claim_path(fresh, tag).exists()
    runner.check_record(fresh, tag, manifest_of(fresh)["identity"])
    assert runner.partial_artifacts(fresh, tag) == [
        f"{tag}.npz",
        f"{tag}.json",
    ]


def test_a_requeued_task_takes_over_its_own_claim(
    fresh, stored, slurm, monkeypatch, capsys
):
    source_out, source_tag = stored
    tag = tag_of(SEED_START)
    claim_by(fresh, tag, "700", "1", restart_count=0)
    seen = {}
    monkeypatch.setattr(
        runner, "run_case", replay(source_out, source_tag, seen)
    )
    as_task(monkeypatch, 700, 1, restart=1)
    assert worker(fresh, tag) == 0
    assert "took over the requeue claim of 700_1" in capsys.readouterr().out
    assert seen["claim"]["restart_count"] == 1
    assert seen["claim"]["previous"]["reason"] == "requeue"
    assert not (slurm / "sacct_queries").exists()
    assert not contract.claim_path(fresh, tag).exists()
    assert contract.record_path(fresh, tag).exists()


def test_the_worker_refuses_another_identity_and_leaves_the_case(
    fresh, slurm, monkeypatch
):
    tag = tag_of(SEED_START)
    edit_manifest(
        fresh,
        lambda m: m["identity"]["source"]["files"].update(
            {"dev/scripts/benchmark_targets.py": "0" * 64}
        ),
    )
    plant(fresh, f"{tag}.npz", "an earlier attempt's\n")
    before = snapshot(fresh)
    monkeypatch.setattr(runner, "run_case", must_not_run)
    as_task(monkeypatch, 700, 1)
    assert worker(fresh, tag) == contract.EXIT_IDENTITY
    assert snapshot(fresh) == before


def test_a_successful_worker_clears_an_earlier_error_file(
    fresh, stored, monkeypatch
):
    source_out, source_tag = stored
    tag = tag_of(SEED_START)
    plant(fresh, f"{tag}.error.txt", "an attempt\n")
    monkeypatch.setattr(runner, "run_case", replay(source_out, source_tag))
    assert worker(fresh, tag) == 0
    assert not contract.error_path(fresh, tag).exists()
    assert contract.record_path(fresh, tag).exists()


def test_a_sigterm_leaves_a_missing_case(fresh, slurm, monkeypatch):
    """Slurm's SIGTERM, at the time limit or on scancel, in the middle of
    the run: the partial files and the claim go, and no error file is
    written, so that `verify` reports the case as missing."""
    tag = tag_of(SEED_START)

    def stopped(out, manifest, tag, label, seed, save_vbmc=False):
        plant(out, f"{tag}.npz", "half\n")
        signal.raise_signal(signal.SIGTERM)
        raise AssertionError("the signal must stop the run")

    monkeypatch.setattr(runner, "run_case", stopped)
    as_task(monkeypatch, 700, 1)
    assert worker(fresh, tag) == 128 + signal.SIGTERM
    assert runner.partial_artifacts(fresh, tag) == []
    for path in (
        contract.claim_path(fresh, tag),
        contract.error_path(fresh, tag),
        contract.record_path(fresh, tag),
    ):
        assert not path.exists()
    ok(cli("verify", "--out", fresh, env=stub_environment(slurm)))
    assert reported(verification(fresh), tag)["status"] == "missing"


def test_a_killed_case_is_interrupted_and_its_resubmission_completes(
    fresh, stored, slurm, monkeypatch
):
    """A task killed outright leaves its files and its claim; `verify`
    reports the case as interrupted, and the next task takes the stale
    claim over and starts from clean."""
    source_out, source_tag = stored
    tag = tag_of(SEED_START + 1)
    plant(fresh, f"{tag}.npz", "the killed attempt's\n")
    claim_by(fresh, tag, "800", "2")
    answer(slurm, "800_2", "OUT_OF_MEMORY\n")
    ok(cli("verify", "--out", fresh, env=stub_environment(slurm)))
    case = reported(verification(fresh), tag)
    assert case["status"] == "interrupted"
    assert case["files"] == [f"{tag}.npz"]
    assert case["claim"]["owner"] == "800_2"

    seen = {}
    monkeypatch.setattr(
        runner, "run_case", replay(source_out, source_tag, seen)
    )
    as_task(monkeypatch, 900, 2)
    assert worker(fresh, tag) == 0
    assert seen["files"] == []
    assert len(retired_claims(fresh, tag, "800_2")) == 1
    ok(cli("verify", "--out", fresh, env=stub_environment(slurm)))
    assert reported(verification(fresh), tag)["status"] == "verified"


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------


def test_verify_reconciles_the_allocation(campaign, tmp_path):
    """Every case of the allocation is placed, and the counts add up."""
    out, _ = campaign
    copy = copied(out, tmp_path)
    ok(cli("verify", "--out", str(copy)))
    report = verification(copy)
    assert report["contract"] == contract.CONTRACT_VERSION
    assert [case["index"] for case in report["cases"]] == list(
        range(1, MAX_SEEDS + 1)
    )
    assert [case["seed"] for case in report["cases"]] == [
        SEED_START + i for i in range(MAX_SEEDS)
    ]
    counts = report["counts"]
    assert counts["verified"] == len(completed_tags(copy))
    assert sum(counts[s] for s in contract.STATUSES) == MAX_SEEDS
    assert counts["stray"] == 0 and report["exit_code"] == 0
    for case in report["cases"]:
        if case["status"] == "verified":
            assert isinstance(case["passes"], bool)
            assert case["differences"]["I_sk"] <= pool_io.TOL_STATS
            assert case["differences"]["J_sjk"] <= pool_io.TOL_STATS
    (condition,) = report["conditions"]
    assert condition["label"] == LABEL
    assert [condition[key] for key in contract.STATUSES] == [
        counts[key] for key in contract.STATUSES
    ]
    assert report["node_feature"] is None
    assert report["verifier_differs_in"] == []


def test_verify_places_every_state(campaign, tmp_path):
    """verified, verify_failed, failed, in_flight, interrupted, partial,
    missing and the stray files, in one directory."""
    out, _ = campaign
    copy = copied(out, tmp_path)
    edit_manifest(copy, lambda m: m["allocation"][0].update(max_seeds=8))
    state = tmp_path / "slurm_state"
    state.mkdir()
    stubs.write_stubs(tmp_path / "bin")
    completed = completed_tags(copy)
    assert len(completed) >= 2
    verified, tampered = completed[:2]
    artifact = copy / f"{tampered}.npz"
    data = bytearray(artifact.read_bytes())
    data[10] ^= 0xFF
    artifact.write_bytes(bytes(data))
    plant(
        copy, f"{tag_of(4003)}.error.txt", f"{tag_of(4003)}: ValueError: x\n"
    )
    plant(copy, f"{tag_of(4004)}.npz")
    claim_by(copy, tag_of(4004), "700", "4")
    answer(state, "700_4", "RUNNING\n")
    plant(copy, f"{tag_of(4005)}.npz")
    claim_by(copy, tag_of(4005), "700", "5")
    answer(state, "700_5", "OUT_OF_MEMORY\n")
    plant(copy, f"{tag_of(4006)}.npz")
    plant(copy, f"{LABEL}/{LABEL}_seed9999.npz")
    plant(copy, f"records/{LABEL}/{LABEL}_seed9998.complete.json", "{}")
    plant(copy, "orphan.npz")
    claim_by(copy, f"{LABEL}/{LABEL}_seed9997", "700", "9")
    result = cli("verify", "--out", copy, env=stub_environment(state))
    assert result.returncode == 1, result.stdout + result.stderr
    report = verification(copy)
    statuses = {case["tag"]: case["status"] for case in report["cases"]}
    expected = {
        tag_of(seed): "verified" if tag_of(seed) in completed else "missing"
        for seed in range(SEED_START, SEED_START + MAX_SEEDS)
    }
    expected.update(
        {
            tampered: "verify_failed",
            tag_of(4003): "failed",
            tag_of(4004): "in_flight",
            tag_of(4005): "interrupted",
            tag_of(4006): "partial",
            tag_of(4007): "missing",
        }
    )
    assert statuses == expected
    assert statuses[verified] == "verified"
    assert (
        "differs from its recorded SHA-256"
        in reported(report, tampered)["error"]
    )
    assert reported(report, tag_of(4003))["reason"].startswith(tag_of(4003))
    assert reported(report, tag_of(4004))["claim"]["owner"] == "700_4"
    assert reported(report, tag_of(4005))["files"] == [f"{tag_of(4005)}.npz"]
    assert reported(report, tag_of(4006))["files"] == [f"{tag_of(4006)}.npz"]
    assert report["stray"] == sorted(
        [
            f"claims/{LABEL}/{LABEL}_seed9997",
            f"records/{LABEL}/{LABEL}_seed9998.complete.json",
            f"{LABEL}/{LABEL}_seed9999.npz",
            "orphan.npz",
        ]
    )
    counts = report["counts"]
    assert counts["stray"] == 4
    (condition,) = report["conditions"]
    for status in contract.STATUSES:
        assert condition[status] == counts[status]
        assert counts[status] == list(statuses.values()).count(status)
    assert (
        f"{reported(report, tag_of(4006))['index']} {tag_of(4006)} partial"
        in result.stdout
    )
    # The finish goes on over failed, in-flight, interrupted and missing
    # cases, and stops on the rest.
    code, _ = contract.finish_decision(
        report, allow_missing=True, allow_running=True
    )
    assert code == contract.FINISH_FATAL
    for name in (
        f"{tampered}.npz",
        f"{tag_of(4006)}.npz",
        f"{LABEL}/{LABEL}_seed9999.npz",
        f"records/{LABEL}/{LABEL}_seed9998.complete.json",
        "orphan.npz",
        f"claims/{LABEL}/{LABEL}_seed9997",
    ):
        (copy / name).unlink()
    contract.record_path(copy, tampered).unlink()
    (copy / f"{tampered}.json").unlink()
    ok(cli("verify", "--out", copy, env=stub_environment(state)))
    report = verification(copy)
    assert report["exit_code"] == 0
    code, _ = contract.finish_decision(
        report, allow_missing=True, allow_running=True
    )
    assert code == 0


def test_verify_checks_the_node_feature_and_the_core(stored, tmp_path):
    """Where the site names a node feature, every record must show it and
    one physical core."""
    out, tag = stored
    copy = copied(out, tmp_path)
    edit_manifest(copy, lambda m: m["site"].update(NODE_FEATURE="stubfeat"))
    others = [t for t in completed_tags(copy) if t != tag]
    for other in others:
        contract.record_path(copy, other).unlink()
        for suffix in pool_io.SUFFIXES:
            (copy / f"{other}{suffix}").unlink()
    path = contract.record_path(copy, tag)

    def host(features, cores):
        record = read_json(path)
        record["identity"]["host"].update(
            node_features=features,
            cpu_affinity={
                "cpus": list(range(len(cores))),
                "physical_cores": cores,
                "core_threads": {c: [i] for i, c in enumerate(cores)},
            },
        )
        contract.write_json(path, record)
        return cli("verify", "--out", copy)

    result = cli("verify", "--out", copy)
    assert result.returncode == 1
    assert "no node features" in reported(verification(copy), tag)["error"]
    features = {
        "node": "n1",
        "available": ["stubfeat", "x"],
        "active": ["stubfeat"],
    }
    ok(host(features, ["0:3"]))
    assert reported(verification(copy), tag)["status"] == "verified"
    assert verification(copy)["node_feature"] == "stubfeat"
    assert host(features, ["0:3", "0:4"]).returncode == 1
    assert (
        "spans 2 physical cores" in reported(verification(copy), tag)["error"]
    )
    other = {"node": "n1", "available": ["x"], "active": ["x"]}
    assert host(other, ["0:3"]).returncode == 1
    assert (
        "lacks the feature stubfeat"
        in reported(verification(copy), tag)["error"]
    )


def test_only_the_source_part_of_a_record_is_compared(stored, tmp_path):
    """Another node may have run a case; other code may not have."""
    out, tag = stored
    copy = copied(out, tmp_path, "elsewhere")
    path = contract.record_path(copy, tag)
    record = read_json(path)
    expected = manifest_of(copy)["identity"]
    record["identity"]["host"].update(
        {
            "hostname": "cluster-node-07",
            "executable": "/scratch/env/bin/python",
            "platform": "Linux-5.14.0-x86_64-with-glibc2.34",
        }
    )
    record["identity"]["imports"]["trees"]["harness"]["path"] = "/elsewhere"
    contract.write_json(path, record)
    assert runner.check_record(copy, tag, expected)["tag"] == tag
    record["identity"]["source"]["files"][
        "dev/scripts/benchmark_targets.py"
    ] = ("0" * 64)
    contract.write_json(path, record)
    with pytest.raises(contract.CompletionError, match="benchmark_targets"):
        runner.check_record(copy, tag, expected)
    record = read_json(path)
    record["identity"]["source"] = expected["source"]
    record["seed"] = 1
    contract.write_json(path, record)
    with pytest.raises(contract.CompletionError, match="seed 1"):
        runner.check_record(copy, tag, expected)


def copied_elsewhere(out, tmp_path, name="pool"):
    """A copy of the campaign whose manifest names a checkout not here.

    What a pool copied from the machine that generated it looks like: the
    manifest's ``gpyreg_source`` is an absolute path of that machine.
    """
    copy = copied(out, tmp_path, name)
    edit_manifest(
        copy,
        lambda m: m.update(
            gpyreg_source=str(tmp_path / "elsewhere" / "gpyreg_1.2.1")
        ),
    )
    return copy


def test_verify_accepts_a_gpyreg_source_at_the_manifest_commit(
    campaign, tmp_path
):
    """A copied pool is verified against a local checkout at the pin."""
    out, _ = campaign
    copy = copied_elsewhere(out, tmp_path)
    manifest = manifest_of(copy)
    # Without the flag the manifest's path is used, and it is not here.
    result = cli("verify", "--out", str(copy))
    assert result.returncode != 0
    assert "no gpyreg package under" in result.stderr
    assert not (copy / "verification.json").exists()
    assert runner.manifest_gpyreg_commit(manifest) == runner.git(
        GPYREG_SOURCE, "rev-parse", "HEAD"
    )
    assert runner.pinned_gpyreg_source(GPYREG_SOURCE, manifest) == str(
        GPYREG_SOURCE
    )
    ok(cli("verify", "--out", copy, "--gpyreg-source", GPYREG_SOURCE))
    report = verification(copy)
    assert report["gpyreg_source"] == str(GPYREG_SOURCE)
    assert report["rounding_factor"] == pool_io.ROUNDING_FACTOR
    # The verifier's identity is recorded next to the pool's, its host
    # part saying where it ran, its source part compared for the report
    # only: this checkout generated the campaign, so nothing differs.
    assert set(report["verifier"]) == {"contract", "source", "imports", "host"}
    assert Path(report["verifier"]["imports"]["modules"]["gpyreg"]) == (
        GPYREG_SOURCE / "gpyreg"
    )
    assert report["verifier_differs_in"] == []
    for case in report["cases"]:
        if case["status"] == "verified":
            assert set(case["relative"]) == {"I_sk", "J_sjk"}
            assert case["condition_number"] >= 1.0
    (condition,) = report["conditions"]
    assert condition["max_condition_number"] >= 1.0
    assert condition["max_relative"] >= 0.0


def perturbed_artifact(out, tmp_path, amount):
    """A copy of the campaign with one stored `I_sk` entry moved.

    The largest entry of the first completed run's ``I_sk`` is shifted by
    ``amount`` in the ``.npz`` and the completion record's entry is
    rewritten to the new file, so that only the recomputation gate can
    tell; returns the copy, the tag and the array's largest magnitude.
    """
    copy = copied(out, tmp_path, "perturbed")
    tag = completed_tags(copy)[0]
    npz = copy / f"{tag}.npz"
    with np.load(npz, allow_pickle=False) as stored_arrays:
        arrays = {k: stored_arrays[k] for k in stored_arrays.files}
    stats = arrays["vp/stats/I_sk"].copy()
    index = np.unravel_index(np.argmax(np.abs(stats)), stats.shape)
    scale = float(np.abs(stats[index]))
    stats[index] += amount
    arrays["vp/stats/I_sk"] = stats
    np.savez_compressed(npz, **arrays)
    path = contract.record_path(copy, tag)
    record = read_json(path)
    record["artifacts"][f"{tag}.npz"] = {
        "sha256": contract.sha256_file(npz),
        "bytes": npz.stat().st_size,
    }
    contract.write_json(path, record)
    return copy, tag, scale


def test_recomputation_gate_allows_amplified_rounding_only(campaign, tmp_path):
    """The gate is the absolute tolerance plus each GP's own rounding.

    On the machine that generated a run the recomputation is exact, so
    the gate's allowance is measured here by moving one stored value: a
    move beyond the allowance fails, whatever the factor below it, and a
    factor sized to that move passes and reports it.
    """
    out, _ = campaign
    tag = completed_tags(out)[0]
    intact = pool_io.verify_run(out / tag)
    assert intact["differences"]["I_sk"] == 0.0
    assert intact["relative"]["I_sk"] == 0.0
    assert intact["condition_number"] >= 1.0
    assert intact["rounding_factor"] == pool_io.ROUNDING_FACTOR
    allowance = intact["tolerance"]["I_sk"]
    assert allowance >= pool_io.TOL_STATS
    amount = 2.0 * allowance
    copy, tag, scale = perturbed_artifact(out, tmp_path, amount)
    record = contract.record_path(copy, tag)
    with pytest.raises(RuntimeError, match="recomputed I_sk differs"):
        pool_io.verify_run(copy / tag, record=record)
    with pytest.raises(RuntimeError, match="recomputed I_sk differs"):
        pool_io.verify_run(copy / tag, rounding_factor=0.0, record=record)
    # A factor whose allowance covers the move: the gate's rounding term
    # is factor * eps * cond(K) relative to the largest stored value.
    eps = np.finfo(float).eps
    factor = float(2.0 * amount / (eps * intact["condition_number"] * scale))
    report = pool_io.verify_run(
        copy / tag, rounding_factor=factor, record=record
    )
    assert report["differences"]["I_sk"] == pytest.approx(amount)
    assert report["relative"]["I_sk"] == pytest.approx(amount / scale)
    assert report["tolerance"]["I_sk"] >= amount
    assert report["rounding_factor"] == factor
    # The same through the command, which records the factor and the
    # per-condition extremes of the gate.
    result = cli("verify", "--out", str(copy))
    assert result.returncode == 1, result.stdout + result.stderr
    case = reported(verification(copy), tag)
    assert case["status"] == "verify_failed"
    assert "recomputed I_sk differs" in case["error"]
    ok(cli("verify", "--out", str(copy), "--rounding-factor", str(factor)))
    report = verification(copy)
    assert report["rounding_factor"] == factor
    case = reported(report, tag)
    assert case["status"] == "verified"
    assert case["relative"]["I_sk"] == pytest.approx(amount / scale)
    (condition,) = report["conditions"]
    assert condition["max_relative"] == pytest.approx(amount / scale)
    assert condition["max_condition_number"] >= intact["condition_number"]


def other_gpyreg(tmp_path):
    """A git checkout of a package named gpyreg that is not the campaign's."""
    other = tmp_path / "other_gpyreg"
    (other / "gpyreg").mkdir(parents=True)
    (other / "gpyreg" / "__init__.py").write_text(
        "# not the campaign's gpyreg\n", encoding="utf-8"
    )
    author = [
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.com",
        "-c",
        "commit.gpgsign=false",
    ]
    subprocess.run(["git", "-C", str(other), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(other), *author, "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(other), *author, "commit", "-q", "-m", "x"],
        check=True,
    )
    return other


def test_verify_refuses_a_gpyreg_source_at_another_commit(campaign, tmp_path):
    """A copied pool is verified against the pinned library alone."""
    out, _ = campaign
    copy = copied(out, tmp_path)
    other = other_gpyreg(tmp_path)
    result = cli("verify", "--out", copy, "--gpyreg-source", other)
    assert result.returncode != 0
    pinned = runner.manifest_gpyreg_commit(manifest_of(copy))
    # The refusal names the manifest's commit, and nothing was verified.
    assert f"not the manifest's {pinned}" in result.stderr
    assert not (copy / "verification.json").exists()
    # A directory that is no git checkout is refused with a plain message.
    plain = tmp_path / "plain"
    (plain / "gpyreg").mkdir(parents=True)
    result = cli("verify", "--out", copy, "--gpyreg-source", plain)
    assert result.returncode != 0
    assert "is not a git checkout" in result.stderr
    assert not (copy / "verification.json").exists()


# --------------------------------------------------------------------------
# The array worker and `run` on one machine
# --------------------------------------------------------------------------


def sidecar_content(path):
    """A sidecar's tree without what differs between two runs of a case:
    the time it was written, the seconds it took and memory addresses."""
    tree = read_json(path)
    for key in TIMING_META:
        tree["meta"].pop(key)
    tree["logger"].pop("total_fun_eval_time")

    def strip(value):
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items()}
        if isinstance(value, list):
            return [strip(v) for v in value]
        if isinstance(value, str):
            return ADDRESS.sub(" at 0x", value)
        return value

    return strip(tree)


def test_the_array_worker_and_run_give_identical_artifacts(stored, tmp_path):
    """One case, run by `run` in the module's campaign and by `worker
    --case` as the task of an array job in another campaign prepared
    alike: every stored array but the evaluations' seconds is equal bit
    for bit, and the sidecar and the record differ in their timings and
    the host part alone."""
    out, tag = stored
    other = tmp_path / "array"
    ok(prepare(other))
    state = tmp_path / "slurm_state"
    state.mkdir()
    stubs.write_stubs(tmp_path / "bin")
    task = stub_environment(
        state,
        SLURM_JOB_ID="1001001",
        SLURM_ARRAY_JOB_ID="1001",
        SLURM_ARRAY_TASK_ID="1",
        SLURMD_NODENAME="node1",
    )
    index = runner.case_lines(manifest_of(other)).index(tag) + 1
    assert index == runner.tag_seed(tag) - SEED_START + 1
    ok(cli("worker", "--out", other, "--case", tag, env=task))
    with np.load(out / f"{tag}.npz", allow_pickle=False) as a, np.load(
        other / f"{tag}.npz", allow_pickle=False
    ) as b:
        assert sorted(a.files) == sorted(b.files)
        for key in a.files:
            assert a[key].dtype == b[key].dtype, key
            assert a[key].shape == b[key].shape, key
            if key not in TIMING_ARRAYS:
                assert a[key].tobytes() == b[key].tobytes(), key
    assert sidecar_content(out / f"{tag}.json") == sidecar_content(
        other / f"{tag}.json"
    )
    ran, arrayed = (
        read_json(contract.record_path(d, tag)) for d in (out, other)
    )
    for record in (ran, arrayed):
        # The size of the sidecar follows the digits of its timings.
        record["verification"].pop("bytes")
    own = set(ran) - set(CONTRACT_FIELDS) - {"wall_s", "target_eval_s"}
    assert own == set(arrayed) - set(CONTRACT_FIELDS) - {
        "wall_s",
        "target_eval_s",
    }
    assert {k: ran[k] for k in own} == {k: arrayed[k] for k in own}
    assert ran["identity"]["source"] == arrayed["identity"]["source"]
    assert ran["identity"]["host"]["slurm"]["job_id"] is None
    assert arrayed["identity"]["host"]["slurm"]["array_job_id"] == "1001"


# --------------------------------------------------------------------------
# The tracked copies
# --------------------------------------------------------------------------


def test_the_tracked_copies_of_a_pool_are_redacted(campaign, tmp_path):
    """The module's pool, finished and then given what a cluster leaves in
    it (the site block, the login node's and the nodes' host parts, the
    paths of the operator's trees, the task logs and the accounting), is
    copied for the repository with none of it."""
    out, _ = campaign
    site = stubs.FakeSite(tmp_path)
    pool = site.home / "runs" / "pool"
    shutil.copytree(out, pool)
    (pool / "verification.json").unlink(missing_ok=True)
    for step in ("verify", "select", "summarize"):
        ok(cli(step, "--out", str(pool)))
    manifest = manifest_of(pool)
    assert manifest["tracked_copies"] == runner.TRACKED_COPIES

    def at_site(manifest):
        manifest["site"] = site.site_block("dev/scripts/svbmc_pool_run.py")
        site.plant(manifest["identity"])
        manifest["gpyreg_source"] = str(site.gpyreg)
        manifest["pip_freeze"].append(
            f"gpyreg @ file://{site.gpyreg.as_posix()}"
        )

    edit_manifest(pool, at_site)
    tags = completed_tags(pool)
    for index, tag in enumerate(tags, start=1):
        site.rewrite(
            contract.record_path(pool, tag),
            lambda record: site.plant(
                record["identity"], node=site.nodes[index % 2], task=index
            ),
        )

    def verified_there(report):
        site.plant(report["verifier"], node=site.nodes[0])
        report["gpyreg_source"] = str(site.gpyreg)

    site.rewrite(pool / "verification.json", verified_there)
    site.write_slurm(pool)
    assert site.leaks(pool)
    target = tmp_path / "handback" / "pools"
    # The interpreter and its libraries, which the host parts of the
    # identities name, lie outside the stand-in site; they are named here,
    # as an operator names them with --path.
    python = [("PYTHON", sys.prefix)]
    contract.redact(
        pool,
        target,
        operator=site.operator(),
        environ={},
        paths=python,
        host="fakelogin9",
        say=lambda message: None,
    )
    assert site.leaks(target) == []
    assert sorted(p.name for p in target.iterdir()) == [
        "manifest.json",
        "redaction.json",
        "selection.json",
        "selection.md",
        "summary.json",
        "summary.md",
        "verification.json",
    ]
    copied_manifest = read_json(target / "manifest.json")
    assert "site" not in copied_manifest
    assert copied_manifest["gpyreg_source"] == "$PYVBMC_GPYREG_SOURCE"
    assert copied_manifest["identity"]["host"]["hostname"] == "login"
    assert copied_manifest["allocation"] == manifest["allocation"]
    report = read_json(target / "verification.json")
    assert report["directory"].startswith("~")
    assert report["verifier"]["host"]["hostname"] == site.family
    assert report["counts"] == read_json(pool / "verification.json")["counts"]
    assert read_json(target / "selection.json")["directory"].startswith("~")
    assert read_json(target / "selection.json")["conditions"] == (
        read_json(pool / "selection.json")["conditions"]
    )
    # A field no rule knows of, holding the username, is refused.
    site.rewrite(pool / "summary.json", lambda s: s.update(by=site.user))
    with pytest.raises(contract.ContractError, match="the username"):
        contract.redact(
            pool,
            tmp_path / "handback" / "again",
            operator=site.operator(),
            environ={},
            paths=python,
            say=lambda message: None,
        )
    assert not (tmp_path / "handback" / "again").exists()


# --------------------------------------------------------------------------
# The flat layout of the pools prepared before the contract
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def flat(campaign, tmp_path_factory):
    """The module's pool in the flat layout, as ``pool_20260914`` holds it.

    Its manifest has no ``contract`` field and holds the flat identity;
    its artifacts lie at the top as ``<label>_seed<seed>.*``, and its
    records hold the flat identity and the hashes by suffix and no
    convergence field.
    """
    out, _ = campaign
    flat_out = tmp_path_factory.mktemp("flat_pool")
    manifest = manifest_of(out)
    identity = runner.identity(GPYREG_SOURCE)
    contract.write_json(
        flat_out / "manifest.json",
        {
            **{
                key: manifest[key]
                for key in (
                    "campaign",
                    "suite",
                    "options",
                    "allocation",
                    "gpyreg_source",
                    "allow_dirty",
                    "created",
                    "allocation_history",
                )
            },
            "identity": identity,
        },
    )
    for tag in completed_tags(out):
        name = tag.split("/")[1]
        for suffix in pool_io.SUFFIXES:
            shutil.copyfile(
                out / f"{tag}{suffix}", flat_out / f"{name}{suffix}"
            )
        record = read_json(pool_io.record_path(out, tag))
        contract.write_json(
            pool_io.record_path(flat_out, name),
            {
                "tag": name,
                "label": record["label"],
                "seed": record["seed"],
                "identity": identity,
                "elapsed_seconds": record["elapsed_seconds"],
                **{
                    key: record[key]
                    for key in (
                        "wall_s",
                        "target_eval_s",
                        "K",
                        "func_count",
                        "elbo",
                        "elbo_sd",
                        "success_flag",
                        "verdict",
                        "metrics",
                        "verification",
                    )
                },
                "hashes": pool_io.artifact_hashes(flat_out, name),
            },
        )
    return flat_out


def flat_tags(out):
    return sorted(
        (
            path.name[: -len(".complete.json")]
            for path in (Path(out) / "records").glob("*.complete.json")
        ),
        key=runner.tag_seed,
    )


def test_a_flat_pool_verifies_selects_and_summarizes(flat, tmp_path):
    copy = copied(flat, tmp_path, "flat")
    tags = flat_tags(copy)
    assert tags
    result = ok(cli("verify", "--out", copy))
    report = verification(copy)
    assert "contract" not in report
    assert report["counts"] == {
        "verified": len(tags),
        "failed": 0,
        "partial": 0,
        "missing": MAX_SEEDS - len(tags),
        "verify_failed": 0,
        "stray": 0,
    }
    assert [c["tag"] for c in report["cases"]] == [
        f"{LABEL}_seed{SEED_START + i}" for i in range(MAX_SEEDS)
    ]
    assert set(report["verifier"]) == {"source", "host"}
    assert "| verified | failed | partial | missing |" in result.stdout
    listed = ok(cli("cases", "--out", copy)).stdout.splitlines()
    assert listed == [f"{LABEL} {SEED_START + i}" for i in range(MAX_SEEDS)]
    ok(cli("select", "--out", copy))
    (condition,) = read_json(copy / "selection.json")["conditions"]
    assert [run["tag"] for run in condition["runs"]] == [
        tag
        for tag in tags
        if read_json(pool_io.record_path(copy, tag))["verdict"]["passes"]
    ][:TARGET]
    # Selected after its verification, the flat pool can be stacked.
    assert runner.stackable_selection(copy)["verification_sha256"] == (
        contract.sha256_file(copy / "verification.json")
    )
    ok(cli("summarize", "--out", copy))
    written = read_json(copy / "summary.json")
    (summary,) = written["conditions"]
    assert summary["completed"] == len(tags)
    assert summary["success_flag"] == sum(
        read_json(pool_io.record_path(copy, tag))["success_flag"]
        for tag in tags
    )
    assert written["totals"]["success_flag"] == summary["success_flag"]
    # The flat records hold no convergence status, and the summary holds
    # no field for one.
    assert not set(runner.CONTRACT_SUMMARY_KEYS) & set(summary)
    text = (copy / "summary.md").read_text(encoding="utf-8")
    assert f"| {LABEL} |" in text
    assert "| success |" in text and "converged" not in text


def test_a_flat_pool_reports_its_faults(flat, tmp_path):
    """A changed artifact, an artifact without its record, a stray one."""
    copy = copied(flat, tmp_path, "flat")
    first = flat_tags(copy)[0]
    artifact = copy / f"{first}.npz"
    data = bytearray(artifact.read_bytes())
    data[10] ^= 0xFF
    artifact.write_bytes(bytes(data))
    plant(copy, "foo_seed1.npz")
    result = cli("verify", "--out", copy)
    assert result.returncode == 1
    report = verification(copy)
    assert reported(report, first)["status"] == "verify_failed"
    assert report["stray"] == ["foo_seed1"]
    pool_io.record_path(copy, first).unlink()
    assert cli("verify", "--out", copy).returncode == 1
    case = reported(verification(copy), first)
    assert case["status"] == "partial"
    assert case["files"] == [f"{first}.npz", f"{first}.json"]


def test_a_flat_pool_is_never_extended(flat, tmp_path):
    copy = copied(flat, tmp_path, "flat")
    before = snapshot(copy)
    result = prepare(copy)
    assert result.returncode != 0 and "flat layout" in result.stderr
    result = cli("worker", "--out", copy, "--case", f"{LABEL} {SEED_START}")
    assert result.returncode == runner.EXIT_USAGE
    result = cli("run", "--out", copy)
    assert result.returncode != 0 and "flat layout" in result.stderr
    assert snapshot(copy) == before


def test_the_stack_harness_reads_every_passing_record_of_a_flat_pool(
    flat, tmp_path
):
    import svbmc_pool_stack as stack

    copy = copied(flat, tmp_path, "flat")
    conditions, identities, labels = stack.pool_conditions([copy])
    assert labels == [LABEL]
    passing = [
        tag
        for tag in flat_tags(copy)
        if read_json(pool_io.record_path(copy, tag))["verdict"]["passes"]
    ]
    assert [entry["name"] for entry in conditions[LABEL]] == passing
    assert identities[0]["selection"]["conditions"] == {
        LABEL: "every passing record"
    }


def test_the_flat_identity_helpers():
    """A flat record written before the split into halves holds both in
    one mapping; the differences name every mismatch."""
    record = runner.identity(GPYREG_SOURCE)
    assert set(record["source"]) == set(runner.SOURCE_KEYS)
    single = dict(record["source"], **record["host"])
    assert runner.identity_source(single) == record["source"]
    assert runner.identity_host(single) == record["host"]
    expected = {"pyvbmc_commit": "a", "runner_sha256": "b", "numpy": "2.0"}
    actual = {"pyvbmc_commit": "a", "runner_sha256": "c", "hostname": "h"}
    assert runner.identity_differences(actual, expected) == [
        "hostname",
        "numpy",
        "runner_sha256",
    ]
    flat_manifest = {"identity": record}
    assert runner.manifest_gpyreg_commit(flat_manifest) == runner.git(
        GPYREG_SOURCE, "rev-parse", "HEAD"
    )


# --------------------------------------------------------------------------
# A campaign through the Slurm driver
# --------------------------------------------------------------------------

#: What a scratch checkout needs to run the pool harness: the package, the
#: harness and the modules it imports, the data the identity hashes and
#: the driver.
DRIVER_FILES = (
    ".gitignore",
    "dev/scripts/svbmc_pool_run.py",
    "dev/scripts/svbmc_pool_io.py",
    "dev/scripts/benchmark_targets.py",
    "dev/scripts/campaign_contract.py",
    "dev/scripts/profile_run.py",
    "dev/scripts/hpc/campaign_env.sh",
    "dev/scripts/hpc/campaign_submit.sh",
    "dev/scripts/hpc/campaign_task.sbatch",
    "dev/scripts/hpc/campaign_finish.sh",
    "dev/scripts/hpc/campaign_redact.sh",
)
POOL_HARNESS = "dev/scripts/svbmc_pool_run.py"


@pytest.fixture(scope="module")
def driver_template(tmp_path_factory):
    """A committed scratch checkout holding the pool harness and the
    driver, for ``test_campaign_driver.World``."""
    import test_campaign_driver as driver

    root = tmp_path_factory.mktemp("pool_driver_template")
    home = root / "home"
    home.mkdir()
    (home / ".gitconfig").write_text("", encoding="utf-8")
    repo = root / "repo"
    tracked = runner.git(ROOT, "ls-files", "pyvbmc", "dev/scripts/data")
    for name in [*tracked.splitlines(), *DRIVER_FILES]:
        source = ROOT / name
        if source.is_file():
            (repo / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, repo / name)
    (
        repo / "dev" / "scripts" / "hpc" / "campaign_requirements.txt"
    ).write_text(
        driver.environment_pins(repo / "dev" / "scripts"), encoding="utf-8"
    )
    driver.git(repo, "init", "-q", home=home)
    driver.git(repo, "add", "-A", home=home)
    driver.git(repo, "commit", "-q", "-m", "pool harness", home=home)
    return root


def test_a_pool_campaign_through_the_slurm_driver(tmp_path, driver_template):
    """Submission, one array task, and the finish: verify, select and
    summarize as batch jobs, and the archive, with the stub Slurm
    commands; the second case is left missing."""
    import test_campaign_driver as driver

    world = driver.World(tmp_path, driver_template, stubs.find_bash())
    settings = {
        "HARNESS": POOL_HARNESS,
        "PYVBMC_GPYREG_SOURCE": driver.posix(GPYREG_SOURCE),
    }
    driver.ok(
        world.submit(
            "c1",
            "--suite",
            "smoke",
            "--only",
            LABEL,
            "--target",
            "1",
            "--max-seeds",
            "2",
            "--seed-start",
            str(SEED_START),
            "--gpyreg-source",
            driver.posix(GPYREG_SOURCE),
            **settings,
        )
    )
    out = world.campaign()
    manifest = manifest_of(out)
    assert manifest["site"]["HARNESS"] == POOL_HARNESS
    assert manifest["site"]["NODE_FEATURE"] == "stubfeat"
    assert manifest["identity"]["source"]["trees"]["harness"]["clean"] is True
    lines = (out / "cases.txt").read_text("utf-8").splitlines()
    assert lines == [tag_of(SEED_START), tag_of(SEED_START + 1)]
    [call] = world.calls()
    assert "--array=1-2%200" in call["args"]

    task = world.task("c1", 1, **settings)
    assert task.returncode == 0, task.stdout + task.stderr
    tag = lines[0]
    path = contract.record_path(out, tag)
    record = read_json(path)
    assert record["identity"]["host"]["slurm"]["array_task_id"] == "1"
    assert record["identity"]["host"]["node_features"]["available"] == [
        "stubfeat"
    ]
    # A Slurm task with --hint=nomultithread runs on one physical core,
    # which this process need not; the record is given one.
    record["identity"]["host"]["cpu_affinity"] = {
        "cpus": [3],
        "physical_cores": ["0:3"],
        "core_threads": {"0:3": [3]},
    }
    contract.write_json(path, record)
    again = world.task("c1", 1, **settings)
    assert again.returncode == 0
    assert "already has a completion record" in again.stdout

    finish = world.finish("c1", "--allow-missing", **settings)
    assert finish.returncode == 0, finish.stdout + finish.stderr
    report = verification(out)
    assert report["counts"]["verified"] == 1
    assert report["counts"]["missing"] == 1
    assert report["node_feature"] == "stubfeat"
    # Each step's job has a "submitted" line and then its exit code's line.
    steps = (out / "slurm" / "steps.txt").read_text("utf-8").splitlines()
    ended = [line.split()[1:3] for line in steps if " rc=" in line]
    assert ended == [
        ["step=verify", "rc=0"],
        ["step=select", "rc=0"],
        ["step=summarize", "rc=0"],
    ]
    selection = read_json(out / "selection.json")
    assert selection["conditions"][0]["runs"] in (
        [{"tag": tag, "seed": SEED_START}],
        [],
    )
    assert read_json(out / "summary.json")["conditions"][0]["completed"] == 1
    assert sorted(world.campaigns.glob("c1.tar.zst.[0-9][0-9][0-9]"))

    # The redaction, in an account whose home holds the scratch world; the
    # interpreter this test runs lies outside it.
    account = {
        "HOME": str(world.root),
        "USER": "fakeoperator",
        "LOGNAME": "fakeoperator",
    }
    target = world.root / "handback" / "pools"
    redaction = world.run(
        "campaign_redact.sh",
        driver.posix(out),
        driver.posix(target),
        "--path",
        f"PYTHON={sys.prefix}",
        env={**settings, **account},
    )
    assert redaction.returncode == 0, redaction.stdout + redaction.stderr
    assert sorted(p.name for p in target.iterdir()) == [
        "manifest.json",
        "redaction.json",
        "selection.json",
        "selection.md",
        "summary.json",
        "summary.md",
        "verification.json",
    ]
    hostname = socket.gethostname().split(".")[0].lower()
    for path in target.iterdir():
        text = path.read_text(encoding="utf-8").lower()
        for forbidden in (
            "fakeoperator",
            hostname,
            driver.posix(world.root),
            str(world.root),
            json.dumps(str(world.root))[1:-1],
        ):
            assert forbidden.lower() not in text, (path.name, forbidden)
    report = read_json(target / "verification.json")
    assert report["verifier"]["host"]["hostname"] == "stubfeat"
    assert report["counts"] == verification(out)["counts"]
