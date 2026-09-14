"""Contracts of the S-VBMC pool generator: artifact, resume, summary.

One short campaign (``normal_D2``, at most three seeds, about a minute of
inference) is generated once for the whole module through the command
line, exactly as a real pool is, and the stored runs are then reloaded and
re-verified. Outside default pytest discovery; run it by path::

    python -m pytest dev/scripts/test_svbmc_pool_run.py -vv
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRIPT = HERE / "svbmc_pool_run.py"
GPYREG_SOURCE = (
    ROOT / "dev" / "scripts" / "runs" / "svbmc_pool_20260913" / "gpyreg"
)
LABEL = "normal_D2"
SEED_START = 4000
TARGET, MAX_SEEDS = 2, 3
AUTHORIZED_BY = "test_svbmc_pool_run"
#: The campaign's extension condition: a suite entry that is not a pool
#: condition (`svbmc_pool_run.POOL_LABELS`).
EXTENSION = "multisensory_s1_D6_noise1.3_svbmc"
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

# The campaign pins gpyreg to a frozen worktree; the modules under test
# read the variable when they are imported, and the workers inherit it.
if (GPYREG_SOURCE / "gpyreg").is_dir():
    os.environ["PYVBMC_GPYREG_SOURCE"] = str(GPYREG_SOURCE)

import svbmc_pool_io as pool_io  # noqa: E402
import svbmc_pool_run as runner  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (GPYREG_SOURCE / "gpyreg").is_dir(),
    reason="the campaign's frozen gpyreg worktree is machine-local",
)


def cli(*args):
    environment = dict(os.environ)
    environment.update({k: "1" for k in runner.THREAD_KEYS})
    environment["MPLBACKEND"] = "Agg"
    environment["PYVBMC_GPYREG_SOURCE"] = str(GPYREG_SOURCE)
    return subprocess.run(
        [sys.executable, "-u", str(SCRIPT), *args],
        cwd=str(ROOT),
        env=environment,
        capture_output=True,
        text=True,
    )


def prepare(out, *extra, target=TARGET):
    return cli(
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
        str(MAX_SEEDS),
        "--seed-start",
        str(SEED_START),
        # The pool scripts and the suite module are developed together, so
        # this campaign is generated from whatever the tree holds.
        "--allow-dirty",
        *extra,
    )


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    """One generated pool directory, shared by every check below."""
    out = tmp_path_factory.mktemp("svbmc_pool")
    assert prepare(out).returncode == 0
    unready = cli("run", "--out", str(out))
    assert unready.returncode != 0
    assert "not marked ready" in unready.stderr
    assert (
        prepare(out, "--ready", "--authorized-by", AUTHORIZED_BY).returncode
        == 0
    )
    result = cli("run", "--out", str(out))
    assert result.returncode == 0, result.stdout + result.stderr
    return out, result.stdout


def completed_tags(out):
    return [
        p.name[: -len(".complete.json")]
        for p in sorted((out / "records").glob("*.complete.json"))
    ]


def test_frozen_gpyreg_is_the_one_imported():
    import gpyreg

    assert Path(gpyreg.__file__).resolve().parent == GPYREG_SOURCE / "gpyreg"


def test_manifest_and_authorization(campaign):
    out, _ = campaign
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["launch_ready"] is True
    assert manifest["authorized_by"] == AUTHORIZED_BY
    assert manifest["allocation"] == [
        {
            "label": LABEL,
            "seed_start": SEED_START,
            "max_seeds": MAX_SEEDS,
            "target_filtered": TARGET,
        }
    ]
    assert manifest["options"] == runner.BASE_OPTIONS
    assert manifest["identity"]["gpyreg_commit"] == runner.git(
        GPYREG_SOURCE, "rev-parse", "HEAD"
    )


def test_every_case_wrote_both_files_and_a_valid_record(campaign):
    out, stdout = campaign
    tags = completed_tags(out)
    assert tags, stdout
    assert len(tags) <= MAX_SEEDS
    expected = runner.identity(GPYREG_SOURCE)
    for tag in tags:
        for suffix in pool_io.SUFFIXES:
            assert (out / f"{tag}{suffix}").exists()
        record = runner.validate_case(out, tag, expected)
        assert record["identity"]["gpyreg_import"].startswith(
            str(GPYREG_SOURCE)
        )
        assert set(record["verdict"]) == {
            "stable",
            "max_J_sjk",
            "s_max",
            "passes",
        }
        assert record["K"] > 0 and record["func_count"] > 0
        assert record["verification"]["checks"] == LIVE_CHECKS
        assert (out / f"{tag}.npz").stat().st_size < 10**6


def test_stopping_rule_and_seeds(campaign):
    out, _ = campaign
    tags = completed_tags(out)
    seeds = sorted(int(t.rsplit("seed", 1)[1]) for t in tags)
    assert seeds == list(range(SEED_START, SEED_START + len(seeds)))
    passing = sum(
        json.loads(pool_io.record_path(out, tag).read_text(encoding="utf-8"))[
            "verdict"
        ]["passes"]
        for tag in tags
    )
    assert passing >= TARGET or len(tags) == MAX_SEEDS


def test_load_run_rebuilds_the_stored_posterior(campaign):
    out, _ = campaign
    tag = completed_tags(out)[0]
    state = pool_io.load_run(out / tag, rng=0)
    assert set(state) == set(pool_io.load_run.KEYS)
    sidecar = json.loads((out / f"{tag}.json").read_text(encoding="utf-8"))
    assert sidecar["vp"]["stats"]["I_sk"] == "@@npz:vp/stats/I_sk"
    with np.load(out / f"{tag}.npz", allow_pickle=False) as stored:
        np.testing.assert_array_equal(
            state["vp"].stats["I_sk"], stored["vp/stats/I_sk"]
        )
        np.testing.assert_array_equal(state["gp"].X, stored["gp/X"])
    for key in pool_io.STATS_KEYS:
        assert key in state["vp"].stats
    meta = state["meta"]
    assert meta["label"] == LABEL and meta["target"] == "normal"
    assert meta["seed"] == int(tag.rsplit("seed", 1)[1])
    assert set(meta["filter"]) == {"stable", "max_J_sjk", "s_max", "passes"}
    assert np.isfinite(meta["metrics"]["gskl"])
    assert meta["identity"]["gpyreg_import"].startswith(str(GPYREG_SOURCE))
    assert state["vp"].rng is not None


def test_verify_run_passes_post_hoc(campaign):
    out, _ = campaign
    for tag in completed_tags(out):
        report = pool_io.verify_run(out / tag)
        assert report["checks"] == [
            "stats_keys",
            "recomputation_gate",
            "dtype_canary",
            "hashes",
        ]
        assert report["differences"]["I_sk"] <= pool_io.TOL_STATS
        assert report["differences"]["J_sjk"] <= pool_io.TOL_STATS
        assert report["dtype_leaves"] > 0


def test_verify_run_compares_the_live_evaluations(campaign):
    """The live checks, against a rebuilt run standing in for the live one.

    A second rebuild of the same artifact holds the same objects, so
    every live check passes on it; changing one recorded evaluation
    makes the comparison of the evaluations fail.
    """
    out, _ = campaign
    tag = completed_tags(out)[0]
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


def test_second_run_skips_every_completed_case(campaign):
    out, _ = campaign
    before = {
        tag: pool_io.artifact_hashes(out, tag) for tag in completed_tags(out)
    }
    result = cli("run", "--out", str(out))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "START" not in result.stdout
    for tag in before:
        assert f"SKIP {tag}" in result.stdout
        assert pool_io.artifact_hashes(out, tag) == before[tag]


def test_summarize_reports_the_counts(campaign):
    out, _ = campaign
    assert cli("summarize", "--out", str(out)).returncode == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    (condition,) = summary["conditions"]
    tags = completed_tags(out)
    assert condition["label"] == LABEL
    assert condition["completed"] == len(tags)
    assert condition["seeds_run"] == len(tags) + condition["failed"]
    assert condition["seed_cap"] == MAX_SEEDS
    assert condition["target_filtered"] == TARGET
    assert condition["filtered"] == sum(
        json.loads(pool_io.record_path(out, tag).read_text(encoding="utf-8"))[
            "verdict"
        ]["passes"]
        for tag in tags
    )
    assert 0.0 <= condition["pass_rate"] <= 1.0
    assert condition["wall_minutes"]["median"] > 0
    assert summary["totals"]["filtered"] == condition["filtered"]
    text = (out / "summary.md").read_text(encoding="utf-8")
    assert LABEL in text and "filtered" in text


def test_changed_artifact_stops_the_sweep(campaign, tmp_path):
    out, _ = campaign
    copy = tmp_path / "tampered"
    shutil.copytree(out, copy)
    tag = completed_tags(copy)[0]
    path = pool_io.record_path(copy, tag)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["hashes"][".npz"] = "0" * 64
    runner.write_json(path, record)
    result = cli("run", "--out", str(copy))
    assert result.returncode != 0
    assert f"{tag}: changed .npz" in result.stderr
    assert f"START {tag}" not in result.stdout


def test_partial_artifact_stops_the_sweep(campaign, tmp_path):
    out, _ = campaign
    copy = tmp_path / "partial"
    shutil.copytree(out, copy)
    # The first seed of the condition, so the sweep reaches it whatever
    # the filters did: its record and sidecar are gone, its .npz is not.
    tag = f"{LABEL}_seed{SEED_START}"
    pool_io.record_path(copy, tag).unlink()
    (copy / f"{tag}.json").unlink()
    result = cli("run", "--out", str(copy))
    assert result.returncode != 0
    assert "incomplete prior attempt" in result.stderr


def test_verify_run_rejects_a_tampered_record(campaign, tmp_path):
    out, _ = campaign
    copy = tmp_path / "hashes"
    shutil.copytree(out, copy)
    tag = completed_tags(copy)[0]
    path = pool_io.record_path(copy, tag)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["hashes"][".json"] = "0" * 64
    runner.write_json(path, record)
    with pytest.raises(RuntimeError, match="recorded hash"):
        pool_io.verify_run(copy / tag)


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


def test_prepare_refuses_a_structural_change(campaign):
    out, _ = campaign
    result = cli(
        "prepare",
        "--out",
        str(out),
        "--suite",
        "smoke",
        "--only",
        LABEL,
        "--target",
        str(TARGET),
        "--max-seeds",
        str(MAX_SEEDS),
        "--seed-start",
        str(SEED_START),
        # everything as prepared, but not the recorded relaxation
    )
    assert result.returncode != 0
    assert "allow_dirty" in result.stderr


def test_revised_history_guards_a_revision(campaign):
    out, _ = campaign
    previous = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    lowered = [
        dict(entry, target_filtered=1) for entry in previous["allocation"]
    ]
    with pytest.raises(RuntimeError, match="--authorized-by"):
        runner.revised_history(out, previous, lowered, None)
    history = runner.revised_history(out, previous, lowered, AUTHORIZED_BY)
    assert history[-1]["allocation"] == previous["allocation"]
    assert history[-1]["authorized_by"] == AUTHORIZED_BY
    assert history[-1]["revised"]
    with pytest.raises(RuntimeError, match="cannot be dropped"):
        runner.revised_history(out, previous, [], AUTHORIZED_BY)
    moved = [
        dict(entry, seed_start=entry["seed_start"] + 1)
        for entry in previous["allocation"]
    ]
    with pytest.raises(RuntimeError, match="first seed"):
        runner.revised_history(out, previous, moved, AUTHORIZED_BY)


def test_revised_allocation_and_pilot_seeds(campaign, tmp_path):
    """A revised target, then a pilot sweep that ignores it.

    The copied campaign already holds its runs, so the sweep reruns
    nothing: the filtered target is lowered to one, and ``--pilot-seeds
    2`` still walks two seeds of the condition.
    """
    out, _ = campaign
    copy = tmp_path / "revised"
    shutil.copytree(out, copy)
    revision = prepare(copy, "--authorized-by", AUTHORIZED_BY, target=1)
    assert revision.returncode == 0, revision.stdout + revision.stderr
    assert "allocation revised" in revision.stdout
    before = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    manifest = json.loads((copy / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["allocation"][0]["target_filtered"] == 1
    assert manifest["allocation_history"][-1] == {
        "allocation": before["allocation"],
        "revised": manifest["allocation_history"][-1]["revised"],
        "authorized_by": AUTHORIZED_BY,
    }
    for key in ("campaign", "suite", "options", "identity", "created"):
        assert manifest[key] == before[key]
    assert manifest["launch_ready"] is True

    result = cli("run", "--out", str(copy), "--pilot-seeds", "2")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "START" not in result.stdout
    finished = json.loads((copy / "finished.json").read_text(encoding="utf-8"))
    condition = finished["conditions"][LABEL]
    assert condition["seeds_run"] == 2
    assert condition["seeds"] == [SEED_START, SEED_START + 1]


def test_a_failed_case_is_skipped_and_counted(tmp_path):
    """A case that failed earlier costs its seed and does not fail a sweep."""
    out = tmp_path / "failures"
    assert (
        prepare(out, "--ready", "--authorized-by", AUTHORIZED_BY).returncode
        == 0
    )
    planted = [f"{LABEL}_seed{SEED_START + i}" for i in range(2)]
    for tag in planted:
        (out / f"{tag}.error.txt").write_text(
            "exit code 1\nplanted failure\n", encoding="utf-8"
        )
    result = cli("run", "--out", str(out), "--pilot-seeds", "2")
    assert result.returncode == 0, result.stdout + result.stderr
    for tag in planted:
        assert f"SKIP {tag} (failed earlier)" in result.stdout
    assert "START" not in result.stdout
    finished = json.loads((out / "finished.json").read_text(encoding="utf-8"))
    assert finished["failed"] == planted
    assert finished["failed_this_sweep"] == []
    condition = finished["conditions"][LABEL]
    assert condition["seeds_run"] == 2 and condition["failed"] == 2

    # A case that fails in this sweep does fail it: an allocation entry
    # naming a target the suite does not define makes its worker exit.
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    manifest["allocation"].append(
        {
            "label": "no_such_target_D2",
            "seed_start": 1,
            "max_seeds": 1,
            "target_filtered": 1,
        }
    )
    runner.write_json(out / "manifest.json", manifest)
    result = cli("run", "--out", str(out), "--pilot-seeds", "1")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "FAILED no_such_target_D2_seed1" in result.stdout
    assert (out / "no_such_target_D2_seed1.error.txt").exists()
    finished = json.loads((out / "finished.json").read_text(encoding="utf-8"))
    assert finished["failed_this_sweep"] == ["no_such_target_D2_seed1"]


def test_prepare_allocates_the_pool_conditions(tmp_path):
    out = tmp_path / "pool"
    result = cli("prepare", "--out", str(out), "--allow-dirty")
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    allocated = [entry["label"] for entry in manifest["allocation"]]
    assert allocated == list(runner.POOL_LABELS)
    assert manifest["suite"] == "svbmc_pool" and EXTENSION not in allocated


def test_the_extension_condition_needs_an_explicit_only(tmp_path):
    out = tmp_path / "extension"
    result = cli(
        "prepare", "--out", str(out), "--only", EXTENSION, "--allow-dirty"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "extension condition" in result.stdout
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert [e["label"] for e in manifest["allocation"]] == [EXTENSION]


def test_check_tree_refuses_an_uncommitted_numerical_source():
    clean = {
        "gpyreg_clean": True,
        "pyvbmc_dirty": "",
        "suite_module_dirty": "",
    }
    assert runner.check_tree(clean, False) == []
    dirty = dict(clean, pyvbmc_dirty=" M pyvbmc/vbmc/vbmc.py")
    with pytest.raises(RuntimeError, match="uncommitted changes"):
        runner.check_tree(dirty, False)
    assert runner.check_tree(dirty, True) == ["pyvbmc/"]
    with pytest.raises(RuntimeError, match="frozen gpyreg"):
        runner.check_tree(dict(clean, gpyreg_clean=False), True)


def test_identity_differences_names_every_mismatch():
    expected = {"pyvbmc_commit": "a", "runner_sha256": "b", "numpy": "2.0"}
    actual = {"pyvbmc_commit": "a", "runner_sha256": "c", "hostname": "h"}
    assert runner.identity_differences(actual, expected) == [
        "hostname",
        "numpy",
        "runner_sha256",
    ]


def test_identity_pins_the_harness_modules():
    record = runner.identity(GPYREG_SOURCE)
    assert record["suite_module_sha256"] == pool_io.sha256(
        HERE / "benchmark_targets.py"
    )
    assert record["io_module_sha256"] == pool_io.sha256(
        HERE / "svbmc_pool_io.py"
    )
    assert record["runner_sha256"] == pool_io.sha256(SCRIPT)


def test_a_stale_log_is_not_a_partial_artifact(tmp_path):
    tag = f"{LABEL}_seed{SEED_START}"
    (tmp_path / f"{tag}.log").write_text("interrupted\n", encoding="utf-8")
    assert runner.case_state(tmp_path, tag, {}) == ("new", None)
    (tmp_path / f"{tag}.npz").write_bytes(b"")
    assert runner.case_state(tmp_path, tag, {}) == (
        "partial",
        [f"{tag}.npz"],
    )


def test_allocation_overrides_and_bounds():
    class Args:
        target, max_seeds, seed_start = 60, 90, 1000
        control_target = control_max_seeds = None
        allocation = ["ring_D2_noise3_svbmc=40/80"]

    from benchmark_targets import suite_configs

    entries = runner.allocation(Args(), suite_configs("svbmc_pool"))
    by_label = {e["label"]: e for e in entries}
    assert by_label["ring_D2_noise3_svbmc"]["target_filtered"] == 40
    assert by_label["ring_D2_noise3_svbmc"]["max_seeds"] == 80
    assert by_label["rosenbrock_D2_noise3_svbmc"]["target_filtered"] == 60
    Args.control_target, Args.control_max_seeds = 30, 45
    entries = runner.allocation(Args(), suite_configs("svbmc_pool"))
    by_label = {e["label"]: e for e in entries}
    assert by_label["gmm_D2_svbmc"]["target_filtered"] == 30
    assert by_label["gmm_D2_svbmc"]["max_seeds"] == 45
    Args.target, Args.max_seeds = 90, 60
    with pytest.raises(RuntimeError, match="exceeds the seed cap"):
        runner.allocation(Args(), suite_configs("svbmc_pool"))


@pytest.mark.parametrize(
    "item", ["ring_D2_noise3_svbmc=40", "=40/80", "ring_D2_noise3_svbmc=x/80"]
)
def test_allocation_rejects_a_malformed_argument(item):
    with pytest.raises(RuntimeError, match="LABEL=TARGET/MAXSEEDS"):
        runner.parse_override(item)
