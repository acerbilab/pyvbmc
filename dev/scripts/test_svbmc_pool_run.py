"""Contracts of the S-VBMC pool generator: artifact, resume, selection,
summary and post-hoc verification.

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
import svbmc_pool_run as runner

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRIPT = HERE / "svbmc_pool_run.py"
#: The campaign's pinned gpyreg worktree, as the harness names it; the
#: runner imports no PyVBMC, so reading it costs nothing.
GPYREG_SOURCE = runner.DEFAULT_GPYREG
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

# The campaign pins gpyreg to a frozen worktree; the artifact module reads
# the variable when it is imported, and the workers inherit it.
if (GPYREG_SOURCE / "gpyreg").is_dir():
    os.environ["PYVBMC_GPYREG_SOURCE"] = str(GPYREG_SOURCE)

import svbmc_pool_io as pool_io  # noqa: E402

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


def prepare(out, target=TARGET):
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
    )


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    """One generated pool directory, shared by every check below."""
    out = tmp_path_factory.mktemp("svbmc_pool")
    assert prepare(out).returncode == 0
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


def test_manifest_records_the_allocation_and_the_identity(campaign):
    out, _ = campaign
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["allocation"] == [
        {
            "label": LABEL,
            "seed_start": SEED_START,
            "max_seeds": MAX_SEEDS,
            "target_filtered": TARGET,
        }
    ]
    assert manifest["options"] == runner.BASE_OPTIONS
    identity = manifest["identity"]
    assert set(identity) == {"source", "host"}
    assert identity["source"]["gpyreg_commit"] == runner.git(
        GPYREG_SOURCE, "rev-parse", "HEAD"
    )
    assert identity["host"]["gpyreg_source"] == str(GPYREG_SOURCE.resolve())
    assert identity["host"]["hostname"] and identity["host"]["executable"]


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
        assert record["identity"]["host"]["gpyreg_import"].startswith(
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
    assert meta["identity"]["host"]["gpyreg_import"].startswith(
        str(GPYREG_SOURCE)
    )
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


def test_manifest_cases_follows_the_allocation_order():
    """Every seed of every condition's range, conditions in manifest order."""
    manifest = {
        "allocation": [
            {"label": "second", "seed_start": 10, "max_seeds": 2},
            {"label": "first", "seed_start": 1000, "max_seeds": 3},
        ]
    }
    assert runner.manifest_cases(manifest) == [
        ("second", 10),
        ("second", 11),
        ("first", 1000),
        ("first", 1001),
        ("first", 1002),
    ]


def test_cases_prints_one_worker_call_per_line(campaign):
    """The array index is the line number; the count goes to stderr."""
    out, _ = campaign
    result = cli("cases", "--out", str(out))
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines() == [
        f"{LABEL} {SEED_START + i}" for i in range(MAX_SEEDS)
    ]
    assert f"{MAX_SEEDS} cases" in result.stderr
    listed = json.loads(
        cli("cases", "--out", str(out), "--format", "json").stdout
    )
    assert listed["count"] == MAX_SEEDS
    assert [
        (case["index"], case["label"], case["seed"])
        for case in listed["cases"]
    ] == [(i + 1, LABEL, SEED_START + i) for i in range(MAX_SEEDS)]


def test_select_takes_the_lowest_seeds_that_pass(campaign, tmp_path):
    out, _ = campaign
    passing = [
        tag
        for tag in completed_tags(out)
        if json.loads(
            pool_io.record_path(out, tag).read_text(encoding="utf-8")
        )["verdict"]["passes"]
    ]
    result = cli("select", "--out", str(out))
    assert result.returncode == 0, result.stdout + result.stderr
    selection = json.loads(
        (out / "selection.json").read_text(encoding="utf-8")
    )
    (condition,) = selection["conditions"]
    assert condition["label"] == LABEL
    assert condition["target_filtered"] == TARGET
    assert [run["tag"] for run in condition["runs"]] == passing[:TARGET]
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
    markdown = (out / "selection.md").read_text(encoding="utf-8")
    assert markdown.startswith("# S-VBMC filtered pool")
    assert "pass rate over the scanned seeds" in markdown
    assert "stops as soon as the target is met" in markdown

    copy = tmp_path / "one"
    shutil.copytree(out, copy)
    assert cli("select", "--out", str(copy), "--target", "1").returncode == 0
    lowered = json.loads((copy / "selection.json").read_text(encoding="utf-8"))
    assert lowered["target_override"] == 1
    assert [r["tag"] for r in lowered["conditions"][0]["runs"]] == passing[:1]


def test_the_stack_harness_reads_the_selection(campaign, tmp_path):
    """The comparison's pool reader prefers `selection.json` to the records.

    The check lives in this module because this is where a pool directory
    is generated; the rest of the comparison harness is exercised by
    `test_svbmc_pool_stack.py`.
    """
    import svbmc_pool_stack as stack

    out, _ = campaign
    copy = tmp_path / "selected"
    shutil.copytree(out, copy)
    (copy / "selection.json").unlink(missing_ok=True)
    conditions, identities, labels = stack.pool_conditions([copy])
    assert labels == [LABEL]
    every = [entry["name"] for entry in conditions[LABEL]]
    assert identities[0]["selection"] == {
        "path": None,
        "generated": None,
        "conditions": {LABEL: "every passing record"},
    }
    assert cli("select", "--out", str(copy), "--target", "1").returncode == 0
    conditions, identities, _ = stack.pool_conditions([copy])
    assert [entry["name"] for entry in conditions[LABEL]] == every[:1]
    assert identities[0]["selection"]["path"] == str(copy / "selection.json")
    assert identities[0]["selection"]["conditions"] == {
        LABEL: "selection.json"
    }

    # A selection whose runs the directory no longer holds names the
    # entry that is wrong, not only the file that is absent.
    (copy / f"{every[0]}.npz").unlink()
    with pytest.raises(RuntimeError, match=r"selection\.json"):
        stack.pool_conditions([copy])
    pool_io.record_path(copy, every[0]).unlink()
    with pytest.raises(RuntimeError, match="no completion record"):
        stack.pool_conditions([copy])


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
    copy = tmp_path / "revised"
    shutil.copytree(out, copy)
    revision = prepare(copy, target=1)
    assert revision.returncode == 0, revision.stdout + revision.stderr
    assert "allocation revised" in revision.stdout
    before = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    manifest = json.loads((copy / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["allocation"][0]["target_filtered"] == 1
    assert manifest["allocation_history"][-1] == {
        "allocation": before["allocation"],
        "revised": manifest["allocation_history"][-1]["revised"],
    }
    for key in ("campaign", "suite", "options", "identity", "created"):
        assert manifest[key] == before[key]

    result = cli("run", "--out", str(copy), "--pilot-seeds", "2")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "START" not in result.stdout
    finished = json.loads((copy / "finished.json").read_text(encoding="utf-8"))
    condition = finished["conditions"][LABEL]
    assert condition["seeds_run"] == 2
    assert condition["seeds"] == [SEED_START, SEED_START + 1]


def bogus_case(out, label="no_such_target_D2", seed=1):
    """Allocate a condition whose target the suite does not define."""
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    manifest["allocation"].append(
        {
            "label": label,
            "seed_start": seed,
            "max_seeds": 1,
            "target_filtered": 1,
        }
    )
    runner.write_json(out / "manifest.json", manifest)
    return label, f"{label}_seed{seed}"


def test_a_failed_case_is_skipped_and_counted(tmp_path):
    """A case that failed earlier costs its seed and does not fail a sweep."""
    out = tmp_path / "failures"
    assert prepare(out).returncode == 0
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
    bogus_case(out)
    result = cli("run", "--out", str(out), "--pilot-seeds", "1")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "FAILED no_such_target_D2_seed1" in result.stdout
    # The worker wrote the record of its own failure; the supervisor kept
    # it rather than replacing it with the exit code and the log tail.
    error = (out / "no_such_target_D2_seed1.error.txt").read_text(
        encoding="utf-8"
    )
    assert error.startswith("no_such_target_D2_seed1: ValueError")
    finished = json.loads((out / "finished.json").read_text(encoding="utf-8"))
    assert finished["failed_this_sweep"] == ["no_such_target_D2_seed1"]


def test_a_failing_worker_records_the_case_and_exits_non_zero(tmp_path):
    """`<tag>.error.txt` is the whole of what a failed array task leaves.

    The worker itself writes it, deletes the artifact files an earlier
    attempt had begun so that no truncated run can be read as one, writes
    no completion record, and exits non-zero, which is how Slurm sees the
    failure. `select` and `summarize` then count the case as failed.
    """
    out = tmp_path / "array"
    assert prepare(out).returncode == 0
    label, tag = bogus_case(out)
    for suffix in runner.ARTIFACT_SUFFIXES:
        (out / f"{tag}{suffix}").write_text("partial", encoding="utf-8")

    result = cli("worker", "--out", str(out), "--label", label, "--seed", "1")
    assert result.returncode != 0
    error = (out / f"{tag}.error.txt").read_text(encoding="utf-8")
    assert error.startswith(f"{tag}: ValueError")
    assert label in error and "Traceback" in error
    assert not pool_io.record_path(out, tag).exists()
    assert runner.partial_artifacts(out, tag) == []

    assert cli("select", "--out", str(out)).returncode == 0
    selection = json.loads(
        (out / "selection.json").read_text(encoding="utf-8")
    )
    (failed,) = [c for c in selection["conditions"] if c["label"] == label]
    assert failed["failed_while_scanning"] == 1
    assert failed["selected"] == 0 and failed["shortfall"] == 1

    assert cli("summarize", "--out", str(out)).returncode == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    (counted,) = [c for c in summary["conditions"] if c["label"] == label]
    assert counted["failed"] == 1 and counted["completed"] == 0
    assert counted["failures"][0]["tag"] == tag
    assert "ValueError" in counted["failures"][0]["reason"]


def test_a_successful_worker_clears_a_stale_error_file(tmp_path, monkeypatch):
    """A rerun that works leaves no trace of the attempt that did not.

    The case itself is stubbed out: what is under test is the bookkeeping
    around it, which every generated case of the module shares.
    """
    tag = f"{LABEL}_seed{SEED_START}"
    (tmp_path / f"{tag}.error.txt").write_text(
        "an attempt\n", encoding="utf-8"
    )
    runner.write_json(tmp_path / "manifest.json", {})
    monkeypatch.setattr(runner, "worker_case", lambda *args: None)
    assert (
        runner.main(
            [
                "worker",
                "--out",
                str(tmp_path),
                "--label",
                LABEL,
                "--seed",
                str(SEED_START),
            ]
        )
        == 0
    )
    assert not (tmp_path / f"{tag}.error.txt").exists()


def test_a_manifest_with_extra_keys_is_read(campaign, tmp_path):
    """Every command reads a manifest by the keys it needs.

    Campaigns prepared by earlier versions of this harness carry fields
    this one does not write, and the pools they generated are read here.
    """
    out, _ = campaign
    copy = tmp_path / "extra_keys"
    shutil.copytree(out, copy)
    manifest = json.loads((copy / "manifest.json").read_text(encoding="utf-8"))
    manifest.update({"released": True, "released_by": "an earlier harness"})
    runner.write_json(copy / "manifest.json", manifest)
    assert cli("cases", "--out", str(copy)).returncode == 0
    assert cli("select", "--out", str(copy)).returncode == 0
    assert cli("summarize", "--out", str(copy)).returncode == 0


def test_prepare_allocates_the_whole_pool_suite(tmp_path):
    """The campaign's approved allocation, with no flag but `--out`.

    Without `--only` the allocation is the eight pool conditions, and
    without allocation flags it is the sizes the PI approved: 100 filtered
    runs per noisy condition and 50 per noiseless control, seed caps 150,
    200 for the ring and 75 for the controls.
    """
    out = tmp_path / "pool"
    result = cli("prepare", "--out", str(out), "--allow-dirty")
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
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
    result = cli(
        "prepare",
        "--out",
        str(out),
        "--only",
        ",".join(wanted),
        "--allow-dirty",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert [e["label"] for e in manifest["allocation"]] == wanted


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
    source = runner.identity(GPYREG_SOURCE)["source"]
    assert set(source) == set(runner.SOURCE_KEYS)
    assert source["suite_module_sha256"] == pool_io.sha256(
        HERE / "benchmark_targets.py"
    )
    assert source["io_module_sha256"] == pool_io.sha256(
        HERE / "svbmc_pool_io.py"
    )
    assert source["runner_sha256"] == pool_io.sha256(SCRIPT)


def test_identity_source_reads_the_flat_records_of_earlier_campaigns():
    """A record written before the split holds both halves in one mapping."""
    record = runner.identity(GPYREG_SOURCE)
    flat = dict(record["source"], **record["host"])
    assert runner.identity_source(flat) == record["source"]
    assert runner.identity_host(flat) == record["host"]


def test_only_the_source_half_of_a_record_is_compared(campaign, tmp_path):
    """Another node may have run a case; other code may not have.

    The host half of a completion record names where the case ran, which
    an array job spreads over the cluster, so it is recorded and not
    compared; the source half is the code and library state the pool is
    generated by, and a difference there stops the campaign.
    """
    out, _ = campaign
    copy = tmp_path / "elsewhere"
    shutil.copytree(out, copy)
    tag = completed_tags(copy)[0]
    path = pool_io.record_path(copy, tag)
    record = json.loads(path.read_text(encoding="utf-8"))
    expected = runner.identity(GPYREG_SOURCE)
    record["identity"]["host"].update(
        {
            "hostname": "cluster-node-07",
            "executable": "/scratch/venv/bin/python",
            "platform": "Linux-5.14.0-x86_64-with-glibc2.34",
        }
    )
    runner.write_json(path, record)
    assert runner.validate_case(copy, tag, expected)["tag"] == tag

    record["identity"]["source"]["suite_module_sha256"] = "0" * 64
    runner.write_json(path, record)
    with pytest.raises(RuntimeError, match="suite_module_sha256"):
        runner.validate_case(copy, tag, expected)


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


def verification(out):
    """The report `verify` wrote in a campaign directory."""
    return json.loads((out / "verification.json").read_text(encoding="utf-8"))


def reported(report, tag):
    """The one reported case of a tag."""
    (case,) = [c for c in report["cases"] if c["tag"] == tag]
    return case


def test_verify_reconciles_the_allocation(campaign, tmp_path):
    """Every case of the allocation is placed, and the counts add up."""
    out, _ = campaign
    copy = tmp_path / "pool"
    shutil.copytree(out, copy)
    result = cli("verify", "--out", str(copy))
    assert result.returncode == 0, result.stdout + result.stderr
    report = verification(copy)
    assert [case["index"] for case in report["cases"]] == list(
        range(1, MAX_SEEDS + 1)
    )
    assert [case["seed"] for case in report["cases"]] == [
        SEED_START + i for i in range(MAX_SEEDS)
    ]
    counts = report["counts"]
    assert counts["verified"] == len(completed_tags(copy))
    assert counts["failed"] == len(list(copy.glob("*.error.txt")))
    # The sweep stops at the filtered target or walks every seed; either
    # way the allocation's cases are each placed exactly once.
    assert (
        counts["verified"]
        + counts["failed"]
        + counts["partial"]
        + counts["missing"]
        + counts["verify_failed"]
        == MAX_SEEDS
    )
    assert counts["stray"] == 0 and counts["partial"] == 0
    for case in report["cases"]:
        if case["status"] == "verified":
            assert isinstance(case["passes"], bool)
            assert case["differences"]["I_sk"] <= pool_io.TOL_STATS
            assert case["differences"]["J_sjk"] <= pool_io.TOL_STATS
    (condition,) = report["conditions"]
    assert condition["label"] == LABEL
    assert [condition[key] for key in runner.VERIFY_STATUSES] == [
        counts[key] for key in runner.VERIFY_STATUSES
    ]


def test_verify_reports_a_tampered_artifact(campaign, tmp_path):
    """A flipped byte in a stored `.npz` fails that case and the command."""
    out, _ = campaign
    copy = tmp_path / "pool"
    shutil.copytree(out, copy)
    tag = completed_tags(copy)[0]
    artifact = copy / f"{tag}.npz"
    stored = bytearray(artifact.read_bytes())
    stored[10] ^= 0xFF
    artifact.write_bytes(bytes(stored))
    result = cli("verify", "--out", str(copy))
    assert result.returncode == 1, result.stdout + result.stderr
    case = reported(verification(copy), tag)
    assert case["status"] == "verify_failed" and case["error"]
    assert f"{tag} verify_failed" in result.stdout


def test_verify_distinguishes_partial_from_missing(campaign, tmp_path):
    """An artifact without its record is partial; nothing at all is missing.

    The second is what a task Slurm killed leaves, which `select` and
    `summarize` cannot see, so `verify` names it with its array index.
    """
    out, _ = campaign
    copy = tmp_path / "pool"
    shutil.copytree(out, copy)
    result = cli("verify", "--out", str(copy))
    assert result.returncode == 0, result.stdout + result.stderr
    before = verification(copy)["counts"]["missing"]
    tag = completed_tags(copy)[0]
    pool_io.record_path(copy, tag).unlink()
    result = cli("verify", "--out", str(copy))
    assert result.returncode == 1, result.stdout + result.stderr
    case = reported(verification(copy), tag)
    assert case["status"] == "partial"
    assert case["files"] == [f"{tag}.npz", f"{tag}.json"]

    gone = tmp_path / "pool2"
    shutil.copytree(out, gone)
    pool_io.record_path(gone, tag).unlink()
    for suffix in (".npz", ".json", ".vbmc.pkl", ".log", ".error.txt"):
        (gone / f"{tag}{suffix}").unlink(missing_ok=True)
    result = cli("verify", "--out", str(gone))
    assert result.returncode == 0, result.stdout + result.stderr
    report = verification(gone)
    case = reported(report, tag)
    assert case["status"] == "missing"
    assert report["counts"]["missing"] == before + 1
    assert f"{case['index']} {tag} missing" in result.stdout


def test_verify_flags_a_stray_artifact(campaign, tmp_path):
    """An artifact the allocation does not name is stray; nothing else is."""
    out, _ = campaign
    copy = tmp_path / "pool"
    shutil.copytree(out, copy)
    (copy / "foo_seed1.npz").write_bytes(b"not an artifact")
    # The campaign's own files and the array's directory are not runs.
    (copy / "slurm").mkdir(exist_ok=True)
    (copy / "slurm" / "sacct.txt").write_text(
        "JobID|State|ExitCode\n", encoding="utf-8"
    )
    (copy / "cases.txt").write_text(
        f"{LABEL} {SEED_START}\n", encoding="utf-8"
    )
    result = cli("verify", "--out", str(copy))
    assert result.returncode == 1, result.stdout + result.stderr
    report = verification(copy)
    assert report["stray"] == ["foo_seed1"]
    assert report["counts"]["stray"] == 1
    assert "stray: 1 (foo_seed1)" in result.stdout
    assert "sacct" not in result.stdout and "cases.txt" not in result.stdout


def test_verify_refuses_a_gpyreg_source_at_another_commit(campaign, tmp_path):
    """A copied pool is verified against the pinned library alone."""
    out, _ = campaign
    copy = tmp_path / "pool"
    shutil.copytree(out, copy)
    (copy / "verification.json").unlink(missing_ok=True)
    other = tmp_path / "other_gpyreg"
    (other / "gpyreg").mkdir(parents=True)
    subprocess.run(["git", "-C", str(other), "init", "-q"], check=True)
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
    subprocess.run(["git", "-C", str(other), *author, "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(other), *author, "commit", "-q", "-m", "x"],
        check=True,
    )
    result = cli("verify", "--out", str(copy), "--gpyreg-source", str(other))
    assert result.returncode != 0
    manifest = json.loads((copy / "manifest.json").read_text(encoding="utf-8"))
    pinned = manifest["identity"]["source"]["gpyreg_commit"]
    # The refusal names the manifest's commit, and nothing was verified.
    assert f"not the manifest's {pinned}" in result.stderr
    assert not (copy / "verification.json").exists()
    # A directory that is no git checkout is refused with a plain message.
    plain = tmp_path / "plain"
    (plain / "gpyreg").mkdir(parents=True)
    result = cli("verify", "--out", str(copy), "--gpyreg-source", str(plain))
    assert result.returncode != 0
    assert "is not a git checkout" in result.stderr
    assert not (copy / "verification.json").exists()
