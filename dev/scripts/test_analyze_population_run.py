"""Checks of the population assessment (run by explicit path).

The statistics are checked against SciPy and an independent count; the
comparison of two arms of array mode runs on campaign directories that the
fixtures here write (sidecars, boost reports, completion records,
verification reports, rescored metrics and the rescoring's record), which
test the reader and the statistics, not VBMC; ``test_population_run.py``
runs the comparison on arms that the harness itself verified and
rescored. Two such arms, run at a stand-in site
(``campaign_slurm_stubs.FakeSite``), are compared as they are and as their
redacted tracked copies.
"""

import copy
import hashlib
import json
from fractions import Fraction

import analyze_population_run as analysis
import campaign_slurm_stubs as stubs
import numpy as np
import pytest
from scipy import stats

runner = analysis.runner
contract = runner.contract


def _permutation(delta):
    nonzero = delta[delta != 0]
    return stats.wilcoxon(
        nonzero,
        alternative="two-sided",
        method=stats.PermutationMethod(n_resamples=np.inf),
    )


@pytest.mark.parametrize("trial", range(30))
def test_exact_signed_rank_matches_scipy_enumeration_with_ties(trial):
    rng = np.random.default_rng(trial)
    size = rng.integers(3, 17)
    # Small integers force ties among absolute differences and zeros.
    delta = rng.integers(-4, 5, size).astype(float)
    if (delta != 0).sum() < 2:
        pytest.skip("fewer than two nonzero differences")
    statistic, pvalue = analysis.exact_signed_rank(delta)
    expected = _permutation(delta)
    assert statistic == expected.statistic
    assert pvalue == expected.pvalue


def test_exact_signed_rank_matches_scipy_exact_beyond_enumeration():
    delta = np.random.default_rng(7).normal(size=30)
    statistic, pvalue = analysis.exact_signed_rank(delta)
    expected = stats.wilcoxon(delta, alternative="two-sided", method="exact")
    assert statistic == expected.statistic
    assert pvalue == pytest.approx(expected.pvalue, rel=1e-12)


def test_exact_signed_rank_degenerate_and_bounds():
    assert analysis.exact_signed_rank([0.0, 0.0, 2.0]) == (0.0, 1.0)
    statistic, pvalue = analysis.exact_signed_rank([1.0, 2.0])
    assert (statistic, pvalue) == (0.0, 0.5)
    statistic, pvalue = analysis.exact_signed_rank(np.arange(1, 9.0))
    assert statistic == 0.0
    assert pvalue == 2 / 2**8
    # Beyond the int64 counts, the counts are Python integers.
    statistic, pvalue = analysis.exact_signed_rank(np.arange(1, 64.0))
    assert statistic == 0.0
    assert pvalue == 2 / 2**63
    statistic, pvalue = analysis.exact_signed_rank(-np.arange(1, 101.0))
    assert statistic == 0.0
    assert pvalue == 2 / 2**100
    with pytest.raises(ValueError):
        analysis.exact_signed_rank([1.0, np.nan])


def independent_signed_rank_pvalue(delta):
    """The two-sided exact p-value, counted with a dictionary and fractions."""
    d = [x for x in delta if x != 0]
    ranks = stats.rankdata(np.abs(d))
    twice = [int(round(2 * r)) for r in ranks]
    ways = {0: 1}
    for unit in twice:
        grown = dict(ways)
        for total, count in ways.items():
            grown[total + unit] = grown.get(total + unit, 0) + count
        ways = grown
    positive = sum(unit for unit, x in zip(twice, d) if x > 0)
    everything = 2 ** len(d)
    less = Fraction(
        sum(c for s, c in ways.items() if s <= positive), everything
    )
    more = Fraction(
        sum(c for s, c in ways.items() if s >= positive), everything
    )
    return float(min(1, 2 * min(less, more)))


def test_exact_signed_rank_at_100_pairs_matches_scipy_exact():
    delta = np.random.default_rng(11).normal(0.2, 1.0, size=100)
    statistic, pvalue = analysis.exact_signed_rank(delta)
    expected = stats.wilcoxon(delta, alternative="two-sided", method="exact")
    assert statistic == expected.statistic
    assert pvalue == pytest.approx(expected.pvalue, rel=1e-9)
    assert pvalue == independent_signed_rank_pvalue(delta)


@pytest.mark.parametrize("trial", range(4))
def test_exact_signed_rank_at_100_pairs_with_ties_is_exact(trial):
    rng = np.random.default_rng(100 + trial)
    delta = rng.integers(-6, 7, 100).astype(float) + (trial - 1.5) / 4
    delta[rng.integers(0, 100, 5)] = 0.0
    statistic, pvalue = analysis.exact_signed_rank(delta)
    assert pvalue == independent_signed_rank_pvalue(delta)
    nonzero = delta[delta != 0]
    ranks = stats.rankdata(np.abs(nonzero))
    assert statistic == min(ranks[nonzero > 0].sum(), ranks[nonzero < 0].sum())


def test_paired_tests_family_and_holm():
    rows = []
    for label, deltas in (("a", [1, 2, 3, 4, 5, 6]), ("b", [1, -1, 2, -2])):
        for seed, d in enumerate(deltas):
            rows.append(
                {
                    "label": label,
                    "delta": {k: float(d) for k in analysis.QUALITY},
                    "old_usable": d > 0,
                    "new_usable": False,
                }
            )
    tests = analysis.paired_tests(rows, metrics=analysis.QUALITY)
    assert len(tests) == 8
    by_key = {(t["label"], t["metric"]): t for t in tests}
    assert by_key[("a", "gskl")]["pvalue"] == 2 / 2**6
    assert by_key[("a", "usable")]["losses"] == 6
    assert by_key[("a", "usable")]["pvalue"] == 2 / 2**6
    assert by_key[("b", "gskl")]["pvalue"] == 1.0
    assert all("holm_adjusted_pvalue" in t for t in tests)
    assert not any(t["holm_rejected"] for t in tests)
    json.dumps(tests)  # plain Python floats and bools only
    plain = analysis.paired_tests(rows, metrics=analysis.QUALITY, adjust=False)
    assert not any("holm_rejected" in t for t in plain)


def test_merge_populations_concatenates_labels():
    first = {"x": {"seeds": [0], "rows": [{"elbo_err": 1.0}], "fails": 0}}
    second = {
        "x": {"seeds": [1], "rows": [{"elbo_err": 2.0}], "fails": 1},
        "y": {"seeds": [0], "rows": [{"elbo_err": 3.0}], "fails": 0},
    }
    merged = analysis.merge_populations([first, second])
    assert merged["x"]["seeds"] == [0, 1]
    assert merged["x"]["fails"] == 1
    np.testing.assert_array_equal(merged["x"]["elbo_err"], [1.0, 2.0])
    assert np.isnan(merged["y"]["gskl"]).all()


def test_paired_tests_without_usability_and_other_alpha():
    rows = [
        {
            "label": "a",
            "delta": {"gskl": -float(d)},
            "old_usable": True,
            "new_usable": True,
        }
        for d in range(1, 9)
    ]
    tests = analysis.paired_tests(
        rows, metrics=("gskl",), usability=False, alpha=0.001
    )
    assert [t["metric"] for t in tests] == ["gskl"]
    assert tests[0]["pvalue"] == 2 / 2**8
    assert not tests[0]["holm_rejected"]
    assert analysis.paired_tests(rows, metrics=("gskl",), usability=False)[0][
        "holm_rejected"
    ]


# --------------------------------------------------------------------------
# Two arms of array mode, on campaign directories written here
# --------------------------------------------------------------------------

LABELS = ("normal_D2", "banana_D2")
SEEDS = list(range(100))
HARNESS = "a" * 40


def source(pyvbmc_commit, files=None):
    return {
        "trees": {
            "harness": {"commit": HARNESS, "clean": True},
            "pyvbmc": {"commit": pyvbmc_commit, "clean": True},
            "gpyreg": {"commit": pyvbmc_commit[:1] * 40, "clean": True},
        },
        "files": files or {"dev/scripts/benchmark_targets.py": "1" * 64},
        "versions": {"python": "3.12.6", "numpy": "2.5.2"},
    }


def base_metrics(label, seed):
    """The accuracy of one seed's run, which both arms start from."""
    rng = np.random.default_rng([LABELS.index(label), seed])
    return {
        "elbo_err": float(rng.uniform(0.0, 0.6)),
        "gskl": float(rng.lognormal(np.log(0.1), 0.8)),
        "mmtv": float(rng.uniform(0.03, 0.15)),
        "rmse": float(rng.uniform(0.0, 0.2)),
    }


def arm_metrics(role, label, seed):
    """In-run and rescored metrics of one case of one arm.

    The candidate halves gsKL on the first configuration and moves the
    second by symmetric noise. The reference's rescored MMTV differs from
    its in-run one, as the metrics of different code do.
    """
    base = base_metrics(label, seed)
    if role == "candidate":
        noise = np.random.default_rng([7, LABELS.index(label), seed])
        base = {
            k: v * float(np.exp(0.05 * noise.standard_normal()))
            for k, v in base.items()
        }
        if label == LABELS[0]:
            base["gskl"] *= 0.5
    rescored = dict(base)
    if role == "reference":
        rescored["mmtv"] += 1e-6
    return base, rescored


def sited(site, identity, node=None):
    """``identity`` with the paths and host part of ``site``, the stand-in
    site of ``campaign_slurm_stubs.FakeSite``: of its login node, or with
    ``node`` of an array task there."""
    identity = copy.deepcopy(identity)
    identity["imports"] = {
        "trees": {
            name: {"path": "", "dirty": []}
            for name in ("harness", "pyvbmc", "gpyreg")
        },
        "modules": {"pyvbmc": "", "gpyreg": ""},
    }
    return site.plant(identity, node=node)


def write_arm(root, role, pyvbmc_commit, files=None, site=None):
    """One arm of array mode, its cases verified.

    With ``site`` the arm was run there: its manifest holds the site block,
    the tracked copies and the login node's identity, each record the
    identity of the node its case ran on, each sidecar its provenance, and
    the directory the task logs, the accounting and a summary.
    """
    path = root / role
    manifest = {
        "harness": "population_run",
        "arm": role,
        "allocation": {
            "suite": "production",
            "labels": list(LABELS),
            "seeds": SEEDS,
        },
        "options": {"tol_elcbo_boost": 0.1},
        "confirmatory": runner.confirmatory_family(None, LABELS),
        "identity": {"source": source(pyvbmc_commit, files), "imports": {}},
    }
    if site is not None:
        manifest["identity"] = sited(site, manifest["identity"])
        manifest.update(
            site=site.site_block("dev/scripts/population_run.py"),
            pip_freeze=[f"gpyreg @ file://{site.gpyreg.as_posix()}"],
            tracked_copies=runner.tracked_copies(role == "candidate"),
        )
    runner.write_json(path / "manifest.json", manifest)
    cases, rescored = [], {}
    for index, line in enumerate(runner.case_lines(manifest), start=1):
        tag, label, seed = runner.parse_case(line)
        stem = f"{label}_seed{seed}"
        files_of = runner.case_files(label, seed)
        in_run, again = arm_metrics(role, label, seed)
        final = {
            **in_run,
            "elbo": -2.0 + in_run["elbo_err"],
            "elbo_sd": 0.01,
            "func_count": 200 + 5 * (seed % 7),
            "success_flag": seed % 10 != 3,
            "wall_s": 30.0 + seed,
            "peak_rss_mb": 200.0,
            "iterations": 20,
            "final_K": 50,
            "n_warps": 0,
        }
        identity = manifest["identity"]
        side = {"label": label, "seed": seed, "final": final}
        if site is not None:
            identity = sited(site, identity, site.nodes[seed % 2])
            side["provenance"] = {
                "source": identity["source"],
                "imports": identity["imports"],
            }
        side_path = path / files_of["sidecar"]
        side_path.parent.mkdir(parents=True, exist_ok=True)
        side_path.write_text(json.dumps(side))
        quality = {k: in_run[k] for k in analysis.QUALITY}
        report = {
            "attempted": True,
            "accepted": True,
            "tolerance": 0.1,
            "scores": {
                "pre": {"elbo": final["elbo"] - 0.05, "elbo_sd": 0.01},
                "candidate": {"elbo": final["elbo"], "elbo_sd": 0.01},
                "returned": {"elbo": final["elbo"], "elbo_sd": 0.01},
            },
            "metrics": {
                stage: dict(quality)
                for stage in ("pre", "candidate", "returned")
            },
        }
        report_path = path / files_of["boost_report"]
        runner.write_json(report_path, report)
        artifacts = {
            files_of["trace"]: {"sha256": fake_sha256(role, "trace", stem)},
            files_of["posterior"]: {
                "sha256": fake_sha256(role, "posterior", stem)
            },
            files_of["sidecar"]: {"sha256": runner.sha256(side_path)},
            files_of["boost_report"]: {"sha256": runner.sha256(report_path)},
        }
        record_path = contract.record_path(path, tag)
        runner.write_json(
            record_path,
            {"tag": tag, "identity": identity, "artifacts": artifacts},
        )
        cases.append(
            {
                "index": index,
                "tag": tag,
                "case": line,
                "status": "verified",
                "record_sha256": runner.sha256(record_path),
            }
        )
        rescored[stem] = {
            "status": "rescored",
            "metrics": again,
            "equal_to_in_run": {k: again[k] == in_run[k] for k in again},
            "artifacts": {
                files_of[key]: artifacts[files_of[key]]["sha256"]
                for key in ("trace", "posterior", "sidecar")
            },
        }
    counts = {status: 0 for status in contract.STATUSES}
    counts.update(verified=len(cases), stray=0)
    runner.write_json(
        path / "verification.json",
        {
            "harness": "population_run",
            "manifest_sha256": runner.sha256(path / "manifest.json"),
            "counts": counts,
            "cases": cases,
            "stray": [],
            "exit_code": 0,
        },
    )
    if site is not None:
        site.write_slurm(path)
        (path / "summary.md").write_text(
            f"# {role}\n\nPrepared on {site.login} in {path}.\n",
            encoding="utf-8",
        )
    return path, rescored


def write_rescoring(candidate, arms, site=None, rescoring_source=None):
    """The rescored metrics of ``arms`` (``{arm directory: cases}``) and the
    rescoring's record, in the candidate, as ``population_run.py rescore``
    writes them: by a process of the candidate's source identity unless
    ``rescoring_source`` names another."""
    identity = {"source": rescoring_source or source(HARNESS)}
    if site is not None:
        identity = sited(site, identity)
    rescored = {}
    for arm, cases in arms.items():
        rel = f"rescored/{arm.name}.json"
        runner.write_json(
            candidate / rel,
            {
                "campaign": {
                    "name": arm.name,
                    "path": str(arm),
                    "manifest_sha256": runner.sha256(arm / "manifest.json"),
                    "verification_sha256": runner.sha256(
                        arm / "verification.json"
                    ),
                },
                "rescoring": {"identity": identity},
                "cases": cases,
            },
        )
        rescored[arm.name] = {
            "file": rel,
            "sha256": runner.sha256(candidate / rel),
            "path": str(arm),
        }
    runner.write_json(
        candidate / runner.RESCORING,
        {"identity": identity, "rescored": rescored},
    )


def fake_sha256(*words):
    """The SHA-256 a record gives a file that the comparison never reads."""
    return hashlib.sha256(" ".join(words).encode()).hexdigest()


@pytest.fixture
def arms(tmp_path):
    reference, before = write_arm(tmp_path, "reference", "b" * 40)
    candidate, after = write_arm(tmp_path, "candidate", HARNESS)
    write_rescoring(candidate, {reference: before, candidate: after})
    return reference, candidate


def test_analyze_arms_on_100_paired_seeds(arms, tmp_path):
    reference, candidate = arms
    result = analysis.analyze_arms(
        reference, candidate, None, tmp_path / "report"
    )
    assert result["paired_cases"] == 200
    assert result["arms"]["reference"]["verified"] == 200
    # The reference arm's in-run MMTV is not its rescored one.
    assert result["arms"]["reference"]["rescored_equal_to_in_run"] == {
        "elbo_err": 200,
        "gskl": 200,
        "mmtv": 0,
        "rmse": 200,
    }
    confirmatory = {
        (t["label"], t["metric"]): t for t in result["confirmatory_tests"]
    }
    assert len(confirmatory) == result["confirmatory_family"]["tests"] == 8
    assert all(
        t["computed"] and t["planned_pairs"] == 100
        for t in confirmatory.values()
    )
    assert result["confirmatory_not_computed"] == []
    halved = confirmatory[(LABELS[0], "gskl")]
    assert halved["n_pairs"] == 100 and halved["improved"] == 100
    assert halved["nonfinite_pairs"] == 0
    assert halved["pvalue"] == 2 / 2**100 and halved["holm_rejected"]
    assert halved["holm_adjusted_pvalue"] == 8 * 2 / 2**100
    assert not any(
        t["holm_rejected"]
        for (label, _), t in confirmatory.items()
        if label == LABELS[1]
    )
    # The paired changes use the rescored metrics of both arms.
    change = next(
        c for c in result["paired_changes"] if c["tag"] == f"{LABELS[1]}_seed4"
    )
    _, old = arm_metrics("reference", LABELS[1], 4)
    assert change["old"]["mmtv"] == old["mmtv"]
    assert len(result["paired_tests"]) == 2 * 5
    assert len(result["ks_tests"]) == 2 * 4
    assert result["flagged_configurations"] == [LABELS[0]]
    for role in ("reference", "candidate"):
        summary = result["boost_summary"][role]
        assert summary["attempted"] == summary["accepted"] == 200
    # The boost stages are scored by each arm's own code, and say so.
    assert "in-run" in result["boost_usability_basis"]
    written = json.loads((tmp_path / "report" / "assessment.json").read_text())
    assert written["paired_cases"] == 200
    comparison = (tmp_path / "report" / "comparison.md").read_text()
    assert "rescored metrics" in comparison and LABELS[0] in comparison


def test_arms_refuse_a_file_changed_after_verification(arms, tmp_path):
    reference, candidate = arms
    path = reference / runner.case_files(LABELS[1], 9)["sidecar"]
    path.write_text(
        path.read_text().replace('"iterations": 20', '"iterations": 21')
    )
    with pytest.raises(AssertionError, match="not the file"):
        analysis.analyze_arms(reference, candidate, None, tmp_path / "r")


def test_arms_refuse_a_failed_or_rewritten_verification(arms, tmp_path):
    reference, candidate = arms
    path = reference / "verification.json"
    report = json.loads(path.read_text())
    report["cases"][3]["status"] = "verify_failed"
    report["exit_code"] = 1
    runner.write_json(path, report)
    with pytest.raises(contract.ContractError, match="failed its verif"):
        analysis.analyze_arms(reference, candidate, None, tmp_path / "r")
    report["cases"][3]["status"] = "verified"
    report["exit_code"] = 0
    runner.write_json(path, report | {"note": "verified again"})
    with pytest.raises(AssertionError, match="another verification"):
        analysis.analyze_arms(reference, candidate, None, tmp_path / "r")


def test_arms_refuse_other_harness_files(tmp_path):
    reference, before = write_arm(
        tmp_path, "reference", "b" * 40, files={"other": "2" * 64}
    )
    candidate, after = write_arm(tmp_path, "candidate", HARNESS)
    write_rescoring(candidate, {reference: before, candidate: after})
    with pytest.raises(AssertionError, match="harness files"):
        analysis.analyze_arms(reference, candidate, None, tmp_path / "r")


def test_arms_refuse_the_same_code(tmp_path):
    reference, before = write_arm(tmp_path, "reference", HARNESS)
    candidate, after = write_arm(tmp_path, "candidate", HARNESS)
    write_rescoring(candidate, {reference: before, candidate: after})
    with pytest.raises(AssertionError, match="the code with itself"):
        analysis.analyze_arms(reference, candidate, None, tmp_path / "r")


def rescored_cases(candidate, arm):
    """The rescored cases of ``arm`` that the candidate holds."""
    path = candidate / "rescored" / f"{arm.name}.json"
    return json.loads(path.read_text())["cases"]


def test_arms_refuse_a_verification_older_than_the_directory(arms, tmp_path):
    # A case the report places as missing has a completion record now.
    reference, candidate = arms
    path = reference / "verification.json"
    report = json.loads(path.read_text())
    case = report["cases"][5]
    case["status"] = "missing"
    del case["record_sha256"]
    report["counts"].update(verified=199, missing=1)
    runner.write_json(path, report)
    with pytest.raises(contract.ContractError, match="is older than"):
        analysis.analyze_arms(reference, candidate, None, tmp_path / "r")


def test_arms_refuse_rescored_metrics_their_rescoring_did_not_write(
    arms, tmp_path
):
    reference, candidate = arms
    path = candidate / "rescored" / "reference.json"
    rescored = json.loads(path.read_text())
    rescored["cases"][f"{LABELS[0]}_seed0"]["metrics"]["gskl"] *= 2
    runner.write_json(path, rescored)
    with pytest.raises(AssertionError, match="not the file its rescoring"):
        analysis.analyze_arms(reference, candidate, None, tmp_path / "r")
    (candidate / runner.RESCORING).unlink()
    with pytest.raises(AssertionError, match="holds no rescoring.json"):
        analysis.analyze_arms(reference, candidate, None, tmp_path / "r")


def test_arms_refuse_a_rescoring_by_other_code_than_the_candidates(
    arms, tmp_path
):
    reference, candidate = arms
    other = source(HARNESS)
    other["versions"]["numpy"] = "2.6.0"
    write_rescoring(
        candidate,
        {
            reference: rescored_cases(candidate, reference),
            candidate: rescored_cases(candidate, candidate),
        },
        rescoring_source=other,
    )
    with pytest.raises(AssertionError, match="not the candidate's"):
        analysis.analyze_arms(reference, candidate, None, tmp_path / "r")


def test_arms_refuse_a_candidate_whose_rescored_metrics_differ(arms, tmp_path):
    # The flag says equal; the values are compared all the same.
    reference, candidate = arms
    cases = rescored_cases(candidate, candidate)
    cases[f"{LABELS[1]}_seed7"]["metrics"]["mmtv"] += 1e-9
    write_rescoring(
        candidate,
        {reference: rescored_cases(candidate, reference), candidate: cases},
    )
    with pytest.raises(AssertionError, match="differ from its in-run"):
        analysis.analyze_arms(reference, candidate, None, tmp_path / "r")


def test_the_confirmatory_family_keeps_the_size_fixed_before_the_runs(
    arms, tmp_path
):
    # Every case of the second configuration failed in the reference arm:
    # its four tests cannot be computed and enter the family at p = 1.
    reference, candidate = arms
    path = reference / "verification.json"
    report = json.loads(path.read_text())
    cases = rescored_cases(candidate, reference)
    for case in report["cases"]:
        _, label, seed = runner.parse_case(case["case"])
        if label == LABELS[1]:
            case["status"] = "failed"
            case["reason"] = "x: RuntimeError: the run broke"
            del case["record_sha256"]
            contract.record_path(reference, case["tag"]).unlink()
            cases[f"{label}_seed{seed}"] = {"status": "failed"}
    report["counts"].update(verified=100, failed=100)
    runner.write_json(path, report)
    write_rescoring(
        candidate,
        {reference: cases, candidate: rescored_cases(candidate, candidate)},
    )
    result = analysis.analyze_arms(reference, candidate, None, tmp_path / "r")
    assert result["paired_cases"] == 100
    tests = {
        (t["label"], t["metric"]): t for t in result["confirmatory_tests"]
    }
    assert len(tests) == 8
    for metric in ("elbo_err", "gskl", "mmtv", "usable"):
        test = tests[(LABELS[1], metric)]
        assert not test["computed"] and "no seed" in test["reason"]
        assert test["pvalue"] == 1.0 and not test["holm_rejected"]
        assert (test["n_pairs"], test["planned_pairs"]) == (0, 100)
    assert len(result["confirmatory_not_computed"]) == 4
    # Holm runs over the eight tests fixed before the runs, not the four
    # that could be computed.
    halved = tests[(LABELS[0], "gskl")]
    assert halved["holm_adjusted_pvalue"] == 8 * 2 / 2**100
    assert result["arms"]["reference"]["verification_counts"]["failed"] == 100


def test_rescored_metrics_that_are_not_finite_are_left_out(arms, tmp_path):
    reference, candidate = arms
    cases = rescored_cases(candidate, reference)
    for seed in range(10):
        entry = cases[f"{LABELS[1]}_seed{seed}"]
        entry["metrics"]["gskl"] = float("inf") if seed < 5 else float("nan")
        entry["equal_to_in_run"]["gskl"] = False
    write_rescoring(
        candidate,
        {reference: cases, candidate: rescored_cases(candidate, candidate)},
    )
    result = analysis.analyze_arms(reference, candidate, None, tmp_path / "r")
    tests = {
        (t["label"], t["metric"]): t for t in result["confirmatory_tests"]
    }
    test = tests[(LABELS[1], "gskl")]
    assert test["computed"] and test["nonfinite_pairs"] == 10
    assert test["n_pairs"] == 100
    assert test["improved"] + test["worsened"] + test["tied"] == 90
    assert result["arms"]["reference"]["rescored_nonfinite"]["gskl"] == 10
    assert result["aggregate"]["reference"]["nonfinite"]["gskl"] == 10
    assert np.isfinite(
        result["aggregate"]["reference"]["metrics"]["gskl"]["max"]
    )
    # A run whose metric is not finite is not usable.
    changes = {c["tag"]: c for c in result["paired_changes"]}
    assert not any(
        changes[f"{LABELS[1]}_seed{seed}"]["old_usable"] for seed in range(10)
    )
    written = json.loads((tmp_path / "r" / "assessment.json").read_text())
    assert written["confirmatory_tests"] == result["confirmatory_tests"]


def test_the_redacted_copies_give_the_same_assessment(tmp_path):
    """The two arms, run at a stand-in site, are redacted into the tracked
    tree of a release gate, which holds none of the site's details; the
    comparison of the copies writes the report the arms themselves give."""
    site = stubs.FakeSite(tmp_path)
    runs = site.home / "runs"
    reference, before = write_arm(runs, "reference", "b" * 40, site=site)
    candidate, after = write_arm(runs, "candidate", HARNESS, site=site)
    write_rescoring(
        candidate, {reference: before, candidate: after}, site=site
    )
    assert site.leaks(reference) and site.leaks(candidate)
    analysis.analyze_arms(reference, candidate, None, tmp_path / "raw")
    tracked = tmp_path / "handback" / "release_gate"
    for arm, name in (
        (reference, "population_before"),
        (candidate, "population_after"),
    ):
        contract.redact(
            arm,
            tracked / name,
            operator=site.operator(),
            environ={},
            host="fakelogin9",
            say=lambda message: None,
        )
    assert site.leaks(tracked) == []
    after_copies = sorted(
        p.relative_to(tracked / "population_after").as_posix()
        for p in (tracked / "population_after").rglob("*")
        if p.is_file()
    )
    assert "rescored/reference.json" in after_copies
    assert runner.RESCORING in after_copies
    assert len(after_copies) == 7 + 3 * len(LABELS) * len(SEEDS)
    result = analysis.analyze_arms(
        tracked / "population_before",
        tracked / "population_after",
        None,
        tmp_path / "copies",
    )
    assert result["paired_cases"] == 200
    assert result["arms"]["reference"]["name"] == "reference"
    for name in ("assessment.json", "comparison.md"):
        assert (tmp_path / "copies" / name).read_bytes() == (
            tmp_path / "raw" / name
        ).read_bytes(), name
    # A copy changed after the redaction is refused.
    sidecar = (
        tracked
        / "population_before"
        / runner.case_files(LABELS[0], 3)["sidecar"]
    )
    sidecar.write_text(sidecar.read_text() + " ")
    with pytest.raises(contract.ContractError, match="not the copy"):
        analysis.analyze_arms(
            tracked / "population_before",
            tracked / "population_after",
            None,
            tmp_path / "again",
        )
