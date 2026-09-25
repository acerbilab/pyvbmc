"""Checks of the population assessment (run by explicit path).

The statistics are checked against SciPy and an independent count; the
comparison of two arms of array mode runs on campaign directories that the
fixtures here write (sidecars, boost reports, completion records,
verification reports and rescored metrics), which test the reader and the
statistics, not VBMC.
"""

import json
from fractions import Fraction

import analyze_population_run as analysis
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


def write_arm(root, role, pyvbmc_commit, files=None):
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
        side_path = path / files_of["sidecar"]
        side_path.parent.mkdir(parents=True, exist_ok=True)
        side_path.write_text(
            json.dumps({"label": label, "seed": seed, "final": final})
        )
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
            files_of["sidecar"]: {"sha256": runner.sha256(side_path)},
            files_of["boost_report"]: {"sha256": runner.sha256(report_path)},
        }
        runner.write_json(
            contract.record_path(path, tag),
            {
                "tag": tag,
                "identity": manifest["identity"],
                "artifacts": artifacts,
            },
        )
        cases.append(
            {"index": index, "tag": tag, "case": line, "status": "verified"}
        )
        rescored[stem] = {
            "status": "rescored",
            "metrics": again,
            "equal_to_in_run": {k: again[k] == in_run[k] for k in again},
            "artifacts": {
                files_of["sidecar"]: artifacts[files_of["sidecar"]]["sha256"]
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
    return path, rescored


def write_rescored(candidate, arm, cases):
    runner.write_json(
        candidate / "rescored" / f"{arm.name}.json",
        {
            "campaign": {
                "name": arm.name,
                "manifest_sha256": runner.sha256(arm / "manifest.json"),
                "verification_sha256": runner.sha256(
                    arm / "verification.json"
                ),
            },
            "rescoring": {"identity": {"source": source(HARNESS)}},
            "cases": cases,
        },
    )


@pytest.fixture
def arms(tmp_path):
    reference, before = write_arm(tmp_path, "reference", "b" * 40)
    candidate, after = write_arm(tmp_path, "candidate", HARNESS)
    write_rescored(candidate, reference, before)
    write_rescored(candidate, candidate, after)
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
    halved = confirmatory[(LABELS[0], "gskl")]
    assert halved["n_pairs"] == 100 and halved["improved"] == 100
    assert halved["pvalue"] == 2 / 2**100 and halved["holm_rejected"]
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
    with pytest.raises(AssertionError):
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
    write_rescored(candidate, reference, before)
    write_rescored(candidate, candidate, after)
    with pytest.raises(AssertionError, match="harness files"):
        analysis.analyze_arms(reference, candidate, None, tmp_path / "r")
