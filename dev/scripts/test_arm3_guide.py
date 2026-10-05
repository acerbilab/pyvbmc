"""Tests of ``arm3_guide.py``: the directions and tests of the guide, on
hand-made inputs, and its readers, on the stacking summary and the
single-run analyses that the release gate's first batch and stage D left
in the repository."""

import json
import math
from pathlib import Path

import analyze_population_run as analysis
import arm3_guide as guide
import numpy as np
import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
NOISY = "rosenbrock_D2_noise3_production"
NOISELESS = "banana_D2"


def test_rank_sums_leave_out_zeros_and_give_ties_their_mean_rank():
    assert guide.rank_sums([0.0, 1.0, -1.0, 2.0, math.nan]) == (4.5, 1.5)
    assert guide.rank_sums([0.0, math.inf - math.inf]) == (0.0, 0.0)
    assert guide.direction_of([0.0]) == "neither"


def test_the_direction_is_the_rank_sums_not_the_median():
    """51 small gains and 49 large losses: the median change is a gain, the
    rank sums and the test a loss."""
    deltas = [-0.01] * 51 + [1.0] * 49
    assert np.median(deltas) < 0
    r_plus, r_minus = guide.rank_sums(deltas)
    assert r_plus > r_minus
    test = guide.signed_rank("mmtv", deltas)
    assert test["direction"] == "worse" and test["pvalue"] < 1e-3


def test_a_test_with_no_change_enters_its_family_at_p_one():
    test = guide.signed_rank("gskl", [0.0, 0.0, math.nan])
    assert not test["computed"] and test["pvalue"] == 1.0
    assert test["nonfinite_pairs"] == 1


def assessment(rows_by_label, usable=None):
    """A stand-in for ``assessment.json``: ``rows_by_label`` maps each label
    to its metrics' paired changes, ``{metric: deltas}``, and ``usable`` to
    its ``(gains, losses)``; the confirmatory tests are computed from them
    as the analysis computes them."""
    rows, tests = [], []
    for label, metrics in rows_by_label.items():
        n = len(next(iter(metrics.values())))
        for i in range(n):
            rows.append(
                {
                    "label": label,
                    "delta": {m: float(d[i]) for m, d in metrics.items()},
                }
            )
        for metric in guide.METRICS:
            statistic, pvalue = analysis.exact_signed_rank(
                np.asarray(metrics[metric], dtype=float)
            )
            tests.append(
                {
                    "label": label,
                    "metric": metric,
                    "n_pairs": n,
                    "statistic": statistic,
                    "pvalue": pvalue,
                    "computed": True,
                }
            )
        gains, losses = (usable or {}).get(label, (0, 0))
        tests.append(
            {
                "label": label,
                "metric": "usable",
                "gains": gains,
                "losses": losses,
                "pvalue": guide.mcnemar("usable", gains, losses)["pvalue"],
                "computed": True,
            }
        )
    return {
        "kind": "two arms of array mode, paired by seed",
        "allocation": {"labels": list(rows_by_label)},
        "paired_changes": rows,
        "confirmatory_tests": tests,
        "confirmatory_not_computed": [],
        "arms": {
            role: {
                "name": role,
                "verification_counts": {"verified": 1, "failed": 0},
                "rescored_nonfinite": {"mmtv": 0},
            }
            for role in ("reference", "candidate")
        },
    }


def changes(seed, shift, n=30):
    """Paired changes with a shift: mostly one way, a few the other."""
    rng = np.random.default_rng(seed)
    return shift + 0.01 * rng.standard_normal(n)


def test_the_fresh_seeds_test_is_one_sided_in_arm3s_favour():
    better = {m: changes(1, -0.004) for m in guide.METRICS}
    part = guide.fresh_part(assessment({guide.FRESH_LABEL: better}))
    mmtv = part["mmtv"]
    assert mmtv["direction"] == "better"
    assert mmtv["one_sided_pvalue"] == pytest.approx(mmtv["pvalue"] / 2)
    assert part["mmtv_lower"] == (mmtv["pvalue"] / 2 <= guide.ALPHA)
    # The same changes the other way round: the analysis's statistic, the
    # smaller rank sum, is the same, and the test does not reject.
    worse = {m: -d for m, d in better.items()}
    flipped = guide.fresh_part(assessment({guide.FRESH_LABEL: worse}))
    assert flipped["mmtv"]["direction"] == "worse"
    assert flipped["mmtv"]["one_sided_pvalue"] is None
    assert not flipped["passes"]
    statistics = [
        analysis.exact_signed_rank(np.asarray(c["mmtv"]))[0]
        for c in (better, worse)
    ]
    assert statistics[0] == statistics[1]


def test_the_fresh_seeds_usability_can_refuse():
    better = {m: changes(2, -0.01) for m in guide.METRICS}
    part = guide.fresh_part(
        assessment({guide.FRESH_LABEL: better}, {guide.FRESH_LABEL: (0, 9)})
    )
    assert part["mmtv_lower"] and part["usability_worse"]
    assert not part["passes"]


def test_each_configuration_is_its_own_holm_family():
    """A loss whose p-value passes Holm over the whole family of 96 tests
    and is refused within its configuration's four."""
    noisy = {m: changes(3, 0.0) for m in guide.METRICS}
    noisy["mmtv"] = changes(3, 0.006)
    noiseless = {m: changes(4, -0.006) for m in guide.METRICS}
    data = assessment({NOISY: noisy, NOISELESS: noiseless})
    p_mmtv = next(
        t["pvalue"]
        for t in data["confirmatory_tests"]
        if t["label"] == NOISY and t["metric"] == "mmtv"
    )
    assert 4 * p_mmtv <= guide.ALPHA < 96 * p_mmtv
    part = guide.population_part(data)
    assert part["noisy_worse"] == [NOISY]
    assert part["noiseless_better"] == [NOISELESS]
    assert part["noiseless_worse"] == []
    assert part["configurations"][NOISY]["noisy"]
    assert not part["configurations"][NOISELESS]["noisy"]


def test_cases_and_tests_the_tests_cannot_see_go_to_the_pi():
    data = assessment({NOISY: {m: changes(5, 0.0) for m in guide.METRICS}})
    data["arms"]["candidate"]["verification_counts"]["failed"] = 2
    data["confirmatory_not_computed"] = [{"label": NOISY, "metric": "gskl"}]
    items = guide.population_part(data)["for_the_pi"]
    assert len(items) == 2 and "failed" in items[0]


def write_pool(root, name, runs, statuses=None):
    """A pool campaign with the files the guide reads: its verification
    report and each verified run's completion record."""
    pool = root / name
    cases = []
    for (label, seed), (metrics, passes) in runs.items():
        tag = f"{label}/{label}_seed{seed}"
        status = (statuses or {}).get((label, seed), "verified")
        cases.append(
            {"tag": tag, "label": label, "seed": seed, "status": status}
        )
        record = pool / "records" / f"{tag}.complete.json"
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(
            json.dumps({"metrics": metrics, "verdict": {"passes": passes}}),
            encoding="utf-8",
        )
    counts = {"verified": sum(c["status"] == "verified" for c in cases)}
    counts["failed"] = len(cases) - counts["verified"]
    (pool / "verification.json").write_text(
        json.dumps({"exit_code": 0, "counts": counts, "cases": cases}),
        encoding="utf-8",
    )
    return pool


def test_the_pools_pair_verified_runs_by_seed(tmp_path):
    noisy, noiseless = "rosenbrock_D2_noise3_svbmc", "gmm_D2_svbmc"
    rng = np.random.default_rng(6)
    reference, candidate = {}, {}
    for label in (noisy, noiseless):
        for seed in range(1000, 1040):
            base = {m: float(rng.uniform(0.05, 0.2)) for m in guide.METRICS}
            reference[(label, seed)] = (base, True)
            shift = 0.05 if label == noisy else 0.0
            noise = {m: 0.002 * rng.standard_normal() for m in guide.METRICS}
            candidate[(label, seed)] = (
                {m: base[m] + shift + noise[m] for m in guide.METRICS},
                True,
            )
    candidate[(noiseless, 1001)][0]["gskl"] = math.inf
    pools = [
        write_pool(tmp_path, "reference", reference),
        write_pool(
            tmp_path, "candidate", candidate, {(noisy, 1000): "failed"}
        ),
    ]
    part = guide.pools_part(*pools)
    assert part["noisy_worse"] == [noisy]
    assert part["noiseless_worse"] == []
    assert part["conditions"][noisy]["paired_seeds"] == 39
    gskl = next(
        t
        for t in part["conditions"][noiseless]["tests"]
        if t["metric"] == "gskl"
    )
    assert gskl["nonfinite_pairs"] == 1
    assert any("not finite" in i for i in part["for_the_pi"])
    assert any("failed" in i for i in part["for_the_pi"])


def test_the_stacking_is_read_as_the_release_pools_stacking_was():
    text = (
        EXPERIMENTS / "release_gate_20261002" / "stacking" / "summary.md"
    ).read_text(encoding="utf-8")
    part = guide.stacking_part(text)
    conditions = part["conditions"]
    assert [c["max_dw"] for c in conditions.values()] == [
        0.0204,
        0.0184,
        0.0168,
        0.0184,
        0.0070,
        0.0269,
        0.0130,
        0.0149,
    ]
    # Criterion 3 fails on Student D8 alone, at M = 2, 4, 8 and 16.
    failing = {k for k, c in conditions.items() if not c["criteria"]["3"]}
    assert failing == {guide.STUDENT}
    gates = dict(conditions[guide.STUDENT]["criterion_3_gates"])
    assert [m for m, g in gates.items() if g == "no"] == ["2", "4", "8", "16"]
    assert all(c["meets"] for c in conditions.values())
    assert part["noisy_meets"] and part["noiseless_meets"]


def test_the_headline_bounds_hold_the_stage_d_pools():
    """Decision 6's bounds were widened to hold stage D at M = 32 (the
    two-level estimate up to +0.19, the raw noiseless ELBO up to +0.11)."""
    text = (
        EXPERIMENTS / "svbmc_pool" / "single_run_20260915" / "added.md"
    ).read_text(encoding="utf-8")
    part = guide.headline_part(text)
    by_m = {r["M"]: r for r in part["rows"][guide.STUDENT]}
    assert by_m[32]["two_level_full"] == 0.19
    assert by_m[32]["capped"] == -1.57
    noiseless = part["rows"]["multisensory_s1_D6_svbmc"]
    assert max(r["raw"] for r in noiseless) == 0.11
    assert part["noisy_switches"] and part["noiseless_keeps_raw"]
    assert part["two_level_outside"] == [] and part["raw_outside"] == []


def test_the_guide_stays_open_until_its_parts_are_read(tmp_path):
    said = guide.verdicts({})
    assert said["warm_up_noisy"].startswith("open")
    assert said["warm_up_noiseless"].startswith("open")
    stacking = EXPERIMENTS / "release_gate_20261002" / "stacking"
    added = EXPERIMENTS / "svbmc_pool" / "single_run_20260915" / "added.md"
    out = tmp_path / "guide"
    assert (
        guide.main(
            [
                "--out",
                str(out),
                "--stacking",
                str(stacking / "summary.md"),
                "--added",
                str(added),
            ]
        )
        == 0
    )
    result = json.loads((out / "guide.json").read_text(encoding="utf-8"))
    assert set(result["parts"]) == {"stacking", "headline"}
    assert result["said"]["headline_noisy"].startswith("the two-level")
    assert "what the guide says" in (out / "guide.md").read_text(
        encoding="utf-8"
    )
