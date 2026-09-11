"""Assess completed population artifacts without running inference or scoring.

A first-stage campaign and any extension campaigns of the same frozen
treatment are pooled. Every case is revalidated against its own campaign's
launch record, the campaigns must share one source/environment identity, and
the pooled population is compared with the golden reference. Each extension
is also reported on its own, with a confirmatory test family fixed before it
ran (`assessment` in its launch manifest).
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

# Initialize the launcher's single-thread environment before NumPy imports.
import population_run as runner  # isort: skip

# isort: split
import numpy as np
from scipy import stats

QUALITY = ("elbo_err", "gskl", "mmtv")
ALPHA = 0.05


def exact_signed_rank(delta):
    """Two-sided exact signed-rank test over all sign assignments.

    Zero differences are dropped, tied absolute differences receive midranks,
    and the null distribution of the positive-rank sum is enumerated by
    dynamic programming over the midranks, so ties are handled exactly at
    any sample size. The result equals SciPy's ``wilcoxon`` with
    ``PermutationMethod(n_resamples=np.inf)``, whose enumeration of the
    ``2**n`` sign vectors is limited to about 20 pairs. Returns the smaller
    rank sum and the p-value.
    """
    d = np.asarray(delta, dtype=float)
    d = d[d != 0]
    m = len(d)
    if m < 2:
        return 0.0, 1.0
    if m > 62:
        raise ValueError("Exact enumeration counts exceed int64 beyond 62.")
    ranks = stats.rankdata(np.abs(d))
    units = np.rint(2 * ranks).astype(np.int64)  # midranks are half-integers
    assert np.array_equal(units, 2 * ranks)
    total = int(units.sum())
    counts = np.zeros(total + 1, dtype=np.int64)
    counts[0] = 1
    for unit in units:
        shifted = counts.copy()
        shifted[unit:] += counts[: total + 1 - unit]
        counts = shifted
    positive = int(units[d > 0].sum())
    n_assignments = float(2**m)
    less = counts[: positive + 1].sum() / n_assignments
    greater = counts[positive:].sum() / n_assignments
    statistic = float(min(ranks[d > 0].sum(), ranks[d < 0].sum()))
    return statistic, float(min(1.0, 2 * min(less, greater)))


def holm(tests, alpha=ALPHA):
    """Annotate one test family in place with Holm-adjusted p-values."""
    order = np.argsort([test["pvalue"] for test in tests])
    adjusted = 0.0
    for rank, index in enumerate(order):
        adjusted = float(
            max(
                adjusted,
                min(1.0, (len(tests) - rank) * tests[index]["pvalue"]),
            )
        )
        tests[index]["holm_adjusted_pvalue"] = adjusted
        tests[index]["holm_rejected"] = bool(adjusted <= alpha)
    assert [test["holm_rejected"] for test in tests] == list(
        runner.golden_trace._holm([test["pvalue"] for test in tests], alpha)
    )
    return tests


def signed_rank_test(label, metric, rows):
    delta = np.array([row["delta"][metric] for row in rows])
    statistic, pvalue = exact_signed_rank(delta)
    return {
        "label": label,
        "metric": metric,
        "method": "exact signed-rank sign permutation",
        "n_pairs": len(rows),
        "improved": int(sum(delta < 0)),
        "worsened": int(sum(delta > 0)),
        "tied": int(sum(delta == 0)),
        "median_paired_change": float(np.median(delta)),
        "statistic": statistic,
        "pvalue": pvalue,
    }


def mcnemar_test(label, rows):
    gains = sum(not row["old_usable"] and row["new_usable"] for row in rows)
    losses = sum(row["old_usable"] and not row["new_usable"] for row in rows)
    pvalue = (
        float(stats.binomtest(gains, gains + losses).pvalue)
        if gains + losses
        else 1.0
    )
    return {
        "label": label,
        "metric": "usable",
        "method": "exact McNemar",
        "n_pairs": len(rows),
        "gains": gains,
        "losses": losses,
        "pvalue": pvalue,
    }


def paired_tests(changes, metrics=(*QUALITY, "func_count"), adjust=True):
    """Test within-configuration seed pairs; one Holm family when `adjust`.

    Signed-rank nulls assume symmetric, independent seed differences.
    Usability uses exact McNemar tests, conditional on the discordant pairs.
    """
    tests = []
    for label in sorted({row["label"] for row in changes}):
        rows = [row for row in changes if row["label"] == label]
        for metric in metrics:
            tests.append(signed_rank_test(label, metric, rows))
        tests.append(mcnemar_test(label, rows))
    return holm(tests) if adjust else tests


def usable(metrics):
    return bool(
        metrics["elbo_err"] < 1
        and metrics["gskl"] < 1
        and metrics["mmtv"] < 0.2
    )


def load_rows(folder):
    return {
        p.stem: json.loads(p.read_text()) for p in folder.glob("*_seed*.json")
    }


def describe(rows):
    finals = [r["final"] for r in rows]
    return {
        "n": len(rows),
        "converged": sum(bool(r["success_flag"]) for r in finals),
        "usable": sum(usable(r) for r in finals),
        "optimizer_seconds": sum(r["wall_s"] for r in finals),
        "evaluations": sum(r["func_count"] for r in finals),
        "metrics": {
            k: {
                "median": float(np.median([r[k] for r in finals])),
                "q90": float(np.quantile([r[k] for r in finals], 0.9)),
                "max": float(max(r[k] for r in finals)),
            }
            for k in (*QUALITY, "func_count", "wall_s", "peak_rss_mb")
        },
    }


def merge_populations(populations):
    """Pool golden-harness populations (`load_population`) by label."""
    merged = {}
    for population in populations:
        for label, entry in population.items():
            target = merged.setdefault(
                label, {"seeds": [], "rows": [], "fails": 0}
            )
            target["seeds"] += entry["seeds"]
            target["rows"] += entry["rows"]
            target["fails"] += entry["fails"]
    trace = runner.golden_trace
    for entry in merged.values():
        for m in trace.METRICS + trace.EXTRA_SCALARS:
            entry[m] = np.array(
                [r.get(m, np.nan) for r in entry["rows"]], dtype=float
            )
    return merged


def verify_reference(reference):
    baseline = load_rows(reference)
    manifest = json.loads(
        (
            runner.ROOT
            / "dev/golden/noisy_extension_20260907/sha256_manifest.json"
        ).read_text()
    )
    for tag in baseline:
        raw = (reference / f"{tag}.json").read_bytes().replace(b"\r\n", b"\n")
        assert (
            hashlib.sha256(raw).hexdigest()
            == manifest["files"][tag]["json_sha256"]
        )
    return baseline


def load_campaign(campaign, reference_population):
    """Revalidate one campaign's artifacts and its own comparison."""
    cases = Path(campaign) / "results"
    launch = json.loads((cases / "launch.json").read_text())
    finished = json.loads((cases / "finished.json").read_text())
    manifest = launch["manifest"]
    expected = {
        f"{c['label']}_seed{s}"
        for c in manifest["allocation"]
        for s in c["seeds"]
    }
    assert len(expected) == manifest["candidate_run_count"]
    rows = load_rows(cases)
    assert set(rows) == expected == set(finished["completed"])
    assert not list(cases.glob("*.error.txt"))
    for tag in sorted(expected):
        runner.validate_case(cases, tag, launch["identity"])
    comparison, _ = runner.golden_trace.compare_populations(
        reference_population, runner.golden_trace.load_population(cases)
    )
    assert comparison.strip() == (cases / "comparison.md").read_text().strip()
    return {
        "name": Path(campaign).name,
        "path": Path(campaign),
        "cases": cases,
        "identity": launch["identity"],
        "manifest": manifest,
        "rows": rows,
        "elapsed_seconds": finished["elapsed_seconds"],
        "finished": finished["finished"],
        "comparison_reproduced_exactly": True,
    }


def paired_change(tag, label, new, old):
    keys = (*QUALITY, "func_count", "wall_s")
    return {
        "tag": tag,
        "label": label,
        "old_usable": usable(old),
        "new_usable": usable(new),
        "old_converged": old["success_flag"],
        "new_converged": new["success_flag"],
        "old": {k: old[k] for k in keys},
        "new": {k: new[k] for k in keys},
        "delta": {k: new[k] - old[k] for k in keys},
    }


def boost_record(cases, tag, label, new):
    record = json.loads(
        runner.case_path(cases, tag, ".boost.json").read_text()
    )
    pre, candidate = record["scores"]["pre"], record["scores"]["candidate"]
    worst_change = None
    expected_accept = False
    if record["attempted"]:
        valid_pre = runner.VBMC._is_valid_final_boost_score(
            pre["elbo"], pre["elbo_sd"]
        )
        valid_candidate = runner.VBMC._is_valid_final_boost_score(
            candidate["elbo"], candidate["elbo_sd"]
        )
        de = candidate["elbo"] - pre["elbo"]
        ds = candidate["elbo_sd"] - pre["elbo_sd"]
        worst_change = min(de, de - 5 * ds)
        expected_accept = bool(
            valid_candidate
            and (not valid_pre or worst_change > -record["tolerance"])
        )
    assert expected_accept == record["accepted"], tag
    quality = record["metrics"]
    assert all("error" not in q for q in quality.values()), (tag, quality)
    for k in QUALITY:
        assert quality["returned"][k] == new[k], (tag, k)
    return {
        "tag": tag,
        "label": label,
        "attempted": record["attempted"],
        "accepted": record["accepted"],
        "worst_score_change": worst_change,
        "scores": record["scores"],
        "quality": {
            stage: {k: value[k] for k in QUALITY}
            for stage, value in quality.items()
        },
        "usable": {stage: usable(value) for stage, value in quality.items()},
    }


def summarize_boosts(boosts):
    return {
        "attempted": sum(b["attempted"] for b in boosts),
        "accepted": sum(b["accepted"] for b in boosts),
        "rejected": sum(b["attempted"] and not b["accepted"] for b in boosts),
        "skipped": sum(not b["attempted"] for b in boosts),
        "usable_pre": sum(b["usable"]["pre"] for b in boosts),
        "usable_candidate": sum(
            b["usable"].get("candidate", b["usable"]["pre"]) for b in boosts
        ),
        "usable_returned": sum(b["usable"]["returned"] for b in boosts),
    }


def configuration_rows(labels, current, baseline):
    rows = {}
    for label in labels:
        new = [r for r in current.values() if r["label"] == label]
        rows[label] = {
            "reference_all": describe(
                [r for r in baseline.values() if r["label"] == label]
            ),
            "reference_matched": describe(
                [baseline[f"{label}_seed{r['seed']}"] for r in new]
            ),
            "candidate": describe(new),
        }
    return rows


def follow_up(extension, first_stage, baseline, changes, boosts):
    """Report an extension on its own, with its pre-specified test family."""
    rows = extension["rows"]
    labels = sorted({r["label"] for r in rows.values()})
    tags = sorted(rows)
    stage_tags = sorted(
        tag for tag, r in first_stage["rows"].items() if r["label"] in labels
    )
    own = [changes[tag] for tag in tags]
    return {
        "campaign": extension["name"],
        "labels": labels,
        "seeds": sorted({r["seed"] for r in rows.values()}),
        "elapsed_seconds": extension["elapsed_seconds"],
        "finished": extension["finished"],
        "question": extension["manifest"].get("assessment"),
        "configurations": configuration_rows(labels, rows, baseline),
        "aggregate": {
            "reference_matched": describe([baseline[tag] for tag in tags]),
            "candidate": describe([rows[tag] for tag in tags]),
        },
        "paired_changes": own,
        "confirmatory_tests": paired_tests(own, metrics=QUALITY),
        "first_stage_tests": paired_tests(
            [changes[tag] for tag in stage_tags], metrics=QUALITY, adjust=False
        ),
        "boost_summary": summarize_boosts([boosts[tag] for tag in tags]),
        "usability_losses": [
            tag
            for tag in tags
            if changes[tag]["old_usable"] and not changes[tag]["new_usable"]
        ],
        "usability_gains": [
            tag
            for tag in tags
            if not changes[tag]["old_usable"] and changes[tag]["new_usable"]
        ],
    }


def analyze(campaign, extensions, out):
    if not __debug__:
        raise RuntimeError(
            "Run without -O: artifact validation uses assertions."
        )
    reference = runner.ROOT / "dev/golden/baseline"
    baseline = verify_reference(reference)
    reference_population = runner.golden_trace.load_population(reference)
    campaigns = [
        load_campaign(path, reference_population)
        for path in (campaign, *extensions)
    ]
    identity = campaigns[0]["identity"]
    assert all(c["identity"] == identity for c in campaigns)
    current, where = {}, {}
    for c in campaigns:
        assert not set(c["rows"]) & set(current), "campaigns overlap"
        current.update(c["rows"])
        where.update({tag: c["cases"] for tag in c["rows"]})
    print(
        f"Validated {len(current)} complete candidate cases in"
        f" {len(campaigns)} campaign(s) and {len(baseline)} reference"
        " sidecars.",
        flush=True,
    )
    comparison, flagged = runner.golden_trace.compare_populations(
        reference_population,
        merge_populations(
            [
                runner.golden_trace.load_population(c["cases"])
                for c in campaigns
            ]
        ),
    )
    labels = sorted({r["label"] for r in current.values()})
    config_rows = configuration_rows(labels, current, baseline)
    tests = []
    for label in labels:
        for k in (*QUALITY, "func_count"):
            a = np.array(
                [
                    r["final"][k]
                    for r in baseline.values()
                    if r["label"] == label
                ]
            )
            b = np.array(
                [
                    r["final"][k]
                    for r in current.values()
                    if r["label"] == label
                ]
            )
            test = stats.ks_2samp(a, b)
            tests.append(
                {
                    "label": label,
                    "metric": k,
                    "statistic": float(test.statistic),
                    "pvalue": float(test.pvalue),
                }
            )
    decisions = runner.golden_trace._holm([t["pvalue"] for t in tests], ALPHA)
    for test, reject in zip(tests, decisions):
        test["holm_rejected"] = bool(reject)
    changes, boosts = {}, {}
    for tag in sorted(current):
        label = current[tag]["label"]
        new, old = current[tag]["final"], baseline[tag]["final"]
        changes[tag] = paired_change(tag, label, new, old)
        boosts[tag] = boost_record(where[tag], tag, label, new)
    ordered_changes = [changes[tag] for tag in sorted(current)]
    ordered_boosts = [boosts[tag] for tag in sorted(current)]
    result = {
        "candidate_identity": identity,
        "campaigns": [
            {
                k: c[k]
                for k in (
                    "name",
                    "elapsed_seconds",
                    "finished",
                    "comparison_reproduced_exactly",
                )
            }
            | {"cases": len(c["rows"]), "stage": c["manifest"].get("stage")}
            for c in campaigns
        ],
        "verified_candidate_cases": len(current),
        "verified_reference_sidecars": len(baseline),
        "elapsed_seconds": sum(c["elapsed_seconds"] for c in campaigns),
        "finished": max(c["finished"] for c in campaigns),
        "comparison_reproduced_exactly": True,
        "flagged_configurations": sorted(flagged),
        "ks_tests": tests,
        "aggregate": {
            "reference_all": describe(list(baseline.values())),
            "reference_matched": describe([baseline[tag] for tag in current]),
            "candidate": describe(list(current.values())),
        },
        "configurations": config_rows,
        "paired_changes": ordered_changes,
        "paired_tests": paired_tests(ordered_changes),
        "boosts": ordered_boosts,
        "boost_summary": summarize_boosts(ordered_boosts),
        "follow_up": [
            follow_up(extension, campaigns[0], baseline, changes, boosts)
            for extension in campaigns[1:]
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    runner.write_json(out / "assessment.json", result)
    (out / "comparison.md").write_text(comparison + "\n", encoding="utf-8")
    for c in campaigns:
        shutil.copyfile(
            c["path"] / "launch_manifest.json",
            out / f"{c['name']}_manifest.json",
        )
    print(
        "Aggregate:",
        json.dumps(
            {
                k: {
                    m: v[m]
                    for m in [
                        "n",
                        "converged",
                        "usable",
                        "optimizer_seconds",
                        "evaluations",
                    ]
                }
                for k, v in result["aggregate"].items()
            }
        ),
        flush=True,
    )
    print("Boost:", result["boost_summary"], flush=True)
    print(
        "Usability losses:",
        [
            x["tag"]
            for x in ordered_changes
            if x["old_usable"] and not x["new_usable"]
        ],
        flush=True,
    )
    print(
        "Usability gains:",
        [
            x["tag"]
            for x in ordered_changes
            if not x["old_usable"] and x["new_usable"]
        ],
        flush=True,
    )
    print(
        "Rejected boosts:",
        [
            b["tag"]
            for b in ordered_boosts
            if b["attempted"] and not b["accepted"]
        ],
        flush=True,
    )
    for report in result["follow_up"]:
        print(f"Follow-up {report['campaign']} seeds {report['seeds']}:")
        for test in report["confirmatory_tests"]:
            detail = (
                f"gains {test['gains']} losses {test['losses']}"
                if test["metric"] == "usable"
                else f"median change {test['median_paired_change']:+.4g},"
                f" improved/worsened {test['improved']}/{test['worsened']}"
            )
            print(
                f"  {test['label']} {test['metric']}: {detail},"
                f" p {test['pvalue']:.4g}, Holm {test['holm_adjusted_pvalue']:.4g}",
                flush=True,
            )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign",
        type=Path,
        default=runner.ROOT / "dev/scripts/runs/population_overnight_20260910",
        help="first-stage campaign directory",
    )
    parser.add_argument(
        "--extension",
        type=Path,
        nargs="*",
        default=[
            runner.ROOT / "dev/scripts/runs/population_extension_20260911"
        ],
        help="extension campaigns of the same treatment; give the flag"
        " without paths to assess the first stage alone",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=runner.ROOT / "dev/experiments/population_extension_20260911",
    )
    args = parser.parse_args()
    analyze(args.campaign, args.extension, args.out)
