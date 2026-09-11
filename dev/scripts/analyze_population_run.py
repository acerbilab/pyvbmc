"""Assess completed population artifacts without running inference or scoring."""

import argparse
import hashlib
import json
from pathlib import Path

# Initialize the launcher's single-thread environment before NumPy imports.
import population_run as runner  # isort: skip

# isort: split
import numpy as np
from scipy import stats

QUALITY = ("elbo_err", "gskl", "mmtv")


def paired_tests(changes):
    """Test within-configuration seed pairs, with one 95-test Holm family.

    Signed-rank tests enumerate all sign assignments to handle ties exactly.
    Their null assumes symmetric, independent seed differences. Usability
    uses exact McNemar tests, conditional on the discordant pairs.
    """
    tests = []
    for label in sorted({row["label"] for row in changes}):
        rows = [row for row in changes if row["label"] == label]
        if len(rows) > 20:
            raise ValueError(
                "Exhaustive paired analysis is limited to 20 seeds."
            )
        for metric in (*QUALITY, "func_count"):
            delta = np.array([row["delta"][metric] for row in rows])
            nonzero = delta[delta != 0]
            if len(nonzero) > 1:
                test = stats.wilcoxon(
                    nonzero,
                    alternative="two-sided",
                    method=stats.PermutationMethod(n_resamples=np.inf),
                )
                statistic, pvalue = float(test.statistic), float(test.pvalue)
            else:
                statistic, pvalue = 0.0, 1.0
            tests.append(
                {
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
            )
        gains = sum(
            not row["old_usable"] and row["new_usable"] for row in rows
        )
        losses = sum(
            row["old_usable"] and not row["new_usable"] for row in rows
        )
        pvalue = (
            float(stats.binomtest(gains, gains + losses).pvalue)
            if gains + losses
            else 1.0
        )
        tests.append(
            {
                "label": label,
                "metric": "usable",
                "method": "exact McNemar",
                "n_pairs": len(rows),
                "gains": gains,
                "losses": losses,
                "pvalue": pvalue,
            }
        )
    order = np.argsort([test["pvalue"] for test in tests])
    adjusted = 0.0
    for rank, index in enumerate(order):
        adjusted = max(
            adjusted, min(1.0, (len(tests) - rank) * tests[index]["pvalue"])
        )
        tests[index]["holm_adjusted_pvalue"] = adjusted
        tests[index]["holm_rejected"] = adjusted <= 0.05
    assert [test["holm_rejected"] for test in tests] == list(
        runner.golden_trace._holm([test["pvalue"] for test in tests], 0.05)
    )
    return tests


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


def analyze(campaign, out):
    if not __debug__:
        raise RuntimeError(
            "Run without -O: artifact validation uses assertions."
        )
    cases = campaign / "results"
    reference = runner.ROOT / "dev/golden/baseline"
    launch = json.loads((cases / "launch.json").read_text())
    finished = json.loads((cases / "finished.json").read_text())
    expected = {
        f"{c['label']}_seed{s}"
        for c in launch["manifest"]["allocation"]
        for s in c["seeds"]
    }
    current = load_rows(cases)
    baseline = load_rows(reference)
    assert set(current) == expected == set(finished["completed"])
    assert not list(cases.glob("*.error.txt"))
    reference_manifest = json.loads(
        (
            runner.ROOT
            / "dev/golden/noisy_extension_20260907/sha256_manifest.json"
        ).read_text()
    )
    for tag in baseline:
        raw = (reference / f"{tag}.json").read_bytes().replace(b"\r\n", b"\n")
        assert (
            hashlib.sha256(raw).hexdigest()
            == reference_manifest["files"][tag]["json_sha256"]
        )
    for tag in sorted(expected):
        runner.validate_case(cases, tag, launch["identity"])
    print(
        f"Validated {len(expected)} complete candidate cases and {len(baseline)} reference sidecars.",
        flush=True,
    )
    comparison, flagged = runner.golden_trace.compare_populations(
        runner.golden_trace.load_population(reference),
        runner.golden_trace.load_population(cases),
    )
    assert comparison.strip() == (cases / "comparison.md").read_text().strip()
    tests, config_rows, changes, boosts = [], {}, [], []
    labels = sorted({r["label"] for r in current.values()})
    for label in labels:
        new = [r for r in current.values() if r["label"] == label]
        all_ref = [r for r in baseline.values() if r["label"] == label]
        paired_ref = [baseline[f"{label}_seed{r['seed']}"] for r in new]
        config_rows[label] = {
            "reference_all": describe(all_ref),
            "reference_matched": describe(paired_ref),
            "candidate": describe(new),
        }
        for k in (*QUALITY, "func_count"):
            a = np.array([r["final"][k] for r in all_ref])
            b = np.array([r["final"][k] for r in new])
            test = stats.ks_2samp(a, b)
            tests.append(
                {
                    "label": label,
                    "metric": k,
                    "statistic": float(test.statistic),
                    "pvalue": float(test.pvalue),
                }
            )
    decisions = runner.golden_trace._holm([t["pvalue"] for t in tests], 0.05)
    for test, reject in zip(tests, decisions):
        test["holm_rejected"] = bool(reject)
    for tag in sorted(expected):
        new, old = current[tag]["final"], baseline[tag]["final"]
        changes.append(
            {
                "tag": tag,
                "label": current[tag]["label"],
                "old_usable": usable(old),
                "new_usable": usable(new),
                "old_converged": old["success_flag"],
                "new_converged": new["success_flag"],
                "old": {k: old[k] for k in (*QUALITY, "func_count", "wall_s")},
                "new": {k: new[k] for k in (*QUALITY, "func_count", "wall_s")},
                "delta": {
                    k: new[k] - old[k]
                    for k in (*QUALITY, "func_count", "wall_s")
                },
            }
        )
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
        boosts.append(
            {
                "tag": tag,
                "label": current[tag]["label"],
                "attempted": record["attempted"],
                "accepted": record["accepted"],
                "worst_score_change": worst_change,
                "scores": record["scores"],
                "quality": {
                    stage: {k: value[k] for k in QUALITY}
                    for stage, value in quality.items()
                },
                "usable": {
                    stage: usable(value) for stage, value in quality.items()
                },
            }
        )
    matched = [baseline[tag] for tag in expected]
    result = {
        "candidate_identity": launch["identity"],
        "verified_candidate_cases": len(expected),
        "verified_reference_sidecars": len(baseline),
        "elapsed_seconds": finished["elapsed_seconds"],
        "finished": finished["finished"],
        "comparison_reproduced_exactly": True,
        "flagged_configurations": sorted(flagged),
        "ks_tests": tests,
        "aggregate": {
            "reference_all": describe(list(baseline.values())),
            "reference_matched": describe(matched),
            "candidate": describe(list(current.values())),
        },
        "configurations": config_rows,
        "paired_changes": changes,
        "paired_tests": paired_tests(changes),
        "boosts": boosts,
        "boost_summary": {
            "attempted": sum(b["attempted"] for b in boosts),
            "accepted": sum(b["accepted"] for b in boosts),
            "rejected": sum(
                b["attempted"] and not b["accepted"] for b in boosts
            ),
            "skipped": sum(not b["attempted"] for b in boosts),
            "usable_pre": sum(b["usable"]["pre"] for b in boosts),
            "usable_candidate": sum(
                b["usable"].get("candidate", b["usable"]["pre"])
                for b in boosts
            ),
            "usable_returned": sum(b["usable"]["returned"] for b in boosts),
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    runner.write_json(out / "assessment.json", result)
    (out / "comparison.md").write_text(comparison + "\n", encoding="utf-8")
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
        [x["tag"] for x in changes if x["old_usable"] and not x["new_usable"]],
        flush=True,
    )
    print(
        "Usability gains:",
        [x["tag"] for x in changes if not x["old_usable"] and x["new_usable"]],
        flush=True,
    )
    print(
        "Rejected boosts:",
        [b["tag"] for b in boosts if b["attempted"] and not b["accepted"]],
        flush=True,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign",
        type=Path,
        default=runner.ROOT / "dev/scripts/runs/population_overnight_20260910",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=runner.ROOT / "dev/experiments/population_assessment_20260911",
    )
    args = parser.parse_args()
    analyze(args.campaign, args.out)
