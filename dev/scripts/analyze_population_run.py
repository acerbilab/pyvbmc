"""Assess completed population artifacts without running inference or scoring.

Two kinds of assessment:

- **Campaigns of one treatment against the golden reference** (the default
  command line; the campaigns of September 2026, run before
  ``population_run.py`` met the campaign contract). A first-stage campaign
  and any extension campaigns of the same frozen treatment are pooled.
  Every case is revalidated against its own campaign's launch record, the
  campaigns must share one source/environment identity, and the pooled
  population is compared with the golden reference, whose sidecars are
  checked against their tracked SHA-256 manifest. Each extension is also
  reported on its own, with a confirmatory test family fixed before it ran
  (``assessment`` in its launch manifest).
- **Two arms of array mode** (``--arms REFERENCE CANDIDATE``): campaigns of
  ``population_run.py`` on one allocation, run by different code, compared
  seed by seed. Each arm is checked against its own ``verification.json``
  (every file read here must be the one its verification checked, under a
  completion record of the arm's identity), and the two arms must share the
  harness, the targets module and its data. The comparison reads each
  arm's sidecars and boost reports and the metrics that ``population_run.py
  rescore`` recomputed with the release code (``--rescored``, by default
  the candidate's ``rescored/``), so that this process imports no package
  but its own. It reports a KS screen per configuration and metric under
  one Holm family, the paired changes, the descriptive paired family (every
  configuration: signed-rank tests of the three accuracy metrics and the
  evaluation count, McNemar tests of usability, one Holm family), the
  confirmatory family fixed in both arms' manifests, and every boost
  decision of each arm checked against the guard.
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


#: The most nonzero differences whose sign-assignment counts fit in int64.
INT64_PAIRS = 62


def exact_signed_rank(delta):
    """Two-sided exact signed-rank test over all sign assignments.

    Zero differences are dropped, tied absolute differences receive midranks,
    and the null distribution of the positive-rank sum is enumerated by
    dynamic programming over the midranks, so ties are handled exactly at
    any sample size. The result equals SciPy's ``wilcoxon`` with
    ``PermutationMethod(n_resamples=np.inf)``, whose enumeration of the
    ``2**n`` sign vectors is limited to about 20 pairs. The counts of the
    ``2**m`` assignments of ``m`` nonzero differences are exact integers:
    int64 up to :data:`INT64_PAIRS` differences and Python integers beyond,
    and each tail probability is their quotient by ``2**m``, correctly
    rounded. Returns the smaller rank sum and the p-value.
    """
    d = np.asarray(delta, dtype=float)
    if not np.all(np.isfinite(d)):
        raise ValueError("The paired differences must be finite.")
    d = d[d != 0]
    m = len(d)
    if m < 2:
        return 0.0, 1.0
    ranks = stats.rankdata(np.abs(d))
    units = np.rint(2 * ranks).astype(np.int64)  # midranks are half-integers
    assert np.array_equal(units, 2 * ranks)
    total = int(units.sum())
    exact_int64 = m <= INT64_PAIRS
    counts = np.zeros(total + 1, dtype=np.int64 if exact_int64 else object)
    counts[0] = 1
    for unit in units:
        unit = int(unit)
        shifted = counts.copy()
        shifted[unit:] += counts[: total + 1 - unit]
        counts = shifted
    positive = int(units[d > 0].sum())
    if exact_int64:
        n_assignments = float(2**m)
        less = counts[: positive + 1].sum() / n_assignments
        greater = counts[positive:].sum() / n_assignments
    else:
        n_assignments = 2**m
        less = int(counts[: positive + 1].sum()) / n_assignments
        greater = int(counts[positive:].sum()) / n_assignments
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


def paired_tests(
    changes,
    metrics=(*QUALITY, "func_count"),
    adjust=True,
    usability=True,
    alpha=ALPHA,
):
    """Test within-configuration seed pairs; one Holm family when `adjust`.

    Signed-rank nulls assume symmetric, independent seed differences.
    Usability uses exact McNemar tests, conditional on the discordant pairs
    (left out when `usability` is false).
    """
    tests = []
    for label in sorted({row["label"] for row in changes}):
        rows = [row for row in changes if row["label"] == label]
        for metric in metrics:
            tests.append(signed_rank_test(label, metric, rows))
        if usability:
            tests.append(mcnemar_test(label, rows))
    return holm(tests, alpha) if adjust else tests


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


#: Pool golden-harness populations (``load_population``) by label.
merge_populations = runner.golden_trace.merge_populations


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


def boost_record(report, tag, label, new):
    """Check one boost report's decision against the guard; summarize it.

    ``report`` is the path of the ``.boost.json``, and ``new`` the run's
    in-run finals, which the report's metrics of the returned posterior
    must equal.
    """
    record = json.loads(Path(report).read_text())
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
        boosts[tag] = boost_record(
            runner.case_path(where[tag], tag, ".boost.json"), tag, label, new
        )
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


# --------------------------------------------------------------------------
# Two arms of array mode
# --------------------------------------------------------------------------


def load_array_campaign(path, rescored_dir):
    """Read one campaign of array mode, checked against its own verification.

    Its ``verification.json`` must reconcile the manifest's allocation, have
    passed (exit code 0) and name the manifest's SHA-256. Every case it
    places as verified must have a completion record whose source identity
    is the manifest's, and the sidecar and boost report read here must be
    the files that record hashes. The rescored metrics
    (``<rescored_dir>/<directory name>.json``) must be those of this
    manifest and this verification, rescored from that sidecar. Cases in
    other states are counted, not read.

    Returns
    -------
    dict
        ``name``, ``path``, ``manifest``, ``identity`` (the manifest's),
        ``cases`` (``{stem: (label, seed, status)}``), ``rows`` (the
        verified cases: ``label``, ``seed``, ``final`` with the rescored
        metrics in place of the in-run ones, ``in_run``, ``equal_to_in_run``),
        ``reports`` (the boost report of each verified case), ``counts``
        (the verification's) and ``rescoring`` (the rescoring process's
        source identity).
    """
    path = Path(path).resolve()
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    verification_path = path / "verification.json"
    verification = json.loads(verification_path.read_text())
    assert [case["case"] for case in verification["cases"]] == (
        runner.case_lines(manifest)
    ), f"{path}: the verification does not reconcile the manifest"
    assert verification["exit_code"] == 0, (path, verification["counts"])
    assert verification["manifest_sha256"] == runner.sha256(manifest_path)
    rescored = json.loads(
        (Path(rescored_dir) / f"{path.name}.json").read_text()
    )
    assert rescored["campaign"]["manifest_sha256"] == runner.sha256(
        manifest_path
    ), f"{path}: rescored for another manifest"
    assert rescored["campaign"]["verification_sha256"] == runner.sha256(
        verification_path
    ), f"{path}: rescored for another verification"
    cases, rows, reports = {}, {}, {}
    for entry in verification["cases"]:
        tag, label, seed = runner.parse_case(entry["case"])
        stem = f"{label}_seed{seed}"
        cases[stem] = (label, seed, entry["status"])
        if entry["status"] != "verified":
            continue
        record = json.loads(runner.contract.record_path(path, tag).read_text())
        assert record["tag"] == tag, tag
        assert not runner.contract.source_differences(
            record["identity"], manifest["identity"]
        ), f"{tag}: the record's identity is not the manifest's"
        files = runner.case_files(label, seed)
        for key in ("sidecar", "boost_report"):
            rel = files[key]
            assert (
                runner.sha256(path / rel) == record["artifacts"][rel]["sha256"]
            ), f"{tag}: {rel} is not the file its verification checked"
        side = json.loads((path / files["sidecar"]).read_text())
        assert (side["label"], side["seed"]) == (label, seed), tag
        again = rescored["cases"][stem]
        assert again["status"] == "rescored", tag
        assert (
            again["artifacts"][files["sidecar"]]
            == record["artifacts"][files["sidecar"]]["sha256"]
        ), f"{tag}: rescored from another sidecar"
        in_run = {key: side["final"][key] for key in runner.RESCORED_METRICS}
        rows[stem] = {
            "label": label,
            "seed": seed,
            "final": {**side["final"], **again["metrics"]},
            "in_run": in_run,
            "equal_to_in_run": again["equal_to_in_run"],
        }
        reports[stem] = path / files["boost_report"]
    return {
        "name": path.name,
        "path": path,
        "manifest": manifest,
        "identity": manifest["identity"],
        "cases": cases,
        "rows": rows,
        "reports": reports,
        "counts": verification["counts"],
        "rescoring": rescored["rescoring"]["identity"]["source"],
    }


def array_population(campaign):
    """The population of one arm (``load_population``'s form), rescored."""
    population = {}
    for stem, (label, seed, status) in campaign["cases"].items():
        entry = population.setdefault(
            label, {"seeds": [], "rows": [], "fails": 0}
        )
        entry["fails"] += status == "failed"
        if stem in campaign["rows"]:
            entry["seeds"].append(seed)
            entry["rows"].append(campaign["rows"][stem]["final"])
    return merge_populations([population])


def ks_screen(reference, candidate, alpha=ALPHA):
    """KS tests per configuration and metric, one Holm family.

    The tests of ``golden_trace.compare_populations``: its metrics, their
    finite values, three or more on each side.
    """
    tests = []
    for label in sorted(set(reference) & set(candidate)):
        for metric in runner.golden_trace.METRICS:
            x = reference[label][metric]
            y = candidate[label][metric]
            x, y = x[np.isfinite(x)], y[np.isfinite(y)]
            if len(x) < 3 or len(y) < 3:
                continue
            test = stats.ks_2samp(x, y)
            tests.append(
                {
                    "label": label,
                    "metric": metric,
                    "n_reference": len(x),
                    "n_candidate": len(y),
                    "statistic": float(test.statistic),
                    "pvalue": float(test.pvalue),
                    "median_shift": float(np.median(y) - np.median(x)),
                }
            )
    decisions = runner.golden_trace._holm([t["pvalue"] for t in tests], alpha)
    for test, reject in zip(tests, decisions):
        test["holm_rejected"] = bool(reject)
    return tests


def _describe(rows):
    return describe(rows) if rows else None


def analyze_arms(reference, candidate, rescored, out):
    """Compare two arms of array mode (module docstring); write the report."""
    if not __debug__:
        raise RuntimeError(
            "Run without -O: artifact validation uses assertions."
        )
    rescored = Path(rescored) if rescored else Path(candidate) / "rescored"
    ref = load_array_campaign(reference, rescored)
    new = load_array_campaign(candidate, rescored)
    for key in ("allocation", "options", "confirmatory"):
        assert (
            ref["manifest"][key] == new["manifest"][key]
        ), f"the arms have different {key}"
    a, b = ref["identity"]["source"], new["identity"]["source"]
    assert a["trees"]["harness"] == b["trees"]["harness"], "harness commits"
    assert a["files"] == b["files"], "the harness files differ"
    assert a["versions"] == b["versions"], "the environments differ"
    assert ref["rescoring"] == new["rescoring"], "rescored by other code"
    rescoring_trees = ref["rescoring"]["trees"]
    assert (
        rescoring_trees["pyvbmc"]
        == rescoring_trees["harness"]
        == (a["trees"]["harness"])
    ), "not rescored by the release code of the arms' harness checkout"
    family = new["manifest"]["confirmatory"]
    labels = new["manifest"]["allocation"]["labels"]
    print(
        f"Validated {len(ref['rows'])} reference and {len(new['rows'])}"
        " candidate cases against their verifications.",
        flush=True,
    )

    ref_population = array_population(ref)
    new_population = array_population(new)
    comparison, flagged = runner.golden_trace.compare_populations(
        ref_population, new_population
    )
    ks = ks_screen(ref_population, new_population)
    assert {t["label"] for t in ks if t["holm_rejected"]} == flagged

    paired = sorted(set(ref["rows"]) & set(new["rows"]))
    changes = [
        paired_change(
            stem,
            new["rows"][stem]["label"],
            new["rows"][stem]["final"],
            ref["rows"][stem]["final"],
        )
        for stem in paired
    ]
    confirmatory = paired_tests(
        [c for c in changes if c["label"] in family["labels"]],
        metrics=family["signed_rank"],
        usability=family["mcnemar_usability"],
        alpha=family["alpha"],
    )
    boosts = {
        role: [
            boost_record(
                arm["reports"][stem],
                stem,
                arm["rows"][stem]["label"],
                {**arm["rows"][stem]["final"], **arm["rows"][stem]["in_run"]},
            )
            for stem in sorted(arm["rows"])
        ]
        for role, arm in (("reference", ref), ("candidate", new))
    }

    def arm_summary(arm):
        return {
            "name": arm["name"],
            "arm": arm["manifest"].get("arm"),
            "source": arm["identity"]["source"],
            "verification_counts": arm["counts"],
            "verified": len(arm["rows"]),
            "rescored_equal_to_in_run": {
                key: sum(
                    bool(row["equal_to_in_run"][key])
                    for row in arm["rows"].values()
                )
                for key in runner.RESCORED_METRICS
            },
        }

    result = {
        "kind": "two arms of array mode, paired by seed",
        "arms": {"reference": arm_summary(ref), "candidate": arm_summary(new)},
        "rescoring_source": ref["rescoring"],
        "allocation": new["manifest"]["allocation"],
        "options": new["manifest"]["options"],
        "confirmatory_family": family,
        "paired_cases": len(paired),
        "flagged_configurations": sorted(flagged),
        "ks_tests": ks,
        "aggregate": {
            "reference": _describe(list(ref["rows"].values())),
            "candidate": _describe(list(new["rows"].values())),
        },
        "configurations": {
            label: {
                role: _describe(
                    [r for r in arm["rows"].values() if r["label"] == label]
                )
                for role, arm in (("reference", ref), ("candidate", new))
            }
            for label in labels
        },
        "paired_changes": changes,
        "paired_tests": paired_tests(changes),
        "confirmatory_tests": confirmatory,
        "boosts": boosts,
        "boost_summary": {
            role: summarize_boosts(records) for role, records in boosts.items()
        },
        "usability_losses": [
            c["tag"]
            for c in changes
            if c["old_usable"] and not c["new_usable"]
        ],
        "usability_gains": [
            c["tag"]
            for c in changes
            if not c["old_usable"] and c["new_usable"]
        ],
    }
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    runner.write_json(out / "assessment.json", result)
    (out / "comparison.md").write_text(
        f"# Compare {ref['name']} (reference) vs {new['name']} (candidate),"
        " rescored metrics\n\n" + comparison + "\n",
        encoding="utf-8",
    )
    print(
        "Aggregate:",
        json.dumps(
            {
                role: {
                    m: value[m]
                    for m in ("n", "converged", "usable", "optimizer_seconds")
                }
                for role, value in result["aggregate"].items()
                if value
            }
        ),
        flush=True,
    )
    print("Flagged by the KS screen:", sorted(flagged), flush=True)
    print("Boost:", result["boost_summary"], flush=True)
    print("Usability losses:", result["usability_losses"], flush=True)
    print("Usability gains:", result["usability_gains"], flush=True)
    rejected = [t for t in confirmatory if t["holm_rejected"]]
    print(
        f"Confirmatory family: {len(confirmatory)} tests, "
        f"{len(rejected)} rejected"
        + (
            ": " + ", ".join(f"{t['label']} {t['metric']}" for t in rejected)
            if rejected
            else ""
        ),
        flush=True,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
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
        "--arms",
        type=Path,
        nargs=2,
        metavar=("REFERENCE", "CANDIDATE"),
        help="compare two arms of array mode instead, seed by seed",
    )
    parser.add_argument(
        "--rescored",
        type=Path,
        help="with --arms: the directory of the rescored metrics"
        " (default: the candidate's rescored/)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="where the report goes (default for the campaigns of one"
        " treatment: dev/experiments/population_extension_20260911)",
    )
    args = parser.parse_args()
    if args.arms:
        if args.out is None:
            parser.error("--arms needs --out")
        analyze_arms(*args.arms, args.rescored, args.out)
    else:
        analyze(
            args.campaign,
            args.extension,
            args.out
            or runner.ROOT / "dev/experiments/population_extension_20260911",
        )
