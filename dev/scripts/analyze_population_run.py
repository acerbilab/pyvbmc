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
  (``population_run.checked_verification``: the report must still describe
  the directory, and every file read here must be the one it checked,
  under a completion record of the arm's identity). The two arms must share
  the allocation, the options, the confirmatory family, the harness
  checkout, the harness files (the targets module and its data among
  them) and the environment's versions, and their package trees must be
  at different commits. The comparison reads each arm's sidecars and boost
  reports and the metrics that ``population_run.py rescore`` recomputed
  with the release code, in the campaign given by ``--rescoring`` (by
  default the candidate), whose ``rescoring.json`` names the rescoring
  process's identity and the SHA-256 of each file of rescored metrics, so
  that this process imports no package but its own. The rescoring's source
  identity must be the candidate's, and the candidate's rescored metrics
  must equal its in-run metrics in every verified case. Either arm, and the
  rescoring campaign, may be a campaign directory or its tracked copies in
  the repository, redacted by ``campaign_contract.py redact``, which give
  the same report. It reports a KS screen per configuration and metric
  under one Holm family, the paired changes, the descriptive paired family
  (every configuration: signed-rank tests of the three accuracy metrics
  and the evaluation count, McNemar tests of usability, one Holm family),
  the confirmatory family fixed in both arms' manifests, at the size they
  fix (:func:`confirmatory_tests`), and every boost decision of each arm
  checked against the guard. A rescored metric that is not finite makes
  its run unusable and leaves its pair out of that metric's signed-rank
  tests, which count such pairs. The usability counts of the boost
  summaries come from each arm's in-run metrics
  (:data:`BOOST_USABILITY_BASIS`).
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


def signed_rank_test(label, metric, rows, nonfinite=False):
    """The exact signed-rank test of one metric's paired differences.

    With ``nonfinite`` the pairs whose difference is not finite (a metric
    that is not finite in one arm or both) are left out of the test and counted
    (``nonfinite_pairs``); the other counts and the median are those of the
    tested pairs. Without it a difference that is not finite raises, as the
    campaigns of one treatment against the golden reference have none.
    """
    delta = np.array([row["delta"][metric] for row in rows], dtype=float)
    finite = np.isfinite(delta)
    if nonfinite:
        delta = delta[finite]
    statistic, pvalue = exact_signed_rank(delta)
    test = {
        "label": label,
        "metric": metric,
        "method": "exact signed-rank sign permutation",
        "n_pairs": len(rows),
        "improved": int(sum(delta < 0)),
        "worsened": int(sum(delta > 0)),
        "tied": int(sum(delta == 0)),
        "median_paired_change": (
            float(np.median(delta)) if len(delta) else float("nan")
        ),
        "statistic": statistic,
        "pvalue": pvalue,
    }
    if nonfinite:
        test["nonfinite_pairs"] = int(sum(~finite))
    return test


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
    nonfinite=False,
):
    """Test within-configuration seed pairs; one Holm family when `adjust`.

    Signed-rank nulls assume symmetric, independent seed differences.
    Usability uses exact McNemar tests, conditional on the discordant pairs
    (left out when `usability` is false). `nonfinite` goes to
    :func:`signed_rank_test`.
    """
    tests = []
    for label in sorted({row["label"] for row in changes}):
        rows = [row for row in changes if row["label"] == label]
        for metric in metrics:
            tests.append(signed_rank_test(label, metric, rows, nonfinite))
        if usability:
            tests.append(mcnemar_test(label, rows))
    return holm(tests, alpha) if adjust else tests


def confirmatory_tests(changes, family, planned_pairs):
    """The confirmatory family of two arms, at the size fixed before the runs.

    ``family`` is the one the arms' manifests fix
    (``population_run.confirmatory_family``); its Holm correction runs over
    ``family["tests"]`` tests whatever the runs gave, so that a failed
    configuration cannot shrink the family and ease the rejection of the
    others. A test of a configuration without a pair of seeds verified in
    both arms, or whose paired differences are none of them finite, cannot
    be computed: it enters the family with p = 1, so that it rejects
    nothing and every other test is adjusted as in the family fixed before
    the runs, and it is flagged (``computed`` false, with the ``reason``).
    Refusing the whole comparison instead would withhold the verdict on
    every other configuration because one failed in an arm. Every test
    holds ``planned_pairs``, the seeds of the allocation, beside the
    ``n_pairs`` it had.
    """

    def not_computed(label, metric, method, reason):
        return {
            "label": label,
            "metric": metric,
            "method": method,
            "n_pairs": 0,
            "pvalue": 1.0,
            "computed": False,
            "reason": reason,
        }

    tests = []
    for label in sorted(family["labels"]):
        rows = [row for row in changes if row["label"] == label]
        for metric in family["signed_rank"]:
            if not rows:
                test = not_computed(
                    label,
                    metric,
                    "exact signed-rank sign permutation",
                    "no seed is verified in both arms",
                )
            else:
                test = signed_rank_test(label, metric, rows, nonfinite=True)
                test["computed"] = test["nonfinite_pairs"] < len(rows)
                if not test["computed"]:
                    test["reason"] = "no paired difference is finite"
            tests.append(test)
        if family["mcnemar_usability"]:
            if not rows:
                test = not_computed(
                    label,
                    "usable",
                    "exact McNemar",
                    "no seed is verified in both arms",
                )
            else:
                test = mcnemar_test(label, rows) | {"computed": True}
            tests.append(test)
    for test in tests:
        test["planned_pairs"] = planned_pairs
    assert len(tests) == family["tests"], (len(tests), family["tests"])
    return holm(tests, family["alpha"])


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


def describe(rows, nonfinite=False):
    """Counts and metric quantiles of a set of runs.

    With ``nonfinite`` the quantiles of each metric are those of its finite
    values (NaN when none is), and ``nonfinite`` counts the others; a run
    with a metric that is not finite is never usable.
    """
    finals = [r["final"] for r in rows]

    def spread(values):
        if nonfinite:
            values = [v for v in values if np.isfinite(v)]
            if not values:
                return {"median": np.nan, "q90": np.nan, "max": np.nan}
        return {
            "median": float(np.median(values)),
            "q90": float(np.quantile(values, 0.9)),
            "max": float(max(values)),
        }

    keys = (*QUALITY, "func_count", "wall_s", "peak_rss_mb")
    result = {
        "n": len(rows),
        "converged": sum(bool(r["success_flag"]) for r in finals),
        "usable": sum(usable(r) for r in finals),
        "optimizer_seconds": sum(r["wall_s"] for r in finals),
        "evaluations": sum(r["func_count"] for r in finals),
        "metrics": {k: spread([r[k] for r in finals]) for k in keys},
    }
    if nonfinite:
        result["nonfinite"] = {
            k: sum(not np.isfinite(r[k]) for r in finals) for k in keys
        }
    return result


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


def read_rescoring(directory):
    """The record of a rescoring (``population_run.RESCORING``).

    ``directory`` is the campaign whose ``rescore`` step rescored the arms,
    or its tracked copies.
    """
    path = Path(directory) / runner.RESCORING
    assert path.is_file(), f"{directory} holds no {runner.RESCORING}"
    return json.loads(path.read_text())


def load_array_campaign(path, rescoring_dir, rescoring):
    """Read one campaign of array mode, checked against its own verification.

    Its ``verification.json`` must pass
    ``population_run.checked_verification``: reconcile the manifest's
    allocation, have passed (exit code 0), name the manifest's SHA-256 and
    still describe the directory (the record of every verified case is the
    one it checked, and no other case has one). Every verified case's
    record must hold the manifest's source identity, and the sidecar and
    boost report read here must be the files that record hashes. The
    rescored metrics must be the file ``rescored/<directory name>.json`` of
    ``rescoring_dir`` whose SHA-256 ``rescoring`` (that directory's
    ``rescoring.json``) records, made by the process that record names, for
    this manifest and this verification, and from the trace, posterior
    arrays and sidecar that each case's record hashes. Cases in other states
    are counted, not read. ``path`` and ``rescoring_dir`` may also be the
    tracked copies of campaigns, redacted by ``campaign_contract.redact``:
    each file is then checked against the SHA-256 its ``redaction.json``
    records, and that of the campaign's own file, which the records and
    reports hash, is compared in its place
    (``campaign_contract.source_sha256``); the campaign's name is the one
    ``redaction.json`` records.

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
    rescoring_dir = Path(rescoring_dir).resolve()
    contract = runner.contract
    name = contract.source_name(path)
    manifest = json.loads((path / "manifest.json").read_text())
    verification = runner.checked_verification(path, manifest)
    manifest_sha256 = contract.source_sha256(path, "manifest.json")
    verification_sha256 = contract.source_sha256(path, "verification.json")
    entry = rescoring["rescored"].get(name)
    assert entry is not None, f"{rescoring_dir} did not rescore {name}"
    rel = f"rescored/{name}.json"
    assert entry["file"] == rel, (name, entry["file"])
    assert (
        contract.source_sha256(rescoring_dir, rel) == entry["sha256"]
    ), f"{rescoring_dir / rel} is not the file its rescoring wrote"
    rescored = json.loads((rescoring_dir / rel).read_text())
    assert (
        rescored["rescoring"]["identity"]["source"]
        == rescoring["identity"]["source"]
    ), f"{rescoring_dir / rel}: rescored by another process"
    assert (
        rescored["campaign"]["manifest_sha256"] == manifest_sha256
    ), f"{path}: rescored for another manifest"
    assert (
        rescored["campaign"]["verification_sha256"] == verification_sha256
    ), f"{path}: rescored for another verification"
    cases, rows, reports = {}, {}, {}
    for entry in verification["cases"]:
        tag, label, seed = runner.parse_case(entry["case"])
        stem = f"{label}_seed{seed}"
        cases[stem] = (label, seed, entry["status"])
        if entry["status"] != "verified":
            continue
        record = json.loads(contract.record_path(path, tag).read_text())
        assert record["tag"] == tag, tag
        assert not contract.source_differences(
            record["identity"], manifest["identity"]
        ), f"{tag}: the record's identity is not the manifest's"
        files = runner.case_files(label, seed)
        for key in ("sidecar", "boost_report"):
            rel = files[key]
            assert (
                contract.source_sha256(path, rel)
                == record["artifacts"][rel]["sha256"]
            ), f"{tag}: {rel} is not the file its verification checked"
        side = json.loads((path / files["sidecar"]).read_text())
        assert (side["label"], side["seed"]) == (label, seed), tag
        again = rescored["cases"][stem]
        assert again["status"] == "rescored", tag
        for key in ("trace", "posterior", "sidecar"):
            assert (
                again["artifacts"][files[key]]
                == record["artifacts"][files[key]]["sha256"]
            ), f"{tag}: rescored from another {key}"
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
        "name": name,
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
    return describe(rows, nonfinite=True) if rows else None


#: What the usability counts of the boost summaries of two arms rest on.
BOOST_USABILITY_BASIS = (
    "each arm's in-run metrics of its pre-boost, candidate and returned "
    "posteriors (its boost reports), computed by that arm's own code; the "
    "pre-boost and candidate posteriors are not rescored, so these counts "
    "are not those of the rescored metrics"
)


def analyze_arms(reference, candidate, rescoring, out):
    """Compare two arms of array mode (module docstring); write the report.

    ``rescoring`` is the campaign whose ``rescore`` step rescored both arms,
    or its tracked copies (None: the candidate). The candidate is the arm
    of the rescoring's own code: the rescoring's source identity must be
    the candidate manifest's, and every verified candidate case's rescored
    metrics must equal its in-run metrics.
    """
    if not __debug__:
        raise RuntimeError(
            "Run without -O: artifact validation uses assertions."
        )
    rescoring_dir = Path(rescoring) if rescoring else Path(candidate)
    record = read_rescoring(rescoring_dir)
    ref = load_array_campaign(reference, rescoring_dir, record)
    new = load_array_campaign(candidate, rescoring_dir, record)
    for key in ("allocation", "options", "confirmatory"):
        assert (
            ref["manifest"][key] == new["manifest"][key]
        ), f"the arms have different {key}"
    differing = runner.pair_differences(new["identity"], ref["identity"])
    assert not differing, differing
    a, b = ref["identity"]["source"], new["identity"]["source"]
    source = record["identity"]["source"]
    assert (
        source == b
    ), "the rescoring's source identity is not the candidate's: " + str(
        runner.contract.source_differences(record["identity"], new["identity"])
    )
    assert ref["rescoring"] == new["rescoring"] == source
    assert (
        source["trees"]["pyvbmc"]
        == source["trees"]["harness"]
        == a["trees"]["harness"]
    ), "not rescored by the release code of the arms' harness checkout"
    unequal = sorted(
        stem
        for stem, row in new["rows"].items()
        if not all(row["equal_to_in_run"].values())
        or not all(
            runner.same_value(row["final"][key], row["in_run"][key])
            for key in runner.RESCORED_METRICS
        )
    )
    assert not unequal, (
        "the candidate's rescored metrics differ from its in-run metrics "
        f"in {len(unequal)} cases, the first {unequal[:1]}"
    )
    family = new["manifest"]["confirmatory"]
    labels = new["manifest"]["allocation"]["labels"]
    assert (
        runner.confirmatory_family(family, labels) == family
    ), "the manifests' confirmatory family is not the one prepare fixes"
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
    confirmatory = confirmatory_tests(
        changes, family, len(new["manifest"]["allocation"]["seeds"])
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
            "rescored_nonfinite": {
                key: sum(
                    not np.isfinite(row["final"][key])
                    for row in arm["rows"].values()
                )
                for key in runner.RESCORED_METRICS
            },
        }

    result = {
        "kind": "two arms of array mode, paired by seed",
        "arms": {"reference": arm_summary(ref), "candidate": arm_summary(new)},
        "rescoring_source": source,
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
        "paired_tests": paired_tests(changes, nonfinite=True),
        "confirmatory_tests": confirmatory,
        "confirmatory_not_computed": [
            f"{t['label']} {t['metric']}"
            for t in confirmatory
            if not t["computed"]
        ],
        "boosts": boosts,
        "boost_summary": {
            role: summarize_boosts(records) for role, records in boosts.items()
        },
        "boost_usability_basis": BOOST_USABILITY_BASIS,
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
    print(
        "Boost (usability from each arm's in-run metrics):",
        result["boost_summary"],
        flush=True,
    )
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
    if result["confirmatory_not_computed"]:
        print(
            "Confirmatory tests not computed (p = 1):",
            result["confirmatory_not_computed"],
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
        "--rescoring",
        type=Path,
        help="with --arms: the campaign whose rescore step rescored both"
        " arms, or its tracked copies (default: the candidate)",
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
        analyze_arms(*args.arms, args.rescoring, args.out)
    else:
        analyze(
            args.campaign,
            args.extension,
            args.out
            or runner.ROOT / "dev/experiments/population_extension_20260911",
        )
