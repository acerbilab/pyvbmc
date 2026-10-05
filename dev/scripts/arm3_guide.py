"""Read the release gate's Arm 3 with the guide fixed before its runs.

``dev/plans/arm-3-warmup-comparison.md`` holds the guide: its decisions 5
and 6 and the section "The comparisons". This script applies the guide's
tests to the batch's outputs and writes, beside each test's inputs and
numbers, what the guide says about the end of warm-up for noisy and for
noiseless targets and about the S-VBMC ELBO headline. The PI reads the
results with it and is not bound by it::

    python dev/scripts/arm3_guide.py --out DIR \\
        [--population DIR] [--fresh DIR] \\
        [--pools-reference DIR --pools-candidate DIR] \\
        [--stacking FILE] [--added FILE]

- ``--population``: the ``--out`` of ``analyze_population_run.py --arms``
  with the after arm as the reference and ``population_arm3`` as the
  candidate (its ``assessment.json``);
- ``--fresh``: the same for ``fresh_release`` and ``fresh_arm3``;
- ``--pools-reference``, ``--pools-candidate``: the after arm's pools and
  Arm 3's, campaign directories restored from their archives, whose
  completion records hold each run's metrics (their tracked copies do not);
- ``--stacking``: the ``summary.md`` of Arm 3's stacking;
- ``--added``: ``single_run/added.md`` of Arm 3's analyses.

Each part of the guide reads its own inputs alone and is reported as not
read without them, so a part can be read as its results arrive. Nothing is
written but ``DIR/guide.json`` and ``DIR/guide.md``.

Directions: every paired change is the candidate's value less the
reference's, so a positive change of an error metric is Arm 3's loss. The
direction of a signed-rank test is that of its rank sums, the positive
changes' against the negative ones' (zeros left out, ties at their mean
rank), not the median's; the analysis's ``statistic`` is the smaller of
the two sums and gives none. A McNemar test of usability or of the pools'
filters finds Arm 3 worse when it loses more runs than it gains.
"""

import argparse
import json
import math
import re
from pathlib import Path

import analyze_population_run as analysis
import benchmark_targets
import numpy as np
from scipy import stats

ALPHA = 0.05
FRESH_LABEL = "rosenbrock_D2_noise3_production"
#: The accuracy metrics of every signed-rank test of the guide.
METRICS = ("elbo_err", "gskl", "mmtv")
#: Decision 6's bounds, included, on the medians as the analysis prints them.
NOISY_ADDED = (-0.25, 0.20)
NOISELESS_ADDED = (-0.15, 0.15)
STUDENT = "student_D8_noise3_svbmc"
#: Criterion 1 of the stacking, read as the release pools' stacking was.
MAX_DW = 0.03


def noisy(label):
    """Whether the targets module gives ``label`` a noise level, which is
    where VBMC's noise handling is on."""
    return benchmark_targets.find_config(label).noise_sd is not None


# --------------------------------------------------------------------------
# Tests and their directions
# --------------------------------------------------------------------------


def rank_sums(deltas):
    """The rank sums of the positive and of the negative changes among the
    finite, nonzero ones, ranked by absolute value with ties at their mean
    rank, as the exact signed-rank test ranks them."""
    d = np.asarray([x for x in deltas if math.isfinite(x) and x != 0])
    if not len(d):
        return 0.0, 0.0
    ranks = stats.rankdata(np.abs(d))
    return float(ranks[d > 0].sum()), float(ranks[d < 0].sum())


def direction_of(deltas):
    r_plus, r_minus = rank_sums(deltas)
    if r_plus > r_minus:
        return "worse"
    if r_plus < r_minus:
        return "better"
    return "neither"


def signed_rank(metric, deltas):
    """The exact two-sided signed-rank test of one metric's paired changes,
    the pairs with a change that is not finite left out and counted; a test
    with no nonzero change is not computed and enters its family at p = 1."""
    finite = [x for x in deltas if math.isfinite(x)]
    test = {
        "metric": metric,
        "n_pairs": len(finite),
        "nonfinite_pairs": len(deltas) - len(finite),
        "rank_sums": rank_sums(finite),
        "direction": direction_of(finite),
    }
    if not any(x != 0 for x in finite):
        return {**test, "computed": False, "pvalue": 1.0}
    _, pvalue = analysis.exact_signed_rank(np.asarray(finite, dtype=float))
    return {**test, "computed": True, "pvalue": float(pvalue)}


def mcnemar(metric, gains, losses):
    pvalue = (
        float(stats.binomtest(gains, gains + losses).pvalue)
        if gains + losses
        else 1.0
    )
    if losses > gains:
        direction = "worse"
    elif gains > losses:
        direction = "better"
    else:
        direction = "neither"
    return {
        "metric": metric,
        "gains": int(gains),
        "losses": int(losses),
        "computed": bool(gains + losses),
        "direction": direction,
        "pvalue": pvalue,
    }


def judged(tests):
    """One family under Holm at :data:`ALPHA`: worse when Holm rejects a
    test that finds Arm 3 worse, better when it rejects one that finds it
    better (both can hold, on different metrics)."""
    analysis.holm(tests, ALPHA)
    rejected = [t for t in tests if t["holm_rejected"]]
    return {
        "tests": tests,
        "worse": any(t["direction"] == "worse" for t in rejected),
        "better": any(t["direction"] == "better" for t in rejected),
    }


# --------------------------------------------------------------------------
# The populations: each configuration's family, and the fresh seeds
# --------------------------------------------------------------------------


def read_assessment(directory):
    path = Path(directory) / "assessment.json"
    assessment = json.loads(path.read_text(encoding="utf-8"))
    assert (
        assessment.get("kind") == "two arms of array mode, paired by seed"
    ), f"{path} is no assessment of two arms"
    return assessment


def deltas_of(assessment, label, metric):
    return [
        float(row["delta"][metric])
        for row in assessment["paired_changes"]
        if row["label"] == label
    ]


def for_the_pi(assessment):
    """What the tests cannot see: cases in another state than verified, the
    tests the analysis could not compute, and rescored metrics that are not
    finite."""
    items = []
    for role, arm in assessment["arms"].items():
        counts = {
            k: v
            for k, v in arm["verification_counts"].items()
            if k != "verified" and v
        }
        if counts:
            items.append(f"the {role} ({arm['name']}) has cases {counts}")
        nonfinite = {k: v for k, v in arm["rescored_nonfinite"].items() if v}
        if nonfinite:
            items.append(
                f"the {role} ({arm['name']}) has rescored metrics that are "
                f"not finite: {nonfinite}"
            )
    for test in assessment.get("confirmatory_not_computed") or []:
        items.append(f"a confirmatory test was not computed: {test}")
    return items


def population_part(assessment):
    """Each configuration's tests of the confirmatory family, under Holm
    within the configuration (decision 5)."""
    labels = assessment["allocation"]["labels"]
    configurations = {}
    for label in labels:
        tests = []
        for test in assessment["confirmatory_tests"]:
            if test["label"] != label:
                continue
            if test["metric"] == "usable":
                entry = mcnemar(
                    "usable", test.get("gains", 0), test.get("losses", 0)
                )
                entry["pvalue"] = float(test["pvalue"])
            else:
                entry = {
                    "metric": test["metric"],
                    "n_pairs": test.get("n_pairs", 0),
                    "rank_sums": rank_sums(
                        deltas_of(assessment, label, test["metric"])
                    ),
                    "direction": direction_of(
                        deltas_of(assessment, label, test["metric"])
                    ),
                    "pvalue": float(test["pvalue"]),
                }
            entry["computed"] = bool(test.get("computed", True))
            if not entry["computed"]:
                entry["pvalue"] = 1.0
            tests.append(entry)
        assert len(tests) == 4, (label, len(tests))
        configurations[label] = {"noisy": noisy(label), **judged(tests)}
    return {
        "configurations": configurations,
        "noisy_worse": sorted(
            k for k, c in configurations.items() if c["noisy"] and c["worse"]
        ),
        "noiseless_worse": sorted(
            k
            for k, c in configurations.items()
            if not c["noisy"] and c["worse"]
        ),
        "noiseless_better": sorted(
            k
            for k, c in configurations.items()
            if not c["noisy"] and c["better"]
        ),
        "for_the_pi": for_the_pi(assessment),
    }


def fresh_part(assessment):
    """The fresh seeds' main test: Arm 3's MMTV lower by the one-sided exact
    signed-rank test, and its usability not worse by the McNemar test, each
    at :data:`ALPHA` (decision 5)."""
    assert assessment["allocation"]["labels"] == [FRESH_LABEL], (
        "the fresh seeds' assessment holds "
        f"{assessment['allocation']['labels']}, not {FRESH_LABEL} alone"
    )
    deltas = deltas_of(assessment, FRESH_LABEL, "mmtv")
    mmtv = signed_rank("mmtv", deltas)
    reported = next(
        t for t in assessment["confirmatory_tests"] if t["metric"] == "mmtv"
    )
    assert math.isclose(mmtv["pvalue"], reported["pvalue"], rel_tol=1e-9), (
        "the two-sided p-value differs from the analysis's",
        mmtv["pvalue"],
        reported["pvalue"],
    )
    # The null distribution is symmetric, so the one-sided p-value in the
    # direction of the changes is half the two-sided one.
    favourable = mmtv["direction"] == "better"
    mmtv["one_sided_pvalue"] = mmtv["pvalue"] / 2 if favourable else None
    lower = bool(favourable and mmtv["pvalue"] / 2 <= ALPHA)
    usable = next(
        t for t in assessment["confirmatory_tests"] if t["metric"] == "usable"
    )
    usability = mcnemar("usable", usable["gains"], usable["losses"])
    usability["pvalue"] = float(usable["pvalue"])
    usability_worse = bool(
        usability["direction"] == "worse" and usability["pvalue"] <= ALPHA
    )
    return {
        "mmtv": mmtv,
        "usability": usability,
        "mmtv_lower": lower,
        "usability_worse": usability_worse,
        "passes": lower and not usability_worse,
        "for_the_pi": for_the_pi(assessment),
    }


# --------------------------------------------------------------------------
# The pools
# --------------------------------------------------------------------------


def pool_runs(directory):
    """The verified runs of a pool campaign: ``{(label, seed): record}``,
    each record its completion record (``svbmc_pool_run.read_record``)."""
    directory = Path(directory)
    report = json.loads(
        (directory / "verification.json").read_text(encoding="utf-8")
    )
    runs = {}
    for case in report["cases"]:
        if case["status"] != "verified":
            continue
        path = directory / "records" / f"{case['tag']}.complete.json"
        runs[(case["label"], int(case["seed"]))] = json.loads(
            path.read_text(encoding="utf-8")
        )
    return report, runs


def pools_part(reference_dir, candidate_dir):
    """Each condition's runs paired by seed over the seeds both pools ran
    and verified: the exact signed-rank tests of the accuracy metrics and
    the McNemar test of the filters, under Holm within the condition."""
    reports, pools = zip(
        *(pool_runs(d) for d in (reference_dir, candidate_dir))
    )
    items = []
    for name, report in zip(("reference", "candidate"), reports):
        if report.get("exit_code") != 0:
            items.append(f"the {name} pool's verification did not pass")
        counts = {
            k: v for k, v in report["counts"].items() if k != "verified" and v
        }
        if counts:
            items.append(f"the {name} pool has cases {counts}")
    ref, new = pools
    labels = sorted({label for label, _ in ref} | {label for label, _ in new})
    conditions = {}
    for label in labels:
        seeds = sorted(
            {s for l, s in ref if l == label}
            & {s for l, s in new if l == label}
        )
        pairs = [(ref[(label, s)], new[(label, s)]) for s in seeds]
        tests = [
            signed_rank(
                metric,
                [
                    float(b["metrics"][metric]) - float(a["metrics"][metric])
                    for a, b in pairs
                ],
            )
            for metric in METRICS
        ]
        gains = sum(
            not a["verdict"]["passes"] and b["verdict"]["passes"]
            for a, b in pairs
        )
        losses = sum(
            a["verdict"]["passes"] and not b["verdict"]["passes"]
            for a, b in pairs
        )
        tests.append(mcnemar("passes", gains, losses))
        conditions[label] = {
            "noisy": noisy(label),
            "paired_seeds": len(seeds),
            **judged(tests),
        }
        if any(t["nonfinite_pairs"] for t in tests if "nonfinite_pairs" in t):
            items.append(
                f"{label}: pairs left out for a metric that is not finite, "
                + str(
                    {
                        t["metric"]: t["nonfinite_pairs"]
                        for t in tests
                        if t.get("nonfinite_pairs")
                    }
                )
            )
    return {
        "conditions": conditions,
        "noisy_worse": sorted(
            k for k, c in conditions.items() if c["noisy"] and c["worse"]
        ),
        "noiseless_worse": sorted(
            k for k, c in conditions.items() if not c["noisy"] and c["worse"]
        ),
        "for_the_pi": items,
    }


# --------------------------------------------------------------------------
# The stacking and the S-VBMC headline
# --------------------------------------------------------------------------


def sections(text):
    """The ``## label`` sections of a summary: ``{label: lines}``."""
    found, label = {}, None
    for line in text.splitlines():
        if line.startswith("## "):
            label = line[3:].strip()
            found[label] = []
        elif label is not None:
            found[label].append(line)
    return found


def tables(lines):
    """The Markdown tables of a section, each a list of ``{column: cell}``."""
    found, header = [], None
    for line in lines:
        if not line.startswith("|"):
            header = None
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if header is None:
            header = cells
            found.append([])
        elif set("".join(cells)) <= set("-: "):
            continue
        else:
            found[-1].append(dict(zip(header, cells)))
    return found


def first_number(cell):
    """The value a cell prints before its interval, or None for ``-``."""
    match = re.match(r"\s*([+-]?\d+(?:\.\d+)?)", cell)
    return float(match.group(1)) if match else None


ALL_CELLS = re.compile(
    r"runtime ratio ([\d.]+) \[([\d.]+), ([\d.]+)\], max\\\|dw\\\| ([\d.]+)"
)


def stacking_part(text):
    """The campaign's criteria on each condition of Arm 3's stacking, read
    as the release pools' stacking was ("The comparisons"): criterion 1 by
    the median of each condition's largest weight difference, criterion 2
    by its flagged cells, criterion 3 by its gate at each ``M`` where both
    arms ran, criterion 4 by the runtime ratio and its interval."""
    conditions = {}
    for label, lines in sections(text).items():
        match = next(
            (ALL_CELLS.search(l) for l in lines if ALL_CELLS.search(l)), None
        )
        assert match, f"{label}: no line of all its cells"
        ratio, low, high, max_dw = (float(g) for g in match.groups())
        found = tables(lines)
        gates = [
            (row["M"], row["not worse"].strip("*"))
            for table in found
            for row in table
            if "not worse" in row
        ]
        flagged = [
            (row["M"], row["flagged"].strip("*"))
            for table in found
            for row in table
            if "flagged" in row
        ]
        assert gates and flagged, f"{label}: tables not found"
        criteria = {
            "1": max_dw <= MAX_DW,
            "2": all(f in ("no", "-") for _, f in flagged),
            "3": all(g in ("yes", "-") for _, g in gates),
            "4": ratio < 1 and high < 1,
        }
        conditions[label] = {
            "noisy": noisy(label),
            "max_dw": max_dw,
            "runtime_ratio": [ratio, low, high],
            "criterion_3_gates": gates,
            "flagged": flagged,
            "criteria": criteria,
            # Criterion 3 fails on Student D8 in every pool set so far,
            # which the guide expects.
            "meets": all(v for k, v in criteria.items() if k != "3")
            and (criteria["3"] or label == STUDENT),
        }
    return {
        "conditions": conditions,
        "noisy_meets": all(
            c["meets"] for c in conditions.values() if c["noisy"]
        ),
        "noiseless_meets": all(
            c["meets"] for c in conditions.values() if not c["noisy"]
        ),
    }


def headline_part(text):
    """Decision 6 on each condition and ``M`` of ``single_run/added.md``."""
    rows = {}
    for label, lines in sections(text).items():
        found = tables(lines)
        (table,) = [t for t in found if t and "two_level_full added" in t[0]]
        rows[label] = [
            {
                "M": int(row["M"]),
                "two_level_full": first_number(row["two_level_full added"]),
                "capped": first_number(row["capped_I_median added"]),
                "raw": first_number(row["raw added [CI]"]),
            }
            for row in table
        ]
    low, high = NOISY_ADDED
    noisy_rows = {k: v for k, v in rows.items() if noisy(k)}
    noiseless_rows = {k: v for k, v in rows.items() if not noisy(k)}
    outside = [
        (k, r["M"], r["two_level_full"])
        for k, v in noisy_rows.items()
        for r in v
        if not low <= r["two_level_full"] <= high
    ]
    student = [
        (r["M"], r["two_level_full"], r["capped"])
        for r in rows.get(STUDENT, [])
        if not abs(r["two_level_full"]) < abs(r["capped"])
    ]
    low, high = NOISELESS_ADDED
    raw_outside = [
        (k, r["M"], r["raw"])
        for k, v in noiseless_rows.items()
        for r in v
        if not low <= r["raw"] <= high
    ]
    return {
        "rows": rows,
        "two_level_outside": outside,
        "student_not_closer": student,
        "student_read": STUDENT in rows,
        "raw_outside": raw_outside,
        "noisy_switches": not outside and not student and STUDENT in rows,
        "noiseless_keeps_raw": not raw_outside,
    }


# --------------------------------------------------------------------------
# What the guide says
# --------------------------------------------------------------------------


def verdicts(parts):
    """The guide's reading of the parts read; a part not read leaves its
    clauses open."""

    def clause(part, key, good):
        if part not in parts:
            return None
        return parts[part][key] == good

    noisy_clauses = {
        "fresh seeds: MMTV lower, usability not worse": clause(
            "fresh", "passes", True
        ),
        "no noisy configuration worse": clause(
            "population", "noisy_worse", []
        ),
        "no noisy pool condition worse": clause("pools", "noisy_worse", []),
        "stacking meets the criteria on noisy conditions": clause(
            "stacking", "noisy_meets", True
        ),
    }
    noiseless_clauses = {
        "some noiseless configuration better": (
            None
            if "population" not in parts
            else bool(parts["population"]["noiseless_better"])
        ),
        "no noiseless configuration worse": clause(
            "population", "noiseless_worse", []
        ),
        "no noiseless pool condition worse": clause(
            "pools", "noiseless_worse", []
        ),
        "stacking meets the criteria on noiseless conditions": clause(
            "stacking", "noiseless_meets", True
        ),
    }

    def reading(clauses, yes, no):
        values = list(clauses.values())
        if any(v is False for v in values):
            return no
        if all(v is True for v in values):
            return yes
        return "open: not every part is read"

    said = {
        "warm_up_noisy": reading(
            noisy_clauses,
            "the port's end of warm-up",
            "MATLAB's end of warm-up",
        ),
        "warm_up_noiseless": reading(
            noiseless_clauses,
            "the port's end of warm-up",
            "MATLAB's end of warm-up",
        ),
        "noisy_clauses": noisy_clauses,
        "noiseless_clauses": noiseless_clauses,
    }
    if "headline" in parts:
        headline = parts["headline"]
        said["headline_noisy"] = (
            "the two-level shrinkage estimate"
            if headline["noisy_switches"]
            else "unchanged, the capped estimate, and the reading to the PI"
        )
        said["headline_noiseless"] = (
            "the raw ELBO, as now"
            if headline["noiseless_keeps_raw"]
            else "unchanged, the raw ELBO, and the reading to the PI"
        )
    return said


def report_md(parts, said):
    lines = [
        "# Arm 3: what the guide says",
        "",
        "The guide of `dev/plans/arm-3-warmup-comparison.md` (decisions 5 and "
        '6, "The comparisons") applied to the batch\'s outputs by '
        "`dev/scripts/arm3_guide.py`. The PI reads the results with it and "
        "is not bound by it.",
        "",
        f"- Noisy targets: **{said['warm_up_noisy']}**.",
        f"- Noiseless targets: **{said['warm_up_noiseless']}**.",
    ]
    if "headline_noisy" in said:
        lines += [
            f"- Headline of noisy stacks: **{said['headline_noisy']}**.",
            "- Headline of noiseless stacks: "
            f"**{said['headline_noiseless']}**.",
        ]
    for name, clauses in (
        ("noisy", said["noisy_clauses"]),
        ("noiseless", said["noiseless_clauses"]),
    ):
        lines += ["", f"## The clauses for {name} targets", ""]
        for text, value in clauses.items():
            mark = {True: "holds", False: "fails", None: "not read"}[value]
            lines.append(f"- {text}: {mark}")
    for part in ("population", "pools"):
        if part not in parts:
            continue
        key = "configurations" if part == "population" else "conditions"
        lines += [
            "",
            f"## {part.capitalize()}: each {key[:-1]} under Holm within it",
            "",
            "| | noisy | test | direction | p | Holm p | rejected |",
            "|---|---|---|---|---|---|---|",
        ]
        for label, entry in parts[part][key].items():
            for test in entry["tests"]:
                lines.append(
                    f"| {label} | {'yes' if entry['noisy'] else 'no'} | "
                    f"{test['metric']} | {test['direction']} | "
                    f"{test['pvalue']:.4g} | "
                    f"{test['holm_adjusted_pvalue']:.4g} | "
                    f"{'**yes**' if test['holm_rejected'] else 'no'} |"
                )
    if "fresh" in parts:
        fresh = parts["fresh"]
        one = fresh["mmtv"]["one_sided_pvalue"]
        lines += [
            "",
            "## The fresh seeds",
            "",
            "- MMTV: rank sums (positive, negative) "
            f"{fresh['mmtv']['rank_sums']}, "
            f"direction {fresh['mmtv']['direction']}, two-sided p "
            f"{fresh['mmtv']['pvalue']:.4g}, one-sided p "
            + (
                f"{one:.4g}"
                if one is not None
                else "none (not in Arm 3's favour)"
            ),
            f"- Usability: gains {fresh['usability']['gains']}, losses "
            f"{fresh['usability']['losses']}, "
            f"p {fresh['usability']['pvalue']:.4g}",
        ]
    if "stacking" in parts:
        lines += [
            "",
            "## The stacking's criteria",
            "",
            "| condition | noisy | max\\|dw\\| | ratio [CI] "
            "| 1 | 2 | 3 | 4 | meets |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for label, c in parts["stacking"]["conditions"].items():
            r = c["runtime_ratio"]
            marks = " | ".join(
                "yes" if c["criteria"][k] else "**no**" for k in "1234"
            )
            lines.append(
                f"| {label} | {'yes' if c['noisy'] else 'no'} | "
                f"{c['max_dw']:.4f} | {r[0]:.3f} [{r[1]:.3f}, {r[2]:.3f}] | "
                f"{marks} | {'yes' if c['meets'] else '**no**'} |"
            )
    if "headline" in parts:
        h = parts["headline"]
        lines += [
            "",
            "## The S-VBMC headline (decision 6)",
            "",
            f"- Two-level estimate outside {list(NOISY_ADDED)} on noisy "
            f"conditions: {h['two_level_outside'] or 'none'}",
            f"- {STUDENT}, where the two-level estimate is not closer to zero "
            f"than the capped one: "
            + (
                str(h["student_not_closer"] or "none")
                if h["student_read"]
                else "not in the table"
            ),
            f"- Raw ELBO outside {list(NOISELESS_ADDED)} on noiseless "
            f"conditions: {h['raw_outside'] or 'none'}",
        ]
    items = [i for p in parts.values() for i in p.get("for_the_pi", [])]
    lines += ["", "## For the PI before the guide is read", ""]
    lines += [f"- {i}" for i in items] or ["- nothing"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--population", type=Path)
    parser.add_argument("--fresh", type=Path)
    parser.add_argument("--pools-reference", type=Path)
    parser.add_argument("--pools-candidate", type=Path)
    parser.add_argument("--stacking", type=Path)
    parser.add_argument("--added", type=Path)
    args = parser.parse_args(argv)
    if bool(args.pools_reference) != bool(args.pools_candidate):
        parser.error(
            "the pools need both --pools-reference and --pools-candidate"
        )
    parts = {}
    if args.population:
        parts["population"] = population_part(read_assessment(args.population))
    if args.fresh:
        parts["fresh"] = fresh_part(read_assessment(args.fresh))
    if args.pools_reference:
        parts["pools"] = pools_part(args.pools_reference, args.pools_candidate)
    if args.stacking:
        parts["stacking"] = stacking_part(
            args.stacking.read_text(encoding="utf-8")
        )
    if args.added:
        parts["headline"] = headline_part(
            args.added.read_text(encoding="utf-8")
        )
    said = verdicts(parts)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "guide.json").write_text(
        json.dumps({"parts": parts, "said": said}, indent=1, default=str),
        encoding="utf-8",
    )
    (args.out / "guide.md").write_text(
        report_md(parts, said), encoding="utf-8"
    )
    print(f"Noisy targets: {said['warm_up_noisy']}", flush=True)
    print(f"Noiseless targets: {said['warm_up_noiseless']}", flush=True)
    for key in ("headline_noisy", "headline_noiseless"):
        if key in said:
            print(f"{key}: {said[key]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
