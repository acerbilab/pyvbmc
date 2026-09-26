"""Contracts of the S-VBMC stacking comparison, in one process and split.

The comparison runs on the two upstream fixture groups instead of a pool
(the harness's ``--fixtures`` mode), with two and three posteriors per cell
and three Adam steps, three times for the whole module: in one process
with both arms at every ``M`` (``comparison``); in one process with both
arms at ``M = 2`` and the integrated arm alone at ``M = 3`` (``mixed``);
and as a campaign of the Slurm driver's contract with the arms of
``mixed``, split into tasks that each run in a worker process of their own
and assembled afterwards (``campaign``). The point is that both
implementations run, that every cell is paired, that the outputs carry
what the campaign's analysis reads, and that the campaign assembles what
the single process computes, not the quality of a three-step fit. Copies
of the finished campaign take it through the contract's other states: a
task that is already complete, one that a live task holds, one that a
signal stops, one whose task was killed outright, and a line that is not
a case; one, given a stand-in site's details, has its tracked copies
redacted. Two two-run pools of the
campaign contract's layout, of the smoke configuration and of the
noiseless GMM condition, generated, verified and selected by
``svbmc_pool_run.py``, check the reading of pools, in either layout, the
gpyreg pin, what ``prepare`` requires of a pool, a campaign on both pools
assembled against the single-process run on them, and the four scripts
that read a pool and that run's cells. The rest needs no cell: the
subsets, the summaries rebuilt from recorded results, the verification of
a recreated baseline and the error of an original arm that died.

Outside default pytest discovery; run it by path, with the machine's gpyreg
checkout and original S-VBMC baseline named::

    PYVBMC_GPYREG_SOURCE=<gpyreg> BASELINE_DIR=<baseline> \\
        python -m pytest dev/scripts/test_svbmc_pool_stack.py -vv

The module skips when either variable is unset; the campaign environment
sets both. Torch comes from the environment, and where the environment
holds none, from the overlay ``<BASELINE_DIR>/deps`` that the baseline
record's recreation installs, which this module puts on the harness's path
itself. Without either, the tests fail.
"""

import copy
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRIPT = HERE / "svbmc_pool_stack.py"
GPYREG = os.environ.get("PYVBMC_GPYREG_SOURCE")
BASELINE = os.environ.get("BASELINE_DIR")
GPYREG_SOURCE = Path(GPYREG).resolve() if GPYREG else None
GROUPS = ("upstream_GMM_noisy", "upstream_Ring")
GRID = (2, 3)
REPETITIONS = (2, 2)
MAX_STEPS = 3
POOL_SIZE = 10  # posteriors per upstream fixture group
#: The arms of ``mixed`` and of the campaign, one per M of ``GRID``.
MIXED_ARMS = ("both", "integrated")

pytestmark = pytest.mark.skipif(
    not (GPYREG and BASELINE),
    reason="PYVBMC_GPYREG_SOURCE and BASELINE_DIR name the gpyreg checkout "
    "and the original S-VBMC baseline, which are machine-local",
)


def _overlay():
    """Where the harness finds Torch: None when the environment holds it,
    and otherwise the overlay ``<BASELINE_DIR>/deps``, when it holds it."""
    if importlib.util.find_spec("torch") is not None or not BASELINE:
        return None
    overlay = Path(BASELINE) / "deps"
    return overlay if (overlay / "torch" / "__init__.py").exists() else None


OVERLAY = _overlay()
# Some checks call the integrated class in this process, so the overlay's
# Torch must be importable here whether or not pytest was started with it
# on PYTHONPATH. Appended, not prepended, so that nothing the environment
# provides is shadowed.
if OVERLAY is not None and str(OVERLAY) not in sys.path:
    sys.path.append(str(OVERLAY))

import campaign_contract as contract  # noqa: E402
import campaign_slurm_stubs as stubs  # noqa: E402
import svbmc_pool_stack as harness  # noqa: E402

# Loose bound on the paired weight difference: Phase 1 of the campaign
# measured at most 0.009 on these groups at three Adam steps against a
# within-arm seed-to-seed spread of the same size, so anything of this
# order means both arms optimized the same objective on the same inputs.
WEIGHT_TOLERANCE = 0.05


def harness_environment(drop=()):
    """The environment every invocation of the harness is given here."""
    environment = dict(os.environ)
    if OVERLAY is not None:
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(OVERLAY), *[p for p in [environment.get("PYTHONPATH")] if p]]
        )
    environment["PYVBMC_GPYREG_SOURCE"] = str(GPYREG_SOURCE)
    environment["BASELINE_DIR"] = str(BASELINE)
    environment["MPLBACKEND"] = "Agg"
    for key in harness.THREAD_KEYS:
        environment[key] = "1"
    for key in drop:
        environment.pop(key, None)
    return environment


def run_script(*arguments, env=None):
    return subprocess.run(
        [sys.executable, "-u", str(SCRIPT), *arguments],
        cwd=str(ROOT),
        env=harness_environment() if env is None else env,
        capture_output=True,
        text=True,
    )


def dateless(text):
    """The lines of a summary that do not carry the date it was built."""
    return [
        line for line in text.splitlines() if not line.startswith("Generated ")
    ]


def untimed(value):
    """A record without what depends on when and how long it ran: every
    ``*_seconds`` field, the runtime ratios, the dates."""
    if isinstance(value, dict):
        return {
            key: untimed(item)
            for key, item in value.items()
            if not key.endswith("_seconds")
            and key not in ("runtime_ratio", "generated")
        }
    if isinstance(value, list):
        return [untimed(item) for item in value]
    return value


def canonical(value):
    """``untimed(value)`` as text, so that a NaN equals a NaN."""
    return json.dumps(untimed(value), sort_keys=True)


def read_outputs(out):
    return {
        name: json.loads((out / f"{name}.json").read_text(encoding="utf-8"))
        for name in ("results", "summary", "sources")
    }


def run_harness(out, *extra):
    return run_script(
        "--fixtures",
        ",".join(GROUPS),
        "--out",
        str(out),
        "--M",
        ",".join(str(m) for m in GRID),
        "--repetitions",
        ",".join(str(r) for r in REPETITIONS),
        "--max-steps",
        str(MAX_STEPS),
        *extra,
    )


@pytest.fixture(scope="module")
def comparison(tmp_path_factory):
    """One finished comparison directory, shared by every check below."""
    out = tmp_path_factory.mktemp("svbmc_stack")
    completed = run_harness(out)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return read_outputs(out) | {"out": out, "stdout": completed.stdout}


@pytest.fixture(scope="module")
def mixed(tmp_path_factory):
    """The same comparison with the integrated arm alone at ``M = 3``."""
    out = tmp_path_factory.mktemp("svbmc_stack_mixed")
    completed = run_harness(out, "--arms", ",".join(MIXED_ARMS))
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return read_outputs(out) | {"out": out, "stdout": completed.stdout}


def test_every_cell_ran_both_arms(comparison):
    cells = comparison["results"]["cells"]
    assert len(cells) == len(GROUPS) * sum(REPETITIONS)
    for group in GROUPS:
        for M, repetitions in zip(GRID, REPETITIONS):
            here = [
                c for c in cells if c["condition"] == group and c["M"] == M
            ]
            assert len(here) == repetitions
            assert {c["repetition"] for c in here} == set(range(repetitions))
    for cell in cells:
        assert len(cell["indices"]) == cell["M"]
        assert len(set(cell["indices"])) == cell["M"]
        assert {"integrated", "original"} == set(cell["arms"])
        for arm in cell["arms"].values():
            assert arm["M_used"] == cell["M"]
            assert len(arm["w"]) == sum(arm["K"])
            assert np.isclose(np.sum(arm["w"]), 1.0)
        # Both arms run on every cell, one at a time, and which goes first
        # alternates, so neither arm always pays a cold cache.
        assert cell["first_arm"] == harness.ARMS[cell["repetition"] % 2]


def test_subsets_of_one_M_are_disjoint(comparison):
    for group in GROUPS:
        for M in GRID:
            runs = [
                i
                for c in comparison["results"]["cells"]
                if c["condition"] == group and c["M"] == M
                for i in c["indices"]
            ]
            assert len(runs) == len(set(runs))
    assert comparison["results"]["settings"]["subsets"] == "disjoint"


def test_cell_schema(comparison):
    for cell in comparison["results"]["cells"]:
        assert cell["cell_seed"] == cell["cell_seed"] & 0xFFFFFFFF
        assert len(cell["entries"]) == cell["M"]
        # One generator seed per posterior, the same ones in both arms.
        assert len(set(cell["entry_seeds"])) == cell["M"]
        assert all(
            arm["entry_seeds"] == cell["entry_seeds"]
            for arm in cell["arms"].values()
        )
        assert "arm_set" not in cell
        for name, arm in cell["arms"].items():
            elbos, metrics = arm["elbos"], arm["metrics"]
            assert arm["headline"] in elbos
            assert "mc" in elbos
            assert set(elbos) <= set(
                k[len("elbo_err_") :]
                for k in metrics
                if k.startswith("elbo_err_")
            )
            # The integrated class returns exactly the draws asked for; the
            # original returns one rounded block per stacked run.
            if name == "integrated":
                assert metrics["n_samples"] == harness.N_DRAWS
            else:
                assert abs(metrics["n_samples"] - harness.N_DRAWS) <= cell["M"]
            assert arm["n_log_joint"] == min(
                harness.N_LOG_JOINT, metrics["n_samples"]
            )
            assert np.isfinite(metrics["mmtv"])
            assert np.isfinite(metrics["gskl"])
            assert np.isclose(
                metrics["gskl_normalized"],
                metrics["gskl"] / np.shape(metrics["post_cov"])[0],
            )
            assert np.isclose(
                arm["elbo_mc"], arm["e_log_joint_mc"] + arm["entropy_ref"]
            )
            for key in ("construction", "optimize", "sample"):
                assert arm[f"{key}_seconds"] > 0
            if name == "integrated":
                assert {
                    "raw",
                    "capped_I_median",
                    "naive",
                    "shrunk_two_level",
                } <= set(elbos)
                assert arm["elbo_sd"] > 0
                assert arm["shrinkage_available"] is True
                assert arm["shrinkage_noise_share"] is None or np.isfinite(
                    arm["shrinkage_noise_share"]
                )
            else:
                assert {"estimated", "debiased_I_median"} <= set(elbos)


def test_monte_carlo_elbo_reference(comparison):
    """Criterion 3's yardstick: the reference, the biases and the KL gap.

    ``elbo_mc`` adds an entropy the harness estimates itself, at the arm's
    own weights, to the arm's Monte Carlo expected log joint, and every
    other ELBO the arm reports is scored against it.
    """
    for cell in comparison["results"]["cells"]:
        assert cell["reference_seconds"] > 0
        for name, arm in cell["arms"].items():
            for key in (
                "entropy_ref",
                "entropy_ref_sd",
                "e_log_joint_mc",
                "e_log_joint_mc_sd",
                "elbo_mc",
                "elbo_mc_sd",
                "kl_gap",  # the fixture targets have a known ln Z
            ):
                assert np.isfinite(arm[key]), (name, key)
            assert arm["entropy_ref_sd"] > 0
            assert arm["e_log_joint_mc_sd"] > 0
            assert len(arm["entropy_ref_batches"]) == (
                harness.N_ENTROPY_BATCHES
            )
            assert np.isclose(
                arm["entropy_ref"], np.mean(arm["entropy_ref_batches"])
            )
            assert np.isclose(
                arm["elbo_mc_sd"],
                np.hypot(arm["entropy_ref_sd"], arm["e_log_joint_mc_sd"]),
            )
            assert np.isclose(
                arm["elbo_mc"], arm["e_log_joint_mc"] + arm["entropy_ref"]
            )
            assert np.isclose(arm["elbos"]["mc"], arm["elbo_mc"])
            assert np.isclose(
                arm["metrics"]["elbo_err_mc"], abs(arm["kl_gap"])
            )
            # Every variant but the reference itself carries its bias.
            assert set(arm["bias"]) == set(arm["elbos"]) - {"mc"}
            for variant, bias in arm["bias"].items():
                assert np.isclose(bias, arm["elbos"][variant] - arm["elbo_mc"])
            # Both estimate the entropy of the same mixture at the same
            # weights, so a gross disagreement means the reference was
            # taken at the wrong weights or on the wrong posteriors.
            assert abs(arm["entropy_ref"] - arm["entropy"]) < 1.0, (
                f"{name}: reference entropy {arm['entropy_ref']} against "
                f"the arm's own {arm['entropy']}"
            )


def test_entropy_reference_moves_with_the_weights_alone():
    """The reference of a cell, called directly on a two-posterior stack.

    Both arms' references are drawn from equally seeded generators, so
    they share their component draws and differ only through the weights;
    that is what makes their biases comparable. The estimate is the mean
    of its batches, its standard error the batches' standard error, and
    the estimator's own generator is left where it was.
    """
    from pyvbmc.svbmc import SVBMC

    entries = harness.fixture_conditions([GROUPS[1]])[GROUPS[1]][:2]
    stacked = SVBMC(
        [harness.load_entry(entry, rng=i) for i, entry in enumerate(entries)],
        seed=0,
    )
    K_total = int(np.sum(stacked.K))
    before = stacked.rng
    equal = np.full(K_total, 1.0 / K_total)
    tilted = np.linspace(1.0, 2.0, K_total)
    tilted /= tilted.sum()

    def reference(w):
        return harness.entropy_reference(
            stacked,
            w,
            np.random.default_rng([7, harness.ENTROPY_REF_STREAM]),
        )

    first, again, other = reference(equal), reference(equal), reference(tilted)
    assert stacked.rng is before
    # The same weights on the same seed give the same draws and the same
    # number; different weights on those draws give a different one.
    assert first == again
    assert first["entropy_ref"] != other["entropy_ref"]
    batches = first["entropy_ref_batches"]
    assert len(batches) == harness.N_ENTROPY_BATCHES
    assert np.isclose(first["entropy_ref"], np.mean(batches))
    assert np.isclose(
        first["entropy_ref_sd"],
        np.std(batches, ddof=1) / np.sqrt(harness.N_ENTROPY_BATCHES),
    )
    with pytest.raises(RuntimeError, match="did not retain the same runs"):
        reference(equal[:-1])
    assert stacked.rng is before


def test_weights_agree_between_the_arms(comparison):
    differences = {
        (cell["condition"], cell["M"], cell["repetition"]): cell["max_abs_dw"]
        for cell in comparison["results"]["cells"]
    }
    worst = max(differences.items(), key=lambda item: item[1])
    assert worst[1] < WEIGHT_TOLERANCE, (
        f"the arms disagree most on {worst[0]}: max|dw| = {worst[1]:.4g}; "
        f"all cells: {differences}"
    )


def test_single_run_rows(comparison):
    rows = comparison["results"]["single_run"]
    assert set(rows) == set(GROUPS)
    for group, group_rows in rows.items():
        assert len(group_rows) == POOL_SIZE
        for row in group_rows:
            assert row["name"].startswith(group)
            assert np.isfinite(row["elbo_err"])
            assert np.isfinite(row["gskl"])
            assert np.isfinite(row["mmtv"])


def test_summary_reports_every_cell_set(comparison):
    summary = comparison["summary"]
    assert [c["condition"] for c in summary["conditions"]] == list(GROUPS)
    assert summary["settings"]["max_steps"] == MAX_STEPS
    for condition in summary["conditions"]:
        assert [entry["M"] for entry in condition["M"]] == list(GRID)
        assert condition["single_run"]["mmtv"]["n"] == POOL_SIZE
        assert condition["all_M"]["cells"] == sum(REPETITIONS)
        for interval in condition["all_M"].values():
            if isinstance(interval, dict):
                assert interval["n"] == sum(REPETITIONS)
        for entry, repetitions in zip(condition["M"], REPETITIONS):
            assert entry["cells"] == repetitions
            for interval in (
                entry["max_abs_dw"],
                entry["runtime_ratio"],
                entry["paired"]["mmtv"],
                entry["arms"]["integrated"]["elbo_err_headline"],
                entry["arms"]["original"]["optimize_seconds"],
            ):
                assert interval["n"] == repetitions
                assert interval["lo"] <= interval["median"] <= interval["hi"]
    text = (comparison["out"] / "summary.md").read_text(encoding="utf-8")
    assert text.startswith("# S-VBMC stacking comparison")
    assert "disjoint within each `M`" in text


def test_summary_reports_the_biases_and_the_criterion_3_gates(comparison):
    """Criterion 3 in the summary: medians, the gate per `M`, the growth."""
    summary = comparison["summary"]
    for condition in summary["conditions"]:
        growth = condition["headline_bias_growth"]
        assert (growth["M_min"], growth["M_max"]) == (min(GRID), max(GRID))
        assert growth["bound"] == harness.GROWTH_BOUND
        assert np.isfinite(growth["growth"])
        assert growth["growth_within_bound"] in (True, False)
        assert growth["growth_within_bound"] == (
            growth["growth"] < growth["bound"]
        )
        for entry, repetitions in zip(condition["M"], REPETITIONS):
            integrated = entry["arms"]["integrated"]
            original = entry["arms"]["original"]
            gate = entry["criterion3"]
            assert gate["headline_bias_not_worse"] in (True, False)
            assert gate["median_bias_headline"] == (
                integrated["bias"]["headline"]["median"]
            )
            assert gate["median_bias_estimated"] == (
                original["bias"]["estimated"]["median"]
            )
            assert gate["headline_bias_not_worse"] == (
                abs(gate["median_bias_headline"])
                <= abs(gate["median_bias_estimated"])
            )
            for interval in (
                integrated["bias"]["raw"],
                integrated["bias"]["shrunk_two_level"],
                original["bias"]["debiased_I_median"],
                integrated["kl_gap"],
                original["kl_gap"],
                integrated["entropy_ref"],
                original["entropy_ref"],
                integrated["elbo_mc_sd"],
                entry["paired"]["bias_headline_minus_estimated"],
            ):
                assert interval["n"] == repetitions
                assert interval["lo"] <= interval["median"] <= interval["hi"]
            availability = integrated["shrinkage_availability"]
            assert availability == {
                "contributing_cells": repetitions,
                "unavailable_cells": 0,
                "not_recorded_cells": 0,
            }
    text = (comparison["out"] / "summary.md").read_text(encoding="utf-8")
    assert "descriptive only" in text
    # The bias against `elbo_mc` is the criterion; the error against ln Z
    # is descriptive, and follows it.
    assert text.index("bias headline int") < text.index("err int")


def test_integrated_report_omits_unavailable_numeric_value():
    class Stack:
        elbo = 1.0
        elbo_details = {
            "raw": 1.0,
            "capped_I_median": 0.9,
            "capped_E_median": 0.8,
            "naive": 0.7,
            "shrunk_two_level": None,
            "shrinkage_noise_share": None,
        }

    elbos, available, share = harness.integrated_elbo_report(Stack())
    assert "shrunk_two_level" not in elbos
    assert available is False
    assert share is None
    encoded = json.dumps(
        {
            "elbos": elbos,
            "shrinkage_available": available,
            "shrinkage_noise_share": share,
        }
    )
    assert '"shrinkage_noise_share": null' in encoded


def _without_shrinkage_fields(results):
    """Copy cells as a historical result that predates shrinkage."""
    rows = copy.deepcopy(results["cells"])
    for row in rows:
        integrated = row["arms"].get("integrated")
        if integrated is None:
            continue
        integrated["elbos"].pop("shrunk_two_level", None)
        integrated["bias"].pop("shrunk_two_level", None)
        integrated["metrics"].pop("elbo_err_shrunk_two_level", None)
        integrated.pop("shrinkage_available", None)
        integrated.pop("shrinkage_noise_share", None)
    return rows


def _without_shrinkage_summary(summary):
    """Copy a summary with only the pre-shrinkage fields."""
    result = copy.deepcopy(summary)
    for condition in result["conditions"]:
        for entry in condition["M"]:
            integrated = entry["arms"].get("integrated")
            if integrated is None:
                continue
            integrated["bias"].pop("shrunk_two_level", None)
            integrated.pop("shrinkage_availability", None)
    return result


def test_added_summary_uses_an_independent_bootstrap_stream(comparison):
    results = comparison["results"]
    settings = results["settings"]
    seed = settings["seed"]
    current_rng = np.random.default_rng(seed)
    historical_rng = np.random.default_rng(seed)
    current = harness.build_summary(
        copy.deepcopy(results["cells"]),
        results["single_run"],
        settings,
        current_rng,
    )
    historical = harness.build_summary(
        _without_shrinkage_fields(results),
        results["single_run"],
        settings,
        historical_rng,
    )
    assert _without_shrinkage_summary(current)["conditions"] == (
        _without_shrinkage_summary(historical)["conditions"]
    )
    assert (
        current_rng.bit_generator.state == historical_rng.bit_generator.state
    )
    for condition in historical["conditions"]:
        for entry in condition["M"]:
            availability = entry["arms"]["integrated"][
                "shrinkage_availability"
            ]
            assert availability["contributing_cells"] == 0
            assert availability["unavailable_cells"] == 0
            assert availability["not_recorded_cells"] == entry["cells"]


def test_summary_counts_unavailable_shrinkage_cells(comparison):
    results = comparison["results"]
    first_condition = results["cells"][0]["condition"]
    first_M = results["cells"][0]["M"]
    rows = [
        copy.deepcopy(row)
        for row in results["cells"]
        if row["condition"] == first_condition and row["M"] == first_M
    ]
    for row in rows:
        integrated = row["arms"]["integrated"]
        integrated["elbos"].pop("shrunk_two_level", None)
        integrated["bias"].pop("shrunk_two_level", None)
        integrated["metrics"].pop("elbo_err_shrunk_two_level", None)
        integrated["shrinkage_available"] = False
        integrated["shrinkage_noise_share"] = None
    settings = copy.deepcopy(results["settings"])
    settings["M"] = [first_M]
    settings["repetitions"] = [len(rows)]
    summary = harness.build_summary(
        rows,
        {first_condition: results["single_run"][first_condition]},
        settings,
        np.random.default_rng(settings["seed"]),
    )
    integrated = summary["conditions"][0]["M"][0]["arms"]["integrated"]
    assert integrated["bias"]["shrunk_two_level"]["n"] == 0
    assert integrated["shrinkage_availability"] == {
        "contributing_cells": 0,
        "unavailable_cells": len(rows),
        "not_recorded_cells": 0,
    }
    text = harness.summary_markdown(summary)
    assert f"(0 / {len(rows)} / 0)" in text


def test_equivalence_tests_cover_every_cell_set(comparison):
    """Criterion 2: one Holm-corrected signed-rank family per metric."""
    summary = comparison["summary"]
    tests = summary["equivalence_tests"]
    assert len(tests) == len(GROUPS) * len(GRID) * len(harness.TEST_METRICS)
    for test in tests:
        assert test["n_pairs"] == dict(zip(GRID, REPETITIONS))[test["M"]]
        assert 0.0 <= test["pvalue"] <= test["holm_adjusted_pvalue"] <= 1.0
        assert test["improved"] + test["worsened"] + test["tied"] == (
            test["n_pairs"]
        )
    for condition in summary["conditions"]:
        for entry in condition["M"]:
            assert set(entry["tests"]) == set(harness.TEST_METRICS)
            for metric, test in entry["tests"].items():
                assert (test["condition"], test["M"], test["metric"]) == (
                    condition["condition"],
                    entry["M"],
                    metric,
                )


def test_summarize_only_rebuilds_the_summary(comparison):
    out = comparison["out"]
    before = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    before_text = (out / "summary.md").read_text(encoding="utf-8")
    completed = run_script("--summarize-only", "--out", str(out))
    assert completed.returncode == 0, completed.stdout + completed.stderr
    after = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    # The bootstrap is seeded from the settings the run recorded, so the
    # rebuilt tables are the same numbers.
    assert after["conditions"] == before["conditions"]
    assert after["equivalence_tests"] == before["equivalence_tests"]
    assert after["settings"] == before["settings"]
    # And the same markdown, but for the line carrying the date it ran.
    after_text = (out / "summary.md").read_text(encoding="utf-8")
    assert dateless(after_text) == dateless(before_text)


def test_summarize_only_describes_the_run_it_summarizes(comparison, tmp_path):
    """An older comparison is described by its settings, not by today's.

    Its draw counts, entropy reference, bootstrap and stacking call are
    the ones it recorded; only a field a results file predating it does
    not carry falls back to this module's constant.
    """
    results = json.loads(
        (comparison["out"] / "results.json").read_text(encoding="utf-8")
    )
    settings = results["settings"]
    for key in (
        "n_samples",
        "n_samples_final",
        "n_log_joint",
        "n_entropy_ref",
        "n_entropy_batches",
        "bootstrap_resamples",
        "max_steps",
        "seed",
    ):
        assert key in settings
    settings["n_entropy_batches"] = settings["n_entropy_batches"] + 1
    settings["n_samples_final"] = 17
    del settings["n_samples"]
    del settings["subsets"]
    path = tmp_path / "results.json"
    path.write_text(json.dumps(results), encoding="utf-8")
    completed = run_script(
        "--summarize-only",
        "--out",
        str(tmp_path),
        "--from-results",
        str(path),
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    text = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert f"in {settings['n_entropy_batches']} batches" in text
    assert "a fresh 17 for the integrated class" in text
    # The one field this file does not carry, and only that one.
    assert f"`n_samples={harness.N_SAMPLES}`" in text
    assert f"{harness.N_ENTROPY_BATCHES} batches" not in text
    # A comparison whose subsets could overlap is not said to be disjoint.
    assert "disjoint" not in text


def test_summarize_only_names_a_results_file_it_cannot_summarize(
    comparison, tmp_path
):
    """A comparison run before the Monte Carlo ELBO reference existed.

    Its cells carry no bias against `elbo_mc`, which every summary is
    stated in and which cannot be recovered from the record, so the
    rebuild must say which field is missing rather than fail on the first
    lookup.
    """
    results = json.loads(
        (comparison["out"] / "results.json").read_text(encoding="utf-8")
    )
    for arm in results["cells"][0]["arms"].values():
        del arm["entropy_ref"]
    path = tmp_path / "results.json"
    path.write_text(json.dumps(results), encoding="utf-8")
    completed = run_script(
        "--summarize-only",
        "--out",
        str(tmp_path),
        "--from-results",
        str(path),
    )
    assert completed.returncode != 0
    assert "entropy_ref" in completed.stderr
    assert "predates" in completed.stderr
    assert not (tmp_path / "summary.json").exists()


def test_summarize_only_refuses_the_flags_it_cannot_use(tmp_path):
    completed = run_script(
        "--summarize-only", "--out", str(tmp_path), "--conditions", "x"
    )
    assert completed.returncode != 0
    assert "--conditions" in completed.stderr


def test_fixture_groups_are_read_from_the_sidecars():
    """A group holds the fixtures whose sidecar names it, and no others."""
    entries = harness.fixture_conditions(["upstream_GMM"])["upstream_GMM"]
    assert len(entries) == POOL_SIZE
    assert not any("noisy" in entry["name"] for entry in entries)
    # Each entry carries the hashes its posterior is checked against.
    for entry in entries:
        files = harness.run_files(harness.entry_base(entry))
        assert entry["sha256"] == {
            suffix: contract.sha256_file(path)
            for suffix, path in files.items()
        }


def test_sources_identify_both_arms(comparison):
    sources = comparison["sources"]
    integrated = sources["arms"]["integrated"]
    original = sources["arms"]["original"]
    # Both arms must import the campaign's frozen gpyreg worktree, and only
    # the original arm may see the pinned upstream package.
    expected = str((GPYREG_SOURCE / "gpyreg").resolve())
    assert integrated["environment"]["host"]["gpyreg_import"] == expected
    assert original["environment"]["host"]["gpyreg_import"] == expected
    # The arms differ in their import paths, which is the point of the
    # subprocess, and in nothing the source half pins.
    assert (
        integrated["environment"]["source"]
        == original["environment"]["source"]
    )
    assert original["svbmc_version"] == "0.1.1"
    checkout = Path(sources["baseline"]["checkout"])
    assert checkout == harness.baseline_checkout(BASELINE)
    upstream = checkout / "src"
    assert Path(original["svbmc_import"]).parent.parent == upstream
    assert str(upstream) in sources["baseline"]["pythonpath"]
    assert str(upstream) not in (integrated["pythonpath"] or "")
    assert (
        original["identity"]["source"]["trees"]["baseline"]["commit"]
        == harness.read_baseline_record()["baseline"]["commit"]
    )
    assert integrated["torch_version"] == original["torch_version"]
    assert sources["baseline_environment"]["clean"] is True
    assert (
        sources["harness"]["sha256"]
        == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    )


def test_existing_comparison_is_not_overwritten(comparison):
    again = run_harness(comparison["out"])
    assert again.returncode != 0
    assert "--overwrite" in again.stderr


def test_integrated_arm_alone(comparison, mixed):
    """The integrated arm alone at `M = 3`: one arm per cell, no paired
    quantity, and the same integrated fits as the two-arm run of the same
    cells."""
    results = mixed["results"]
    assert results["settings"]["arms"] == list(harness.ARMS)
    assert results["settings"]["arms_by_M"] == list(MIXED_ARMS)
    cells = results["cells"]
    assert len(cells) == len(GROUPS) * sum(REPETITIONS)
    alone = [cell for cell in cells if cell["M"] == GRID[1]]
    for cell in alone:
        assert set(cell["arms"]) == {"integrated"}
        assert cell["first_arm"] == "integrated"
        assert cell["max_abs_dw"] is None
        outcome = cell["arms"]["integrated"]
        assert all(key in outcome for key in harness.REFERENCE_FIELDS)
    assert "the integrated arm alone at M = 3" in mixed["stdout"]
    # The integrated arm is seeded by the cell and rebuilds its posteriors,
    # and so is the original, so the same cells give the same fits whether
    # or not the other arm runs beside them, and whatever ran before.
    twins = {
        (c["condition"], c["M"], c["repetition"]): c
        for c in comparison["results"]["cells"]
    }
    for cell in cells:
        twin = twins[(cell["condition"], cell["M"], cell["repetition"])]
        assert cell["entry_seeds"] == twin["entry_seeds"]
        for arm, outcome in cell["arms"].items():
            assert canonical(outcome) == canonical(twin["arms"][arm])
    summary = mixed["summary"]
    assert len(summary["equivalence_tests"]) == len(GROUPS) * len(
        harness.TEST_METRICS
    )
    for condition in summary["conditions"]:
        assert condition["all_M"]["runtime_ratio"]["n"] == REPETITIONS[0]
        entry = condition["M"][1]
        assert entry["arms_present"] == ["integrated"]
        assert set(entry["arms"]) == {"integrated"}
        assert entry["paired"] == {}
        assert entry["tests"] == {}
        assert entry["runtime_ratio"]["n"] == 0
        gate = entry["criterion3"]
        assert gate["median_bias_headline"] is not None
        assert gate["median_bias_estimated"] is None
        assert gate["headline_bias_not_worse"] is None
        growth = condition["headline_bias_growth"]
        assert growth["growth"] is not None
        assert growth["growth_within_bound"] is not None


def filtered_results(source, out, keep_M):
    """A copy of a run's results file holding only the cells of some M."""
    results = json.loads((source / "results.json").read_text(encoding="utf-8"))
    results["cells"] = [c for c in results["cells"] if c["M"] in keep_M]
    settings = results["settings"]
    kept = [
        (M, R, A)
        for M, R, A in zip(
            settings["M"], settings["repetitions"], settings["arms_by_M"]
        )
        if M in keep_M
    ]
    settings["M"] = [M for M, _, _ in kept]
    settings["repetitions"] = [R for _, R, _ in kept]
    settings["arms_by_M"] = [A for _, _, A in kept]
    settings["arms"] = [
        arm
        for arm in harness.ARMS
        if any(arm in harness.ARM_SETS[A] for A in settings["arms_by_M"])
    ]
    out.mkdir(parents=True, exist_ok=True)
    path = out / "results.json"
    path.write_text(json.dumps(results), encoding="utf-8")
    return path


def test_summarize_only_merges_a_two_arm_run_with_an_integrated_run(
    comparison, mixed, tmp_path
):
    """Both arms up to one M and the integrated arm beyond it summarize
    together: paired quantities where both ran, growth across every M."""
    low = filtered_results(comparison["out"], tmp_path / "low", {GRID[0]})
    high = filtered_results(mixed["out"], tmp_path / "high", {GRID[1]})
    out = tmp_path / "merged"
    result = run_script(
        "--summarize-only",
        "--out",
        str(out),
        "--from-results",
        str(low),
        "--from-results",
        str(high),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    settings = summary["settings"]
    assert settings["M"] == list(GRID)
    assert settings["repetitions"] == list(REPETITIONS)
    assert settings["arms"] == list(harness.ARMS)
    assert settings["arms_by_M"] == list(MIXED_ARMS)
    assert [part["arms"] for part in settings["merged_from"]] == [
        list(harness.ARMS),
        ["integrated"],
    ]
    assert len(summary["equivalence_tests"]) == len(GROUPS) * len(
        harness.TEST_METRICS
    )
    for condition in summary["conditions"]:
        low_entry, high_entry = condition["M"]
        assert (low_entry["M"], high_entry["M"]) == GRID
        assert low_entry["arms_present"] == list(harness.ARMS)
        assert low_entry["tests"] and low_entry["paired"]
        assert high_entry["arms_present"] == ["integrated"]
        assert high_entry["tests"] == {} and high_entry["paired"] == {}
        growth = condition["headline_bias_growth"]
        assert (growth["M_min"], growth["M_max"]) == GRID
        assert growth["growth"] is not None
        assert condition["all_M"]["runtime_ratio"]["n"] == REPETITIONS[0]
    text = (out / "summary.md").read_text(encoding="utf-8")
    assert "summarized together" in text
    # The same cell in two files is refused by name.
    result = run_script(
        "--summarize-only",
        "--out",
        str(tmp_path / "dup"),
        "--from-results",
        str(comparison["out"] / "results.json"),
        "--from-results",
        str(mixed["out"] / "results.json"),
    )
    assert result.returncode != 0
    assert "summarized once" in result.stderr
    # And --arms cannot apply to a summary, which runs no cell.
    result = run_script(
        "--summarize-only",
        "--out",
        str(tmp_path / "flag"),
        "--from-results",
        str(low),
        "--arms",
        "integrated",
    )
    assert result.returncode != 0
    assert "--arms" in result.stderr


# --------------------------------------------------------------------------
# The subsets, the arms and the pools' gpyreg, without a cell
# --------------------------------------------------------------------------


def test_plan_draws_disjoint_subsets_within_each_M():
    """Repetition `r` takes the `r`-th block of `M` runs of a permutation
    drawn for the condition and `M` alone."""
    conditions = {
        "a": [{"name": f"a{i}", "seed": 1000 + i} for i in range(50)],
        "b": [{"name": f"b{i}", "seed": 2000 + i} for i in range(40)],
    }
    indices = {"a": 3, "b": 5}
    grid, repetitions = [2, 4, 16, 64], [20, 12, 3, 1]
    arms = ["both", "integrated", "both", "both"]
    plan, skipped = harness.plan_cells(
        conditions, grid, repetitions, 7, indices, arms
    )
    by_set = {}
    for cell in plan:
        by_set.setdefault((cell["condition"], cell["M"]), []).append(cell)
        assert len(set(cell["indices"])) == cell["M"]
        assert cell["indices"] == sorted(cell["indices"])
        key = [7, indices[cell["condition"]], cell["M"], cell["repetition"]]
        assert cell["cell_seed"] == int(
            np.random.SeedSequence(key).generate_state(1, dtype=np.uint32)[0]
        )
        assert cell["entries"] == [
            conditions[cell["condition"]][i]["name"] for i in cell["indices"]
        ]
    for (condition, M), cells in by_set.items():
        runs = [i for cell in cells for i in cell["indices"]]
        assert len(runs) == len(set(runs)), (condition, M)
        assert [c["repetition"] for c in cells] == list(range(len(cells)))
        # The blocks of one permutation, in order.
        order = np.random.default_rng([7, indices[condition], M]).permutation(
            len(conditions[condition])
        )
        for cell in cells:
            r = cell["repetition"]
            assert cell["indices"] == sorted(order[r * M : (r + 1) * M])
    assert {key: len(cells) for key, cells in by_set.items()} == {
        ("a", 2): 20,
        ("a", 4): 12,
        ("a", 16): 3,
        ("b", 2): 20,
        ("b", 4): 10,
        ("b", 16): 2,
    }
    assert skipped == [
        {"condition": "a", "M": 64, "pool": 50, "requested": 1, "drawn": 0},
        {"condition": "b", "M": 4, "pool": 40, "requested": 12, "drawn": 10},
        {"condition": "b", "M": 16, "pool": 40, "requested": 3, "drawn": 2},
        {"condition": "b", "M": 64, "pool": 40, "requested": 1, "drawn": 0},
    ]
    # The arms of every M, the two-arm cells alternating which goes first.
    for cell in plan:
        if cell["M"] == 4:
            assert cell["arm_set"] == ["integrated"]
            assert cell["first_arm"] == "integrated"
        else:
            assert cell["arm_set"] == list(harness.ARMS)
            assert cell["first_arm"] == harness.ARMS[cell["repetition"] % 2]
    # Independent across M: M = 4 does not reuse the blocks of M = 2.
    pairs = by_set[("a", 2)]
    assert by_set[("a", 4)][0]["indices"] != sorted(
        pairs[0]["indices"] + pairs[1]["indices"]
    )
    # And the same draw whichever conditions an invocation selects.
    alone, _ = harness.plan_cells(
        {"b": conditions["b"]}, grid, repetitions, 7, {"b": 5}, arms
    )
    assert alone == [cell for cell in plan if cell["condition"] == "b"]


def test_arm_sets():
    assert harness.arm_sets("both", [2, 3]) == ["both", "both"]
    assert harness.arm_sets("both,integrated", [2, 3]) == [
        "both",
        "integrated",
    ]
    for wrong in ("both,integrated,both", "neither", "both,neither"):
        with pytest.raises(ValueError, match="--arms"):
            harness.arm_sets(wrong, [2, 3])
    assert harness.describe_arms(
        [2, 3, 4], ["both", "integrated", "both"]
    ) == ("both arms at M = 2, 4; the integrated arm alone at M = 3")


def test_pool_gpyreg_commit_reads_both_manifest_shapes():
    flat = {"identity": {"source": {"gpyreg_commit": "abc"}}}
    trees = {
        "identity": {
            "source": {"trees": {"gpyreg": {"commit": "def", "clean": True}}}
        }
    }
    assert harness.pool_gpyreg_commit(flat) == "abc"
    assert harness.pool_gpyreg_commit(trees) == "def"
    assert harness.pool_gpyreg_commit({"identity": {"source": {}}}) is None


def test_a_recreated_baseline_verifies_by_content(tmp_path):
    """The baseline cloned to another path verifies there, and only as
    committed."""
    record = harness.read_baseline_record()
    original = harness.baseline_checkout(BASELINE)
    moved = tmp_path / "elsewhere"
    clone = moved / "source"
    subprocess.run(
        [
            "git",
            "clone",
            "--quiet",
            "--no-hardlinks",
            str(original),
            str(clone),
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(clone),
            "checkout",
            "--quiet",
            record["baseline"]["commit"],
        ],
        check=True,
    )
    checkout = harness.baseline_checkout(moved)
    assert checkout == clone.resolve()
    report = harness.verify_baseline(record, checkout)
    assert report["checkout"] == str(clone.resolve())
    assert report["commit"] == record["baseline"]["commit"]
    assert report["files_sha256_committed"] == {
        name: entry["sha256"]
        for name, entry in record["baseline"]["files_sha256_committed"].items()
    }
    package = clone / "src" / "svbmc"
    changed = package / "utils.py"
    text = changed.read_bytes()
    changed.write_bytes(text + b"\n# changed\n")
    with pytest.raises(contract.IdentityError, match="utils.py differs"):
        harness.verify_baseline(record, checkout)
    changed.write_bytes(text)
    harness.verify_baseline(record, checkout)
    (package / "extra.py").write_text("", encoding="utf-8")
    with pytest.raises(contract.IdentityError, match="extra.py"):
        harness.verify_baseline(record, checkout)
    (package / "extra.py").unlink()
    # A directory that holds no checkout, or two of them, names none.
    with pytest.raises(contract.IdentityError, match="BASELINE_DIR"):
        harness.baseline_checkout(tmp_path / "nothing")
    shutil.copytree(clone, moved / "second")
    with pytest.raises(contract.IdentityError, match="2 subdirectories"):
        harness.baseline_checkout(moved)


# --------------------------------------------------------------------------
# The campaign: prepared, split into tasks, verified and assembled
# --------------------------------------------------------------------------

CAMPAIGN_ARGUMENTS = (
    "--fixtures",
    ",".join(GROUPS),
    "--M",
    ",".join(str(m) for m in GRID),
    "--repetitions",
    ",".join(str(r) for r in REPETITIONS),
    "--arms",
    ",".join(MIXED_ARMS),
    "--max-steps",
    str(MAX_STEPS),
    "--split-from",
    str(GRID[1]),
    # The harness and its tests are developed together, so the campaigns
    # here are computed from whatever the checkout holds.
    "--allow-dirty",
)


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    """The comparison of ``mixed`` as a campaign: prepared, every task run
    by a worker process of its own (in the reverse of the case order, and
    without ``BASELINE_DIR`` for the integrated arm alone), verified and
    assembled."""
    out = tmp_path_factory.mktemp("svbmc_stack_campaign")
    prepared = run_script("prepare", "--out", str(out), *CAMPAIGN_ARGUMENTS)
    assert prepared.returncode == 0, prepared.stdout + prepared.stderr
    listed = run_script("cases", "--out", str(out))
    assert listed.returncode == 0, listed.stderr
    lines = listed.stdout.splitlines()
    logs = {}
    for line in reversed(lines):
        drop = () if "original" in line else ("BASELINE_DIR",)
        result = run_script(
            "worker",
            "--out",
            str(out),
            "--case",
            line,
            env=harness_environment(drop),
        )
        assert result.returncode == 0, line + result.stdout + result.stderr
        logs[line] = result.stdout
    verified = run_script("verify", "--out", str(out))
    assert verified.returncode == 0, verified.stdout + verified.stderr
    assembled = run_script("assemble", "--out", str(out))
    assert assembled.returncode == 0, assembled.stdout + assembled.stderr
    return read_outputs(out) | {
        "out": out,
        "lines": lines,
        "logs": logs,
        "manifest": contract.read_json(out / "manifest.json"),
        "verification": contract.read_json(out / "verification.json"),
    }


def test_campaign_cases(campaign):
    lines, out = campaign["lines"], campaign["out"]
    assert [line.split()[0] for line in lines] == [
        f"{group}/{task}"
        for group in GROUPS
        for task in ("M2", "M3_r0", "M3_r1")
    ]
    assert lines[0] == (
        f"{GROUPS[0]}/M2 M=2 repetitions=0-1 arms=integrated+original"
    )
    assert lines[1] == f"{GROUPS[0]}/M3_r0 M=3 repetitions=0 arms=integrated"
    raw = subprocess.run(
        [sys.executable, str(SCRIPT), "cases", "--out", str(out)],
        cwd=str(ROOT),
        env=harness_environment(),
        capture_output=True,
    )
    assert raw.returncode == 0 and b"\r" not in raw.stdout
    (out / "cases.txt").write_bytes(raw.stdout)
    assert contract.read_cases(out / "cases.txt") == lines
    for name, indices in (
        ("M3", [2, 3, 5, 6]),
        ("both", [1, 4]),
        ("integrated", [2, 3, 5, 6]),
    ):
        listed = run_script("cases", "--out", str(out), "--subset", name)
        assert listed.returncode == 0, listed.stderr
        assert listed.stdout.splitlines() == [
            f"{i} {lines[i - 1]}" for i in indices
        ]
        subset = out / "subsets" / f"{name}.txt"
        subset.parent.mkdir(exist_ok=True)
        subset.write_text(listed.stdout, encoding="utf-8", newline="\n")
        assert contract.read_subset(subset, lines) == indices
    for wrong in ("M9", "odd"):
        listed = run_script("cases", "--out", str(out), "--subset", wrong)
        assert listed.returncode != 0


def test_campaign_manifest(campaign):
    manifest = campaign["manifest"]
    assert manifest["contract"] == contract.CONTRACT_VERSION
    assert manifest["campaign"] == harness.CAMPAIGN
    assert manifest["kind"] == "fixture"
    settings = manifest["settings"]
    assert (settings["M"], settings["repetitions"]) == (
        list(GRID),
        list(REPETITIONS),
    )
    assert settings["arms_by_M"] == list(MIXED_ARMS)
    assert (settings["seed"], settings["max_steps"]) == (0, MAX_STEPS)
    assert settings["subsets"] == "disjoint"
    assert manifest["split_from"] == GRID[1]
    assert [c["label"] for c in manifest["conditions"]] == list(GROUPS)
    for condition in manifest["conditions"]:
        assert len(condition["entries"]) == POOL_SIZE
        for entry in condition["entries"]:
            assert set(entry["sha256"]) == set(harness.RUN_SUFFIXES)
    assert [f["condition"] for f in manifest["fixtures"]] == list(GROUPS)
    plan = manifest["plan"]
    assert len(plan) == len(GROUPS) * sum(REPETITIONS)
    for group in GROUPS:
        for M in GRID:
            runs = [
                i
                for cell in plan
                if cell["condition"] == group and cell["M"] == M
                for i in cell["indices"]
            ]
            assert len(runs) == len(set(runs)) == M * REPETITIONS[0]
    record = harness.read_baseline_record()
    baseline = manifest["baseline"]
    assert baseline["commit"] == record["baseline"]["commit"]
    assert baseline["torch_version"] == record["torch"]["version"]
    assert baseline["verification"]["files_sha256_committed"] == (
        baseline["files_sha256_committed"]
    )
    source = manifest["identity"]["source"]
    assert set(source["trees"]) == {"harness", "gpyreg", "baseline"}
    assert source["trees"]["baseline"]["commit"] == baseline["commit"]
    assert source["versions"]["torch"] == record["torch"]["version"]
    assert set(source["files"]) == {
        name + ("/" if (ROOT / name).is_dir() else "")
        for name in harness.HARNESS_FILES
    }
    assert set(manifest["site"]) == set(contract.SETTINGS)
    assert manifest["pip_freeze"]
    assert manifest["finishing_steps"] == [["assemble"]]
    assert contract.finishing_steps(manifest) == [["assemble"]]


def test_assembled_campaign_equals_the_single_process_run(campaign, mixed):
    """The acceptance of Phase 5: split into tasks, run in any order in
    processes of their own and assembled, the comparison gives the cells,
    the single-run rows and the summary of one process, but for the
    seconds it measures."""
    split, whole = campaign["results"], mixed["results"]
    assert split["settings"] == whole["settings"]
    assert split["skipped"] == whole["skipped"]
    assert split["kind"] == whole["kind"]
    assert canonical(split["single_run"]) == canonical(whole["single_run"])
    assert [
        (c["condition"], c["M"], c["repetition"]) for c in split["cells"]
    ] == [(c["condition"], c["M"], c["repetition"]) for c in whole["cells"]]
    for a, b in zip(split["cells"], whole["cells"]):
        assert canonical(a) == canonical(b), (a["condition"], a["M"])
    summaries = campaign["summary"], mixed["summary"]
    for key in ("conditions", "equivalence_tests", "settings"):
        assert canonical(summaries[0][key]) == canonical(summaries[1][key])
    sources = campaign["sources"]
    assert sources["verification"]["verified"] == len(campaign["lines"])
    assert set(sources["tasks"]) == {
        line.split()[0] for line in campaign["lines"]
    }


def test_campaign_verifies(campaign):
    report = campaign["verification"]
    assert report["campaign"] == harness.CAMPAIGN
    assert report["exit_code"] == 0
    assert report["counts"]["verified"] == len(campaign["lines"])
    assert report["stray"] == []
    for case in report["cases"]:
        assert case["status"] == "verified"
        assert case["cells"] == (2 if case["tag"].endswith("/M2") else 1)


def test_tasks_of_the_integrated_arm_alone_need_no_baseline(campaign):
    """They ran without BASELINE_DIR, and their identity leaves it out."""
    out = campaign["out"]
    record = harness.read_baseline_record()
    for line, log in campaign["logs"].items():
        tag = line.split()[0]
        completion = contract.read_json(contract.record_path(out, tag))
        trees = completion["identity"]["source"]["trees"]
        assert completion["cells"] == len(completion["artifacts"])
        assert completion["peak_rss_bytes"]["harness"] is None or (
            completion["peak_rss_bytes"]["harness"] > 0
        )
        if "original" in line:
            assert "baseline" in trees
            assert completion["baseline"]["files_sha256_committed"] == {
                n: e["sha256"]
                for n, e in record["baseline"][
                    "files_sha256_committed"
                ].items()
            }
            assert completion["original_arm"]["svbmc_version"] == "0.1.1"
            assert set(completion["warm_up_seconds"]) == set(harness.ARMS)
        else:
            assert "baseline" not in trees
            assert completion["baseline"] is None
            assert completion["original_arm"] is None
            assert completion["peak_rss_bytes"]["original_arm"] is None
            assert set(completion["warm_up_seconds"]) == {"integrated"}
        # Each task warms up before its first timed cell.
        assert log.index("warmed up integrated") < log.index("DONE")


def snapshot(directory):
    """Every file under a directory with its bytes and modification time."""
    return {
        path.relative_to(directory).as_posix(): (
            path.read_bytes(),
            path.stat().st_mtime_ns,
        )
        for path in sorted(Path(directory).rglob("*"))
        if path.is_file()
    }


def test_a_completed_task_exits_at_once(campaign):
    out = campaign["out"]
    before = snapshot(out)
    result = run_script(
        "worker", "--out", str(out), "--case", campaign["lines"][0]
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "already has a completion record" in result.stdout
    assert "warmed up" not in result.stdout
    assert snapshot(out) == before


def campaign_copy(campaign, tmp_path, missing=()):
    """A copy of the finished campaign, before its assembly, with the tasks
    of the case indices ``missing`` (from 1) undone: no record, no cell."""
    out = tmp_path / "campaign"
    shutil.copytree(campaign["out"], out)
    for name in ("results", "summary", "sources", "verification"):
        (out / f"{name}.json").unlink(missing_ok=True)
    (out / "summary.md").unlink(missing_ok=True)
    manifest = contract.read_json(out / "manifest.json")
    tasks = harness.campaign_tasks(manifest)
    for index in missing:
        task = tasks[index - 1]
        contract.record_path(out, task["tag"]).unlink()
        for path in task["paths"]:
            (out / path).unlink()
    return out, manifest, tasks


@pytest.fixture
def slurm(tmp_path):
    """Stub Slurm commands first on the harness's PATH: the state directory
    whose files answer the accounting, and the environment."""
    state = tmp_path / "state"
    (state / "sacct").mkdir(parents=True)
    directory = stubs.write_stubs(tmp_path / "bin")
    return state, stubs.stub_environment(
        directory, state, base=harness_environment()
    )


def claim_by(out, tag, job, task):
    contract.write_json(
        contract.claim_path(out, tag),
        contract.new_claim(tag, task={"job": job, "array_task": task}),
    )


def status_of(out, environment, index):
    result = run_script("verify", "--out", str(out), env=environment)
    report = contract.read_json(out / "verification.json")
    return result.returncode, report["cases"][index - 1]


def test_a_live_claim_refuses_a_second_worker(campaign, tmp_path, slurm):
    """A task another live task holds is refused, and its files stay."""
    state, environment = slurm
    out, _, tasks = campaign_copy(campaign, tmp_path, missing=[3])
    task = tasks[2]
    partial = out / task["paths"][0]
    partial.write_text("the running task's cell\n", encoding="utf-8")
    claim_by(out, task["tag"], "900", "4")
    (state / "sacct" / "900_4").write_text("RUNNING\n", encoding="utf-8")
    before = snapshot(out)
    result = run_script(
        "worker", "--out", str(out), "--case", task["line"], env=environment
    )
    assert result.returncode == contract.EXIT_CLAIMED, (
        result.stdout + result.stderr
    )
    assert "claimed by 900_4" in result.stdout
    assert snapshot(out) == before
    code, case = status_of(out, environment, 3)
    assert code == 0 and case["status"] == "in_flight"


def test_a_two_arm_task_without_the_baseline_is_refused(campaign, tmp_path):
    """Without BASELINE_DIR the original arm's identity cannot be
    established: the worker refuses before it claims or touches anything."""
    out, _, tasks = campaign_copy(campaign, tmp_path, missing=[4])
    task = tasks[3]
    assert "original" in task["arms"]
    before = snapshot(out)
    result = run_script(
        "worker",
        "--out",
        str(out),
        "--case",
        task["line"],
        env=harness_environment(drop=("BASELINE_DIR",)),
    )
    assert result.returncode == contract.EXIT_IDENTITY, (
        result.stdout + result.stderr
    )
    assert "BASELINE_DIR is not set" in result.stdout
    assert snapshot(out) == before


def test_an_interrupted_task_is_taken_over(campaign, tmp_path, slurm):
    """A task killed outright leaves its files and a stale claim; the next
    worker takes the claim over, removes the files and completes it."""
    state, environment = slurm
    out, _, tasks = campaign_copy(campaign, tmp_path, missing=[6])
    task = tasks[5]
    (out / task["paths"][0]).write_text("half a cell\n", encoding="utf-8")
    claim_by(out, task["tag"], "800", "1")
    (state / "sacct" / "800_1").write_text("OUT_OF_MEMORY\n", encoding="utf-8")
    code, case = status_of(out, environment, 6)
    assert code == 0 and case["status"] == "interrupted"
    assert case["files"] == task["paths"]
    result = run_script(
        "worker", "--out", str(out), "--case", task["line"], env=environment
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "took over the stale claim of 800_1" in result.stdout
    assert "removed 1 files of an earlier attempt" in result.stdout
    # The retired claim is claims/<tag>.stale.800_1, with or without a
    # further .<key> after the owner.
    retired = contract.claim_path(out, task["tag"] + ".stale.800_1")
    (match,) = retired.parent.glob(f"{retired.name}*")
    assert match.name == retired.name or match.name.startswith(
        f"{retired.name}."
    )
    code, _ = status_of(out, environment, 6)
    report = contract.read_json(out / "verification.json")
    assert code == 0 and report["counts"]["verified"] == len(tasks)
    rerun = contract.read_json(out / task["paths"][0])["cell"]
    first = campaign["results"]["cells"][task["cells"][0]]
    assert canonical(rerun) == canonical(first)


def test_a_stop_signal_cleans_up_the_task(campaign, tmp_path, monkeypatch):
    """SIGTERM, which Slurm sends at the time limit and on scancel, in the
    middle of a two-arm task: its written cell is removed, the original
    arm's subprocess ended, the claim released, no error file written, and
    ``verify`` counts the task as missing. Run in this process, which sends
    itself the signal, as a process on Windows must."""
    out, manifest, tasks = campaign_copy(campaign, tmp_path, missing=[1])
    task = tasks[0]
    environment = harness_environment()
    for key in (
        "PYTHONPATH",
        "PYVBMC_GPYREG_SOURCE",
        "BASELINE_DIR",
        "MPLBACKEND",
        *harness.THREAD_KEYS,
    ):
        if key in environment:
            monkeypatch.setenv(key, environment[key])
    written, servers = [], []
    write_cell, arm = harness.write_cell, harness.OriginalArm

    def write_then_stop(path, tag, row):
        write_cell(path, tag, row)
        written.append(Path(path))
        signal.raise_signal(signal.SIGTERM)

    def recorded_arm(*args, **kwargs):
        servers.append(arm(*args, **kwargs))
        return servers[-1]

    monkeypatch.setattr(harness, "write_cell", write_then_stop)
    monkeypatch.setattr(harness, "OriginalArm", recorded_arm)
    code = harness.main(["worker", "--out", str(out), "--case", task["line"]])
    assert code == 128 + signal.SIGTERM
    assert len(written) == 1 and not written[0].exists()
    assert not any((out / path).exists() for path in task["paths"])
    assert len(servers) == 1 and servers[0].process.poll() is not None
    assert not contract.claim_path(out, task["tag"]).exists()
    assert not contract.error_path(out, task["tag"]).exists()
    assert not contract.record_path(out, task["tag"]).exists()
    report = harness.campaign_report(out, manifest, tasks)
    assert report["cases"][0]["status"] == "missing"
    assert report["exit_code"] == 0


def test_assemble_merges_a_complete_campaign_only(campaign, tmp_path):
    out, _, _ = campaign_copy(campaign, tmp_path, missing=[4])
    result = run_script("assemble", "--out", str(out))
    assert result.returncode != 0
    assert "complete campaign only" in result.stderr
    assert not (out / "results.json").exists()


def test_verify_fails_a_changed_or_stray_cell(campaign, tmp_path):
    out, _, tasks = campaign_copy(campaign, tmp_path)
    changed = out / tasks[1]["paths"][0]
    changed.write_bytes(changed.read_bytes() + b"\n")
    stray = out / GROUPS[0] / f"M9_r0{harness.CELL_SUFFIX}"
    stray.write_text("{}", encoding="utf-8")
    result = run_script("verify", "--out", str(out))
    assert result.returncode == 1
    report = contract.read_json(out / "verification.json")
    assert report["cases"][1]["status"] == "verify_failed"
    assert "SHA-256" in report["cases"][1]["error"]
    assert report["stray"] == [f"{GROUPS[0]}/M9_r0{harness.CELL_SUFFIX}"]


def test_prepare_refuses_to_change_a_prepared_campaign(campaign, tmp_path):
    out, _, _ = campaign_copy(campaign, tmp_path)
    again = run_script("prepare", "--out", str(out), *CAMPAIGN_ARGUMENTS)
    assert again.returncode == 0, again.stdout + again.stderr
    assert "unchanged" in again.stdout
    other = run_script(
        "prepare", "--out", str(out), *CAMPAIGN_ARGUMENTS, "--seed", "1"
    )
    assert other.returncode != 0
    assert "prepared differently" in other.stderr
    assert "settings" in other.stderr and "plan" in other.stderr


def test_the_tracked_copies_of_a_campaign_are_redacted(campaign, tmp_path):
    """The finished campaign, given what a cluster leaves in it (the site
    block, the login node's and the nodes' host parts, the paths of the
    operator's trees and baseline, the task logs and the accounting), is
    copied for the repository with none of it and its results unchanged."""
    site = stubs.FakeSite(tmp_path)
    out = site.home / "runs" / "stacking"
    shutil.copytree(campaign["out"], out)
    manifest = contract.read_json(out / "manifest.json")
    assert manifest["tracked_copies"] == harness.TRACKED_COPIES
    checkout = str(site.home / "baseline" / "source")

    def at_site(value):
        value["site"] = site.site_block("dev/scripts/svbmc_pool_stack.py")
        site.plant(value["identity"])
        value["gpyreg_source"] = str(site.gpyreg)
        value["baseline"]["verification"]["checkout"] = checkout

    site.rewrite(out / "manifest.json", at_site)
    for index, line in enumerate(campaign["lines"], start=1):
        site.rewrite(
            contract.record_path(out, line.split()[0]),
            lambda record: site.plant(
                record["identity"], node=site.nodes[index % 2], task=index
            ),
        )

    def assembled_there(value):
        site.plant(value["identity"], node=site.nodes[0])
        value["gpyreg_source"] = str(site.gpyreg)
        value["baseline"]["verification"]["checkout"] = checkout
        for number, task in enumerate(value["tasks"].values()):
            task["host"]["hostname"] = site.nodes[number % 2]

    site.rewrite(out / "sources.json", assembled_there)
    site.write_slurm(out)
    assert site.leaks(out)
    target = tmp_path / "handback" / "stacking"
    # What lies outside the stand-in site: the checkout the campaign ran
    # from (the fixtures under it), the interpreter and its libraries, and
    # the baseline with its Torch overlay, which the records and sources
    # name. They are named here, as an operator names them with --path.
    outside = [
        ("CHECKOUT", str(ROOT)),
        ("PYTHON", sys.prefix),
        ("BASELINE", str(Path(BASELINE).resolve())),
    ]
    contract.redact(
        out,
        target,
        operator=site.operator(),
        environ={},
        paths=outside,
        host="fakelogin9",
        say=lambda message: None,
    )
    assert site.leaks(target) == []
    assert sorted(p.name for p in target.iterdir()) == [
        "manifest.json",
        "redaction.json",
        "results.json",
        "sources.json",
        "summary.json",
        "summary.md",
        "verification.json",
    ]
    for name in ("results.json", "summary.json", "summary.md"):
        assert (target / name).read_bytes() == (out / name).read_bytes()
    sources = contract.read_json(target / "sources.json")
    assert {
        task["host"]["hostname"] for task in sources["tasks"].values()
    } == {site.family}
    assert sources["baseline"]["verification"]["checkout"].startswith("~")
    copied = contract.read_json(target / "manifest.json")
    assert "site" not in copied and copied["plan"] == manifest["plan"]
    # A field no rule knows of, holding the username, is refused.
    site.rewrite(out / "results.json", lambda r: r.update(by=site.user))
    with pytest.raises(contract.ContractError, match="the username"):
        contract.redact(
            out,
            tmp_path / "handback" / "again",
            operator=site.operator(),
            environ={},
            paths=outside,
            say=lambda message: None,
        )
    assert not (tmp_path / "handback" / "again").exists()


# --------------------------------------------------------------------------
# A pool generated by svbmc_pool_run.py
# --------------------------------------------------------------------------

RUNNER = HERE / "svbmc_pool_run.py"
POOL_LABEL = "normal_D2"
#: The condition of the second pool: the noiseless GMM of the release
#: pools, whose runs take seconds.
SECOND_LABEL = "gmm_D2_svbmc"


def run_runner(*arguments):
    return subprocess.run(
        [sys.executable, "-u", str(RUNNER), *arguments],
        cwd=str(ROOT),
        env=harness_environment(),
        capture_output=True,
        text=True,
    )


def generated_pool(tmp_path_factory, name, suite, label):
    """A two-run pool of one condition, generated, verified and selected, as
    a campaign's finish leaves it."""
    out = tmp_path_factory.mktemp(name)
    prepared = run_runner(
        "prepare",
        "--out",
        str(out),
        "--suite",
        suite,
        "--only",
        label,
        "--target",
        "2",
        "--max-seeds",
        "2",
        "--seed-start",
        "4000",
        "--gpyreg-source",
        str(GPYREG_SOURCE),
        "--allow-dirty",
    )
    assert prepared.returncode == 0, prepared.stdout + prepared.stderr
    for command in ("run", "verify", "select"):
        result = run_runner(command, "--out", str(out))
        assert result.returncode == 0, result.stdout + result.stderr
    return out


@pytest.fixture(scope="module")
def pool(tmp_path_factory):
    """A two-run pool of the smoke configuration."""
    return generated_pool(tmp_path_factory, "svbmc_pool", "smoke", POOL_LABEL)


@pytest.fixture(scope="module")
def second_pool(tmp_path_factory):
    """A two-run pool of the noiseless GMM condition."""
    return generated_pool(
        tmp_path_factory, "svbmc_pool_gmm", "svbmc_pool", SECOND_LABEL
    )


#: The grid of the comparisons on the two pools: both arms, one cell per
#: condition at ``M = 2``.
POOL_GRID = ("--M", "2", "--repetitions", "1", "--max-steps", str(MAX_STEPS))


def pool_arguments(*pools):
    return [word for pool in pools for word in ("--pool", str(pool))]


@pytest.fixture(scope="module")
def pool_comparison(pool, second_pool, tmp_path_factory):
    """The single-process comparison on the two pools."""
    out = tmp_path_factory.mktemp("pool_comparison")
    result = run_script(
        *pool_arguments(pool, second_pool), "--out", str(out), *POOL_GRID
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return read_outputs(out) | {"out": out}


def other_gpyreg(tmp_path):
    """A git checkout of a package named gpyreg at a commit no pool names."""
    other = tmp_path / "other_gpyreg"
    (other / "gpyreg").mkdir(parents=True)
    (other / "gpyreg" / "__init__.py").write_text("\n", encoding="utf-8")
    author = [
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.com",
        "-c",
        "commit.gpgsign=false",
    ]
    subprocess.run(["git", "-C", str(other), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(other), *author, "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(other), *author, "commit", "-q", "-m", "x"],
        check=True,
    )
    return other


def relaid(pool, out, nested):
    """A copy of a pool in the flat layout of the September pools or in the
    per-condition layout of the campaign contract's, whichever the runner
    wrote: files, records and selection moved to the other tags."""
    shutil.copytree(pool, out, ignore=shutil.ignore_patterns("records"))
    records = sorted((pool / "records").rglob("*.complete.json"))
    moved = {}
    for path in records:
        record = json.loads(path.read_text(encoding="utf-8"))
        label, seed, old = record["label"], record["seed"], record["tag"]
        tag = (
            f"{label}/{label}_seed{seed}" if nested else f"{label}_seed{seed}"
        )
        moved[old] = tag
        for suffix in harness.RUN_SUFFIXES:
            source = out / f"{old}{suffix}"
            target = out / f"{tag}{suffix}"
            target.parent.mkdir(parents=True, exist_ok=True)
            if source != target:
                shutil.move(source, target)
        record["tag"] = tag
        target = out / "records" / f"{tag}.complete.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record), encoding="utf-8")
    selection = json.loads(
        (out / "selection.json").read_text(encoding="utf-8")
    )
    for condition in selection["conditions"]:
        for run in condition["runs"]:
            run["tag"] = moved[run["tag"]]
    (out / "selection.json").write_text(
        json.dumps(selection), encoding="utf-8"
    )
    return out, moved


@pytest.mark.parametrize("nested", [False, True], ids=["flat", "nested"])
def test_a_pool_is_read_in_either_layout(pool, tmp_path, nested):
    """A run's files are `<pool>/<tag>.npz` and `.json` whatever the tag
    holds, from the selection or, without one, from the records."""
    out, moved = relaid(pool, tmp_path / "pool", nested)
    conditions, identities, labels = harness.pool_conditions([out])
    assert labels == [POOL_LABEL]
    entries = conditions[POOL_LABEL]
    assert [e["name"] for e in entries] == sorted(
        moved.values(), key=lambda tag: int(tag.rsplit("seed", 1)[1])
    )
    assert identities[0]["selection"]["conditions"] == {
        POOL_LABEL: "selection.json"
    }
    for entry in entries:
        assert Path(entry["path"]) == out / entry["name"]
        files = harness.run_files(entry["path"])
        assert entry["sha256"] == {
            s: contract.sha256_file(p) for s, p in files.items()
        }
    (out / "selection.json").unlink()
    again, identities, _ = harness.pool_conditions([out])
    assert identities[0]["selection"]["conditions"] == {
        POOL_LABEL: "every passing record"
    }
    assert [e["name"] for e in again[POOL_LABEL]] == [
        e["name"] for e in entries
    ]


def test_a_pool_posterior_is_the_pool_readers(pool):
    """The harness rebuilds the posterior alone, and it is the one the pool's
    own reader rebuilds with the whole run; a changed file is refused."""
    import svbmc_pool_io as pool_io

    entries, _, _ = harness.pool_conditions([pool])
    entry = entries[POOL_LABEL][0]
    vp = harness.load_entry(entry, rng=5)
    full = pool_io.load_run(entry["path"], rng=5)["vp"]
    for name in ("w", "eta", "mu", "sigma", "lambd"):
        np.testing.assert_array_equal(getattr(vp, name), getattr(full, name))
    assert set(vp.stats) == set(full.stats)
    for key in ("I_sk", "J_sjk", "elbo", "stable"):
        np.testing.assert_array_equal(vp.stats[key], full.stats[key])
    assert vp.parameter_transformer == full.parameter_transformer
    np.testing.assert_array_equal(vp.sample(50)[0], full.sample(50)[0])
    tampered = dict(entry, sha256=dict(entry["sha256"], **{".npz": "0" * 64}))
    with pytest.raises(RuntimeError, match="differs from the SHA-256"):
        harness.load_entry(tampered, rng=5)


def copied_pool(pool, tmp_path):
    """The pool as a copy from another machine: its manifest names a
    gpyreg checkout that does not exist here."""
    copy = tmp_path / "pool"
    shutil.copytree(pool, copy)
    path = copy / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["gpyreg_source"] = str(tmp_path / "elsewhere" / "gpyreg_1.2.1")
    path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return copy


def run_pool_harness(copy, out, *extra, env=None):
    return run_script(
        "--pool",
        str(copy),
        "--out",
        str(out),
        "--M",
        "2",
        "--repetitions",
        "1",
        "--max-steps",
        str(MAX_STEPS),
        *extra,
        env=env,
    )


def test_pool_accepts_a_gpyreg_source_at_the_manifest_commit(pool, tmp_path):
    """A pool copied from another machine is stacked against a local
    checkout at its manifest's gpyreg commit, and nothing else."""
    copy = copied_pool(pool, tmp_path)
    manifest = json.loads((copy / "manifest.json").read_text(encoding="utf-8"))
    pinned = harness.pool_gpyreg_commit(manifest)
    assert pinned
    # The manifest's path is not here, so without a checkout named nothing
    # runs.
    result = run_pool_harness(
        copy,
        tmp_path / "out_manifest",
        env=harness_environment(drop=("PYVBMC_GPYREG_SOURCE",)),
    )
    assert result.returncode != 0
    assert "no gpyreg package under" in result.stderr
    # A checkout at another commit is refused by name, before any cell.
    other = other_gpyreg(tmp_path)
    result = run_pool_harness(
        copy, tmp_path / "out_other", "--gpyreg-source", str(other)
    )
    assert result.returncode != 0
    assert f"not the manifest's {pinned}" in result.stderr
    assert not (tmp_path / "out_other" / "results.json").exists()
    # The frozen worktree is at the pin, so the comparison runs on it.
    out = tmp_path / "out_pinned"
    result = run_pool_harness(copy, out, "--gpyreg-source", str(GPYREG_SOURCE))
    assert result.returncode == 0, result.stdout + result.stderr
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert results["kind"] == "run"
    assert [cell["condition"] for cell in results["cells"]] == [POOL_LABEL]
    assert all(arm in results["cells"][0]["arms"] for arm in harness.ARMS)
    sources = json.loads((out / "sources.json").read_text(encoding="utf-8"))
    assert sources["gpyreg_source"] == str(GPYREG_SOURCE.resolve())
    assert sources["gpyreg_source_origin"] == "--gpyreg-source"
    assert sources["pools"][0]["gpyreg_source"] == manifest["gpyreg_source"]
    assert sources["pools"][0]["gpyreg_commit"] == pinned
    expected = str((GPYREG_SOURCE / "gpyreg").resolve())
    for arm in harness.ARMS:
        environment = sources["arms"][arm]["environment"]
        assert environment["host"]["gpyreg_import"] == expected
        assert environment["source"]["gpyreg_commit"] == pinned


def test_a_campaign_on_two_pools_equals_the_single_process_run(
    pool, second_pool, pool_comparison, tmp_path
):
    """The contract on two pools of the campaign contract's layout: the
    runs' hashes and the pools' verifications in the manifest, a changed
    run refused by the worker, the single-run rows from the pool records,
    and the assembly of tasks run in processes of their own equal to the
    single-process run's cells, rows and summaries."""
    pools = (pool, second_pool)
    out = tmp_path / "campaign"
    arguments = (*pool_arguments(*pools), *POOL_GRID, "--arms", "both")
    prepared = run_script(
        "prepare", "--out", str(out), *arguments, "--allow-dirty"
    )
    assert prepared.returncode == 0, prepared.stdout + prepared.stderr
    manifest = contract.read_json(out / "manifest.json")
    assert manifest["kind"] == "run"
    assert manifest["allow_dirty"] is True
    assert [record["directory"] for record in manifest["pools"]] == [
        str(directory.resolve()) for directory in pools
    ]
    for record, directory in zip(manifest["pools"], pools):
        assert record["verification"]["verification_sha256"] == (
            contract.sha256_file(directory / "verification.json")
        )
    assert [c["label"] for c in manifest["conditions"]] == [
        POOL_LABEL,
        SECOND_LABEL,
    ]
    assert set(manifest["identity"]["source"]["trees"]) == {
        "harness",
        "gpyreg",
        "baseline",
    }
    lines = run_script("cases", "--out", str(out)).stdout.splitlines()
    assert [line.split()[0] for line in lines] == [
        f"{POOL_LABEL}/M2",
        f"{SECOND_LABEL}/M2",
    ]
    entry = manifest["conditions"][0]["entries"][0]
    npz = harness.run_files(entry["path"])[".npz"]
    kept = npz.read_bytes()
    npz.write_bytes(kept + b"\0")
    try:
        failed = run_script("worker", "--out", str(out), "--case", lines[0])
    finally:
        npz.write_bytes(kept)
    assert failed.returncode == 1
    error = contract.error_path(out, lines[0].split()[0]).read_text("utf-8")
    assert "differs from the SHA-256" in error
    for line in reversed(lines):
        result = run_script("worker", "--out", str(out), "--case", line)
        assert result.returncode == 0, line + result.stdout + result.stderr
    for command in ("verify", "assemble"):
        result = run_script(command, "--out", str(out))
        assert result.returncode == 0, result.stdout + result.stderr
    assert not contract.error_path(out, lines[0].split()[0]).exists()
    split, whole = read_outputs(out), pool_comparison
    assert split["results"]["settings"] == whole["results"]["settings"]
    assert split["results"]["skipped"] == whole["results"]["skipped"]
    assert canonical(split["results"]["single_run"]) == canonical(
        whole["results"]["single_run"]
    )
    cells = split["results"]["cells"], whole["results"]["cells"]
    assert [(c["condition"], c["repetition"]) for c in cells[0]] == [
        (POOL_LABEL, 0),
        (SECOND_LABEL, 0),
    ]
    assert len(cells[1]) == len(cells[0])
    for a, b in zip(*cells):
        assert set(a["arms"]) == set(harness.ARMS)
        assert canonical(a) == canonical(b), a["condition"]
    for key in ("conditions", "equivalence_tests", "settings"):
        assert canonical(split["summary"][key]) == canonical(
            whole["summary"][key]
        )
    for condition in manifest["conditions"]:
        rows = split["results"]["single_run"][condition["label"]]
        directory = Path(condition["entries"][0]["path"]).parents[1]
        records = [
            contract.read_json(
                directory / "records" / f"{e['name']}.complete.json"
            )
            for e in condition["entries"]
        ]
        assert [row["name"] for row in rows] == [r["tag"] for r in records]
        for row, record in zip(rows, records):
            for key in ("elbo_err", "gskl", "mmtv"):
                assert row[key] == record["metrics"][key]


def test_prepare_stacks_a_pool_only_after_its_verification(pool, tmp_path):
    """A pool without a passing verification, or selected before it, is
    refused before anything is written; selected after it, it is taken."""
    copy = tmp_path / "pool"
    shutil.copytree(pool, copy)
    out = tmp_path / "campaign"
    arguments = ("--pool", str(copy), *POOL_GRID, "--arms", "integrated")

    def prepare():
        return run_script(
            "prepare", "--out", str(out), *arguments, "--allow-dirty"
        )

    (copy / "verification.json").unlink()
    refused = prepare()
    assert refused.returncode != 0
    assert "holds no verification.json" in refused.stderr
    assert run_runner("verify", "--out", str(copy)).returncode == 0
    refused = prepare()
    assert refused.returncode != 0
    assert "another verification.json" in refused.stderr
    assert not out.exists()
    assert run_runner("select", "--out", str(copy)).returncode == 0
    accepted = prepare()
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr


def test_prepare_refuses_a_dirty_harness_without_allow_dirty(
    tmp_path, monkeypatch
):
    """Only the gpyreg and baseline trees must be clean; a dirty harness
    checkout needs --allow-dirty, which the manifest records."""
    real = harness.campaign_identity

    def dirty(*args, **kwargs):
        record = real(*args, **kwargs)
        record["source"]["trees"]["harness"]["clean"] = False
        record["imports"]["trees"]["harness"]["dirty"] = [" M somewhere.py"]
        return record

    monkeypatch.setattr(harness, "campaign_identity", dirty)
    out = tmp_path / "campaign"
    arguments = [
        "prepare",
        "--out",
        str(out),
        "--fixtures",
        GROUPS[0],
        "--M",
        "2",
        "--repetitions",
        "1",
        "--arms",
        "integrated",
    ]
    with pytest.raises(RuntimeError, match="--allow-dirty") as refused:
        harness.main(arguments)
    assert "harness checkout ( M somewhere.py)" in str(refused.value)
    assert not (out / "manifest.json").exists()
    assert harness.main([*arguments, "--allow-dirty"]) == 0
    assert contract.read_json(out / "manifest.json")["allow_dirty"] is True


def test_the_worker_refuses_a_line_outside_the_campaign(campaign, tmp_path):
    """A line that is not a case, or a directory that is not a campaign of
    this harness, exits 64 and touches nothing."""
    out, _, tasks = campaign_copy(campaign, tmp_path, missing=[2])
    before = snapshot(out)
    for line in (
        tasks[1]["line"] + " extra",
        tasks[1]["tag"],
        "no_such_condition/M2 M=2 repetitions=0-1 arms=integrated",
    ):
        result = run_script("worker", "--out", str(out), "--case", line)
        assert result.returncode == contract.EXIT_USAGE, (
            line + result.stdout + result.stderr
        )
        assert "is not a case of" in result.stderr
    assert snapshot(out) == before
    other = tmp_path / "other"
    other.mkdir()
    contract.write_json(other / "manifest.json", {"campaign": "svbmc_pool"})
    result = run_script(
        "worker", "--out", str(other), "--case", tasks[1]["line"]
    )
    assert result.returncode == contract.EXIT_USAGE
    assert "is not a manifest of" in result.stderr
    assert sorted(p.name for p in other.iterdir()) == ["manifest.json"]


def test_an_original_arm_that_died_says_how(tmp_path):
    """The error of a subprocess that ended without replying gives its exit
    code, and the signal that killed it where one did."""

    def ended(code):
        arm = object.__new__(harness.OriginalArm)
        arm.process = subprocess.Popen(
            [sys.executable, "-c", code], stdout=subprocess.PIPE, text=True
        )
        try:
            with pytest.raises(RuntimeError) as error:
                arm._read()
        finally:
            arm.process.stdout.close()
        return str(error.value)

    assert "exited with code 3" in ended("import sys; sys.exit(3)")
    if hasattr(signal, "SIGKILL"):
        message = ended("import os, signal; os.kill(os.getpid(), 9)")
        assert "was killed by SIGKILL (exit code -9)" in message


READERS = (
    "svbmc_cap_kappa.py",
    "svbmc_shrink_elbo.py",
    "svbmc_single_run_bias.py",
    "svbmc_shrink_optimize.py",
)


@pytest.mark.parametrize("script", READERS)
def test_a_pool_reader_reads_the_contract_layout(
    script, pool, pool_comparison, tmp_path
):
    """Each script that reads a pool and a comparison's cells finds the runs
    of a pool of the campaign contract, whose tags name the condition's
    directory, and refuses a gpyreg checkout at a commit other than the one
    that pool's manifest records."""
    cells = pool_comparison["out"] / "results.json"
    out = tmp_path / "out"
    common = ["--pool", str(pool), "--out", str(out), "--cells", str(cells)]
    common += ["--conditions", POOL_LABEL]

    def run(*extra):
        return subprocess.run(
            [sys.executable, "-u", str(HERE / script), *common, *extra],
            cwd=str(ROOT),
            env=harness_environment(),
            capture_output=True,
            text=True,
        )

    refused = run("--gpyreg-source", str(other_gpyreg(tmp_path)))
    assert refused.returncode != 0
    assert "not the manifest's" in refused.stderr
    assert not out.exists()
    result = run("--gpyreg-source", str(GPYREG_SOURCE))
    assert result.returncode == 0, result.stdout + result.stderr
    stacked = [
        cell
        for cell in pool_comparison["results"]["cells"]
        if cell["condition"] == POOL_LABEL
    ]
    assert stacked and all(
        "/" in name for cell in stacked for name in cell["entries"]
    )
    rows = [
        json.loads(line)
        for line in (out / "cells.jsonl").read_text("utf-8").splitlines()
    ]
    assert len(rows) == len(stacked)
    assert {row["condition"] for row in rows} == {POOL_LABEL}
    if script == "svbmc_single_run_bias.py":
        runs = [
            json.loads(line)
            for line in (out / "runs.jsonl").read_text("utf-8").splitlines()
        ]
        selection = contract.read_json(pool / "selection.json")
        assert sorted(run["name"] for run in runs) == sorted(
            run["tag"] for run in selection["conditions"][0]["runs"]
        )
    assert (out / "summary.json").exists() and (out / "sources.json").exists()
