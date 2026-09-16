"""Contracts of the S-VBMC stacking comparison: schema, pairing, agreement.

One short comparison is run once for the whole module through the command
line, on the two upstream fixture groups instead of a pool (the harness's
``--fixtures`` mode), with two and three posteriors per cell and three Adam
steps. The point is that both implementations run, that every cell is
paired, and that the outputs carry what the campaign's analysis reads, not
the quality of a three-step fit. Two further checks need no cell: that
``--summarize-only`` rebuilds the same tables from the recorded results,
and that a fixture group holds the fixtures its sidecars name. Outside
default pytest discovery; run it by path, with Torch from the recorded
overlay::

    PYTHONPATH="<TORCH_PATH>" python -m pytest \\
        dev/scripts/test_svbmc_pool_stack.py -vv

``TORCH_PATH`` is ``dev/experiments/svbmc_pool/baseline_environment.json``'s
Torch overlay, which this module also puts on the harness's path itself, so
the tests pass whether or not pytest was started with it.
"""

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from svbmc_pool_run import DEFAULT_GPYREG, THREAD_KEYS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRIPT = HERE / "svbmc_pool_stack.py"
GPYREG_SOURCE = DEFAULT_GPYREG
BASELINE_RECORD = (
    ROOT / "dev" / "experiments" / "svbmc_pool" / "baseline_environment.json"
)
GROUPS = ("upstream_GMM_noisy", "upstream_Ring")
GRID = (2, 3)
REPETITIONS = (2, 2)
MAX_STEPS = 3
POOL_SIZE = 10  # posteriors per upstream fixture group

# The campaign pins gpyreg to a frozen worktree; the harness reads the
# variable when it imports PyVBMC, and the arms inherit it.
if (GPYREG_SOURCE / "gpyreg").is_dir():
    os.environ["PYVBMC_GPYREG_SOURCE"] = str(GPYREG_SOURCE)

import svbmc_pool_stack as harness  # noqa: E402

# Loose bound on the paired weight difference: Phase 1 of the campaign
# measured at most 0.009 on these groups at three Adam steps against a
# within-arm seed-to-seed spread of the same size, so anything of this
# order means both arms optimized the same objective on the same inputs.
WEIGHT_TOLERANCE = 0.05


def _overlay():
    """The recorded Torch overlay, or ``None`` when it is not on this box."""
    if not BASELINE_RECORD.exists():
        return None
    record = json.loads(BASELINE_RECORD.read_text(encoding="utf-8"))
    overlay = Path(record["torch"]["overlay_dir"])
    return overlay if (overlay / "torch" / "__init__.py").exists() else None


OVERLAY = _overlay()
# Two checks below call the integrated class in this process, so the
# overlay's Torch must be importable here whether or not pytest was
# started with it on PYTHONPATH. Appended, not prepended: with PYTHONPATH
# set this is a no-op, and without it nothing the venv provides is
# shadowed.
if OVERLAY is not None and str(OVERLAY) not in sys.path:
    sys.path.append(str(OVERLAY))
pytestmark = pytest.mark.skipif(
    OVERLAY is None or not (GPYREG_SOURCE / "gpyreg").is_dir(),
    reason="the baseline Torch overlay and the frozen gpyreg worktree of "
    "the S-VBMC campaign are machine-local",
)


def harness_environment():
    """The environment every invocation of the harness is given here."""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(OVERLAY), *[p for p in [environment.get("PYTHONPATH")] if p]]
    )
    environment["PYVBMC_GPYREG_SOURCE"] = str(GPYREG_SOURCE)
    environment["MPLBACKEND"] = "Agg"
    for key in THREAD_KEYS:
        environment[key] = "1"
    return environment


def run_script(*arguments):
    return subprocess.run(
        [sys.executable, "-u", str(SCRIPT), *arguments],
        cwd=str(ROOT),
        env=harness_environment(),
        capture_output=True,
        text=True,
    )


def dateless(text):
    """The lines of a summary that do not carry the date it was built."""
    return [
        line for line in text.splitlines() if not line.startswith("Generated ")
    ]


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
    return {
        name: json.loads((out / f"{name}.json").read_text(encoding="utf-8"))
        for name in ("results", "summary", "sources")
    } | {"out": out, "stdout": completed.stdout}


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
    assert {c["first_arm"] for c in cells} == {"integrated", "original"}


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
    assert (
        (comparison["out"] / "summary.md")
        .read_text(encoding="utf-8")
        .startswith("# S-VBMC stacking comparison")
    )


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
    # worker, and in nothing the source half pins.
    assert (
        integrated["environment"]["source"]
        == original["environment"]["source"]
    )
    assert original["svbmc_version"] == "0.1.1"
    upstream = sources["path_sets"]["BASELINE_PATH"].split(os.pathsep)[-1]
    assert Path(original["svbmc_import"]).parent.parent == Path(upstream)
    assert upstream not in (integrated["pythonpath"] or "")
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


RUNNER = HERE / "svbmc_pool_run.py"
POOL_LABEL = "normal_D2"


def run_runner(*arguments):
    return subprocess.run(
        [sys.executable, "-u", str(RUNNER), *arguments],
        cwd=str(ROOT),
        env=harness_environment(),
        capture_output=True,
        text=True,
    )


@pytest.fixture(scope="module")
def pool(tmp_path_factory):
    """A two-run pool of the smoke configuration, generated and selected."""
    out = tmp_path_factory.mktemp("svbmc_pool")
    prepared = run_runner(
        "prepare",
        "--out",
        str(out),
        "--suite",
        "smoke",
        "--only",
        POOL_LABEL,
        "--target",
        "2",
        "--max-seeds",
        "2",
        "--seed-start",
        "4000",
        "--allow-dirty",
    )
    assert prepared.returncode == 0, prepared.stdout + prepared.stderr
    for command in ("run", "select"):
        result = run_runner(command, "--out", str(out))
        assert result.returncode == 0, result.stdout + result.stderr
    return out


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


def run_pool_harness(copy, out, *extra):
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
    )


def test_pool_accepts_a_gpyreg_source_at_the_manifest_commit(pool, tmp_path):
    """A pool copied from another machine is stacked against a local
    checkout at its manifest's gpyreg commit, and nothing else."""
    copy = copied_pool(pool, tmp_path)
    manifest = json.loads((copy / "manifest.json").read_text(encoding="utf-8"))
    pinned = manifest["identity"]["source"]["gpyreg_commit"]
    # The manifest's path is not here, so without the flag nothing runs.
    result = run_pool_harness(copy, tmp_path / "out_manifest")
    assert result.returncode != 0
    assert "no gpyreg package under" in result.stderr
    # A checkout at another commit is refused by name, before any cell.
    other = tmp_path / "other_gpyreg"
    (other / "gpyreg").mkdir(parents=True)
    author = [
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.com",
        "-c",
        "commit.gpgsign=false",
    ]
    subprocess.run(["git", "-C", str(other), "init", "-q"], check=True)
    (other / "gpyreg" / "__init__.py").write_text("\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(other), *author, "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(other), *author, "commit", "-q", "-m", "x"],
        check=True,
    )
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
    expected = str((GPYREG_SOURCE / "gpyreg").resolve())
    for arm in harness.ARMS:
        environment = sources["arms"][arm]["environment"]
        assert environment["host"]["gpyreg_import"] == expected
        assert environment["source"]["gpyreg_commit"] == pinned


@pytest.fixture(scope="module")
def integrated_only(tmp_path_factory):
    """The same comparison run with the integrated arm alone."""
    out = tmp_path_factory.mktemp("svbmc_stack_integrated")
    completed = run_harness(out, "--arms", "integrated")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return {
        name: json.loads((out / f"{name}.json").read_text(encoding="utf-8"))
        for name in ("results", "summary", "sources")
    } | {"out": out, "stdout": completed.stdout}


def test_integrated_arm_alone(comparison, integrated_only):
    """`--arms integrated`: one arm per cell, no paired quantity, and the
    same integrated fits as the two-arm run of the same cells."""
    results = integrated_only["results"]
    assert results["settings"]["arms"] == ["integrated"]
    assert comparison["results"]["settings"]["arms"] == list(harness.ARMS)
    cells = results["cells"]
    assert len(cells) == len(GROUPS) * sum(REPETITIONS)
    for cell in cells:
        assert set(cell["arms"]) == {"integrated"}
        assert cell["first_arm"] == "integrated"
        assert cell["max_abs_dw"] is None
        outcome = cell["arms"]["integrated"]
        assert all(key in outcome for key in harness.REFERENCE_FIELDS)
    assert "the integrated arm alone" in integrated_only["stdout"]
    assert not (integrated_only["out"] / "original_arm.log").exists()
    assert integrated_only["sources"]["arms"]["original"] is None
    # The integrated arm is seeded by the cell and rebuilds its posteriors,
    # so the same cells give the same fit whether or not the original arm
    # runs beside it.
    twins = {
        (c["condition"], c["M"], c["repetition"]): c
        for c in comparison["results"]["cells"]
    }
    for cell in cells:
        twin = twins[(cell["condition"], cell["M"], cell["repetition"])]
        assert cell["entry_seeds"] == twin["entry_seeds"]
        np.testing.assert_allclose(
            cell["arms"]["integrated"]["w"],
            twin["arms"]["integrated"]["w"],
            rtol=0,
            atol=1e-8,
        )
        assert cell["arms"]["integrated"]["elbo_mc"] == pytest.approx(
            twin["arms"]["integrated"]["elbo_mc"], abs=1e-6
        )
    summary = integrated_only["summary"]
    assert summary["equivalence_tests"] == []
    for condition in summary["conditions"]:
        assert condition["all_M"]["runtime_ratio"]["n"] == 0
        assert condition["all_M"]["max_abs_dw"]["n"] == 0
        for entry in condition["M"]:
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
    text = (integrated_only["out"] / "summary.md").read_text(encoding="utf-8")
    for condition in summary["conditions"]:
        assert f"## {condition['condition']}" in text


def filtered_results(source, out, keep_M):
    """A copy of a run's results file holding only the cells of some M."""
    results = json.loads((source / "results.json").read_text(encoding="utf-8"))
    results["cells"] = [c for c in results["cells"] if c["M"] in keep_M]
    settings = results["settings"]
    kept = [
        (M, R)
        for M, R in zip(settings["M"], settings["repetitions"])
        if M in keep_M
    ]
    settings["M"] = [M for M, _ in kept]
    settings["repetitions"] = [R for _, R in kept]
    out.mkdir(parents=True, exist_ok=True)
    path = out / "results.json"
    path.write_text(json.dumps(results), encoding="utf-8")
    return path


def test_summarize_only_merges_a_two_arm_run_with_an_integrated_run(
    comparison, integrated_only, tmp_path
):
    """Both arms up to one M and the integrated arm beyond it summarize
    together: paired quantities where both ran, growth across every M."""
    low = filtered_results(comparison["out"], tmp_path / "low", {GRID[0]})
    high = filtered_results(
        integrated_only["out"], tmp_path / "high", {GRID[1]}
    )
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
        str(integrated_only["out"] / "results.json"),
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
