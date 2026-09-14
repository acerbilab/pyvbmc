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

import hashlib
import json
import os
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
                arm["elbo_mc"], arm["e_log_joint_mc"] + arm["entropy"]
            )
            for key in ("construction", "optimize", "sample"):
                assert arm[f"{key}_seconds"] > 0
            if name == "integrated":
                assert {"raw", "capped_I_median", "naive"} <= set(elbos)
                assert arm["elbo_sd"] > 0
            else:
                assert {"estimated", "debiased_I_median"} <= set(elbos)


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
    completed = run_script("--summarize-only", "--out", str(out))
    assert completed.returncode == 0, completed.stdout + completed.stderr
    after = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    # The bootstrap is seeded from the settings the run recorded, so the
    # rebuilt tables are the same numbers.
    assert after["conditions"] == before["conditions"]
    assert after["equivalence_tests"] == before["equivalence_tests"]
    assert after["settings"] == before["settings"]


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
    assert integrated["environment"]["gpyreg_import"] == expected
    assert original["environment"]["gpyreg_import"] == expected
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
