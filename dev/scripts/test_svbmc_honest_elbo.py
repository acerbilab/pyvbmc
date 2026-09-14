"""Contracts of the cross-run expected-log-joint estimator.

One short pool (``normal_D2``, two seeds, about a minute of inference) is
generated once for the whole module through the pool generator's command
line, and a cells file in the shape of ``svbmc_pool_stack.py``'s
``results.json`` is written for it with naive-stacking weights, so that the
estimator runs end to end without Torch or a real comparison. The checks
are the ones a real campaign relies on: the own-run Monte Carlo values
reproduce the stored statistics, the recomputed raw expected log joint
equals the recorded one, a Gaussian target fitted by two runs is fully
cross-covered and estimated within a small tolerance of the truth, the
outputs carry what the analysis reads, ``--summarize-only`` rebuilds the
same summary, and ``--self-check`` runs without cells. Outside default
pytest discovery; run it by path::

    python -m pytest dev/scripts/test_svbmc_honest_elbo.py -vv
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import svbmc_pool_run as runner

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRIPT = HERE / "svbmc_honest_elbo.py"
GPYREG_SOURCE = runner.DEFAULT_GPYREG
LABEL = "normal_D2"
SEED_START = 4100
SEEDS = 2
CELL_SEED = 123
DRAWS = 200
HEADLINE = f"ratio{2.0:g}"

# The campaign pins gpyreg to a frozen worktree: the estimator's
# subprocesses read the variable, and this process gets the same pin on
# its path before anything imports PyVBMC.
if (GPYREG_SOURCE / "gpyreg").is_dir():
    runner.activate_gpyreg(GPYREG_SOURCE)

import svbmc_honest_elbo as honest  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (GPYREG_SOURCE / "gpyreg").is_dir(),
    reason="the campaign's frozen gpyreg worktree is machine-local",
)


def cli(script, *args):
    environment = dict(os.environ)
    environment.update({k: "1" for k in runner.THREAD_KEYS})
    environment["MPLBACKEND"] = "Agg"
    environment["PYVBMC_GPYREG_SOURCE"] = str(GPYREG_SOURCE)
    return subprocess.run(
        [sys.executable, "-u", str(script), *args],
        cwd=str(ROOT),
        env=environment,
        capture_output=True,
        text=True,
    )


@pytest.fixture(scope="module")
def pool(tmp_path_factory):
    """A two-run pool on the smoke suite's Gaussian target."""
    out = tmp_path_factory.mktemp("honest_pool")
    prepared = cli(
        runner.__file__,
        "prepare",
        "--out",
        str(out),
        "--suite",
        "smoke",
        "--only",
        LABEL,
        "--target",
        str(SEEDS),
        "--max-seeds",
        str(SEEDS),
        "--seed-start",
        str(SEED_START),
        # The estimator and the pool scripts are developed together, so
        # this pool is generated from whatever the tree holds.
        "--allow-dirty",
    )
    assert prepared.returncode == 0, prepared.stdout + prepared.stderr
    result = cli(
        runner.__file__, "run", "--out", str(out), "--pilot-seeds", str(SEEDS)
    )
    assert result.returncode == 0, result.stdout + result.stderr
    tags = sorted(
        p.name[: -len(".complete.json")]
        for p in (out / "records").glob("*.complete.json")
    )
    assert len(tags) == SEEDS
    return out, tags


@pytest.fixture(scope="module")
def cells(pool):
    """A cells file with naive-stacking weights for the pool's two runs.

    The reference terms are computed here from the runs' posteriors:
    ``e_log_joint_mc`` from draws of the naive mixture through the target,
    the entropy terms set to zero (they cancel in every bias), and the raw
    ELBO from the stored corrected expected log joints, which is what the
    integrated class records.
    """
    from benchmark_targets import find_config
    from svbmc_pool_io import load_run

    from pyvbmc.svbmc._jacobian import expected_log_jacobian

    out, tags = pool
    problem = find_config(LABEL).make(seed=0)
    runs = [load_run(out / tag, rng=i) for i, tag in enumerate(tags)]
    weights, I_corr, samples = [], [], []
    for state in runs:
        vp = state["vp"]
        w_run = np.ravel(vp.w) / np.sum(vp.w) / len(runs)
        weights.append(w_run)
        I_corr.append(
            np.mean(np.asarray(vp.stats["I_sk"]), axis=0)
            - expected_log_jacobian(vp)
        )
        samples.append(vp.sample(20_000, orig_flag=True)[0])
    w = np.concatenate(weights)
    G_raw = float(w @ np.concatenate(I_corr))
    draws = np.concatenate(samples)
    e_mc = np.asarray(problem.log_density_vec(draws), dtype=float)
    mean, sd = float(np.mean(e_mc)), float(
        np.std(e_mc, ddof=1) / np.sqrt(e_mc.size)
    )

    def arm(names):
        return {
            "K": [int(s["vp"].K) for s in runs],
            "w": w.tolist(),
            "elbos": {name: G_raw for name in names},
            "entropy": 0.0,
            "entropy_ref": 0.0,
            "entropy_ref_sd": 0.0,
            "e_log_joint_mc": mean,
            "e_log_joint_mc_sd": sd,
            "elbo_mc": mean,
            "elbo_mc_sd": sd,
            "kl_gap": float(problem.ln_Z - mean),
            "bias": {name: G_raw - mean for name in names},
        }

    cell = {
        "condition": LABEL,
        "condition_index": 0,
        "M": len(runs),
        "repetition": 0,
        "entries": tags,
        "seeds": [int(t.rsplit("seed", 1)[1]) for t in tags],
        "cell_seed": CELL_SEED,
        # Both arms' records, with each arm's variant names, so that
        # either can be scored; the original arm's values are the same.
        "arms": {
            arm_name: arm(list(honest.VARIANTS[arm_name].values()))
            for arm_name in honest.VARIANTS
        },
    }
    path = out / "cells_results.json"
    path.write_text(
        json.dumps({"cells": [cell], "settings": {"seed": 0}}, indent=1),
        encoding="utf-8",
    )
    return path


@pytest.fixture(scope="module")
def scored(pool, cells, tmp_path_factory):
    out, _ = pool
    target = tmp_path_factory.mktemp("honest_out")
    result = cli(
        SCRIPT,
        "--pool",
        str(out),
        "--cells",
        str(cells),
        "--out",
        str(target),
        "--draws",
        str(DRAWS),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return target, result.stdout


def test_frozen_gpyreg_is_the_one_imported():
    import gpyreg

    assert (
        Path(gpyreg.__file__).resolve().parent
        == (GPYREG_SOURCE / "gpyreg").resolve()
    )


def test_outputs_and_settings(scored):
    target, stdout = scored
    for name in ("results.json", "summary.json", "summary.md", "sources.json"):
        assert (target / name).exists(), name
    results = json.loads((target / "results.json").read_text(encoding="utf-8"))
    settings = results["settings"]
    assert settings["draws"] == DRAWS
    assert set(settings["rules"]) == set(
        honest.coverage_rules(honest.RATIOS, honest.SD_CAP)
    )
    assert settings["headline_rule"] == HEADLINE in settings["rules"]
    assert len(results["runs"]) == SEEDS and len(results["cells"]) == 1
    assert results["skipped"] == []
    lines = (target / "cells.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["condition"] == LABEL
    figures = sorted(p.name for p in (target / "figures").glob("*.png"))
    assert figures == [
        "bias_by_condition.png",
        "calibration.png",
        "component_errors.png",
        "coverage.png",
        "sparsity.png",
    ]
    assert "[cell] normal_D2 M=2 r=0" in stdout
    sources = json.loads((target / "sources.json").read_text(encoding="utf-8"))
    assert sources["gpyreg"]["import"].startswith(str(GPYREG_SOURCE.resolve()))
    assert sources["cells"]["sha256"]
    assert sources["threads"] == {k: "1" for k in runner.THREAD_KEYS}


def test_own_run_values_reproduce_the_stored_statistics(scored):
    target, _ = scored
    results = json.loads((target / "results.json").read_text(encoding="utf-8"))
    for run in results["runs"]:
        checks = run["checks"]
        assert not checks["flagged"], run["tag"]
        assert abs(checks["own_offset_z"]) < honest.Z_FLAG
        assert checks["own_mc_max_z"] < honest.Z_FLAG_MAX
        # An unbounded, unwarped or warped Gaussian run has a Jacobian that
        # is constant over every component: exact agreement.
        assert checks["jac_max_abs"] < 1e-9
    cell = results["cells"][0]
    assert not cell["checks"]["flagged"]
    assert abs(cell["checks"]["raw_consistency"]) < honest.RAW_TOLERANCE


def test_two_runs_on_a_gaussian_cover_each_other(scored):
    target, _ = scored
    results = json.loads((target / "results.json").read_text(encoding="utf-8"))
    cell = results["cells"][0]
    headline = cell["honest"][HEADLINE][honest.HEADLINE_METHOD]
    # Exact likelihoods: both GPs are certain to thousandths of a nat on
    # every component, below the rule's absolute floor, so each run
    # covers the other whatever the ratio of their two tiny SDs.
    assert headline["weight_covered"] > 0.9, [
        (r["tag"], r["pred_sd_median"]) for r in results["runs"]
    ]
    # Two exact-likelihood runs on a Gaussian: the other run's GP knows
    # the target where the component sits to a few thousandths of a nat.
    truth = cell["truth_strat"]["G"]
    assert abs(headline["G"] - truth) < 0.02
    assert len(cell["decomposition"]) == SEEDS
    for entry in cell["decomposition"]:
        assert entry["own"]["n_within"] > 0
        assert set(entry["others"]) == set(cell["entries"]) - {entry["run"]}
    assert abs(cell["truth_strat"]["bias"]) < 5 * np.hypot(
        cell["truth_strat"]["se"], cell["reference"]["e_log_joint_mc_sd"]
    )
    # The `none` rule (every other run, no coverage test) is a superset.
    none = cell["honest"]["none"][honest.HEADLINE_METHOD]
    assert none["weight_covered"] >= headline["weight_covered"]
    for rule in cell["honest"].values():
        for method in rule.values():
            assert np.isfinite(method["G"]) and method["mc_se"] >= 0


def test_per_component_arrays(scored):
    target, _ = scored
    results = json.loads((target / "results.json").read_text(encoding="utf-8"))
    cell = results["cells"][0]
    path = target / "cells" / honest.cell_filename(cell)
    with np.load(path, allow_pickle=False) as data:
        M, K_total = data["est"].shape
        assert M == SEEDS and K_total == cell["K_total"]
        for name in (
            "se",
            "se_diff",
            "v",
            "mean_fn",
            "n_eff",
            "n_within",
            "data_offset",
            "cover",
            "a",
        ):
            assert data[name].shape == (M, K_total), name
        assert np.all(data["v"] >= 0) and np.all(data["n_eff"] >= 0)
        assert np.all(data["n_within"] >= 0)
        assert np.allclose(data["w"].sum(), 1.0)
        assert np.all(data["a"].sum(axis=0)[data["covered"]] > 0.999)
        assert not np.any(data["cover"][data["run_index"], np.arange(K_total)])
        assert np.isclose(
            float(data["w"] @ data["honest_G"]),
            cell["honest"][HEADLINE][honest.HEADLINE_METHOD]["G"],
        )


def test_summarize_only_rebuilds_the_summary(scored):
    target, _ = scored
    before = json.loads((target / "summary.json").read_text(encoding="utf-8"))
    result = cli(
        SCRIPT, "--summarize-only", "--out", str(target), "--no-figures"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    after = json.loads((target / "summary.json").read_text(encoding="utf-8"))
    before.pop("generated")
    after.pop("generated")
    assert before == after
    markdown = (target / "summary.md").read_text(encoding="utf-8")
    assert "## Headline across conditions" in markdown
    assert f"| {LABEL} | {SEEDS} | 1 |" in markdown


def test_self_check_runs_without_cells(pool, tmp_path):
    out, tags = pool
    target = tmp_path / "self"
    result = cli(
        SCRIPT, "--pool", str(out), "--out", str(target), "--self-check"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    results = json.loads((target / "results.json").read_text(encoding="utf-8"))
    assert [r["tag"] for r in results["runs"]] == tags
    assert results["cells"] == []
    assert results["settings"]["cells"] is None
    markdown = (target / "summary.md").read_text(encoding="utf-8")
    assert "--self-check" in markdown
    # The run table is written even though no cell was scored.
    assert all(f"| {tag} |" in markdown for tag in tags)


def test_combine_median_and_precision_on_synthetic_arrays():
    """The combination rules on a hand-built cell of three runs."""
    K = 3
    run_index = np.array([0, 1, 2])
    est = np.array([[1.0, 5.0, 9.0], [2.0, 6.0, 10.0], [3.0, 7.0, 11.0]])
    se = np.full((3, K), 0.1)
    v = np.array([[0.25, 1.21, 100.0], [4.0, 0.25, 1.0], [1.0, 1.21, 0.25]])
    # Four draws per component with the same alternating pattern in every
    # run: each run's standard error is `se`, and the runs' Monte Carlo
    # errors are perfectly correlated, as they are when every run
    # averages the same draws.
    pattern = np.array([-1.0, 1.0, -1.0, 1.0]) * np.sqrt(3.0)
    g = est[:, :, None] + se[:, :, None] * pattern
    np.testing.assert_allclose(g.std(axis=2, ddof=1) / 2, se)
    arrays = {
        "est": est,
        "se": se,
        "v": v,
        "g": g,
        "w": np.array([0.5, 0.3, 0.2]),
        "run_index": run_index,
        "I_corr": np.array([1.5, 6.5, 11.5]),
        "bq_sd": np.sqrt(v[run_index, np.arange(K)]),
    }
    # Ratio 2 with a cap of sqrt(5): component 0 (own sd 0.5) is covered by
    # run 2 (sd 1) but not run 1 (sd 2 exceeds 2 * 0.5 = 1); component 1
    # (own sd 0.5) by neither other run (sd 1.1 > 1), with ratio 3 by
    # both; component 2 (own sd 0.5) by run 1 only (run 0's sd is 10).
    summary, G, cover, a = honest.combine(
        arrays, (2.0, np.sqrt(5), 0.0), "median"
    )
    assert cover.tolist() == [
        [False, False, False],
        [False, False, True],
        [True, False, False],
    ]
    np.testing.assert_allclose(G, [3.0, 6.5, 10.0])
    assert summary["weight_covered"] == pytest.approx(0.7)
    assert summary["components_covered"] == 2
    # One covering run per covered component: its standard error; the
    # fallback component contributes none.
    assert summary["mc_se"] == pytest.approx(
        np.sqrt(0.5**2 * 0.1**2 + 0.2**2 * 0.1**2)
    )
    summary3, G3, cover3, a3 = honest.combine(
        arrays, (3.0, np.sqrt(5), 0.0), "precision"
    )
    assert cover3[:, 1].tolist() == [True, False, True]
    # Precision weights 1/1 and 1/1 on estimates 5 and 7: the mean, 6.
    assert G3[1] == pytest.approx(6.0)
    assert np.isclose(a3[:, 1].sum(), 1.0)
    # The median of two is their mean; of three, the middle one. With
    # perfectly correlated draws the standard error of the mean of two
    # runs is the mean of their standard errors, not its 1/sqrt(2).
    none, G_none, cover_none, _ = honest.combine(
        arrays, (np.inf, np.inf, 0.0), "median"
    )
    assert cover_none.sum() == 6
    np.testing.assert_allclose(G_none, [2.5, 6.0, 9.5])
    assert none["mc_se"] == pytest.approx(
        0.1 * np.sqrt(np.sum(arrays["w"] ** 2))
    )
    # The floor alone: with a floor of 1.05 and a ratio of 1 (which admits
    # no other run, since every other run's SD here exceeds the own
    # run's), run 2 (sd 1) covers component 0 and run 1 (sd 1) covers
    # component 2.
    _, _, cover_floor, _ = honest.combine(
        arrays, (1.0, np.sqrt(5), 1.05), "median"
    )
    assert cover_floor.tolist() == [
        [False, False, False],
        [False, False, True],
        [True, False, False],
    ]
    # Own run included: every component is covered by all three runs.
    _, G_pooled, cover_pooled, _ = honest.combine(
        arrays, (np.inf, np.inf, 0.0), "median", exclude_own=False
    )
    assert cover_pooled.all()
    np.testing.assert_allclose(G_pooled, [2.0, 6.0, 10.0])


class _KnownGP:
    """A stand-in GP whose mean is a known log joint in its run's space.

    ``predict`` returns ``log p(x(u)) + log|J(u)|`` for a fixed target, as
    a run's GP models it, with a small constant variance, so that every
    run's estimate of a component's expected log joint must equal the
    truth exactly (the same draws feed every run and the truth).
    """

    class _Posterior:
        hyp = np.zeros(0)

    class _Count:
        @staticmethod
        def hyperparameter_count(*_):
            return 0

    class _Mean:
        @staticmethod
        def compute(_hyp, U):
            return np.zeros(len(U))

    def __init__(self, pt, log_density, X_orig):
        self.pt = pt
        self.log_density = log_density
        self.X = np.asarray(pt(X_orig), dtype=float).reshape(len(X_orig), -1)
        self.y = (log_density(X_orig) + pt.log_abs_det_jacobian(self.X))[
            :, None
        ]
        self.posteriors = np.array([self._Posterior()])
        self.covariance = self.noise = self._Count()
        self.mean = self._Mean()

    def predict(self, U):
        U = np.asarray(U, dtype=float)
        x = self.pt.inverse(U)
        f = self.log_density(x) + self.pt.log_abs_det_jacobian(U)
        return f[:, None], np.full((len(U), 1), 0.01)


def _known_run(tag, pt, mu, sig, gp):
    K = mu.shape[1]
    return {
        "tag": tag,
        "K": K,
        "D": mu.shape[0],
        "pt": pt,
        "gp": gp,
        "mu": mu,
        "sig": sig,
        "w_own": np.full(K, 1.0 / K),
        "I_corr": np.zeros(K),
        "jac": np.zeros(K),
        "bq_sd": np.full(K, 0.1),
    }


def test_evaluate_maps_draws_between_different_transformers():
    """Two runs with different transforms agree on every component.

    One run is unbounded with its own centering and scaling, the other
    bounded (probit) on a box; the stand-in GPs know the same target in
    their own coordinates, so the cross-run estimates must reproduce the
    own-run ones and the truth to rounding, which pins the composition
    inverse transform, forward transform and log-Jacobian.
    """
    from pyvbmc.parameter_transformer import ParameterTransformer

    D = 2

    def log_density(x):
        x = np.atleast_2d(x)
        return -0.5 * np.sum(x**2, axis=1) - D * 0.5 * np.log(2 * np.pi)

    class Problem:
        ln_Z = 0.0
        log_density_vec = staticmethod(log_density)

    unbounded = ParameterTransformer(
        D,
        np.full((1, D), -np.inf),
        np.full((1, D), np.inf),
        np.array([[-3.0, -1.0]]),
        np.array([[3.0, 5.0]]),
    )
    bounded = ParameterTransformer(
        D,
        np.array([[-6.0, -6.0]]),
        np.array([[6.0, 8.0]]),
        np.array([[-2.0, -2.0]]),
        np.array([[2.0, 2.0]]),
    )
    rng = np.random.default_rng(3)
    X_train = rng.normal(size=(40, D))
    runs = []
    for tag, pt in (("unbounded", unbounded), ("bounded", bounded)):
        centers = pt(np.array([[0.0, 0.0], [1.0, -0.5]]))
        mu = np.asarray(centers, dtype=float).T
        sig = np.full((D, 2), 0.3)
        runs.append(
            _known_run(tag, pt, mu, sig, _KnownGP(pt, log_density, X_train))
        )
    w = np.full(4, 0.25)
    arrays = honest.evaluate(runs, w, Problem, np.random.default_rng(5), 50)
    est, truth = arrays["est"], arrays["truth"]
    np.testing.assert_allclose(est[0], est[1], rtol=0, atol=1e-9)
    np.testing.assert_allclose(est[0], truth, rtol=0, atol=1e-9)
    # The per-draw series of both runs coincide, so the standard error of
    # their difference from the truth vanishes.
    assert np.all(arrays["se_diff"] < 1e-9)
    # Every training point sits within reach of the components centred
    # at the origin, seen from either run.
    assert np.all(arrays["n_within"][:, [0, 2]] > 0)
    assert np.all(np.abs(arrays["data_offset"][:, [0, 2]]) < 1e-9)


def test_cli_rejects_a_headline_ratio_outside_the_grid(pool, tmp_path):
    out, _ = pool
    result = cli(
        SCRIPT,
        "--pool",
        str(out),
        "--cells",
        str(out / "cells_results.json"),
        "--out",
        str(tmp_path),
        "--ratios",
        "1.5,3",
    )
    assert result.returncode == 2
    assert "--headline-ratio 2 is not among --ratios" in result.stderr
    result = cli(
        SCRIPT,
        "--pool",
        str(out),
        "--cells",
        str(out / "cells_results.json"),
        "--out",
        str(tmp_path),
        "--self-check",
    )
    assert result.returncode == 2 and "takes no --cells" in result.stderr


def test_original_arm_is_scored_with_its_variant_names(pool, cells, tmp_path):
    out, _ = pool
    result = cli(
        SCRIPT,
        "--pool",
        str(out),
        "--cells",
        str(cells),
        "--out",
        str(tmp_path),
        "--arm",
        "original",
        "--draws",
        "50",
        "--no-figures",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    results = json.loads(
        (tmp_path / "results.json").read_text(encoding="utf-8")
    )
    cell = results["cells"][0]
    assert cell["arm"] == "original"
    assert set(cell["recorded_bias"]) == {"raw", "capped_I", "capped_E"}
    assert "raw_consistency" in cell["checks"]


def test_run_checks_flag_a_broken_mapping():
    K = 4
    good = {
        "w": np.full(K, 0.25),
        "own_mc": np.array([1.0, 2.0, 3.0, 4.0]) + 0.01,
        "own_mc_se": np.full(K, 0.02),
        "I_corr": np.array([1.0, 2.0, 3.0, 4.0]),
        "jac_mc": np.zeros(K),
        "jac_mc_se": np.zeros(K),
        "jac": np.zeros(K),
    }
    checks = honest.run_checks(good)
    assert not checks["flagged"] and checks["jac_max_z"] == 0.0
    broken = dict(good, own_mc=good["I_corr"] + 1.0)
    assert honest.run_checks(broken)["flagged"]
    shifted = dict(good, jac_mc=np.full(K, 0.3), jac_mc_se=np.full(K, 0.05))
    assert honest.run_checks(shifted)["flagged"]
