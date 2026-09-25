"""Cross-run estimate of a stacked posterior's expected log joint.

The stacked ELBO that S-VBMC maximizes is optimistic on noisy targets: the
optimizer gives weight to whichever run's surrogate overestimated the
expected log joint of a region, so the reported value collects the
selected errors, and the more runs cover a region the larger the largest
error. The Phase 2 proposal of ``dev/2026-09-12-svbmc-elbo-optimism.md``
is to measure the expected log joint of the final mixture with noise that
is independent of the noise the weights were selected on: component by
component, with the Gaussian processes of the *other* runs that cover the
component, the component's own run left out. This script prototypes that
estimator on the artifacts of the run-pool campaign
(``dev/plans/svbmc-benchmark-campaign.md``) and scores it, next to the
raw and capped estimates, against the Monte Carlo reference of the
stacking comparison. The package is untouched: everything here reads the
pool artifacts through ``svbmc_pool_io.load_run`` and the comparison's
``results.json``.

Estimator. For a cell of the comparison (a subset of ``M`` runs and the
weights ``w`` one arm returned), every component ``k`` of every run ``m``
contributes ``draws`` points from its diagonal Gaussian in run ``m``'s
transformed space, mapped to the original space with run ``m``'s inverse
transform. Every run ``r`` of the cell then maps those points into its own
transformed space, evaluates its log-Jacobian there and predicts with its
GP. The GP of a run models the log joint plus the run's log-Jacobian in
the run's transformed space (``function_logger.py`` stores every value
that way), so the mean over the draws of ``predict mean - log-Jacobian``
is run ``r``'s estimate of the component's expected log joint in the
common original space, with a Monte Carlo standard error from the draws,
and the mean over the draws of the predictive variance is run ``r``'s
self-reported uncertainty about it (an upper bound on the variance of the
integral, which would need the covariance between the draws). Run ``r``
*covers* component ``k`` under the coverage rule ``ratio`` when its mean
predictive SD on the component is at most ``ratio`` times that of the
component's own run or below the absolute floor ``sd_floor`` (0.1 nats:
on a noiseless target both SDs are thousandths of a nat and their ratio
says nothing), and in either case at most the absolute cap ``sd_cap``
(the S-VBMC filter's scale, ``sqrt(5)`` nats). The covering runs other
than the component's own are combined by the median of their estimates,
by precision weighting with their mean predictive variances, or by the
plain mean; a component no other run covers keeps its own stored,
Jacobian-corrected estimate. The stack's honest expected log joint is the
weighted sum over components. A grid of rules is evaluated on every cell
(``ratio1.5`` ... ``ratio5`` with the cap, ``cap_only`` with the cap alone,
``none`` with every other run), so the sensitivity to the rule is a
column rather than an assumption; the headline is ``--headline-ratio``
(2) with the median.

Scoring. Every estimate of the expected log joint (raw ``sum w I_corr``,
capped at the component median and at the run median as the class caps
them, honest under every rule and method) is scored by its bias against
``e_log_joint_mc`` of the cell, the mean of the target's noiseless log
density over the arm's own draws of the stacked posterior, so the
comparison isolates the expected log joint term: adding the cell's
``entropy_ref`` to every estimate gives the ELBOs and leaves every bias
unchanged, and ``bias_arm_entropy`` adds the arm's own entropy instead,
which is what the class would report. The same component draws also give
the target's true expected log joint per component (``truth``), from
which three things follow that the reference alone cannot give: the
weighted sum ``G_true`` is a stratified estimate of ``e_log_joint_mc``
(recorded as the ``truth_strat`` bias, a consistency check between the
two Monte Carlo routes); every run's own error against the truth,
component by component, shows the run's own optimism; and dividing the
errors by the self-reported standard deviations (the run's
Bayesian-quadrature SD ``sqrt(J_kk)`` for its own components, the GP's
mean predictive SD for the cross-run estimates) answers the optimism
note's error-scale question as ``z`` statistics per cell. The two SDs are
not the same notion of uncertainty: the quadrature SD is the mean over
hyperparameter samples of the per-sample variance, while the predictive
variance also carries the spread of the per-sample means. Monte Carlo
standard errors respect that every evaluating run and the truth average
over the *same* draws of a component: the standard error of a combined
estimate is that of the combined per-draw series, and a cross-run ``z``
divides by the standard error of the per-draw difference between the
estimate and the truth.

Every pair of evaluating run and component also records ``n_eff`` (the
run's training inputs weighted by the component's Gaussian kernel relative
to its mode; a point at the mode counts one), ``n_within`` (the training
inputs within two component SDs), ``data_offset`` (the mean over those
inputs of the stored noisy value, Jacobian removed, minus the true log
joint there: what the run's data say about the component, before any
fit) and ``mean_fn`` (the run's GP mean function alone, Jacobian removed,
averaged over the component's draws: where the fit falls back to without
data). For the heaviest component of every run of a cell these are
written out as a ``decomposition`` record with every run's estimate, so
that a run's own optimism and another run's under- or over-prediction can
be read against the data each run had at that component.

Checks. On every run the own-run Monte Carlo estimate must reproduce the
stored ``I_corr`` within sampling error (the weighted offset over
components within ``Z_FLAG`` standard errors, no component beyond
``Z_FLAG_MAX``) and the Monte Carlo log-Jacobian must reproduce the
deterministic ``expected_log_jacobian`` the same way; these gate the
mapping and the units. On every cell the recomputed raw expected log joint
must equal the arm's ``raw ELBO - entropy`` (``raw_consistency``; exact
for the integrated arm, whose Jacobian corrections are the same
deterministic ones, within the original arm's 20-draw Monte Carlo
correction otherwise), which gates the weights and the component order.
``--self-check`` runs the run-level checks on every artifact of a pool
without cells, which is how a pool generated elsewhere is first looked at.
A cell whose scoring fails (an arm that dropped a run, an artifact that
does not load) is recorded under ``skipped`` with the reason and the sweep
continues; every finished cell is appended to ``cells.jsonl`` as it
completes.

Usage::

    python dev/scripts/svbmc_honest_elbo.py --pool DIR [--pool DIR2] \\
        --cells RESULTS.json --out DIR [--arm integrated] [--draws 100] \\
        [--seed 0] [--ratios 1.5,2,3,5] [--headline-ratio 2] \\
        [--sd-cap 2.2360679775] [--sd-floor 0.1] [--conditions L1,L2] \\
        [--gpyreg-source PATH] [--no-figures]
    python dev/scripts/svbmc_honest_elbo.py --pool DIR --out DIR --self-check
    python dev/scripts/svbmc_honest_elbo.py --summarize-only --out DIR

``--cells`` is the ``results.json`` of ``svbmc_pool_stack.py``; a cell's
runs are located by tag among the pool directories, and the arm's
recorded weights, component counts and reference terms are read from it.
gpyreg is imported from the frozen worktree the pools' manifests name
(they must agree), or from ``--gpyreg-source``, before PyVBMC is
imported; the process refuses to continue when gpyreg resolves
elsewhere. Every draw comes from a generator keyed by ``--seed``, a
stream constant and the cell's seed (or the run's tag), so a cell is
reproducible from its record. Nothing here needs Torch.

Outputs under ``--out``: ``results.json`` (every cell row, every run's
checks, the skipped cells, the settings), ``cells.jsonl`` (one line per
finished cell, written as the sweep goes),
``cells/<condition>_M<M>_r<rep>.npz`` (the per-component arrays of a
cell: every run's estimate, standard error, mean predictive variance,
``n_eff``, ``n_within``, ``data_offset`` and ``mean_fn`` for every
component, the truth, the own-run values, the weights, the headline
coverage and combination weights), ``summary.json`` and ``summary.md``
(per condition and ``M``: medians over cells with 10 000-resample
bootstrap 95 % intervals of every bias, the cross-covered weight under
every rule, the checks, the runs' own errors, the decomposition of the
heaviest components), ``figures/*.png`` (bias by condition, component
errors own against honest, coverage sensitivity, calibration of the
self-reported uncertainties, errors against effective training points)
and ``sources.json`` (commits, import paths, versions, thread settings,
the hashes of this file and of the cells file, the pools' identities).
``--summarize-only`` rebuilds the summaries and the figures from a
finished ``results.json``; the three figures that read the per-cell
arrays are skipped, with a message, when ``cells/`` is absent.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

# The pool generator's module body pins BLAS to one thread and selects the
# non-interactive matplotlib backend, both of which must happen before
# NumPy is imported, so this import stays above the one below; it imports
# nothing from PyVBMC, so the gpyreg pin can still be activated in `main`
# before the first PyVBMC import.
from svbmc_pool_run import (  # noqa: E402
    THREAD_KEYS,
    activate_gpyreg,
    git,
    write_json,
)

# isort: split
import numpy as np  # noqa: E402

#: Draws per component.
DRAWS = 100
#: Coverage ratios of the rule grid: a run covers a component when its
#: mean predictive SD on the component is at most this multiple of the
#: component's own run's.
RATIOS = (1.5, 2.0, 3.0, 5.0)
HEADLINE_RATIO = 2.0
#: Absolute cap on a covering run's mean predictive SD, in nats: the
#: S-VBMC filter's scale for the Bayesian-quadrature SD of a component.
SD_CAP = float(np.sqrt(5.0))
#: Absolute floor, in nats: a run whose mean predictive SD on a component
#: is below it covers the component whatever the own run's SD. On a
#: noiseless target both SDs are thousandths of a nat and their ratio says
#: nothing about accuracy; on a noisy target the SDs sit far above the
#: floor and the ratio decides.
SD_FLOOR = 0.1
#: Combinations of the covering runs' estimates.
METHODS = ("median", "precision", "mean")
HEADLINE_METHOD = "median"
#: Stream constants keying the draws of a cell and of a run's self-check.
DRAW_STREAM = 23
SELF_STREAM = 29
#: Rows per ``gp.predict`` call (bounds the ``(N, rows)`` kernel matrix).
PREDICT_ROWS = 20_000
BOOTSTRAP_RESAMPLES = 10_000
#: Own-run checks flag a run or cell when the weighted offset of the
#: Monte Carlo values from the stored ones exceeds ``Z_FLAG`` standard
#: errors, or when a single component exceeds ``Z_FLAG_MAX``: a broken
#: mapping gives offsets of many nats and ``z`` in the hundreds, while the
#: largest of 1600 per-component t statistics on skewed draws reaches 5
#: now and then.
Z_FLAG = 4.0
Z_FLAG_MAX = 6.0
#: The ELBO variants of each arm's record, in the names of the cells.
VARIANTS = {
    "integrated": {
        "raw": "raw",
        "capped_I": "capped_I_median",
        "capped_E": "capped_E_median",
    },
    "original": {
        "raw": "estimated",
        "capped_I": "debiased_I_median",
        "capped_E": "debiased_E_median",
    },
}
#: Agreement required of the recomputed raw expected log joint with the
#: integrated arm's record (both use the deterministic Jacobian terms).
RAW_TOLERANCE = 1e-8
#: Mahalanobis radius, in component SDs, of ``n_within``.
WITHIN_RADIUS = 2.0


def sha256(path):
    """SHA-256 of a text file with line endings normalized to LF.

    The script and the cells file are text; a checkout with CRLF line
    endings must record the same hash as one with LF, and as the git blob.
    """
    return hashlib.sha256(
        Path(path).read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest()


def rule_name(ratio):
    return f"ratio{float(ratio):g}"


def coverage_rules(ratios, sd_cap, sd_floor=SD_FLOOR):
    """The rule grid: ``name -> (ratio, cap, floor)``.

    A run covers a component when its mean predictive SD on the component
    is at most ``ratio`` times the own run's or at most ``floor``, and in
    either case at most ``cap``.
    """
    rules = {
        rule_name(r): (float(r), float(sd_cap), float(sd_floor))
        for r in ratios
    }
    rules["cap_only"] = (np.inf, float(sd_cap), float(sd_floor))
    rules["none"] = (np.inf, np.inf, 0.0)
    return rules


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------


class Runs:
    """The pools' artifacts, rebuilt once each and cached by tag.

    A run's tag is its artifact's path in the pool directory without the
    suffix: ``<label>/<label>_seed<seed>`` in a pool of the campaign
    contract, ``<label>_seed<seed>`` in one of the flat layout.
    """

    def __init__(self, pool_dirs):
        self.paths = {}
        for directory in pool_dirs:
            directory = Path(directory)
            for npz in sorted(
                [*directory.glob("*.npz"), *directory.glob("*/*.npz")]
            ):
                tag = npz.relative_to(directory).with_suffix("").as_posix()
                if tag in self.paths:
                    raise RuntimeError(
                        f"the artifact {tag} is in more than one pool "
                        f"directory ({self.paths[tag].parent} and "
                        f"{npz.parent})"
                    )
                self.paths[tag] = npz.with_suffix("")
        self.cache = {}

    def tags(self):
        return sorted(self.paths)

    def get(self, tag):
        if tag not in self.paths:
            raise RuntimeError(f"no pool directory holds the artifact {tag}")
        if tag not in self.cache:
            self.cache[tag] = prepare_run(tag, self.paths[tag])
        return self.cache[tag]


def prepare_run(tag, path):
    """Rebuild one run and precompute what every cell reads from it."""
    from svbmc_pool_io import load_run

    from pyvbmc.svbmc._jacobian import expected_log_jacobian

    state = load_run(path, rng=0)
    vp, gp, pt = state["vp"], state["gp"], state["pt"]
    if pt is not vp.parameter_transformer:
        raise RuntimeError(
            f"{tag}: the posterior does not share the transformer"
        )
    I_sk = np.asarray(vp.stats["I_sk"], dtype=np.float64)
    J_sjk = np.asarray(vp.stats["J_sjk"], dtype=np.float64)
    K = int(vp.K)
    diag = np.arange(K)
    jac = np.asarray(expected_log_jacobian(vp), dtype=np.float64)
    I = I_sk.mean(axis=0)
    identity = state["meta"]["identity"]
    return {
        "tag": tag,
        "label": state["meta"]["label"],
        "seed": int(state["meta"]["seed"]),
        "vp": vp,
        "gp": gp,
        "pt": pt,
        "K": K,
        "D": int(vp.D),
        "N": int(gp.X.shape[0]),
        "Ns": int(gp.posteriors.size),
        "mu": np.asarray(vp.mu, dtype=np.float64),
        "sig": np.asarray(vp.lambd, dtype=np.float64)
        * np.asarray(vp.sigma, dtype=np.float64),
        "w_own": np.ravel(vp.w) / np.sum(vp.w),
        "I": I,
        "jac": jac,
        "I_corr": I - jac,
        "bq_sd": np.sqrt(np.maximum(J_sjk[:, diag, diag].mean(axis=0), 0.0)),
        "noisy": int(vp.stats["uncertainty_handling_level"]) > 0,
        "gpyreg_commit": identity.get("source", identity).get("gpyreg_commit"),
    }


def predict(gp, U, rows=PREDICT_ROWS):
    """GP posterior mean and variance at ``U``, averaged over samples."""
    f = np.empty(len(U), dtype=np.float64)
    s2 = np.empty(len(U), dtype=np.float64)
    for start in range(0, len(U), rows):
        stop = min(start + rows, len(U))
        mean, variance = gp.predict(U[start:stop])
        f[start:stop] = np.ravel(mean)
        s2[start:stop] = np.ravel(variance)
    return f, s2


def mean_function(gp, U):
    """The GP's mean function at ``U``, averaged over the samples."""
    D = U.shape[1]
    cov_N = gp.covariance.hyperparameter_count(D)
    noise_N = gp.noise.hyperparameter_count()
    values = [
        np.ravel(gp.mean.compute(p.hyp[cov_N + noise_N :], U))
        for p in gp.posteriors
    ]
    return np.mean(values, axis=0)


# --------------------------------------------------------------------------
# One cell
# --------------------------------------------------------------------------


def evaluate(runs, w, problem, rng, draws):
    """Every run's estimate of every component's expected log joint.

    ``runs`` are the cell's runs in stacking order, ``w`` the normalized
    weights of all their components in that order. Returns the
    per-component arrays described in the module docstring, with every
    ``(M, K_total)`` array indexed by evaluating run and component, and
    the per-draw series ``g`` of shape ``(M, K_total, draws)`` (predict
    mean minus log-Jacobian) and ``truth_draws`` of shape ``(K_total,
    draws)`` that the standard errors are built from.
    """
    M = len(runs)
    K = [r["K"] for r in runs]
    offsets = np.concatenate([[0], np.cumsum(K)]).astype(int)
    K_total = int(offsets[-1])
    D = runs[0]["D"]
    n = int(draws)
    if w.shape != (K_total,):
        raise RuntimeError(
            f"the weights hold {w.size} components, the runs {K_total}"
        )
    run_index = np.repeat(np.arange(M), K)

    def per_component(values):
        v = np.reshape(values, (K_total, n))
        return v.mean(axis=1), v.std(axis=1, ddof=1) / np.sqrt(n)

    # Draws from every component in its run's space, mapped to the
    # original space with the run's inverse transform.
    X = np.empty((K_total * n, D), dtype=np.float64)
    U_own = np.empty_like(X)
    for m, r in enumerate(runs):
        Z = rng.standard_normal((K[m], n, D))
        U = r["mu"].T[:, None, :] + r["sig"].T[:, None, :] * Z
        block = slice(offsets[m] * n, offsets[m + 1] * n)
        U_own[block] = U.reshape(K[m] * n, D)
        X[block] = r["pt"].inverse(U_own[block])
    truth_draws = np.reshape(
        np.asarray(problem.log_density_vec(X), dtype=np.float64),
        (K_total, n),
    )
    truth = truth_draws.mean(axis=1)
    truth_se = truth_draws.std(axis=1, ddof=1) / np.sqrt(n)

    # Every run evaluates every draw: its transform, its log-Jacobian, its
    # GP and its mean function. The own block keeps the draws themselves
    # rather than the round trip through the original space (identical
    # up to rounding, and up to the clipping of an extreme draw on a
    # bounded coordinate).
    g = np.empty((M, K_total, n))
    v = np.empty((M, K_total))
    mean_fn = np.empty((M, K_total))
    lj_own = np.empty(K_total * n)
    for rr, r in enumerate(runs):
        block = slice(offsets[rr] * n, offsets[rr + 1] * n)
        U = np.asarray(r["pt"](X), dtype=np.float64).reshape(K_total * n, D)
        U[block] = U_own[block]
        lj = np.asarray(r["pt"].log_abs_det_jacobian(U), dtype=np.float64)
        f, s2 = predict(r["gp"], U)
        g[rr] = np.reshape(f - lj, (K_total, n))
        v[rr] = np.reshape(s2, (K_total, n)).mean(axis=1)
        mean_fn[rr] = np.reshape(
            mean_function(r["gp"], U) - lj, (K_total, n)
        ).mean(axis=1)
        lj_own[block] = lj[block]
    est = g.mean(axis=2)
    se = g.std(axis=2, ddof=1) / np.sqrt(n)
    se_diff = (g - truth_draws[None]).std(axis=2, ddof=1) / np.sqrt(n)

    # What each run's training data say about each component: the run's
    # training inputs mapped into the component's run's space, weighted
    # by the component's Gaussian kernel relative to its mode (`n_eff`),
    # counted within two SDs (`n_within`), and their stored values with
    # the Jacobian removed against the true log joint there
    # (`data_offset`, the noise realization the run saw).
    n_eff = np.empty((M, K_total))
    n_within = np.empty((M, K_total), dtype=int)
    data_offset = np.full((M, K_total), np.nan)
    for rr, r in enumerate(runs):
        X_train_u = np.asarray(r["gp"].X, dtype=np.float64)
        X_train = r["pt"].inverse(X_train_u)
        if "train_residual" not in r:
            # The run's stored values against the truth at its training
            # inputs depend on the run alone; cached across cells.
            r["train_residual"] = (
                np.ravel(r["gp"].y)
                - np.asarray(
                    r["pt"].log_abs_det_jacobian(X_train_u), dtype=float
                )
                - np.asarray(problem.log_density_vec(X_train), dtype=float)
            )
        residual = r["train_residual"]
        for m, other in enumerate(runs):
            U_m = np.asarray(other["pt"](X_train), dtype=np.float64)
            U_m = U_m.reshape(-1, D)
            scaled = (U_m[:, None, :] - other["mu"].T[None, :, :]) / other[
                "sig"
            ].T[None, :, :]
            d2 = np.sum(scaled**2, axis=2)  # (N_r, K_m)
            block = slice(offsets[m], offsets[m + 1])
            n_eff[rr, block] = np.exp(-0.5 * d2).sum(axis=0)
            within = d2 < WITHIN_RADIUS**2
            count = within.sum(axis=0)
            n_within[rr, block] = count
            total = (residual[:, None] * within).sum(axis=0)
            data_offset[rr, block] = np.where(
                count > 0, total / np.maximum(count, 1), np.nan
            )
    columns = np.arange(K_total)
    jac_mc, jac_se = per_component(lj_own)
    return {
        "K": np.asarray(K, dtype=int),
        "w": w,
        "run_index": run_index,
        "g": g,
        "truth_draws": truth_draws,
        "est": est,
        "se": se,
        "se_diff": se_diff,
        "v": v,
        "mean_fn": mean_fn,
        "n_eff": n_eff,
        "n_within": n_within,
        "data_offset": data_offset,
        "truth": truth,
        "truth_se": truth_se,
        "own_mc": est[run_index, columns],
        "own_mc_se": se[run_index, columns],
        "jac_mc": jac_mc,
        "jac_mc_se": jac_se,
        "I_corr": np.concatenate([r["I_corr"] for r in runs]),
        "jac": np.concatenate([r["jac"] for r in runs]),
        "bq_sd": np.concatenate([r["bq_sd"] for r in runs]),
    }


def combine(arrays, rule, method, exclude_own=True):
    """Honest expected log joint under one coverage rule and method.

    Returns the cell-level numbers and the per-component estimate,
    coverage mask (evaluating run by component) and combination weights.
    The Monte Carlo standard error of a combined estimate is that of the
    combined per-draw series (every evaluating run averages the same
    draws), from ``arrays["g"]``.
    """
    est, v = arrays["est"], arrays["v"]
    w, run_index = arrays["w"], arrays["run_index"]
    g = arrays["g"]
    n = g.shape[2]
    M, K_total = est.shape
    ratio, cap, floor = rule
    columns = np.arange(K_total)
    sd = np.sqrt(v)
    own_sd = sd[run_index, columns]
    cover = ((sd <= ratio * own_sd[None, :]) | (sd <= floor)) & (sd <= cap)
    if exclude_own:
        cover[run_index, columns] = False
    G = np.empty(K_total)
    mc_se = np.zeros(K_total)
    a = np.zeros((M, K_total))
    fallback = np.zeros(K_total, dtype=bool)
    for k in range(K_total):
        R = np.flatnonzero(cover[:, k])
        if R.size == 0:
            G[k] = arrays["I_corr"][k]
            fallback[k] = True
            continue
        if method == "median":
            order = R[np.argsort(est[R, k])]
            half = R.size // 2
            picks = order[[half]] if R.size % 2 else order[[half - 1, half]]
            a[picks, k] = 1.0 / picks.size
        elif method == "precision":
            precision = 1.0 / np.maximum(v[R, k], np.finfo(float).tiny)
            a[R, k] = precision / precision.sum()
        elif method == "mean":
            a[R, k] = 1.0 / R.size
        else:
            raise ValueError(f"unknown method {method!r}")
        G[k] = a[:, k] @ est[:, k]
        series = a[:, k] @ g[:, k, :]
        mc_se[k] = series.std(ddof=1) / np.sqrt(n)
    # GP-side uncertainty of the weighted sum, under independence across
    # components and under full correlation within each evaluating run
    # (a fallback component is evaluated by its own run's quadrature).
    gp_var_k = np.sum(a**2 * v, axis=0)
    gp_var_k[fallback] = arrays["bq_sd"][fallback] ** 2
    per_run = np.sum(w[None, :] * a * sd, axis=1)
    np.add.at(
        per_run, run_index[fallback], w[fallback] * arrays["bq_sd"][fallback]
    )
    covered = ~fallback
    summary = {
        "G": float(w @ G),
        "mc_se": float(np.sqrt(np.sum(w**2 * mc_se**2))),
        "gp_sd_indep": float(np.sqrt(np.sum(w**2 * gp_var_k))),
        "gp_sd_corr": float(np.sqrt(np.sum(per_run**2))),
        "weight_covered": float(w[covered].sum()),
        "components_covered": int(covered.sum()),
        "covering_runs_weighted": float(w @ cover.sum(axis=0)),
    }
    return summary, G, cover, a


def weighted_quantile(values, weights, q):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(np.interp(q * cumulative[-1], cumulative, values[order]))


def z_stats(z, weights):
    """Weighted median, median absolute value and tail fraction of ``z``."""
    z = np.asarray(z, dtype=float)
    weights = np.asarray(weights, dtype=float)
    keep = np.isfinite(z) & (weights > 0)
    if not np.any(keep):
        return {"n": 0, "median": None, "abs_median": None, "frac_gt2": None}
    z, weights = z[keep], weights[keep] / weights[keep].sum()
    return {
        "n": int(z.size),
        "median": weighted_quantile(z, weights, 0.5),
        "abs_median": weighted_quantile(np.abs(z), weights, 0.5),
        "frac_gt2": float(weights[np.abs(z) > 2].sum()),
    }


def run_checks(arrays):
    """The own-run consistency statistics of a cell or a self-check."""
    w = arrays["w"]
    tiny = np.finfo(float).tiny
    own_diff = arrays["own_mc"] - arrays["I_corr"]
    own_z = own_diff / np.maximum(arrays["own_mc_se"], tiny)
    own_offset = float(w @ own_diff)
    own_offset_se = float(np.sqrt(np.sum(w**2 * arrays["own_mc_se"] ** 2)))
    jac_diff = arrays["jac_mc"] - arrays["jac"]
    jac_abs = np.abs(jac_diff)
    # A constant Jacobian (unbounded, unwarped run) has zero SE and a
    # difference of rounding size, which counts as zero.
    exact = jac_abs <= 1e-9
    jac_z = np.where(
        exact, 0.0, jac_diff / np.maximum(arrays["jac_mc_se"], tiny)
    )
    jac_offset = float(w @ np.where(exact, 0.0, jac_diff))
    jac_offset_se = float(
        np.sqrt(np.sum(np.where(exact, 0.0, w * arrays["jac_mc_se"]) ** 2))
    )
    own_offset_z = own_offset / max(own_offset_se, tiny)
    jac_offset_z = 0.0 if jac_offset_se == 0 else jac_offset / jac_offset_se
    return {
        "own_mc_max_z": float(np.max(np.abs(own_z))),
        "own_offset": own_offset,
        "own_offset_z": float(own_offset_z),
        "jac_max_z": float(np.max(np.abs(jac_z))),
        "jac_max_abs": float(np.max(jac_abs)),
        "jac_offset": jac_offset,
        "jac_offset_z": float(jac_offset_z),
        "flagged": bool(
            np.max(np.abs(own_z)) > Z_FLAG_MAX
            or np.max(np.abs(jac_z)) > Z_FLAG_MAX
            or abs(own_offset_z) > Z_FLAG
            or abs(jac_offset_z) > Z_FLAG
        ),
    }


def _optional(value):
    value = float(value)
    return None if not np.isfinite(value) else value


def decomposition(runs, arrays, cover):
    """The heaviest component of every run, seen by every run of the cell."""
    w = arrays["w"]
    offsets = np.concatenate([[0], np.cumsum(arrays["K"])]).astype(int)
    rows = []
    for m, r in enumerate(runs):
        block = slice(offsets[m], offsets[m + 1])
        k = offsets[m] + int(np.argmax(w[block]))

        def data(rr):
            return {
                "n_eff": float(arrays["n_eff"][rr, k]),
                "n_within": int(arrays["n_within"][rr, k]),
                "data_offset": _optional(arrays["data_offset"][rr, k]),
                "mean_fn": float(arrays["mean_fn"][rr, k]),
            }

        rows.append(
            {
                "run": r["tag"],
                "component": int(k - offsets[m]),
                "weight": float(w[k]),
                "truth": float(arrays["truth"][k]),
                "truth_se": float(arrays["truth_se"][k]),
                "own": {
                    "I_corr": float(arrays["I_corr"][k]),
                    "error": float(arrays["I_corr"][k] - arrays["truth"][k]),
                    "bq_sd": float(arrays["bq_sd"][k]),
                    **data(m),
                },
                "others": {
                    runs[rr]["tag"]: {
                        "estimate": float(arrays["est"][rr, k]),
                        "error": float(
                            arrays["est"][rr, k] - arrays["truth"][k]
                        ),
                        "sd": float(np.sqrt(arrays["v"][rr, k])),
                        "covered": bool(cover[rr, k]),
                        **data(rr),
                    }
                    for rr in range(len(runs))
                    if rr != m
                },
            }
        )
    return rows


def score_cell(cell, arm, runs, problem, rng, draws, rules, methods, headline):
    """One cell: every estimate, its bias and the diagnostics."""
    record = cell["arms"][arm]
    names = VARIANTS[arm]
    K_recorded = [int(k) for k in record["K"]]
    K_runs = [r["K"] for r in runs]
    if K_recorded != K_runs:
        raise RuntimeError(
            f"the cell records components {K_recorded} but the runs hold "
            f"{K_runs}: the arm did not retain the cell's runs as listed"
        )
    w = np.asarray(record["w"], dtype=np.float64).ravel()
    w = w / w.sum()
    started = time.perf_counter()
    arrays = evaluate(runs, w, problem, rng, draws)
    evaluate_seconds = time.perf_counter() - started

    I_corr = arrays["I_corr"]
    offsets = np.concatenate([[0], np.cumsum(arrays["K"])]).astype(int)
    E_corr = np.array(
        [
            r["w_own"] @ I_corr[offsets[m] : offsets[m + 1]]
            for m, r in enumerate(runs)
        ]
    )
    G_raw = float(w @ I_corr)
    estimates = {
        "raw": G_raw,
        "capped_I": float(min(G_raw, np.median(I_corr))),
        "capped_E": float(min(G_raw, np.median(E_corr))),
    }
    e_mc = float(record["e_log_joint_mc"])
    e_mc_sd = float(record["e_log_joint_mc_sd"])
    entropy_arm = float(record["entropy"])
    entropy_ref = float(record["entropy_ref"])
    G_true = float(w @ arrays["truth"])
    G_true_se = float(np.sqrt(np.sum(w**2 * arrays["truth_se"] ** 2)))

    honest = {}
    per_component = None
    for name, rule in rules.items():
        honest[name] = {}
        for method in methods:
            summary, G_k, cover, a = combine(arrays, rule, method)
            summary["bias"] = summary["G"] - e_mc
            summary["bias_arm_entropy"] = summary["bias"] + (
                entropy_arm - entropy_ref
            )
            summary["elbo"] = summary["G"] + entropy_ref
            summary["elbo_arm_entropy"] = summary["G"] + entropy_arm
            honest[name][method] = summary
            if name == headline[0] and method == headline[1]:
                per_component = {"G": G_k, "cover": cover, "a": a}
    if per_component is None:
        raise RuntimeError(f"the headline {headline} is not in the grid")
    pooled, _, _, _ = combine(
        arrays, rules[headline[0]], headline[1], exclude_own=False
    )
    pooled["bias"] = pooled["G"] - e_mc

    # Calibration of the self-reported uncertainties against the truth.
    truth, truth_se = arrays["truth"], arrays["truth_se"]
    own_z = (I_corr - truth) / np.sqrt(arrays["bq_sd"] ** 2 + truth_se**2)
    cover = per_component["cover"]
    rows, cols = np.nonzero(cover)
    cross_z = (arrays["est"][rows, cols] - truth[cols]) / np.sqrt(
        arrays["v"][rows, cols] + arrays["se_diff"][rows, cols] ** 2
    )
    honest_error = per_component["G"] - truth

    # Per run: own error with the stacked weights, and how the other runs
    # see its components under the headline rule.
    run_rows = []
    run_index = arrays["run_index"]
    for m, r in enumerate(runs):
        mine = run_index == m
        share = float(w[mine].sum())
        own_error = (
            float(w[mine] @ (I_corr[mine] - truth[mine]) / share)
            if share > 0
            else None
        )
        seen_by = {}
        for rr, other in enumerate(runs):
            if rr == m:
                continue
            covered = mine & cover[rr]
            covered_share = float(w[covered].sum())
            seen_by[other["tag"]] = {
                "weight_covered": (
                    covered_share / share if share > 0 else None
                ),
                "error": (
                    float(
                        w[covered]
                        @ (arrays["est"][rr, covered] - truth[covered])
                        / covered_share
                    )
                    if covered_share > 0
                    else None
                ),
            }
        run_rows.append(
            {
                "tag": r["tag"],
                "weight_share": share,
                "own_error": own_error,
                "seen_by": seen_by,
            }
        )

    n_eff = arrays["n_eff"]
    sparsity = {
        "own_n_eff_median": weighted_quantile(
            n_eff[run_index, np.arange(len(w))], w, 0.5
        ),
        "cross_n_eff_median": (
            weighted_quantile(n_eff[rows, cols], w[cols], 0.5)
            if rows.size
            else None
        ),
    }

    checks = run_checks(arrays)
    raw_arm = float(record["elbos"][names["raw"]]) - entropy_arm
    checks["raw_consistency"] = float(G_raw - raw_arm)
    if arm == "integrated" and abs(checks["raw_consistency"]) > RAW_TOLERANCE:
        checks["flagged"] = True

    row = {
        "condition": cell["condition"],
        "M": int(cell["M"]),
        "repetition": int(cell["repetition"]),
        "cell_seed": int(cell["cell_seed"]),
        "arm": arm,
        "entries": [r["tag"] for r in runs],
        "K": K_runs,
        "K_total": int(sum(K_runs)),
        "draws": int(draws),
        "reference": {
            "e_log_joint_mc": e_mc,
            "e_log_joint_mc_sd": e_mc_sd,
            "entropy_ref": entropy_ref,
            "entropy_ref_sd": float(record["entropy_ref_sd"]),
            "entropy_arm": entropy_arm,
            "elbo_mc": float(record["elbo_mc"]),
            "elbo_mc_sd": float(record["elbo_mc_sd"]),
            "kl_gap": float(record["kl_gap"]),
            "ln_Z": None if problem.ln_Z is None else float(problem.ln_Z),
        },
        "estimates": estimates,
        "bias": {k: float(v - e_mc) for k, v in estimates.items()},
        "recorded_bias": {
            k: float(record["bias"][name]) for k, name in names.items()
        },
        "truth_strat": {"G": G_true, "se": G_true_se, "bias": G_true - e_mc},
        "honest": honest,
        "pooled": pooled,
        "calibration": {
            "own": z_stats(own_z, w),
            "cross": z_stats(cross_z, w[cols]),
            "honest_error_weighted": float(w @ honest_error),
            "own_error_weighted": float(w @ (I_corr - truth)),
        },
        "sparsity": sparsity,
        "runs": run_rows,
        "decomposition": decomposition(runs, arrays, cover),
        "checks": checks,
        "evaluate_seconds": evaluate_seconds,
    }
    return row, arrays, per_component


def self_check(run, problem, rng, draws):
    """The own-run checks and the run's own error, on one artifact."""
    started = time.perf_counter()
    arrays = evaluate([run], run["w_own"], problem, rng, draws)
    checks = run_checks(arrays)
    error = run["I_corr"] - arrays["truth"]
    z = error / np.sqrt(run["bq_sd"] ** 2 + arrays["truth_se"] ** 2)
    return {
        "tag": run["tag"],
        "condition": run["label"],
        "seed": run["seed"],
        "D": run["D"],
        "K": run["K"],
        "N": run["N"],
        "Ns": run["Ns"],
        "noisy": bool(run["noisy"]),
        "gpyreg_commit": run["gpyreg_commit"],
        "checks": checks,
        "own_error": float(run["w_own"] @ error),
        "own_error_se": float(
            np.sqrt(np.sum(run["w_own"] ** 2 * arrays["truth_se"] ** 2))
        ),
        "own_z": z_stats(z, run["w_own"]),
        "bq_sd_median": float(np.median(run["bq_sd"])),
        "pred_sd_median": float(np.median(np.sqrt(arrays["v"][0]))),
        "seconds": time.perf_counter() - started,
    }


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------


def bootstrap_median(values, rng, resamples=BOOTSTRAP_RESAMPLES):
    """Median of ``values`` with a percentile bootstrap 95 % interval."""
    finite = np.asarray(
        [v for v in values if v is not None and np.isfinite(v)], dtype=float
    )
    if finite.size == 0:
        return {"n": 0, "median": None, "lo": None, "hi": None}
    draws = np.median(
        finite[rng.integers(0, finite.size, size=(resamples, finite.size))],
        axis=1,
    )
    return {
        "n": int(finite.size),
        "median": float(np.median(finite)),
        "lo": float(np.percentile(draws, 2.5)),
        "hi": float(np.percentile(draws, 97.5)),
    }


HONEST_KEYS = (
    "bias",
    "bias_arm_entropy",
    "weight_covered",
    "mc_se",
    "gp_sd_indep",
    "gp_sd_corr",
    "covering_runs_weighted",
)


def build_summary(results, rng):
    settings = results["settings"]
    rules = list(settings["rules"])
    methods = list(settings["methods"])
    head_rule, head_method = (
        settings["headline_rule"],
        settings["headline_method"],
    )
    conditions = {r["condition"]: {} for r in results["runs"]}
    for row in results["cells"]:
        conditions.setdefault(row["condition"], {}).setdefault(
            row["M"], []
        ).append(row)
    out = []
    for condition in sorted(conditions):
        by_M = []
        for M in sorted(conditions[condition]):
            rows = conditions[condition][M]

            def agg(get):
                return bootstrap_median([get(r) for r in rows], rng)

            honest = {
                rule: {
                    method: {
                        key: agg(
                            lambda r, rule=rule, method=method, key=key: r[
                                "honest"
                            ][rule][method][key]
                        )
                        for key in HONEST_KEYS
                    }
                    for method in methods
                }
                for rule in rules
            }
            head = [r["honest"][head_rule][head_method] for r in rows]
            within = [
                abs(h["bias"])
                <= 2
                * np.hypot(h["mc_se"], r["reference"]["e_log_joint_mc_sd"])
                for h, r in zip(head, rows)
            ]
            medians = {
                key: float(np.median([abs(r["bias"][key]) for r in rows]))
                for key in ("raw", "capped_I", "capped_E")
            }
            medians["honest"] = float(
                np.median([abs(h["bias"]) for h in head])
            )
            # A verdict on the median absolute biases with a tie band of
            # one reference standard error, so that Monte Carlo noise
            # does not flip it.
            tie = float(
                np.median([r["reference"]["e_log_joint_mc_sd"] for r in rows])
            )

            def verdict(other):
                gap = medians["honest"] - medians[other]
                if abs(gap) <= tie:
                    return "tie"
                return "worse" if gap > 0 else "better"

            by_M.append(
                {
                    "M": M,
                    "cells": len(rows),
                    "bias": {
                        key: agg(lambda r, key=key: r["bias"][key])
                        for key in ("raw", "capped_I", "capped_E")
                    },
                    "recorded_bias": {
                        key: agg(lambda r, key=key: r["recorded_bias"][key])
                        for key in ("raw", "capped_I", "capped_E")
                    },
                    "truth_strat_bias": agg(
                        lambda r: r["truth_strat"]["bias"]
                    ),
                    "reference_sd": agg(
                        lambda r: r["reference"]["e_log_joint_mc_sd"]
                    ),
                    "kl_gap": agg(lambda r: r["reference"]["kl_gap"]),
                    "honest": honest,
                    "pooled_bias": agg(lambda r: r["pooled"]["bias"]),
                    "headline_within_se": float(np.mean(within)),
                    "median_abs_bias": medians,
                    "verdict_tie_band": tie,
                    "headline_vs_raw": verdict("raw"),
                    "headline_vs_capped": verdict("capped_I"),
                    "calibration": {
                        side: {
                            stat: agg(
                                lambda r, side=side, stat=stat: r[
                                    "calibration"
                                ][side][stat]
                            )
                            for stat in ("median", "abs_median", "frac_gt2")
                        }
                        for side in ("own", "cross")
                    },
                    "sparsity": {
                        key: agg(lambda r, key=key: r["sparsity"][key])
                        for key in ("own_n_eff_median", "cross_n_eff_median")
                    },
                    "checks": {
                        "own_mc_max_z": max(
                            r["checks"]["own_mc_max_z"] for r in rows
                        ),
                        "own_offset_max_z": max(
                            abs(r["checks"]["own_offset_z"]) for r in rows
                        ),
                        "jac_max_z": max(
                            r["checks"]["jac_max_z"] for r in rows
                        ),
                        "jac_offset_max_z": max(
                            abs(r["checks"]["jac_offset_z"]) for r in rows
                        ),
                        "raw_consistency_max": max(
                            abs(r["checks"]["raw_consistency"]) for r in rows
                        ),
                        "flagged_cells": sum(
                            bool(r["checks"]["flagged"]) for r in rows
                        ),
                    },
                    "decomposition": rows[0]["decomposition"],
                    "decomposition_cell": {
                        "repetition": rows[0]["repetition"],
                        "cell_seed": rows[0]["cell_seed"],
                    },
                }
            )
        runs = [r for r in results["runs"] if r["condition"] == condition]
        out.append(
            {
                "condition": condition,
                "noisy": any(r["noisy"] for r in runs),
                "by_M": by_M,
                "runs": [
                    {
                        "tag": r["tag"],
                        "own_error": r["own_error"],
                        "own_error_se": r["own_error_se"],
                        "own_z_median": r["own_z"]["median"],
                        "own_abs_z_median": r["own_z"]["abs_median"],
                        "bq_sd_median": r["bq_sd_median"],
                        "pred_sd_median": r["pred_sd_median"],
                        "own_mc_max_z": r["checks"]["own_mc_max_z"],
                        "own_offset_z": r["checks"]["own_offset_z"],
                        "jac_max_z": r["checks"]["jac_max_z"],
                        "jac_offset_z": r["checks"]["jac_offset_z"],
                        "flagged": r["checks"]["flagged"],
                    }
                    for r in runs
                ],
            }
        )
    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "settings": settings,
        "skipped": results.get("skipped", []),
        "conditions": out,
    }


def number(value, digits=3):
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def interval(entry, digits=3):
    if entry is None or entry.get("median") is None:
        return "-"
    return (
        f"{entry['median']:.{digits}f} "
        f"[{entry['lo']:.{digits}f}, {entry['hi']:.{digits}f}]"
    )


def _verdict_text(verdict):
    return {"worse": "**worse**", "better": "better", "tie": "a tie"}[verdict]


def decomposition_markdown(rows):
    """The heaviest component of every run of a cell, seen by every run."""
    lines = [
        "| Run, heaviest component (weight) | truth | own I_corr (error; "
        "BQ sd; points within 2 SD, their data offset; mean fn) | other "
        "runs: estimate (error; pred sd; covered; points within 2 SD, "
        "their data offset; mean fn) |",
        "|---|---:|---|---|",
    ]
    for row in rows:
        own = row["own"]
        others = "; ".join(
            f"{tag}: {o['estimate']:.3f} ({o['error']:+.3f}; {o['sd']:.3f}; "
            f"{'yes' if o['covered'] else 'no'}; {o['n_within']}, "
            f"{number(o['data_offset'])}; {o['mean_fn']:.1f})"
            for tag, o in row["others"].items()
        )
        lines.append(
            f"| {row['run']}, component {row['component']} "
            f"({row['weight']:.3f}) | {row['truth']:.3f} ± "
            f"{row['truth_se']:.3f} | {own['I_corr']:.3f} "
            f"({own['error']:+.3f}; {own['bq_sd']:.3f}; {own['n_within']}, "
            f"{number(own['data_offset'])}; {own['mean_fn']:.1f}) | "
            f"{others} |"
        )
    return lines


def summary_markdown(summary):
    settings = summary["settings"]
    rule, method = settings["headline_rule"], settings["headline_method"]
    if settings["cells"] is None:
        lines = [
            "# Own-run checks of pool artifacts (`--self-check`)",
            "",
            f"Generated {summary['generated']} from the artifacts of the "
            f"pool alone, {settings['draws']} draws per component, seed "
            f"{settings['seed']}. For every run the own-run Monte Carlo "
            "estimate of each component's expected log joint is compared "
            "with the stored, Jacobian-corrected `I_corr` and the Monte "
            "Carlo log-Jacobian with the deterministic one (`abs z` per "
            "component and the weighted offset in standard errors; a run "
            f"is flagged beyond {settings['z_flag_max']:g} for a component "
            f"or {settings['z_flag']:g} for the offset); `own error` is the "
            "run's own expected log joint against the target's true one "
            "over its draws, weighted by the run's own weights; `BQ sd` and "
            "`pred sd` are the medians over components of the run's "
            "quadrature SD and of its GP's mean predictive SD on the "
            "component.",
        ]
        for condition in summary["conditions"]:
            lines += _run_table(condition)
        return "\n".join(lines) + "\n"
    origin = (
        f"the cells of `{Path(settings['cells']).name}` "
        f"(arm `{settings['arm']}`)"
    )
    lines = [
        "# Honest (cross-run) expected log joint of stacked posteriors",
        "",
        f"Generated {summary['generated']} from {origin}, "
        f"{settings['draws']} draws per component, seed {settings['seed']}. "
        "Every estimate of the stack's expected log joint is scored by its "
        "bias against the cell's `e_log_joint_mc` (the mean of the "
        "target's noiseless log density over the arm's own draws of the "
        "stacked posterior); adding the cell's `entropy_ref` to every "
        "estimate gives the ELBOs and leaves every bias unchanged. "
        "`raw` is `sum w I_corr`, `capped_I` and `capped_E` the class's "
        "component-median and run-median caps of it; an honest value "
        "replaces every component's expected log joint by the covering "
        "other runs' GP estimates under a coverage rule (`ratioR`: a "
        "run's mean predictive SD on the component at most `R` times the "
        f"component's own run's or below {settings['sd_floor']:g} nats, "
        f"and at most {settings['sd_cap']:.3f} nats; `cap_only`: the cap "
        "alone; `none`: every other run) combined by the `median`, "
        "`precision` weighting or the `mean`, the component's own value "
        f"when no other run covers it. The headline is `{rule}` with the "
        f"`{method}`. `truth_strat` is the bias of the stratified "
        "Monte Carlo truth (`sum w E_true`) against `e_log_joint_mc`, a "
        "consistency check between the two Monte Carlo routes, and "
        "`pooled` the headline rule with the component's own run "
        "included (not honest). Medians over the cells of a condition and "
        f"`M` with a {settings['bootstrap_resamples']:,}-resample "
        "bootstrap 95 % interval. `covered` is the stacked weight on "
        "components at least one other run covers. `mc_se` is the "
        "Monte Carlo standard error of the honest value from the draws "
        "(the covering runs average the same draws), `gp_sd` its GP-side "
        "standard deviation under independence across components and "
        "under full correlation within each evaluating run. `within 2 SE` "
        "is the fraction of cells whose headline bias is within twice the "
        "combined Monte Carlo standard error of the estimate and of the "
        "reference.",
    ]
    if summary.get("skipped"):
        lines += [
            "",
            f"Skipped cells: {len(summary['skipped'])} (see `results.json`).",
        ]
    if any(c["by_M"] for c in summary["conditions"]):
        lines += [
            "",
            "## Headline across conditions",
            "",
            "| Condition | M | cells | raw | capped_I | honest "
            f"({rule}, {method}) | honest ({rule}, precision) | covered | "
            "mc_se | within 2 SE | median abs bias raw / capped_I / "
            "honest | KL gap |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
        ]
        for condition in summary["conditions"]:
            for entry in condition["by_M"]:
                h = entry["honest"][rule]
                mab = entry["median_abs_bias"]
                lines.append(
                    f"| {condition['condition']} | {entry['M']} | "
                    f"{entry['cells']} | {interval(entry['bias']['raw'])} | "
                    f"{interval(entry['bias']['capped_I'])} | "
                    f"{interval(h[method]['bias'])} | "
                    f"{interval(h['precision']['bias'])} | "
                    f"{number(h[method]['weight_covered']['median'])} | "
                    f"{number(h[method]['mc_se']['median'])} | "
                    f"{entry['headline_within_se']:.2f} | "
                    f"{mab['raw']:.3f} / {mab['capped_I']:.3f} / "
                    f"{mab['honest']:.3f} | {interval(entry['kl_gap'], 2)} |"
                )
    for condition in summary["conditions"]:
        lines += [
            "",
            f"## {condition['condition']}"
            + (" (noisy)" if condition["noisy"] else " (noiseless)"),
        ]
        for entry in condition["by_M"]:
            lines += [
                "",
                f"### M = {entry['M']} ({entry['cells']} cells)",
                "",
                "| Estimator | bias | bias with the arm's entropy | covered "
                "| mc_se | gp_sd (indep / corr) |",
                "|---|---:|---:|---:|---:|---:|",
                f"| raw | {interval(entry['bias']['raw'])} | "
                f"{interval(entry['recorded_bias']['raw'])} | - | - | - |",
                f"| capped_I | {interval(entry['bias']['capped_I'])} | "
                f"{interval(entry['recorded_bias']['capped_I'])} | - | - | "
                "- |",
                f"| capped_E | {interval(entry['bias']['capped_E'])} | "
                f"{interval(entry['recorded_bias']['capped_E'])} | - | - | "
                "- |",
            ]
            for r in settings["rules"]:
                for m in settings["methods"]:
                    h = entry["honest"][r][m]
                    lines.append(
                        f"| honest {r} {m} | {interval(h['bias'])} | "
                        f"{interval(h['bias_arm_entropy'])} | "
                        f"{number(h['weight_covered']['median'])} | "
                        f"{number(h['mc_se']['median'])} | "
                        f"{number(h['gp_sd_indep']['median'])} / "
                        f"{number(h['gp_sd_corr']['median'])} |"
                    )
            mab = entry["median_abs_bias"]
            own_c, cross_c = (
                entry["calibration"]["own"],
                entry["calibration"]["cross"],
            )
            checks = entry["checks"]
            lines += [
                f"| pooled ({rule}, own included) | "
                f"{interval(entry['pooled_bias'])} | - | - | - | - |",
                f"| truth_strat | {interval(entry['truth_strat_bias'])} | - "
                "| - | - | - |",
                "",
                f"Reference `e_log_joint_mc_sd` median "
                f"{number(entry['reference_sd']['median'])}; the headline "
                f"bias is within 2 SE in {entry['headline_within_se']:.0%} "
                "of the cells; median absolute bias raw "
                f"{mab['raw']:.3f}, capped_I {mab['capped_I']:.3f}, "
                f"capped_E {mab['capped_E']:.3f}, honest {mab['honest']:.3f} "
                "(against raw the honest estimate is "
                f"{_verdict_text(entry['headline_vs_raw'])}, against "
                f"capped_I {_verdict_text(entry['headline_vs_capped'])}; "
                f"tie band {entry['verdict_tie_band']:.3f}, one reference "
                "standard error). Calibration of the self-reported SDs "
                "(weighted over components): own z median "
                f"{interval(own_c['median'], 2)}, abs z median "
                f"{interval(own_c['abs_median'], 2)}, abs z > 2 fraction "
                f"{interval(own_c['frac_gt2'], 2)}; cross z median "
                f"{interval(cross_c['median'], 2)}, abs z median "
                f"{interval(cross_c['abs_median'], 2)}, abs z > 2 fraction "
                f"{interval(cross_c['frac_gt2'], 2)}. Effective training "
                "points on a component (weighted median): own run "
                f"{interval(entry['sparsity']['own_n_eff_median'], 1)}, "
                "covering other runs "
                f"{interval(entry['sparsity']['cross_n_eff_median'], 1)}. "
                f"Checks: own-run max abs z {checks['own_mc_max_z']:.2f} "
                f"(weighted offset abs z {checks['own_offset_max_z']:.2f}), "
                f"Jacobian max abs z {checks['jac_max_z']:.2f} (offset abs z "
                f"{checks['jac_offset_max_z']:.2f}), raw consistency "
                f"{checks['raw_consistency_max']:.1e}, flagged cells "
                f"{checks['flagged_cells']}.",
            ]
            if entry["M"] <= 4:
                lines += [
                    "",
                    "Heaviest component of every run, repetition "
                    f"{entry['decomposition_cell']['repetition']}:",
                    "",
                    *decomposition_markdown(entry["decomposition"]),
                ]
        lines += _run_table(condition)
    return "\n".join(lines) + "\n"


def _run_table(condition):
    lines = [
        "",
        f"## {condition['condition']}"
        + (" (noisy)" if condition["noisy"] else " (noiseless)")
        if not condition["by_M"]
        else "",
        "| Run | own error | se | own z median | own abs z median | "
        "BQ sd median | pred sd median | own-run max abs z (offset z) | "
        "Jacobian max abs z (offset z) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if condition["by_M"]:
        lines.pop(1)
    for r in condition["runs"]:
        lines.append(
            f"| {r['tag']} | {number(r['own_error'])} | "
            f"{number(r['own_error_se'])} | "
            f"{number(r['own_z_median'], 2)} | "
            f"{number(r['own_abs_z_median'], 2)} | "
            f"{number(r['bq_sd_median'])} | "
            f"{number(r['pred_sd_median'])} | "
            f"{r['own_mc_max_z']:.2f} ({r['own_offset_z']:+.2f}) | "
            f"{r['jac_max_z']:.2f} ({r['jac_offset_z']:+.2f})"
            + (" (flagged)" if r["flagged"] else "")
            + " |"
        )
    return lines


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

SERIES = {"raw": "#2a78d6", "capped": "#eb6834", "honest": "#1baf7a"}
CONDITION_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
CONDITION_MARKERS = ("o", "s", "D", "^", "v")
INK, INK_SOFT, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def _style():
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SOFT,
            "axes.titlecolor": INK,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.color": INK_SOFT,
            "ytick.color": INK_SOFT,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "lines.linewidth": 1.5,
            "font.size": 9,
            "text.color": INK,
        }
    )


def _short(condition):
    return condition.replace("_svbmc", "")


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.15)


def figure_bias(summary, out):
    import matplotlib.pyplot as plt

    settings = summary["settings"]
    rule, method = settings["headline_rule"], settings["headline_method"]
    Ms = sorted({e["M"] for c in summary["conditions"] for e in c["by_M"]})
    n_conditions = max(1, len(summary["conditions"]))
    fig, axes = plt.subplots(
        1,
        len(Ms),
        figsize=(max(5.5, 2.0 * n_conditions) * len(Ms) ** 0.5, 3.8),
        squeeze=False,
        sharey=True,
    )
    series = [
        ("raw", SERIES["raw"], "o", True, lambda e: e["bias"]["raw"]),
        (
            "capped_I",
            SERIES["capped"],
            "o",
            True,
            lambda e: e["bias"]["capped_I"],
        ),
        (
            f"honest {rule} {method}",
            SERIES["honest"],
            "o",
            True,
            lambda e: e["honest"][rule][method]["bias"],
        ),
        (
            f"honest {rule} precision",
            SERIES["honest"],
            "s",
            False,
            lambda e: e["honest"][rule]["precision"]["bias"],
        ),
    ]
    offsets = np.linspace(-0.27, 0.27, len(series))
    for ax, M in zip(axes[0], Ms):
        labels = []
        for i, condition in enumerate(summary["conditions"]):
            entry = next((e for e in condition["by_M"] if e["M"] == M), None)
            labels.append(_short(condition["condition"]))
            if entry is None:
                continue
            for offset, (name, color, marker, filled, get) in zip(
                offsets, series
            ):
                stat = get(entry)
                if stat["median"] is None:
                    continue
                ax.errorbar(
                    i + offset,
                    stat["median"],
                    yerr=[
                        [stat["median"] - stat["lo"]],
                        [stat["hi"] - stat["median"]],
                    ],
                    fmt=marker,
                    color=color,
                    markerfacecolor=color if filled else SURFACE,
                    markeredgewidth=1.5,
                    markersize=6,
                    capsize=2,
                    elinewidth=1,
                    label=name if i == 0 else None,
                )
        ax.axhline(0, color=INK_SOFT, linewidth=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(f"M = {M}: bias against e_log_joint_mc")
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("bias (nats); median over cells, 95 % bootstrap")
    axes[0][0].legend(loc="best")
    _save(fig, out / "bias_by_condition.png")
    plt.close(fig)


def _cell_files(results, out):
    """The per-cell arrays available for the figures, one per condition/M."""
    cells_dir = Path(out).parent / "cells"
    picks = {}
    for row in results["cells"]:
        key = (row["condition"], row["M"])
        current = picks.get(key)
        if current is None or row["repetition"] < current["repetition"]:
            picks[key] = row
    available, missing = {}, []
    for key, row in sorted(picks.items()):
        path = cells_dir / cell_filename(row)
        if path.exists():
            available[key] = (row, path)
        else:
            missing.append(path.name)
    return available, missing


def figure_components(results, out, available):
    import matplotlib.pyplot as plt

    settings = results["settings"]
    rule, method = settings["headline_rule"], settings["headline_method"]
    picks = {}
    for (condition, M), (row, path) in available.items():
        if condition not in picks or M > picks[condition][0]["M"]:
            picks[condition] = (row, path)
    conditions = sorted(picks)
    cols = min(3, len(conditions))
    rows_n = int(np.ceil(len(conditions) / cols))
    fig, axes = plt.subplots(
        rows_n, cols, figsize=(3.6 * cols, 3.6 * rows_n), squeeze=False
    )
    for ax in axes.ravel()[len(conditions) :]:
        ax.axis("off")
    for ax, condition in zip(axes.ravel(), conditions):
        row, path = picks[condition]
        with np.load(path, allow_pickle=False) as data:
            own_error = data["I_corr"] - data["truth"]
            honest_error = data["honest_G"] - data["truth"]
            covered = data["covered"]
            w = data["w"]
        size = 6 + 140 * w / max(w.max(), np.finfo(float).tiny)
        lim = float(np.max(np.abs(np.concatenate([own_error, honest_error]))))
        lim = max(lim * 1.08, 0.1)
        ax.plot(
            [-lim, lim], [-lim, lim], color=GRID, linewidth=1, linestyle="--"
        )
        ax.axhline(0, color=INK_SOFT, linewidth=0.7)
        ax.axvline(0, color=INK_SOFT, linewidth=0.7)
        ax.scatter(
            own_error[covered],
            honest_error[covered],
            s=size[covered],
            color=SERIES["honest"],
            edgecolor=SURFACE,
            linewidth=0.6,
            alpha=0.9,
            label="cross-covered",
        )
        if np.any(~covered):
            ax.scatter(
                own_error[~covered],
                honest_error[~covered],
                s=size[~covered],
                facecolor="none",
                edgecolor=INK_SOFT,
                linewidth=0.9,
                label="own value kept",
            )
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(
            f"{_short(condition)} (M = {row['M']}, r = {row['repetition']})"
        )
        weight_covered = row["honest"][rule][method]["weight_covered"]
        ax.text(
            0.03,
            0.97,
            f"covered weight {weight_covered:.2f}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            color=INK_SOFT,
        )
        ax.set_xlabel("own estimate − truth (nats)")
        ax.set_ylabel("honest estimate − truth (nats)")
        ax.legend(loc="lower right")
    fig.suptitle(
        "Per-component error, own expected log joint against the honest "
        f"one ({rule}, {method}); marker area follows the stacked weight",
        fontsize=10,
    )
    _save(fig, out / "component_errors.png")
    plt.close(fig)


def figure_coverage(summary, out):
    import matplotlib.pyplot as plt

    settings = summary["settings"]
    rules = list(settings["rules"])
    method = settings["headline_method"]
    conditions = [c for c in summary["conditions"] if c["by_M"]]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    x = np.arange(len(rules))
    for i, condition in enumerate(conditions):
        entry = max(condition["by_M"], key=lambda e: e["M"])
        color = CONDITION_COLORS[i % len(CONDITION_COLORS)]
        marker = CONDITION_MARKERS[i % len(CONDITION_MARKERS)]
        label = f"{_short(condition['condition'])} (M = {entry['M']})"
        axes[0].plot(
            x,
            [
                entry["honest"][r][method]["weight_covered"]["median"]
                for r in rules
            ],
            marker=marker,
            color=color,
            markersize=5,
            label=label,
        )
        axes[1].plot(
            x,
            [entry["honest"][r][method]["bias"]["median"] for r in rules],
            marker=marker,
            color=color,
            markersize=5,
            label=label,
        )
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(rules, rotation=20, ha="right")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("stacked weight cross-covered")
    axes[0].set_title("Coverage under each rule")
    axes[1].axhline(0, color=INK_SOFT, linewidth=0.8)
    axes[1].set_ylabel(f"honest bias ({method}), median over cells (nats)")
    axes[1].set_title("Honest bias under each rule")
    axes[1].set_yscale("symlog", linthresh=1.0)
    axes[0].legend(loc="lower left")
    _save(fig, out / "coverage.png")
    plt.close(fig)


def figure_calibration(out, available):
    import matplotlib.pyplot as plt

    own, cross = [], []
    for row, path in available.values():
        with np.load(path, allow_pickle=False) as data:
            truth, truth_se = data["truth"], data["truth_se"]
            own.append(
                (data["I_corr"] - truth)
                / np.sqrt(data["bq_sd"] ** 2 + truth_se**2)
            )
            rows, cols = np.nonzero(data["cover"])
            cross.append(
                (data["est"][rows, cols] - truth[cols])
                / np.sqrt(
                    data["v"][rows, cols] + data["se_diff"][rows, cols] ** 2
                )
            )
    own = np.concatenate(own)
    cross = np.concatenate(cross) if cross else np.empty(0)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    grid = np.linspace(-6, 6, 241)
    normal = np.exp(-0.5 * grid**2) / np.sqrt(2 * np.pi)
    for ax, z, color, title in (
        (
            axes[0],
            own,
            SERIES["raw"],
            "own run: (I_corr − truth) / sqrt(BQ sd² + MC se²)",
        ),
        (
            axes[1],
            cross,
            SERIES["honest"],
            "other runs: (estimate − truth) / sqrt(pred sd² + MC se²)",
        ),
    ):
        if z.size:
            ax.hist(
                np.clip(z, -6, 6),
                bins=np.linspace(-6, 6, 49),
                density=True,
                color=color,
                alpha=0.85,
                edgecolor=SURFACE,
                linewidth=0.5,
            )
            ax.text(
                0.02,
                0.95,
                f"n = {z.size}\nmedian z {np.median(z):+.2f}\n"
                f"median abs z {np.median(np.abs(z)):.2f}\n"
                f"abs z > 2: {np.mean(np.abs(z) > 2):.0%}",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                color=INK_SOFT,
            )
        ax.plot(
            grid,
            normal,
            color=INK_SOFT,
            linewidth=1,
            linestyle="--",
            label="standard normal",
        )
        ax.set_title(title)
        ax.set_xlabel("z (values beyond ±6 clipped)")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper right")
    axes[0].set_ylabel("density over components (unweighted)")
    _save(fig, out / "calibration.png")
    plt.close(fig)


def figure_sparsity(out, available):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    conditions = sorted({key[0] for key in available})
    for i, condition in enumerate(conditions):
        color = CONDITION_COLORS[i % len(CONDITION_COLORS)]
        marker = CONDITION_MARKERS[i % len(CONDITION_MARKERS)]
        first = True
        for (name, M), (row, path) in available.items():
            if name != condition:
                continue
            with np.load(path, allow_pickle=False) as data:
                columns = np.arange(len(data["w"]))
                own_n = data["n_eff"][data["run_index"], columns]
                own_error = data["I_corr"] - data["truth"]
                rows_c, cols_c = np.nonzero(data["cover"])
                cross_n = data["n_eff"][rows_c, cols_c]
                cross_error = (
                    data["est"][rows_c, cols_c] - data["truth"][cols_c]
                )
            label = _short(condition) if first else None
            first = False
            axes[0].scatter(
                np.maximum(own_n, 1e-2),
                own_error,
                s=9,
                color=color,
                marker=marker,
                alpha=0.7,
                edgecolor="none",
                label=label,
            )
            axes[1].scatter(
                np.maximum(cross_n, 1e-2),
                cross_error,
                s=9,
                color=color,
                marker=marker,
                alpha=0.7,
                edgecolor="none",
                label=label,
            )
    for ax, title in zip(
        axes,
        (
            "own run: I_corr − truth",
            "covering other runs: estimate − truth (headline rule)",
        ),
    ):
        ax.axhline(0, color=INK_SOFT, linewidth=0.8)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_xlabel("effective training points on the component")
        ax.set_title(title)
    axes[0].set_ylabel("error of the expected log joint (nats, symlog)")
    axes[1].legend(loc="lower right", markerscale=2)
    _save(fig, out / "sparsity.png")
    plt.close(fig)


def make_figures(results, summary, out):
    """Write the figures; the per-cell figures only when their arrays exist."""
    _style()
    figures = Path(out) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    if not results["cells"]:
        return figures
    figure_bias(summary, figures)
    figure_coverage(summary, figures)
    available, missing = _cell_files(results, figures)
    if missing:
        print(
            f"{len(missing)} per-cell array file(s) missing under "
            f"{Path(out) / 'cells'}; the component, calibration and "
            "sparsity figures use the cells that are present"
            + (
                " (none: those figures are left as they are)"
                if not available
                else ""
            )
        )
    if available:
        figure_components(results, figures, available)
        figure_calibration(figures, available)
        figure_sparsity(figures, available)
    return figures


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def cell_filename(row):
    return f"{row['condition']}_M{row['M']}_r{row['repetition']}.npz"


def save_cell_arrays(path, arrays, per_component):
    np.savez_compressed(
        path,
        K=arrays["K"],
        w=arrays["w"],
        run_index=arrays["run_index"],
        est=arrays["est"],
        se=arrays["se"],
        se_diff=arrays["se_diff"],
        v=arrays["v"],
        mean_fn=arrays["mean_fn"],
        n_eff=arrays["n_eff"],
        n_within=arrays["n_within"],
        data_offset=arrays["data_offset"],
        truth=arrays["truth"],
        truth_se=arrays["truth_se"],
        own_mc=arrays["own_mc"],
        own_mc_se=arrays["own_mc_se"],
        jac_mc=arrays["jac_mc"],
        jac_mc_se=arrays["jac_mc_se"],
        I_corr=arrays["I_corr"],
        jac=arrays["jac"],
        bq_sd=arrays["bq_sd"],
        honest_G=per_component["G"],
        cover=per_component["cover"],
        a=per_component["a"],
        covered=per_component["cover"].any(axis=0),
    )


def read_manifests(pool_dirs):
    manifests = []
    for directory in pool_dirs:
        path = Path(directory) / "manifest.json"
        if not path.exists():
            raise RuntimeError(f"{directory} holds no manifest.json")
        manifests.append(
            {
                "directory": str(Path(directory).resolve()),
                "manifest": json.loads(path.read_text(encoding="utf-8")),
            }
        )
    return manifests


def gpyreg_source(manifests, override):
    if override is not None:
        return str(Path(override).resolve())
    sources = {m["manifest"]["gpyreg_source"] for m in manifests}
    if len(sources) != 1:
        raise RuntimeError(
            f"the pools name different gpyreg sources {sorted(sources)}; "
            "pass --gpyreg-source"
        )
    return sources.pop()


def sources_record(manifests, source, cells_path, settings, tags):
    import gpyreg

    import pyvbmc

    gp_dir = Path(gpyreg.__file__).resolve().parent
    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(__file__),
        },
        "pyvbmc": {
            "commit": git(ROOT, "rev-parse", "HEAD"),
            "dirty": git(
                ROOT, "status", "--porcelain", "--", "pyvbmc", "dev/scripts"
            ),
            "import": str(Path(pyvbmc.__file__).resolve().parent),
        },
        "gpyreg": {
            "source": source,
            "import": str(gp_dir),
            "commit": git(gp_dir.parent, "rev-parse", "HEAD"),
        },
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "threads": {k: os.environ.get(k) for k in THREAD_KEYS},
        "cells": (
            None
            if cells_path is None
            else {
                "path": str(Path(cells_path).resolve()),
                "sha256": sha256(cells_path),
            }
        ),
        "pools": [
            {
                "directory": m["directory"],
                "identity": m["manifest"].get("identity"),
                "gpyreg_source": m["manifest"].get("gpyreg_source"),
            }
            for m in manifests
        ],
        "runs": sorted(tags),
        "settings": settings,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pool", action="append", type=Path, default=[])
    parser.add_argument("--cells", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--arm", choices=("integrated", "original"), default="integrated"
    )
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ratios", default=",".join(f"{r:g}" for r in RATIOS))
    parser.add_argument("--headline-ratio", type=float, default=HEADLINE_RATIO)
    parser.add_argument("--sd-cap", type=float, default=SD_CAP)
    parser.add_argument("--sd-floor", type=float, default=SD_FLOOR)
    parser.add_argument("--conditions", default=None)
    parser.add_argument("--gpyreg-source", type=Path, default=None)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    if args.summarize_only:
        return args
    if not args.pool:
        parser.error("--pool is required unless --summarize-only")
    if args.self_check and args.cells is not None:
        parser.error("--self-check takes no --cells")
    if not args.self_check and args.cells is None:
        parser.error("--cells is required unless --self-check")
    if args.draws < 2:
        parser.error("--draws must be at least 2")
    try:
        args.ratio_list = tuple(
            float(x) for x in args.ratios.split(",") if x.strip()
        )
    except ValueError:
        parser.error(
            f"--ratios must be comma-separated numbers: {args.ratios}"
        )
    if not args.ratio_list:
        parser.error("--ratios must name at least one ratio")
    if rule_name(args.headline_ratio) not in coverage_rules(
        args.ratio_list, args.sd_cap, args.sd_floor
    ):
        parser.error(
            f"--headline-ratio {args.headline_ratio:g} is not among "
            f"--ratios {args.ratios}"
        )
    return args


def summarize_only(args):
    results = json.loads(
        (args.out / "results.json").read_text(encoding="utf-8")
    )
    rng = np.random.default_rng(results["settings"]["seed"])
    summary = build_summary(results, rng)
    write_json(args.out / "summary.json", summary)
    (args.out / "summary.md").write_text(
        summary_markdown(summary), encoding="utf-8"
    )
    if not args.no_figures:
        make_figures(results, summary, args.out)
    print(f"summaries rebuilt under {args.out}")
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.summarize_only:
        return summarize_only(args)
    manifests = read_manifests(args.pool)
    source = gpyreg_source(manifests, args.gpyreg_source)
    activate_gpyreg(source)
    from benchmark_targets import find_config

    rules = coverage_rules(args.ratio_list, args.sd_cap, args.sd_floor)
    headline = (rule_name(args.headline_ratio), HEADLINE_METHOD)
    settings = {
        "arm": args.arm,
        "draws": int(args.draws),
        "seed": int(args.seed),
        "ratios": list(args.ratio_list),
        "sd_cap": float(args.sd_cap),
        "sd_floor": float(args.sd_floor),
        "rules": {k: [float(x) for x in rule] for k, rule in rules.items()},
        "methods": list(METHODS),
        "headline_rule": headline[0],
        "headline_method": headline[1],
        "z_flag": Z_FLAG,
        "z_flag_max": Z_FLAG_MAX,
        "within_radius": WITHIN_RADIUS,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "cells": None if args.cells is None else str(args.cells),
        "pools": [m["directory"] for m in manifests],
        "self_check": bool(args.self_check),
    }
    only = set(args.conditions.split(",")) if args.conditions else None
    runs = Runs(args.pool)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cells").mkdir(exist_ok=True)
    problems = {}

    def problem_of(label):
        if label not in problems:
            problems[label] = find_config(label).make(seed=0)
        return problems[label]

    started = time.perf_counter()
    cells = []
    if args.cells is not None:
        recorded = json.loads(args.cells.read_text(encoding="utf-8"))
        cells = [
            c
            for c in recorded["cells"]
            if args.arm in c.get("arms", {})
            and (only is None or c["condition"] in only)
        ]
        if not cells:
            raise RuntimeError(
                f"{args.cells} holds no cell of arm {args.arm}"
                + ("" if only is None else f" for {sorted(only)}")
            )
    if args.self_check:
        tags = [
            t
            for t in runs.tags()
            if only is None or runs.get(t)["label"] in only
        ]
        if not tags:
            raise RuntimeError("the pools hold no artifact to check")
    else:
        tags = sorted({t for c in cells for t in c["entries"]})

    run_rows = []
    for tag in tags:
        run = runs.get(tag)
        rng = np.random.default_rng(
            [int(args.seed), SELF_STREAM, zlib.crc32(tag.encode("utf-8"))]
        )
        row = self_check(run, problem_of(run["label"]), rng, args.draws)
        run_rows.append(row)
        checks = row["checks"]
        print(
            f"[run] {tag}: own-run max abs z {checks['own_mc_max_z']:.2f} "
            f"(offset z {checks['own_offset_z']:+.2f}), Jacobian max abs z "
            f"{checks['jac_max_z']:.2f} (offset z "
            f"{checks['jac_offset_z']:+.2f}), own error "
            f"{row['own_error']:+.3f} +- {row['own_error_se']:.3f}"
            + (" FLAGGED" if checks["flagged"] else ""),
            flush=True,
        )

    cell_rows, skipped = [], []
    jsonl = args.out / "cells.jsonl"
    jsonl.write_text("", encoding="utf-8")
    for cell in cells:
        label = f"{cell['condition']} M={cell['M']} r={cell['repetition']}"
        cell_started = time.perf_counter()
        try:
            cell_runs = [runs.get(tag) for tag in cell["entries"]]
            rng = np.random.default_rng(
                [int(args.seed), DRAW_STREAM, int(cell["cell_seed"])]
            )
            row, arrays, per_component = score_cell(
                cell,
                args.arm,
                cell_runs,
                problem_of(cell["condition"]),
                rng,
                args.draws,
                rules,
                METHODS,
                headline,
            )
        except Exception as error:  # noqa: BLE001
            skipped.append(
                {
                    "condition": cell["condition"],
                    "M": int(cell["M"]),
                    "repetition": int(cell["repetition"]),
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
            print(
                f"[cell] {label}: SKIPPED ({skipped[-1]['reason']})",
                flush=True,
            )
            continue
        row["seconds"] = time.perf_counter() - cell_started
        save_cell_arrays(
            args.out / "cells" / cell_filename(row), arrays, per_component
        )
        cell_rows.append(row)
        with jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        head = row["honest"][headline[0]][headline[1]]
        print(
            f"[cell] {label}: bias raw {row['bias']['raw']:+.3f}, capped_I "
            f"{row['bias']['capped_I']:+.3f}, honest {head['bias']:+.3f} "
            f"+- {head['mc_se']:.3f} (covered {head['weight_covered']:.2f}), "
            f"truth_strat {row['truth_strat']['bias']:+.3f}; "
            f"{row['seconds']:.1f} s"
            + (" FLAGGED" if row["checks"]["flagged"] else ""),
            flush=True,
        )

    results = {
        "campaign": "svbmc_pool_phase2",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": time.perf_counter() - started,
        "settings": settings,
        "runs": run_rows,
        "cells": cell_rows,
        "skipped": skipped,
    }
    write_json(args.out / "results.json", results)
    write_json(
        args.out / "sources.json",
        sources_record(manifests, source, args.cells, settings, tags),
    )
    summary = build_summary(results, np.random.default_rng(args.seed))
    write_json(args.out / "summary.json", summary)
    (args.out / "summary.md").write_text(
        summary_markdown(summary), encoding="utf-8"
    )
    if not args.no_figures:
        make_figures(results, summary, args.out)
    flagged = [r["tag"] for r in run_rows if r["checks"]["flagged"]] + [
        cell_filename(r) for r in cell_rows if r["checks"]["flagged"]
    ]
    print(
        f"{len(run_rows)} runs checked, {len(cell_rows)} cells scored, "
        f"{len(skipped)} skipped in {results['elapsed_seconds']:.0f} s; "
        f"outputs under {args.out}"
        + (f"; FLAGGED: {flagged}" if flagged else "")
    )
    return 1 if flagged or skipped else 0


if __name__ == "__main__":
    sys.exit(main())
