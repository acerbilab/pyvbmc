"""Ground truths for the benchmark targets: MCMC samples, posterior
moments and the log normalizing constant.

Companion of ``benchmark_targets.py``: every target it defines has bounds,
a plausible box and a log density, and this script turns one of them into
the reference a benchmark run is scored against. The defaults are the
real-data targets ``timing``, ``multisensory_s1`` and ``multisensory_s2``;
any other name the targets module knows works as well, as in ``--targets
halfnormal --quick``. ``logreg`` and ``halfnormal`` carry a normalizing
constant and moments of their own, so a short run on either says whether
the estimators here are working.

Method
------
Everything happens in PyVBMC's unbounded coordinates ``u``, where the box
constraints are gone and the posterior is closest to a Gaussian mixture.
With ``T`` the parameter transformer of the target's bounds, the target
there is

    g(u) = log p(T^-1(u)) + log |det J_{T^-1}(u)| ,

whose integral over R^D is the normalizing constant of ``p`` over the box.

1. **MAP and Laplace.** ``-g`` is minimized from 8 starts drawn uniformly
   in the plausible box (L-BFGS-B, numerical gradient); the best point is
   ``u*``. Its Hessian comes from central differences with a step of 1e-3
   in the transformed coordinates, where the transformer gives the
   plausible box width one in every dimension; it is symmetrized and
   raised to positive definiteness by lifting any non-positive eigenvalue
   to a small floor (recorded when it happens). The Laplace estimate is
   ``g(u*) + D/2 log(2 pi) - 1/2 log det H`` and ``L = chol(H^-1)`` is the
   whitening factor.
2. **Chains.** gpyreg's slice sampler runs in whitened coordinates ``z``,
   ``u = u* + L z``, so the strong parameter correlations do not stall the
   coordinate-wise moves. Chains start dispersed at ``u* + 2 L eps``,
   ``eps ~ N(0, I)``, are burned in for ``--burn-frac`` sweeps per stored
   draw per chain and thinned, and are drawn in chunks: every chunk is
   reported and written to ``chains/{name}_chain{k}.npz``. A rerun reuses
   a chain file that already holds the target number of draws, and draws
   every other chain from scratch, so the unit of resumption is one whole
   chain: a chain that finished is never redrawn, a chain interrupted
   halfway is redrawn from its start.
3. **Diagnostics.** Classic split R-hat and an autocorrelation-based
   effective sample size with Geyer's initial-positive-sequence
   truncation, per dimension, over the chains.
4. **Proposal.** A ``BayesianGaussianMixture`` (scikit-learn, used only to
   fit) on the first half of the draws, components under weight 1e-3
   dropped and the rest renormalized, plus a defensive component of weight
   0.05 carrying the draws' mean and four times their covariance. Its
   density and sampling are implemented here, exactly.
5. **lnZ.** Geyer's estimator between the second half of the chains and
   ``--n-proposal`` draws from that mixture -- with one normalized
   proposal this is the optimal bridge sampling estimator -- iterated to
   1e-10 in log space from the importance-sampling estimate. The standard
   error follows Fruehwirth-Schnatter (2004), with the second half's
   effective sample size in place of its length. The proposal draws are
   the only new target evaluations: the chain term reuses the log
   densities the sampler already returned.
6. **Cross-checks**, recorded but never asserted: defensive importance
   sampling from the same proposal draws with its effective sample size,
   the Laplace estimate, the value benchflow stores for a target that has one,
   and, for a target whose truth ``benchmark_targets.py`` already carries,
   the differences and z-scores against it.

A run prints a ``WARN`` line, and carries on, when the chains missed
R-hat 1.01 or 400 effective draws in some dimension, or when Geyer's
estimate and the importance-sampling estimate disagree by more than three
of their combined standard errors. These are the plan's acceptance
conditions, made visible in the log; the decision to keep a truth is the
reader's.

Outputs
-------
Under ``--out`` (default ``dev/scripts/data/truths``), per target, written
through a temporary name and renamed into place:

``{name}.npz``
    ``samples`` ``(n, D)``, all stored draws in the *original* parameter
    space; ``sample_logp`` ``(n,)``, the original-space log density at each
    draw; ``mean`` ``(D,)`` and ``cov`` ``(D, D)`` of those draws;
    ``ln_z`` and ``ln_z_se``.
``{name}.json``
    settings and seeds, per-dimension R-hat and effective sample sizes,
    the importance-sampling and Laplace cross-checks, the MAP, the
    proposal summary, the comparisons, target evaluation counts, the
    generating commit, timestamps and the elapsed seconds per stage.
``chains/{name}_chain{k}.npz``
    the whitened draws and their log densities, with the MAP and whitening
    they were drawn under, so a rerun can tell a reusable chain from a
    stale one. Under the default ``--out`` this subdirectory is gitignored
    while the truths beside it are tracked.

Usage
-----
``dev/scripts/runs/`` is gitignored, so an overnight log belongs there::

    mkdir -p dev/scripts/runs/truths_20260911
    python -u dev/scripts/make_benchmark_truths.py \
        --targets timing multisensory_s1 multisensory_s2 \
        > dev/scripts/runs/truths_20260911/run.log 2>&1
    python dev/scripts/make_benchmark_truths.py --check \
        --targets timing multisensory_s1 multisensory_s2

Progress lines carry a timestamp and are flushed, so the log of an
overnight run can be followed. ``--quick`` shortens every stage for a
smoke test; ``--check`` reloads the stored files, verifies that the stored
moments are the moments of the stored draws and that the stored log
densities are reproduced by the target, prints the diagnostics, and exits
nonzero if any of that fails (it needs no scikit-learn).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))

from benchmark_targets import (  # noqa: E402
    MULTISENSORY_S1_LN_Z_BENCHFLOW,
    make_problem,
)

DEFAULT_OUT = REPO_ROOT / "dev" / "scripts" / "data" / "truths"

# The targets generated when --targets is not given. DEFAULT_D holds the
# dimension of every target this script is meant for, the real-data ones
# and the two with a known truth, so none of them needs --D; any other
# target does.
DEFAULT_TARGETS = ("timing", "multisensory_s1", "multisensory_s2")
DEFAULT_D = {
    "timing": 5,
    "multisensory_s1": 6,
    "multisensory_s2": 6,
    "logreg": 5,
    "halfnormal": 2,
}

# Values stored by benchflow, reported as a difference and never asserted.
# For multisensory_s1 three estimators sharing no code agree about 0.29
# above benchflow's stored constant; the comment beside the constant in
# benchmark_targets.py records what that comparison is worth.
KNOWN_LN_Z = {"multisensory_s1": MULTISENSORY_S1_LN_Z_BENCHFLOW}

# VBMC's own default bounded transform, so the sampling space is the one
# the algorithm under test works in.
BOUNDED_TRANSFORM = "probit"

FULL_SETTINGS = {"n_samples": 50_000, "n_proposal": 20_000, "thin": 5}
QUICK_SETTINGS = {"n_samples": 2_000, "n_proposal": 2_000, "thin": 2}

N_MAP_STARTS = 8
# The transformer gives the plausible box width one in every transformed
# dimension, so this is the finite-difference step itself.
HESSIAN_REL_STEP = 1e-3
EIGENVALUE_FLOOR = 1e-8  # relative, when the Hessian is not PD
MAX_START_HALVINGS = 8  # of the dispersion of a chain's start point
N_MIXTURE_COMPONENTS = 10
MIXTURE_WEIGHT_FLOOR = 1e-3
BROAD_WEIGHT = 0.05
BROAD_INFLATION = 4.0
BRIDGE_TOL = 1e-10
BRIDGE_MAX_ITER = 1000
CHUNKS_PER_CHAIN = 10  # progress reports and partial saves per chain

# Conditions the plan asks a truth to meet, reported as WARN lines.
RHAT_GATE = 1.01
ESS_GATE = 400
AGREEMENT_SIGMAS = 3.0

CHECK_N_DRAWS = 100  # stored draws re-evaluated by --check
CHECK_LOGP_TOL = 1e-8  # absolute, plus 1e-12 of the stored value
CHECK_MOMENT_RTOL = 1e-10
CHECK_SCALAR_RTOL = 1e-12  # npz against sidecar


# --------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------


def _log(msg):
    """Print one timestamped, flushed progress line."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _git(*args):
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_state():
    """The commit the truth was generated at, and whether the checkout had
    uncommitted or untracked changes at the time."""
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=all")
    return (
        commit.strip() if commit is not None else None,
        bool(status.strip()) if status is not None else None,
    )


def _atomic_bytes(path, data):
    """Write ``data`` to ``path`` through a temporary name in the same
    directory, so an interrupted write never leaves a half file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _atomic_npz(path, arrays):
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    _atomic_bytes(path, buf.getvalue())


def _atomic_json(path, obj):
    data = json.dumps(_jsonable(obj), indent=2).encode("utf-8")
    _atomic_bytes(path, data + b"\n")


def _jsonable(obj):
    """Recursively convert NumPy scalars and arrays to JSON types."""
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    return obj


def _logmeanexp(a):
    return logsumexp(a) - np.log(a.size)


def _fmt(value, spec=".6f"):
    """Format a number that may be missing, NaN or infinite."""
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return format(number, spec) if math.isfinite(number) else str(number)


def _target_seed_sequence(seed, name):
    """A per-target seed sequence, so a target's stream does not depend on
    which other targets were asked for or in what order."""
    digest = hashlib.sha256(name.encode("utf-8")).digest()[:4]
    return np.random.SeedSequence([int(seed), int.from_bytes(digest, "big")])


# --------------------------------------------------------------------------
# The target in transformed space
# --------------------------------------------------------------------------


class TransformedTarget:
    """A benchmark target pushed forward to unbounded coordinates.

    ``g`` maps an ``(n, D)`` array of transformed points to
    ``log p(T^-1(u)) + log |det J(u)|``, whose integral over R^D is the
    normalizing constant of the target over its box. Rows and calls are
    counted so the report can say how many target evaluations each stage
    cost.
    """

    # Value returned to the optimizer where ``g`` is not finite: large
    # enough to be rejected, finite so L-BFGS-B's line search survives it.
    SENTINEL = 1e20

    def __init__(self, problem, transformer):
        self.problem = problem
        self.transformer = transformer
        self.D = int(problem.D)
        self.n_calls = 0
        self.n_rows = 0

    def g(self, U):
        U = np.atleast_2d(np.asarray(U, dtype=np.float64))
        X = self.transformer.inverse(U)
        values = np.asarray(
            self.problem.log_density_vec(X), dtype=np.float64
        ).ravel()
        out = values + self.transformer.log_abs_det_jacobian(U)
        self.n_calls += 1
        self.n_rows += U.shape[0]
        return np.asarray(out, dtype=np.float64)

    def g1(self, u):
        """``g`` at a single point, as a float."""
        return float(self.g(np.reshape(u, (1, self.D)))[0])

    def neg_g1(self, u):
        value = self.g1(u)
        return -value if np.isfinite(value) else self.SENTINEL


def make_transformer(problem):
    """PyVBMC's transformer for the problem's bounds."""
    from pyvbmc.parameter_transformer import ParameterTransformer

    D = int(problem.D)

    def row(v):
        return np.asarray(v, dtype=np.float64).reshape(1, D)

    return ParameterTransformer(
        D,
        row(problem.lb),
        row(problem.ub),
        row(problem.plb),
        row(problem.pub),
        transform_type=BOUNDED_TRANSFORM,
    )


# --------------------------------------------------------------------------
# MAP, Hessian, Laplace
# --------------------------------------------------------------------------


def find_map(target, problem, rng):
    """Multi-start minimization of ``-g``; returns the best point in
    transformed space, ``g`` there, and the value of every start."""
    D = target.D
    plb = np.asarray(problem.plb, dtype=np.float64).reshape(1, D)
    pub = np.asarray(problem.pub, dtype=np.float64).reshape(1, D)
    starts_x = plb + rng.random((N_MAP_STARTS, D)) * (pub - plb)
    starts_u = np.asarray(
        target.transformer(starts_x), dtype=np.float64
    ).reshape(N_MAP_STARTS, D)

    best_u, best_value, values = None, np.inf, []
    for k in range(N_MAP_STARTS):
        result = minimize(
            target.neg_g1,
            starts_u[k],
            method="L-BFGS-B",
            options={"maxiter": 10_000, "maxfun": 100_000},
        )
        value = float(np.ravel(result.fun)[0])
        values.append(value)
        if np.isfinite(value) and value < best_value:
            best_value = value
            best_u = np.asarray(result.x, dtype=np.float64).copy()
        _log(f"    start {k + 1}/{N_MAP_STARTS}: g = {-value:.6f}")
    if best_u is None:
        raise RuntimeError("every MAP start failed to find a finite value")
    return best_u, -best_value, values


def hessian_fd(target, u_star, widths):
    """Hessian of ``-g`` at ``u_star`` by central differences.

    The step is ``HESSIAN_REL_STEP`` times ``widths``, the plausible box's
    widths in transformed space, which the transformer makes one in every
    dimension; they are measured rather than assumed. All ``1 + 2 D^2``
    points go through one batched call.
    """
    D = u_star.size
    h = HESSIAN_REL_STEP * np.asarray(widths, dtype=np.float64).ravel()
    points, index = [u_star.copy()], {}
    for i in range(D):
        for si in (1, -1):
            index[(i, si)] = len(points)
            point = u_star.copy()
            point[i] += si * h[i]
            points.append(point)
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    index[(i, j, si, sj)] = len(points)
                    point = u_star.copy()
                    point[i] += si * h[i]
                    point[j] += sj * h[j]
                    points.append(point)

    values = target.g(np.asarray(points, dtype=np.float64))
    if not np.all(np.isfinite(values)):
        raise RuntimeError("the target is not finite around the MAP")
    f0 = values[0]

    H = np.zeros((D, D), dtype=np.float64)
    for i in range(D):
        plus, minus = values[index[(i, 1)]], values[index[(i, -1)]]
        H[i, i] = -(plus - 2.0 * f0 + minus) / h[i] ** 2
    for i in range(D):
        for j in range(i + 1, D):
            mixed = (
                values[index[(i, j, 1, 1)]]
                - values[index[(i, j, 1, -1)]]
                - values[index[(i, j, -1, 1)]]
                + values[index[(i, j, -1, -1)]]
            )
            H[i, j] = H[j, i] = -mixed / (4.0 * h[i] * h[j])
    return 0.5 * (H + H.T), float(f0), h


def make_positive_definite(H):
    """Lift any non-positive eigenvalue of ``H`` to a small floor, and say
    whether that was needed."""
    eigenvalues, vectors = np.linalg.eigh(H)
    floor = EIGENVALUE_FLOOR * max(float(np.max(eigenvalues)), 1.0)
    info = {
        "regularized": bool(np.min(eigenvalues) < floor),
        "min_eigenvalue": float(np.min(eigenvalues)),
        "max_eigenvalue": float(np.max(eigenvalues)),
        "floor": float(floor),
    }
    if not info["regularized"]:
        return H, info
    lifted = np.maximum(eigenvalues, floor)
    H = vectors @ np.diag(lifted) @ vectors.T
    H = 0.5 * (H + H.T)
    info["min_eigenvalue_after"] = float(np.min(lifted))
    return H, info


def laplace_and_whitening(H, g_star, D):
    """``ln_z_laplace`` and the whitening factor ``L = chol(H^-1)``."""
    covariance = np.linalg.inv(H)
    covariance = 0.5 * (covariance + covariance.T)
    L = np.linalg.cholesky(covariance)
    log_det_L = float(np.sum(np.log(np.diag(L))))
    ln_z_laplace = g_star + 0.5 * D * np.log(2 * np.pi) + log_det_L
    return L, log_det_L, float(ln_z_laplace)


# --------------------------------------------------------------------------
# Chains
# --------------------------------------------------------------------------


def chain_path(out_dir, name, k):
    return out_dir / "chains" / f"{name}_chain{k}.npz"


def _save_chain(path, z, f_vals, state, thin, burn, n_target, k):
    u_star, L, g_star = state
    _atomic_npz(
        path,
        {
            "z": np.asarray(z, dtype=np.float64),
            "f_vals": np.asarray(f_vals, dtype=np.float64),
            "u_star": np.asarray(u_star, dtype=np.float64),
            "L": np.asarray(L, dtype=np.float64),
            "g_star": np.float64(g_star),
            "thin": np.int64(thin),
            "burn": np.int64(burn),
            "n_target": np.int64(n_target),
            "chain": np.int64(k),
        },
    )


def _load_chain(path, n_target, thin, burn, state):
    """The stored chain if it is complete and was drawn from this target in
    this whitening, otherwise ``None`` and the reason it cannot be used."""
    u_star, L, g_star = state
    if not path.exists():
        return None, "no file"
    try:
        with np.load(path, allow_pickle=False) as f:
            z = np.asarray(f["z"], dtype=np.float64)
            f_vals = np.asarray(f["f_vals"], dtype=np.float64).ravel()
            stored_u = np.asarray(f["u_star"], dtype=np.float64)
            stored_L = np.asarray(f["L"], dtype=np.float64)
            stored_g = float(f["g_star"])
            stored_thin = int(f["thin"])
            stored_burn = int(f["burn"])
    except (OSError, ValueError, KeyError) as exc:
        return None, f"unreadable ({exc})"
    if z.ndim != 2 or z.shape[1] != u_star.size:
        return None, "wrong shape"
    if z.shape[0] < n_target:
        return None, f"partial ({z.shape[0]} of {n_target} draws)"
    if stored_thin != int(thin):
        return None, f"thinning {stored_thin}, wanted {thin}"
    if stored_burn != int(burn):
        return None, f"burn-in {stored_burn} sweeps, wanted {burn}"
    if not np.allclose(stored_u, u_star, rtol=1e-10, atol=1e-12):
        return None, "different MAP"
    if not np.allclose(stored_L, L, rtol=1e-10, atol=1e-12):
        return None, "different whitening"
    # The MAP alone does not identify the target: a changed log density can
    # keep its argmax. Its value at the MAP is the cheap discriminator.
    if not abs(stored_g - g_star) < 1e-9:
        return None, f"g at the MAP {stored_g:.10g}, now {g_star:.10g}"
    return (z[:n_target], f_vals[:n_target]), "complete"


def run_chain(target, k, state, n_target, thin, burn, chunk, seed_seq, path):
    """One whitened slice-sampling chain, resumable through ``path``.

    ``state`` is the ``(u*, L, g(u*))`` the chain is drawn in, stored with
    the draws so a later run can tell a reusable chain from a stale one.
    Returns the stored draws in whitened coordinates, their log densities
    under ``g``, and the number of target evaluations spent (zero when the
    chain was reused).
    """
    u_star, L, _ = state
    reused, reason = _load_chain(path, n_target, thin, burn, state)
    if reused is not None:
        _log(
            f"  chain {k}: reusing {path.name}, {n_target} draws already "
            "stored, not resampling"
        )
        return reused[0], reused[1], 0
    _log(f"  chain {k}: sampling from scratch ({reason})")

    from gpyreg.slice_sample import SliceSampler

    rng = np.random.default_rng(seed_seq)
    D = u_star.size
    spent0 = target.n_rows

    # A dispersed start can land outside the support of a target that is
    # -inf somewhere; pull it back towards the MAP until it is inside.
    z0 = 2.0 * rng.standard_normal(D)
    halvings = 0
    while not np.isfinite(target.g1(u_star + L @ z0)):
        if halvings >= MAX_START_HALVINGS:
            raise RuntimeError(
                f"chain {k}: no finite start point near the MAP after "
                f"{MAX_START_HALVINGS} halvings of the dispersion"
            )
        z0 = 0.5 * z0
        halvings += 1
    if halvings:
        _log(
            f"  chain {k}: the dispersed start was outside the support, "
            f"halved it {halvings} time(s)"
        )

    sampler = SliceSampler(
        lambda z: target.g1(u_star + L @ z),
        z0,
        widths=1.0,
        # Its own diagnostics would run on every chunk and be discarded:
        # the R-hat and effective sizes reported come from split_rhat_ess
        # over whole chains.
        options={"display": "off", "diagnostics": False},
        rng=rng,
    )

    z_parts, f_parts, drawn = [], [], 0
    first = True
    while drawn < n_target:
        n_chunk = int(min(chunk, n_target - drawn))
        started = time.perf_counter()
        result = sampler.sample(
            n_chunk, thin=int(thin), burn=int(burn) if first else 0
        )
        first = False
        z_parts.append(np.asarray(result["samples"], dtype=np.float64))
        f_parts.append(np.asarray(result["f_vals"], dtype=np.float64).ravel())
        drawn += n_chunk
        z = np.vstack(z_parts)
        f_vals = np.concatenate(f_parts)
        _save_chain(path, z, f_vals, state, thin, burn, n_target, k)
        elapsed = time.perf_counter() - started
        _log(
            f"  chain {k}: {drawn}/{n_target} draws, "
            f"{elapsed:.1f} s for the last {n_chunk} "
            f"({elapsed / n_chunk * 1e3:.1f} ms/draw), "
            f"mean g {f_vals[-n_chunk:].mean():.4f}, saved"
        )
    return z, f_vals, target.n_rows - spent0


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------


def _autocovariance(x):
    """Autocovariance of a 1-D sequence by FFT (divisor ``n``)."""
    n = x.size
    size = 1 << int(np.ceil(np.log2(max(2 * n, 2))))
    centered = x - x.mean()
    spectrum = np.fft.rfft(centered, size)
    acov = np.fft.irfft(spectrum * np.conjugate(spectrum), size)[:n]
    return acov / n


def split_rhat_ess(chains):
    """Classic split R-hat and effective sample size per dimension.

    ``chains`` is ``(m, n, D)``. Each chain is split in half, giving ``2m``
    sequences; R-hat is the square root of the ratio of the pooled variance
    estimate to the within-sequence variance, and the effective sample size
    divides the total draws by the integrated autocorrelation time summed
    over Geyer's initial positive sequence of autocorrelation pairs.
    """
    chains = np.asarray(chains, dtype=np.float64)
    m, n, D = chains.shape
    half = n // 2
    if m < 1 or half < 4:
        nan = np.full(D, np.nan)
        return nan, nan
    split = np.concatenate(
        [chains[:, :half, :], chains[:, n - half :, :]], axis=0
    )
    M, N = split.shape[0], half

    means = split.mean(axis=1)
    variances = split.var(axis=1, ddof=1)
    W = variances.mean(axis=0)
    B = N * means.var(axis=0, ddof=1) if M > 1 else np.zeros(D)
    var_hat = (N - 1) / N * W + B / N
    with np.errstate(divide="ignore", invalid="ignore"):
        rhat = np.sqrt(var_hat / W)

    ess = np.empty(D, dtype=np.float64)
    max_pair = (N - 2) // 2
    for d in range(D):
        acov = np.array(
            [_autocovariance(split[j, :, d]) for j in range(M)],
            dtype=np.float64,
        )
        mean_acov = acov.mean(axis=0) * N / (N - 1)
        rho = 1.0 - (W[d] - mean_acov) / var_hat[d]
        rho[0] = 1.0
        tau, t = -1.0, 0
        while t <= max_pair:
            pair = rho[2 * t] + rho[2 * t + 1]
            if pair <= 0.0:
                break
            tau += 2.0 * pair
            t += 1
        # Stan's floor on tau, so a chain that looks antithetic over its
        # whole length cannot report an unbounded effective size.
        tau = max(tau, 1.0 / np.log10(M * N))
        ess[d] = M * N / tau
    return rhat, ess


# --------------------------------------------------------------------------
# Proposal mixture
# --------------------------------------------------------------------------


class GaussianMixture:
    """A Gaussian mixture with an exact log density and exact sampling.

    Built from weights, means and full covariances; a Cholesky factor per
    component serves both. Independent of how the components were found.
    """

    def __init__(self, weights, means, covariances):
        weights = np.asarray(weights, dtype=np.float64).ravel()
        self.weights = weights / weights.sum()
        self.means = np.asarray(means, dtype=np.float64)
        self.covariances = np.asarray(covariances, dtype=np.float64)
        self.K, self.D = self.means.shape
        self.chols = np.array(
            [np.linalg.cholesky(c) for c in self.covariances]
        )
        self.log_weights = np.log(self.weights)
        self.log_norms = np.array(
            [
                -np.sum(np.log(np.diag(chol)))
                - 0.5 * self.D * np.log(2 * np.pi)
                for chol in self.chols
            ]
        )

    def logpdf(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        parts = np.empty((self.K, X.shape[0]), dtype=np.float64)
        for k in range(self.K):
            z = np.linalg.solve(self.chols[k], (X - self.means[k]).T)
            parts[k] = (
                self.log_weights[k]
                + self.log_norms[k]
                - 0.5 * np.sum(z**2, axis=0)
            )
        return logsumexp(parts, axis=0)

    def rvs(self, n, rng):
        which = rng.choice(self.K, size=int(n), p=self.weights)
        eps = rng.standard_normal((int(n), self.D))
        out = np.empty((int(n), self.D), dtype=np.float64)
        for k in range(self.K):
            mask = which == k
            if np.any(mask):
                out[mask] = self.means[k] + eps[mask] @ self.chols[k].T
        return out


def fit_proposal(Z, random_state):
    """A variational Gaussian mixture on ``Z`` plus a broad component.

    scikit-learn fits the mixture and nothing else: the components it
    returns are pruned of weights below ``MIXTURE_WEIGHT_FLOOR``,
    renormalized, shrunk to make room for a defensive component carrying
    the draws' mean and ``BROAD_INFLATION`` times their covariance, and
    handed to :class:`GaussianMixture`.
    """
    # Lazy, so --check runs where scikit-learn is not installed.
    from sklearn.mixture import BayesianGaussianMixture

    fitted = BayesianGaussianMixture(
        n_components=N_MIXTURE_COMPONENTS,
        covariance_type="full",
        weight_concentration_prior_type="dirichlet_process",
        max_iter=2000,
        random_state=int(random_state),
    ).fit(Z)

    weights = np.asarray(fitted.weights_, dtype=np.float64)
    keep = weights >= MIXTURE_WEIGHT_FLOOR
    if not np.any(keep):
        keep = weights >= weights.max()
    kept = weights[keep] / weights[keep].sum()
    means = np.asarray(fitted.means_, dtype=np.float64)[keep]
    covariances = np.asarray(fitted.covariances_, dtype=np.float64)[keep]

    broad_mean = Z.mean(axis=0)
    broad_cov = BROAD_INFLATION * np.cov(Z, rowvar=False)
    mixture = GaussianMixture(
        np.concatenate([(1.0 - BROAD_WEIGHT) * kept, [BROAD_WEIGHT]]),
        np.vstack([means, broad_mean[None, :]]),
        np.concatenate([covariances, broad_cov[None, :, :]], axis=0),
    )
    summary = {
        "n_components": int(mixture.K),
        "weights": mixture.weights.tolist(),
        "n_fitted": int(N_MIXTURE_COMPONENTS),
        "n_kept": int(np.sum(keep)),
        "weight_floor": MIXTURE_WEIGHT_FLOOR,
        "broad_weight": BROAD_WEIGHT,
        "broad_inflation": BROAD_INFLATION,
        "converged": bool(fitted.converged_),
        "n_iter": int(fitted.n_iter_),
        "random_state": int(random_state),
    }
    return mixture, summary


# --------------------------------------------------------------------------
# Normalizing constant
# --------------------------------------------------------------------------


def geyer_ln_z(l1, l2, n1_eff):
    """Geyer's estimator (optimal bridge sampling) of ``ln Z``.

    ``l1`` are ``g(x_i) - log q(x_i)`` at draws from the target, ``l2`` the
    same at draws from the proposal, with ``q`` normalized. The fixed point
    is iterated in log space from the importance-sampling estimate; the
    standard error is Fruehwirth-Schnatter's (2004), with ``n1_eff`` the
    effective sample size of the target draws, never more than their
    number. Entries of ``l2`` at ``-inf`` (a proposal draw outside the
    target's support) carry zero weight through every sum and belong in
    ``n2``.
    """
    n1, n2 = int(l1.size), int(l2.size)
    n1_eff = min(float(n1_eff), float(n1))
    log_s1 = np.log(n1 / (n1 + n2))
    log_s2 = np.log(n2 / (n1 + n2))
    log_r = float(logsumexp(l2) - np.log(n2))

    delta, iterations = np.inf, 0
    for iterations in range(1, BRIDGE_MAX_ITER + 1):
        numerator = _logmeanexp(l2 - np.logaddexp(log_s1 + l2, log_s2 + log_r))
        denominator = _logmeanexp(-np.logaddexp(log_s1 + l1, log_s2 + log_r))
        updated = float(numerator - denominator)
        delta = abs(updated - log_r)
        log_r = updated
        if delta < BRIDGE_TOL:
            break

    # f1 = q / (s1 p~ + s2 r q) at the target draws, f2 = p~ / (same) at the
    # proposal draws; both in log space, rescaled before the variances,
    # which are invariant to the common factor.
    log_f1 = -np.logaddexp(log_s1 + l1, log_s2 + log_r)
    log_f2 = l2 - np.logaddexp(log_s1 + l2, log_s2 + log_r)
    f1 = np.exp(log_f1 - log_f1.max())
    f2 = np.exp(log_f2 - log_f2.max())
    variance = np.var(f1, ddof=1) / (n1_eff * f1.mean() ** 2) + np.var(
        f2, ddof=1
    ) / (n2 * f2.mean() ** 2)
    return log_r, float(np.sqrt(variance)), int(iterations), float(delta)


def importance_sampling_ln_z(l2):
    """Defensive importance sampling from the same proposal draws."""
    n2 = int(l2.size)
    ln_z = float(logsumexp(l2) - np.log(n2))
    weights = np.exp(l2 - l2.max())
    ess = float(weights.sum() ** 2 / np.sum(weights**2))
    se = float(np.sqrt(max(n2 / ess - 1.0, 0.0) / n2))
    return ln_z, se, ess


# --------------------------------------------------------------------------
# Comparisons with what is already known
# --------------------------------------------------------------------------


def compare_with_known(name, problem, ln_z, ln_z_se, mean, cov, ess):
    """Differences against a value benchflow stores and against the truth the
    target already carries, reported and never asserted."""
    out = {}
    if name in KNOWN_LN_Z:
        known = KNOWN_LN_Z[name]
        out["known_ln_z"] = {
            "value": known,
            "difference": float(ln_z - known),
        }
        _log(
            f"  benchflow ln_z {known:.6f}: difference "
            f"{ln_z - known:+.6f} ({(ln_z - known) / ln_z_se:+.2f} se)"
        )
    if problem.ln_Z is not None:
        stored = float(problem.ln_Z)
        out["target_ln_z"] = {
            "value": stored,
            "difference": float(ln_z - stored),
            "z_score": float((ln_z - stored) / ln_z_se),
        }
        _log(
            f"  target ln_Z {stored:.6f}: difference {ln_z - stored:+.6f} "
            f"({(ln_z - stored) / ln_z_se:+.2f} se)"
        )
    if problem.true_mean is not None:
        true_mean = np.asarray(problem.true_mean, dtype=np.float64).ravel()
        se = np.sqrt(np.diag(cov) / np.maximum(ess, 1.0))
        z = (mean - true_mean) / se
        out["target_mean"] = {
            "value": true_mean.tolist(),
            "difference": (mean - true_mean).tolist(),
            "z_score": z.tolist(),
            "max_abs_z": float(np.max(np.abs(z))),
        }
        _log(
            "  target mean: max |z| "
            f"{np.max(np.abs(z)):.2f}, z = " + " ".join(f"{v:+.2f}" for v in z)
        )
    if problem.true_cov is not None:
        true_cov = np.asarray(problem.true_cov, dtype=np.float64)
        sd, true_sd = np.sqrt(np.diag(cov)), np.sqrt(np.diag(true_cov))
        ratio = sd / true_sd
        out["target_cov"] = {
            "sd_ratio": ratio.tolist(),
            "max_abs_cov_difference": float(np.max(np.abs(cov - true_cov))),
        }
        _log(
            "  target sd ratio: "
            + " ".join(f"{v:.3f}" for v in ratio)
            + f" (max |cov diff| {np.max(np.abs(cov - true_cov)):.4g})"
        )
    return out


# --------------------------------------------------------------------------
# The generator
# --------------------------------------------------------------------------


def generate(name, D, out_dir, settings):
    """Generate and store the ground truth of one target."""
    started_at, started = _now(), time.perf_counter()
    elapsed = {}
    _log(f"{name} (D = {D}): starting")

    seeds = _target_seed_sequence(settings["seed"], name)
    map_seq, chain_seq, fit_seq, proposal_seq = seeds.spawn(4)

    problem = make_problem(name, int(D))
    if int(problem.D) != int(D):
        raise ValueError(f"{name} was built at D = {problem.D}, not {D}")
    transformer = make_transformer(problem)
    target = TransformedTarget(problem, transformer)

    # --- MAP, Hessian, Laplace ---------------------------------------
    stage = time.perf_counter()
    _log(f"  MAP: {N_MAP_STARTS} starts")
    u_star, g_star, start_values = find_map(
        target, problem, np.random.default_rng(map_seq)
    )
    x_star = np.asarray(
        transformer.inverse(u_star.reshape(1, D)), dtype=np.float64
    ).ravel()
    _log(f"  MAP: g = {g_star:.6f} at x = {np.array2string(x_star)}")

    plb_u = np.asarray(transformer(np.asarray(problem.plb).reshape(1, D)))
    pub_u = np.asarray(transformer(np.asarray(problem.pub).reshape(1, D)))
    widths = np.abs(pub_u - plb_u).ravel()
    H, g_at_map, steps = hessian_fd(target, u_star, widths)
    H, hessian_info = make_positive_definite(H)
    hessian_info["step"] = steps.tolist()
    if hessian_info["regularized"]:
        _log(
            "  Hessian: not positive definite (min eigenvalue "
            f"{hessian_info['min_eigenvalue']:.4g}), lifted to "
            f"{hessian_info['floor']:.4g}"
        )
    L, log_det_L, ln_z_laplace = laplace_and_whitening(H, g_at_map, D)
    _log(f"  Laplace: ln_z = {ln_z_laplace:.6f}")
    elapsed["map_and_laplace"] = time.perf_counter() - stage
    evals_map = target.n_rows

    # --- Chains --------------------------------------------------------
    stage = time.perf_counter()
    n_chains = int(settings["chains"])
    per_chain = int(settings["n_samples"]) // n_chains
    if per_chain < 8:
        raise ValueError("--n-samples is too small for --chains")
    burn = int(round(settings["burn_frac"] * per_chain))
    chunk = max(1, int(np.ceil(per_chain / CHUNKS_PER_CHAIN)))
    thin = int(settings["thin"])
    _log(
        f"  chains: {n_chains} x {per_chain} draws, thin {thin}, "
        f"burn-in {burn} sweeps, chunks of {chunk}"
    )
    (out_dir / "chains").mkdir(parents=True, exist_ok=True)
    chain_seqs = chain_seq.spawn(n_chains)
    state = (u_star, L, g_at_map)
    Z_chains, F_chains, chain_evals = [], [], []
    for k in range(n_chains):
        z, f_vals, spent = run_chain(
            target,
            k,
            state,
            per_chain,
            thin,
            burn,
            chunk,
            chain_seqs[k],
            chain_path(out_dir, name, k),
        )
        Z_chains.append(z)
        F_chains.append(f_vals)
        chain_evals.append(int(spent))
    Z_chains = np.asarray(Z_chains, dtype=np.float64)
    F_chains = np.asarray(F_chains, dtype=np.float64)
    elapsed["chains"] = time.perf_counter() - stage

    # --- Diagnostics ---------------------------------------------------
    stage = time.perf_counter()
    rhat, ess = split_rhat_ess(Z_chains)
    _log(
        f"  diagnostics: max R-hat {np.max(rhat):.4f}, min ESS "
        f"{np.min(ess):.0f} of {n_chains * per_chain} draws"
    )
    _log("    R-hat " + " ".join(f"{v:.4f}" for v in rhat))
    _log("    ESS   " + " ".join(f"{v:.0f}" for v in ess))
    if np.max(rhat) > RHAT_GATE:
        _log(
            f"  WARN: max R-hat {np.max(rhat):.4f} is above {RHAT_GATE}; "
            "the chains have not mixed"
        )
    if np.min(ess) < ESS_GATE:
        _log(
            f"  WARN: min effective sample size {np.min(ess):.0f} is below "
            f"{ESS_GATE}; sample longer or thin less"
        )
    elapsed["diagnostics"] = time.perf_counter() - stage

    # --- Proposal ------------------------------------------------------
    stage = time.perf_counter()
    half = per_chain // 2
    Z_first = Z_chains[:, :half, :].reshape(-1, D)
    Z_second = Z_chains[:, half:, :].reshape(-1, D)
    F_second = F_chains[:, half:].reshape(-1)
    random_state = int(fit_seq.generate_state(1, dtype=np.uint32)[0])
    mixture, proposal_summary = fit_proposal(Z_first, random_state)
    _log(
        f"  proposal: {mixture.K} components from {len(Z_first)} draws, "
        f"weights " + " ".join(f"{w:.3f}" for w in mixture.weights)
    )
    elapsed["proposal_fit"] = time.perf_counter() - stage

    # --- Geyer ---------------------------------------------------------
    stage = time.perf_counter()
    n_proposal = int(settings["n_proposal"])
    proposal_rng = np.random.default_rng(proposal_seq)
    Z_proposal = mixture.rvs(n_proposal, proposal_rng)
    U_proposal = u_star + Z_proposal @ L.T
    _log(f"  proposal: evaluating the target at {n_proposal} draws")
    evals_before = target.n_rows
    g_proposal = target.g(U_proposal)
    # q as a density over u: the whitening's constant Jacobian moves the
    # mixture's density from z to u, so log r below is ln Z itself.
    log_q_second = mixture.logpdf(Z_second) - log_det_L
    log_q_proposal = mixture.logpdf(Z_proposal) - log_det_L
    l1 = F_second - log_q_second
    l2 = g_proposal - log_q_proposal

    # A proposal draw where the target is zero contributes zero to every
    # sum below and still belongs in n2: dropping it would raise both
    # estimates by log(n2 / n2_kept). Only a NaN has to go.
    n_zero = int(np.sum(np.isneginf(l2)))
    n_nan = int(np.sum(np.isnan(l2)))
    if n_zero or n_nan:
        _log(
            f"  proposal draws off the target: {n_zero} with zero density "
            f"(kept, they weigh nothing), {n_nan} returning NaN (dropped)"
        )
    if n_nan:
        l2 = l2[~np.isnan(l2)]
    proposal_draws = {
        "n": int(n_proposal),
        "zero_density": n_zero,
        "nan": n_nan,
        "used": int(l2.size),
    }

    rhat_second, ess_second = split_rhat_ess(Z_chains[:, half:, :])
    n1_eff = float(min(np.min(ess_second), l1.size))
    ln_z, ln_z_se, iterations, final_delta = geyer_ln_z(l1, l2, n1_eff)
    ln_z_is, ln_z_is_se, is_ess = importance_sampling_ln_z(l2)
    elapsed["geyer"] = time.perf_counter() - stage
    evals_proposal = target.n_rows - evals_before

    _log(
        f"  Geyer: ln_z = {ln_z:.6f} +- {ln_z_se:.6f} "
        f"({iterations} iterations, last step {final_delta:.2g}; "
        f"n1 = {l1.size}, n1_eff = {n1_eff:.0f}, n2 = {l2.size})"
    )
    _log(
        f"  importance sampling: ln_z = {ln_z_is:.6f} +- {ln_z_is_se:.6f} "
        f"(ESS {is_ess:.0f} of {l2.size}; difference from Geyer "
        f"{ln_z_is - ln_z:+.6f})"
    )
    _log(
        f"  Laplace: ln_z = {ln_z_laplace:.6f} (difference "
        f"{ln_z_laplace - ln_z:+.6f})"
    )
    if final_delta >= BRIDGE_TOL:
        _log(
            f"  WARN: the Geyer fixed point stopped at {BRIDGE_MAX_ITER} "
            f"iterations with a last step of {final_delta:.3g}, above the "
            f"tolerance {BRIDGE_TOL:.3g}"
        )
    combined_se = float(np.hypot(ln_z_se, ln_z_is_se))
    if abs(ln_z - ln_z_is) > AGREEMENT_SIGMAS * combined_se:
        _log(
            f"  WARN: Geyer and importance sampling differ by "
            f"{ln_z - ln_z_is:+.6f}, more than {AGREEMENT_SIGMAS:g} of "
            f"their combined standard error {combined_se:.6f}"
        )

    # --- Samples and moments in the original space ---------------------
    stage = time.perf_counter()
    Z_all = Z_chains.reshape(-1, D)
    U_all = u_star + Z_all @ L.T
    X_all = np.asarray(transformer.inverse(U_all), dtype=np.float64)
    log_jacobian = np.asarray(
        transformer.log_abs_det_jacobian(U_all), dtype=np.float64
    )
    sample_logp = F_chains.reshape(-1) - log_jacobian
    mean = X_all.mean(axis=0)
    cov = np.cov(X_all, rowvar=False)
    _log("  posterior mean " + np.array2string(mean, precision=6))
    _log(
        "  posterior sd   "
        + np.array2string(np.sqrt(np.diag(cov)), precision=6)
    )

    comparisons = compare_with_known(
        name, problem, ln_z, ln_z_se, mean, cov, ess
    )

    _atomic_npz(
        out_dir / f"{name}.npz",
        {
            "samples": X_all,
            "sample_logp": sample_logp,
            "mean": np.asarray(mean, dtype=np.float64),
            "cov": np.asarray(cov, dtype=np.float64),
            "ln_z": np.float64(ln_z),
            "ln_z_se": np.float64(ln_z_se),
        },
    )
    elapsed["write"] = time.perf_counter() - stage
    elapsed["total"] = time.perf_counter() - started
    git_commit, git_dirty = _git_state()

    sidecar = {
        "name": name,
        "D": int(D),
        "started": started_at,
        "finished": _now(),
        "elapsed_s": {k: float(v) for k, v in elapsed.items()},
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "settings": {
            "seed": int(settings["seed"]),
            "quick": bool(settings["quick"]),
            "n_samples": int(n_chains * per_chain),
            "n_samples_requested": int(settings["n_samples"]),
            "chains": n_chains,
            "draws_per_chain": per_chain,
            "burn_frac": float(settings["burn_frac"]),
            "burn_sweeps": burn,
            "thin": thin,
            "chunk": chunk,
            "n_proposal": n_proposal,
            "bounded_transform": BOUNDED_TRANSFORM,
        },
        "diagnostics": {
            "rhat": rhat.tolist(),
            "ess": ess.tolist(),
            "rhat_max": float(np.max(rhat)),
            "ess_min": float(np.min(ess)),
            "rhat_second_half": rhat_second.tolist(),
            "ess_second_half": ess_second.tolist(),
            "n1_eff": n1_eff,
        },
        "ln_z": float(ln_z),
        "ln_z_se": float(ln_z_se),
        "bridge_iterations": int(iterations),
        "bridge_final_step": float(final_delta),
        "ln_z_is": float(ln_z_is),
        "ln_z_is_se": float(ln_z_is_se),
        "is_ess": float(is_ess),
        "ln_z_laplace": float(ln_z_laplace),
        "hessian": hessian_info,
        "map": {
            "x": x_star.tolist(),
            "u": u_star.tolist(),
            "g": float(g_star),
            "log_density": float(
                g_star
                - np.ravel(
                    transformer.log_abs_det_jacobian(u_star.reshape(1, D))
                )[0]
            ),
            "start_values": [float(-v) for v in start_values],
        },
        "proposal": proposal_summary,
        "proposal_draws": proposal_draws,
        "comparisons": comparisons,
        "target_evaluations": {
            "map_and_hessian": int(evals_map),
            "chains": [int(v) for v in chain_evals],
            "proposal": int(evals_proposal),
            "total": int(target.n_rows),
        },
    }
    _atomic_json(out_dir / f"{name}.json", sidecar)
    _log(
        f"{name}: wrote {out_dir / (name + '.npz')} and the sidecar "
        f"({elapsed['total']:.1f} s total)"
    )
    for key, value in elapsed.items():
        _log(f"    stage {key}: {value:.2f} s")
    return sidecar


# --------------------------------------------------------------------------
# Check
# --------------------------------------------------------------------------


def check_target(name, D, out_dir):
    """Reload a stored truth and verify it against itself and the target."""
    npz_path, json_path = out_dir / f"{name}.npz", out_dir / f"{name}.json"
    if not npz_path.exists() or not json_path.exists():
        _log(f"{name}: FAIL, {npz_path.name} or {json_path.name} is missing")
        return False

    try:
        with np.load(npz_path, allow_pickle=False) as f:
            samples = np.asarray(f["samples"])
            sample_logp = np.asarray(f["sample_logp"])
            mean = np.asarray(f["mean"])
            cov = np.asarray(f["cov"])
            ln_z = float(f["ln_z"])
            ln_z_se = float(f["ln_z_se"])
        with open(json_path, encoding="utf-8") as f:
            sidecar = json.load(f)
    except (OSError, ValueError, KeyError) as exc:
        _log(f"{name}: FAIL, the stored files cannot be read ({exc})")
        return False

    ok = True
    for label, array in (
        ("samples", samples),
        ("sample_logp", sample_logp),
        ("mean", mean),
        ("cov", cov),
    ):
        if array.dtype != np.float64:
            _log(f"{name}: FAIL, {label} is {array.dtype}, not float64")
            ok = False

    n, stored_D = samples.shape
    _log(
        f"{name}: {n} draws in D = {stored_D}, ln_z = {_fmt(ln_z)} "
        f"+- {_fmt(ln_z_se)}"
    )

    mean_again = samples.mean(axis=0)
    cov_again = np.cov(samples, rowvar=False)
    mean_diff = float(np.max(np.abs(mean_again - mean)))
    cov_diff = float(np.max(np.abs(cov_again - cov)))
    scale = max(float(np.max(np.abs(mean))), 1.0)
    if mean_diff > CHECK_MOMENT_RTOL * scale:
        _log(f"{name}: FAIL, stored mean differs by {mean_diff:.3g}")
        ok = False
    scale = max(float(np.max(np.abs(cov))), 1.0)
    if cov_diff > CHECK_MOMENT_RTOL * scale:
        _log(f"{name}: FAIL, stored cov differs by {cov_diff:.3g}")
        ok = False
    _log(
        f"  moments reproduce the draws: max |mean diff| {mean_diff:.3g}, "
        f"max |cov diff| {cov_diff:.3g}"
    )

    index = np.unique(
        np.linspace(0, n - 1, min(CHECK_N_DRAWS, n)).astype(np.int64)
    )
    problem = make_problem(name, int(stored_D))
    if int(problem.D) != int(D):
        _log(f"{name}: FAIL, stored D = {stored_D} but --D says {D}")
        ok = False
    recomputed = np.asarray(
        problem.log_density_vec(samples[index]), dtype=np.float64
    ).ravel()
    difference = np.abs(recomputed - sample_logp[index])
    tolerance = CHECK_LOGP_TOL + 1e-12 * np.abs(sample_logp[index])
    if np.any(difference > tolerance):
        worst = int(np.argmax(difference - tolerance))
        _log(
            f"{name}: FAIL, the target does not reproduce sample_logp at "
            f"draw {index[worst]}: {difference[worst]:.3g} > "
            f"{tolerance[worst]:.3g}"
        )
        ok = False
    _log(
        f"  target reproduces sample_logp at {index.size} draws: "
        f"max |difference| {np.max(difference):.3g}"
    )

    # The npz carries the numbers a consumer reads; the sidecar carries the
    # same two for the record. A disagreement means one of the pair was
    # rewritten without the other.
    for key, stored in (("ln_z", ln_z), ("ln_z_se", ln_z_se)):
        recorded = sidecar.get(key)
        if recorded is None or not math.isclose(
            float(recorded),
            stored,
            rel_tol=CHECK_SCALAR_RTOL,
            abs_tol=CHECK_SCALAR_RTOL,
        ):
            _log(
                f"{name}: FAIL, {key} is {_fmt(stored, '.12g')} in the npz "
                f"and {_fmt(recorded, '.12g')} in the sidecar"
            )
            ok = False

    settings = sidecar.get("settings", {})
    diagnostics = sidecar.get("diagnostics", {})
    quick = bool(settings.get("quick", False))
    _log(
        "  settings: {} chains x {}, thin {}, burn-in {} sweeps, {} "
        "proposal draws, seed {}{}".format(
            settings.get("chains", "?"),
            settings.get("draws_per_chain", "?"),
            settings.get("thin", "?"),
            settings.get("burn_sweeps", "?"),
            settings.get("n_proposal", "?"),
            settings.get("seed", "?"),
            ", quick" if quick else "",
        )
    )

    rhat_max = diagnostics.get("rhat_max")
    ess_min = diagnostics.get("ess_min")
    _log(
        f"  diagnostics: max R-hat {_fmt(rhat_max, '.4f')}, min ESS "
        f"{_fmt(ess_min, '.0f')}"
    )
    for label, spec, values in (
        ("R-hat", ".4f", diagnostics.get("rhat")),
        ("ESS  ", ".0f", diagnostics.get("ess")),
    ):
        if values:
            _log(f"    {label} " + " ".join(_fmt(v, spec) for v in values))
    if rhat_max is None:
        _log(f"{name}: FAIL, the sidecar records no R-hat")
        ok = False
    elif float(rhat_max) > RHAT_GATE and not quick:
        _log(
            f"{name}: FAIL, max R-hat {_fmt(rhat_max, '.4f')} is above "
            f"{RHAT_GATE}; the chains have not mixed"
        )
        ok = False
    elif float(rhat_max) > RHAT_GATE:
        _log(
            f"  max R-hat {_fmt(rhat_max, '.4f')} is above {RHAT_GATE}, "
            "tolerated because the run was quick"
        )

    _log(
        "  cross-checks: importance sampling "
        f"{_fmt(sidecar.get('ln_z_is'))} +- "
        f"{_fmt(sidecar.get('ln_z_is_se'))} "
        f"(ESS {_fmt(sidecar.get('is_ess'), '.0f')}), Laplace "
        f"{_fmt(sidecar.get('ln_z_laplace'))}"
    )
    for key, value in sidecar.get("comparisons", {}).items():
        _log(f"  comparison {key}: {json.dumps(value)}")
    _log(f"{name}: {'OK' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Generate the ground truth (MCMC samples, moments, ln Z) of a "
            "benchmark target of dev/scripts/benchmark_targets.py."
        )
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=list(DEFAULT_TARGETS),
        metavar="NAME",
        help="target names (default: %(default)s)",
    )
    parser.add_argument(
        "--D",
        type=int,
        default=None,
        help="dimension, when the target does not have a known one "
        f"({', '.join(f'{k} {v}' for k, v in DEFAULT_D.items())})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help=f"stored draws in total, split evenly across the chains "
        f"(default: {FULL_SETTINGS['n_samples']})",
    )
    parser.add_argument(
        "--chains", type=int, default=4, help="default: %(default)s"
    )
    parser.add_argument(
        "--burn-frac",
        type=float,
        default=0.2,
        help="burn-in sweeps as a fraction of the stored draws per chain "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--thin",
        type=int,
        default=None,
        help=f"slice-sampling sweeps per stored draw (default: "
        f"{FULL_SETTINGS['thin']})",
    )
    parser.add_argument(
        "--n-proposal",
        type=int,
        default=None,
        help=f"draws from the proposal mixture, the only new target "
        f"evaluations (default: {FULL_SETTINGS['n_proposal']})",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="default: %(default)s"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="short settings for a smoke test (2000 draws, 2000 proposal "
        "draws, thinning 2)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="reload the stored files, verify their internal consistency "
        "and print the diagnostics instead of generating",
    )
    return parser.parse_args(argv)


def resolve_dimension(name, requested):
    if requested is not None:
        return int(requested)
    if name in DEFAULT_D:
        return DEFAULT_D[name]
    raise SystemExit(f"{name} has no default dimension; pass --D")


def main(argv=None):
    args = parse_args(argv)
    os.environ.setdefault("MPLBACKEND", "Agg")
    # --quick lowers the defaults; an explicit value still wins, and 0 is
    # a value, not an omission.
    defaults = QUICK_SETTINGS if args.quick else FULL_SETTINGS

    def pick(given, fallback):
        return fallback if given is None else given

    settings = {
        "n_samples": pick(args.n_samples, defaults["n_samples"]),
        "n_proposal": pick(args.n_proposal, defaults["n_proposal"]),
        "thin": pick(args.thin, defaults["thin"]),
        "chains": args.chains,
        "burn_frac": args.burn_frac,
        "seed": args.seed,
        "quick": args.quick,
    }
    out_dir = Path(args.out)

    if args.check:
        _log(f"checking {', '.join(args.targets)} in {out_dir}")
        failures = [
            name
            for name in args.targets
            if not check_target(name, resolve_dimension(name, args.D), out_dir)
        ]
        if failures:
            _log(f"FAILED: {', '.join(failures)}")
            return 1
        _log("all checks passed")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    _log(
        f"generating {', '.join(args.targets)} into {out_dir} "
        f"(seed {settings['seed']}, "
        f"{settings['n_samples']} draws, {settings['chains']} chains, "
        f"thin {settings['thin']}, {settings['n_proposal']} proposal draws"
        + (", quick" if args.quick else "")
        + ")"
    )
    started = time.perf_counter()
    for name in args.targets:
        generate(name, resolve_dimension(name, args.D), out_dir, settings)
    _log(f"done in {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
