"""Export one PyVBMC run as the trace that the 3D animation plays back.

The animation page (``docsrc/source/_static/vbmc3d/index.html``) renders a
two-dimensional run as a landscape: the GP surrogate of the log joint as a
wireframe, the evaluations as probes standing on it, the variational
mixture on the floor and the acquisition function while points are being
chosen. It computes nothing itself; every surface it draws is a grid
written here from a real run.

The run is observed, not altered. ``active_sample`` offers a
selection-policy hook that is called before and after each point is
chosen; the recorder uses it to evaluate the GP and the acquisition
function on the display grid with the state the search is about to use
(the GP after the rank-one updates of the points already acquired in the
iteration), and returns no selection, so the run takes the path it takes
without an observer. The recorder draws nothing from the run's generator.
The GP and the posterior at the end of each iteration are read from the
iteration history afterwards.

The surrogate that the page draws is the GP averaged over the
hyperparameter samples that a display grid can resolve, not over all of
them (``drawable_samples`` says why and records how many were left out of
each state). The acquisition function, the posterior and every number on
the page are the run's own.

Everything is exported in the caller's coordinates: grid points go
through the transformer current at that moment, the GP mean has the
log-Jacobian removed, and each mixture component becomes a center and a
2 x 2 matrix ``M`` whose columns are the images of the component's
principal axes (exact for an unbounded problem, whose transform is
affine), so its covariance is ``M M^T``.

Usage, from the repository root::

    python -u dev/scripts/export_animation_trace.py --seed 8

writes ``trace.js`` (``window.VBMC_TRACE = {...}``) next to the page. The
fields are described in ``build_trace``.
"""

import argparse
import base64
import importlib
import json
import sys
import time
import zlib
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

import pyvbmc
from pyvbmc import VBMC

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "docsrc" / "source" / "_static" / "vbmc3d" / "trace.js"

# Display window in the caller's coordinates (square, so that the floor of
# the scene is not stretched).
X_RANGE = (-7.0, 7.0)
Y_RANGE = (-5.0, 9.0)

# Quantization ranges. Heights are stored as log1p(top - z) so that the
# resolution is finest near the top of the landscape, where the eye is.
Z_SPAN = 1.0e5
ACQ_SPAN = 40.0
SD_HALF = 1.0


# --------------------------------------------------------------------------
# Target
# --------------------------------------------------------------------------

BANANA_SIG1 = 2.0
BANANA_B = 0.5
LOBE_W = 0.22
LOBE_MU = np.array([0.0, 4.6])
LOBE_SD = np.array([0.75, 0.6])


def log_density_vec(X):
    """Normalized log density of the target at the rows of ``X``.

    A twisted Gaussian (a banana opening upward; the twist has unit
    Jacobian, so the density is normalized) mixed with a Gaussian lobe
    between its arms. The log normalizing constant is 0.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    z1 = X[:, 0]
    z2 = X[:, 1] - BANANA_B * (X[:, 0] ** 2 - BANANA_SIG1**2)
    log_banana = (
        -0.5 * (z1 / BANANA_SIG1) ** 2
        - 0.5 * z2**2
        - np.log(BANANA_SIG1)
        - np.log(2 * np.pi)
    )
    r = (X - LOBE_MU) / LOBE_SD
    log_lobe = (
        -0.5 * np.sum(r**2, axis=1)
        - np.sum(np.log(LOBE_SD))
        - np.log(2 * np.pi)
    )
    return logsumexp(
        np.stack([log_banana, log_lobe]),
        axis=0,
        b=np.array([[1 - LOBE_W], [LOBE_W]]),
    )


def log_density(x):
    return float(log_density_vec(x)[0])


def true_moments():
    """Exact mean and covariance of the target."""
    means = np.array([[0.0, 0.0], LOBE_MU])
    covs = np.array(
        [
            np.diag(
                [BANANA_SIG1**2, 1 + 2 * BANANA_B**2 * BANANA_SIG1**4]
            ),
            np.diag(LOBE_SD**2),
        ]
    )
    w = np.array([1 - LOBE_W, LOBE_W])
    mean = w @ means
    second = sum(wi * (c + np.outer(m, m)) for wi, m, c in zip(w, means, covs))
    return mean, second - np.outer(mean, mean)


def gskl(vp):
    """Gaussianized symmetrized KL divergence of a posterior from the
    target, as the benchmark computes it."""
    from pyvbmc.stats import kl_div_mvn

    mean, cov = vp.moments(orig_flag=True, cov_flag=True)
    true_mean, true_cov = true_moments()
    return float(0.5 * np.sum(kl_div_mvn(mean, cov, true_mean, true_cov)))


# --------------------------------------------------------------------------
# Grid evaluation in the caller's coordinates
# --------------------------------------------------------------------------


def make_grid(n):
    gx = np.linspace(*X_RANGE, n)
    gy = np.linspace(*Y_RANGE, n)
    XX, YY = np.meshgrid(gx, gy)  # row index is y, column index is x
    return np.column_stack([XX.ravel(), YY.ravel()])


def gp_on_grid(gp, transformer, grid, keep=None):
    """GP mean and SD of the log joint at original-space grid points,
    combined over the hyperparameter samples ``keep`` (all of them by
    default) as ``GP.predict`` combines them: the mean of the means, and
    the mean of the variances plus the sample variance of the means."""
    U = transformer(grid)
    mu, s2 = gp.predict(U, add_noise=False, separate_samples=True)
    if keep is not None:
        mu, s2 = mu[:, keep], s2[:, keep]
    between = np.var(mu, axis=1, ddof=1) if mu.shape[1] > 1 else 0.0
    mean = mu.mean(axis=1) - np.ravel(transformer.log_abs_det_jacobian(U))
    return mean, np.sqrt(np.maximum(s2.mean(axis=1) + between, 0.0))


# A hyperparameter sample is drawn if its smallest length scale is at least
# this many cells of the display grid.
RESOLVABLE_CELLS = 2.0


def sample_length_scales(gp, transformer):
    """Smallest length scale of each hyperparameter sample, in the caller's
    coordinates."""
    J = _jacobian(transformer, gp.D)
    smallest = []
    for posterior in gp.posteriors:
        ell = np.exp(posterior.hyp[: gp.D])
        P = J.T @ np.diag(1.0 / ell**2) @ J
        smallest.append(1.0 / np.sqrt(np.linalg.eigvalsh(P).max()))
    return np.array(smallest)


def drawable_samples(gp, transformer, cell):
    """Mask of the hyperparameter samples that enter the displayed GP.

    The GP that VBMC uses is the average over its hyperparameter samples.
    Every sample's mean passes through the observations, but a sample
    with a length scale far below their spacing does so as a needle and
    sits at its mean function everywhere else, so the averaged mean has a
    needle at each observation too: the observations stand above (or
    below) a surface that does not bend towards them. After the first
    fit, on ten points, the hyperparameter posterior is broad and a few
    samples of this kind are common (length scales of a thousandth of the
    plausible box in one coordinate); they disappear as evaluations
    accumulate. A grid of cell size ``cell`` cannot draw a needle, and drawn
    wider it reads as a surrogate that misses its data, which is the
    opposite of what it does. So the displayed average leaves out the
    samples whose smallest length scale is below ``RESOLVABLE_CELLS``
    cells, keeping at least the better-resolved half of them.
    """
    ells = sample_length_scales(gp, transformer)
    keep = ells >= RESOLVABLE_CELLS * cell
    if 2 * keep.sum() < keep.size:
        keep = np.zeros(keep.size, dtype=bool)
        keep[np.argsort(ells)[-((keep.size + 1) // 2) :]] = True
    return keep


def gp_kernel(gp, transformer, keep=None):
    """Output scale and input metric of the GP's kernel in the caller's
    coordinates, ``[sf, p11, p12, p22]``.

    Two points ``d`` apart are at squared scaled distance ``rho2 = d^T P d``,
    with ``P = J^T diag(1 / ell^2) J`` and ``J`` the Jacobian of the
    transform (constant for an unbounded problem); ``sf`` and ``ell`` are
    geometric means over the hyperparameter samples ``keep``. The page uses
    them to pin the surrogate at its data. Conditioning on one noiseless
    observation alone leaves a predictive SD of ``sf sqrt(1 - exp(-rho2))``
    at scaled distance ``rho`` from it. For one hyperparameter sample that
    is an upper bound on the SD, up to the observation noise, since more
    observations only lower it, and it is zero at the observation. The
    displayed SD also carries the spread between hyperparameter samples,
    which this does not bound but which vanishes at an observation too. The
    page caps the grid's SD by it around each observation, which restores
    the zero that a display grid coarser than the length scale cannot hold.
    """
    D = gp.D
    hyp = np.array([posterior.hyp for posterior in gp.posteriors])
    if keep is not None:
        hyp = hyp[keep]
    ell = np.exp(hyp[:, :D].mean(axis=0))
    sf = float(np.exp(hyp[:, D].mean()))
    J = _jacobian(transformer, D)
    P = J.T @ np.diag(1.0 / ell**2) @ J
    return [float(f"{v:.6g}") for v in (sf, P[0, 0], P[0, 1], P[1, 1])]


def _jacobian(transformer, D):
    """Jacobian of the transform at the center of the display window."""
    center = np.array([[np.mean(X_RANGE), np.mean(Y_RANGE)]])
    J = np.empty((D, D))
    for d in range(D):
        step = np.zeros((1, D))
        step[0, d] = 0.5
        J[:, d] = (transformer(center + step) - transformer(center - step))[0]
    return J


def min_length_scale(gp, transformer):
    """Smallest length scale of any hyperparameter sample, in the caller's
    coordinates (see ``drawable_samples``)."""
    return float(sample_length_scales(gp, transformer).min())


def run_min_length_scale(vbmc):
    """``min_length_scale`` over every recorded iteration of a run."""
    history = vbmc.iteration_history
    return min(
        min_length_scale(
            vbmc.get_gp(t), history["vp"][t].parameter_transformer
        )
        for t in range(len(history["iter"]))
    )


def vp_components(vp):
    """Centers ``(K, 2)``, axis matrices ``(K, 2, 2)`` and weights ``(K,)``
    of the mixture components in the caller's coordinates."""
    inv = vp.parameter_transformer.inverse
    mu = vp.mu.T  # (K, D)
    scale = vp.sigma.T * vp.lambd.T  # (K, D)
    centers = inv(mu)
    M = np.empty((vp.K, 2, 2))
    for d in range(2):
        step = np.zeros_like(mu)
        step[:, d] = scale[:, d]
        # Central difference over one standard deviation: exact for an
        # affine transform.
        M[:, :, d] = 0.5 * (inv(mu + step) - inv(mu - step))
    return centers, M, np.ravel(vp.w)


# --------------------------------------------------------------------------
# Observer of the active-sampling loop
# --------------------------------------------------------------------------


def check_recordable(vbmc):
    """Refuse a run whose acquisition function the recorder cannot record.

    The recorder evaluates the acquisition function on the display grid
    when ``active_sample`` announces a search, which is before it draws
    the importance samples of the noisy acquisitions or computes the
    variance of the log joint that a custom acquisition may ask for. The
    grid would then come from the previous search's quantities. The
    acquisition of a noiseless target needs neither.
    """
    acquisitions = vbmc.options["search_acq_fcn"]
    acquisition = acquisitions[0]
    if (
        len(acquisitions) != 1
        or isinstance(acquisition, str)
        or acquisition.acq_info.get("importance_sampling")
        or acquisition.acq_info.get("compute_var_log_joint")
    ):
        raise ValueError(
            "the recorder handles one acquisition function that needs no "
            "importance samples and no variance of the log joint"
        )


class TraceRecorder:
    """Read-only selection policy: records, never selects.

    It evaluates the run's acquisition function itself (see
    ``check_recordable`` for which ones that is valid for).
    """

    def __init__(self, grid, detail_iters, cell):
        self.grid = grid
        self.cell = cell
        self.detail_iters = detail_iters
        self.substeps = {}  # iteration -> list of dicts
        self._current = None

    def __enter__(self):
        self._module = importlib.import_module("pyvbmc.vbmc.active_sample")
        if self._module._selection_policy_callback is not None:
            raise RuntimeError("another selection policy is installed")
        self._module._selection_policy_callback = self
        return self

    def __exit__(self, *exc):
        self._module._selection_policy_callback = None
        return False

    def start(self, *, gp, vp, function_logger, optim_state, options):
        iteration = int(optim_state["iter"])
        record = {"iter": iteration}
        transformer = function_logger.parameter_transformer
        self._transformer = transformer
        if iteration <= self.detail_iters:
            acq_fcn = options["search_acq_fcn"][0]
            U = transformer(self.grid)
            acq = acq_fcn(U, gp, vp, function_logger, optim_state)
            record["acq"] = -np.ravel(acq)  # log scale, larger is better
            keep = drawable_samples(gp, transformer, self.cell)
            record["gp_mean"], record["gp_sd"] = gp_on_grid(
                gp, transformer, self.grid, keep
            )
            record["gp_k"] = gp_kernel(gp, transformer, keep)
            record["gp_shown"] = [int(keep.sum()), int(keep.size)]
        self._current = record

    def select(self, **kwargs):
        return None

    def finish(self, *, selected, cache_index, repeat):
        record, self._current = self._current, None
        x_tran = np.array(selected, dtype=float).reshape(1, -1)
        record["x_orig"] = self._transformer.inverse(x_tran)[0]
        self.substeps.setdefault(record["iter"], []).append(record)


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------


def _round(a, digits=5):
    return np.round(np.asarray(a, dtype=float), digits).tolist()


class Encoder:
    """Quantizes the grids of a trace and packs them into one block.

    A grid is referred to by its index among the grids of its width: 16
    bits for heights, which the page compares with the observed values, 8
    bits for SDs and acquisition values, which set colors and the amplitude
    of the motion. The block is every 16-bit grid, then every 8-bit grid,
    each stored as its second difference (along x, then along y, modulo
    2^16 or 2^8; the fields are smooth, so the differences are small),
    compressed with zlib and
    written in base64. The page inflates it with ``DecompressionStream``.
    """

    def __init__(self, z_top, n):
        self.z_top = float(z_top)
        self.n = n
        self.wide = []
        self.narrow = []

    def _add(self, store, unit, levels, dtype):
        q = np.round(np.clip(unit, 0.0, 1.0) * levels).astype(np.int64)
        q = q.reshape(self.n, self.n)
        d = np.diff(np.diff(q, axis=1, prepend=0), axis=0, prepend=0)
        store.append((d % (levels + 1)).astype(dtype).tobytes())
        return len(store) - 1

    def z(self, values):
        depth = np.clip(self.z_top - np.asarray(values), 0.0, Z_SPAN)
        unit = np.log1p(depth) / np.log1p(Z_SPAN)
        return self._add(self.wide, unit, 65535, "<u2")

    def sd(self, values):
        values = np.asarray(values)
        return self._add(self.narrow, values / (values + SD_HALF), 255, "u1")

    def acq(self, values):
        values = np.asarray(values, dtype=float)
        finite = np.isfinite(values)
        top = values[finite].max() if finite.any() else 0.0
        rel = np.where(finite, values - top, -ACQ_SPAN)
        depth = np.clip(-rel, 0.0, ACQ_SPAN) / ACQ_SPAN
        return self._add(self.narrow, 1.0 - np.sqrt(depth), 255, "u1")

    def block(self):
        raw = b"".join(self.wide) + b"".join(self.narrow)
        return base64.b64encode(zlib.compress(raw, 9)).decode("ascii")


def mixture_logpdf(X, centers, M, w):
    """Log density of the Gaussian mixture with covariances ``M M^T``."""
    out = np.full((len(w), X.shape[0]), -np.inf)
    for k in range(len(w)):
        cov = M[k] @ M[k].T
        d = X - centers[k]
        maha = np.sum(d @ np.linalg.inv(cov) * d, axis=1)
        logdet = np.linalg.slogdet(cov)[1]
        out[k] = np.log(w[k]) - 0.5 * maha - 0.5 * logdet - np.log(2 * np.pi)
    return logsumexp(out, axis=0)


def vp_entry(vp, grid):
    """Components of a posterior as the page receives them, after checking
    that they give its density: the page evaluates the posterior on its
    grid from them, which is exact for an unbounded problem."""
    centers, M, w = vp_components(vp)
    entry = {
        "w": _round(w, 7),
        "c": _round(centers, 7),
        "M": _round(M.reshape(-1, 4), 7),
    }
    exact = np.ravel(vp.pdf(grid, orig_flag=True, log_flag=True))
    rebuilt = mixture_logpdf(
        grid,
        np.array(entry["c"]),
        np.array(entry["M"]).reshape(-1, 2, 2),
        np.array(entry["w"]),
    )
    core = exact > exact.max() - 40.0
    if np.max(np.abs(exact - rebuilt)[core]) > 1e-3:
        raise RuntimeError("the mixture components do not give vp.pdf")
    return entry


# --------------------------------------------------------------------------
# The run and the trace
# --------------------------------------------------------------------------


def run(seed, max_fun_evals):
    plb = np.array([[-5.0, -3.5]])
    pub = np.array([[5.0, 7.5]])
    lb = np.full((1, 2), -np.inf)
    ub = np.full((1, 2), np.inf)
    x0 = np.array([[-3.0, 5.5]])
    options = {"display": "off", "max_fun_evals": max_fun_evals}
    return VBMC(log_density, x0, lb, ub, plb, pub, options=options, seed=seed)


def build_trace(vbmc, recorder, results, grid, n_grid, seed):
    """Assemble the trace.

    ``meta``: grid size and ranges, quantization constants, the number of
    grids of each width, and the run's provenance, which the page does not
    read (``seed``, ``gskl``, ``min_ell``, ``target``, ``pyvbmc``).
    ``points``: every evaluation in order, ``[x, y, log density]``.
    ``iterations[t]``: counters (``n_evals``, ``K``, ``elbo``, ``elbo_sd``,
    and for the record ``warmup`` and ``r_index``), ``actions`` (what the
    run logged for the iteration: ``rotoscale``, ``trim data``, ``end
    warm-up`` and the like; the page takes captions from them, since an
    iteration's shape does not say what happened in it), ``active``
    (indices of the evaluations in the GP training set), the grids of the GP refitted at the end of the
    iteration (``gp_mean``, ``gp_sd``), its kernel scales (``gp_k``, see
    ``gp_kernel``), how many of its hyperparameter samples the grids
    average over (``gp_shown``, ``[shown, all]``, see
    ``drawable_samples``), the components of the posterior (``vp``: weights
    ``w``, centers ``c`` and axis matrices ``M``, row-major; the page
    evaluates the posterior on its grid from them) and ``sub``, one entry
    per acquired point with its index into ``points`` and, for the first
    iterations, the grids of the acquisition function and of the GP the
    search used.
    ``final``: the returned posterior (after the final boost).
    ``truth``: the target's log density on the grid.
    ``grids``: the block that holds every grid (``Encoder``); a grid field
    is an index into it. Heights are ``log1p(z_top - z) / log1p(z_span)``
    on 16 bits, SDs ``s / (s + sd_half)`` and acquisitions ``1 - sqrt((max
    a - a) / acq_span)``, clipped at 0, on 8 bits; row index y, column
    index x.
    """
    history = vbmc.iteration_history
    n_iter = len(history["iter"])
    cell = (X_RANGE[1] - X_RANGE[0]) / (n_grid - 1)
    logger = vbmc.function_logger
    # Every evaluation, including the ones that warm-up trimming removed
    # from the training set (their `X_flag` is False).
    X_all = logger.X_orig[: logger.Xn + 1]
    y_all = np.ravel(logger.y_orig[: logger.Xn + 1])

    def point_index(x_orig):
        d = np.sum((X_all - x_orig) ** 2, axis=1)
        j = int(np.argmin(d))
        if d[j] > 1e-12:
            raise RuntimeError("an acquired point is not in the logger")
        return j

    floats = []  # every height grid, to fix the shared reference
    iterations = []
    for t in range(n_iter):
        vp = history["vp"][t]
        transformer = vp.parameter_transformer
        gp = vbmc.get_gp(t)
        keep = drawable_samples(gp, transformer, cell)
        gp_mean, gp_sd = gp_on_grid(gp, transformer, grid, keep)
        elbo = float(history["elbo"][t])
        X_train = transformer.inverse(gp.X)
        active = sorted(point_index(x) for x in X_train)

        sub = []
        for record in recorder.substeps.get(t, []):
            entry = {"point": point_index(record["x_orig"])}
            if "acq" in record:
                entry["_acq"] = record["acq"]
                entry["_gp_mean"] = record["gp_mean"]
                entry["_gp_sd"] = record["gp_sd"]
                entry["gp_k"] = record["gp_k"]
                entry["gp_shown"] = record["gp_shown"]
                floats.append(record["gp_mean"])
            sub.append(entry)

        floats.append(gp_mean)
        iterations.append(
            {
                "n_evals": int(history["func_count"][t]),
                "K": int(vp.K),
                "elbo": round(elbo, 4),
                "elbo_sd": round(float(history["elbo_sd"][t]), 4),
                "warmup": bool(history["warmup"][t]),
                "r_index": _finite(history["r_index"][t]),
                "actions": [str(a) for a in history["logging_action"][t]],
                "active": active,
                "gp_k": gp_kernel(gp, transformer, keep),
                "gp_shown": [int(keep.sum()), int(keep.size)],
                "vp": vp_entry(vp, grid),
                "sub": sub,
                "_gp_mean": gp_mean,
                "_gp_sd": gp_sd,
            }
        )

    vp = vbmc.vp
    final_elbo = float(results["elbo"])
    truth = log_density_vec(grid)
    floats.append(truth)

    finite_top = max(float(np.max(f[np.isfinite(f)])) for f in floats)
    enc = Encoder(finite_top + 0.5, n_grid)

    for it in iterations:
        it["gp_mean"] = enc.z(it.pop("_gp_mean"))
        it["gp_sd"] = enc.sd(it.pop("_gp_sd"))
        for entry in it["sub"]:
            if "_acq" in entry:
                entry["acq"] = enc.acq(entry.pop("_acq"))
                entry["gp_mean"] = enc.z(entry.pop("_gp_mean"))
                entry["gp_sd"] = enc.sd(entry.pop("_gp_sd"))

    truth_index = enc.z(truth)
    return {
        "meta": {
            "n": n_grid,
            "grids": {"u16": len(enc.wide), "u8": len(enc.narrow)},
            "x_range": list(X_RANGE),
            "y_range": list(Y_RANGE),
            "z_top": enc.z_top,
            "z_span": Z_SPAN,
            "sd_half": SD_HALF,
            "acq_span": ACQ_SPAN,
            "ln_z_true": 0.0,
            "target": "twisted Gaussian with a lobe",
            "seed": seed,
            "gskl": round(gskl(vbmc.vp), 5),
            "min_ell": round(run_min_length_scale(vbmc), 4),
            "pyvbmc": version("pyvbmc"),
            "n_init": int(iterations[0]["n_evals"]),
            "plausible": [[-5.0, -3.5], [5.0, 7.5]],
        },
        "truth": truth_index,
        "points": _round(np.column_stack([X_all, y_all]), 4),
        "iterations": iterations,
        "final": {
            "K": int(vp.K),
            "elbo": round(final_elbo, 4),
            "elbo_sd": round(float(results["elbo_sd"]), 4),
            "vp": vp_entry(vp, grid),
        },
        "grids": enc.block(),
    }


def _finite(value):
    value = float(value)
    return round(value, 4) if np.isfinite(value) else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--max-fun-evals", type=int, default=200)
    parser.add_argument(
        "--detail-iters",
        type=int,
        default=8,
        help="iterations whose per-point acquisition and GP grids are kept",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--sweep",
        metavar="A:B",
        help="run seeds A to B - 1 (or a comma-separated list) and print "
        "each run's ELBO, gsKL and smallest hyperparameter-sample length "
        "scale instead of writing a trace (the target's log evidence is "
        "0; see min_length_scale)",
    )
    args = parser.parse_args(argv)

    print(f"pyvbmc from {pyvbmc.__file__}", flush=True)
    if args.sweep:
        if ":" in args.sweep:
            first, last = (int(v) for v in args.sweep.split(":"))
            seeds = range(first, last)
        else:
            seeds = [int(v) for v in args.sweep.split(",")]
        print(
            "seed  iters  evals     elbo  elbo_sd     gskl  min_ell",
            flush=True,
        )
        for seed in seeds:
            vbmc = run(seed, args.max_fun_evals)
            vp, results = vbmc.optimize()
            print(
                f"{seed:4d}  {results['iterations']:5d}  "
                f"{results['func_count']:5d}  {results['elbo']:7.3f}  "
                f"{results['elbo_sd']:7.3f}  {gskl(vp):7.4f}  "
                f"{run_min_length_scale(vbmc):7.3f}",
                flush=True,
            )
        return 0
    grid = make_grid(args.grid)
    vbmc = run(args.seed, args.max_fun_evals)
    check_recordable(vbmc)
    t0 = time.time()
    cell = (X_RANGE[1] - X_RANGE[0]) / (args.grid - 1)
    with TraceRecorder(grid, args.detail_iters, cell) as recorder:
        vp, results = vbmc.optimize()
    print(
        f"run finished in {time.time() - t0:.0f} s: "
        f"{results['iterations']} iterations, "
        f"{results['func_count']} evaluations, "
        f"elbo {results['elbo']:.3f} +- {results['elbo_sd']:.3f}, "
        f"K {vp.K}, {results['message']}",
        flush=True,
    )

    trace = build_trace(vbmc, recorder, results, grid, args.grid, args.seed)
    for t, it in enumerate(trace["iterations"]):
        shown, total = it["gp_shown"]
        if shown < total:
            print(
                f"iteration {t}: the displayed GP averages {shown} of "
                f"{total} hyperparameter samples",
                flush=True,
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(trace, separators=(",", ":"))
    args.out.write_text(f"window.VBMC_TRACE={payload};\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(payload) / 1e6:.2f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
