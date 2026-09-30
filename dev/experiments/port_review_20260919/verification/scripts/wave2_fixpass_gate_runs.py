"""Seeded default runs for the gate of the wave-2 fix pass.

The first phase of the pass lands only commits that leave default
trajectories alone. This script records a few short seeded runs at the
default options, noiseless and noisy (the noisy ones through
`specify_target_noise` in the options dict), and saves every number that
describes the trajectory; run on the starting commit and again after a
batch of cherry-picks, the two files must agree bit for bit.

The first four runs are those of the port review's gates. The last two,
added for the gate of the release, pass a prior object with `prior=`: the
FAQ's list of `scipy.stats.uniform` marginals built from the hard bounds,
whose support falls short of the upper bound by rounding, so that the
support check passes only within its slack; and a `SplineTrapezoidal` with
its pivots at the plausible bounds. A record made before these two were
added holds the first four only, and compares with a new one recorded with
`--only` and their names.

Usage:
    python -u wave2_fixpass_gate_runs.py --out before.npz
    python -u wave2_fixpass_gate_runs.py --compare before.npz after.npz

Run with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1.
"""

import argparse
import hashlib
import logging
import sys
import time

import numpy as np


def rosenbrock(x):
    x = np.atleast_2d(x)
    return float(
        -np.sum(
            100.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2
        )
        - 0.5 * np.sum(x**2) / 9.0
    )


def two_blobs(x):
    x = np.atleast_2d(x)
    a = -0.5 * np.sum((x - 2.0) ** 2) / 0.3**2
    b = -0.5 * np.sum((x + 2.0) ** 2) / 0.6**2
    return float(np.logaddexp(a, b))


def rosenbrock_loglik(x):
    """The Rosenbrock term of `rosenbrock` alone, for a run given a prior."""
    x = np.atleast_2d(x)
    return float(
        -np.sum(
            100.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2
        )
    )


def faq_uniform_prior(LB, UB, PLB, PUB):
    """The FAQ's independent uniform priors over the hard bounds."""
    from scipy.stats import uniform

    return [
        uniform(loc=low, scale=high - low)
        for low, high in zip(np.ravel(LB), np.ravel(UB))
    ]


def spline_trapezoidal_prior(LB, UB, PLB, PUB):
    """A smoothed trapezoid over the hard bounds, flat on the plausible box."""
    from pyvbmc.priors import SplineTrapezoidal

    return SplineTrapezoidal(LB, PLB, PUB, UB)


def make_noisy(fun, sd, seed):
    """Target with Gaussian noise of known standard deviation.

    The noise comes from its own seeded generator, so the sequence of
    returned values depends only on the sequence of calls.
    """
    noise_rng = np.random.default_rng(seed)

    def noisy(x):
        return fun(x) + sd * noise_rng.standard_normal(), sd

    return noisy


def build_runs():
    inf = np.inf
    runs = {}
    # name: (target factory, D, lb, ub, plb, pub, x0, options, seed)
    runs["rosenbrock_D2_unbounded"] = (
        lambda: rosenbrock,
        2,
        -inf,
        inf,
        -3.0,
        3.0,
        0.0,
        {},
        3,
    )
    runs["two_blobs_D3_bounded"] = (
        lambda: two_blobs,
        3,
        -10.0,
        10.0,
        -3.0,
        3.0,
        1.0,
        {},
        4,
    )
    runs["rosenbrock_D2_noisy"] = (
        lambda: make_noisy(rosenbrock, 1.0, 11),
        2,
        -10.0,
        10.0,
        -3.0,
        3.0,
        0.0,
        {"specify_target_noise": True},
        5,
    )
    # A noisy run whose budget the user sets: the noisy defaults must leave
    # a user-set option alone.
    runs["two_blobs_D3_noisy_user_budget"] = (
        lambda: make_noisy(two_blobs, 0.5, 12),
        3,
        -10.0,
        10.0,
        -3.0,
        3.0,
        1.0,
        {"specify_target_noise": True, "max_fun_evals": 90},
        6,
    )
    # Runs given a prior object, whose target is the log likelihood alone.
    # Uniform marginals from bounds -2.9 and 3.3 have the support
    # [-2.9, 3.2999999999999994], inside the upper bound by one rounding.
    runs["rosenbrock_D2_uniform_prior"] = (
        lambda: rosenbrock_loglik,
        2,
        -2.9,
        3.3,
        -1.0,
        2.0,
        0.0,
        {},
        7,
        faq_uniform_prior,
    )
    runs["rosenbrock_D2_spline_trapezoidal_prior"] = (
        lambda: rosenbrock_loglik,
        2,
        -2.9,
        3.3,
        -1.0,
        2.0,
        0.0,
        {},
        8,
        spline_trapezoidal_prior,
    )
    return runs


def record(name, spec):
    from pyvbmc import VBMC

    factory, D, lb, ub, plb, pub, x0, options, seed, *prior_factory = spec
    full = lambda v: np.full((1, D), float(v))  # noqa: E731
    prior = {}
    if prior_factory:
        prior["prior"] = prior_factory[0](
            full(lb), full(ub), full(plb), full(pub)
        )
    t0 = time.time()
    vbmc = VBMC(
        factory(),
        full(x0),
        full(lb),
        full(ub),
        full(plb),
        full(pub),
        options=dict(display="off", **options),
        seed=seed,
        **prior,
    )
    if prior:
        support_lb, support_ub = vbmc.prior.support()
        inside_lb = np.max(np.ravel(support_lb) - lb)
        inside_ub = np.max(ub - np.ravel(support_ub))
        print(
            f"{name}: the prior support lies inside the hard bounds by "
            f"{inside_lb:.3g} below and {inside_ub:.3g} above",
            flush=True,
        )
    vp, results = vbmc.optimize()
    hist = vbmc.iteration_history
    fl = vbmc.function_logger
    n_iter = len(hist["iter"])
    out = {}

    def scalar_series(key):
        return np.array(
            [
                np.nan if hist[key][i] is None else float(hist[key][i])
                for i in range(n_iter)
            ]
        )

    for key in ["elbo", "elbo_sd", "r_index", "lcb_max", "sKL", "N"]:
        out[key] = scalar_series(key)
    out["func_count"] = scalar_series("func_count")
    out["warmup"] = np.array([bool(hist["warmup"][i]) for i in range(n_iter)])
    out["stable"] = np.array([bool(hist["stable"][i]) for i in range(n_iter)])
    out["K"] = np.array([hist["vp"][i].K for i in range(n_iter)])
    actions = [str(hist["logging_action"][i]) for i in range(n_iter)]
    out["n_warps"] = np.array(
        [sum(("rotoscale" in a) or ("warp" in a) for a in actions)]
    )
    out["X_orig"] = fl.X_orig[fl.X_flag]
    out["y_orig"] = fl.y_orig[fl.X_flag]
    out["X_all"] = fl.X_orig[: fl.Xn + 1]
    out["y_all"] = fl.y_orig[: fl.Xn + 1]
    out["X_flag"] = fl.X_flag[: fl.Xn + 1]
    out["vp_mu"] = vp.mu
    out["vp_sigma"] = vp.sigma
    out["vp_lambd"] = vp.lambd
    out["vp_w"] = vp.w
    out["final_elbo"] = np.array([results["elbo"], results["elbo_sd"]])
    out["best_iter"] = np.array([results["best_iter"]])
    out["rng_next"] = vbmc.rng.standard_normal(4)

    digest = hashlib.sha256()
    for key in sorted(out):
        digest.update(np.ascontiguousarray(out[key]).tobytes())
    print(
        f"{name}: {n_iter} iterations, {int(out['func_count'][-1])} "
        f"evaluations, warps {int(out['n_warps'][0])}, "
        f"elbo {results['elbo']:.6f} +- {results['elbo_sd']:.6f}, "
        f"best_iter {results['best_iter']}, "
        f"results['iterations'] {results['iterations']}, "
        f"end of warm-up at {int(np.sum(out['warmup']))}, "
        f"{time.time() - t0:.0f} s",
        flush=True,
    )
    print(f"    actions: {actions}", flush=True)
    print(f"    sha256 {digest.hexdigest()}", flush=True)
    return {f"{name}/{k}": v for k, v in out.items()}


def compare(path_a, path_b):
    a = np.load(path_a, allow_pickle=False)
    b = np.load(path_b, allow_pickle=False)
    keys = sorted(set(a.files) | set(b.files))
    differing = []
    for key in keys:
        if key not in a.files or key not in b.files:
            differing.append((key, "missing on one side"))
            continue
        x, y = a[key], b[key]
        if x.shape != y.shape or x.dtype != y.dtype:
            differing.append((key, f"shape/dtype {x.shape} {y.shape}"))
        elif x.tobytes() != y.tobytes():
            if np.issubdtype(x.dtype, np.floating):
                with np.errstate(invalid="ignore"):
                    gap = np.nanmax(np.abs(x - y))
                differing.append((key, f"max abs difference {gap:.3e}"))
            else:
                differing.append((key, "differs"))
    print(f"{len(keys)} arrays compared, {len(differing)} differ", flush=True)
    for key, what in differing:
        print(f"  {key}: {what}", flush=True)
    return 1 if differing else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out")
    parser.add_argument("--compare", nargs=2)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    if args.compare:
        sys.exit(compare(*args.compare))

    import pyvbmc

    logging.disable(logging.CRITICAL)
    print("pyvbmc:", pyvbmc.__file__, flush=True)
    everything = {}
    for name, spec in build_runs().items():
        if args.only and name not in args.only:
            continue
        everything.update(record(name, spec))
    np.savez(args.out, **everything)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
