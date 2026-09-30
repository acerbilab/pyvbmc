"""O4 F1: a hyperparameter fixed by equal bounds, without a prior, while
another coordinate has one. Gradient of the log posterior at the fixed value,
what L-BFGS-B does with it in `fit`, and the test configuration of
`test_fitting_with_fixed_bounds`."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gpyreg as gpr
import numpy as np
import scipy as sp
from wave7_O4_common import attach, banner, fd_grad, make_gp

banner()
np.set_printoptions(precision=4, suppress=False, linewidth=120)

rng = np.random.default_rng(11)
N, D = 20, 2
X = rng.uniform(-1, 1, (N, D))
y = (-0.5 * np.sum(X**2, 1) + 0.02 * rng.standard_normal(N)).reshape(-1, 1)
FIX = np.log(1e-2)
NOISE = (
    D + 1
)  # index of noise_log_scale in [ell_1..ell_D, log sf, noise, mean...]


def build(prior_mode, fixed=True, widen=0.0):
    gp = make_gp(D, "negquad", (1, 0, 0))
    attach(gp, X, y)
    b = gp.get_recommended_bounds()
    if fixed:
        b["noise_log_scale"] = (
            np.array([FIX - widen]),
            np.array([FIX + widen]),
        )
    gp.set_bounds(b)
    pri = {k: None for k in b}
    if prior_mode == "elsewhere":
        pri["covariance_log_lengthscale"] = ("gaussian", (0.0, 1.0))
    elif prior_mode == "on_fixed":
        pri["covariance_log_lengthscale"] = ("gaussian", (0.0, 1.0))
        pri["noise_log_scale"] = ("gaussian", (FIX, 1e3))
    elif prior_mode == "none":
        pri = None
    gp.set_priors(pri)
    return gp


print("\n== 1. Gradient at the fixed value ==")
gp = build("elsewhere")
lb, ub = gp.lower_bounds, gp.upper_bounds
# A sensible point: the plausible starting point of each block.
hyp = np.concatenate(
    [
        gp.covariance.get_bounds_info(gp.X, gp.y)["x0"],
        gp.noise.get_bounds_info(gp.X, gp.y)["x0"],
        gp.mean.get_bounds_info(gp.X, gp.y)["x0"],
    ]
)
hyp[NOISE] = FIX
print(
    "bounds of noise_log_scale:",
    lb[NOISE],
    ub[NOISE],
    "equal:",
    lb[NOISE] == ub[NOISE],
)
print("no_prior flag:", gp.no_prior)
lp, dlp = gp.log_posterior(hyp, compute_grad=True)
lz, dlz = gp.log_likelihood(hyp, compute_grad=True)
print("log_posterior value:", lp, " (finite at the fixed value)")
print("grad log_posterior :", dlp)
print("grad log_likelihood:", dlz)
# Transcription of gplite_hypprior.m: dlp = zeros, Gaussian rows only.
ref_prior_grad = np.zeros_like(hyp)
ref_prior_grad[:D] = -(hyp[:D] - 0.0) / 1.0**2
print("MATLAB-style expected grad (dlZ + dlp):", dlz + ref_prior_grad)
print("NaN coordinates of grad log_posterior:", np.where(np.isnan(dlp))[0])
# FD of log_posterior along the free coordinates only (fixed coordinate kept)
free = np.ones(hyp.size, bool)
free[NOISE] = False
fd = fd_grad(lambda h: gp.log_posterior(h), hyp)
print("FD of log_posterior (free coords):", fd[free])
print(
    "max |analytic - FD| on free coords:", np.max(np.abs(dlp[free] - fd[free]))
)

print("\n== 2. Same point, three other set-ups ==")
for mode, fixed in (("none", True), ("on_fixed", True), ("elsewhere", False)):
    g = build(mode, fixed=fixed)
    h = hyp.copy()
    _, d = g.log_posterior(h, compute_grad=True)
    _, d2 = g._GP__gp_obj_fun(h, True, False)
    print(
        f"prior={mode:9s} fixed={fixed!s:5s} no_prior={g.no_prior!s:5s}: NaN in "
        f"log_posterior grad = {np.any(np.isnan(d))}, NaN in fit objective grad = {np.any(np.isnan(d2))}"
    )

print("\n== 3. fit(n_samples=0) ==")
results = {}
for label, mode, widen in (
    ("fixed, prior elsewhere only", "elsewhere", 0.0),
    ("fixed, prior also on fixed", "on_fixed", 0.0),
    ("pair widened by 1e-9", "elsewhere", 1e-9),
    ("fixed, no priors at all", "none", 0.0),
):
    g = build(mode, widen=widen)
    hyp_fit, res, _ = g.fit(
        options={"n_samples": 0, "init_N": 128, "opts_N": 2},
        rng=np.random.default_rng(0),
    )
    h = hyp_fit[0]
    _, d = g.log_posterior(h, compute_grad=True)
    fr = np.ones(h.size, bool)
    fr[NOISE] = False
    print(
        f"{label:30s}: nit={res.nit:3d} status={res.status} "
        f"msg={res.message!r}\n{'':32s}log_lik={g.log_likelihood(h):9.4f} "
        f"log_post={g.log_posterior(h):9.4f} |grad free|_inf={np.max(np.abs(d[fr])):.3e} "
        f"noise={h[NOISE]:.6f}"
    )
    results[label] = (g, h)

g, h0 = results["fixed, prior elsewhere only"]
fr = np.ones(h0.size, bool)
fr[NOISE] = False


def obj_free(hf):
    h = h0.copy()
    h[fr] = hf
    v, dv = g.log_posterior(h, compute_grad=True)
    return -v, -dv[fr]


r = sp.optimize.minimize(
    obj_free,
    h0[fr],
    jac=True,
    method="L-BFGS-B",
    bounds=list(zip(g.lower_bounds[fr], g.upper_bounds[fr])),
    tol=1e-5,
)
h1 = h0.copy()
h1[fr] = r.x
print(
    f"hand L-BFGS-B on the free coordinates from that result: nit={r.nit}, "
    f"log_post {g.log_posterior(h0):.4f} -> {g.log_posterior(h1):.4f}"
)

print(
    "\n== 4. L-BFGS-B directly, NaN in one gradient entry of a fixed variable =="
)


def quad(x):
    f = np.sum((x - 1.0) ** 2)
    gr = 2 * (x - 1.0)
    gr[2] = np.nan
    return f, gr


r = sp.optimize.minimize(
    quad,
    np.zeros(3),
    jac=True,
    method="L-BFGS-B",
    bounds=[(-5, 5), (-5, 5), (0.0, 0.0)],
)
print(f"toy: nit={r.nit} status={r.status} msg={r.message!r} x={r.x}")
r = sp.optimize.minimize(
    lambda x: (np.sum((x - 1.0) ** 2), np.r_[2 * (x[:2] - 1.0), 0.0]),
    np.zeros(3),
    jac=True,
    method="L-BFGS-B",
    bounds=[(-5, 5), (-5, 5), (0.0, 0.0)],
)
print(f"toy, finite gradient: nit={r.nit} status={r.status} x={r.x}")

print("\n== 5. Configuration of test_fitting_with_fixed_bounds ==")
Xt = np.reshape(np.linspace(-10, 10, 20), (-1, 1))
yt = 1 + np.sin(Xt)
for opts in ({"n_samples": 0}, {}):
    for lo, hi in ((0.5, 0.5), (0.5 - 1e-9, 0.5 + 1e-9)):
        gpt = gpr.GP(
            D=1,
            covariance=gpr.covariance_functions.Matern(3),
            mean=gpr.mean_functions.ConstantMean(),
            noise=gpr.noise_functions.GaussianNoise(constant_add=True),
        )
        gpt.set_priors(
            {
                "covariance_log_outputscale": None,
                "covariance_log_lengthscale": None,
                "noise_log_scale": ("gaussian", (np.log(1e-3), 1.0)),
                "mean_const": None,
            }
        )
        gpt.set_bounds(
            {
                "covariance_log_outputscale": (-np.inf, np.inf),
                "covariance_log_lengthscale": (-np.inf, np.inf),
                "noise_log_scale": (-np.inf, np.inf),
                "mean_const": (lo, hi),
            }
        )
        hy, res, sres = gpt.fit(
            X=Xt, y=yt, options=opts, rng=np.random.default_rng(3)
        )
        _, d = gpt.log_posterior(res.x, compute_grad=True)
        print(
            f"opts={opts!s:18s} mean_const in [{lo!r}, {hi!r}]: optimizer nit={res.nit} "
            f"msg={res.message!r} log_post(opt)={-res.fun:.4f} grad(opt)={d}"
        )
