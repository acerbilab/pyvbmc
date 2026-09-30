"""W5, the guards of `kl_div(gauss_flag=False)` and the density they guard.

`variational_posterior.py:1492-1493` and `:1499-1500` write
`q[q == 0 | np.isinf(q)] = ...` where `vbmc_kldiv.m:75-76` and `:82-83` have
`q(q == 0 | ~isfinite(q)) = ...`. In MATLAB `==` binds tighter than `|`; in
Python `|` binds tighter than `==`. This script settles:

1. which entries the Python mask selects, against MATLAB's;
2. what the zero guard does to two posteriors that do not overlap (the
   divergence is capped near `-log(realmin)`);
3. whether the original-space density can be non-finite, on which the two
   reviewers disagree: a one-component posterior of SD `s` in the inference
   space of a probit transform on the unit box, `s` from 1 to 60, 1e5 draws
   each, counting infinite densities, the rows on which the log form is
   finite and the linear form `y / exp(logJ)` is not (`:917-928`), and what
   `kl_div(gauss_flag=False)` then returns.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys

import numpy as np

import pyvbmc
from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.variational_posterior import VariationalPosterior

print("pyvbmc:", pyvbmc.__file__, flush=True)

print("\n--- 1. the mask", flush=True)
q = np.array([0.0, 1.0, np.inf, np.nan, 0.5])
print("   q                          :", q)
print("   q == 0 | np.isinf(q)       :", q == 0 | np.isinf(q))
print("   (q == 0) | np.isinf(q)     :", (q == 0) | np.isinf(q))
print(
    "   (q == 0) | ~np.isfinite(q) :", (q == 0) | ~np.isfinite(q), " (MATLAB)"
)


def one_component(pt, mean, sd, seed):
    vp = VariationalPosterior(1, 1, np.zeros((1, 1)))
    vp.parameter_transformer = pt
    vp.mu = np.array([[mean]])
    vp.sigma = np.array([[sd]])
    vp.lambd = np.ones((1, 1))
    vp.w = np.ones((1, 1))
    vp.rng = np.random.default_rng(seed)
    return vp


print(
    "\n--- 2. two unit Gaussians 80 apart, unbounded (exact KL 3200 each way)",
    flush=True,
)
pt = ParameterTransformer(1)
a, b = one_component(pt, 0.0, 1.0, 1), one_component(pt, 80.0, 1.0, 2)
with np.errstate(all="ignore"):
    print(
        "   kl_div(gauss_flag=False):",
        a.kl_div(vp2=b, N=20000, gauss_flag=False),
    )
print("   -log(realmin) =", -np.log(sys.float_info.min))

print(
    "\n--- 3. probit transform on [0, 1], posterior N(0, s^2) in the inference"
    " space",
    flush=True,
)
pt = ParameterTransformer(
    1, np.array([[0.0]]), np.array([[1.0]]), transform_type="probit"
)
for s in (1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 30.0, 60.0):
    vp1 = one_component(pt, 0.0, s, 3)
    vp2 = one_component(pt, 0.5, s, 4)
    x, _ = vp1.sample(int(1e5), orig_flag=True)
    with np.errstate(all="ignore"):
        lin = vp1.pdf(x, orig_flag=True)
        log = vp1.pdf(x, orig_flag=True, log_flag=True)
        kls = vp1.kl_div(vp2=vp2, N=int(1e5), gauss_flag=False)
    n_inf = int(np.sum(np.isinf(lin)))
    n_lost = int(np.sum(np.isinf(lin) & (log < np.log(sys.float_info.max))))
    on_bound = int(np.sum((x <= 0) | (x >= 1)))
    print(
        f"   s = {s:4.0f}: infinite densities {n_inf:6d} of 1e5 (of which "
        f"{n_lost} have a representable value), draws on a bound {on_bound:6d}, "
        f"max log density {np.max(log[np.isfinite(log)]):8.1f}; kl_div = {kls}"
    )

print(
    "\n--- 4. y / exp(logJ) against exp(log y - logJ), D = 2, probit on the"
    " unit box, posterior N(0, 4.5^2 I), at the image of u = (-27.3, -27.3)",
    flush=True,
)
pt2 = ParameterTransformer(
    2, np.zeros((1, 2)), np.ones((1, 2)), transform_type="probit"
)
vp = VariationalPosterior(2, 1, np.zeros((1, 2)))
vp.parameter_transformer = pt2
vp.mu = np.zeros((2, 1))
vp.sigma = np.array([[4.5]])
vp.lambd = np.ones((2, 1))
vp.w = np.ones((1, 1))
u = np.array([[-27.3, -27.3]])
x = pt2.inverse(u)
with np.errstate(all="ignore"):
    print(
        "   x =",
        x,
        " strictly inside the box:",
        bool(np.all((x > 0) & (x < 1))),
    )
    print("   log|J| at u          :", pt2.log_abs_det_jacobian(pt2(x)))
    print(
        "   pdf, log_flag=True   :", vp.pdf(x, orig_flag=True, log_flag=True)
    )
    print("   pdf, log_flag=False  :", vp.pdf(x, orig_flag=True))
    print(
        "   exp of the log value :",
        np.exp(vp.pdf(x, orig_flag=True, log_flag=True)),
    )
