"""P9-12b. A seed sweep behind the two KS p-values of P9_12 that fell below
0.05, so that a single-seed fluctuation is not read as a sampler defect.

Settles: whether the third column of a D=3 `Trapezoidal` and the
single-dimension `UniformBox`/`SmoothBox` draws of my MATLAB transcription
deviate systematically from the closed-form CDF, over ten seeds each, and
the uniformity of the resulting p-values.
"""

import matlab_transcription as M
import numpy as np
from scipy.stats import kstest

import pyvbmc
from pyvbmc.priors import SmoothBox, Trapezoidal, UniformBox

print("pyvbmc.__file__ =", pyvbmc.__file__)
print()

SQ2PI = np.sqrt(2 * np.pi)


def cdf_unif(x, a, b):
    return np.clip((np.asarray(x, float) - a) / (b - a), 0.0, 1.0)


def cdf_trapez(x, a, u, v, b):
    x = np.asarray(x, float)
    h = 2.0 / (b - a + v - u)
    F = np.zeros_like(x)
    m = (x >= a) & (x < u)
    F[m] = h * (x[m] - a) ** 2 / (2 * (u - a))
    m = (x >= u) & (x < v)
    F[m] = h * (u - a) / 2 + h * (x[m] - u)
    m = (x >= v) & (x < b)
    F[m] = 1 - h * (b - x[m]) ** 2 / (2 * (b - v))
    F[x >= b] = 1.0
    return F


def cdf_smooth(x, a, b, s):
    from scipy.stats import norm as spnorm

    x = np.asarray(x, float)
    C = 1.0 / ((b - a) + SQ2PI * s)
    F = np.empty_like(x)
    m = x < a
    F[m] = C * s * SQ2PI * spnorm.cdf((x[m] - a) / s)
    m = (x >= a) & (x <= b)
    F[m] = C * s * SQ2PI * 0.5 + C * (x[m] - a)
    m = x > b
    F[m] = 1 - C * s * SQ2PI * (1 - spnorm.cdf((x[m] - b) / s))
    return F


a3 = np.array([-3.0, 0.0, 10.0])
u3 = np.array([-2.0, 0.5, 11.0])
v3 = np.array([1.0, 0.75, 18.0])
b3 = np.array([2.0, 4.0, 20.0])

print("--- D=3 Trapezoidal, per-column KS, ten seeds ---")
p = Trapezoidal(a3, u3, v3, b3)
for seed in range(10):
    draws = p.sample(100000, rng=np.random.default_rng(100 + seed))
    out = []
    for d in range(3):
        r = kstest(
            draws[:, d],
            lambda t, d=d: cdf_trapez(t, a3[d], u3[d], v3[d], b3[d]),
        )
        out.append(f"d{d} p={r.pvalue:.3f}")
    print(f"  seed {seed}: " + "  ".join(out))
print()

print("--- the same third column alone, n = 400000, five seeds ---")
p1 = Trapezoidal(a3[2], u3[2], v3[2], b3[2])
for seed in range(5):
    draws = p1.sample(400000, rng=np.random.default_rng(200 + seed)).ravel()
    r = kstest(draws, lambda t: cdf_trapez(t, a3[2], u3[2], v3[2], b3[2]))
    print(
        f"  seed {seed}: D={r.statistic:.5f} p={r.pvalue:.3f}  "
        f"mean={draws.mean():.5f}"
    )
from scipy.integrate import quad

a, u, v, b = a3[2], u3[2], v3[2], b3[2]
exact_mean = quad(
    lambda t: t * float(p1.pdf(np.array([[t]]), keepdims=False)[0]), a, b
)[0]
print(f"  exact mean by quadrature = {exact_mean:.5f}")
print()

print("--- my MATLAB transcription, ten seeds each ---")
n = 200000
for name, mrnd, F, args in (
    (
        "munifboxrnd(-3,2)",
        lambda r: M.munifboxrnd(np.array([[-3.0]]), np.array([[2.0]]), n, r),
        cdf_unif,
        (-3.0, 2.0),
    ),
    (
        "msmoothboxrnd(0,1,1)",
        lambda r: M.msmoothboxrnd(
            np.array([[0.0]]), np.array([[1.0]]), np.array([[1.0]]), n, r
        ),
        cdf_smooth,
        (0.0, 1.0, 1.0),
    ),
):
    ps = []
    for seed in range(10):
        draws = mrnd(np.random.default_rng(300 + seed)).ravel()
        ps.append(kstest(draws, lambda t: F(t, *args)).pvalue)
    print(
        f"  {name:24s} p-values = "
        + " ".join(f"{q:.3f}" for q in ps)
        + f"   (min {min(ps):.3f}, median {np.median(ps):.3f})"
    )
