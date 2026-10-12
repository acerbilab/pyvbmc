"""P9-12c. Are the occasional low KS p-values of the third column of a D=3
`Trapezoidal` a real deviation or seed noise?

Settles: the distribution of the KS p-value over 40 seeds for the same
marginal drawn three ways -- as column 2 of a D=3 prior, as a standalone
one-dimensional prior, and from my MATLAB transcription -- tested for
uniformity against U(0, 1).
"""

import matlab_transcription as M
import numpy as np
from scipy.stats import kstest, uniform

import pyvbmc
from pyvbmc.priors import Trapezoidal

print("pyvbmc.__file__ =", pyvbmc.__file__)
print()

a3 = np.array([-3.0, 0.0, 10.0])
u3 = np.array([-2.0, 0.5, 11.0])
v3 = np.array([1.0, 0.75, 18.0])
b3 = np.array([2.0, 4.0, 20.0])
a, u, v, b = a3[2], u3[2], v3[2], b3[2]


def cdf(x):
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


n = 50000
nseeds = 40
p3 = Trapezoidal(a3, u3, v3, b3)
p1 = Trapezoidal(a, u, v, b)

rows = {}
rows["D=3 column 2"] = [
    kstest(p3.sample(n, rng=np.random.default_rng(1000 + s))[:, 2], cdf).pvalue
    for s in range(nseeds)
]
rows["D=1 standalone"] = [
    kstest(
        p1.sample(n, rng=np.random.default_rng(2000 + s)).ravel(), cdf
    ).pvalue
    for s in range(nseeds)
]
rows["MATLAB mtrapezrnd"] = [
    kstest(
        M.mtrapezrnd(
            np.array([[a]]),
            np.array([[u]]),
            np.array([[v]]),
            np.array([[b]]),
            n,
            np.random.default_rng(3000 + s),
        ).ravel(),
        cdf,
    ).pvalue
    for s in range(nseeds)
]

for name, ps in rows.items():
    ps = np.array(ps)
    unif = kstest(ps, uniform.cdf)
    print(
        f"{name:20s} {nseeds} seeds: below 0.05 = {int((ps<0.05).sum())}, "
        f"below 0.01 = {int((ps<0.01).sum())}, median = "
        f"{np.median(ps):.3f}; KS of the p-values against U(0,1): "
        f"D={unif.statistic:.3f} p={unif.pvalue:.3f}"
    )
print()

print("--- pooled draws, one big sample per route (n = 2,000,000) ---")
big = 2000000
for name, draws in (
    ("D=3 column 2", p3.sample(big, rng=np.random.default_rng(77))[:, 2]),
    ("D=1 standalone", p1.sample(big, rng=np.random.default_rng(78)).ravel()),
):
    r = kstest(draws, cdf)
    print(
        f"{name:20s} D={r.statistic:.6f}  p={r.pvalue:.4f}  "
        f"mean={draws.mean():.6f}"
    )
from scipy.integrate import quad

print(
    "exact mean =",
    quad(
        lambda t: t * float(p1.pdf(np.array([[t]]), keepdims=False)[0]), a, b
    )[0],
)
