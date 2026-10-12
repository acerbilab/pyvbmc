"""P7-12: sample() against vbmc_rnd.m, outside the remainder of the balanced draw.

Checks the pieces the comparison reviewer read line by line: the N < 1
return, the unbalanced draw, the shuffle, the Gaussian and heavy-tailed
formulas, the gamrnd / rng.gamma parameterization, the K == 1 branch and
the inverse transform.  MATLAB's side is established by reading
vbmc_rnd.m; what is executed here is the Python and the distributional
identity of the two rules.
"""

import numpy as np
from scipy import stats

import pyvbmc
from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.variational_posterior import VariationalPosterior

print("pyvbmc.__file__ =", pyvbmc.__file__)


def make_vp(D, K, seed=13, pt=None):
    vp = VariationalPosterior(
        D,
        K,
        x0=np.zeros((1, D)),
        parameter_transformer=pt,
        rng=np.random.default_rng(seed),
    )
    vp.mu = np.linspace(-2, 2, D * K).reshape(D, K)
    vp.sigma = np.full((1, K), 0.6)
    vp.lambd = np.full((D, 1), 1.0)
    vp.w = np.ones((1, K)) / K
    return vp


print("\n== N < 1 ==")
vp = make_vp(3, 2)
for N in [0, -5]:
    x, i = vp.sample(N)
    print(
        f"  N={N}: X.shape={x.shape} I.shape={i.shape}"
        "   (MATLAB vbmc_rnd.m:42-44: zeros(0,D), zeros(0,1))"
    )

print("\n== the t scaling: rng.gamma(df/2, df/2) and gamrnd(df/2, df/2) ==")
df = 5.0
g = np.random.default_rng(0).gamma(df / 2, df / 2, 400000)
t = df / 2 / np.sqrt(g)
print("  both take (shape, scale); G = Gamma(df/2, scale=df/2) = df*chi2_df/4")
print("  so t = df/2/sqrt(G) = sqrt(df/chi2_df), the multivariate-t scale")
print(
    "  E[t^2] sample:", np.mean(t**2), " theory df/(df-2) =", df / (df - 2)
)
ks = stats.kstest(1.0 / t**2, lambda q: stats.chi2.cdf(q * df, df))
print(
    "  KS of 1/t^2 against chi2_df/df: statistic",
    ks.statistic,
    "p =",
    ks.pvalue,
)

print("\n== the drawn law, K > 1 and K == 1, normal and heavy-tailed ==")
for K in [1, 3]:
    for dfv in [np.inf, 5.0]:
        vp = make_vp(1, K, seed=21)
        vp.mu = np.zeros((1, K))
        vp.sigma = np.full((1, K), 1.0)
        x, _ = vp.sample(200000, orig_flag=False, df=dfv)
        if np.isinf(dfv):
            ks = stats.kstest(x.ravel(), "norm")
        else:
            ks = stats.kstest(x.ravel(), lambda q: stats.t.cdf(q, dfv))
        print(
            f"  K={K} df={dfv}: KS statistic {ks.statistic:.5f} "
            f"p={ks.pvalue:.3f}"
        )

print("\n== the shuffle: randperm(numel(I), N) against shuffle + [:N] ==")
vp = make_vp(2, 3, seed=5)
vp.w = np.array([[0.5, 0.3, 0.2]])
counts = np.zeros(3)
reps = 4000
for r in range(reps):
    vp._rng = np.random.default_rng(r)
    _, i = vp.sample(10, orig_flag=False, balance_flag=True)
    counts += np.bincount(i.astype(int), minlength=3)
print(
    "  balanced N=10, empirical component frequencies:",
    counts / counts.sum(),
    " weights:",
    vp.w.ravel(),
)
print("  (the remainder rule itself is the orchestrator's finding)")

print("\n== the unbalanced draw with unnormalized weights ==")
vp = make_vp(2, 3, seed=5)
vp.w = np.array([[0.25, 0.15, 0.1]])  # sums to 0.5
print("  sum(w) =", vp.w.sum())
try:
    x, i = vp.sample(10, orig_flag=False, balance_flag=False)
    print(
        "  unbalanced: ok, counts =", np.bincount(i.astype(int), minlength=3)
    )
except Exception as e:
    print(f"  unbalanced: {type(e).__name__}: {e}")
try:
    x, i = vp.sample(10, orig_flag=False, balance_flag=True)
    print("  balanced: ok, counts =", np.bincount(i.astype(int), minlength=3))
except Exception as e:
    print(f"  balanced: {type(e).__name__}: {e}")
print(
    "  MATLAB catrnd scales u by cdf(end), so an unnormalized w draws from"
    " w/sum(w) without complaint (vbmc_rnd.m:118-123)"
)

print("\n== the inverse transform ==")
pt = ParameterTransformer(2, np.array([[0.0, 0.0]]), np.array([[1.0, 1.0]]))
vp = make_vp(2, 2, seed=8, pt=pt)
x_o, _ = vp.sample(2000, orig_flag=True)
x_t, _ = vp.sample(2000, orig_flag=False)
print(
    "  orig-space draws inside the box:", bool(np.all((x_o > 0) & (x_o < 1)))
)
print("  transformed-space draws unconstrained, range:", x_t.min(), x_t.max())
