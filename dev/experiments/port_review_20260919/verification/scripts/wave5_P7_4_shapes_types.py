"""P7-4: the four shape and type observations of sample / moments / pdf.

(a) the documented and actual shape and dtype of the index array I of sample;
(b) the 0-d covariance of moments(orig_flag=True, cov_flag=True) for D = 1;
(c) a negative df: accepted by pdf, refused by sample;
(d) a float N: refused by sample through kl_div and mtv, accepted by moments.
"""

import numpy as np

import pyvbmc
from pyvbmc.variational_posterior import VariationalPosterior

print("pyvbmc.__file__ =", pyvbmc.__file__)


def make_vp(D, K, seed=3):
    vp = VariationalPosterior(
        D, K, x0=np.zeros((1, D)), rng=np.random.default_rng(seed)
    )
    vp.mu = np.linspace(-1, 1, D * K).reshape(D, K)
    vp.sigma = np.full((1, K), 0.7)
    vp.lambd = np.ones((D, 1))
    vp.w = np.ones((1, K)) / K
    return vp


print("\n== (a) the index array I ==")
for K in [1, 3]:
    vp = make_vp(2, K)
    x, i = vp.sample(5)
    print(
        f"  K={K}: X.shape={x.shape} I.shape={i.shape} I.dtype={i.dtype}"
        f"  I={i}"
    )
vp = make_vp(2, 3)
x, i = vp.sample(0)
print(f"  N=0: X.shape={x.shape} I.shape={i.shape} I.dtype={i.dtype}")
print("  docstring promises: 'I is an N-by-1 array'")
print(
    "  MATLAB vbmc_rnd: K>1 -> I is N-by-1 of 1-based indices; "
    "K==1 -> ones(N,1) (only when nargout>1); N<1 -> zeros(0,1)"
)

print("\n== (b) moments(orig_flag=True, cov_flag=True) for D = 1 ==")
for D in [1, 2]:
    vp = make_vp(D, 2)
    m_o, c_o = vp.moments(2000, True, True)
    m_t, c_t = vp.moments(2000, False, True)
    print(
        f"  D={D}: orig mean.shape={m_o.shape} cov.shape={c_o.shape} "
        f"cov.ndim={np.ndim(c_o)} | transformed cov.shape={np.shape(c_t)}"
    )

print("\n== (c) negative df ==")
vp = make_vp(2, 2)
xq = np.array([[0.1, 0.2]])
print("  pdf(df=-3) =", vp.pdf(xq, orig_flag=False, df=-3.0))
print("  pdf(df=+3) =", vp.pdf(xq, orig_flag=False, df=3.0))
try:
    s, _ = vp.sample(4, orig_flag=False, df=-3.0)
    print("  sample(df=-3) ->", s)
except Exception as e:
    print(f"  sample(df=-3) raises {type(e).__name__}: {e}")
print(
    "  MATLAB: vbmc_pdf.m:87-102 takes df = abs(df) (product of univariate t);"
    " vbmc_rnd.m:87 calls gamrnd(df/2, df/2, ...) with a negative shape"
)

print("\n== (d) a float N ==")
vp = make_vp(2, 2)
vp2 = make_vp(2, 2, seed=4)
for name, call in [
    ("moments(N=1e3)", lambda: vp.moments(1e3, True, False)),
    (
        "kl_div(N=1e3, gauss_flag=True)",
        lambda: vp.kl_div(vp2=vp2, N=1e3, gauss_flag=True),
    ),
    (
        "kl_div(N=1e3, gauss_flag=False)",
        lambda: vp.kl_div(vp2=vp2, N=1e3, gauss_flag=False),
    ),
    ("mtv(N=1e3)", lambda: vp.mtv(vp2=vp2, N=1e3)),
    ("sample(1e3)", lambda: vp.sample(1e3)),
]:
    try:
        out = call()
        print(f"  {name}: ok -> {np.ravel(out)[:2]}")
    except Exception as e:
        print(f"  {name}: {type(e).__name__}: {e}")
