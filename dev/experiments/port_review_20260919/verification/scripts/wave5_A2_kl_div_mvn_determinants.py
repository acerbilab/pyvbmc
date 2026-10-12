"""W5, `kl_div_mvn` on raw determinants.

`pyvbmc/stats/kl_div_mvn.py:34-39` computes `det(sigma1)` and `det(sigma2)`,
returns `(inf, inf)` when either is zero and otherwise takes
`log(detq2 / detq1)`, as `shared/mvnkl.m` does without the guard. The
determinant of a `D`-dimensional covariance of scale `s**2` is `s**(2*D)`,
which leaves the range of a double for small or large `s` at moderate `D`
although the matrix is well conditioned. This script settles, for two
Gaussians with covariances `s**2 * I` and `4 * s**2 * I` and the same mean
(both divergences are then independent of `s`):

1. for which `(D, s)` the function returns a non-finite value or loses
   digits, against the closed form and against the same formula with
   `slogdet`;
2. what `vp.kl_div(gauss_flag=True)`, the route of `sKL` at the default
   options, returns for two posteriors of that kind;
3. what MATLAB's arithmetic gives in the same cases (`log(0/0)`, `log(x/0)`,
   `log(inf/inf)`), by the IEEE rules that both languages follow.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

import pyvbmc
from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.stats import kl_div_mvn
from pyvbmc.variational_posterior import VariationalPosterior

print("pyvbmc:", pyvbmc.__file__, flush=True)


def exact(D):
    # KL(N(0, S) || N(0, 4S)) and KL(N(0, 4S) || N(0, S)), any S
    kl1 = 0.5 * (D / 4 - D + D * np.log(4.0))
    kl2 = 0.5 * (4 * D - D - D * np.log(4.0))
    return np.array([kl1, kl2])


def with_slogdet(mu1, s1, mu2, s2):
    D = mu1.size
    dmu = (mu2 - mu1).reshape(-1, 1)
    lndet = np.linalg.slogdet(s2)[1] - np.linalg.slogdet(s1)[1]
    kl1 = 0.5 * (
        np.trace(np.linalg.solve(s2, s1))
        + (dmu.T @ np.linalg.solve(s2, dmu)).item()
        - D
        + lndet
    )
    kl2 = 0.5 * (
        np.trace(np.linalg.solve(s1, s2))
        + (dmu.T @ np.linalg.solve(s1, dmu)).item()
        - D
        - lndet
    )
    return np.array([kl1, kl2])


print("\n--- 1. kl_div_mvn(0, s^2 I, 0, 4 s^2 I)", flush=True)
print(
    "   D      s    det(S1)        kl_div_mvn                 slogdet form"
    "               exact"
)
for D in (2, 6, 10, 15, 20):
    for s in (1e-12, 1e-9, 1e-8, 1e-6, 1.0, 1e6, 1e8, 1e9):
        S1 = s**2 * np.eye(D)
        S2 = 4 * S1
        mu = np.zeros(D)
        with np.errstate(all="ignore"):
            got = kl_div_mvn(mu, S1, mu, S2)
            det1 = np.linalg.det(S1)
        ref = with_slogdet(mu, S1, mu, S2)
        ok = np.allclose(got, exact(D), rtol=1e-9, atol=0)
        if not ok:
            print(
                f"  {D:2d}  {s:6.0e}  {det1:10.3e}  {np.array2string(got, precision=6):26s}"
                f" {np.array2string(ref, precision=6):26s} "
                f"{np.array2string(exact(D), precision=6)}"
            )
print(
    "   (rows printed: those where kl_div_mvn is off the closed form by more"
    " than 1e-9 relative)"
)

print(
    "\n--- 2. vp.kl_div(gauss_flag=True) on two one-component posteriors",
    flush=True,
)
for D, s in ((20, 1e-8), (20, 1e-9), (10, 1e-9), (20, 1e9)):
    vps = []
    for scale in (1.0, 2.0):
        vp = VariationalPosterior(D, 1, np.zeros((1, D)))
        vp.parameter_transformer = ParameterTransformer(D)
        vp.mu = np.zeros((D, 1))
        vp.sigma = np.array([[s * scale]])
        vp.lambd = np.ones((D, 1))
        vp.w = np.ones((1, 1))
        vp.rng = np.random.default_rng(0)
        vps.append(vp)
    with np.errstate(all="ignore"):
        kls = vps[0].kl_div(vp2=vps[1], N=int(1e5), gauss_flag=True)
    print(
        f"   D = {D}, SD = {s:g}: kl_div = {kls}, exact {exact(D)} up to "
        f"Monte Carlo error; sKL = {max(0, 0.5 * np.sum(kls))}"
    )

print("\n--- 3. the same cases under MATLAB's formula (no guard)", flush=True)
with np.errstate(all="ignore"):
    zero, tiny, inf = np.float64(0.0), np.float64(1e-320), np.float64(np.inf)
    print("   log(0 / 0)     =", np.log(zero / zero))
    print("   log(tiny / 0)  =", np.log(tiny / zero))
    print("   log(inf / inf) =", np.log(inf / inf))
