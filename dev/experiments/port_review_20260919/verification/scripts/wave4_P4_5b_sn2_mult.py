"""P4-5b: the rank-one update on a posterior whose factorization was retried.

``__core_computation`` multiplies the noise by ``sn2_mult`` (x10 per
retry) when the Cholesky factorization fails, and ``GP.update``'s
rank-one path extends that factor using the *stored* ``sn2_mult``.  This
script looks for the mildest state that retries (``sn2_mult == 10``),
then compares the rank-one update with a full rebuild in *relative*
terms, and both against a reference solve of the same linear system, so
that a disagreement can be told apart from the conditioning of the state
itself.
"""

import copy
import warnings

import numpy as np
from wave4_P4_common import banner

banner()

import gpyreg as gpr  # noqa: E402


def make_gp(X, y, s2, params, hyp):
    gp = gpr.GP(
        D=X.shape[1],
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(*params),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.update(X_new=X, y_new=y, s2_new=s2, hyp=hyp, compute_posterior=True)
    return gp


rng = np.random.default_rng(1)
D = 2
mean_hyp = [0.3] + [0.0] * D + [np.log(1.0)] * D

best = None
for log_sf in np.arange(6.0, 20.0, 0.5):
    for n_dup in (4, 6, 8):
        Xb = rng.standard_normal((3, D))
        Xd = np.vstack([Xb] * n_dup) + 1e-12 * rng.standard_normal(
            (3 * n_dup, D)
        )
        yd = (-0.5 * np.sum(Xd**2, 1)).reshape(-1, 1)
        s2d = np.full((Xd.shape[0], 1), 1e-3)
        hyp = np.array(
            [np.log(1.0), np.log(1.0), log_sf, np.log(1e-3)] + mean_hyp
        ).reshape(1, -1)
        gp = make_gp(Xd, yd, s2d, (1, 1, 0), hyp)
        m = gp.posteriors[0].sn2_mult
        if m is not None and m > 1:
            best = (log_sf, n_dup, m, Xd, yd, s2d, hyp)
            break
    if best is not None:
        break

if best is None:
    print("no retried factorization found in the sweep")
    raise SystemExit

log_sf, n_dup, mult, Xd, yd, s2d, hyp = best
print(
    f"mildest retry found: log sf = {log_sf}, N = {Xd.shape[0]}, "
    f"sn2_mult = {mult}"
)

gp0 = make_gp(Xd, yd, s2d, (1, 1, 0), hyp)
K = gp0.covariance.compute(hyp.ravel()[:3], Xd)
sn2 = gp0.noise.compute(hyp.ravel()[3:4], Xd, yd, s2d)
A = K + mult * np.diag(np.ravel(np.broadcast_to(np.ravel(sn2), Xd.shape[0])))
print("cond(K + sn2_mult*diag(sn2)) =", np.linalg.cond(A))
print("sf2 =", np.exp(2 * hyp.ravel()[2]))

x_new = rng.standard_normal((1, D))
y_new = np.array([[-0.8]])
Xt = rng.standard_normal((5, D))

gp_r1 = copy.deepcopy(gp0)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    gp_r1.update(
        X_new=x_new,
        y_new=y_new,
        s2_new=np.array([[1e-3]]),
        compute_posterior=True,
    )
gp_full = make_gp(
    np.vstack([Xd, x_new]),
    np.vstack([yd, y_new]),
    np.vstack([s2d, [[1e-3]]]),
    (1, 1, 0),
    hyp,
)

p1, p2 = gp_r1.posteriors[0], gp_full.posteriors[0]
print("sn2_mult after: rank-one", p1.sn2_mult, " full", p2.sn2_mult)

# Reference alpha from a direct solve of the enlarged system with the
# SAME sn2_mult as each path used.
Xe = np.vstack([Xd, x_new])
ye = np.vstack([yd, y_new])
s2e = np.vstack([s2d, [[1e-3]]])
Ke = gp_full.covariance.compute(hyp.ravel()[:3], Xe)
sn2e = np.ravel(gp_full.noise.compute(hyp.ravel()[3:4], Xe, ye, s2e))
m_e = np.ravel(gp_full.mean.compute(hyp.ravel()[4:], Xe))


def ref_alpha(mult_used):
    Amat = Ke + mult_used * np.diag(sn2e)
    return np.linalg.solve(Amat, (ye.ravel() - m_e)).reshape(-1, 1)


for name, p, mult_used in (
    ("rank-one", p1, p1.sn2_mult),
    ("full    ", p2, p2.sn2_mult),
):
    ref = ref_alpha(mult_used if mult_used else 1)
    rel = np.abs(p.alpha - ref).max() / max(np.abs(ref).max(), 1e-300)
    print(
        f"  {name}: max|alpha - ref| / max|ref| = {rel:.3e}"
        f"  (max|alpha| = {np.abs(p.alpha).max():.3e})"
    )

m1, v1 = gp_r1.predict(Xt, separate_samples=True)
m2, v2 = gp_full.predict(Xt, separate_samples=True)
print(
    "  predictions: max|dmu| =",
    np.abs(m1 - m2).max(),
    " relative:",
    np.abs(m1 - m2).max() / max(np.abs(m2).max(), 1e-300),
)
print(
    "  variances:   max|ds2| =",
    np.abs(v1 - v2).max(),
    " relative:",
    np.abs(v1 - v2).max() / max(np.abs(v2).max(), 1e-300),
)
print("  s2 values (rank-one):", v1.ravel())
print("  s2 values (full)    :", v2.ravel())
