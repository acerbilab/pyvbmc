"""O1 first question, variance branch: varG, J_sjk and var_ss of
_gp_log_joint against an independent quadrature of gpyreg's latent posterior
covariance (predict_full, add_noise=False) on a shared tensor grid, at
separated and wide components, K = 3, the three mean functions, plus a
heteroskedastic GP and the same GP after a rank-one update with a point of
lower noise than every training point.
"""

import sys
from pathlib import Path

import gpyreg as gpr
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from wave7_O1_common import MEANS, banner, build_gp

from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.vbmc.variational_optimization import _gp_log_joint

banner()


def make_vp(D, K, rng):
    vp = VariationalPosterior(D, K, rng=0)
    vp.mu = np.linspace(-1.5, 1.5, K)[None, :] * np.ones((D, 1)) + 0.2 * (
        rng.standard_normal((D, K))
    )
    vp.sigma = np.array([[0.25, 0.9, 0.5]])[:, :K]
    vp.lambd = np.array([[1.0], [0.7]])[:D]
    w = np.array([0.5, 0.3, 0.2])[:K]
    vp.w = (w / w.sum()).reshape(1, -1)
    vp.eta = np.log(vp.w)
    return vp


def grid_weights(vp, n1d):
    """Shared tensor grid and, per component, density x cell volume."""
    D, K = vp.D, vp.K
    s = vp.sigma.ravel()[None, :] * vp.lambd.ravel()[:, None]  # (D, K)
    lo = np.min(vp.mu - 8 * s, axis=1)
    hi = np.max(vp.mu + 8 * s, axis=1)
    axes = [np.linspace(lo[d], hi[d], n1d) for d in range(D)]
    mesh = np.meshgrid(*axes, indexing="ij")
    X = np.column_stack([m.ravel() for m in mesh])
    cell = np.prod([a[1] - a[0] for a in axes])
    Q = np.zeros((X.shape[0], K))
    for k in range(K):
        z = (X - vp.mu[:, k]) / s[:, k]
        Q[:, k] = np.exp(-0.5 * np.sum(z**2, axis=1)) / np.prod(
            np.sqrt(2 * np.pi) * s[:, k]
        )
    return X, Q * cell


def check(label, gp, vp, n1d):
    Ns = len(gp.posteriors)
    out = _gp_log_joint(vp, gp, False, False, True, True, True)
    G_s, J = out[0], out[6]  # avg_flag off: per-sample G; J (Ns, K, K)
    X, Q = grid_weights(vp, n1d)
    _, C = gp.predict_full(X, add_noise=False)  # (M, M, Ns)
    worst = 0.0
    for s in range(Ns):
        J_ref = Q.T @ C[:, :, s] @ Q
        rel = np.max(np.abs(J[s] - J_ref)) / np.max(np.abs(J_ref))
        worst = max(worst, rel)
    # averaged outputs rebuilt from per-sample ones
    G, _, varG, _, var_ss = _gp_log_joint(vp, gp, False, True, True, True)
    w = vp.w.ravel()
    vs = np.einsum("sjk,j,k->s", J, w, w)
    if Ns > 1:
        G_rb = G_s.mean()
        varG_rb = vs.mean() + np.var(G_s, ddof=1)
        vss_rb = np.var(G_s, ddof=1) + np.std(vs, ddof=1)
    else:
        G_rb, varG_rb, vss_rb = G_s[0], vs[0], 0.0
    print(
        f"{label:34s} J_sjk vs grid quad: {worst:.1e} | "
        f"G {abs(G-G_rb):.1e} varG {abs(varG-varG_rb)/varG_rb:.1e} "
        f"var_ss {abs(var_ss-vss_rb):.1e} (varG={varG:.3e})",
        flush=True,
    )


rng = np.random.default_rng(3)
for mean in ("zero", "const", "negquad"):
    for D, n1d in ((1, 1601), (2, 45)):
        gp = build_gp(D, N=15, Ns=2, seed=40 + D, mean=mean, ln_sn=-2.0)
        check(f"{mean} D={D} K=3 Ns=2", gp, make_vp(D, 3, rng), n1d)

# heteroskedastic user noise, then a rank-one update with a lower noise
D = 1
r = np.random.default_rng(9)
gp = gpr.GP(
    D=D,
    covariance=gpr.covariance_functions.SquaredExponential(),
    mean=MEANS["negquad"](),
    noise=gpr.noise_functions.GaussianNoise(
        constant_add=True, user_provided_add=True
    ),
)
X = r.uniform(-2.5, 2.5, size=(15, 1))
y = -0.5 * (X - 0.3) ** 2 / 1.5
s2 = 0.05 * (0.5 + r.random((15, 1)))
hyp = np.array(
    [
        [np.log(1.1), np.log(2.0), -3.0, 0.1, 0.3, np.log(1.3)],
        [np.log(0.9), np.log(1.7), -3.0, -0.1, 0.2, np.log(1.5)],
    ]
)
gp.update(X_new=X, y_new=y, s2_new=s2, hyp=hyp)
vp = make_vp(D, 3, rng)
check("negquad D=1 hetero Ns=2", gp, vp, 1601)
sl_before = [p.sl for p in gp.posteriors]
gp.update(
    X_new=np.array([[0.4]]),
    y_new=np.array([[-0.001]]),
    s2_new=np.array([[1e-4]]),
)
print(
    "  after rank-one update: stored sl",
    [p.sl for p in gp.posteriors],
    "(before",
    sl_before,
    "); 1/sW[0]^2 =",
    [1 / p.sW[0, 0] ** 2 for p in gp.posteriors],
    "; min(sn2)*mult now ~",
    1e-4 + np.exp(2 * -3.0),
)
check("negquad D=1 hetero, rank-one upd.", gp, vp, 1601)
