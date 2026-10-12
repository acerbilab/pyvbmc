"""Shows that gpyreg's own `test_rank_one_update_with_heteroskedastic_noise`
(`gpyreg/testing/test_gaussian_process.py:483`, case
"provided_without_s2_new") already creates the divergence between
`Posterior.sl` and `min(sn2) * sn2_mult` that `GP.quad` ignores, so one
`quad(compute_var=True)` call added to that test would catch row G2-2.

Run: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python G2_9_rank_one_test_gap.py
"""
import gpyreg as gpr
import numpy as np

N, D = 15, 2
rng = np.random.default_rng(6)
X = rng.uniform(-3, 3, size=(N, D))
y = np.sin(X[:, 0:1]) + np.cos(X[:, 1:2])
hyp = np.array([[0.0, 0.0, 0.0, np.log(0.1), 0.2, 0.0]])
s2 = rng.uniform(0.01, 0.5, size=(N, 1))
mk = lambda: gpr.GP(
    D=D,
    covariance=gpr.covariance_functions.SquaredExponential(),
    mean=gpr.mean_functions.ConstantMean(),
    noise=gpr.noise_functions.GaussianNoise(
        constant_add=True, user_provided_add=True, scale_user_provided=True
    ),
)
gp = mk()
gp.update(X_new=X[:-1], y_new=y[:-1], s2_new=s2[:-1], hyp=hyp)
gp.update(X_new=X[-1:], y_new=y[-1:], s2_new=None)
s2r = s2.copy()
s2r[-1] = 0.0
gpr_ = mk()
gpr_.update(X_new=X, y_new=y, s2_new=s2r, hyp=hyp)
p, q = gp.posteriors[0], gpr_.posteriors[0]
print(
    "rank-one stored sl =",
    p.sl,
    " full recomputation sl =",
    q.sl,
    " ratio =",
    p.sl / q.sl,
)
sn2 = gp.noise.compute(p.hyp[3:5], gp.X, gp.y, gp.s2)
print(
    "min(sn2)*sn2_mult that quad would recompute =",
    float(np.min(sn2) * p.sn2_mult),
)
print(
    "the test's own invariant L'L*sl:",
    np.allclose(
        p.L.T @ p.L * p.sl, q.L.T @ q.L * q.sl, rtol=1e-10, atol=1e-12
    ),
)
Fa = gp.quad(np.zeros((1, D)), np.ones((1, D)), compute_var=True)
Fb = gpr_.quad(np.zeros((1, D)), np.ones((1, D)), compute_var=True)
print("quad on the rank-one GP: mean", Fa[0].ravel(), "var", Fa[1].ravel())
print("quad on the full GP    : mean", Fb[0].ravel(), "var", Fb[1].ravel())
