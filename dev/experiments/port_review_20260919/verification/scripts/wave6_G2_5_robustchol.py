"""Settles the three claims about `GP.__robust_cholesky` and
`GP.random_function` (gaussian_process.py:2613-2637, :2593):

(a) the sign-convention block at :2620-2622 indexes rows, not one entry per
    column, so `U[negidx] *= -1` flips scattered entries and `T.T @ T` no
    longer equals the matrix -- against `gplite_rnd.m:97-99`, whose linear
    index picks one entry per column and negates whole columns;
(b) when a surviving eigenvalue is negative the function returns an all-zero
    `T` of the full shape, so `random_function` returns the predictive mean
    with no warning, where MATLAB's `zeros(0,...)` makes the caller fail;
(c) how often each path is taken on realistic test grids: the direct
    Cholesky, the eigendecomposition with `p == 0` (where (a) fires), and
    the eigendecomposition with `p > 0` (where (b) fires and masks (a)).

Run: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python G2_5_robustchol.py
Needs `matlab_gplite_g2.py` next to it.
"""

import os
import sys

import numpy as np
import scipy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gpyreg as gpr
import wave6_G2_matlab_gplite_g2 as M

print("gpyreg:", gpr.__file__)
np.set_printoptions(precision=6, suppress=True, linewidth=170)


def gpyreg_robust_cholesky(sigma):
    """gaussian_process.py:2613-2637, copied verbatim."""
    try:
        T = sp.linalg.cholesky(sigma, check_finite=False)
        return T, "direct"
    except sp.linalg.LinAlgError:
        D, U = sp.linalg.eig((sigma + sigma.T) / 2)
        maxidx = np.argmax(np.abs(U), axis=0)
        negidx = U[maxidx] < 0
        U[negidx] *= -1
        D = np.real(D)
        tol = np.abs(np.spacing(np.max(D))) * D.shape[0]
        t = np.abs(D) > tol
        D = D[t]
        p = np.sum(D < 0)
        if p == 0:
            T = np.dot(np.diag(np.sqrt(D)), np.real(U[:, t]).T)
            return T, "eig p==0"
        return np.zeros(sigma.shape), "eig p>0 (zeros)"


print("\n=== A. the sign block on a rank-deficient PSD matrix ===")
rng = np.random.default_rng(0)
n, r = 6, 2
B = rng.standard_normal((n, r))
C = B @ B.T
C = (C + C.T) / 2
Tg, path = gpyreg_robust_cholesky(C.copy())
Tm, p = M.robustchol(C.copy())
print("  path taken by gpyreg:", path)
print("  max|C| =", np.max(np.abs(C)))
print("  gpyreg max|T'T - C| =", np.max(np.abs(Tg.T @ Tg - C)))
print("  MATLAB max|T'T - C| =", np.max(np.abs(Tm.T @ Tm - C)))
# The shapes of the two masks
Dv, U = sp.linalg.eig((C + C.T) / 2)
maxidx = np.argmax(np.abs(U), axis=0)
py_mask = U[maxidx] < 0
ml_mask = U[maxidx, np.arange(n)] < 0
print(
    "  gpyreg negidx shape:",
    py_mask.shape,
    " entries flagged:",
    int(np.sum(py_mask)),
    "of",
    py_mask.size,
)
print("  MATLAB negidx shape:", ml_mask.shape, "value:", ml_mask)
print("  a per-column flip leaves T'T unchanged: max|T'T - C| =", end=" ")
U2 = np.real(U).copy()
U2[:, ml_mask] *= -1
Dr = np.real(Dv)
tol = np.abs(np.spacing(np.max(Dr))) * Dr.size
t = np.abs(Dr) > tol
T2 = np.diag(np.sqrt(Dr[t])) @ U2[:, t].T
print(np.max(np.abs(T2.T @ T2 - C)))
print(
    "  sp.linalg.eig imaginary part of the eigenvectors: max =",
    np.max(np.abs(np.imag(U))),
)
# Does `sp.linalg.eig` ever return complex eigenvectors here?  Sweep
# rank-deficient symmetric matrices of several sizes and ranks.
worst_im, worst_tag = 0.0, None
for nn in (4, 5, 6, 8, 10, 20, 40):
    for rr in (1, 2, 3, max(1, nn // 2)):
        for sd in range(6):
            rr2 = np.random.default_rng(sd)
            BB = rr2.standard_normal((nn, rr))
            CC = BB @ BB.T
            CC = (CC + CC.T) / 2
            _, UU = sp.linalg.eig(CC)
            im = float(np.max(np.abs(np.imag(UU))))
            if im > worst_im:
                worst_im, worst_tag = im, (nn, rr, sd)
print(
    "  sweep of rank-deficient symmetric matrices: largest imaginary"
    f" part {worst_im:.3e} at (n, rank, seed) = {worst_tag}"
)

print("\n=== B. end to end: random_function through the eig path ===")
D = 1
gp = gpr.GP(
    D=D,
    covariance=gpr.covariance_functions.SquaredExponential(),
    mean=gpr.mean_functions.NegativeQuadratic(),
    noise=gpr.noise_functions.GaussianNoise(constant_add=True),
)
Xtr = np.linspace(-2, 2, 12).reshape(-1, 1)
ytr = np.sin(2 * Xtr)
h = np.array([np.log(0.6), np.log(1.2), np.log(0.05), 0.2, 0.0, np.log(1.5)])
gp.update(X_new=Xtr, y_new=ytr, hyp=h[None, :])
Xs = np.array([[0.0], [0.0], [1.0], [1.0], [-1.5]])
_, cov = gp.predict_full(Xs, add_noise=False)
Cs = (cov[:, :, 0] + cov[:, :, 0].T) / 2
try:
    sp.linalg.cholesky(Cs, check_finite=False)
    print("  direct Cholesky succeeded (duplicate test points)")
except sp.linalg.LinAlgError:
    print("  direct Cholesky FAILED -> the fallback is entered")
Tg, path = gpyreg_robust_cholesky(Cs.copy())
print("  path:", path, " max|T'T - C| =", np.max(np.abs(Tg.T @ Tg - Cs)))
draws = np.hstack(
    [
        gp.random_function(Xs, rng=np.random.default_rng(1000 + i))
        for i in range(8000)
    ]
)
emp = np.cov(draws)
print("  empirical covariance of 8000 draws:\n", emp)
print("  predict_full covariance:\n", Cs)
dsd = np.sqrt(np.diag(Cs))
print(
    "  predictive corr(x1,x2) =",
    Cs[0, 1] / (dsd[0] * dsd[1]),
    " empirical corr =",
    emp[0, 1] / np.sqrt(emp[0, 0] * emp[1, 1]),
)

print("\n=== C. which path a realistic grid takes ===")
header = (
    f"{'D':>2} {'Ntrain':>6} {'Ngrid':>6} {'noise':>8} "
    f"{'direct':>7} {'eig p==0':>9} {'eig p>0':>8}"
)
print(header)
for Dg in (1, 2, 3):
    for Ntr in (8, 20, 40, 80):
        for lnsn in (np.log(1e-3), np.log(0.1)):
            counts = {"direct": 0, "eig p==0": 0, "eig p>0 (zeros)": 0}
            for rep in range(12):
                rg = np.random.default_rng(500 + rep)
                Xt = rg.uniform(-2, 2, size=(Ntr, Dg))
                yt = np.sum(np.sin(Xt), 1).reshape(-1, 1)
                g = gpr.GP(
                    D=Dg,
                    covariance=gpr.covariance_functions.SquaredExponential(),
                    mean=gpr.mean_functions.NegativeQuadratic(),
                    noise=gpr.noise_functions.GaussianNoise(constant_add=True),
                )
                hh = np.zeros(Dg + 1 + 1 + 1 + 2 * Dg)
                hh[0:Dg] = np.log(0.7)
                hh[Dg] = np.log(1.2)
                hh[Dg + 1] = lnsn
                hh[Dg + 3 + Dg :] = np.log(1.5)
                g.update(X_new=Xt, y_new=yt, hyp=hh[None, :])
                if Dg == 1:
                    Xg = np.linspace(-2.5, 2.5, 100).reshape(-1, 1)
                elif Dg == 2:
                    a = np.linspace(-2.5, 2.5, 10)
                    G1, G2 = np.meshgrid(a, a)
                    Xg = np.column_stack([G1.ravel(), G2.ravel()])
                else:
                    a = np.linspace(-2.5, 2.5, 5)
                    G1, G2, G3 = np.meshgrid(a, a, a)
                    Xg = np.column_stack([G1.ravel(), G2.ravel(), G3.ravel()])
                _, cv = g.predict_full(Xg, add_noise=False)
                Cg = (cv[:, :, 0] + cv[:, :, 0].T) / 2
                _, pth = gpyreg_robust_cholesky(Cg.copy())
                counts[pth] += 1
            print(
                f"{Dg:>2} {Ntr:>6} {Xg.shape[0]:>6} {np.exp(lnsn):>8.4g} "
                f"{counts['direct']:>7} {counts['eig p==0']:>9} "
                f"{counts['eig p>0 (zeros)']:>8}   (12 repetitions)"
            )

print("\n=== D. the zeros path makes the draws deterministic ===")
Dg = 1
rg = np.random.default_rng(77)
Xt = rg.uniform(-2, 2, size=(40, 1))
yt = np.sin(2 * Xt)
g = gpr.GP(
    D=1,
    covariance=gpr.covariance_functions.SquaredExponential(),
    mean=gpr.mean_functions.NegativeQuadratic(),
    noise=gpr.noise_functions.GaussianNoise(constant_add=True),
)
hh = np.array([np.log(0.7), np.log(1.2), np.log(1e-3), 0.0, 0.0, np.log(1.5)])
g.update(X_new=Xt, y_new=yt, hyp=hh[None, :])
Xg = np.linspace(-2.5, 2.5, 100).reshape(-1, 1)
_, cv = g.predict_full(Xg, add_noise=False)
Cg = (cv[:, :, 0] + cv[:, :, 0].T) / 2
Dv = np.real(sp.linalg.eig(Cg)[0])
tol = np.abs(np.spacing(np.max(Dv))) * Dv.size
surv = Dv[np.abs(Dv) > tol]
print(
    "  tolerance =",
    tol,
    " surviving eigenvalues:",
    surv.size,
    " negative among them:",
    int(np.sum(surv < 0)),
    " smallest:",
    np.min(surv),
)
Tg, path = gpyreg_robust_cholesky(Cg.copy())
print("  path:", path, " T is all zeros:", bool(np.all(Tg == 0)))
d1 = g.random_function(Xg, rng=np.random.default_rng(1))
d2 = g.random_function(Xg, rng=np.random.default_rng(2))
mu, _ = g.predict(Xg)
print(
    "  two draws with different generators identical:",
    bool(np.array_equal(d1, d2)),
    " equal to predict's mean:",
    bool(np.allclose(d1, mu)),
)
Tm, pm = M.robustchol(Cg.copy())
print(
    "  MATLAB transcription: p =",
    pm,
    " T shape =",
    Tm.shape,
    "(MATLAB's `zeros(0,...)`, so `T'*randn(0,1) + fmu` fails on sizes)",
)
