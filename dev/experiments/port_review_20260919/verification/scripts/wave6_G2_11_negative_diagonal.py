"""Searches for a configuration in which `GP.predict_full`'s diagonal is
negative while `GP.predict` clips the same entries at zero
(gaussian_process.py:2023 against the absence of a clamp in `predict_full`),
to quantify the minor observation that the two disagree in sign near machine
precision.

Run: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python G2_11_negative_diagonal.py
"""
import gpyreg as gpr
import numpy as np

best = (0.0, None)
for lnsn in (np.log(1e-4), np.log(1e-6), np.log(1e-7), np.log(3e-3)):
    for lnsf in (0.0, 2.0, 5.0, 10.0):
        for lnell in (np.log(0.3), np.log(3.0), np.log(30.0)):
            for N in (20, 40, 60):
                rng = np.random.default_rng(3)
                X = np.linspace(-2, 2, N).reshape(-1, 1)
                y = np.sin(3 * X)
                gp = gpr.GP(
                    D=1,
                    covariance=gpr.covariance_functions.SquaredExponential(),
                    mean=gpr.mean_functions.ZeroMean(),
                    noise=gpr.noise_functions.GaussianNoise(constant_add=True),
                )
                try:
                    gp.update(
                        X_new=X, y_new=y, hyp=np.array([[lnell, lnsf, lnsn]])
                    )
                except Exception:
                    continue
                Xt = np.concatenate(
                    [X, X + 1e-12, np.linspace(-2, 2, 3 * N).reshape(-1, 1)]
                )
                try:
                    _, cv = gp.predict_full(Xt, add_noise=False)
                    _, s2 = gp.predict(Xt, add_noise=False)
                except Exception:
                    continue
                dg = np.diag(cv[:, :, 0])
                nneg = int(np.sum(dg < 0))
                if nneg > best[0]:
                    best = (
                        nneg,
                        (
                            lnsn,
                            lnsf,
                            lnell,
                            N,
                            float(np.min(dg)),
                            float(np.min(s2)),
                            gp.posteriors[0].L_chol,
                        ),
                    )
print("most negative-diagonal entries found:", best[0])
print(
    "at (log sn, log sf, log ell, N, min diag, min predict s2, L_chol) =",
    best[1],
)
