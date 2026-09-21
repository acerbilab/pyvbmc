"""Settles the claim that `GP.predict_full(add_noise=True)` adds a broadcast
column instead of a diagonal when the noise function returns a per-point
value (gaussian_process.py:1852).

Prints the added matrix, the symmetry of the result, the agreement of the
diagonal with `predict(add_noise=True)`, and the scalar-noise case where the
line is correct. Also shows that with `s2_star=None` the user-provided
parameterizations degenerate to the scalar case.

Run: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python G2_1_predict_full_noise.py
"""

import gpyreg as gpr
import numpy as np

print("gpyreg:", gpr.__file__)
np.set_printoptions(precision=6, suppress=True, linewidth=160)

D = 2
rng = np.random.default_rng(5)
X = rng.uniform(-2, 2, size=(18, D))
y = np.sum(np.sin(X), 1).reshape(-1, 1)
s2_tr = np.abs(rng.standard_normal((18, 1))) * 0.05 + 0.01
Xs = rng.uniform(-2, 2, size=(4, D))
ys = rng.standard_normal((4, 1))
s2s = np.array([[0.5], [0.1], [0.3], [0.05]])


def make(params, s2_train):
    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(
            constant_add=params[0] == 1,
            user_provided_add=params[1] > 0,
            scale_user_provided=params[1] == 2,
            rectified_linear_output_dependent_add=params[2] == 1,
        ),
    )
    n_cov, n_mean = D + 1, 1 + 2 * D
    n_noise = gp.noise.hyperparameter_count()
    h = np.zeros(n_cov + n_noise + n_mean)
    h[0:D] = np.log(1.0)
    h[D] = np.log(1.2)
    i = n_cov
    if params[0] == 1:
        h[i] = np.log(0.2)
        i += 1
    if params[1] == 2:
        h[i] = np.log(1.2)
        i += 1
    if params[2] == 1:
        h[i] = 0.4
        h[i + 1] = np.log(0.3)
        i += 2
    h[n_cov + n_noise] = 0.5
    h[n_cov + n_noise + 1 + D :] = np.log(2.0)
    gp.update(X_new=X, y_new=y, s2_new=s2_train, hyp=h[None, :])
    return gp


for params, tag, s2_arg, y_arg in [
    ((1, 1, 0), "user_provided_add, s2_star given", s2s, None),
    ((1, 2, 0), "scale_user_provided, s2_star given", s2s, None),
    ((1, 0, 1), "output-dependent, y_star given", None, ys),
    ((1, 1, 0), "user_provided_add, s2_star=None", None, None),
    ((1, 0, 0), "constant only (scalar noise)", None, None),
]:
    gp = make(params, s2_tr if params[1] > 0 else None)
    _, cov_f = gp.predict_full(Xs, y_arg, s2_arg, add_noise=False)
    _, cov_t = gp.predict_full(Xs, y_arg, s2_arg, add_noise=True)
    added = cov_t[:, :, 0] - cov_f[:, :, 0]
    sn2_star = gp.noise.compute(
        gp.posteriors[0].hyp[D + 1 : D + 1 + gp.noise.hyperparameter_count()],
        Xs,
        y_arg,
        s2_arg,
    )
    mult = gp.posteriors[0].sn2_mult
    print(f"\n--- {tag} ---")
    print(
        "  noise function returns shape",
        np.shape(sn2_star),
        "value",
        np.ravel(sn2_star),
    )
    print("  added matrix:\n", added)
    print(
        "  symmetric:",
        bool(np.allclose(added, added.T)),
        "  equals diag(sn2_star*mult):",
        bool(np.allclose(added, np.diag(np.ravel(sn2_star) * mult))),
    )
    ev = np.linalg.eigvalsh((cov_t[:, :, 0] + cov_t[:, :, 0].T) / 2)
    print("  eigenvalues of the symmetrized returned cov:", ev)
    _, s2p = gp.predict(Xs, y_arg, s2_arg, add_noise=True)
    print(
        "  |diag(cov_full) - predict s2| =",
        np.max(np.abs(np.diag(cov_t[:, :, 0]) - s2p.ravel())),
    )
    print(
        "  latent off-diagonal magnitude (max |cov_f| off diag) =",
        np.max(np.abs(cov_f[:, :, 0] - np.diag(np.diag(cov_f[:, :, 0])))),
    )
