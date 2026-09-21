"""Checks the test notes of both G2 reports on `test_predict_lpd`
(`gpyreg/testing/test_gaussian_process.py:1206`) and its isotropic twin
(`test_gaussian_process_isotropic.py:1045`): the hyperparameter count, the
blocks `predict` actually reads, the zeroed `s2_star`, the vacuous `np.pi`
factor, and whether un-zeroing `s2_star` breaks the test's own formula.

It rebuilds the two GPs exactly as the tests do (nothing is imported from the
test modules and no test is run).

Run: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python G2_8_tests.py
"""

import gpyreg as gpr
import gpyreg.isotropic_covariance_functions as iso
import numpy as np
import scipy.stats

print("gpyreg:", gpr.__file__)
np.set_printoptions(precision=6, suppress=True, linewidth=170)

D = 3
row = [
    0.0,
    0.0,
    0.0,  # log ell
    1.0,  # log sf2
    np.log(np.pi),  # "log std. dev. of noise"
    -(D / 2) * np.log(2 * np.pi),  # "MVN mode"
    0.0,
    0.0,
    0.0,  # "Mode location"
    0.0,
    0.0,
    0.0,  # "log scale"
]
hyp = np.array([row, row])

for tag, cov in [
    (
        "ARD (test_gaussian_process.py:1206)",
        gpr.covariance_functions.SquaredExponential(),
    ),
    (
        "isotropic (test_gaussian_process_isotropic.py:1045)",
        iso.SquaredExponentialIsotropic(),
    ),
]:
    gp = gpr.GP(
        D=D,
        covariance=cov,
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(user_provided_add=True),
    )
    n_cov = gp.covariance.hyperparameter_count(D)
    n_noise = gp.noise.hyperparameter_count()
    n_mean = gp.mean.hyperparameter_count(D)
    print(f"\n=== {tag} ===")
    print(
        f"  cov_N = {n_cov}, noise_N = {n_noise}, mean_N = {n_mean}"
        f"  -> the GP expects {n_cov + n_noise + n_mean} hyperparameters;"
        f" the test's rows carry {hyp.shape[1]}"
    )
    gp.update(hyp=hyp.copy())
    print(
        "  update() stored the row as is:",
        bool(np.array_equal(gp.posteriors[0].hyp, hyp[0])),
    )
    print("  blocks predict reads:")
    print("    covariance :", gp.posteriors[0].hyp[0:n_cov])
    print(
        "    noise      :",
        gp.posteriors[0].hyp[n_cov : n_cov + n_noise],
        "(empty: user_provided_add alone has no hyperparameter)",
    )
    print(
        "    mean       :",
        gp.posteriors[0].hyp[n_cov + n_noise : n_cov + n_noise + n_mean],
    )
    print("    never read :", gp.posteriors[0].hyp[n_cov + n_noise + n_mean :])

    X_star = np.arange(-9, 9).reshape((-1, 3))
    rng = np.random.default_rng(0)
    offset = rng.normal(size=(6, 1))
    y_star = (
        scipy.stats.multivariate_normal.logpdf(
            X_star, mean=np.zeros((D,))
        ).reshape(-1, 1)
        + offset
    )
    s2_zero = np.zeros((6, 1))
    f_mu, f_s2, lpd = gp.predict(
        X_star, y_star, s2_star=s2_zero, return_lpd=True
    )
    sn2 = gp.noise.compute(
        gp.posteriors[0].hyp[n_cov : n_cov + n_noise], X_star, y_star, s2_zero
    )
    print(
        "  with the test's s2_star = zeros: sn2_star =",
        np.ravel(sn2),
        " (np.spacing(1) =",
        np.spacing(1.0),
        ")",
    )
    lhs = scipy.stats.norm.logpdf(
        y_star, loc=f_mu, scale=np.sqrt(np.pi * s2_zero + f_s2)
    )
    print(
        "  the test's assertion holds:",
        bool(np.allclose(lpd, lhs)),
        " max|difference| =",
        float(np.max(np.abs(lpd - lhs))),
    )
    print(
        "  f_s2 returned (add_noise=False) equals the latent variance;"
        " the lpd uses f_s2 + np.spacing(1): max|lpd - logN(y;mu,f_s2)| =",
        float(
            np.max(
                np.abs(
                    lpd - scipy.stats.norm.logpdf(y_star, f_mu, np.sqrt(f_s2))
                )
            )
        ),
    )

    # Un-zero s2_star: which formula matches?
    s2_nz = np.full((6, 1), 0.7)
    f_mu2, f_s22, lpd2 = gp.predict(
        X_star, y_star, s2_star=s2_nz, return_lpd=True
    )
    pi_form = scipy.stats.norm.logpdf(
        y_star, loc=f_mu2, scale=np.sqrt(np.pi * s2_nz + f_s22)
    )
    plain = scipy.stats.norm.logpdf(
        y_star, loc=f_mu2, scale=np.sqrt(s2_nz + f_s22)
    )
    print("  with s2_star = 0.7:")
    print(
        "    max|lpd - logN(..., pi*s2_star + f_s2)| =",
        float(np.max(np.abs(lpd2 - pi_form))),
    )
    print(
        "    max|lpd - logN(..., s2_star + f_s2)|    =",
        float(np.max(np.abs(lpd2 - plain))),
    )

    # Two identical hyperparameter rows: separate_samples is untested in
    # substance.
    mu_s, s2_s = gp.predict(X_star, separate_samples=True)
    print(
        "  the two hyperparameter rows are identical, so the per-sample"
        " columns coincide:",
        bool(np.allclose(mu_s[:, 0], mu_s[:, 1])),
    )
