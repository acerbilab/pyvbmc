import gpyreg as gpr
import numpy as np
from scipy.linalg import solve_triangular

from pyvbmc.variational_posterior import VariationalPosterior


def check_variance_regularization(acq_fcn, ns_gp, M):
    D = 1
    X = np.array([[-1.0], [0.0], [1.0]])
    y = np.array([[-0.5], [0.0], [-0.5]])
    hyp = np.array(
        [
            [
                0.0,
                0.0,
                -2.0,
                -0.5 * np.log(2 * np.pi),
                0.0,
                0.0,
            ],
            [
                0.15,
                0.1,
                -1.8,
                -0.5 * np.log(2 * np.pi),
                0.0,
                0.0,
            ],
        ][:ns_gp]
    )
    gp = gpr.GP(
        D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    gp.update(X_new=X, y_new=y, hyp=hyp)

    Xa = np.array([[-0.75], [0.25], [1.5]])
    __, f_s2a = gp.predict(Xa, separate_samples=True)
    K_Xa_X = np.empty((ns_gp, Xa.shape[0], X.shape[0]))
    C_tmp = np.empty((ns_gp, X.shape[0], Xa.shape[0]))
    cov_N = gp.covariance.hyperparameter_count(D)
    for s, posterior in enumerate(gp.posteriors):
        K_Xa_X[s] = gp.covariance.compute(posterior.hyp[:cov_N], Xa, X)
        if posterior.L_chol:
            sn2_eff = 1 / posterior.sW[0] ** 2
            C_tmp[s] = (
                solve_triangular(
                    posterior.L,
                    solve_triangular(
                        posterior.L,
                        K_Xa_X[s].T,
                        trans=True,
                        check_finite=False,
                    ),
                    check_finite=False,
                )
                / sn2_eff
            )
        else:
            C_tmp[s] = posterior.L @ K_Xa_X[s].T

    ln_ell = np.column_stack(
        [posterior.hyp[:D] for posterior in gp.posteriors]
    )
    gp_length_scale = np.exp(np.mean(ln_ell, axis=1))
    gp.temporary_data["X_rescaled"] = X / gp_length_scale
    noise_N = gp.noise.hyperparameter_count()
    sn2_new = np.column_stack(
        [
            np.broadcast_to(
                gp.noise.compute(
                    posterior.hyp[cov_N : cov_N + noise_N], X, y
                ).reshape(-1),
                (X.shape[0],),
            )
            for posterior in gp.posteriors
        ]
    )
    gp.temporary_data["sn2_new"] = np.mean(sn2_new, axis=1)
    optim_state = {
        "integer_vars": None,
        "lb_eps_orig": -np.inf,
        "ub_eps_orig": np.inf,
        "gp_length_scale": gp_length_scale,
        "active_importance_sampling": {
            "X": Xa,
            "f_s2": f_s2a,
            "ln_weights": np.zeros((ns_gp, Xa.shape[0])),
            "K_Xa_X": K_Xa_X,
            "C_tmp": C_tmp,
        },
    }
    vp = VariationalPosterior(D, 1)
    X_eval = np.array([[-1.0], [0.35], [2.5]])[:M]

    optim_state["variance_regularized_acq_fcn"] = False
    base = acq_fcn(X_eval, gp, vp, None, optim_state)
    f_mu, f_s2 = gp.predict(X_eval, separate_samples=True)
    var_tot = np.mean(f_s2, axis=1)
    if ns_gp > 1:
        var_tot += np.var(f_mu, axis=1, ddof=1)

    if M == 1:
        tol_var = 2 * var_tot[0]
    else:
        assert np.ptp(var_tot) > 0
        tol_var = 0.5 * (np.min(var_tot) + np.max(var_tot))
    mask = var_tot < tol_var
    if M > 1:
        assert np.any(mask) and np.any(~mask)

    expected = base.copy()
    expected[mask] += tol_var / var_tot[mask] - 1
    optim_state["variance_regularized_acq_fcn"] = True
    optim_state["tol_gp_var"] = tol_var
    batch = acq_fcn(X_eval, gp, vp, None, optim_state)
    pointwise = np.array(
        [acq_fcn(x, gp, vp, None, optim_state)[0] for x in X_eval]
    )

    assert batch.shape == (M,)
    assert np.allclose(batch, expected)
    assert np.allclose(batch, pointwise)
