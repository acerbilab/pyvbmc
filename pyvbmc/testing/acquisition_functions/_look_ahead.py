"""References for the integrated look-ahead acquisitions (VIQR, IMIQR).

Both acquisitions integrate a transform of the GP's predictive standard
deviation *after* a hypothetical observation at the candidate,

    s_new^2(theta; x) = s^2(theta) - C(theta, x)^2 / (s^2(x) + sn2(x)),

over a Gaussian. The helpers here compute that integral by Gauss-Hermite
quadrature (a converged reference for the acquisition's Monte Carlo
estimate) and recompute it from a given set of weighted samples with the
GP's own ``predict_full`` (an exact check of the acquisition's formula,
independent of how the samples were drawn).
"""

import numpy as np


def look_ahead_sd(gp, thetas, x, sn2_new, chunk=2000):
    """Predictive sd at ``thetas`` after observing at ``x`` with noise
    variance ``sn2_new``, from full covariances of chunks of ``thetas`` with
    ``x`` appended (one hyperparameter sample)."""
    assert len(gp.posteriors) == 1
    x = np.atleast_2d(x)
    s2_new = np.empty(thetas.shape[0])
    for lo in range(0, thetas.shape[0], chunk):
        block = thetas[lo : lo + chunk]
        __, cov = gp.predict_full(np.vstack([block, x]))
        cov = cov[:, :, 0]
        c = cov[:-1, -1]
        s2_new[lo : lo + block.shape[0]] = np.diag(cov)[:-1] - c**2 / (
            cov[-1, -1] + sn2_new
        )
    return np.sqrt(np.maximum(s2_new, 0.0))


def gauss_hermite_reference(gp, vp, X_eval, sn2_new, transform, n_nodes=30):
    """``E_vp[transform(s_new(theta; x))]`` for every row of ``X_eval`` by
    tensor Gauss-Hermite quadrature under the single-component Gaussian
    ``vp`` (``n_nodes`` nodes per axis; 30 is converged to 1e-5 on the
    scenarios of the tests, 20 to 1e-4)."""
    assert vp.K == 1
    D = vp.D
    nodes, weights = np.polynomial.hermite.hermgauss(n_nodes)
    grids = np.meshgrid(*([nodes] * D), indexing="ij")
    z = np.sqrt(2.0) * np.column_stack([g.ravel() for g in grids])
    w = np.prod(
        np.meshgrid(*([weights] * D), indexing="ij"), axis=0
    ).ravel() / np.pi ** (D / 2)
    scale = (vp.sigma[0, 0] * vp.lambd[:, 0])[None, :]
    thetas = vp.mu[:, 0][None, :] + scale * z
    sn2_new = np.broadcast_to(sn2_new, (X_eval.shape[0],))
    return np.array(
        [
            np.sum(w * transform(look_ahead_sd(gp, thetas, x, sn2)))
            for x, sn2 in zip(X_eval, sn2_new)
        ]
    )


def weighted_samples_reference(gp, Xa, weights, X_eval, sn2_new, transform):
    """``sum_a weights[a] transform(s_new(Xa[a]; x))`` for every row of
    ``X_eval``: the acquisition's estimate recomputed from its own
    importance samples and normalized weights."""
    sn2_new = np.broadcast_to(sn2_new, (X_eval.shape[0],))
    return np.array(
        [
            np.sum(weights * transform(look_ahead_sd(gp, Xa, x, sn2)))
            for x, sn2 in zip(X_eval, sn2_new)
        ]
    )
