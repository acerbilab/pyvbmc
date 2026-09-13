"""Log densities of every stacked component at stratified draws.

The Monte Carlo entropy of the stacked posterior needs, for ``n_samples``
draws from every component, the log density of every component at every
draw, all in the original parameter space. This module builds that matrix.
Work shared by the components of one run is done once per run: one inverse
transform maps the draws of all its components to the original space, and
its forward transform and log-Jacobian are evaluated once for all rows
before the diagonal-Gaussian log densities of its components are formed in
one broadcast over a bounded workspace. The values equal those of a
component-by-component evaluation up to floating-point rounding: for a
warped run, the matrix product of the rotation in the inverse transform
rounds differently for a different number of rows.
"""

import numpy as np

# Bound on the ``(rows, K_m, D)`` workspace holding one run's densities.
_WORKSPACE_BYTES = 32 * 1024**2
# Written as SciPy's normal distribution writes it (``log(sqrt(2 pi))``).
_NORMAL_LOG_CONSTANT = np.log(np.sqrt(2.0 * np.pi))


def _component_parameters(vp):
    """Return a run's component means and scales as ``(K, D)`` arrays."""
    mu = np.asarray(vp.mu, dtype=np.float64)
    sigma = np.asarray(vp.lambd * vp.sigma, dtype=np.float64)
    return mu.T, sigma.T


def component_log_densities(
    vp_list, n_samples, rng, *, workspace_bytes=_WORKSPACE_BYTES
):
    """Draw from every component and evaluate every component's log density.

    Parameters
    ----------
    vp_list : list of VariationalPosterior
        The retained runs, in stacking order. Read, never modified.
    n_samples : int
        Draws per component, at least 1.
    rng : numpy.random.Generator
        Source of the draws: one ``standard_normal((n_samples, D))`` call
        per component, in run order and then component order.
    workspace_bytes : int, optional
        Bound on the temporary array holding one run's densities; the rows
        are processed in chunks that fit it, at least one row at a time.
        Chunking does not change the values. Default 32 MiB.

    Returns
    -------
    logq : np.ndarray, shape (K_total * n_samples, K_total)
        ``logq[s, j]`` is the log density of component ``j`` in the original
        space at draw ``s``. Rows ``mk * n_samples`` to ``(mk + 1) *
        n_samples`` hold the draws of component ``mk``, components numbered
        in run order and then component order. Mixture log weights are not
        included.
    """
    n_samples = int(n_samples)
    if n_samples < 1:
        raise ValueError("`n_samples` should be at least 1.")
    counts = [int(np.asarray(vp.mu).shape[1]) for vp in vp_list]
    K_total = int(sum(counts))
    D = int(np.asarray(vp_list[0].mu).shape[0])
    S = K_total * n_samples

    # Step 1: draws from every component in its run's transformed space,
    # mapped to the original space with one inverse transform per run.
    X_orig = np.empty((S, D), dtype=np.float64)
    start = 0
    for vp, K_m in zip(vp_list, counts):
        mu, sigma = _component_parameters(vp)
        U = np.empty((K_m * n_samples, D), dtype=np.float64)
        for k in range(K_m):
            z = rng.standard_normal((n_samples, D))
            U[k * n_samples : (k + 1) * n_samples] = z * sigma[k] + mu[k]
        rows = slice(start * n_samples, (start + K_m) * n_samples)
        X_orig[rows] = vp.parameter_transformer.inverse(U)
        start += K_m

    # Step 2: log q_mk(x) in the original space for every draw and every
    # component: a diagonal normal in the run's transformed space, Jacobian
    # corrected. The transform and its Jacobian are evaluated once per run;
    # the run's components are evaluated together in a bounded workspace.
    logq = np.empty((S, K_total), dtype=np.float64)
    start = 0
    for vp, K_m in zip(vp_list, counts):
        transform = vp.parameter_transformer
        mu, sigma = _component_parameters(vp)
        log_sigma = np.log(sigma)
        U = np.asarray(transform(X_orig), dtype=np.float64).reshape(S, D)
        jac = np.asarray(
            transform.log_abs_det_jacobian(U), dtype=np.float64
        ).reshape(S, 1)

        chunk_rows = min(S, max(1, workspace_bytes // (8 * K_m * D)))
        workspace = np.empty((chunk_rows, K_m, D), dtype=np.float64)
        columns = slice(start, start + K_m)
        for r0 in range(0, S, chunk_rows):
            r1 = min(r0 + chunk_rows, S)
            terms = workspace[: r1 - r0]
            np.subtract(U[r0:r1, np.newaxis, :], mu[np.newaxis], out=terms)
            terms /= sigma
            np.square(terms, out=terms)
            terms *= -0.5
            terms -= _NORMAL_LOG_CONSTANT
            terms -= log_sigma
            logq[r0:r1, columns] = terms.sum(axis=2) - jac[r0:r1]
        start += K_m

    return logq
