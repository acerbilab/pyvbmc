"""Log densities of every stacked component at stratified draws.

The Monte Carlo entropy of the stacked posterior needs, for ``n_samples``
draws from every component, the log density of every component at every
draw, all in the original parameter space: a matrix with ``S = K_total *
n_samples`` rows, one per draw, and ``K_total`` columns. It is computed in
two stages.

1. Draw from every component and map the draws to the original space with
   one inverse transform per run; then evaluate each run's forward
   transform and log-Jacobian at all ``S`` draws in one call. For a warped
   run, the matrix product of the rotation rounds differently for a
   different number of rows. The values therefore equal those of a
   component-by-component evaluation only up to floating-point rounding,
   and the transforms are never evaluated on part of the rows, which would
   change them.
2. Form the log densities of every run's components, diagonal normals in
   the run's transformed space corrected by the log-Jacobian, for a range
   of rows: one broadcast per run over a bounded workspace. The values in
   a row do not depend on the range they are computed in.

:func:`component_log_densities` returns the whole matrix.
:func:`log_density_chunks` yields the same values a chunk of rows at a
time, so that the entropy of a large stack can be reduced without holding
the matrix. It keeps the first stage's transformed draws and log-Jacobians
of every run, ``M * S * (D + 1)`` values for ``M`` runs, which every chunk
reads.
"""

import numpy as np

# Bound on the ``(rows, K_m, D)`` workspace holding one run's densities.
_WORKSPACE_BYTES = 32 * 1024**2
# Bound on a chunk of rows of the matrix together with the temporaries of
# its size that the entropy's weighted log-sum-exp makes (the sum with the
# log weights, and its difference from the row maxima inside
# ``torch.logsumexp``): ``_CHUNK_COPIES`` arrays of the chunk's size.
_CHUNK_BYTES = 64 * 1024**2
_CHUNK_COPIES = 3
# Written as SciPy's normal distribution writes it (``log(sqrt(2 pi))``).
_NORMAL_LOG_CONSTANT = np.log(np.sqrt(2.0 * np.pi))


class _Run:
    """One run's components and its transform of every draw."""

    __slots__ = ("columns", "mu", "sigma", "log_sigma", "U", "jac")

    def __init__(self, columns, mu, sigma, U, jac):
        self.columns = columns
        self.mu = mu
        self.sigma = sigma
        self.log_sigma = np.log(sigma)
        self.U = U
        self.jac = jac


def _component_parameters(vp):
    """Return a run's component means and scales as ``(K, D)`` arrays."""
    mu = np.asarray(vp.mu, dtype=np.float64)
    sigma = np.asarray(vp.lambd * vp.sigma, dtype=np.float64)
    return mu.T, sigma.T


def _component_counts(vp_list, n_samples):
    """Validate ``n_samples``; return it with the runs' component counts."""
    n_samples = int(n_samples)
    if n_samples < 1:
        raise ValueError("`n_samples` should be at least 1.")
    counts = [int(np.asarray(vp.mu).shape[1]) for vp in vp_list]
    return n_samples, counts


def _draw(vp_list, counts, n_samples, rng):
    """Draw from every component; return the draws in the original space.

    One ``standard_normal((n_samples, D))`` call per component, in run
    order and then component order, and one inverse transform per run.
    """
    D = int(np.asarray(vp_list[0].mu).shape[0])
    S = int(sum(counts)) * n_samples
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
    return X_orig


def _transformed_runs(vp_list, counts, X_orig):
    """Yield each run's forward transform and log-Jacobian at all draws."""
    S, D = X_orig.shape
    start = 0
    for vp, K_m in zip(vp_list, counts):
        transform = vp.parameter_transformer
        mu, sigma = _component_parameters(vp)
        U = np.asarray(transform(X_orig), dtype=np.float64).reshape(S, D)
        jac = np.asarray(
            transform.log_abs_det_jacobian(U), dtype=np.float64
        ).reshape(S, 1)
        yield _Run(slice(start, start + K_m), mu, sigma, U, jac)
        start += K_m


def _block_rows(n_rows, run, workspace_bytes):
    """Rows of one run's densities that fit the workspace, at least one."""
    return min(n_rows, max(1, workspace_bytes // (8 * run.mu.size)))


def _workspace(runs, n_rows, workspace_bytes):
    """A flat workspace for blocks of the densities of any of ``runs``."""
    size = max(
        _block_rows(n_rows, run, workspace_bytes) * run.mu.size for run in runs
    )
    return np.empty(size, dtype=np.float64)


def _fill_log_densities(runs, r0, r1, out, workspace, workspace_bytes):
    """Write rows ``r0:r1`` of the columns of ``runs`` to ``out``.

    ``out[: r1 - r0, run.columns]`` receives the log density in the
    original space of every component of ``run`` at the draws ``r0:r1``,
    for every run in ``runs``, evaluated in blocks of rows that fit
    ``workspace_bytes`` and held at the start of ``workspace``.
    """
    for run in runs:
        K_m, D = run.mu.shape
        block = _block_rows(r1 - r0, run, workspace_bytes)
        blocks = workspace[: block * K_m * D].reshape(block, K_m, D)
        for b0 in range(r0, r1, block):
            b1 = min(b0 + block, r1)
            terms = blocks[: b1 - b0]
            np.subtract(
                run.U[b0:b1, np.newaxis, :], run.mu[np.newaxis], out=terms
            )
            terms /= run.sigma
            np.square(terms, out=terms)
            terms *= -0.5
            terms -= _NORMAL_LOG_CONSTANT
            terms -= run.log_sigma
            out[b0 - r0 : b1 - r0, run.columns] = (
                terms.sum(axis=2) - run.jac[b0:b1]
            )


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
        are processed in blocks that fit it, at least one row at a time.
        The blocks do not change the values. Default 32 MiB.

    Returns
    -------
    logq : np.ndarray, shape (K_total * n_samples, K_total)
        ``logq[s, j]`` is the log density of component ``j`` in the original
        space at draw ``s``. Rows ``mk * n_samples`` to ``(mk + 1) *
        n_samples`` hold the draws of component ``mk``, components numbered
        in run order and then component order. Mixture log weights are not
        included.
    """
    n_samples, counts = _component_counts(vp_list, n_samples)
    X_orig = _draw(vp_list, counts, n_samples, rng)
    S = X_orig.shape[0]
    logq = np.empty((S, int(sum(counts))), dtype=np.float64)
    # The matrix is filled one run's columns at a time, so only one run's
    # transformed draws are held at once.
    for run in _transformed_runs(vp_list, counts, X_orig):
        workspace = _workspace([run], S, workspace_bytes)
        _fill_log_densities([run], 0, S, logq, workspace, workspace_bytes)
    return logq


def log_density_chunks(
    vp_list,
    n_samples,
    rng,
    *,
    chunk_rows=None,
    workspace_bytes=_WORKSPACE_BYTES,
):
    """Draw from every component and yield the log densities by rows.

    The draws and every run's transforms are made when this function is
    called, consuming ``rng`` exactly as :func:`component_log_densities`
    does; the returned iterator then fills the rows of that function's
    matrix, with the same values, one chunk at a time.

    Parameters
    ----------
    vp_list : list of VariationalPosterior
        The retained runs, in stacking order. Read, never modified.
    n_samples : int
        Draws per component, at least 1.
    rng : numpy.random.Generator
        Source of the draws: one ``standard_normal((n_samples, D))`` call
        per component, in run order and then component order.
    chunk_rows : int, optional
        Rows per chunk, at least 1. By default the most that keep a chunk,
        with the two temporaries of its size that the entropy's weighted
        log-sum-exp makes, within 64 MiB; at least one.
    workspace_bytes : int, optional
        Bound on the temporary array holding one run's densities, as for
        :func:`component_log_densities`. Default 32 MiB.

    Returns
    -------
    chunks : iterator of (int, int, np.ndarray)
        ``(r0, r1, logq_rows)`` for consecutive ranges of rows that cover
        all ``K_total * n_samples`` rows in order. ``logq_rows``, of shape
        ``(r1 - r0, K_total)``, equals ``logq[r0:r1]`` of
        :func:`component_log_densities`. It is one array reused for every
        chunk, so each chunk is overwritten by the next.
    """
    n_samples, counts = _component_counts(vp_list, n_samples)
    K_total = int(sum(counts))
    if chunk_rows is None:
        chunk_rows = max(1, _CHUNK_BYTES // (_CHUNK_COPIES * 8 * K_total))
    elif int(chunk_rows) < 1:
        raise ValueError("`chunk_rows` should be at least 1.")
    X_orig = _draw(vp_list, counts, n_samples, rng)
    runs = list(_transformed_runs(vp_list, counts, X_orig))
    S = X_orig.shape[0]
    return _chunks(runs, S, K_total, min(S, int(chunk_rows)), workspace_bytes)


def _chunks(runs, S, K_total, chunk_rows, workspace_bytes):
    """Yield ``(r0, r1, logq_rows)`` for consecutive chunks of rows."""
    rows = np.empty((chunk_rows, K_total), dtype=np.float64)
    workspace = _workspace(runs, chunk_rows, workspace_bytes)
    for r0 in range(0, S, chunk_rows):
        r1 = min(r0 + chunk_rows, S)
        logq_rows = rows[: r1 - r0]
        _fill_log_densities(
            runs, r0, r1, logq_rows, workspace, workspace_bytes
        )
        yield r0, r1, logq_rows
