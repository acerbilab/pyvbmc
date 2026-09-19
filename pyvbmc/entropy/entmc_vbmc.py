from numbers import Integral

import numpy as np

from pyvbmc.calibration.profile import DEFAULT_CHUNK_ELEMENTS
from pyvbmc.rng import get_rng
from pyvbmc.variational_posterior import VariationalPosterior


def _block_layout(Ns: int, D: int, K: int, budget: int) -> tuple[int, int]:
    """Components and samples of the blocks a budget of doubles allows.

    The budget bounds one ``(components, samples, D, K)`` tensor of
    standardized distances. Samples are blocked only once one component's
    tensor alone exceeds the budget, and one sample's ``D x K`` slab is
    always used whole.
    """
    per_component = Ns * D * K
    g = int(max(1, min(K, budget // max(1, per_component))))
    if per_component <= budget:
        return g, Ns
    return g, int(max(1, budget // max(1, D * K)))


def _entmc_vbmc(
    vp: VariationalPosterior,
    Ns: int,
    grad_flags: tuple = tuple([True] * 4),
    jacobian_flag: bool = True,
    rng=None,
    *,
    budget: int,
):
    r"""Monte Carlo estimate of entropy of variational posterior.

    Parameters
    ----------
    vp : VariationalPosterior
        An instance of VariationalPosterior class.
    Ns : int
        Number of samples to draw per component. Ns > 0. Rounded up to an
        even number (antithetic sampling).
    grad_flags : tuple of bool, len(grad_flags)=4, default=tuple([True] * 4)
        Whether to compute the gradients for [mu, sigma, lambda, w].
    jacobian_flag : bool
        Whether variational parameters are transformed.
        The variational parameters and corresponding transformations are:
        sigma (log), lambda (log), w (softmax).
    rng : None, int, SeedSequence or np.random.Generator, optional
        Random generator (or seed) for the Monte Carlo samples. By default
        ``vp.rng`` is used.

    Returns
    -------
    H: float
        Estimated entropy of vp by Monte Carlo method.
    dH: np.ndarray
        Estimated entropy gradient by Monte Carlo method.
        :math:`dH = \left[\nabla_{\mu_1}^{T} H, ..., \nabla_{\mu_K}^{T} H,
        \nabla_{\sigma}^{T} H, \nabla_{\lambda}^{T} H,
        \nabla_{\omega}^{T} H\right]`
        Only the blocks requested through ``grad_flags`` are present.

    Notes
    -----
    ``Ns`` antithetic samples are drawn from each component, the mixture
    density is evaluated at every sample, and ``H`` is the average of
    ``-log q`` weighted by the component weights; the gradients use the
    reparameterization trick (the sample of component ``j`` is
    ``mu_j + sigma_j * lambda * epsilon``). All components' samples come
    from one ``standard_normal((K, Ns / 2, D))`` draw, which consumes the
    generator exactly as ``K`` successive draws of ``(Ns / 2, D)`` would.
    The density and the gradient terms are computed as a broadcast over a
    ``(components, samples, D, K)`` tensor of standardized distances, in
    blocks bounded by the fixed calibration profile (unless one sample's
    ``D x K`` slab already exceeds it): blocks of components, and blocks of
    samples within a component when one component's tensor alone exceeds
    the budget. The canonical blocks are the blocks of
    ``DEFAULT_CHUNK_ELEMENTS`` doubles, and the mixture densities and every
    sum over samples and components are taken over them whatever budget is
    in force, so neither the estimate nor its gradients depend on it: a
    computed block either subdivides one canonical block, whose component
    densities it fills in, or is a union of whole canonical blocks that it
    is split back into. The mixture sum over the components is a
    matrix-vector product, whose rounding follows the number of samples in
    a canonical block, so the estimate depends on the BLAS build at the
    level of its rounding.
    """
    if (
        isinstance(budget, (bool, np.bool_))
        or not isinstance(budget, Integral)
        or budget < 1
    ):
        raise ValueError("budget must be a positive integer.")
    rng = vp.rng if rng is None else get_rng(rng)

    D = int(vp.D)
    K = int(vp.K)
    mu = np.asarray(vp.mu, dtype=float)  # [D, K]
    if mu.shape != (D, K):
        raise ValueError(f"vp.mu must have shape {(D, K)}, got {mu.shape}.")
    mu_t = mu.T  # [K, D]
    sigma = np.asarray(vp.sigma, dtype=float).reshape(-1)  # [K, ]
    lambd = np.asarray(vp.lambd, dtype=float).reshape(-1)  # [D, ]
    w = np.asarray(vp.w, dtype=float).reshape(-1)  # [K, ]
    eta = np.asarray(vp.eta, dtype=float).reshape(-1)  # [K, ]

    grad_mu, grad_sigma, grad_lambd, grad_w = (bool(f) for f in grad_flags)
    need_lsum = grad_mu or grad_sigma or grad_lambd

    sigmalambd = sigma * lambd[:, None]  # [D, K]
    # Common normalization factor, and per component nf_k = nconst / sigma^D
    nconst = 1 / (2 * np.pi) ** (D / 2) / np.prod(lambd)
    nf = nconst / (sigma**D)  # [K, ]
    wnf = w * nf  # [K, ]: w_k * norm_k = wnf_k * exp(-d2_k / 2)
    # Factor turning delta_k = (x - mu_k) / (sigma_k lambd) and
    # exp(-d2_k / 2) into (x - mu_k) / (sigma_k lambd)^2 * w_k * norm_k
    C = wnf / sigmalambd  # [D, K]

    # Make sure Ns is even
    Ns = int(np.ceil(Ns / 2)) * 2
    half = Ns // 2

    # Antithetic samples for every component (component j's block is what
    # a draw of (half, D) for that component alone would have produced)
    eps_half = rng.standard_normal((K, half, D))
    epsilon = np.concatenate([eps_half, -eps_half], axis=1)  # [K, Ns, D]
    del eps_half  # epsilon is the one (K, Ns, D) array kept for the call

    # Sums over the samples of each component; the weights w_j and 1 / Ns
    # are applied at the end.
    sum_logq = np.zeros(K)  # sum_n log q(x_jn)
    mu_acc = np.zeros((K, D)) if grad_mu else None
    sigma_acc = np.zeros(K) if grad_sigma else None
    lambd_acc = np.zeros(D) if grad_lambd else None
    w_acc = np.zeros(K) if grad_w else None

    budget = int(budget)
    # Everything from the mixture density on runs over the canonical blocks;
    # the budget in force only decides how the component densities and the
    # location sums of a canonical block are produced.
    g_c, step_c = _block_layout(Ns, D, K, DEFAULT_CHUNK_ELEMENTS)
    g_x, step_x = _block_layout(Ns, D, K, budget)
    # A computed block subdivides one canonical block or is the union of
    # whole ones. Components are grouped only in a block that spans every
    # sample, so that canonical blocks reach the sums in the canonical order.
    if step_x >= Ns:
        step_x = Ns
    elif step_x > step_c:
        step_x = (step_x // step_c) * step_c
    if g_x > g_c:
        g_x = (g_x // g_c) * g_c if step_x >= Ns else g_c
    fill = g_x < g_c or step_x < step_c
    split = g_x > g_c or step_x > step_c
    outer_g, outer_n = max(g_c, g_x), max(step_c, step_x)

    def _component_terms(jj, nn):
        """Component densities and the location sum at a block's samples."""
        eps_b = epsilon[jj, nn]  # [g, n, D]
        # Samples of the block's components (reparameterization)
        X_b = eps_b * lambd * sigma[jj, None, None] + mu_t[jj, None, :]
        # Standardized distance of every sample to every component
        delta = (X_b[..., None] - mu) / sigmalambd  # [g, n, D, K]
        d2 = np.einsum("jndk,jndk->jnk", delta, delta)  # [g, n, K]
        E = np.exp(-0.5 * d2)  # [g, n, K]
        if not need_lsum:
            return E, None
        # sum_k (x - mu_k) / (sigma_k lambd)^2 * w_k * norm_k
        return E, np.einsum("jndk,dk,jnk->jnd", delta, C, E)  # [g, n, D]

    def _filled_block(jj, nn):
        """One canonical block filled in from budget-sized rectangles."""
        gl, nl = jj.stop - jj.start, nn.stop - nn.start
        E_blk = np.empty((gl, nl, K))
        lsum_blk = np.empty((gl, nl, D)) if need_lsum else None
        for a0 in range(0, gl, g_x):
            a1 = min(gl, a0 + g_x)
            for b0 in range(0, nl, step_x):
                b1 = min(nl, b0 + step_x)
                E_part, lsum_part = _component_terms(
                    slice(jj.start + a0, jj.start + a1),
                    slice(nn.start + b0, nn.start + b1),
                )
                E_blk[a0:a1, b0:b1] = E_part
                if need_lsum:
                    lsum_blk[a0:a1, b0:b1] = lsum_part
        return E_blk, lsum_blk

    def _canonical_parts(jj, nn, E_blk, lsum_blk):
        """The canonical blocks of a computed block, C-contiguous."""
        if not split:
            yield jj, nn, E_blk, lsum_blk
            return
        for a0 in range(0, jj.stop - jj.start, g_c):
            a1 = min(jj.stop - jj.start, a0 + g_c)
            for b0 in range(0, nn.stop - nn.start, step_c):
                b1 = min(nn.stop - nn.start, b0 + step_c)
                lsum_part = None
                if need_lsum:
                    lsum_part = np.ascontiguousarray(lsum_blk[a0:a1, b0:b1])
                yield (
                    slice(jj.start + a0, jj.start + a1),
                    slice(nn.start + b0, nn.start + b1),
                    np.ascontiguousarray(E_blk[a0:a1, b0:b1]),
                    lsum_part,
                )

    for j0 in range(0, K, outer_g):
        jo = slice(j0, min(K, j0 + outer_g))
        for n0 in range(0, Ns, outer_n):
            no = slice(n0, min(Ns, n0 + outer_n))
            computed = (
                _filled_block(jo, no) if fill else _component_terms(jo, no)
            )
            for jj, nn, E, lsum in _canonical_parts(jo, no, *computed):
                eps_b = epsilon[jj, nn]  # [g, n, D]
                q = E @ wnf  # [g, n]: mixture density at each sample
                sum_logq[jj] += np.log(q).sum(1)

                if need_lsum:
                    r = lsum / q[..., None]  # [g, n, D]
                    if grad_mu:
                        mu_acc[jj] += r.sum(1)
                    if grad_sigma:
                        sigma_acc[jj] += (r * eps_b * lambd).sum((1, 2))
                    if grad_lambd:
                        lambd_acc += np.einsum(
                            "j,jnd,jnd->d", w[jj] * sigma[jj], eps_b, r
                        )
                if grad_w:
                    # sum_j w_j sum_n exp(-d2_jnk / 2) / q_jn (nf_k below)
                    w_acc += np.einsum("j,jnk,jn->k", w[jj], E, 1.0 / q)

    H = -(w * sum_logq).sum() / Ns

    mu_grad = (w[:, None] * mu_acc).T / Ns if grad_mu else np.empty(0)
    sigma_grad = w * sigma_acc / Ns if grad_sigma else np.empty(0)
    lambd_grad = lambd_acc / Ns if grad_lambd else np.empty(0)
    w_grad = -(sum_logq + nf * w_acc) / Ns if grad_w else np.empty(0)

    # Correct for standard log reparameterization of SIGMA
    if jacobian_flag and grad_sigma:
        sigma_grad = sigma_grad * sigma

    # Correct for standard log reparameterization of LAMBDA
    if jacobian_flag and grad_lambd:
        lambd_grad = lambd_grad * lambd

    # Correct for standard softmax reparameterization of W
    if jacobian_flag and grad_w:
        eta_exp = np.exp(eta)
        eta_sum = eta_exp.sum()
        J_w = (
            -np.outer(eta_exp, eta_exp) / eta_sum**2
            + np.diag(eta_exp) / eta_sum
        )
        w_grad = J_w @ w_grad

    dH = np.concatenate([mu_grad.ravel("F"), sigma_grad, lambd_grad, w_grad])

    return H, dH


def entmc_vbmc(
    vp: VariationalPosterior,
    Ns: int,
    grad_flags: tuple = tuple([True] * 4),
    jacobian_flag: bool = True,
    rng=None,
):
    profile = vp._resolve_calibration()
    if any(bool(flag) for flag in grad_flags):
        budget = profile.entropy_grad_chunk_elements
    else:
        budget = profile.entropy_value_chunk_elements
    return _entmc_vbmc(
        vp,
        Ns,
        grad_flags,
        jacobian_flag,
        rng,
        budget=budget,
    )


entmc_vbmc.__doc__ = _entmc_vbmc.__doc__
