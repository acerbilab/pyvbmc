import numpy as np

from pyvbmc.variational_posterior import VariationalPosterior


def entmc_vbmc(
    vp: VariationalPosterior,
    Ns: int,
    grad_flags: tuple = tuple([True] * 4),
    jacobian_flag: bool = True,
):
    r"""Monte Carlo estimate of entropy of variational posterior.

    Parameters
    ----------
    vp : VariationalPosterior
        An instance of VariationalPosterior class.
    Ns : int
        Number of samples to draw. Ns > 0.
    grad_flags : tuple of bool, len(grad_flags)=4, default=tuple([True] * 4)
        Whether to compute the gradients for [mu, sigma, lambda, w].
    jacobian_flag : bool
        Whether variational parameters are transformed.
        The variational parameters and corresponding transformations are:
        sigma (log), lambda (log), w (softmax).

    Returns
    -------
    H: float
        Estimated entropy of vp by Monte Carlo method.
    dH: np.ndarray
        Estimated entropy gradient by Monte Carlo method.
        :math:`dH = \left[\nabla_{\mu_1}^{T} H, ..., \nabla_{\mu_K}^{T} H,
        \nabla_{\sigma}^{T} H, \nabla_{\lambda}^{T} H,
        \nabla_{\omega}^{T} H\right]`

    """

    D = vp.D
    K = vp.K
    mu = vp.mu  # [D,K]
    sigma = vp.sigma.ravel()  # [1,K] -> [K, ]
    lambd = vp.lambd.ravel()  # [D,1] -> [D, ]
    w = vp.w.ravel()  # [1,K] -> [K, ]
    eta = vp.eta.ravel()  # [1,K] -> [K, ]

    # Check which gradients are computed
    mu_grad = np.zeros([D, K]) if grad_flags[0] else np.empty(0)
    sigma_grad = np.zeros(K) if grad_flags[1] else np.empty(0)
    lambd_grad = np.zeros(D) if grad_flags[2] else np.empty(0)
    w_grad = np.zeros(K) if grad_flags[3] else np.empty(0)

    sigmalambd = sigma * lambd[..., None]  # [D, K]
    nconst = (
        1 / (2 * np.pi) ** (D / 2) / np.prod(lambd)
    )  # Common normalization factor

    H = 0

    # Make sure Ns is even
    Ns = np.ceil(Ns / 2).astype(int) * 2
    epsilon = np.zeros([Ns, D])

    for j in range(K):
        # Draw Monte Carlo samples from the j-th component
        # Antithetic sampling
        epsilon[: Ns // 2, :] = np.random.randn(Ns // 2, D)
        epsilon[Ns // 2 :, :] = -epsilon[: Ns // 2, :]

        Xs = epsilon * lambd * sigma[j] + mu[:, j]  # [Ns, D]

        # Compute pdf
        ys = np.zeros(Ns)
        for k in range(K):
            d2 = ((Xs - mu[:, k]) / (sigma[k] * lambd)) ** 2  # [Ns, D]
            d2 = d2.sum(1)
            nn = w[k] * nconst / (sigma[k] ** D) * np.exp(-0.5 * d2)
            ys += nn

        H += -w[j] * np.log(ys).sum() / Ns

        # Compute gradient via reparameterization trick
        if any(grad_flags):
            # Full mixture (for sample from the j-th component)
            norm_j1 = ((Xs[..., None] - mu) / sigmalambd) ** 2  # [Ns, D, K]
            norm_j1 = norm_j1.sum(1)  # [Ns, K]
            norm_j1 = np.exp(-0.5 * norm_j1)
            norm_j1 = nconst / (sigma**D) * norm_j1  # [Ns, K]

            q_j = (w * norm_j1).sum(1)  # [Ns, ]

            # Compute sum for gradient wrt mu
            lsum = (Xs[..., None] - mu) / sigmalambd**2  # [Ns, D, K]
            lsum = lsum * w * norm_j1[:, None, :]  # [Ns, D, K]
            lsum = lsum.sum(2)  # [Ns, D]

            if grad_flags[0]:
                mu_grad[:, j] = w[j] * (lsum / q_j[..., None]).sum(0) / Ns

            if grad_flags[1]:
                # Compute sum for gradient wrt sigma
                isum = (lsum * epsilon * lambd).sum(1)  # [Ns, ]
                sigma_grad[j] = (w[j] * isum / q_j).sum() / Ns

            if grad_flags[2]:
                lambd_grad += (
                    w[j] * sigma[j] * epsilon * lsum / q_j[..., None]
                ).sum(0) / Ns

            if grad_flags[3]:
                w_grad[j] -= np.log(q_j).sum() / Ns
                w_grad[:] -= (w[j] * norm_j1 / q_j[:, None]).sum(0) / Ns

    # Correct for standard log reparameterization of SIGMA
    if jacobian_flag and grad_flags[1]:
        sigma_grad = sigma_grad * sigma

    # Correct for standard log reparameterization of LAMBDA
    if jacobian_flag and grad_flags[2]:
        lambd_grad = lambd_grad * lambd

    # Correct for standard softmax reparameterization of W
    if jacobian_flag and grad_flags[3]:
        eta_exp = np.exp(eta)
        eta_sum = eta_exp.sum()
        J_w = (
            -eta_exp[:, None] * eta_exp[None, :] / eta_sum**2
            + np.diag(eta_exp) / eta_sum
        )
        w_grad = J_w @ w_grad

    dH = np.concatenate([mu_grad.ravel("F"), sigma_grad, lambd_grad, w_grad])

    return H, dH
