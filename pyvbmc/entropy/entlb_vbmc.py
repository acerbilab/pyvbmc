import numpy as np

from pyvbmc.variational_posterior import VariationalPosterior


def entlb_vbmc(
    vp: VariationalPosterior,
    grad_flags: tuple = tuple([True] * 4),
    jacobian_flag: bool = True,
):
    r"""Entropy lower bound for variational posterior by Jensen's inequality.

    Parameters
    ----------
    vp : VariationalPosterior
        An instance of VariationalPosterior class.
    grad_flags : tuple of bool, len(grad_flags)=4, optional
        Whether to compute the gradients for [mu, sigma, lambda, w].
    jacobian_flag : bool, optional
        Whether variational parameters are transformed. The variational
        parameters and corresponding transformations are:
        sigma (log), lambda (log), w (softmax).

    Returns
    -------
    H: float
        Entropy lower bound of vp [1]_.
    dH: np.ndarray
        Gradient of entropy lower bound.

    Raises
    ------
    NotImplementedError
        Not implemented for K > BigK.

    References
    ----------
    .. [1] Gershman, S. J., Hoffman, M. D., & Blei, D. M. (2012).
        Nonparametric variational inference. Proceedings of the 29th
        International Conference on Machine Learning, 235–242.

    """
    BigK = np.inf  # large number of components

    D = vp.D
    K = vp.K
    mu = vp.mu  # [D, K]
    mu_t = mu.T  # [K, D]
    sigma = vp.sigma.ravel()  # [1,K] -> [K, ]
    lambd = vp.lambd.ravel()  # [D,1] -> [D, ]
    w = vp.w.ravel()  # [1,K] -> [K, ]
    eta = vp.eta.ravel()  # [1,K] -> [K, ]

    # Check which gradients are computed
    mu_grad = np.zeros((D, K)) if grad_flags[0] else np.empty(0)
    sigma_grad = np.zeros(K) if grad_flags[1] else np.empty(0)
    lambd_grad = np.zeros(D) if grad_flags[2] else np.empty(0)
    w_grad = np.zeros(K) if grad_flags[3] else np.empty(0)

    if K == 1:
        # Entropy of single component, uses exact expression
        H = (
            0.5 * D * (1 + np.log(2 * np.pi))
            + D * np.log(sigma).sum()
            + np.log(lambd).sum()
        )

        if grad_flags[0]:
            mu_grad = np.zeros(D)

        if grad_flags[1]:
            sigma_grad = D / sigma

        if grad_flags[2]:
            lambd_grad = 1 / lambd

        if grad_flags[3]:
            w_grad = np.zeros(1)

    elif K > BigK:
        raise NotImplementedError("Not implemented yet for K > BigK.")
    else:
        # Multiple components
        sumsigma2 = sigma[:, None] ** 2 + sigma[None, :] ** 2
        sumsigma = np.sqrt(sumsigma2)  # [K, K]

        nconst = 1 / (2 * np.pi) ** (D / 2) / np.prod(lambd)

        d2 = (
            (mu_t[:, None, :] - mu_t[None, :, :])
            / (sumsigma[..., None] * lambd)
        ) ** 2  # [K, K, D]
        d2 = d2.sum(2)  # [K, K]
        gamma = nconst / sumsigma**D * np.exp(-0.5 * d2)  # [K, K]
        gammasum = (w * gamma).sum(1)  # [K, ]

        H = -(w * np.log(gammasum)).sum()

        if any(grad_flags):
            gammafrac = (
                gamma / gammasum
            )  # [K, K], gammafrac[i,j]=gamma[i,j]/gammasum[j]
            wgammafrac = (
                w * gammafrac
            )  # [K, K], wgammafrac[i,j] = w[j]*gamma[i,j]/gammasum[j]

            if grad_flags[0]:
                dmu = (mu_t[:, None, :] - mu_t[None, :, :]) / (
                    sumsigma2[..., None] * lambd**2
                )  # [K, K, D], dmu[i,j,:] = (mu_i - mu_j)/sumsigma2[i]/lambd^2

            if grad_flags[1]:
                dsigma = -D / sumsigma2 + 1 / sumsigma2**2 * np.sum(
                    ((mu_t[:, None, :] - mu_t[None, :, :]) / lambd) ** 2, 2
                )  # [K, K]

            # Loop over mixture components
            for j in range(K):
                if grad_flags[0]:
                    m1 = (wgammafrac[j, :][:, None] * dmu[:, j, :]).sum(0)
                    m2 = (
                        dmu[:, j, :] * gamma[:, [j]] * w[:, None]
                    ) / gammasum[
                        j
                    ]  # [K, D]
                    m2 = m2.sum(0)

                    mu_grad[:, j] = -w[j] * (m1 + m2)

                if grad_flags[1]:
                    # Compute terms of gradient with respect to sigma_j
                    s1 = (wgammafrac[j, :] * dsigma[:, j]).sum(0)
                    s2 = (dsigma[:, j] * gamma[:, j] * w) / gammasum[j]
                    s2 = s2.sum(0)

                    sigma_grad[j] = -w[j] * sigma[j] * (s1 + s2)

            if grad_flags[2]:
                dmu2 = (
                    (mu_t[:, None, :] - mu_t[None, :, :]) ** 2
                    / sumsigma2[..., None]
                    / lambd**2
                )  # [K, K, D]

                lambd_grad[:] = (
                    -np.sum(
                        w[:, None]
                        * np.sum(
                            w[:, None, None] * gamma[:, :, None] * (dmu2 - 1),
                            0,
                        )
                        / gammasum[:, None],
                        0,
                    )
                    / lambd
                )

            if grad_flags[3]:
                w_grad[:] = -np.log(gammasum) - wgammafrac.sum(1)

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
