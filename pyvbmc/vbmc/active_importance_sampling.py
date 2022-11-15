import copy
import sys
from math import ceil

import gpyreg as gpr
import numpy as np
from scipy.linalg import solve_triangular


def active_importance_sampling(vp, gp, acq_fcn, options):
    """
    Set up importance sampling acquisition functions (viqr/imiqr).

    This function samples from the base importance sampling (IS) density in
    three steps:
    1) Use importance sampling-resampling (ISR) to sample from the
    base IS density based on a proposal distribuion which is a mixture of
    a smoothed variational posterior and on box-uniform mixture centered
    around the current training points
    2) Optionally use MCMC initialized with the ISR samples to sample from
    the base IS density
    3) Compute IS statistics and book-keeping

    Parameters
    ----------
    gp : gpr.GP
        The GP surrogate.
    vp : VariationalPosterior
        The variational posterior.
    acq_fcn : AbstractAcqFcn
        The acquisition function callable.
    options : Options
        The VBMC options.

    Returns
    -------
    active_is : dict
        A dictionary of importance sampling values and bookkeeping.
    """
    # Do we simply sample from the variational posterior?
    only_vp_flag = acq_fcn.acq_info.get(
        "variational_importance_sampling", False
    )

    D = gp.X.shape[1]
    Ns_gp = len(gp.posteriors)  # Number of gp hyperparameter samples

    # Input space bounds and typical scales (for MCMC only)
    widths = np.std(gp.X, axis=0, ddof=1)
    max_bnd = 0.5
    diam = np.amax(gp.X, axis=0) - np.amin(gp.X, axis=0)
    lb_tran = np.amin(gp.X, axis=0) - max_bnd * diam
    ub_tran = np.amax(gp.X, axis=0) + max_bnd * diam

    active_is = {}
    active_is["ln_weights"] = None
    active_is["X"] = None
    active_is["f_s2"] = None

    if only_vp_flag:
        # Step 0: Simply sample from variational posterior.

        Na = ceil(
            options.eval(
                "active_importance_sampling_mcmc_samples",
                {"K": vp.K, "n_vars": D, "D": D},
            )
        )

        if not np.isfinite(Na) or not np.isscalar(Na) or Na <= 0:
            raise ValueError(
                "options['active_importance_sampling_mcmc_samples']"
                + "should evaluate to a positive integer."
            )

        Xa, __ = vp.sample(Na, orig_flag=False)

        f_mu, f_s2 = gp.predict(Xa, separate_samples=True)

        if acq_fcn.acq_info.get("mcmc_importance_sampling"):
            # Compute fractional effective sample size (ESS)
            fESS = fess(vp, f_mu, Xa)

            if fESS < options["active_importance_sampling_fess_thresh"]:
                log_p_fun = lambda x: acq_fcn.is_log_full(x, vp=vp, gp=gp)

                # Get MCMC options
                Nmcmc_samples = (
                    Na * options["active_importance_sampling_mcmc_thin"]
                )
                thin = 1
                burn_in = 0
                sampler_opts, __, __ = get_mcmc_opts(Nmcmc_samples)
                # W = Na  # walkers, not applicable for simple slice sampling.

                # Perform a single MCMC step for all samples.
                # Contrary to MATLAB, we are using simple slice sampling.
                # Better (e.g. ensemble slice) sampling methods could
                # later be implemented.
                sampler = gpr.slice_sample.SliceSampler(
                    log_p_fun, Xa, widths, lb_tran, ub_tran, sampler_opts
                )
                results = sampler.sample(Nmcmc_samples, thin, burn_in)
                Xa = results["samples"]
                # Xa = eis_sample_lite(log_p_fun, Xa, Nmcmc_samples, W, widths,
                # lb_tran, ub_tran, sample_opts)
                Xa = Xa[-Na:, :]
                f_mu, f_s2 = gp.predict(Xa, separate_samples=True)

        ln_y = acq_fcn.is_log_base(Xa, f_mu=f_mu, f_s2=f_s2)

        active_is["f_s2"] = f_s2
        active_is["ln_weights"] = ln_y.T
        active_is["X"] = Xa

    else:
        # Step 1: Importance sampling-resampling

        Nvp_samples = options["active_importance_sampling_vp_samples"]
        Nbox_samples = options["active_importance_sampling_box_samples"]
        w_vp = Nvp_samples / (Nvp_samples + Nbox_samples)

        rect_delta = 2 * np.std(gp.X, ddof=1, axis=0)

        # Smoothed posterior for importance sampling-resampling
        if Nvp_samples > 0:
            scale_vec = np.array([0.05, 0.2, 1.0])

            vp_is = copy.deepcopy(vp)
            for i in range(len(scale_vec)):
                vp_is.K = vp_is.K + vp.K
                vp_is.w = np.hstack((vp_is.w, vp.w))
                vp_is.mu = np.hstack((vp_is.mu, vp.mu))
                vp_is.sigma = np.hstack(
                    (vp_is.sigma, np.sqrt(vp.sigma**2 + scale_vec[i] ** 2))
                )
            vp_is.w = vp_is.w / np.sum(vp_is.w)

            # Sample from smoothed posterior
            Xa_vp, __ = vp_is.sample(Nvp_samples, orig_flag=False)
            ln_weights, f_s2a_vp = active_sample_proposal_pdf(
                Xa_vp, gp, vp_is, w_vp, rect_delta, acq_fcn
            )
            if active_is.get("ln_weights") is None:
                active_is["ln_weights"] = ln_weights.T
            else:
                active_is["ln_weights"] = np.append(
                    active_is["ln_weights"], ln_weights.T, axis=1
                )
            if active_is.get("X") is None:
                active_is["X"] = Xa_vp
            else:
                active_is["X"] = np.append(active_is["X"], Xa_vp, axis=0)
            if active_is.get("f_s2") is None:
                active_is["f_s2"] = f_s2a_vp
            else:
                active_is["f_s2"] = np.append(
                    active_is["f_s2"], f_s2a_vp, axis=0
                )
        else:
            vp_is = None

        # Box-uniform sampling around training inputs
        if Nbox_samples > 0:
            jj = np.random.randint(0, len(gp.X), size=(Nbox_samples,))
            Xa_box = (
                gp.X[jj, :] + (2 * np.random.rand(jj.size, D) - 1) * rect_delta
            )
            ln_weights, f_s2a_box = active_sample_proposal_pdf(
                Xa_box, gp, vp_is, w_vp, rect_delta, acq_fcn
            )
            if active_is.get("ln_weights") is None:
                active_is["ln_weights"] = ln_weights.T
            else:
                active_is["ln_weights"] = np.append(
                    active_is["ln_weights"], ln_weights.T, axis=1
                )
            if active_is.get("X") is None:
                active_is["X"] = Xa_box
            else:
                active_is["X"] = np.append(active_is["X"], Xa_box, axis=0)
            if active_is.get("f_s2") is None:
                active_is["f_s2"] = f_s2a_box
            else:
                active_is["f_s2"] = np.append(
                    active_is["f_s2"], f_s2a_box, axis=0
                )

        active_is["ln_weights"][
            ~np.isfinite(active_is["ln_weights"])
        ] = -np.inf

        # Step 2 (optional): MCMC sample

        Nmcmc_samples = options["active_importance_sampling_mcmc_samples"]

        if Nmcmc_samples > 0:
            active_is_old = copy.deepcopy(active_is)

            active_is["ln_weights"] = np.zeros((Ns_gp, Nmcmc_samples))
            active_is["X"] = np.zeros((Ns_gp, Nmcmc_samples, D))
            active_is["f_s2"] = np.zeros((Nmcmc_samples, Ns_gp))

            # Consider only one GP sample at a time
            gp1 = copy.deepcopy(gp)
            for s in range(Ns_gp):
                gp1.posteriors = np.array(
                    [gp.posteriors[s]]
                )  # Assign current GP sample

                # quasi-random grid for D <= 2, not implemented
                # See activeimportancesampling_vbmc.m, lines 159 to 165.

                log_p_fun = lambda x: acq_fcn.is_log_full(x, vp=vp, gp=gp1)

                # Get MCMC Options
                thin = options["active_importance_sampling_mcmc_thin"]
                burn_in = ceil(thin * Nmcmc_samples / 2)
                sampler_opts, __, __ = get_mcmc_opts(Nmcmc_samples)

                # Walkers = 2 * (D + 1)  # For ensemble slice sampling,
                # not applicable for simple slice sampling.

                # Use importance sampling-resampling
                f_mu, f_s2 = gp1.predict(
                    active_is_old["X"], separate_samples=True
                )
                ln_weights = active_is_old["ln_weights"][s, :].reshape(
                    -1, 1
                ) + acq_fcn.is_log_added(f_mu=f_mu, f_s2=f_s2)
                ln_weights_max = np.amax(ln_weights, axis=1).reshape(-1, 1)
                if np.any(ln_weights_max == -np.inf):
                    raise ValueError("Invalid value.")
                weights = np.exp(ln_weights - ln_weights_max).ravel()
                weights = weights / np.sum(weights)
                # x0 = np.zeros((Walkers, D))
                # Select x0 without replacement by weight:
                index = np.random.choice(
                    a=len(weights), p=weights, replace=False
                )
                x0 = active_is_old["X"][index, :]
                x0 = np.maximum(
                    np.minimum(x0, ub_tran), lb_tran
                )  # Force inside bounds

                # Contrary to MATLAB, we are using simple slice sampling.
                # Better (e.g. ensemble slice) sampling methods could
                # later be implemented.
                sampler = gpr.slice_sample.SliceSampler(
                    log_p_fun, x0, widths, lb_tran, ub_tran, sampler_opts
                )
                results = sampler.sample(Nmcmc_samples, thin, burn_in)
                Xa, log_p = results["samples"], results["f_vals"]
                f_mu, f_s2 = gp1.predict(Xa, separate_samples=True)

                # Fixed log weight for importance sampling
                # (log fixed integrand)
                ln_y = acq_fcn.is_log_base(Xa, f_mu=f_mu, f_s2=f_s2)

                active_is["f_s2"][:, s] = f_s2.ravel()
                active_is["ln_weights"][s, :] = ln_y.T - log_p.T
                active_is["X"][s, :, :] = Xa

    # Step 3: Pre-compute quantities for importance sampling calculations:

    # Pre-compute cross-kernel matrix on importance points
    if active_is["X"].ndim == 3:
        K_Xa_X = np.zeros((Ns_gp, active_is["X"].shape[1], gp.X.shape[0]))
        C_tmp = np.zeros((Ns_gp, gp.X.shape[0], active_is["X"].shape[1]))
    else:
        K_Xa_X = np.zeros((Ns_gp, active_is["X"].shape[0], gp.X.shape[0]))
        C_tmp = np.zeros((Ns_gp, gp.X.shape[0], active_is["X"].shape[0]))

    for s in range(Ns_gp):
        if active_is["X"].ndim == 3:
            Xa = active_is["X"][s, :, :]
        else:
            Xa = active_is["X"]
        cov_N = gp.covariance.hyperparameter_count(gp.D)
        hyp = gp.posteriors[s].hyp[0:cov_N]  # just covariance hyperparameters
        L = gp.posteriors[s].L
        L_chol = gp.posteriors[s].L_chol
        sn2_eff = 1 / gp.posteriors[s].sW[0] ** 2
        if isinstance(
            gp.covariance, gpr.covariance_functions.SquaredExponential
        ):
            K_Xa_X[s, :, :] = gp.covariance.compute(hyp, Xa, gp.X)
        else:
            raise ValueError(
                "Covariance functions besides"
                + "SquaredExponential are not supported yet."
            )

        if L_chol:
            C_tmp[s, :, :] = (
                solve_triangular(
                    L,
                    solve_triangular(
                        L, K_Xa_X[s, :, :].T, trans=True, check_finite=False
                    ),
                    check_finite=False,
                )
                / sn2_eff
            )
        else:
            C_tmp[s, :, :] = L @ K_Xa_X[s, :, :].T
    active_is["K_Xa_X"] = K_Xa_X
    active_is["C_tmp"] = C_tmp

    # Omitted port, integrated mean functions:
    # activeimportancesampling_vbmc.m, lines 257 to 266.

    active_is["ln_weights"] = renormalize_weights(active_is["ln_weights"])
    return active_is


def active_sample_proposal_pdf(Xa, gp, vp_is, w_vp, rect_delta, acq_fcn):
    r"""Compute importance weights for proposal pdf.

    Parameters
    ----------
    Xa : np.ndarray
        Array of ``N`` importance sampling evaluation points, of shape ``(N, D)``
        where ``D`` is the problem dimension.
    gp : gpyreg.GP
        The GP object.
    vp_is : VariationalPosterior
        The smoothed VP for importance sampling.
    w_vp : float
        The ratio of VP samples: ``n_vp / (n_vp + n_box)``.
    rect_delta : np.ndarray
        The half-widths (in each dimension) of the rectangle used for
        box-uniform sampling.
    acq_fcn : AbstractAcqFcn
        The acquisition function callable.
    vp : VariationalPosterior
        The unsmoothed VP.
    Returns
    -------
    ln_weights : np.ndarray
        The log importance weights, of shape ``(N, Ns_gp)`` where ``Ns_gp`` is the
        number of GP posterior hyperparameter samples.
    f_s2 : np.ndarray
        The predicted GP variance at the importance sampling points, of shape
        ``(N, Ns_gp)`` where ``Ns_gp`` is the number of GP posterior hyperparameter
        samples.
    """
    N, D = gp.X.shape
    Na = Xa.shape[0]

    f_mu, f_s2 = gp.predict(Xa, separate_samples=True)

    Ntot = 1 + N  # Total number of mixture elements

    if w_vp < 1:
        temp_lpdf = np.zeros((Na, Ntot))
    else:
        temp_lpdf = np.zeros((Na, 1))

    # Mixture of variational posteriors
    if w_vp > 0:
        temp_lpdf[:, 0] = vp_is.pdf(
            Xa, orig_flag=False, log_flag=True
        ).T + np.log(w_vp)
    else:
        temp_lpdf[:, 0] = -np.inf

    # Fixed log weight for importance sampling (log fixed integrand)
    ln_y = acq_fcn.is_log_base(Xa, f_mu=f_mu, f_s2=f_s2)

    # Mixture of box-uniforms
    if w_vp < 1:
        VV = np.product(2 * rect_delta)

        for i in range(N):
            mask = np.all(np.abs(Xa[:] - gp.X[i, :]) < rect_delta, axis=1)
            temp_lpdf[mask, i + 1] = np.log((1 - w_vp) / VV / N)
            temp_lpdf[~mask, i + 1] = -np.inf

        m_max = np.amax(temp_lpdf, axis=1)
        if np.any(m_max == -np.inf):
            raise ValueError("Invalid value.")
        l_pdf = np.log(
            np.sum(np.exp(temp_lpdf - m_max.reshape(-1, 1)), axis=1)
        )
        ln_weights = ln_y - (l_pdf + m_max).reshape(-1, 1)
    else:
        ln_weights = ln_y - temp_lpdf

    return ln_weights, f_s2


def get_mcmc_opts(Ns, thin=1, burn_in=None):
    r"""Get standard MCMC options.

    Parameters
    ----------
    Ns : int, optional
        The number of MCMC samples to return, after thinning and burn-in.
    thin : int, optional
        The thinning parameter will omit ``thin-1`` out of ``thin`` values in
        the generated sequence (after burn-in). Default 1.
    burn_in : int, optional
        The burn-in omits the first ``burn_in`` points before returning any
        samples. Default ``thin * Ns / 2``.

    Returns
    -------
    sampler_opts : dict
        The default sampler options: ``"display" : off`` and
        ``diagnostics : False''.
    thin : int
        The selected value for ``thin``.
    burn_in : int
        The selected value for ``burn_in``.
    """
    sampler_opts = {}
    if burn_in is None:
        burn_in = ceil(thin * Ns / 2)
    sampler_opts["display"] = "off"
    sampler_opts["diagnostics"] = False

    return sampler_opts, thin, burn_in


def fess(vp, gp, X=100):
    r"""Compute fractional effective sample size through importance sampling.

    Parameters
    ----------
    vp : VariationalPosterior
        The ``VP`` object for calculating importance sampling weights.
    gp : gpr.GP or np.ndarray, shape(N, Ns_gp)
        The ``GP`` object for calculating the average posterior mean, or an
        ``ndarray`` of separate GP posterior means, over ``Ns_gp``
        hyperparameter samples.
    X : np.ndarray(N, D) or int
        The domain points at which to calculate the VP pdf for importance
        sampling weights, or an integer number of samples to draw from the
        VP for the same purpose.

    Returns
    -------
    fESS : float
        The estimated fractional Effective Samples Size.

    Raises
    ------
    ValueError
        If the number of provided GP means does not match the number of VP
        samples.
    """
    # If a single number is passed, interpret it as the number of samples
    if np.isscalar(X):
        N = X
        X = vp.sample(N, orig_flag=False)
    else:
        N = X.shape[0]

    # Can pass the GP, or the GP means directly:
    if isinstance(gp, gpr.GP):
        f_bar, __ = gp.predict(X)
        f_bar = f_bar.ravel()
    else:
        f_bar = np.mean(gp, axis=1)

    if f_bar.shape[0] != X.shape[0]:
        raise ValueError("Mismatch between number of samples from VP and GP.")

    # Compute effective sample size (ESS) with importance sampling
    v_ln_pdf = np.maximum(
        vp.pdf(X, orig_flag=False, log_flag=True), np.log(sys.float_info.min)
    ).ravel()
    ln_weights = f_bar - np.atleast_2d(v_ln_pdf)
    weight = np.exp(ln_weights - np.amax(ln_weights))
    weight = weight / np.sum(weight)

    return (1 / np.sum(weight**2)) / N  # Fractional ESS


def renormalize_weights(ln_w):
    M = np.amax(ln_w)
    return ln_w - (M + np.log(np.sum(np.exp(ln_w - M))))
