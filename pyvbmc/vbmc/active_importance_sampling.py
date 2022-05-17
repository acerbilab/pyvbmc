import numpy as np
import copy
import math
import sys
import gpyreg as gpr

def active_importance_sampling(vp, gp, acqfcn, options):
    """
    Setup importance sampling acquisition functions (viqr/imiqr).

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
    acqfcn : AbstractAcqFcn
        The acquisition function.

    Returns
    -------
    active_is : dict
        A dictionary of IS statistics and bookkeeping.
    """

    # Does the importance sampling step use the variational posterior?
    isample_vp_flag = getattr(acqfcn, "importance_sampling_vp", False)
    # Do we simply sample from the variational posterior?
    only_vp_flag = getattr(acqfcn, "variational_importance_sampling", False)

    D = gp.X.shape[1]
    Ns_gp = len(gp.posteriors)  # Number of gp hyperparameter samples

    # Input space bounds and typical scales (for MCMC only)
    widths = np.std(gp.X, axis=0, ddof=1)
    max_bnd = 0.5
    diam = np.amax(gp.X, axis=0) - np.amin(gp.X, axis=0)
    LB = np.amin(gp.X, axis=0) - max_bnd * diam
    UB = np.amax(gp.X, axis=0) + max_bnd * diam

    active_is = {}
    active_is["ln_weights"] = None
    active_is["X"] = None
    active_is["f_s2"] = None

    if only_vp_flag:
        # Step 0: Simply sample from variational posterior.

        Na = options["activeimportancesamplingmcmcsamples"]
        Xa, __ = vp.sample(Na, origflag=False)

        f_mu, f_s2 = gp.predict(Xa, separate_samples=True)

        if hasattr(acqfcn, "mcmc_importance_sampling")\
           and acqfcn.mcmc_importance_sampling:
            # Compute fractional effective sample size (ESS)
            fESS = fess_vbmc(vp, f_mu, Xa)

            if fESS < options["activeimportancesamplingfessthresh"]:
                Xa_old = copy.deepcopy(Xa)

                if isample_vp_flag:
                    log_p_fun = lambda x : log_isbasefun(x, acqfcn, gp, vp)
                else:
                    log_p_fun = lambda x : log_isbasefun(x, acqfcn, gp, None)

                # Get MCMC options
                Nmcmc_samples = Na * options["activeimportancesamplingmcmcthin"]
                thin = 1
                burn_in = 0
                sampler_opts, __, __ = get_mcmc_opts()
                W = Na # walkers, not used (see below).

                # Perform a single MCMC step for all samples.
                # Contrary to MATLAB, we are using simple slice sampling.
                # Better (e.g. ensemble slice) sampling methods could
                # later be implemented.
                sampler = gpr.slice_sample.SliceSampler(log_p_fun, Xa, widths, LB, UB, sampler_opts)
                results = sampler.sample(Nmcmc_samples, thin, burn_in)
                Xa = results["samples"]
                # Xa = eis_sample_lite(log_p_fun, Xa, Nmcmc_samples, W, widths, LB, UB, sample_opts)
                Xa = Xa[-Na:, :]
                f_mu, f_s2 = gp.predict(Xa, separate_samples=True)

        if isample_vp_flag:
            v_ln_pdf = np.maximum(vp.pdf(Xa, origflag=False, logflag=True),
                                  np.log(np.finfo(np.float64).min))
            ln_y = acqfcn.is_log_f1(v_ln_pdf, f_mu, f_s2)
        else:
            ln_y = acqfcn.is_log_f1(None, f_mu, f_s2)

        active_is["f_s2"] = f_s2
        active_is["ln_weights"] = ln_y.T
        active_is["X"] = Xa

    else:
        # Step 1: Importance sampling-resampling

        Nvp_samples = options["activeimportancesamplingvpsamples"]
        Nbox_samples = options["activeimportancesamplingboxsamples"]
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
                vp_is.sigma = np.hstack((vp_is.sigma,
                                        vp.sigma**2 + scale_vec[i]**2))
            vp_is.w = vp_is.w / np.sum(vp_is.w)

            # Sample from smoothed posterior
            Xa_vp, __ = vp_is.sample(Nvp_samples, origflag=False)
            ln_weights, f_s2a_vp = activesample_proposalpdf(Xa_vp, gp, vp_is, w_vp, rect_delta, acqfcn, vp, isample_vp_flag)
            if active_is.get("ln_weights") is None:
                active_is["ln_weights"] = ln_weights.T
            else:
                active_is["ln_weights"] = np.append(active_is["ln_weights"], ln_weights.T, axis=1)
            if active_is.get("X") is None:
                active_is["X"] = Xa_vp
            else:
                active_is["X"] = np.append(active_is["X"], Xa_vp, axis=0)
            if active_is.get("f_s2") is None:
                active_is["f_s2"] = f_s2a_vp
            else:
                active_is["f_s2"] = np.append(active_is["f_s2"], f_s2a_vp, axis=0)
        else:
            vp_is = None

        # Box-uniform sampling around training inputs
        if Nbox_samples > 0:
            jj = np.random.randint(0, len(gp.X), size=(Nbox_samples,))
            Xa_box = gp.X[jj, :] + (2 * np.random.rand(jj.size, D) - 1) * rect_delta
            ln_weights, f_s2a_box = activesample_proposalpdf(Xa_box, gp, vp_is, w_vp, rect_delta, acqfcn, vp, isample_vp_flag)
            if active_is.get("ln_weights") is None:
                active_is["ln_weights"] = ln_weights.T
            else:
                active_is["ln_weights"] = np.append(active_is["ln_weights"], ln_weights.T, axis=1)
            if active_is.get("X") is None:
                active_is["X"] = Xa_box
            else:
                active_is["X"] = np.append(active_is["X"], Xa_box, axis=0)
            if active_is.get("f_s2") is None:
                active_is["f_s2"] = f_s2a_box
            else:
                active_is["f_s2"] = np.append(active_is["f_s2"], f_s2a_box, axis=0)

        active_is["ln_weights"][~np.isfinite(active_is["ln_weights"])] = -np.inf

        # Step 2 (optional): MCMC sample

        Nmcmc_samples = options["activeimportancesamplingmcmcsamples"]

        if Nmcmc_samples > 0:
            active_is_old = copy.deepcopy(active_is)

            active_is["ln_weights"] = np.zeros((Ns_gp, Nmcmc_samples))
            active_is["X"] = np.zeros((Nmcmc_samples, D, Ns_gp))
            active_is["f_s2"] = np.zeros((Nmcmc_samples, Ns_gp))

            # Consider only one GP sample at a time
            gp1 = copy.deepcopy(gp)
            for s in range(Ns_gp):
                gp1.posteriors = np.array([gp.posteriors[s]])  # Assign current GP sample

                # quasi-random grid for D <= 2, not implemented
                # See activeimportancesampling_vbmc.m, lines 159 to 165.

                if isample_vp_flag:
                    log_p_fun = lambda x : log_isbasefun(x, acqfcn, gp1, vp)
                else:
                    log_p_fun = lambda x : log_isbasefun(x , acqfcn, gp1, None)

                # Get MCMC Options
                thin = options["activeimportancesamplingmcmcthin"]
                burn_in = math.ceil(thin * Nmcmc_samples/2)
                sampler_opts, __, __ = get_mcmc_opts(Nmcmc_samples)

                # Not used (see comment below regarding simple slice sampling.)
                # Walkers = 2 * (D + 1)
                Walkers = 1

                # Use importance sampling-resampling
                f_mu, f_s2 = gp1.predict(active_is_old["X"], separate_samples=True)
                ln_weights = active_is_old["ln_weights"][s, :].reshape(-1, 1)\
                       + acqfcn.is_log_f2(f_mu, f_s2)
                ln_weights_max = np.amax(ln_weights, axis=1).reshape(-1, 1)
                assert np.all(ln_weights_max != -np.inf)
                w = np.exp(ln_weights - ln_weights_max).reshape(-1)
                w = w / np.sum(w)
                # x0 = np.zeros((Walkers, D))
                # Select without replacement by weight w:
                index = np.random.choice(a=len(w), p=w, replace=False)
                x0 = active_is_old["X"][index, :]
                x0 = np.maximum(np.minimum(x0, UB), LB)  # Force inside bounds

                # Contrary to MATLAB, we are using simple slice sampling.
                # Better (e.g. ensemble slice) sampling methods could
                # later be implemented.
                sampler = gpr.slice_sample.SliceSampler(log_p_fun, x0, widths, LB, UB, sampler_opts)
                results = sampler.sample(Nmcmc_samples, thin, burn_in)
                Xa, log_p = results["samples"], results["f_vals"]
                # Xa, log_p = eis_sample_lite(log_p_fun, x0, Nmcmc_samples, W, widths, LB, UB, sample_opts)
                f_mu, f_s2 = gp1.predict(Xa, separate_samples=True)

                # Fixed log weight for importance sampling (log fixed integrand)
                if isample_vp_flag:
                    v_ln_pdf = np.maximum(vp.pdf(Xa, origflag=False, logflag=True),
                                          np.log(sys.float_info.min))
                    ln_y = acqfcn.is_log_f1(v_ln_pdf, f_mu, f_s2)
                else:
                    ln_y = acqfcn.is_log_f1(None, f_mu, f_s2)

                active_is["f_s2"][:, s] = f_s2.reshape(-1)
                active_is["ln_weights"][s, :] = ln_y.T - log_p.T
                active_is["X"][:, :, s] = Xa

    # Step 3: Pre-compute quantities for importance sampling calculations:

    # Pre-compute cross-kernel matrix on importance points
    K_Xa_X = np.zeros(
        (active_is["X"].shape[0], gp.X.shape[0], Ns_gp)
        )
    for s in range(Ns_gp):
        if active_is["X"].ndim == 3:
            Xa = active_is["X"][:, :, s]
        else:
            Xa = active_is["X"]
        cov_N = gp.covariance.hyperparameter_count(gp.D)
        hyp = gp.posteriors[s].hyp[0:cov_N]  # just covariance hyperparameters
        if isinstance(gp.covariance,
                      gpr.covariance_functions.SquaredExponential):
            K_Xa_X[:, :, s] = gp.covariance.compute(hyp, Xa, gp.X)
        else:
            raise ValueError("Covariance functions besides" ++
                             "SquaredExponential are not supported yet.")
    active_is["K_Xa_X"] = K_Xa_X

    # Omitted port, integrated mean functions:
    # activeimportancesampling_vbmc.m, lines 257 to 266.

    return active_is


def activesample_proposalpdf(Xa, gp, vp_is, w_vp, rect_delta, acqfcn, vp, isample_vp_flag):
    r"""Compute importance weights for proposal pdf.

    Parameters
    ----------

    Returns
    -------

    """
    N, D = gp.X.shape
    Na = Xa.shape[0]

    f_mu, f_s2 = gp.predict(Xa, separate_samples=True)

    Ntot = 1 + N  # Total number of mixture elements

    if w_vp < 1:
        temp_lpdf = np.zeros((Na, Ntot))

    # Mixture of variational posteriors
    if w_vp > 0:
        temp_lpdf[:, 0] = vp_is.pdf(Xa, origflag=False, logflag=True).T
    else:
        temp_lpdf[:, 0] = -np.inf

    # Fixed log weight for importance sampling (log fixed integrand)
    if isample_vp_flag:
        v_ln_pdf = np.maximum(vp.pdf(Xa, origflag=False, logflag=True),
                              np.log(sys.float_info.min))
        ln_y = acqfcn.is_log_f1(v_ln_pdf, f_mu, f_s2)
    else:
        ln_y = acqfcn.is_log_f1(None, f_mu, f_s2)

    # Mixture of box-uniforms
    if w_vp < 1:
        VV = np.product(2 * rect_delta)

        for i in range(N):
            temp_lpdf[:, i+1] = np.log(
                (1 - w_vp) * np.all(
                    np.abs(Xa - gp.X[i, :]) < rect_delta, axis=1
                ) / VV / N
            )

        m_max = np.amax(temp_lpdf, axis=1)
        assert np.all(m_max != -np.inf)
        l_pdf = np.log(np.sum(np.exp(temp_lpdf - m_max.reshape(-1, 1)),
                              axis=1))
        ln_weights = ln_y - (l_pdf + m_max).reshape(-1, 1)
    else:
        ln_weights = ln_y - temp_lpdf

    return ln_weights, f_s2


def log_isbasefun(x, acq_fcn, gp, vp=None):
    r"""Base importance sampling proposal log pdf.

    Parameters
    ----------
    x : np.ndarray, shape (N, D)
        The points at which to evaluate the log pdf of the proposal.
    acq_fcn : AbstractAcqFcn
        The acquisition function (must implement ``is_log_f`` method which
        evaluates the log pdf.
    gp : gpr.GP
        The GP surrogate.
    vp : VariationalPosterior, optional
        The current VP. If ``None``, no importance sampling weights
        will be added to the log pdf.

    Returns
    -------
    is_log_f : np.ndarray, shape (N,)
        The proposal log pdf evaluated at the input points.
    """

    f_mu, f_s2 = gp.predict(x.reshape(1, -1))

    if vp is None:
        return acq_fcn.is_log_f1(0, f_mu, f_s2)
    else:
        v_ln_pdf = np.maximum(vp.pdf(x, origflag=False, logflag=True),
                              np.log(sys.float_info.min))
        return acq_fcn.is_log_f(v_ln_pdf, f_mu, f_s2)


def get_mcmc_opts(Ns=100, thin=1, burn_in=None):
    r"""Get standard MCMC options.

    Parameters
    ----------
    Ns : int, optional
        The number of MCMC samples to return, after thinning and burn-in.
        Default 100.
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
        burn_in = math.ceil(thin * Ns / 2)
    sampler_opts["display"] = 'off'
    sampler_opts["diagnostics"] = False

    return sampler_opts, thin, burn_in


def fess_vbmc(vp, gp, X=100):
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
    """

    # If a single number is passed, interpret it as the number of samples
    if np.isscalar(X):
        N = X
        X = vp.sample(N, origflag=False)
    else:
        N = X.shape[0]

    # Can pass the GP, or the GP means directly:
    if isinstance(gp, gpr.GP):
        fbar, __ = gp.predict(X)
    else:
        fbar = np.mean(gp, axis=1)

    if fbar.shape[0] != X.shape[0]:
        raise ValueError("Mismatch between number of samples from VP and GP.")

    # Compute effective sample size (ESS) with importance sampling
    v_ln_pdf = np.maximum(vp.pdf(X, origflag=False, logflag=True),
                          np.log(sys.float_info.min))
    ln_weights = fbar - v_ln_pdf
    weight = np.exp(ln_weights - np.amax(ln_weights, axis=1))
    weight = weight / sum(weight)

    return (1 / sum(weight**2)) / N  # Fractional ESS
