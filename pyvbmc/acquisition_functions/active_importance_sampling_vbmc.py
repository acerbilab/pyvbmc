import logging
import numpy as np
import copy
import math
import sys
import gpyreg as gpr
from .utilities import sq_dist

def active_importance_sampling_vbmc(vp, gp, acqfcn, options):
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
    isample_vp_flag = hasattr(acqfcn, "importance_sampling_vp")\
        and acqfcn.importance_sampling_vp
    # Do we simply sample from the variational posterior?
    only_vp_flag = hasattr(acqfcn, "variational_importance_sampling")\
        and acqfcn.variational_importance_sampling

    D = gp.X.shape[1]
    Ns_gp = len(gp.posteriors)  # Number of gp hyperparameter samples

    # Input space bounds and typical scales (for MCMC only)
    widths = np.std(gp.X, axis=0, ddof=1)
    max_bnd = 0.5
    diam = np.amax(gp.X, axis=0) - np.amin(gp.X, axis=0)
    LB = np.amin(gp.X, axis=0) - max_bnd * diam
    UB = np.amax(gp.X, axis=0) + max_bnd * diam

    active_is = {}
    active_is["log_weight"] = None
    active_is["Xa"] = None
    active_is["f_s2a"] = None

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
                sampler = gpr.SliceSampler(log_p_fun, Xa, widths, LB, UB, sampler_opts)
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

        active_is["f_s2a"] = f_s2
        active_is["ln_w"] = ln_y.T
        active_is["Xa"] = Xa

    else:
        # Step 1: Importance sampling-resampling

        Nvp_samples = options["activeimportancesamplingvpsamples"]
        Nbox_samples = options["activeimportancesamplingboxsamples"]
        w_vp = Nvp_samples / (Nvp_samples + Nbox_samples)

        rect_delta = 2 * np.std(gp.X, ddof=1)

        # Smoothed posterior for importance sampling-resampling
        if Nvp_samples > 0:
            scale_vec = np.array([0.05, 0.2, 1.0])

            vp_is = copy.deepcopy(vp)
            for i in range(len(scale_vec)):
                vp_is.K = vp_is.K + vp.K
                vp_is.w = np.append(vp_is.w, vp.w)
                vp_is.mu = np.append(vp_is.mu, vp.mu)
                vp_is.sigma = np.append(vp_is.sigma, vp.sigma)
            vp_is.w = vp_is.w / np.sum(vp_is.w)

            # Sample from smoothed posterior
            Xa_vp = vp_is.sample(Nvp_samples, origflag=False)
            ln_w, f_s2a_vp = activesample_proposalpdf(Xa_vp, gp, vp_is, w_vp, rect_delta, acqfcn, vp, isample_vp_flag)
            active_is["ln_w"] = np.append(active_is["ln_w"], ln_w.T)
            active_is["Xa"] = np.append(active_is["Xa"], Xa_vp)
            active_is["f_s2a"] = np.append(active_is["f_s2a"], f_s2a_vp)
        else:
            vp_is = None

        # Box-uniform sampling around training inputes
        if Nbox_samples > 0:
            jj = np.random.randint(0, len(gp.X), size=(1, Nbox_samples))
            Xa_box = gp.X[jj, :] + (2 * np.random.rand(jj.size, D) - 1) * rect_delta
            ln_w, f_s2a_box = activesample_proposalpdf(Xa_box, vp_is, w_vp, rect_delta, acqfcn, vp, isample_vp_flag)
            active_is["ln_w"] = np.append(active_is["ln_w"], ln_w.T)
            active_is["Xa"] = np.append(active_is["Xa"], Xa_box)
            active_is["f_s2a"] = np.append(active_is["f_s2a"], f_s2a_box)

        active_is["ln_w"][~np.isfinite(active_is["ln_w"])] = -np.inf

        # Step 2 (optional): MCMC sample

        Nmcmc_samples = options["activeimportancesamplingmcmcsamples"]

        if Nmcmc_samples > 0:
            active_is_old = copy.deepcopy(active_is)

            active_is["ln_w"] = np.zeros((Ns_gp, Nmcmc_samples))
            active_is["Xa"] = np.zeros((Nmcmc_samples, D, Ns_gp))
            active_is["f_s2a"] = np.zeros((Nmcmc_samples, Ns_gp))

            # Consider only one GP sample at a time
            gp1 = copy.deepcopy(gp)
            for s in range(Ns_gp):
                gp1.post = gp.post[s]  # Assign current GP sample

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
                Walkers = 2 * (D + 1)

                # Use importance sampling-resampling
                f_mu, f_s2 = gp.predict(active_is_old["Xa"], separate_samples=True)
                ln_w = active_is_old["ln_w"]
                w = np.exp(ln_w - np.amax(ln_w, axis=1))
                x0 = np.zeros(W, D)
                # Select without replacement by weight w:
                indices = np.random.choice(a=len(w), p=w, replace=False)
                x0[indices, :] = active_is_old["Xa"][indices, :]

                # Contrary to MATLAB, we are using simple slice sampling.
                # Better (e.g. ensemble slice) sampling methods could
                # later be implemented.
                sampler = gpr.SliceSampler(log_p_fun, x0, widths, LB, UB, sampler_opts)
                results = sampler.sample(Nmcmc_samples, thin, burn_in)
                Xa, log_p = results["samples"], results["f_vals"]
                # Xa, log_p = eis_sample_lite(log_p_fun, x0, Nmcmc_samples, W, widths, LB, UB, sample_opts)
                f_mu, f_s2 = gp.predict(Xa, separate_samples=True)

                # Fixed log weight for importance sampling (log fixed integrand)
                if isample_vp_flag:
                    v_ln_pdf = np.maximum(vp.pdf(Xa, origflag=False, logflag=True),
                                          np.log(sys.float_info.min))
                    ln_y = acqfcn.is_log_f1(v_ln_pdf, f_mu, f_s2)
                else:
                    ln_y = acqfcn.is_log_f1(None, f_mu, f_s2)

                active_is["f_s2a"][:, s] = f_s2
                active_is["ln_w"][s, :] = ln_y.T - log_p.T
                active_is["Xa"][:, :, s] = Xa

    # Step 3: Pre-compute quantities for importance sampling calculations:

    # Pre-compute cross-kernel matrix on importance points
    Kax_mat = np.zeros(
        (active_is["Xa"].shape[0], gp.X.shape[0], Ns_gp)
        )
    for s in range(Ns_gp):
        if len(active_is["Xa"].shape) == 2:
            Xa = active_is["Xa"]
        else:
            Xa = active_is["Xa"][:, :, s]
        cov_N = gp.covariance.hyperparameter_count(gp.D)
        hyp = gp.posteriors[s].hyp[0:cov_N]  # just covariance hyperparameters
        if isinstance(gp.covariance,
                      gpr.covariance_functions.SquaredExponential):
            Kax_mat[:, :, s] = gp.covariance.compute(hyp, Xa, gp.X)
        else:
            raise ValueError("Covariance functions besides" ++
                             "SquaredExponential are not supported yet.")
    active_is["Kax_mat"] = Kax_mat

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
    N, D = gp.Xa.shape
    Na = Xa.shape[0]

    f_mu, f_s2 = gp.predict(Xa, separate_samples=True)

    Ntot = 1 + N  # Total number of mixture elements

    if w_vp < 1:
        temp_lpdf = np.zeros(Na, Ntot)

    # Mixture of variational posteriors
    if w_vp > 0:
        temp_lpdf[:, 0] = vp_is.pdf(Xa, origflag=False, logflag=True)
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

        for ii in range(N):
            temp_lpdf[:, ii+2] = np.log(
                (1 - w_vp) * np.all(
                    np.abs(Xa - gp.X[ii, :]) < rect_delta, axis=1
                ) / VV / N
            )

        m_max = np.amax(temp_lpdf, axis=1)
        l_pdf = np.log(np.sum(np.exp(temp_lpdf - m_max), axis=1))
        ln_w = ln_y - (l_pdf + m_max)
    else:
        ln_w = ln_y - temp_lpdf

    return ln_w, f_s2


def log_isbasefun(x, acq_fcn, gp, vp=None):
    r"""Base importance sampling proposal log pdf.

    Parameters
    ---------

    Returns
    -------
    """

    u = 0.6745  # inverse normal cdf of 0.75
    f_mu, f_s2 = gp.predict(x)
    f_s = np.sqrt(f_s2)

    if vp is None:
        return acq_fcn.is_log_f(0, f_mu, f_s2)
    else:
        v_ln_pdf = np.maximum(vp.pdf(x, origflag=False, logflag=True),
                              np.log(sys.float_info.min))
        return acq_fcn.is_log_f(v_ln_pdf, f_mu, f_s2)


def get_mcmc_opts(Ns=100, thin=1, burn_in=None):
    r"""Get standard MCMC options.

    Parameters
    ---------

    Returns
    -------
    """

    sampler_opts = {}
    if burn_in is None:
        burn_in = math.ceil(thin * Ns / 2)
    sampler_opts["display"] = 'off'
    sampler_opts["diagnostics"] = False

    return sampler_opts, thin, burn_in


def fess_vbmc(vp, gp, X=100):
    r"""Compute fractional effective sample size through importance sampling.
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
    ln_w = fbar - v_ln_pdf
    weight = np.exp(ln_w - np.amax(ln_w, axis=1))
    weight = weight / sum(weight)

    return (1 / sum(weight**2)) / N  # Fractional ESS
