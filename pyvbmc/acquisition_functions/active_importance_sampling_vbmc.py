import logging
import numpy as np
import copy

def active_importance_sampling_vbmc(vp, gp, acqfcn, acqinfo, options):
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
    acqfun : AbstractAcqFcn
        The acquisition function.
    acqinfo : dict
        The information of the acquisition function.

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
    Ns_gp = len(gp.post)# Number of gp hyperparameter samples

    # Input space bounds and typical scales (for MCMC only)
    widths = np.std(gp.X, axis=0, ddof=1)
    max_bnd = 0.5
    diam = max(gp.X) - min(gp.X)
    LB = min(gp.X) - max_bnd * diam
    UB = max(gp.X) + max_bnd * diam

    active_is = {}
    active_is["log_weight"] = None
    active_is["Xa"] = None
    active_is["fs2a"] = None

    if only_vp_flag:
        # Step 0: Simply sample from variational posterior.

        Na = options["activeimportancesamplingmcmcsamples"]
        Xa = vp.sample(Na, origflag=False)

        fmu, fs1 = gp.predict(Xa, separate_samples=True)

        if hasattr(acqfcn, "mcmc_importance_sampling")\
           and acqfcn.mcmc_importance_sampling:
            # Compute fractional effective sample size (ESS)
            fESS = fess_vbmc(vp, fmu, Xa)

            if fESS < options["activeimportancesamplingfessthresh"]:
                Xa_old = copy.deepcopy(Xa)

                if isample_vp_flag:
                    log_p_fun = lambda x : log_isbasefun(x, acqfcn, gp, vp)
                else:
                    log_p_fun = lambda x : log_isbasefun(x, acqfcn, gp, None)

                # Get MCMC options
                Nmcmc_samples = Na * options["activeimportancesamplingmcmcthin"]
                thin = 1
                burnin = 0
                sample_opts = get_mcmcopts(None, thin, burnin)
                log_p_funs = log_p_fun
                W = Na # walkers

                # Perform a single MCMC step for all samples
                Xa = eis_sample_lite(log_p_funs, Xa, Nmcmc_samples, W, widths, LB, UB, sample_opts)
                Xa = Xa[-Na:, :]
                fmu, fs2 = gp.predict(Xa, add_noise=True)

        if isample_vp_flag:
            vlnpdf = np.max(vp.pdf(Xa, origflag=False, logflag=True),
                            np.log(np.finfo(np.float64).min))
            lny = acqfun('islogf1', vlnpdf, None, None, fmu, fs2)
        else:
            lny = acqfun('islogf1', None, None, None, fmu, fs2)

        active_is["fs2a"] = fs2
        active_is["lnw"] = lny.T
        active_is["Xa"] = Xa

        return active_is

    else:
        # Step 1: Importance sampling-resampling

        Nvp_samples = options["activeimportancesamplingvpsamples"]
        Nbox_samples = options["activeimportancesamplingboxsamples"]
        w_vp = Nvp_samples / (Nvp_samples + Nbox_samples)

        rect_delta = 2 * std(gp.X, ddof=1)

        # Smoothed posterior for importance sampling-resampling
        if Nvp_samples > 0:
            scale_vec = np.array([0.05, 0.2, 1.0])

            vp_is = copy.deepcopy(vp)
            for i in range(length(scale_vec)):
                vp_is.K = vp_is.K + vp.K
                vp_is.w = np.append(vp_is.w, vp.w)
                vp_is.mu = np.append(vp_is.mu, vp.mu)
                vp_is.sigma = np.append(vp_is.sigma, vp.sigma)
            vp_is.w = vp_is.w / np.sum(vp_is.w)

            # Sample from smoothed posterior
            Xa_vp = vp_is.sample(Nvp_samples, origflag=False)
            lnw, fs2a_vp = activesample_proposalpdf(Xa_vp, gp, vp_is, w_vp, rect_delta, acqfcn, vp, isample_vp_flag)
            active_is["lnw"] = np.append(active_is["lnw"], lnw.T)
            active_is["Xa"] = np.append(active_is["Xa"], Xa_vp)
            active_is["fs2a"] = np.append(active_is["fs2a"], fs2a_vp)
        else:
            vp_is = None

        # Box-uniform sampling around training inputes
        if Nbox_samples > 0:
            jj = np.random.randint(0, len(gp.X), size=(1, Nbox_samples))
            Xa_box = gp.X[jj, :] + (2 * np.random.rand(jj.size, D) - 1) * rect_delta
            lnw, fs2a_box = activesample_proposalpdf(Xa_box, vp_is, w_vp, rect_delta, acqfcn, vp, isample_vp_flag)
            active_is["lnw"] = np.append(active_is["lnw"], lnw.T)
            active_is["Xa"] = np.append(active_is["Xa"], Xa_box)
            active_is["fs2a"] = np.append(active_is["fs2a"], fs2a_box)

        active_is["lnw"][~np.isfinite(active_is["lnw"])] = -np.inf

        # Step 2 (optional): MCMC sample

        Nmcmc_samples = options["activeimportancesamplingmcmcsamples"]

        if Nmcmc_samples > 0:
            active_is_old = copy.deepcopy(active_is)

            active_is["lnw"] = np.zeros((Ns_gp, Nmcmc_samples))
            active_is["Xa"] = np.zeros((Nmcmc_samples, D, Ns_gp))
            active_is["fs2a"] = np.zeros((Nmcmc_samples, Ns_gp))

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
