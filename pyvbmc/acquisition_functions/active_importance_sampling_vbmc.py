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
                    logpfun = lambda x : log_isbasefun(x, acqfcn, gp, vp)
                else:
                    logpfun = lambda x : log_isbasefun(x, acqfcn, gp, None)
