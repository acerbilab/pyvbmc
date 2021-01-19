"""
This is a perliminary version of the VBMC loop in order to identify possible objects
"""


import vbmc
from gaussianProcess import gplite
from math import sqrt

gp = None
hypstruct = None
stats = None


optimState = dict()
options = dict()

loopiter = 0
finished = False

while not finished:

    # t_iter = tic;
    # timer = timer_init();   #Initialize iteration timer

    loopiter += 1
    optimState.iter = loopiter
    vp_old = vp
    action = ""
    optimState.redoRotoscaling = False

    if loopiter == 1 and optimState.Warmup:
        action = "start warm-up"

    # Switch to stochastic entropy towards the end if still on deterministic
    if (
        optimState.EntropySwitch
        and optimState.funccount
        >= options.EntropyForceSwitch * options.MaxFunEvals
    ):
        optimState.EntropySwitch = False
        if not action:
            action = "entropy switch"
        else:
            action += "entropy switch"  # [action ', entropy switch']

    """
    Actively sample new points into the training set
    """

    # t = tic;
    # optimState.trinfo = vp.trinfo;
    # if loopiter == 1; new_funevals = options.FunEvalStart; else; new_funevals = options.FunEvalsPerIter; end
    # if optimState.Xn > 0
    #    optimState.ymax = max(optimState.y(optimState.X_flag));
    # end

    if optimState.SkipActiveSampling:
        optimState.SkipActiveSampling = False
    else:
        if gp and options.SeparateSearchGP and not options.VarActiveSample:
            # Train a distinct GP for active sampling
            if loopiter % 2 == 0:
                meantemp = optimState.gpMeanfun
                optimState.gpMeanfun = "const"
                gp_search, hypstruct_search = vbmc.__2gptrain_vbmc(
                    hypstruct_search, optimState, stats, options
                )
                optimState.gpMeanfun = meantemp
            else:
                gp_search = gp
        else:
            gp_search = gp

        # Perform active sampling
        if options.VarActiveSample:
            # FIX TIMER HERE IF USING THIS
            # [optimState,vp,t_active,t_func] = variationalactivesample_vbmc(optimState,new_funevals,funwrapper,vp,vp_old,gp_search,options)
            print("Function currently not supported")
            exit(1)
        else:
            optimState.hypstruct = hypstruct
            optimState, vp, gp, timer = vbmc.__1activesample_vbmc(
                optimState,
                new_funevals,
                funwrapper,
                vp,
                vp_old,
                gp_search,
                stats,
                timer,
                options,
            )
            hypstruct = optimState.hypstruct

    # optimState.N = optimState.Xn;  % Number of training inputs
    # optimState.Neff = sum(optimState.nevals(optimState.X_flag));

    """
    Train GP
    """

    # t = tic
    gp, hypstruct, Ns_gp, optimState = vbmc.__2gptrain_vbmc(
        hypstruct, optimState, stats, options
    )
    # timer.gpTrain = timer.gpTrain + toc(t)

    # Check if reached stable sampling regime
    if Ns_gp == options.StableGPSamples and optimState.StopSampling == 0:
        optimState.StopSampling = optimState.N

    # Estimate of GP noise around the top high posterior density region
    # optimState.sn2hpd = estimate_GPnoise(gp)

    """
    Optimize variational parameters
    """

    # t = tic;

    if not vp.optimize_mu:
        # Variational components fixed to training inputs
        # vp.mu = gp.X'
        # Knew = size(vp.mu,2)
        print("")
    else:
        # Update number of variational mixture components
        Knew = vbmc.__3updateK(optimState, stats, options)

    # Decide number of fast/slow optimizations
    Nfastopts = ceil(evaloption_vbmc(options.NSelbo, K))

    if optimState.RecomputeVarPost or options.AlwaysRefitVarPost:
        # Full optimizations
        Nslowopts = options.ElboStarts
        optimState.RecomputeVarPost = False
    else:
        # Only incremental change from previous iteration
        Nfastopts = ceil(Nfastopts * options.NSelboIncr)
        Nslowopts = 1

    # Run optimization of variational parameters
    if optimState.Warmup and options.BOWarmup:
        vp_fields = {"elbo", "elbo_sd", "G", "H", "varG", "varH"}
        for i in len(vp_fields):  # i = 1:numel(vp_fields)
            vp.stats.vp_fields[i] = None
            # vp.stats.(vp_fields{i}) = NaN
        varss = None
        pruned = 0
    else:
        [vp, varss, pruned] = vbmc.__3vpoptimize_vbmc(
            Nfastopts, Nslowopts, vp, gp, Knew, optimState, options, prnt
        )
        # optimState.vp_repo{end+1} = get_vptheta(vp)

    optimState.vpK = vp.K
    optimState.H = vp.stats.entropy  # Save current entropy

    # Get real variational posterior (might differ from training posterior)
    vp_real = vp.vptrain2real(0, options)
    elbo = vp_real.stats.elbo
    elbo_sd = vp_real.stats.elbo_sd

    # timer.variationalFit = timer.variationalFit + toc(t)

    """
    Finalize iteration
    """

    # t = tic;

    # Compute symmetrized KL-divergence between old and new posteriors
    Nkl = 1e5
    sKL = max(0, 0.5 * sum(vbmc_kldiv(vp, vp_old, Nkl, options.KLgauss)))

    # Evaluate max LCB of GP prediction on all training inputs
    _, _, fmu, fs2 = gplite.gplite_pred(gp, gp.X, gp.y, gp.s2)
    optimState.lcbmax = max(fmu - options.ELCBOImproWeight * sqrt(fs2))

    if options.AdaptiveEntropyAlpha:
        # Evaluate deterministic entropy
        Hl = vp.entlb_vbmc(0, 0)
        Hu = vp.entub_vbmc(0, 0)
        optimState.entropy_alpha = max(
            0, min(1, (vp.stats.entropy - Hl) / (Hu - Hl))
        )
        # optimState.entropy_alpha

    # Compare variational posterior's moments with ground truth
    if (
        options.TrueMean and options.TrueCov
    ):  # and all(isfinite(options.TrueMean(:)))&& all(isfinite(options.TrueCov(:)))
        mubar_orig, Sigma_orig = vp_real.vbmc_moments(1, 1e6)
        # [kl(1),kl(2)] = mvnkl(mubar_orig,Sigma_orig,options.TrueMean,options.TrueCov)
        sKL_true = 0.5 * sum(kl)
    else:
        sKL_true = None

    # Record moments in transformed space
    mubar, Sigma = vp.moments(0)
    if not optimState.RunMean or not optimState.RunCov:
        # optimState.RunMean = mubar(:)
        optimState.RunCov = Sigma
        optimState.LastRunAvg = optimState.N
    else:
        Nnew = optimState.N - optimState.LastRunAvg
        wRun = options.MomentsRunWeight ^ Nnew
        # optimState.RunMean = wRun*optimState.RunMean + (1-wRun)*mubar(:)
        optimState.RunCov = wRun * optimState.RunCov + (1 - wRun) * Sigma
        optimState.LastRunAvg = optimState.N

    # timer.finalize = toc(t);
    # timer.totalruntime = NaN;   # Update at the end of iteration
    # timer

    # Record all useful stats
    stats = savestats(
        stats,
        optimState,
        vp,
        elbo,
        elbo_sd,
        varss,
        sKL,
        sKL_true,
        gp,
        hypstruct.full,
        Ns_gp,
        pruned,
        timer,
        options.Diagnostics,
    )

    """
    Check termination conditions and warmup
    """

    (
        optimState,
        stats,
        isFinished_flag,
        exitflag,
        action,
        msg,
    ) = vbmc.__4vbmc_termination(optimState, action, stats, options)
    vp.stats.stable = stats.stable(optimState.iter)  # Save stability

    # Check if we are still warming-up
    if optimState.Warmup and loopiter > 1:
        if options.RecomputeLCBmax:
            optimState.lcbmax_vec = recompute_lcbmax(
                gp, optimState, stats, options
            )  #'
        optimState, action, trim_flag = vbmc.__4vbmc_warmup(
            optimState, stats, action, options
        )
        if trim_flag:
            # Re-update GP after trimming
            gp = gpreupdate(gp, optimState, options)
        if not optimState.Warmup:
            vp.optimize_mu = logical(options.VariableMeans)
            vp.optimize_weights = logical(options.VariableWeights)
            if options.BOWarmup:
                optimState.gpMeanfun = options.gpMeanFun
                hypstruct.hyp = []

            # Switch to main algorithm options
            options = options_main
            hypstruct.runcov = []  # Reset GP hyperparameter covariance
            optimState.vp_repo = []  # Reset VP repository
            optimState.acqInfo = getAcqInfo(
                options.SearchAcqFcn
            )  # Re-get acq info

    stats.warmup[loopiter] = optimState.Warmup

    # Check and update fitness shaping / output warping threshold
    if optimState.OutwarpDelta and optimState.R < options.WarpTolReliability:
        Xrnd = vp.vbmc_rnd(2e4, 0)
        ymu = gp.gplite_pred(gp, Xrnd, [], [], 0, 1)
        ydelta = max([0, optimState.ymax - quantile(ymu, 1e-3)])
        if (
            ydelta > optimState.OutwarpDelta * options.OutwarpThreshTol
            and optimState.R < 1
        ):
            optimState.OutwarpDelta = (
                optimState.OutwarpDelta * options.OutwarpThreshMult
            )

    if options.AcqHedge:
        # Update hedge values
        optimState.hedge = acqhedge_vbmc(
             "upd", optimState.hedge, stats, options 
                    )

    """
    Write iteration output
    """
   
    #Stopped GP sampling this iteration?
    if(Ns_gp == options.StableGPSamples and stats.gpNsamples(max(1,loopiter-1)) > options.StableGPSamples):
        if(Ns_gp == 0):
            if (isempty(action)):
                    action = 'switch to GP opt'
            else:
                action = [action ', switch to GP opt']
        else:
            if(isempty(action)):
                action = 'stable GP sampling'
            else:
                action = [action ', stable GP sampling']
    
    if(prnt > 2):
        if(options.BOWarmup and optimState.Warmup):
            print(displayFormat_warmup,loopiter,optimState.funccount,max(optimState.y_orig),action)            
        else:
            if(optimState.Cache.active):
                print(displayFormat,loopiter,optimState.funccount,optimState.cachecount,elbo,elbo_sd,sKL,vp.K,optimState.R,action)
            elif(optimState.UncertaintyHandlingLevel > 0  and options.MaxRepeatedObservations > 0):
                print(displayFormat,loopiter,optimState.funccount,optimState.N,elbo,elbo_sd,sKL,vp.K,optimState.R,action)              
            else:
                print(displayFormat,loopiter,optimState.funccount,elbo,elbo_sd,sKL,vp.K,optimState.R,action)
    
    stats.timer(loopiter).totalruntime = toc(t0)
    
vp_old = vp

"""
Pick "best" variational solution to return (and real vp, if train vp differs)    
"""

vp,elbo,elbo_sd,idx_best = vbmc.__5best_vbmc(stats,loopiter,options.BestSafeSD,options.BestFracBack,options.RankCriterion,0)
#new_final_vp_flag = idx_best ~= loopiter
gp = stats.gp(idx_best)
vp.gp = gp; #Add GP to variational posterior

"""
Last variational optimization with large number of components
"""

vp,elbo,elbo_sd,changedflag = vbmc.__5finalboost_vbmc(vp,idx_best,optimState,stats,options)
if(changedflag):
    new_final_vp_flag = True

if(new_final_vp_flag and prnt > 2):
    #Recompute symmetrized KL-divergence
    sKL = max(0,0.5*sum(vbmc_kldiv(vp,vp_old,Nkl,options.KLgauss)))

