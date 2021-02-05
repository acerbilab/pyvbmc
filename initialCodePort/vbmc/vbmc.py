import math
from gaussian_process import GP_Lite
from .optimstate_vbmc import OptimState
from .options_vbmc import Options_VBMC
from .timer_vbmc import Timer
from entropy.entlb_vbmc import entlb_vbmc
from entropy.entub_vbmc import entub_vbmc
from .stats_vbmc import Stats
from variational_posterior import VP


class VBMC(object):
    """
    The VBMC algorithm class
    """

    def __init__(self):
        pass

    def algorithm(self, options=Options_VBMC):
        """
        This is a perliminary version of the VBMC loop in order to identify possible objects
        """

        gp = None
        hypstruct = None
        stats = Stats()
        timer = Timer()

        prnt = 1  # configure verbosity of the algorithm

        optimState = OptimState()

        loopiter = 0

        vp = VP()
        isFinished_flag = False

        while not isFinished_flag:

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

            timer.start_timer("activeSampling")
            optimState.trinfo = vp.trinfo
            if loopiter == 1:
                new_funevals = options.FunEvalStart
            else:
                new_funevals = options.FunEvalsPerIter

            if optimState.Xn > 0:
                optimState.ymax = max(optimState.y(optimState.X_flag))

            if optimState.SkipActiveSampling:
                optimState.SkipActiveSampling = False
            else:
                if (
                    gp
                    and options.SeparateSearchGP
                    and not options.VarActiveSample
                ):
                    # Train a distinct GP for active sampling
                    if loopiter % 2 == 0:
                        meantemp = optimState.gpMeanfun
                        optimState.gpMeanfun = "const"
                        gp_search, hypstruct_search = self.__2gptrain_vbmc(
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
                    optimState, vp, gp, timer = self.__1activesample_vbmc(
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

            optimState.N = optimState.Xn  # Number of training inputs
            optimState.Neff = sum(optimState.nevals(optimState.X_flag))

            timer.stop_timer("activeSampling")

            """
            Train GP
            """

            timer.start_timer("gpTrain")

            gp, hypstruct, Ns_gp, optimState = self.__2gptrain_vbmc(
                hypstruct, optimState, stats, options
            )

            timer.stop_timer("gpTrain")

            # Check if reached stable sampling regime
            if (
                Ns_gp == options.StableGPSamples
                and optimState.StopSampling == 0
            ):
                optimState.StopSampling = optimState.N

            """
            Optimize variational parameters
            """

            timer.start_timer("variationalFit")

            if not vp.optimize_mu:
                # Variational components fixed to training inputs
                # vp.mu = gp.X'
                # Knew = size(vp.mu,2)
                print("")
            else:
                # Update number of variational mixture components
                Knew = self.__3updateK(optimState, stats, options)

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
                [vp, varss, pruned] = self.__3vpoptimize_vbmc(
                    Nfastopts,
                    Nslowopts,
                    vp,
                    gp,
                    Knew,
                    optimState,
                    options,
                    prnt,
                )
                # optimState.vp_repo{end+1} = get_vptheta(vp)

            optimState.vpK = vp.K
            optimState.H = vp.stats.entropy  # Save current entropy

            # Get real variational posterior (might differ from training posterior)
            vp_real = vp.vptrain2real(0, options)
            elbo = vp_real.stats.elbo
            elbo_sd = vp_real.stats.elbo_sd

            timer.stop_timer("variationalFit")

            """
            Finalize iteration
            """

            timer.start_timer("finalize")

            # Compute symmetrized KL-divergence between old and new posteriors
            Nkl = 1e5
            sKL = max(
                0, 0.5 * sum(vbmc_kldiv(vp, vp_old, Nkl, options.KLgauss))
            )

            # Evaluate max LCB of GP prediction on all training inputs
            _, _, fmu, fs2 = GP_Lite.gplite_pred(gp, gp.X, gp.y, gp.s2)
            optimState.lcbmax = max(fmu - options.ELCBOImproWeight * sqrt(fs2))

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
                optimState.RunCov = (
                    wRun * optimState.RunCov + (1 - wRun) * Sigma
                )
                optimState.LastRunAvg = optimState.N

            timer.stop_timer("finalize")
            # timer.totalruntime = NaN;   # Update at the end of iteration
            # timer

            # Record all useful stats
            stats.save_iteration(
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
                options.Diagnostics
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
            ) = self.__4vbmc_termination(optimState, action, stats, options)
            vp.stats.stable = stats.stable(optimState.iter)  # Save stability

            # Check if we are still warming-up
            if optimState.Warmup and loopiter > 1:
                if options.RecomputeLCBmax:
                    optimState.lcbmax_vec = recompute_lcbmax(
                        gp, optimState, stats, options
                    )  #'
                optimState, action, trim_flag = self.__4vbmc_warmup(
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
            if (
                optimState.OutwarpDelta
                and optimState.R < options.WarpTolReliability
            ):
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
                optimState.hedge = self.acqhedge_vbmc(
                    "upd", optimState.hedge, stats, options
                )

            """
            Write iteration output
            """

            # Stopped GP sampling this iteration?
            if (
                Ns_gp == options.StableGPSamples
                and stats.gpNsamples(max(1, loopiter - 1))
                > options.StableGPSamples
            ):
                if Ns_gp == 0:
                    if not action:
                        action = "switch to GP opt"
                    else:
                        action += ", switch to GP opt"  # [action ', switch to GP opt']
                else:
                    if not action:
                        action = "stable GP sampling"
                    else:
                        action += ", stable GP sampling"  # [action ', stable GP sampling']

            if prnt > 2:
                if options.BOWarmup and optimState.Warmup:
                    print(
                        displayFormat_warmup,
                        loopiter,
                        optimState.funccount,
                        max(optimState.y_orig),
                        action,
                    )
                else:
                    if optimState.Cache.active:
                        print(
                            displayFormat,
                            loopiter,
                            optimState.funccount,
                            optimState.cachecount,
                            elbo,
                            elbo_sd,
                            sKL,
                            vp.K,
                            optimState.R,
                            action,
                        )
                    elif (
                        optimState.UncertaintyHandlingLevel > 0
                        and options.MaxRepeatedObservations > 0
                    ):
                        print(
                            displayFormat,
                            loopiter,
                            optimState.funccount,
                            optimState.N,
                            elbo,
                            elbo_sd,
                            sKL,
                            vp.K,
                            optimState.R,
                            action,
                        )
                    else:
                        print(
                            displayFormat,
                            loopiter,
                            optimState.funccount,
                            elbo,
                            elbo_sd,
                            sKL,
                            vp.K,
                            optimState.R,
                            action,
                        )

            stats.timer(loopiter).totalruntime = toc(t0)

        """
        End of VBMC loop
        """

        vp_old = vp

        """
        Pick "best" variational solution to return (and real vp, if train vp differs)    
        """

        vp, elbo, elbo_sd, idx_best = self.__5best_vbmc(
            stats,
            loopiter,
            options.BestSafeSD,
            options.BestFracBack,
            options.RankCriterion,
            0,
        )
        # new_final_vp_flag = idx_best != loopiter
        gp = stats.gp(idx_best)
        vp.gp = gp
        # Add GP to variational posterior

        """
        Last variational optimization with large number of components
        """

        vp, elbo, elbo_sd, changedflag = self.__5finalboost_vbmc(
            vp, idx_best, optimState, stats, options
        )
        if changedflag:
            new_final_vp_flag = True

        if new_final_vp_flag and prnt > 2:
            # Recompute symmetrized KL-divergence
            sKL = max(
                0, 0.5 * sum(vbmc_kldiv(vp, vp_old, Nkl, options.KLgauss))
            )

    # Initial methods

    # Initial:

    # - boundscheck_vbmc(x0,LB,UB,PLB,PUB,prnt) -> BOUNDSCHECK Initial check of bounds.

    # - setupoptions_vbmc(nvars,defopts,options) %SETUPOPTIONS_VBMC Initialize OPTIONS struct for VBMC.

    # - setupvars_vbmc(x0,LB,UB,PLB,PUB,K,optimState,options,prnt) %INITVARS Initialize variational posterior, transforms and variables for VBMC.

    # - timer_init() %TIMER_INIT Initialize iteration timer.

    # - initFromVP(vp,LB,UB,PLB,PUB,prnt)

    # VMBC loop

    # Warping

    # - warp_input_vbmc(vp,optimState,gp,options) %WARP_INPUT_VBMC Perform input warping of variables.

    # - warp_gpandvp_vbmc(trinfo,vp_old,gp_old) %WARP_GPANDVP_VBMC Update GP hyps and variational posterior after warping.

    # Active Sampling

    def __1activesample_vbmc(
        self, optimState, Ns, funwrapper, vp, vp_old, gp, stats, timer, options
    ):
        """
        Actively sample points iteratively based on acquisition function.
        """

        NSsearch = options.NSsearch  # Number of points for acquisition fcn
        t_func = 0

        # time_active = tic

        if isempty(gp):

            # No GP yet, just use provided points or sample from plausible box
            [optimState, t_func] = initdesign_vbmc(
                optimState, Ns, funwrapper, t_func, options
            )

        else:  # Active uncertainty sampling

            SearchAcqFcn = options.SearchAcqFcn
            gp_old = []

            if options.AcqHedge and numel(SearchAcqFcn) > 1:
                # Choose acquisition function via hedge strategy
                optimState.hedge = acqhedge_vbmc(
                    "acq", optimState.hedge, [], options
                )
                idxAcq = optimState.hedge.chosen

            # Compute time cost (used by some acquisition functions)
            if optimState.iter > 2:
                deltaNeff = max(
                    1,
                    stats.Neff[optimState.iter - 1]
                    - stats.Neff[optimState.iter - 2],
                )
            else:
                deltaNeff = stats.Neff[0]

            timer_iter = stats.timer[optimState.iter - 1]

            gpTrain_vec = [stats.timer.gpTrain]

            if options.ActiveVariationalSamples > 0:
                options_activevar = options
                options_activevar.TolWeight = 0
                options_activevar.NSentFine = options.NSent
                options_activevar.ELCBOmidpoint = False
                Ns_activevar = options.ActiveVariationalSamples

            # Perform GP (and possibly variational) update after each active sample
            ActiveSampleFullUpdate_flag = (
                options.ActiveSampleVPUpdate or options.ActiveSampleGPUpdate
            ) and (
                (
                    (
                        optimState.iter
                        - options.ActiveSampleFullUpdatePastWarmup
                    )
                    <= optimState.LastWarmup
                )
                or (
                    stats.rindex(end) > options.ActiveSampleFullUpdateThreshold
                )
            )

            if ActiveSampleFullUpdate_flag and Ns > 1:
                RecomputeVarPost_old = optimState.RecomputeVarPost
                entropy_alpha_old = optimState.entropy_alpha

                options_update = options

                options_update.GPTolOpt = options.GPTolOptActive
                options_update.GPTolOptMCMC = options.GPTolOptMCMCActive
                options_update.TolWeight = 0
                options_update.NSent = options.NSentActive
                options_update.NSentFast = options.NSentFastActive
                options_update.NSentFine = options.NSentFineActive
                hypstruct = []
                vp0 = vp

            for iS in range(Ns):

                optimState.N = optimState.Xn  # Number of training inputs
                optimState.Neff = sum(optimState.nevals(optimState.X_flag))

                if options.ActiveVariationalSamples > 0:
                    vp, _, output = vpsample_vbmc(
                        Ns_activevar,
                        0,
                        vp,
                        gp,
                        optimState,
                        options_activevar,
                        options.ScaleLowerBound,
                    )
                    if "stepsize" in output:
                        optimState.mcmc_stepsize = output.stepsize

                Nextra = evaloption_vbmc(options.SampleExtraVPMeans, vp.K)
                if Nextra > 0:
                    vp_base = vp
                    NsFromGP = 4e3
                    Nextra = evaloption_vbmc(options.SampleExtraVPMeans, vp.K)
                    gpbeta = -options.OptimisticVariationalBound

                    x_range = max(gp.X) - min(gp.X)
                    LB_extra = min(gp.X) - 0.1 * x_range
                    UB_extra = max(gp.X) + 0.1 * x_range

                    [X_hpd, y_hpd] = gethpd_vbmc(gp.X, gp.y, options.HPDFrac)
                    Nextra = min(size(X_hpd, 1), Nextra)
                    # xx = gp.X(randperm(size(X_hpd,1),Nextra),:)
                    xx = gp.X
                    Nextra = size(xx, 1)

                    OptimizeMu = vp.optimize_mu
                    OptimizeWeights = vp.optimize_weights
                    vp.optimize_mu = false
                    vp.optimize_weights = true
                    vp.K = vp.K + Nextra
                    # vp.mu = [vp.mu,xx(round(linspace(1,size(xx,1),Nextra)),:)']
                    vp.sigma = [
                        vp.sigma,
                        exp(mean(log(vp.sigma))) * ones(1, Nextra),
                    ]
                    vp.w = [vp.w, exp(mean(log(vp.w))) * ones(1, Nextra)]
                    # vp.w = vp.w(:)'/sum(vp.w)
                    options_vp = options
                    options_vp.NSent = 0
                    options_vp.NSentFast = 0
                    options_vp.NSentFine = 0
                    options_vp.TolWeight = 0
                    options_vp.ELCBOWeight = (
                        -options.OptimisticVariationalBound
                    )
                    vp = vpoptimize_vbmc(
                        0, 1, vp, gp, [], optimState, options_vp, 0
                    )
                    vp.optimize_mu = OptimizeMu
                    vp.optimize_weights = OptimizeWeights

                if not options.AcqHedge:
                    idxAcq = randi(numel(SearchAcqFcn))

                """
                Pre-computations for acquisition functions
                """

                # Re-evaluate variance of the log joint if requested
                if optimState.acqInfo[idxAcq].get("compute_varlogjoint"):
                    _, _, varF = gplogjoint(vp, gp, 0, 0, 0, 1)
                    optimState.varlogjoint_samples = varF

                # Evaluate noise at each training point
                Ns_gp = numel(gp.post)
                sn2new = zeros(size(gp.X, 1), Ns_gp)

                # fix in python
                # for s in range(Ns_gp):
                # hyp_noise = gp.post(s).hyp(gp.Ncov+1:gp.Ncov+gp.Nnoise) #Get noise hyperparameters
                # if 'S' in optimState:
                #     s2 = (optimState.S(optimState.X_flag).^2).*optimState.nevals(optimState.X_flag)
                # else:
                #     s2 = []

                if options.NoiseShaping:
                    s2 = noiseshaping_vbmc(s2, gp.y, options)

                # sn2new(:,s) = gplite_noisefun(hyp_noise,gp.X,gp.noisefun,gp.y,s2)

                gp.sn2new = mean(sn2new, 2)

                # Evaluate GP input length scale (use geometric mean)
                D = size(gp.X, 2)
                ln_ell = zeros(D, Ns_gp)
                # fix in python
                # for s in range(Ns_gp):
                #     ln_ell(:,s) = gp.post(s).hyp(1:D)
                # optimState.gplengthscale = exp(mean(ln_ell,2))';

                # Rescale GP training inputs by GP length scale
                # fix in python
                # gp.X_rescaled = bsxfun(@rdivide,gp.X,optimState.gplengthscale)

                # Algorithmic time per iteration (from last iteration)
                t_base = (
                    timer_iter.activeSampling
                    + timer_iter.variationalFit
                    + timer_iter.finalize
                    + timer_iter.gpTrain
                )

                # Estimated increase in cost for a new training input
                if optimState.iter > 3:
                    len = 10
                    # xx = log(stats.N(max(end-len,ceil(end/2)):end))
                    # yy = log(gpTrain_vec(max(end-len,ceil(end/2)):end))
                    if numel(unique(xx)) > 1:
                        p = polyfit(xx, yy, 1)
                    #    gpTrain_diff = diff(exp(polyval(p,log([stats.N(end),stats.N(end)+1]))))
                    else:
                        gpTrain_diff = 0

                else:
                    gpTrain_diff = 0

                # Algorithmic cost per function evaluation
                optimState.t_algoperfuneval = t_base / deltaNeff + max(
                    0, gpTrain_diff
                )

                # Prepare for importance sampling based acquisition function
                if optimState.acqInfo[idxAcq].get("importance_sampling"):
                    optimState.ActiveImportanceSampling = (
                        activeimportancesampling_vbmc(
                            vp,
                            gp,
                            SearchAcqFcn[idxAcq],
                            optimState.acqInfo[idxAcq],
                            options,
                        )
                    )

                """
                Start active search
                """

                optimState.acqrand = rand()  # Seed for random acquisition fcn

                # Create search set from cache and randomly generated
                [Xsearch, idx_cache] = getSearchPoints(
                    NSsearch, optimState, vp, gp, options
                )
                Xsearch = real2int_vbmc(
                    Xsearch, vp.trinfo, optimState.integervars
                )

                # check how to do in python
                # acqEval = @(Xs_,vp_,gp_,optimState_,transpose_flag_)acqwrapper_vbmc(Xs_,vp_,gp_,optimState_,transpose_flag_,SearchAcqFcn{idxAcq},optimState.acqInfo{idxAcq})

                # Evaluate acquisition function
                acq_fast = acqEval(Xsearch, vp, gp, optimState, 0)

                if options.SearchCacheFrac > 0:
                    [_, ord] = sort(acq_fast, "ascend")
                    # optimState.SearchCache = Xsearch(ord,:)
                    idx = ord(1)
                else:
                    [_, idx] = min(acq_fast)

                # Xacq = Xsearch(idx,:)
                idx_cache_acq = idx_cache(idx)

                # Remove selected points from search set
                # Xsearch(idx,:) = []
                # idx_cache(idx) = []

                # Additional search via optimization
                if not strcmpi(options.SearchOptimizer, "none"):
                    # fval_old = acqEval(Xacq(1,:),vp,gp,optimState,0)
                    # x0 = real2int_vbmc(Xacq(1,:),vp.trinfo,optimState.integervars)
                    if all(
                        isfinite([optimState.LB_search, optimState.UB_search])
                    ):
                        LB = min([x0, optimState.LB_search])
                        UB = max([x0, optimState.UB_search])
                    else:
                        x_range = max(gp.X) - min(gp.X)
                        # LB = min([gp.X;x0]) - 0.1*x_range
                        # UB = max([gp.X;x0]) + 0.1*x_range

                    if optimState.acqInfo[idxAcq].get(log_flag):
                        TolFun = 1e-2
                    else:
                        TolFun = max(1e-12, abs(fval_old * 1e-3))

                # check if necessary in python
                fmincon_opts.Display = "off"
                fmincon_opts.TolFun = TolFun
                fmincon_opts.MaxFunEvals = options.SearchMaxFunEvals
                try:
                    # xsearch_optim,fval_optim,_,out_optim = fmincon(@(x) acqEval(x,vp,gp,optimState,0),x0,[],[],[],[],LB,UB,[],fmincon_opts)
                    nevals = out_optim.funcCount
                except:
                    print("Active search failed.\n")
                    fval_optim = float("inf")

                # if fval_optim < fval_old:
                #     Xacq(1,:) = real2int_vbmc(xsearch_optim,vp.trinfo,optimState.integervars)
                #     idx_cache_acq(1) = 0

                # Add random jitter
                if rand() < 1 / 3 and 0:
                    X_hpd = gethpd_vbmc(gp.X, gp.y, options.HPDFrac)
                    Sigma = diag(var(X_hpd)) * exp(randn())
                    # Xacq(1,:) = mvnrnd(Xacq(1,:),Sigma)
                    # Xacq(1,:) = real2int_vbmc(Xacq(1,:),vp.trinfo,optimState.integervars)

                if (
                    options.UncertaintyHandling
                    and options.MaxRepeatedObservations > 0
                ):
                    if (
                        optimState.RepeatedObservationsStreak
                        >= options.MaxRepeatedObservations
                    ):
                        # Maximum number of consecutive repeated observations
                        # (to prevent getting stuck in a wrong belief state)
                        optimState.RepeatedObservationsStreak = 0
                    else:
                        # Re-evaluate acquisition function on training set
                        X_train = get_traindata_vbmc(optimState, options)

                        # Disable variance-based regularization first
                        oldflag = optimState.VarianceRegularizedAcqFcn
                        optimState.VarianceRegularizedAcqFcn = False

                        # Use current cost of GP instead of future cost
                        old_t_algoperfuneval = optimState.t_algoperfuneval
                        optimState.t_algoperfuneval = t_base / deltaNeff
                        acq_train = acqEval(X_train, vp, gp, optimState, 0)
                        optimState.VarianceRegularizedAcqFcn = oldflag
                        optimState.t_algoperfuneval = old_t_algoperfuneval
                        [acq_train, idx_train] = min(acq_train)

                        # acq_now = acqEval(Xacq(1,:),vp,gp,optimState,0)

                        if acq_train < options.RepeatedAcqDiscount * acq_now:
                            # Xacq(1,:) = X_train(idx_train,:)
                            optimState.RepeatedObservationsStreak = (
                                optimState.RepeatedObservationsStreak + 1
                            )
                        else:
                            optimState.RepeatedObservationsStreak = 0

                if options.UncertaintyHandling and rand() < 0:
                    # fix in python:
                    # acqpeak = @acqfsn2reg_vbmc

                    # Evaluate acquisition function on training set
                    X_train = get_traindata_vbmc(optimState, options)

                    # Disable variance-based regularization first
                    oldflag = optimState.VarianceRegularizedAcqFcn
                    optimState.VarianceRegularizedAcqFcn = False

                    # Use current cost of GP instead of future cost
                    old_t_algoperfuneval = optimState.t_algoperfuneval
                    optimState.t_algoperfuneval = t_base / deltaNeff
                    acq_train = acqpeak(X_train, vp, gp, optimState, 0)
                    optimState.VarianceRegularizedAcqFcn = oldflag
                    optimState.t_algoperfuneval = old_t_algoperfuneval
                    [acq_train, idx_train] = min(acq_train)
                    # Xacq(1,:) = X_train(idx_train,:)

                # y_orig = [NaN; optimState.Cache.y_orig(:)]; #First position is NaN (not from cache)
                yacq = y_orig(idx_cache_acq + 1)
                idx_nn = not math.isnan(yacq)
                # if any(idx_nn):
                #    yacq(idx_nn) = yacq(idx_nn) + warpvars_vbmc(Xacq(idx_nn,:),'logp',optimState.trinfo)

                # xnew = Xacq(1,:)
                idxnew = 1

                # See if chosen point comes from starting cache
                idx = idx_cache_acq(idxnew)
                if idx > 0:
                    y_orig = optimState.Cache.y_orig(idx)
                else:
                    y_orig = float("nan")
                timer_func = tic
                if math.isnan(y_orig):
                    # Function value is not available, evaluate
                    try:
                        [ynew, optimState, idx_new] = funlogger_vbmc(
                            funwrapper, xnew, optimState, "iter"
                        )
                    except:  # func_error:
                        print("pause")
                        # Pause
                else:
                    [ynew, optimState, idx_new] = funlogger_vbmc(
                        funwrapper, xnew, optimState, "add", y_orig
                    )
                    # Remove point from starting cache
                    # optimState.Cache.X_orig(idx,:) = []
                    # optimState.Cache.y_orig(idx) = []

                t_func = t_func + toc(timer_func)

                if "S" in optimState:
                    s2new = optimState.S(idx_new) ^ 2
                else:
                    s2new = []

                tnew = optimState.funevaltime(idx_new)

                if not "acqtable" in optimState:
                    optimState.acqtable = []
                [_, _, fmu, fs2] = gplite_pred(gp, xnew)
                v = [idxAcq, ynew, fmu, sqrt(fs2)]
                optimState.acqtable = [optimState.acqtable, v]

                if Nextra > 0:
                    vp = vp_base

                if iS < Ns:

                    if not isempty(gp_old):
                        gp = gp_old

                    if ActiveSampleFullUpdate_flag:
                        # Quick GP update
                        if isempty(hypstruct):
                            hypstruct = optimState.hypstruct

                        fESS_thresh = options.ActiveSamplefESSThresh
                        gptmp = []
                        if fESS_thresh < 1:
                            gptmp = gpreupdate(gp, optimState, options)
                            fESS = fess_vbmc(vp, gptmp, 100)
                        else:
                            fESS = 0

                        if fESS <= fESS_thresh:
                            if options.ActiveSampleGPUpdate:
                                t = tic
                                [
                                    gp,
                                    hypstruct,
                                    Ns_gp,
                                    optimState,
                                ] = self.__2gptrain_vbmc(
                                    hypstruct,
                                    optimState,
                                    stats,
                                    options_update,
                                )
                                timer.gpTrain = timer.gpTrain + toc(t)
                            else:
                                if isempty(gptmp):
                                    gp = gpreupdate(gp, optimState, options)
                                else:
                                    gp = gptmp

                            if options.ActiveSampleVPUpdate:
                                # Quick variational optimization

                                t = tic
                                # Decide number of fast optimizations
                                Nfastopts = ceil(
                                    options_update.NSelboIncr
                                    * evaloption_vbmc(
                                        options_update.NSelbo, vp.K
                                    )
                                )
                                if options.UpdateRandomAlpha:
                                    optimState.entropy_alpha = 1 - sqrt(rand())

                                vp = vpoptimize_vbmc(
                                    Nfastopts,
                                    1,
                                    vp,
                                    gp,
                                    [],
                                    optimState,
                                    options_update,
                                    0,
                                )
                                # optimState.vp_repo{end+1} = get_vptheta(vp)
                                timer.variationalFit = (
                                    timer.variationalFit + toc(t)
                                )

                        else:
                            gp = gptmp

                    else:
                        # Perform simple rank-1 update if no noise and first sample
                        t = tic
                        update1 = (
                            (
                                len(s2new) == 0
                                or optimState.nevals(idx_new) == 1
                            )
                            and not options.NoiseShaping
                            and not options.IntegrateGPMean
                        )
                        if update1:
                            gp = gplite_post(
                                gp, xnew, ynew, [], [], [], s2new, 1
                            )
                            # gp.t(end+1) = tnew
                        else:
                            gp = gpreupdate(gp, optimState, options)

                        # timer.gpTrain = timer.gpTrain + toc(t)

                # Check if active search bounds need to be expanded
                delta_search = 0.05 * (
                    optimState.UB_search - optimState.LB_search
                )

                # ADD DIFFERENT CHECKS FOR INTEGER VARIABLES!
                idx = abs(xnew - optimState.LB_search) < delta_search
                optimState.LB_search[idx] = max(
                    optimState.LB(idx),
                    optimState.LB_search(idx) - delta_search(idx),
                )
                idx = abs(xnew - optimState.UB_search) < delta_search
                optimState.UB_search[idx] = min(
                    optimState.UB[idx],
                    optimState.UB_search[idx] + delta_search[idx],
                )

                # Hard lower/upper bounds on search
                prange = optimState.PUB - optimState.PLB
                LB_searchmin = max(
                    optimState.PLB - 2 * prange * options.ActiveSearchBound,
                    optimState.LB,
                )
                UB_searchmin = min(
                    optimState.PUB + 2 * prange * options.ActiveSearchBound,
                    optimState.UB,
                )
                return (optimState, vp, gp, timer)

    def __1acqhedge_vbmc(self, action, hedge, stats, options):
        """
        ACQPORTFOLIO Evaluate and update portfolio of acquisition functions. (unused)
        """
        pass

    def __1getAcqInfo(self, SearchAcqFcn):
        """
        GETACQINFO Get information from acquisition function(s)
        """
        pass

    def __1gpreupdate(self, gp, optimState, options):
        """
        GPREUPDATE Quick posterior reupdate of Gaussian process
        """
        pass

    # GP Training

    def __2gptrain_vbmc(self, hypstruct, optimState, stats, options):
        """
        GPTRAIN_VBMC Train Gaussian process model.
        """
        # return [gp,hypstruct,Ns_gp,optimState]
        pass

    # Variational optimization / training of variational posterior:

    def __3updateK(self, optimState, stats, options):
        """
        UPDATEK Update number of variational mixture components.
        """
        pass

    def __3vpoptimize_vbmc(
        self, Nfastopts, Nslowopts, vp, gp, K, optimState, options, prnt
    ):
        """
        VPOPTIMIZE Optimize variational posterior.
        """
        pass

    # Loop termination:

    def __4vbmc_warmup(self, optimState, stats, action, options):
        """
        check if warmup ends
        """
        pass

    def __4vbmc_termination(self, optimState, action, stats, options):
        """
        Compute stability index and check termination conditions.
        """
        pass

    def __4recompute_lcbmax(self, gp, optimState, stats, options):
        """
        RECOMPUTE_LCBMAX Recompute moving LCB maximum based on current GP.
        """
        pass

    # Finalizing:

    def __5finalboost_vbmc(self, vp, idx_best, optimState, stats, options):
        """
        FINALBOOST_VBMC Final boost of variational components.
        """
        pass

    def __5best_vbmc(
        self, stats, idx, SafeSD, FracBack, RankCriterion, RealFlag
    ):
        """
        VBMC_BEST Return best variational posterior from stats structure.
        """
        pass

    def acqhedge_vbmc(self):
        pass


# helper functions:

# - savestats(stats, optimState,vp,elbo,elbo_sd,varss,sKL,sKL_true,gp,hypstruct.full, Ns_gp,pruned,timer,options.Diagnostics);

# - funlogger_vbmc(fun,x,optimState,state,varargin) %FUNLOGGER_VBMC Call objective function and do some bookkeeping.

# - evaloption_vbmc(option,N) %GETVALUE_VBMC Return option value that could be a function handle.

# - mvnkl(Mu1,Sigma1,Mu2,Sigma2) %MVNKL Kullback-Leibler divergence between two multivariate normal pdfs.
