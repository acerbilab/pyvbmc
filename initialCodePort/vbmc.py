class VBMC(object):
    """
    The VBMC algorithm class
    """
    
    def __init__(self):
        pass



    #Initial methods

        # Initial:

        # - boundscheck_vbmc(x0,LB,UB,PLB,PUB,prnt) -> BOUNDSCHECK Initial check of bounds.

        # - setupoptions_vbmc(nvars,defopts,options) %SETUPOPTIONS_VBMC Initialize OPTIONS struct for VBMC.

        # - setupvars_vbmc(x0,LB,UB,PLB,PUB,K,optimState,options,prnt) %INITVARS Initialize variational posterior, transforms and variables for VBMC.

        # - timer_init() %TIMER_INIT Initialize iteration timer.

        # - initFromVP(vp,LB,UB,PLB,PUB,prnt)

    #VMBC loop

    #Warping

		# - warp_input_vbmc(vp,optimState,gp,options) %WARP_INPUT_VBMC Perform input warping of variables.

		# - warp_gpandvp_vbmc(trinfo,vp_old,gp_old) %WARP_GPANDVP_VBMC Update GP hyps and variational posterior after warping.

    #Active Sampling

    def __1activesample_vbmc(self, optimState,Ns,funwrapper,vp,vp_old,gp,stats,timer,options):
        """
        Actively sample points iteratively based on acquisition function.
        """
        pass

    def __1acqhedge_vbmc(self, action,hedge,stats,options):
        """
        ACQPORTFOLIO Evaluate and update portfolio of acquisition functions. (unused)
        """
        pass

    def __1getAcqInfo(self, SearchAcqFcn):
        """
        GETACQINFO Get information from acquisition function(s)
        """
        pass

    def __1gpreupdate(self, gp,optimState,options):
        """
        GPREUPDATE Quick posterior reupdate of Gaussian process
        """
        pass

    # GP Training
    
    def __2gptrain_vbmc(self, hypstruct,optimState,stats,options):
        """
        GPTRAIN_VBMC Train Gaussian process model.
        """
        pass

    #Variational optimization / training of variational posterior:

    def __3updateK(self, optimState,stats,options):
        """
		UPDATEK Update number of variational mixture components.
        """
        pass

    def __3vpoptimize_vbmc(self, Nfastopts,Nslowopts,vp,gp,K,optimState,options,prnt):
        """
        VPOPTIMIZE Optimize variational posterior.
        """
        pass

    #Loop termination:
    
    def __4vbmc_warmup(self, optimState,stats,action,options):
        """
        check if warmup ends
        """
        pass    

    def __4vbmc_termination(self, optimState,action,stats,options):
        """
        Compute stability index and check termination conditions.
        """
        pass

    def __4recompute_lcbmax(self, gp,optimState,stats,options):
        """
        RECOMPUTE_LCBMAX Recompute moving LCB maximum based on current GP.
        """
        pass

    #Finalizing:

    def __5finalboost_vbmc(self, vp,idx_best,optimState,stats,options):
        """
        FINALBOOST_VBMC Final boost of variational components.
        """
        pass

    def __5best_vbmc(self, stats,idx,SafeSD,FracBack,RankCriterion,RealFlag):
        """
        VBMC_BEST Return best variational posterior from stats structure.
        """
        pass

	# helper functions:

	# - savestats(stats, optimState,vp,elbo,elbo_sd,varss,sKL,sKL_true,gp,hypstruct.full, Ns_gp,pruned,timer,options.Diagnostics);    

	# - funlogger_vbmc(fun,x,optimState,state,varargin) %FUNLOGGER_VBMC Call objective function and do some bookkeeping.

	# - evaloption_vbmc(option,N) %GETVALUE_VBMC Return option value that could be a function handle.

	# - mvnkl(Mu1,Sigma1,Mu2,Sigma2) %MVNKL Kullback-Leibler divergence between two multivariate normal pdfs.