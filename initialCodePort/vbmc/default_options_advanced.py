# Advanced options (do not modify unless you *know* what you are doing)
default_options_advanced = {
    # Explicit noise handling'
    "defopts.UncertaintyHandling": [],
    # Array with indices of integer variables'
    "defopts.IntegerVars": [],
    # Base observation noise magnitude (standard deviation)'
    "defopts.NoiseSize": [],
    # Max number of consecutive repeated measurements for noisy inputs'
    "defopts.MaxRepeatedObservations": 0,
    # Multiplicative discount True acquisition fcn
    # to repeat measurement at the same location'
    "defopts.RepeatedAcqDiscount": 1,
    # Number of initial target fcn evals'
    "defopts.FunEvalStart": "max(D,10)",
    # Base step size for stochastic gradient descent'
    "defopts.SGDStepSize": 0.005,
    # Skip active sampling the first iteration after warmup'
    "defopts.SkipActiveSamplingAfterWarmup": False,
    # Use ranking criterion to pick best non-converged solution'
    "defopts.RankCriterion": True,
    # Required stable iterations to switch entropy approximation'
    "defopts.TolStableEntropyIters": 6,
    # Use variable component means for variational posterior'
    "defopts.VariableMeans": True,
    # Use variable mixture weight for variational posterior'
    "defopts.VariableWeights": True,
    # Penalty multiplier for small mixture weights'
    "defopts.WeightPenalty": 0.1,
    # Run in diagnostics mode, get additional info'
    "defopts.Diagnostics": False,
    # Output function'
    "defopts.OutputFcn": [],
    # Fraction of allowed exceptions when computing iteration stability'
    "defopts.TolStableExcptFrac": 0.2,
    # Evaluated fcn values at X0'
    "defopts.Fvals": [],
    # Use Optimization Toolbox (if empty, determine at runtime)'
    "defopts.OptimToolbox": [],
    # Weighted proposal fcn for uncertainty search'
    "defopts.ProposalFcn": [],
    # Automatic nonlinear rescaling of variables'
    "defopts.NonlinearScaling": True,
    # Fast search acquisition fcn(s)'
    "defopts.SearchAcqFcn": "@acqf_vbmc        ",
    # Samples for fast acquisition fcn eval per new point'
    "defopts.NSsearch": 2 ^ 13,
    # Total samples for Monte Carlo approx. of the entropy'
    "defopts.NSent": "@(K) 100*K.^(2/3) ",
    # Total samples for preliminary Monte Carlo approx. of the entropy'
    "defopts.NSentFast": 0,
    # Total samples for refined Monte Carlo approx. of the entropy'
    "defopts.NSentFine": "@(K) 2^12*K       ",
    # Total samples for Monte Carlo approx. of the entropy (final boost)'
    "defopts.NSentBoost": "@(K) 200*K.^(2/3) ",
    # Total samples for preliminary Monte Carlo
    # approx. of the entropy (final boost)'
    "defopts.NSentFastBoost": [],
    # Total samples for refined Monte Carlo
    # approx. of the entropy (final boost)'
    "defopts.NSentFineBoost": [],
    # Total samples for Monte Carlo approx. of the entropy (active sampling)'
    "defopts.NSentActive": "@(K) 20*K.^(2/3)  ",
    # Total samples for preliminary Monte Carlo
    # approx. of the entropy (active sampling)'
    "defopts.NSentFastActive": 0,
    # Total samples for refined Monte Carlo
    # approx. of the entropy (active sampling)'
    "defopts.NSentFineActive": "@(K) 200*K        ",
    # Samples for fast approximation of the ELBO'
    "defopts.NSelbo": "@(K) 50*K         ",
    # Multiplier to samples for fast approx. of ELBO for incremental iterations'
    "defopts.NSelboIncr": 0.1,
    # Starting points to refine optimization of the ELBO'
    "defopts.ElboStarts": 2,
    # Max GP hyperparameter samples (decreases with training points)'
    "defopts.NSgpMax": 80,
    # Max GP hyperparameter samples during warmup'
    "defopts.NSgpMaxWarmup": 8,
    # Max GP hyperparameter samples during main algorithm'
    "defopts.NSgpMaxMain": "Inf",
    # Fcn evals without improvement before stopping warmup'
    "defopts.WarmupNoImproThreshold": "20 + 5 * nvars",
    # Also check for max fcn value improvement before stopping warmup'
    "defopts.WarmupCheckMax": True,
    # Force stable GP hyperparameter sampling
    # (reduce samples or start optimizing)'
    "defopts.StableGPSampling": "200 + 10 * nvars",
    # Force stable GP hyperparameter sampling
    # after reaching this number of components'
    "defopts.StableGPvpK": "Inf",
    # Number of GP samples when GP is stable (0 = optimize)'
    "defopts.StableGPSamples": 0,
    # Thinning for GP hyperparameter sampling'
    "defopts.GPSampleThin": 5,
    # Initial design points for GP hyperparameter training'
    "defopts.GPTrainNinit": 1024,
    # Final design points for GP hyperparameter training'
    "defopts.GPTrainNinitFinal": 64,
    # Initial design method for GP hyperparameter training'
    "defopts.GPTrainInitMethod": "rand",
    # Tolerance for optimization of GP hyperparameters'
    "defopts.GPTolOpt": 1e-5,
    # Tolerance for optimization of GP hyperparameters preliminary to MCMC'
    "defopts.GPTolOptMCMC": 1e-2,
    # Tolerance for optimization of GP hyperparameters during active sampling'
    "defopts.GPTolOptActive": 1e-4,
    # Tolerance for optimization of GP hyperparameters preliminary
    # to MCMC during active sampling'
    "defopts.GPTolOptMCMCActive": 1e-2,
    # Threshold True GP variance used by regulatized acquisition fcns'
    "defopts.TolGPVar": 1e-4,
    # Threshold True GP variance, used to stabilize sampling'
    "defopts.TolGPVarMCMC": 1e-4,
    # GP mean function'
    "defopts.gpMeanFun": "negquad",
    # GP integrated mean function'
    "defopts.gpIntMeanFun": 0,
    # Max variational components as a function of training points'
    "defopts.KfunMax": "@(N) N.^(2/3)     ",
    # Variational components during warmup'
    "defopts.Kwarmup": 2,
    # Added variational components for stable solution'
    "defopts.AdaptiveK": 2,
    # High Posterior Density region (fraction of training inputs)'
    "defopts.HPDFrac": 0.8,
    # Uncertainty weight True ELCBO for computing lower bound improvement'
    "defopts.ELCBOImproWeight": 3,
    # Minimum fractional length scale'
    "defopts.TolLength": 1e-6,
    # Size of cache for storing fcn evaluations'
    "defopts.CacheSize": 500,
    # Fraction of search points from starting cache (if nonempty)'
    "defopts.CacheFrac": 0.5,
    # Stochastic optimizer for varational parameters'
    "defopts.StochasticOptimizer": "adam",
    # Stopping threshold for stochastic optimization'
    "defopts.TolFunStochastic": 1e-3,
    # Max iterations for stochastic optimization'
    "defopts.MaxIterStochastic": "100 * (2 + nvars)",
    # Set stochastic optimization stepsize via GP hyperparameters'
    "defopts.GPStochasticStepsize": False,
    # Tolerance True ELBO uncertainty for stopping
    # (if variational posterior is stable)'
    "defopts.TolSD": 0.1,
    # Stopping threshold True change of
    # variational posterior per training point'
    "defopts.TolsKL": "0.01 * sqrt(nvars)",
    # Number of stable fcn evals for stopping warmup'
    "defopts.TolStableWarmup": 15,
    # MCMC sampler for variational posteriors'
    "defopts.VariationalSampler": "malasample",
    # Required ELCBO improvement per fcn eval before termination'
    "defopts.TolImprovement": 0.01,
    # Use Gaussian approximation for symmetrized KL-divergence b\w iters'
    "defopts.KLgauss": True,
    # True mean of the target density (for debugging)'
    "defopts.TrueMean": [],
    # True covariance of the target density (for debugging)'
    "defopts.TrueCov": [],
    # Min number of fcn evals'
    "defopts.MinFunEvals": "5 * nvars",
    # Min number of iterations'
    "defopts.MinIter": "nvars",
    # Fraction of search points from heavy-tailed variational posterior'
    "defopts.HeavyTailSearchFrac": 0.25,
    # Fraction of search points from multivariate normal'
    "defopts.MVNSearchFrac": 0.25,
    # Fraction of search points from multivariate normal fitted to HPD points'
    "defopts.HPDSearchFrac": 0,
    # Fraction of search points
    # from uniform random box based True training inputs'
    "defopts.BoxSearchFrac": 0.25,
    # Fraction of search points from previous iterations'
    "defopts.SearchCacheFrac": 0,
    # Always fully refit variational posterior'
    "defopts.AlwaysRefitVarPost": False,
    # Perform warm-up stage'
    "defopts.Warmup": True,
    # Special OPTIONS struct for warmup stage'
    "defopts.WarmupOptions": [],
    # Stop warm-up when ELCBO increase below threshold (per fcn eval)'
    "defopts.StopWarmupThresh": 0.2,
    # Max log-likelihood difference for points kept after warmup'
    "defopts.WarmupKeepThreshold": "10 * nvars",
    # Max log-likelihood difference for points kept
    # after a false-alarm warmup stop'
    "defopts.WarmupKeepThresholdFalseAlarm": "100 * (nvars + 2)",
    # Reliability index required to stop warmup'
    "defopts.StopWarmupReliability": 100,
    # Optimization method for active sampling'
    "defopts.SearchOptimizer": "cmaes",
    # Initialize CMA-ES search SIGMA from variational posterior'
    "defopts.SearchCMAESVPInit": True,
    # Take bestever solution from CMA-ES search'
    "defopts.SearchCMAESbest": False,
    # Max number of acquisition fcn evaluations during search'
    "defopts.SearchMaxFunEvals": "500 * (nvars + 2)",
    # Weight of previous trials (per trial)
    # for running avg of variational posterior moments'
    "defopts.MomentsRunWeight": 0.9,
    # Upper threshold True reliability index
    #  for full retraining of GP hyperparameters'
    "defopts.GPRetrainThreshold": 1,
    # Compute full ELCBO also at best midpoint'
    "defopts.ELCBOmidpoint": True,
    # Multiplier to widths from previous posterior for GP sampling
    # (Inf = do not use previous widths)'
    "defopts.GPSampleWidths": 5,
    # Weight of previous trials (per trial)
    # for running avg of GP hyperparameter covariance'
    "defopts.HypRunWeight": 0.9,
    # Use weighted hyperparameter posterior covariance'
    "defopts.WeightedHypCov": True,
    # Minimum weight for weighted hyperparameter posterior covariance'
    "defopts.TolCovWeight": 0,
    # MCMC sampler for GP hyperparameters'
    "defopts.GPHypSampler": "slicesample",
    # Switch to covariance sampling below this threshold of stability index'
    "defopts.CovSampleThresh": 10,
    # Optimality tolerance for optimization of deterministic entropy'
    "defopts.DetEntTolOpt": 1e-3,
    # Switch from deterministic entropy
    # to stochastic entropy when reaching stability'
    "defopts.EntropySwitch": False,
    # Force switch to stochastic entropy at this fraction of total fcn evals'
    "defopts.EntropyForceSwitch": 0.8,
    # Alpha value for lower/upper deterministic entropy interpolation'
    "defopts.DetEntropyAlpha": 0,
    # Randomize deterministic entropy alpha during active sample updates'
    "defopts.UpdateRandomAlpha": False,
    # Online adaptation of alpha value
    # for lower/upper deterministic entropy interpolation'
    "defopts.AdaptiveEntropyAlpha": False,
    # Start with deterministic entropy only with this number of vars or more'
    "defopts.DetEntropyMinD": 5,
    # Fractional tolerance for constraint violation of variational parameters'
    "defopts.TolConLoss": 0.01,
    # SD multiplier of ELCBO for computing best variational solution'
    "defopts.BestSafeSD": 5,
    # When computing best solution, lacking stability
    # go back up to this fraction of iterations'
    "defopts.BestFracBack": 0.25,
    # Threshold mixture component weight for pruning'
    "defopts.TolWeight": 1e-2,
    # Multiplier to threshold for pruning mixture weights'
    "defopts.PruningThresholdMultiplier": "@(K) 1/sqrt(K)   ",
    # Annealing for hyperprior width of GP negative quadratic mean'
    "defopts.AnnealedGPMean": "@(N,NMAX) 0       ",
    # Strict hyperprior for GP negative quadratic mean'
    "defopts.ConstrainedGPMean": False,
    # Empirical Bayes prior over some GP hyperparameters'
    "defopts.EmpiricalGPPrior": False,
    # Minimum GP observation noise'
    "defopts.TolGPNoise": "sqrt(1e-5)",
    # Prior mean over GP input length scale (in plausible units)'
    "defopts.GPLengthPriorMean": "sqrt(D/6)",
    # Prior std over GP input length scale (in plausible units)'
    "defopts.GPLengthPriorStd": "0.5*log(1e3)",
    # Upper bound True GP input lengths based True plausible box (0 = ignore)'
    "defopts.UpperGPLengthFactor": 0,
    # Initial samples ("plausible" is uniform in the plausible box)'
    "defopts.InitDesign": "plausible",
    # Stricter upper bound True GP negative quadratic mean function'
    "defopts.gpQuadraticMeanBound": True,
    # Bandwidth parameter for GP smoothing (in units of plausible box)'
    "defopts.Bandwidth": 0,
    # Heuristic output warping ("fitness shaping")'
    "defopts.FitnessShaping": False,
    # Output warping starting threshold'
    "defopts.OutwarpThreshBase": "10 * nvars",
    # Output warping threshold multiplier when failed sub-threshold check'
    "defopts.OutwarpThreshMult": 1.25,
    # Output warping base threshold tolerance (fraction of current threshold)'
    "defopts.OutwarpThreshTol": 0.8,
    # Temperature for posterior tempering (allowed values T = 1,2,3,4)'
    "defopts.Temperature": 1,
    # Use separate GP with constant mean for active search'
    "defopts.SeparateSearchGP": False,
    # Discount observations from from extremely low-density regions'
    "defopts.NoiseShaping": False,
    # Threshold from max observed value to start discounting'
    "defopts.NoiseShapingThreshold": "10 * nvars",
    # Proportionality factor of added noise wrt distance from threshold'
    "defopts.NoiseShapingFactor": 0.05,
    # Hedge True multiple acquisition functions'
    "defopts.AcqHedge": False,
    # Past iterations window to judge acquisition fcn improvement'
    "defopts.AcqHedgeIterWindow": 4,
    # Portfolio value decay per function evaluation'
    "defopts.AcqHedgeDecay": 0.9,
    # MCMC variational steps before each active sampling'
    "defopts.ActiveVariationalSamples": 0,
    # Apply lower bound True variational components
    # scale during variational sampling'
    "defopts.ScaleLowerBound": True,
    # Perform variational optimization after each active sample'
    "defopts.ActiveSampleVPUpdate": False,
    # Perform GP training after each active sample'
    "defopts.ActiveSampleGPUpdate": False,
    # # iters past warmup to continue update after each active sample'
    "defopts.ActiveSampleFullUpdatePastWarmup": 2,
    # Perform full update during active sampling if stability above threshold'
    "defopts.ActiveSampleFullUpdateThreshold": 3,
    # Use previous variational posteriors to initialize optimization'
    "defopts.VariationalInitRepo": False,
    # Extra variational components sampled from GP profile'
    "defopts.SampleExtraVPMeans": 0,
    # Uncertainty weight True ELCBO during active sampling'
    "defopts.OptimisticVariationalBound": 0,
    # # importance samples from smoothed variational posterior'
    "defopts.ActiveImportanceSamplingVPSamples": 100,
    # # importance samples from box-uniform centered True training inputs'
    "defopts.ActiveImportanceSamplingBoxSamples": 100,
    # # importance samples through MCMC'
    "defopts.ActiveImportanceSamplingMCMCSamples": 100,
    # Thinning for importance sampling MCMC'
    "defopts.ActiveImportanceSamplingMCMCThin": 1,
    # fractional ESS threhsold to update GP and VP'
    "defopts.ActiveSamplefESSThresh": 1,
    # % fractional ESS threhsold to do MCMC while active importance sampling'
    "defopts.ActiveImportanceSamplingfESSThresh": 0.9,
    # Active search bound multiplier'
    "defopts.ActiveSearchBound": 2,
    # Try integrating GP mean function'
    "defopts.IntegrateGPMean": False,
    # Tolerance True closeness to bound constraints (fraction of total range)'
    "defopts.TolBoundX": 1e-5,
    # Recompute LCB max for each iteration based True current GP estimate'
    "defopts.RecomputeLCBmax": True,
    # Input transform for bounded variables'
    "defopts.BoundedTransform": "logit",
    # Use double GP'
    "defopts.DoubleGP": False,
    # Warp every this number of iterations'
    "defopts.WarpEveryIters": 5,
    # Increase delay between warpings'
    "defopts.IncrementalWarpDelay": True,
    # Threshold True reliability index to perform warp'
    "defopts.WarpTolReliability": 3,
    # Rotate and scale input'
    "defopts.WarpRotoScaling": True,
    # Regularization weight towards
    # diagonal covariance matrix for N training inputs'
    "defopts.WarpCovReg": 0,
    # Threshold True correlation matrix for roto-scaling'
    "defopts.WarpRotoCorrThresh": 0.05,
    # Min number of variational components to perform warp'
    "defopts.WarpMinK": 5,
    # Immediately undo warp if not improving ELBO'
    "defopts.WarpUndoCheck": True,
    # Improvement of ELBO required to keep a warp proposal'
    "defopts.WarpTolImprovement": 0.1,
    # Multiplier tolerance of ELBO SD after warp proposal'
    "defopts.WarpTolSDMultiplier": 2,
    # Base tolerance True ELBO SD after warp proposal'
    "defopts.WarpTolSDBase": 1,
}