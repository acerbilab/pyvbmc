class Options_VBMC(object):
    """
    Default Options of VBMC algorithm
    Appoach to model as a class inspired by
    https://stackoverflow.com/questions/211695/what-is-an-easy-way-to-create-a-trivial-one-off-python-object
    """

    """
    Basic default options
    """
    Display                 = None #'iter        % Level of display ("iter", "notify", "final", or "off")';
    Plot                    = None #'off         % Plot marginals of variational posterior at each iteration';
    MaxIter                 = None #'50*(2+nvars)% Max number of iterations';
    MaxFunEvals             = None #50*(2+nvars) % Max number of target fcn evals';
    FunEvalsPerIter         = None #5            % Number of target fcn evals per iteration';
    TolStableCount          = None #60           % Required stable fcn evals for termination';
    RetryMaxFunEvals        = None #0            % Max number of target fcn evals on retry (0 = no retry)';
    MinFinalComponents      = None #50           % Number of variational components to refine posterior at termination';
    SpecifyTargetNoise      = None #no           % Target log joint function returns noise estimate (SD) as second output';


    """
    Advanced options (do not modify unless you *know* what you are doing)
    """
    UncertaintyHandling     = None #[]           % Explicit noise handling';
    IntegerVars             = None #[]           % Array with indices of integer variables';
    NoiseSize               = None #[]           % Base observation noise magnitude (standard deviation)';
    MaxRepeatedObservations = None #0            % Max number of consecutive repeated measurements for noisy inputs';
    RepeatedAcqDiscount     = None #1            % Multiplicative discount on acquisition fcn to repeat measurement at the same location';
    FunEvalStart            = None #max(D,10)    % Number of initial target fcn evals';
    SGDStepSize             = None #0.005        % Base step size for stochastic gradient descent';
    SkipActiveSamplingAfterWarmup  = None #no    % Skip active sampling the first iteration after warmup';
    RankCriterion           = None #yes          % Use ranking criterion to pick best non-converged solution';
    TolStableEntropyIters   = None #6            % Required stable iterations to switch entropy approximation';
    VariableMeans           = None #yes          % Use variable component means for variational posterior';
    VariableWeights         = None #yes          % Use variable mixture weight for variational posterior';
    WeightPenalty           = None #0.1          % Penalty multiplier for small mixture weights';
    Diagnostics             = None #off          % Run in diagnostics mode, get additional info';
    OutputFcn               = None #[]           % Output function';
    TolStableExcptFrac      = None #0.2          % Fraction of allowed exceptions when computing iteration stability';
    Fvals                   = None #[]           % Evaluated fcn values at X0';
    OptimToolbox            = None #[]           % Use Optimization Toolbox (if empty, determine at runtime)';
    ProposalFcn             = None #[]           % Weighted proposal fcn for uncertainty search';
    NonlinearScaling   = None #on                % Automatic nonlinear rescaling of variables';
    SearchAcqFcn       = None #@acqf_vbmc        % Fast search acquisition fcn(s)';
    NSsearch           = None #2^13              % Samples for fast acquisition fcn eval per new point';
    NSent              = None #@(K) 100*K.^(2/3) % Total samples for Monte Carlo approx. of the entropy';
    NSentFast          = None #0                 % Total samples for preliminary Monte Carlo approx. of the entropy';
    NSentFine          = None #@(K) 2^12*K       % Total samples for refined Monte Carlo approx. of the entropy';
    NSentBoost         = None #@(K) 200*K.^(2/3) % Total samples for Monte Carlo approx. of the entropy (final boost)';
    NSentFastBoost     = None #[]                % Total samples for preliminary Monte Carlo approx. of the entropy (final boost)';
    NSentFineBoost     = None #[]                % Total samples for refined Monte Carlo approx. of the entropy (final boost)';
    NSentActive        = None #@(K) 20*K.^(2/3)  % Total samples for Monte Carlo approx. of the entropy (active sampling)';
    NSentFastActive    = None #0                 % Total samples for preliminary Monte Carlo approx. of the entropy (active sampling)';
    NSentFineActive    = None #@(K) 200*K        % Total samples for refined Monte Carlo approx. of the entropy (active sampling)';
    NSelbo             = None #@(K) 50*K         % Samples for fast approximation of the ELBO';
    NSelboIncr         = None #0.1               % Multiplier to samples for fast approx. of ELBO for incremental iterations';
    ElboStarts         = None #2                 % Starting points to refine optimization of the ELBO';
    NSgpMax            = None #80                % Max GP hyperparameter samples (decreases with training points)';
    NSgpMaxWarmup      = None #8                 % Max GP hyperparameter samples during warmup';
    NSgpMaxMain        = None #Inf               % Max GP hyperparameter samples during main algorithm';
    WarmupNoImproThreshold = None #20 + 5*nvars  % Fcn evals without improvement before stopping warmup';
    WarmupCheckMax     = None #yes               % Also check for max fcn value improvement before stopping warmup';
    StableGPSampling   = None #200 + 10*nvars    % Force stable GP hyperparameter sampling (reduce samples or start optimizing)';
    StableGPvpK        = None #Inf               % Force stable GP hyperparameter sampling after reaching this number of components';
    StableGPSamples    = None #0                 % Number of GP samples when GP is stable (0 = optimize)';
    GPSampleThin       = None #5                 % Thinning for GP hyperparameter sampling';
    GPTrainNinit       = None #1024              % Initial design points for GP hyperparameter training';
    GPTrainNinitFinal  = None #64                % Final design points for GP hyperparameter training';
    GPTrainInitMethod  = None #rand              % Initial design method for GP hyperparameter training';
    GPTolOpt           = None #1e-5              % Tolerance for optimization of GP hyperparameters';
    GPTolOptMCMC       = None #1e-2              % Tolerance for optimization of GP hyperparameters preliminary to MCMC';
    GPTolOptActive     = None #1e-4              % Tolerance for optimization of GP hyperparameters during active sampling';
    GPTolOptMCMCActive = None #1e-2              % Tolerance for optimization of GP hyperparameters preliminary to MCMC during active sampling';
    TolGPVar           = None #1e-4              % Threshold on GP variance used by regulatized acquisition fcns';
    TolGPVarMCMC       = None #1e-4              % Threshold on GP variance, used to stabilize sampling';
    gpMeanFun          = None #negquad           % GP mean function';
    gpIntMeanFun       = None #0                 % GP integrated mean function';
    KfunMax            = None #@(N) N.^(2/3)     % Max variational components as a function of training points';
    Kwarmup            = None #2                 % Variational components during warmup';
    AdaptiveK          = None #2                 % Added variational components for stable solution';
    HPDFrac            = None #0.8               % High Posterior Density region (fraction of training inputs)';
    ELCBOImproWeight   = None #3                 % Uncertainty weight on ELCBO for computing lower bound improvement';
    TolLength          = None #1e-6              % Minimum fractional length scale';
    CacheSize          = None #500               % Size of cache for storing fcn evaluations';
    CacheFrac          = None #0.5               % Fraction of search points from starting cache (if nonempty)';
    StochasticOptimizer = None #adam             % Stochastic optimizer for varational parameters';
    TolFunStochastic   = None #1e-3              % Stopping threshold for stochastic optimization';
    MaxIterStochastic  = None #100*(2+nvars)     % Max iterations for stochastic optimization';
    GPStochasticStepsize = None #off               % Set stochastic optimization stepsize via GP hyperparameters';
    TolSD              = None #0.1               % Tolerance on ELBO uncertainty for stopping (iff variational posterior is stable)';
    TolsKL             = None #0.01*sqrt(nvars)  % Stopping threshold on change of variational posterior per training point';
    TolStableWarmup    = None #15                % Number of stable fcn evals for stopping warmup';
    VariationalSampler = None #malasample        % MCMC sampler for variational posteriors';
    TolImprovement     = None #0.01              % Required ELCBO improvement per fcn eval before termination';
    KLgauss            = None #yes               % Use Gaussian approximation for symmetrized KL-divergence b\w iters';
    TrueMean           = None #[]                % True mean of the target density (for debugging)';
    TrueCov            = None #[]                % True covariance of the target density (for debugging)';
    MinFunEvals        = None #5*nvars           % Min number of fcn evals';
    MinIter            = None #nvars             % Min number of iterations';
    HeavyTailSearchFrac = None #0.25               % Fraction of search points from heavy-tailed variational posterior';
    MVNSearchFrac      = None #0.25              % Fraction of search points from multivariate normal';
    HPDSearchFrac      = None #0                 % Fraction of search points from multivariate normal fitted to HPD points';
    BoxSearchFrac      = None #0.25              % Fraction of search points from uniform random box based on training inputs';
    SearchCacheFrac    = None #0                 % Fraction of search points from previous iterations';
    AlwaysRefitVarPost = None #no                % Always fully refit variational posterior';
    Warmup             = None #on                % Perform warm-up stage';
    WarmupOptions      = None #[]                % Special OPTIONS struct for warmup stage';
    StopWarmupThresh   = None #0.2               % Stop warm-up when ELCBO increase below threshold (per fcn eval)';
    WarmupKeepThreshold = None #10*nvars         % Max log-likelihood difference for points kept after warmup';
    WarmupKeepThresholdFalseAlarm = None #100*(nvars+2) % Max log-likelihood difference for points kept after a false-alarm warmup stop';
    StopWarmupReliability = None #100            % Reliability index required to stop warmup';
    SearchOptimizer    = None #cmaes             % Optimization method for active sampling';
    SearchCMAESVPInit  = None #yes               % Initialize CMA-ES search SIGMA from variational posterior';
    SearchCMAESbest    = None #no                % Take bestever solution from CMA-ES search';
    SearchMaxFunEvals  = None #500*(nvars+2)     % Max number of acquisition fcn evaluations during search';
    MomentsRunWeight   = None #0.9               % Weight of previous trials (per trial) for running avg of variational posterior moments';
    GPRetrainThreshold = None #1                 % Upper threshold on reliability index for full retraining of GP hyperparameters';
    ELCBOmidpoint      = None #on                % Compute full ELCBO also at best midpoint';
    GPSampleWidths     = None #5                 % Multiplier to widths from previous posterior for GP sampling (Inf = do not use previous widths)';
    HypRunWeight       = None #0.9               % Weight of previous trials (per trial) for running avg of GP hyperparameter covariance';
    WeightedHypCov     = None #on                % Use weighted hyperparameter posterior covariance';
    TolCovWeight       = None #0                 % Minimum weight for weighted hyperparameter posterior covariance';
    GPHypSampler       = None #slicesample       % MCMC sampler for GP hyperparameters';
    CovSampleThresh    = None #10                % Switch to covariance sampling below this threshold of stability index';
    DetEntTolOpt       = None #1e-3              % Optimality tolerance for optimization of deterministic entropy';
    EntropySwitch      = None #off               % Switch from deterministic entropy to stochastic entropy when reaching stability';
    EntropyForceSwitch = None #0.8               % Force switch to stochastic entropy at this fraction of total fcn evals';
    DetEntropyAlpha    = None #0                 % Alpha value for lower/upper deterministic entropy interpolation';
    UpdateRandomAlpha  = None #no                % Randomize deterministic entropy alpha during active sample updates';
    AdaptiveEntropyAlpha = None #no              % Online adaptation of alpha value for lower/upper deterministic entropy interpolation';
    DetEntropyMinD     = None #5                 % Start with deterministic entropy only with this number of vars or more';
    TolConLoss         = None #0.01              % Fractional tolerance for constraint violation of variational parameters';
    BestSafeSD         = None #5                 % SD multiplier of ELCBO for computing best variational solution';
    BestFracBack       = None #0.25              % When computing best solution, lacking stability go back up to this fraction of iterations';
    TolWeight          = None #1e-2              % Threshold mixture component weight for pruning';
    PruningThresholdMultiplier = None #@(K) 1/sqrt(K)   % Multiplier to threshold for pruning mixture weights';
    AnnealedGPMean     = None #@(N,NMAX) 0       % Annealing for hyperprior width of GP negative quadratic mean';
    ConstrainedGPMean  = None #no                % Strict hyperprior for GP negative quadratic mean';
    EmpiricalGPPrior   = None #no                % Empirical Bayes prior over some GP hyperparameters';
    TolGPNoise         = None #sqrt(1e-5)        % Minimum GP observation noise';
    GPLengthPriorMean  = None #sqrt(D/6)         % Prior mean over GP input length scale (in plausible units)';
    GPLengthPriorStd   = None #0.5*log(1e3)      % Prior std over GP input length scale (in plausible units)';
    UpperGPLengthFactor = None #0                % Upper bound on GP input lengths based on plausible box (0 = ignore)';
    InitDesign         = None #plausible         % Initial samples ("plausible" is uniform in the plausible box)';
    gpQuadraticMeanBound = None #yes             % Stricter upper bound on GP negative quadratic mean function';
    Bandwidth          = None #0                 % Bandwidth parameter for GP smoothing (in units of plausible box)';
    FitnessShaping     = None #no                % Heuristic output warping ("fitness shaping")';
    OutwarpThreshBase  = None #10*nvars          % Output warping starting threshold';
    OutwarpThreshMult  = None #1.25              % Output warping threshold multiplier when failed sub-threshold check';
    OutwarpThreshTol   = None #0.8               % Output warping base threshold tolerance (fraction of current threshold)';
    Temperature        = None #1                 % Temperature for posterior tempering (allowed values T = 1,2,3,4)';
    SeparateSearchGP   = None #no                % Use separate GP with constant mean for active search';
    NoiseShaping       = None #no                % Discount observations from from extremely low-density regions';
    NoiseShapingThreshold = None #10*nvars       % Threshold from max observed value to start discounting';
    NoiseShapingFactor = None #0.05              % Proportionality factor of added noise wrt distance from threshold';
    AcqHedge           = None #no                % Hedge on multiple acquisition functions';
    AcqHedgeIterWindow = None #4                 % Past iterations window to judge acquisition fcn improvement';
    AcqHedgeDecay      = None #0.9               % Portfolio value decay per function evaluation';
    ActiveVariationalSamples = None #0           % MCMC variational steps before each active sampling';
    ScaleLowerBound    = None #yes               % Apply lower bound on variational components scale during variational sampling';
    ActiveSampleVPUpdate = None #no              % Perform variational optimization after each active sample';
    ActiveSampleGPUpdate = None #no              % Perform GP training after each active sample';
    ActiveSampleFullUpdatePastWarmup = None #2   % # iters past warmup to continue update after each active sample';
    ActiveSampleFullUpdateThreshold = None #3    % Perform full update during active sampling if stability above threshold';
    VariationalInitRepo = None #no               % Use previous variational posteriors to initialize optimization';
    SampleExtraVPMeans = None #0                 % Extra variational components sampled from GP profile';
    OptimisticVariationalBound = None #0         % Uncertainty weight on ELCBO during active sampling';
    ActiveImportanceSamplingVPSamples   = None #100 % # importance samples from smoothed variational posterior';
    ActiveImportanceSamplingBoxSamples  = None #100 % # importance samples from box-uniform centered on training inputs';
    ActiveImportanceSamplingMCMCSamples = None #100 % # importance samples through MCMC';
    ActiveImportanceSamplingMCMCThin    = None #1   % Thinning for importance sampling MCMC';
    ActiveSamplefESSThresh  = None #1            % fractional ESS threhsold to update GP and VP';
    ActiveImportanceSamplingfESSThresh = None #0.9 % % fractional ESS threhsold to do MCMC while active importance sampling';
    ActiveSearchBound  = None #2                  % Active search bound multiplier';
    IntegrateGPMean    = None #no                   % Try integrating GP mean function';
    TolBoundX          = None #1e-5              % Tolerance on closeness to bound constraints (fraction of total range)';
    RecomputeLCBmax    = None #yes              % Recompute LCB max for each iteration based on current GP estimate';
    BoundedTransform   = None #logit            % Input transform for bounded variables';
    DoubleGP           = None #no                % Use double GP';
    WarpEveryIters     = None #5                 % Warp every this number of iterations';
    IncrementalWarpDelay = None #yes             % Increase delay between warpings';
    WarpTolReliability = None #3                 % Threshold on reliability index to perform warp';
    WarpRotoScaling    = None #yes               % Rotate and scale input';
    WarpCovReg         = None #0                 % Regularization weight towards diagonal covariance matrix for N training inputs';
    WarpRotoCorrThresh = None #0.05              % Threshold on correlation matrix for roto-scaling';
    WarpMinK           = None #5                 % Min number of variational components to perform warp';
    WarpUndoCheck      = None #yes               % Immediately undo warp if not improving ELBO';
    WarpTolImprovement = None #0.1               % Improvement of ELBO required to keep a warp propsal';

    """
    Advanced options for unsupported/untested features (do *not* modify)
    """

    WarpNonlinear      = None #off               % Nonlinear input warping';
    ELCBOWeight        = None #0                 % Uncertainty weight during ELCBO optimization';
    VarParamsBack      = None #0                 % Check variational posteriors back to these previous iterations';
    AltMCEntropy       = None #no                % Use alternative Monte Carlo computation for the entropy';
    VarActiveSample    = None #no                % Variational active sampling';
    FeatureTest        = None #no                % Test a new experimental feature';
    BOWarmup           = None #no                % Bayesian-optimization-like warmup stage';
    gpOutwarpFun       = None #[]                % GP default output warping function';

def _validateOptions():
    '''
    We need the functionality to validate the options
    '''
    pass