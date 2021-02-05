from .options_vbmc import Options_VBMC


class OptimState(object):
    """
    Contains the current state of the VMBC algorithm
    """

    def __init__(self, options: Options_VBMC, K, LB, UB, PLB, PUB) -> None:

        nvars = size(LB, 2)

        # Integer variables
        self.integervars = false(1, nvars)
        # if options.IntegerVars is not None:
        #     self.integervars(options.IntegerVars) = True
        #     for d = find(optimState.integervars)
        # if (~isfinite(LB(d)) && floor(LB(d)) ~= 0.5) || ...
        #         (~isfinite(UB(d)) && floor(UB(d)) ~= 0.5)
        #         error('Hard bounds of integer variables need to be set at +/- 0.5 points from their boundary values (e.g., -0.5 and 10.5 for a variable that takes values from 0 to 10).');

        self.LB = LB
        self.UB = UB
        self.PLB = PLB
        self.PUB = PUB

        # iteration number
        self.iter: int = 0

        # Estimate of GP observation noise around the high posterior density region
        self.sn2hpd: float = float("inf")

        # Fully recompute variational posterior
        self.RecomputeVarPost: bool = True

        # When was the last warping action performed (number of training inputs)
        self.LastWarping: float = float("-inf")

        # Number of warpings performed
        self.WarpingCount: int = 0

        # When GP hyperparameter sampling is switched with optimization
        if options.NSgpMax > 0:
            self.StopSampling: float = 0
        else:
            self.StopSampling: float = float("inf")

        # Fully recompute variational posterior
        self.RecomputeVarPost: bool = True

        # Start with warm-up?
        self.Warmup: bool = options.Warmup

        if self.Warmup:
            self.LastWarmup: float = float("inf")
        else:
            self.LastWarmup: float = 0

        # Number of stable function evaluations during warmup with small increment
        self.WarmupStableCount: int = 0

        # Proposal function for search
        if options.ProposalFcn is None:
            self.ProposalFcn = (
                None  # @(x) proposal_vbmc(x,optimState.PLB,optimState.PUB)
            )
        else:
            self.ProposalFcn = options.ProposalFcn

        # Quality of the variational posterior
        self.R: float = float("inf")

        # Start with adaptive sampling
        self.SkipActiveSampling = False

        # Running mean and covariance of variational posterior in transformed space
        self.RunMean = []
        self.RunCov = []

        # Last time running average was updated
        self.LastRunAvg: float = float("nan")

        # Current number of components for variational posterior
        self.vpK: int = K

        # Number of variational components pruned in last iteration
        self.pruned: int = 0

        # Need to switch from deterministic entropy to stochastic entropy
        self.EntropySwitch: bool = options.EntropySwitch

        # Only use deterministic entropy if NVARS larger than a fixed number
        if nvars < options.DetEntropyMinD:
            self.EntropySwitch: bool = False

        # Tolerance threshold on GP variance (used by some acquisition fcns)
        self.TolGPVar = options.TolGPVar

        # Copy maximum number of fcn. evaluations, used by some acquisition fcns.
        self.MaxFunEvals = options.MaxFunEvals

        # By default, apply variance-based regularization to acquisition functions
        self.VarianceRegularizedAcqFcn: bool = True

        # Setup search cache
        self.SearchCache = []

        # Set uncertainty handling level
        # (0: none; 1: unkown noise level; 2: user-provided noise)
        if options.SpecifyTargetNoise:
            self.UncertaintyHandlingLevel: int = 2
        elif options.UncertaintyHandling:
            self.UncertaintyHandlingLevel: int = 1
        else:
            self.UncertaintyHandlingLevel: int = 0

        # Empty hedge struct for acquisition functions
        if options.AcqHedge:
            self.hedge = []

        # List of points at the end of each iteration
        self.iterList.u = []
        self.iterList.fval = []
        self.iterList.fsd = []
        self.iterList.fhyp = []

        self.delta = options.Bandwidth * (self.PUB - self.PLB)

        # Posterior tempering temperature
        if options.Temperature is not None:
            self.temperature: int = options.Temperature
        else:
            self.temperature: int = 1

        # Deterministic entropy approximation lower/upper factor
        self.entropy_alpha = options.DetEntropyAlpha

        # Repository of variational solutions
        self.vp_repo = []

        # Repeated measurement streak
        self.RepeatedObservationsStreak = 0

        # List of data trimming events
        self.DataTrimList = []

        # Expanding search bounds
        prange = self.PUB - self.PLB
        self.LB_search = max(
            self.PLB - prange * options.ActiveSearchBound, self.LB
        )
        self.UB_search = min(
            self.PUB + prange * options.ActiveSearchBound, self.UB
        )

        # Does the starting cache contain function values?
        self.Cache.active: bool = (
            None  # any(isfinite(optimState.Cache.y_orig))
        )

        self.acqInfo = None

        self.cachecount = None

        self.funccount = None
        self.gpMeanfun = None
        self.gpOutwarpfun = None
        self.H = None
        self.hypstruct = None

        self.lcbmax = None
        self.lcbmax_vec = None

        self.N = None
        self.Neff = None
        self.nevals = None
        self.OutwarpDelta = None

        self.totalfunevaltime = None
        self.totaltime = None

        self.Xn = None
        self.X_flag = None
        self.ymax = None
        self.y_orig = None
