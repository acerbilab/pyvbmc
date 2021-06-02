import math
import sys

import numpy as np
from entropy import entlb_vbmc, entub_vbmc
from function_logger import FunctionLogger
from parameter_transformer import ParameterTransformer
from timer import Timer
from variational_posterior import VariationalPosterior

from .options import Options


class VBMC:
    """
    The VBMC algorithm class
    """

    def __init__(
        self,
        fun: callable,
        x0: np.ndarray = None,
        lower_bounds: np.ndarray = None,
        upper_bounds: np.ndarray = None,
        plausible_lower_bounds: np.ndarray = None,
        plausible_upper_bounds: np.ndarray = None,
        user_options: dict = None,
    ):
        # Initialize variables and algorithm structures
        if x0 is None:
            if (
                plausible_lower_bounds is None
                or plausible_upper_bounds is None
            ):
                raise ValueError(
                    """vbmc:UnknownDims If no starting point is
                 provided, PLB and PUB need to be specified."""
                )
            else:
                self.x0 = np.full((plausible_lower_bounds.shape), np.NaN)
        else:
            self.x0 = x0

        self.D = self.x0.shape[1]

        # Empty LB and UB are Infs
        if lower_bounds is None:
            self.lower_bounds = np.ones((1, self.D)) * -np.inf
        else:
            self.lower_bounds = lower_bounds

        if upper_bounds is None:
            self.upper_bounds = np.ones((1, self.D)) * np.inf
        else:
            self.upper_bounds = upper_bounds

        # Check/fix boundaries and starting points
        (
            self.x0,
            self.lower_bounds,
            self.upper_bounds,
            self.plausible_lower_bounds,
            self.plausible_upper_bounds,
        ) = self._boundscheck(
            fun,
            self.x0,
            self.lower_bounds,
            self.upper_bounds,
            plausible_lower_bounds,
            plausible_upper_bounds,
        )

        self.options = Options(
            "./vbmc/option_configs/advanced_vbmc_options.ini",
            evalutation_parameters={"D": self.D},
            user_options=user_options,
        )

        self.K = self.options.get("kwarmup")
        # starting point
        if not np.all(np.isfinite(self.x0)):
            # print('Initial starting point is invalid or not provided.
            # Starting from center of plausible region.\n');
            self.x0 = 0.5 * (
                self.plausible_lower_bounds + self.plausible_upper_bounds
            )

        self.parameter_transformer = ParameterTransformer(
            self.D,
            self.lower_bounds,
            self.upper_bounds,
            self.plausible_lower_bounds,
            self.plausible_upper_bounds,
        )

        noise_flag = None
        uncertainty_handling_level = None
        self.function_logger = FunctionLogger(
            fun,
            self.D,
            noise_flag,
            uncertainty_handling_level,
            self.options.get("cachesize"),
            self.parameter_transformer,
        )

        # Initialize variational posterior
        self.vp = VariationalPosterior(
            D=self.D,
            K=self.K,
            x0=self.x0,
            parameter_transformer=self.parameter_transformer,
        )
        if not self.options.get("warmup"):
            self.vp.optimize_mu = self.options.get("variablemeans")
            self.vp.optimize_weights = self.options.get("variableweights")

        self.optimState = self._init_optimstate(fun)

        self.x0 = self.parameter_transformer(self.x0)

    def _boundscheck(
        self,
        fun: callable,
        x0: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        plausible_lower_bounds: np.ndarray = None,
        plausible_upper_bounds: np.ndarray = None,
    ):
        """
        Private function to do the initial check of the VBMC bounds.
        """

        N0, D = x0.shape

        if plausible_lower_bounds is None or plausible_upper_bounds is None:
            if N0 > 1:
                width = x0.max(0) - x0.min(0)
                if plausible_lower_bounds is None:
                    plausible_lower_bounds = x0.min(0) - width / N0
                    plausible_lower_bounds = np.maximum(
                        plausible_lower_bounds, lower_bounds
                    )
                if plausible_upper_bounds is None:
                    plausible_upper_bounds = x0.max(0) + width / N0
                    plausible_upper_bounds = np.minimum(
                        plausible_upper_bounds, upper_bounds
                    )

                idx = plausible_lower_bounds == plausible_upper_bounds
                if np.any(idx):
                    plausible_lower_bounds[idx] = lower_bounds[idx]
                    plausible_upper_bounds[idx] = upper_bounds[idx]
                    # warning('vbmc:pbInitFailed')
            else:
                # warning('vbmc:pbUnspecified')
                if plausible_lower_bounds is None:
                    plausible_lower_bounds = np.copy(lower_bounds)
                if plausible_upper_bounds is None:
                    plausible_upper_bounds = np.copy(upper_bounds)

        # check that all bounds are row vectors with D elements
        if (
            np.ndim(lower_bounds) != 2
            or np.ndim(upper_bounds) != 2
            or np.ndim(plausible_lower_bounds) != 2
            or np.ndim(plausible_upper_bounds) != 2
            or lower_bounds.shape != (1, D)
            or upper_bounds.shape != (1, D)
            or plausible_lower_bounds.shape != (1, D)
            or plausible_upper_bounds.shape != (1, D)
        ):
            raise ValueError(
                """All input vectors (x0, lower_bounds, upper_bounds,
                 plausible_lower_bounds, plausible_upper_bounds), if specified,
                 need to be row vectors with D elements."""
            )

        # check that plausible bounds are finite
        if np.any(np.invert(np.isfinite(plausible_lower_bounds))) or np.any(
            np.invert(np.isfinite(plausible_upper_bounds))
        ):
            raise ValueError(
                "Plausible interval bounds PLB and PUB need to be finite."
            )

        # Test that all vectors are real-valued
        if (
            np.any(np.invert(np.isreal(x0)))
            or np.any(np.invert(np.isreal(lower_bounds)))
            or np.any(np.invert(np.isreal(upper_bounds)))
            or np.any(np.invert(np.isreal(plausible_lower_bounds)))
            or np.any(np.invert(np.isreal(plausible_upper_bounds)))
        ):
            raise ValueError(
                """All input vectors (x0, lower_bounds, upper_bounds,
                 plausible_lower_bounds, plausible_upper_bounds), if specified,
                 need to be real valued."""
            )

        # Fixed variables (all bounds equal) are not supported
        fixidx = (
            (lower_bounds == upper_bounds)
            & (upper_bounds == plausible_lower_bounds)
            & (plausible_lower_bounds == plausible_upper_bounds)
        )
        if np.any(fixidx):
            raise ValueError(
                """vbmc:FixedVariables VBMC does not support fixed 
            variables. Lower and upper bounds should be different."""
            )

        # Test that plausible bounds are different
        if np.any(plausible_lower_bounds == plausible_upper_bounds):
            raise ValueError(
                """vbmc:MatchingPB:For all variables,
            plausible lower and upper bounds need to be distinct."""
            )

        # Check that all X0 are inside the bounds
        if np.any(x0 < lower_bounds) or np.any(x0 > upper_bounds):
            raise ValueError(
                """vbmc:InitialPointsNotInsideBounds: The starting
            points X0 are not inside the provided hard bounds LB and UB."""
            )

        # % Compute "effective" bounds (slightly inside provided hard bounds)
        bounds_range = upper_bounds - lower_bounds
        bounds_range[np.isinf(bounds_range)] = 1e3
        scale_factor = 1e-3
        realmin = sys.float_info.min
        LB_eff = lower_bounds + scale_factor * bounds_range
        LB_eff[np.abs(lower_bounds) <= realmin] = (
            scale_factor * bounds_range[np.abs(lower_bounds) <= realmin]
        )
        UB_eff = upper_bounds - scale_factor * bounds_range
        UB_eff[np.abs(upper_bounds) <= realmin] = (
            -scale_factor * bounds_range[np.abs(upper_bounds) <= realmin]
        )
        # Infinities stay the same
        LB_eff[np.isinf(lower_bounds)] = lower_bounds[np.isinf(lower_bounds)]
        UB_eff[np.isinf(upper_bounds)] = upper_bounds[np.isinf(upper_bounds)]

        if np.any(LB_eff >= UB_eff):
            raise ValueError(
                """vbmc:StrictBoundsTooClose: Hard bounds LB and UB
                are numerically too close. Make them more separate."""
            )

        # Fix when provided X0 are almost on the bounds -- move them inside
        if np.any(x0 < LB_eff) or np.any(x0 > UB_eff):
            # warning('vbmc:InitialPointsTooClosePB')
            x0 = np.maximum((np.minimum(x0, UB_eff)), LB_eff)

        # Test order of bounds (permissive)
        ordidx = (
            (lower_bounds <= plausible_lower_bounds)
            & (plausible_lower_bounds < plausible_upper_bounds)
            & (plausible_upper_bounds <= upper_bounds)
        )
        if np.any(np.invert(ordidx)):
            raise ValueError(
                """vbmc:StrictBounds: For each variable, hard and
            plausible bounds should respect the ordering LB < PLB < PUB < UB."""
            )

        # Test that plausible bounds are reasonably separated from hard bounds
        if np.any(LB_eff > plausible_lower_bounds) or np.any(
            plausible_upper_bounds > UB_eff
        ):
            # warning('vbmc:TooCloseBounds', ...
            plausible_lower_bounds = np.maximum(plausible_lower_bounds, LB_eff)
            plausible_upper_bounds = np.minimum(plausible_upper_bounds, UB_eff)

        # Check that all X0 are inside the plausible bounds,
        # move bounds otherwise
        if np.any(x0 <= LB_eff) or np.any(x0 >= UB_eff):
            # "warning('vbmc:InitialPointsOutsidePB', ...")
            plausible_lower_bounds = np.minimum(
                plausible_lower_bounds, x0.min(0)
            )
            plausible_upper_bounds = np.maximum(
                plausible_upper_bounds, x0.max(0)
            )

        # Test order of bounds
        ordidx = (
            (lower_bounds < plausible_lower_bounds)
            & (plausible_lower_bounds < plausible_upper_bounds)
            & (plausible_upper_bounds < upper_bounds)
        )
        if np.any(np.invert(ordidx)):
            raise ValueError(
                """vbmc:StrictBounds: For each variable, hard and
            plausible bounds should respect the ordering LB < PLB < PUB < UB."""
            )

        # Check that variables are either bounded or unbounded
        # (not half-bounded)
        if (
            np.any(np.isfinite(lower_bounds))
            and np.any(np.invert(np.isfinite(upper_bounds)))
            or np.any(np.invert(np.isfinite(lower_bounds)))
            and np.any(np.isfinite(upper_bounds))
        ):
            raise ValueError(
                """vbmc:HalfBounds: Each variable needs to be unbounded or
            bounded. Variables bounded only below/above are not supported."""
            )

        return (
            x0,
            lower_bounds,
            upper_bounds,
            plausible_lower_bounds,
            plausible_upper_bounds,
        )

    def _init_optimstate(self, fun):
        """
        A private function to init the optimstate dict that contains infomration
        about VBMC variables.
        """
        # Record starting points (original coordinates)
        y_orig = np.array(self.options.get("fvals")).flatten()
        if len(y_orig) == 0:
            y_orig = np.full([self.x0.shape[0]], np.nan)
        if len(self.x0) != len(y_orig):
            raise ValueError(
                """vbmc:MismatchedStartingInputs The number of
            points in X0 and of their function values as specified in
            self.options.fvals are not the same."""
            )

        optimState = dict()
        optimState["Cache"] = dict()
        optimState["Cache"]["X_orig"] = self.x0
        optimState["Cache"]["y_orig"] = y_orig

        # Integer variables
        optimState["integervars"] = np.full(self.D, False)
        if len(self.options.get("integervars")) > 0:
            integeridx = self.options.get("integervars") != 0
            optimState["integervars"][integeridx] = True
            if (
                np.any(np.isinf(self.lower_bounds[:, integeridx]))
                or np.any(np.isinf(self.upper_bounds[:, integeridx]))
                or np.any(self.lower_bounds[:, integeridx] % 1 != 0.5)
                or np.any(self.upper_bounds[:, integeridx] % 1 != 0.5)
            ):
                raise ValueError(
                    """Hard bounds of integer variables need to be
                 set at +/- 0.5 points from their boundary values (e.g., -0.5 
                 nd 10.5 for a variable that takes values from 0 to 10)"""
                )

        # fprintf('Index of variable restricted to integer values: %s.\n'
        optimState["LB_orig"] = self.lower_bounds
        optimState["UB_orig"] = self.upper_bounds
        optimState["PLB_orig"] = self.plausible_lower_bounds
        optimState["PUB_orig"] = self.plausible_upper_bounds
        eps_orig = (self.upper_bounds - self.lower_bounds) * self.options.get(
            "tolboundx"
        )
        # inf - inf raises warning in numpy, but output is correct
        with np.errstate(invalid="ignore"):
            optimState["LBeps_orig"] = self.lower_bounds + eps_orig
            optimState["UBeps_orig"] = self.upper_bounds - eps_orig

        # Transform variables
        optimState["LB"] = self.parameter_transformer(self.lower_bounds)
        optimState["UB"] = self.parameter_transformer(self.upper_bounds)
        optimState["PLB"] = self.parameter_transformer(
            self.plausible_lower_bounds
        )
        optimState["PUB"] = self.parameter_transformer(
            self.plausible_upper_bounds
        )

        # Before first iteration
        optimState["iter"] = 0

        # Estimate of GP observation noise around the high posterior
        # density region
        optimState["sn2hpd"] = np.inf

        # Does the starting cache contain function values?
        optimState["Cache.active"] = np.any(
            np.isfinite(optimState.get("Cache").get("y_orig"))
        )

        # When was the last warping action performed (number of iterations)
        optimState["LastWarping"] = -np.inf

        # When was the last warping action performed and not undone
        # (number of iterations)
        optimState["LastSuccessfulWarping"] = -np.inf

        # Number of warpings performed
        optimState["WarpingCount"] = 0

        # When GP hyperparameter sampling is switched with optimization
        if self.options.get("nsgpmax") > 0:
            optimState["StopSampling"] = 0
        else:
            optimState["StopSampling"] = np.Inf

        # Fully recompute variational posterior
        optimState["RecomputeVarPost"] = True

        # Start with warm-up?
        optimState["Warmup"] = self.options.get("warmup")
        if self.options.get("warmup"):
            optimState["LastWarmup"] = np.inf
        else:
            optimState["LastWarmup"] = 0

        # Number of stable function evaluations during warmup
        # with small increment
        optimState["WarmupStableCount"] = 0

        # Proposal function for search
        if self.options.get("proposalfcn") is None:
            optimState["ProposalFcn"] = "@(x)proposal_vbmc"
        else:
            optimState["ProposalFcn"] = self.options.get("proposalfcn")

        # Quality of the variational posterior
        optimState["R"] = np.inf

        # Start with adaptive sampling
        optimState["SkipActiveSampling"] = False

        # Running mean and covariance of variational posterior
        # in transformed space
        optimState["RunMean"] = []
        optimState["RunCov"] = []
        # Last time running average was updated
        optimState["LastRunAvg"] = np.NaN

        # Current number of components for variational posterior
        optimState["vpK"] = self.K

        # Number of variational components pruned in last iteration
        optimState["pruned"] = 0

        # Need to switch from deterministic entropy to stochastic entropy
        optimState["EntropySwitch"] = self.options.get("entropyswitch")

        # Only use deterministic entropy if NVARS larger than a fixed number
        if self.D < self.options.get("detentropymind"):
            optimState["EntropySwitch"] = False

        # Tolerance threshold on GP variance (used by some acquisition fcns)
        optimState["TolGPVar"] = self.options.get("tolgpvar")

        # Copy maximum number of fcn. evaluations, used by some acquisition fcns.
        optimState["MaxFunEvals"] = self.options.get("MaxFunEvals")

        # By default, apply variance-based regularization
        # to acquisition functions
        optimState["VarianceRegularizedAcqFcn"] = True

        # Setup search cache
        optimState["SearchCache"] = []

        # Set uncertainty handling level
        # (0: none; 1: unknown noise level; 2: user-provided noise)
        if self.options.get("specifytargetnoise"):
            optimState["UncertaintyHandlingLevel"] = 2
        elif self.options.get("uncertaintyhandling"):
            optimState["UncertaintyHandlingLevel"] = 1
        else:
            optimState["UncertaintyHandlingLevel"] = 0

        # Empty hedge struct for acquisition functions
        if self.options.get("acqhedge"):
            optimState.hedge = []

        # List of points at the end of each iteration
        # Is this required?
        optimState["iterlist"] = dict()
        optimState["iterlist"]["u"] = []
        optimState["iterlist"]["fval"] = []
        optimState["iterlist"]["fsd"] = []
        optimState["iterlist"]["fhyp"] = []

        optimState["delta"] = self.options.get("bandwidth") * (
            optimState.get("PUB") - optimState.get("PLB")
        )

        # Deterministic entropy approximation lower/upper factor
        optimState["entropy_alpha"] = self.options.get("detentropyalpha")

        # Repository of variational solutions
        optimState["vp_repo"] = []

        # Repeated measurement streak
        optimState["RepeatedObservationsStreak"] = 0

        # List of data trimming events
        optimState["DataTrimList"] = []

        if (
            self.options.get("noiseshaping")
            and optimState["gpNoisefun"][2] == 0
        ):
            optimState["gpNoisefun"][2] = 1

        optimState["gpMeanfun"] = self.options.get("gpmeanfun")
        valid_gpmeanfuns = [
            "zero",
            "const",
            "negquad",
            "se",
            "negquadse",
            "negquadfixiso",
            "negquadfix",
            "negquadsefix",
            "negquadonly",
            "negquadfixonly",
            "negquadlinonly",
            "negquadmix",
        ]

        if not optimState["gpMeanfun"] in valid_gpmeanfuns:
            raise ValueError(
                """vbmc:UnknownGPmean:Unknown/unsupported GP mean
            function. Supported mean functions are zero, const,
            egquad, and se"""
            )
        optimState["gntMeanfun"] = self.options.get("gpintmeanfun")
        # more logic here in matlab
        optimState["gpOutwarpfun"] = self.options.get("gpoutwarpfun")

        # Starting threshold on y for output warping
        if (
            self.options.get("fitnessshaping")
            or optimState.get("gpOutwarpfun") is not None
        ):
            optimState["OutwarpDelta"] = self.options.get("outwarpthreshbase")
        else:
            optimState["OutwarpDelta"] = []

        return optimState

    def optimize(self):
        """
        This is a perliminary version of the VBMC loop in order to identify
        possible objects
        """
        pass

    def __1acqhedge_vbmc(self, action, hedge, stats, options):
        """
        ACQPORTFOLIO Evaluate and update portfolio of acquisition functions.
        (unused)
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
