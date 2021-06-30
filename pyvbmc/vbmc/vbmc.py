import sys

import numpy as np
from pyvbmc.function_logger import FunctionLogger
from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.stats.entropy import kldiv_mvn
from pyvbmc.timer import Timer
from pyvbmc.variational_posterior import VariationalPosterior

from .options import Options
from .stats import Stats


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
        """
        Initialize an instance of the VBMC algorithm.

        Parameters
        ----------
        fun : callable
            A given target log posterior FUN.
        x0 : np.ndarray, optional
            [description], by default None
        lower_bounds, upper_bounds : np.ndarray, optional
            Lower_bounds (LB) and upper_bounds (UB) define a set
            of strict lower and upper bounds coordinate vector, X, so that the
            posterior has support on LB < X < UB.
            If scalars, the bound is replicated in each dimension. Use
            empty matrices for LB and UB if no bounds exist. Set LB[i] = -Inf
            and UB[i] = Inf if the i-th coordinate is unbounded (while other
            coordinates may be bounded). Note that if LB and UB contain
            unbounded variables, the respective values of PLB and PUB need to be
            specified (see below), by default None
        plausible_lower_bounds, plausible_upper_bounds : np.ndarray, optional
            Specifies a set of plausible_lower_bounds (PLB) and
            plausible_upper_bounds (PUB) such that LB < PLB < PUB < UB.
            Both PLB and PUB need to be finite. PLB and PUB represent a
            "plausible" range, which should denote a region of high posterior
            probability mass. Among other things, the plausible box is used to
            draw initial samples and to set priors over hyperparameters of the
            algorithm. When in doubt, we found that setting PLB and PUB using
            the topmost ~68% percentile range of the prior (e.g, mean +/- 1 SD
            for a Gaussian prior) works well in many cases (but note that
            additional information might afford a better guess), both are
            by default None.
        user_options : dict, optional
            Modified options can be passed as a dict. Please refer to the
            respective VBMC options page for the default options. If no
            user_options are passed, the default options are used.

        Raises
        ------
        ValueError
            [description]
        """
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
                x0 = np.full((plausible_lower_bounds.shape), np.NaN)

        self.D = x0.shape[1]

        # Empty LB and UB are Infs
        if lower_bounds is None:
            lower_bounds = np.ones((1, self.D)) * -np.inf

        if upper_bounds is None:
            upper_bounds = np.ones((1, self.D)) * np.inf

        # Check/fix boundaries and starting points
        (
            self.x0,
            self.lower_bounds,
            self.upper_bounds,
            self.plausible_lower_bounds,
            self.plausible_upper_bounds,
        ) = self._boundscheck(
            x0,
            lower_bounds,
            upper_bounds,
            plausible_lower_bounds,
            plausible_upper_bounds,
        )

        self.options = Options(
            "./pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini",
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

        self.optim_state = self._init_optim_state()

        self.function_logger = FunctionLogger(
            fun=fun,
            D=self.D,
            noise_flag=self.optim_state.get("uncertainty_handling_level") > 0,
            uncertainty_handling_level=self.optim_state.get(
                "uncertainty_handling_level"
            ),
            cache_size=self.options.get("cachesize"),
            parameter_transformer=self.parameter_transformer,
        )

        self.x0 = self.parameter_transformer(self.x0)

        self.stats = Stats()

    def _boundscheck(
        self,
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

    def _init_optim_state(self):
        """
        A private function to init the optim_state dict that contains
        information about VBMC variables.
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

        optim_state = dict()
        optim_state["cache"] = dict()
        optim_state["cache"]["x_orig"] = self.x0
        optim_state["cache"]["y_orig"] = y_orig

        # Does the starting cache contain function values?
        optim_state["cache_active"] = np.any(
            np.isfinite(optim_state.get("cache").get("y_orig"))
        )

        # Integer variables
        optim_state["integervars"] = np.full(self.D, False)
        if len(self.options.get("integervars")) > 0:
            integeridx = self.options.get("integervars") != 0
            optim_state["integervars"][integeridx] = True
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
        optim_state["lb_orig"] = self.lower_bounds
        optim_state["ub_orig"] = self.upper_bounds
        optim_state["plb_orig"] = self.plausible_lower_bounds
        optim_state["pub_orig"] = self.plausible_upper_bounds
        eps_orig = (self.upper_bounds - self.lower_bounds) * self.options.get(
            "tolboundx"
        )
        # inf - inf raises warning in numpy, but output is correct
        with np.errstate(invalid="ignore"):
            optim_state["lb_eps_orig"] = self.lower_bounds + eps_orig
            optim_state["ub_eps_orig"] = self.upper_bounds - eps_orig

        # Transform variables (Transform of lower_bounds and upper bounds can
        # create warning but we are aware of this and output is correct)
        with np.errstate(divide="ignore"):
            optim_state["lb"] = self.parameter_transformer(self.lower_bounds)
            optim_state["ub"] = self.parameter_transformer(self.upper_bounds)
        optim_state["plb"] = self.parameter_transformer(
            self.plausible_lower_bounds
        )
        optim_state["pub"] = self.parameter_transformer(
            self.plausible_upper_bounds
        )

        # Before first iteration
        optim_state["iter"] = 0

        # Estimate of GP observation noise around the high posterior
        # density region
        optim_state["sn2hpd"] = np.inf

        # When was the last warping action performed (number of iterations)
        optim_state["last_warping"] = -np.inf

        # When was the last warping action performed and not undone
        # (number of iterations)
        optim_state["last_successful_warping"] = -np.inf

        # Number of warpings performed
        optim_state["warping_count"] = 0

        # When GP hyperparameter sampling is switched with optimization
        if self.options.get("nsgpmax") > 0:
            optim_state["stop_sampling"] = 0
        else:
            optim_state["stop_sampling"] = np.Inf

        # Fully recompute variational posterior
        optim_state["recompute_var_post"] = True

        # Start with warm-up?
        optim_state["warmup"] = self.options.get("warmup")
        if self.options.get("warmup"):
            optim_state["last_warmup"] = np.inf
        else:
            optim_state["last_warmup"] = 0

        # Number of stable function evaluations during warmup
        # with small increment
        optim_state["warmup_stable_count"] = 0

        # Proposal function for search
        if self.options.get("proposalfcn") is None:
            optim_state["proposalfcn"] = "@(x)proposal_vbmc"
        else:
            optim_state["proposalfcn"] = self.options.get("proposalfcn")

        # Quality of the variational posterior
        optim_state["r"] = np.inf

        # Start with adaptive sampling
        optim_state["skip_active_sampling"] = False

        # Running mean and covariance of variational posterior
        # in transformed space
        optim_state["run_mean"] = []
        optim_state["run_cov"] = []
        # Last time running average was updated
        optim_state["last_run_avg"] = np.NaN

        # Current number of components for variational posterior
        optim_state["vpk"] = self.K

        # Number of variational components pruned in last iteration
        optim_state["pruned"] = 0

        # Need to switch from deterministic entropy to stochastic entropy
        optim_state["entropy_switch"] = self.options.get("entropyswitch")

        # Only use deterministic entropy if D larger than a fixed number
        if self.D < self.options.get("detentropymind"):
            optim_state["entropy_switch"] = False

        # Tolerance threshold on GP variance (used by some acquisition fcns)
        optim_state["tol_gp_var"] = self.options.get("tolgpvar")

        # Copy maximum number of fcn. evaluations,
        # used by some acquisition fcns.
        optim_state["max_fun_evals"] = self.options.get("maxfunevals")

        # By default, apply variance-based regularization
        # to acquisition functions
        optim_state["variance_regularized_acqfcn"] = True

        # Setup search cache
        optim_state["search_cache"] = []

        # Set uncertainty handling level
        # (0: none; 1: unknown noise level; 2: user-provided noise)
        if self.options.get("specifytargetnoise"):
            optim_state["uncertainty_handling_level"] = 2
        elif len(self.options.get("uncertaintyhandling")) == 0:
            optim_state["uncertainty_handling_level"] = 1
        else:
            optim_state["uncertainty_handling_level"] = 0

        # Empty hedge struct for acquisition functions
        if self.options.get("acqhedge"):
            optim_state["hedge"] = []

        # List of points at the end of each iteration
        optim_state["iterlist"] = dict()
        optim_state["iterlist"]["u"] = []
        optim_state["iterlist"]["fval"] = []
        optim_state["iterlist"]["fsd"] = []
        optim_state["iterlist"]["fhyp"] = []

        optim_state["delta"] = self.options.get("bandwidth") * (
            optim_state.get("pub") - optim_state.get("plb")
        )

        # Deterministic entropy approximation lower/upper factor
        optim_state["entropy_alpha"] = self.options.get("detentropyalpha")

        # Repository of variational solutions
        optim_state["vp_repo"] = []

        # Repeated measurement streak
        optim_state["repeated_observations_streak"] = 0

        # List of data trimming events
        optim_state["data_trim_list"] = []

        # Expanding search bounds
        prange = optim_state.get("pub") - optim_state.get("plb")
        optim_state["lb_search"] = np.maximum(
            optim_state.get("plb")
            - prange * self.options.get("activesearchbound"),
            optim_state.get("lb"),
        )
        optim_state["ub_search"] = np.minimum(
            optim_state.get("pub")
            + prange * self.options.get("activesearchbound"),
            optim_state.get("ub"),
        )

        # Initialize Gaussian process settings
        # Squared exponential kernel with separate length scales
        optim_state["gp_covfun"] = 1

        if optim_state.get("uncertainty_handling_level") == 0:
            # Observation noise for stability
            optim_state["gp_noisefun"] = [1, 0]
        elif optim_state.get("uncertainty_handling_level") == 1:
            # Infer noise
            optim_state["gp_noisefun"] = [1, 2]
        elif optim_state.get("uncertainty_handling_level") == 2:
            # Provided heteroskedastic noise
            optim_state["gp_noisefun"] = [1, 1]

        if (
            self.options.get("noiseshaping")
            and optim_state["gp_noisefun"][1] == 0
        ):
            optim_state["gp_noisefun"][1] = 1

        optim_state["gp_meanfun"] = self.options.get("gpmeanfun")
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

        if not optim_state["gp_meanfun"] in valid_gpmeanfuns:
            raise ValueError(
                """vbmc:UnknownGPmean:Unknown/unsupported GP mean
            function. Supported mean functions are zero, const,
            egquad, and se"""
            )
        optim_state["int_meanfun"] = self.options.get("gpintmeanfun")
        # more logic here in matlab
        optim_state["gp_outwarpfun"] = self.options.get("gpoutwarpfun")

        # Starting threshold on y for output warping
        if (
            self.options.get("fitnessshaping")
            or optim_state.get("gp_outwarpfun") is not None
        ):
            optim_state["outwarp_delta"] = self.options.get(
                "outwarpthreshbase"
            )
        else:
            optim_state["outwarp_delta"] = []

        return optim_state

    def optimize(self):
        """
        Execute the VBMC loop. TBD.
        """
        is_finished = False
        # the iterations of pyvbmc start at 0
        iteration = -1
        timer = Timer()
        gp = None

        while not is_finished:
            iteration += 1
            self.optim_state["iter"] = iteration
            self.optim_state["redo_roto_scaling"] = False

            if self.optim_state.get("entropy_switch") and (
                self.optim_state.get("func_count")
                >= self.optim_state.get("entropy_force_switch")
                * self.optim_state.get("max_fun_evals")
            ):
                self.optim_state["entropy_switch"] = False

            # Actively sample new points into the training set
            timer.start_timer("activeSampling")

            if iteration == 0:
                new_funevals = self.options.get("funevalstart")
            else:
                new_funevals = self.options.get("funevalsperiter")

            # if optimState.Xn > 0:
            #     optimState.ymax = max(optimState.y(optimState.X_flag))

            if self.optim_state.get("skipactivesampling"):
                self.optim_state["skipactivesampling"] = False
            else:
                if (
                    gp is not None
                    and self.options.get("separatesearchgp")
                    and not self.options.get("varactivesample")
                ):
                    # Train a distinct GP for active sampling
                    if iteration % 2 == 0:
                        meantemp = self.optim_state.get("gp_meanfun")
                        self.optim_state["gp_meanfun"] = "const"
                        gp_search = self._train_gp()
                        self.optim_state["gp_meanfun"] = meantemp
                    else:
                        gp_search = gp
                else:
                    gp_search = gp

                # Perform active sampling
                if self.options.get("varactivesample"):
                    # FIX TIMER HERE IF USING THIS
                    # [optimState,vp,t_active,t_func] =
                    # variationalactivesample_vbmc(optimState,new_funevals,
                    # funwrapper,vp,vp_old,gp_search,options)
                    sys.exit("Function currently not supported")
                else:
                    self._activesample(new_funevals)

            # optimState.N = optimState.Xn  # Number of training inputs
            # optimState.Neff = sum(optimState.nevals(optimState.X_flag))

            timer.stop_timer("activeSampling")

            # train gp

            timer.start_timer("gpTrain")

            Ns_gp = self._train_gp()

            timer.stop_timer("gpTrain")

            # Check if reached stable sampling regime
            if (
                Ns_gp == self.options.get("stablegpsamples")
                and self.optim_state.get("stop_sampling") == 0
            ):
                self.optim_state["stop_sampling"] = self.optim_state.get("N")

            # Optimize variational parameters
            timer.start_timer("variationalFit")

            if not self.vp.optimize_mu:
                # Variational components fixed to training inputs
                self.vp.mu = gp.X.T
                Knew = self.vp.mu.shape[1]
            else:
                # Update number of variational mixture components
                Knew = self._updateK()

            # Decide number of fast/slow optimizations
            N_fastopts = 3
            # np.ceil(evaloption_vbmc(options.NSelbo, K))

            if self.optim_state.get("recompute_varpost") or (
                self.options.get("alwaysrefitvarpost")
            ):
                # Full optimizations
                N_slowopts = self.options.get("elbostarts")
                self.optim_state["recompute_varpost"] = False
            else:
                # Only incremental change from previous iteration
                N_fastopts = np.ceil(
                    N_fastopts * self.options.get("nselboincr")
                )
                N_slowopts = 1

                # Run optimization of variational parameters
                varss, pruned = self._optimize_vp(
                    N_fastopts,
                    N_slowopts,
                    Knew,
                )
                # optimState.vp_repo{end+1} = get_vptheta(vp)

            self.optim_state["vpK"] = self.vp.K
            # Save current entropy
            self.optim_state["H"] = self.vp  # .stats.entropy

            # Get real variational posterior (might differ from training posterior)
            # vp_real = vp.vptrain2real(0, self.options)
            vp_real = self.vp
            elbo = vp_real  # .stats.elbo
            elbo_sd = vp_real  # .stats.elbo_sd

            timer.stop_timer("variationalFit")

            # Finalize iteration

            timer.start_timer("finalize")

            # Compute symmetrized KL-divergence between old and new posteriors
            Nkl = 1e5

            # remove later
            vp_old = self.vp
            sKL = max(
                0,
                0.5
                * np.sum(
                    self.vp.kldiv(
                        vp2=vp_old,
                        N=Nkl,
                        gaussflag=self.options.get("klgauss"),
                    )
                ),
            )

            # Evaluate max LCB of GP prediction on all training inputs
            # _, _, fmu, fs2 = GP_Lite.gplite_pred(gp, gp.X, gp.y, gp.s2)
            fmu = [3, 3, 3]
            fs2 = [4]
            self.optim_state["lcbmax"] = np.max(
                fmu - self.options.get("elcboimproweight") * np.sqrt(fs2)
            )

            # Compare variational posterior's moments with ground truth
            if (
                self.options.get("truemean")
                and self.options.get("truecov")
                and np.all(np.isfinite(self.options.get("truemean")))
                and np.all(np.isfinite(self.options.get("truecov")))
            ):
                mubar_orig, sigma_orig = vp_real.moments(1e6, True, True)

                kl = kldiv_mvn(
                    mubar_orig,
                    sigma_orig,
                    self.options.get("truemean"),
                    self.options.get("truecov"),
                )
                sKL_true = 0.5 * np.sum(kl)
            else:
                sKL_true = None

            # Record moments in transformed space
            mubar, sigma = self.vp.moments(origflag=False, covflag=True)
            if len(self.optim_state.get("run_mean")) == 0 or len(
                self.optim_state.get("run_cov") == 0
            ):
                self.optim_state["run_mean"] = mubar.reshape(1, -1)
                self.optim_state["run_cov"] = sigma
                self.optim_state["last_run_avg"] = self.optim_state.get("N")
            else:
                Nnew = self.optim_state.get("N") - self.optim_state.get(
                    "last_run_avg"
                )
                wRun = self.options.get("momentsrunweight ") ** Nnew
                self.optim_state["run_mean"] = wRun * self.optim_state.get(
                    "run_mean"
                ) + (1 - wRun) * mubar.reshape(1, -1)
                self.optim_state["run_cov"] = (
                    wRun * self.optim_state.get("run_cov") + (1 - wRun) * sigma
                )
                self.optim_state["last_run_avg"] = self.optim_state.get("N")

            timer.stop_timer("finalize")
            # timer.totalruntime = NaN;   # Update at the end of iteration
            # timer

            # Record all useful stats
            self.stats.record_iteration(
                self.optim_state,
                self.vp,
                elbo,
                elbo_sd,
                varss,
                sKL,
                sKL_true,
                gp,
                Ns_gp,
                pruned,
                timer,
            )

            # stats.warmup[loopiter] = optimState.Warmup

            # Check warmup
            if (
                self.optim_state.get("iter") > 2
                and self.optim_state.get("stop_gp_sampling") == 0
                and not self.optim_state.get("warmup")
            ):
                if self._is_gp_sampling_finished():
                    self.optim_state[
                        "stop_gp_sampling"
                    ] = self.optim_state.get("N")

            # Check termination conditions
            # is_finished = self._check_termination_conditions()

            #  Save stability
            # vp.stats.stable = stats.stable(optimState.iter)

            # Check if we are still warming-up
            if self.optim_state.get("warmup") and iteration > 0:
                if self.options.get("recomputelcbmax"):
                    self.optim_state["lcbmax_vec"] = self._recompute_lcbmax().T
                trim_flag = self._check_warmup_end()
                if trim_flag:
                    # Re-update GP after trimming
                    gp = self._reupdate_gp(gp)
                if not self.optim_state.get("warmup"):
                    self.vp.optimize_mu = self.options.get("variablemeans")
                    self.vp.optimize_weights = self.options.get(
                        "variableweights"
                    )

                    # Switch to main algorithm options
                    # options = options_main
                    # Reset GP hyperparameter covariance
                    # hypstruct.runcov = []
                    # Reset VP repository
                    self.optim_state["vp_repo"] = []
                    # Re-get acq info
                    # self.optim_state['acqInfo'] = getAcqInfo(
                    #    options.SearchAcqFcn
                    # )

            # Check and update fitness shaping / output warping threshold
            if (
                self.optim_state.get("outwarp_delta") is not None
                and self.optim_state.get("R") is not None
                and (
                    self.optim_state.get("R")
                    < self.options.get("warptolreliability")
                )
            ):
                Xrnd = self.vp.sample(N=int(2e4), origflag=False)
                ymu = gp.gplite_pred(gp, Xrnd, [], [], 0, 1)
                ydelta = max(
                    [0, self.optim_state["ymax"] - np.quantile(ymu, 1e-3)]
                )
                if (
                    ydelta
                    > self.optim_state.get("outwarp_delta")
                    * self.options.get("outwarpthreshtol")
                    and self.optim_state.get("R") is not None
                    and self.optim_state.get("R") < 1
                ):
                    self.optim_state["outwarp_delta"] = self.optim_state.get(
                        "outwarp_delta"
                    ) * self.options.get("outwarpthreshmult")

            # Write iteration output

            # Pick "best" variational solution to return (and real vp, if train vp differs)
            # idx_best = self._determine_best_vp()
            idx_best = 1
            # Last variational optimization with large number of components
            changed_flag = self._finalboost(idx_best)
            # remove later
            is_finished = iteration > 2

    # active sampling
    def _activesample(self, new_funevals):
        pass

    def _reupdate_gp(self, gp):
        """
        Quick posterior reupdate of Gaussian process.

        Wait for interface of GPlite before implementing.
        """
        return gp

    # GP Training

    def _train_gp(self):
        """
        Train Gaussian process model.

        Wait for interface of GPlite before implementing.
        """
        ns_gp = 10
        gp = {}
        return ns_gp, gp

    # Variational optimization / training of variational posterior:

    def _updateK(self):
        """
        Update number of variational mixture components.
        """
        return self.vp.K

    def _optimize_vp(self, Nfastopts, Nslowopts, K=None):
        """
        Optimize variational posterior.
        """
        # use interface for vp optimzation?
        if K is None:
            K = self.vp.K
        varss = []
        pruned = 0
        return varss, pruned

    # Loop termination:

    def _check_warmup_end(self):
        """
        vbmc_warmup.m
        optim_state, stats, action, options
        check if warmup ends
        """
        return True

    def _check_termination_conditions(self):
        """
        Private method to determine the status of termination conditions.

        It also saves the reliability index, ELCBO improvement and stableflag
        to the iteration_history object.
        """
        isFinished_flag = False

        # Maximum number of new function evaluations
        if self.optim_state.get("func_count") >= self.options.get(
            "maxfunevals"
        ):
            isFinished_flag = True
            # msg "Inference terminated

        # Maximum number of iterations
        iteration = self.optim_state.get("iter")
        if iteration >= self.options.get("maxiter"):
            isFinished_flag = True
            # msg = "Inference terminated

        # Quicker stability check for entropy switching
        if self.optim_state.get("entropy_switch"):
            tol_stable_iters = self.options.get("tolstableentropyiters")
        else:
            tol_stable_iters = int(
                np.ceil(
                    self.options.get("tolstablecount")
                    / self.options.get("funevalsperiter")
                )
            )

        rindex, ELCBO_improvement = self._compute_reliability_index(
            tol_stable_iters
        )

        # Store reliability index
        # this will be improved with the new iteration_history object.
        if "rindex" in self.stats:
            if len(self.stats["rindex"]) > iteration:
                self.stats["rindex"][iteration] = rindex
            else:
                self.stats["rindex"] = np.append(
                    self.stats.get("rindex"), [rindex]
                )

        if "elcbo_impro" in self.stats:
            if len(self.stats["elcbo_impro"]) > iteration:
                self.stats["elcbo_impro"][iteration] = ELCBO_improvement
            else:
                self.stats["elcbo_impro"] = np.append(
                    self.stats.get("elcbo_impro"), [ELCBO_improvement]
                )
        self.optim_state["R"] = rindex

        # Check stability termination condition
        stableflag = False
        if (
            iteration >= tol_stable_iters
            and rindex < 1
            and ELCBO_improvement < self.options.get("tolimprovement")
        ):
            # Count how many good iters in the recent past (excluding current)
            stable_count = np.sum(
                self.stats.get("rindex")[
                    iteration - tol_stable_iters : iteration - 2
                ]
                < 1
            )
            # Iteration is stable if almost all recent iterations are stable
            if (
                stable_count
                >= tol_stable_iters
                - np.floor(
                    tol_stable_iters * self.options.get("tolstableexcptfrac")
                )
                - 1
            ):
                if self.optim_state.get("entropy_switch"):
                    # If stable but entropy switch is On,
                    # turn it off and continue
                    self.optim_state["entropy_switch"] = False
                else:
                    isFinished_flag = True
                    stableflag = True
                    # "msg = 'Inference terminated:"

        # Store stability flag
        # this will be improved with the new iteration_history object.
        if "stable" in self.stats:
            if len(self.stats["stable"]) > iteration:
                self.stats["stable"][iteration] = stableflag
            else:
                self.stats["stable"] = np.append(
                    self.stats.get("stable"), [stableflag]
                )
        # Prevent early termination
        if self.optim_state.get("func_count") < self.options.get(
            "minfunevals"
        ) or iteration < self.options.get("miniter"):
            isFinished_flag = False

        return isFinished_flag

    def _compute_reliability_index(self, tol_stable_iters):
        """
        Private function to compute the reliability index.
        """
        iteration_idx = self.optim_state.get("iter")
        if self.optim_state.get("iter") < 3:
            rindex = np.Inf
            ELCBO_improvement = np.NaN
            return rindex, ELCBO_improvement

        sn = np.sqrt(self.optim_state.get("sn2hpd"))
        tol_sn = np.sqrt(sn / self.options.get("tolsd")) * self.options.get(
            "tolsd"
        )
        tol_sd = min(
            max(self.options.get("tolsd"), tol_sn),
            self.options.get("tolsd") * 10,
        )

        rindex_vec = np.full((3), np.NaN)
        rindex_vec[0] = (
            np.abs(
                self.stats.get("elbo")[iteration_idx]
                - self.stats.get("elbo")[iteration_idx - 1]
            )
            / tol_sd
        )
        rindex_vec[1] = self.stats.get("elbo_sd")[iteration_idx] / tol_sd
        rindex_vec[2] = self.stats.get("sKL")[
            iteration_idx
        ] / self.options.get("tolskl")

        # Compute average ELCBO improvement per fcn eval in the past few iters
        idx0 = int(
            max(
                1,
                self.optim_state.get("iter")
                - np.ceil(0.5 * tol_stable_iters)
                + 1,
            )
            - 1
        )
        xx = self.stats.get("funccount")[idx0:iteration_idx]
        yy = (
            self.stats.get("elbo")[idx0:iteration_idx]
            - self.options.get("elcboimproweight")
            * self.stats.get("elbo_sd")[idx0:iteration_idx]
        )
        ELCBO_improvement = np.polyfit(xx, yy, 1)[0]
        return np.mean(rindex_vec), ELCBO_improvement

    def _is_gp_sampling_finished(self):
        """
        Private function to check if the MCMC sampling of the Gaussian Process
        is finished.
        """
        finished_flag = False
        # Stop sampling after sample variance has stabilized below ToL
        iteration = self.optim_state.get("iter")

        w1 = np.zeros((iteration + 1))
        w1[iteration] = 1
        w2 = np.exp(-(self.stats.get("N")[-1] - self.stats.get("N") / 10))
        w2 = w2 / np.sum(w2)
        w = 0.5 * w1 + 0.5 * w2
        if np.sum(w * self.stats.get("gp_sample_var")) < self.options.get(
            "tolgpvarmcmc"
        ):
            finished_flag = True

        return finished_flag

    def _recompute_lcbmax(self):
        """
        RECOMPUTE_LCBMAX Recompute moving LCB maximum based on current GP.
        """
        return np.array([])

    # Finalizing:

    def _finalboost(self, idx_best):
        """
        FINALBOOST_VBMC Final boost of variational components.
        """
        changed_flag = True
        return changed_flag

    def _determine_best_vp(
        self,
        max_idx: int = None,
        safe_sd: float = 5,
        frac_back: float = 0.25,
        rank_citerion_flag: bool = False,
    ):
        """
        Return best variational posterior from the iteration_history object.

        Parameters
        ----------
        max_idx : int, optional
            Check up to this iteration (default, last), by default None
        safe_sd : float, optional
            Penalization for uncertainty, by default 5
        frac_back : float, optional
            If no past stable iteration, go back up to this fraction of
            iterations, by default 0.25
        rank_citerion_flag : bool, optional
            If True use new ranking criterion method to pick best solution.
            It finds a solution that combines ELCBO, stability, and recency,
            by default False

        Returns
        -------
        [type]
            [description]
        """

        # Check up to this iteration (default, last)
        if max_idx is None:
            max_idx = self.stats.get("iter")[-1]

        if self.stats.get("stable")[max_idx]:
            # If the current iteration is stable, return it
            idx_best = max_idx

        else:
            # Otherwise, find best solution according do various criteria

            if rank_citerion_flag:
                # Find solution that combines ELCBO, stability, and recency

                rank = np.zeros((max_idx + 1, 4))
                # Rank by position
                rank[:, 0] = np.arange(1, max_idx + 2)[::-1]

                # Rank by ELCBO
                lnZ_iter = self.stats.get("elbo")[: max_idx + 1]
                lnZsd_iter = self.stats.get("elbo_sd")[: max_idx + 1]
                elcbo = lnZ_iter - safe_sd * lnZsd_iter
                order = elcbo.argsort()[::-1]
                rank[order, 1] = np.arange(1, max_idx + 2)

                # Rank by reliability index
                order = self.stats.get("rindex")[: max_idx + 1].argsort()
                rank[order, 2] = np.arange(1, max_idx + 2)

                # Rank penalty to all non-stable iterations
                rank[:, 3] = max_idx
                rank[self.stats.get("stable")[: max_idx + 1], 3] = 1

                idx_best = np.argmin(np.sum(rank, 1))

            else:
                # Find recent solution with best ELCBO
                laststable = np.argwhere(
                    self.stats.get("stable")[: max_idx + 1] == True
                )
                if len(laststable) == 0:
                    idx_start = max(
                        1, int(np.ceil(max_idx - max_idx * frac_back))
                    )
                else:
                    idx_start = np.ravel(laststable)[-1]

                lnZ_iter = self.stats.get("elbo")[idx_start : max_idx + 1]
                lnZsd_iter = self.stats.get("elbo_sd")[idx_start : max_idx + 1]
                elcbo = lnZ_iter - safe_sd * lnZsd_iter
                idx_best = idx_start + np.argmax(elcbo)

        # Return best variational posterior, its ELBO and SD
        vp = self.stats.get("vp")[idx_best]
        elbo = self.stats.get("elbo")[idx_best]
        elbo_sd = self.stats.get("elbo_sd")[idx_best]

        return vp, elbo, elbo_sd, idx_best
