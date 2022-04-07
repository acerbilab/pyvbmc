import copy
import logging
import math
import os
import sys

import gpyreg as gpr
import matplotlib.pyplot as plt
import numpy as np
from pyvbmc.function_logger import FunctionLogger
from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.stats import kldiv_mvn
from pyvbmc.timer import Timer
from pyvbmc.variational_posterior import VariationalPosterior

from .active_sample import active_sample
from .gaussian_process_train import reupdate_gp, train_gp
from .iteration_history import IterationHistory
from .options import Options
from .variational_optimization import optimize_vp, update_K

from pyvbmc.whitening import warp_input_vbmc, warp_gpandvp_vbmc


class VBMC:
    """
    Posterior and model inference via Variational Bayesian Monte Carlo (VBMC).
    
    VBMC computes a variational approximation of the full posterior and a lower 
    bound on the normalization constant (marginal likelhood or model evidence) 
    for a provided unnormalized log posterior. 

    Initialize a ``VBMC`` object to set up the inference problem, then run
    ``optimize()``. See the examples for more details.

    Parameters
    ----------
    fun : callable
        A given target log posterior `fun`. `fun` accepts input `x` and returns 
        the value of the target log-joint, that is the unnormalized 
        log-posterior density, at `x`.
    x0 : np.ndarray, optional
        Starting point for the inference. Ideally `x0` is a point in the
        proximity of the mode of the posterior. Default is ``None``.
    lower_bounds, upper_bounds : np.ndarray, optional
        `lower_bounds` (`LB`) and `upper_bounds` (`UB`) define a set
        of strict lower and upper bounds for the coordinate vector, `x`, so 
        that the posterior has support on `LB` < `x` < `UB`.
        If scalars, the bound is replicated in each dimension. Use
        ``None`` for `LB` and `UB` if no bounds exist. Set `LB` [`i`] = -``inf``
        and `UB` [`i`] = ``inf`` if the `i`-th coordinate is unbounded (while 
        other coordinates may be bounded). Note that if `LB` and `UB` contain
        unbounded variables, the respective values of `PLB` and `PUB` need to 
        be specified (see below), by default ``None``.
    plausible_lower_bounds, plausible_upper_bounds : np.ndarray, optional
        Specifies a set of `plausible_lower_bounds` (`PLB`) and
        `plausible_upper_bounds` (`PUB`) such that `LB` < `PLB` < `PUB` < `UB`.
        Both `PLB` and `PUB` need to be finite. `PLB` and `PUB` represent a
        "plausible" range, which should denote a region of high posterior
        probability mass. Among other things, the plausible box is used to
        draw initial samples and to set priors over hyperparameters of the
        algorithm. When in doubt, we found that setting `PLB` and `PUB` using
        the topmost ~68% percentile range of the prior (e.g, mean +/- 1 SD
        for a Gaussian prior) works well in many cases (but note that
        additional information might afford a better guess). Both are
        by default ``None``.
    user_options : dict, optional
        Additional options can be passed as a dict. Please refer to the
        VBMC options page for the default options. If no `user_options` are 
        passed, the default options are used.

    Raises
    ------
    ValueError
        When neither `x0` or (`plausible_lower_bounds` and
        `plausible_upper_bounds`) are specified.
    ValueError
        When various checks for the bounds (LB, UB, PLB, PUB) of VBMC fail.

    Notes
    -----
    The current version of ``VBMC`` only supports noiseless evaluations of the 
    log posterior [1]_. Noisy evaluations as in [2]_ are not implemented yet.

    References
    ----------
    .. [1] Acerbi, L. (2018). "Variational Bayesian Monte Carlo". In Advances 
       in Neural Information Processing Systems 31 (NeurIPS 2018), pp. 8213-8223.
    .. [2] Acerbi, L. (2020). "Variational Bayesian Monte Carlo with Noisy
       Likelihoods". In Advances in Neural Information Processing Systems 33 
       (NeurIPS 2020).

    Examples
    --------
    For `VBMC` usage examples, please look up the Jupyter notebook tutorials
    in the pyvbmc documentation: 
    https://lacerbi.github.io/pyvbmc/_examples/pyvbmc_example_1.html
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
                x0 = np.full((plausible_lower_bounds.shape), np.NaN)

        self.D = x0.shape[1]
        # load basic and advanced options and validate the names
        pyvbmc_path = os.path.dirname(os.path.realpath(__file__))
        basic_path = pyvbmc_path + "/option_configs/basic_vbmc_options.ini"
        self.options = Options(
            basic_path,
            evaluation_parameters={"D": self.D},
            user_options=user_options,
        )

        advanced_path = (
            pyvbmc_path + "/option_configs/advanced_vbmc_options.ini"
        )
        self.options.load_options_file(
            advanced_path,
            evaluation_parameters={"D": self.D},
        )

        self.options.validate_option_names([basic_path, advanced_path])

        # set up root logger (only changes stuff if not initialized yet)
        logging.basicConfig(stream=sys.stdout, format="%(message)s")

        # Create an initial logger for initialization messages:
        self.logger = self._init_logger("_init")


        # variable to keep track of logging actions
        self.logging_action = []

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

        self.iteration_history = IterationHistory(
            [
                "rindex",
                "elcbo_impro",
                "stable",
                "elbo",
                "vp",
                "warmup",
                "iter",
                "elbo_sd",
                "lcbmax",
                "data_trim_list",
                "gp",
                "gp_hyp_full",
                "Ns_gp",
                "timer",
                "optim_state",
                "sKL",
                "sKL_true",
                "pruned",
                "varss",
                "func_count",
                "n_eff",
                "logging_action",
            ]
        )

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
                self.logger.warning(
                    "PLB and/or PUB not specified. Estimating"
                    + "plausible bounds from starting set X0..."
                )
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
                    self.logger.warning(
                        "vbmc:pbInitFailed: Some plausible bounds could not be "
                        + "determined from starting set. Using hard upper/lower"
                        + " bounds for those instead."
                    )
            else:
                self.logger.warning(
                    "vbmc:pbUnspecified: Plausible lower/upper bounds PLB and"
                    "/or PUB not specified and X0 is not a valid starting set. "
                    + "Using hard upper/lower bounds instead."
                )
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
            self.logger.warning(
                "vbmc:InitialPointsTooClosePB: The starting points X0 are on "
                + "or numerically too close to the hard bounds LB and UB. "
                + "Moving the initial points more inside..."
            )
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
            self.logger.warning(
                "vbmc:TooCloseBounds: For each variable, hard "
                + "and plausible bounds should not be too close. "
                + "Moving plausible bounds."
            )
            plausible_lower_bounds = np.maximum(plausible_lower_bounds, LB_eff)
            plausible_upper_bounds = np.minimum(plausible_upper_bounds, UB_eff)

        # Check that all X0 are inside the plausible bounds,
        # move bounds otherwise
        if np.any(x0 <= plausible_lower_bounds)\
           or np.any(x0 >= plausible_upper_bounds):
            self.logger.warning(
                "vbmc:InitialPointsOutsidePB. The starting points X0"
                + " are not inside the provided plausible bounds PLB and "
                + "PUB. Expanding the plausible bounds..."
            )
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
        # Iterations are from 0 onwards in optimize so we should have -1
        # here. In MATLAB this was 0.
        optim_state["iter"] = -1

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
        optim_state["R"] = np.inf

        # Start with adaptive sampling
        optim_state["skip_active_sampling"] = False

        # Running mean and covariance of variational posterior
        # in transformed space
        optim_state["run_mean"] = []
        optim_state["run_cov"] = []
        # Last time running average was updated
        optim_state["last_run_avg"] = np.NaN

        # Current number of components for variational posterior
        optim_state["vpK"] = self.K

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
        elif len(self.options.get("uncertaintyhandling")) > 0:
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

        # Repository of variational solutions (not used in Python)
        # optim_state["vp_repo"] = []

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
            optim_state["gp_noisefun"] = [1, 0, 0]
        elif optim_state.get("uncertainty_handling_level") == 1:
            # Infer noise
            optim_state["gp_noisefun"] = [1, 2, 0]
        elif optim_state.get("uncertainty_handling_level") == 2:
            # Provided heteroskedastic noise
            optim_state["gp_noisefun"] = [1, 1, 0]

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

        # Starting threshold on y for output warping
        if self.options.get("fitnessshaping"):
            optim_state["outwarp_delta"] = self.options.get(
                "outwarpthreshbase"
            )
        else:
            optim_state["outwarp_delta"] = []

        return optim_state

    def optimize(self):
        """
        Run inference on an initialized ``VBMC`` object. 
        
        VBMC computes a variational approximation of the full posterior and the
        ELBO (evidence lower bound), a lower bound on the log normalization 
        constant (log marginal likelhood or log model evidence) for the provided 
        unnormalized log posterior.

        Returns
        -------
        vp : VariationalPosterior
            The ``VariationalPosterior`` computed by VBMC.
        elbo : float
            An estimate of the ELBO for the returned `vp`.
        elbo_sd : float
            The standard deviation of the estimate of the ELBO. Note that this 
            standard deviation is *not* representative of the error between the 
            `elbo` and the true log marginal likelihood.
        success_flag : bool
           `success_flag` is ``True`` if the inference reached stability within 
           the provided budget of function evaluations, suggesting convergence.
           If ``False``, the returned solution has not stabilized and should
           not be trusted.
        results_dict : dict
            A dictionary with additional information about the VBMC run.
        """
        is_finished = False
        # the iterations of pyvbmc start at 0
        iteration = -1
        timer = Timer()
        gp = None
        hyp_dict = {}
        success_flag = True
        # Initialize main logger with potentially new options:
        self.logger = self._init_logger()
        # set up strings for logging of the iteration
        display_format = self._setup_logging_display_format()

        if self.optim_state["uncertainty_handling_level"] > 0:
            self.logger.info(
                "Beginning variational optimization assuming NOISY observations"
                + " of the log-joint"
            )
        else:
            self.logger.info(
                "Beginning variational optimization assuming EXACT observations"
                + " of the log-joint."
            )

        self._log_column_headers()

        while not is_finished:
            iteration += 1
            self.optim_state["iter"] = iteration
            self.optim_state["redo_roto_scaling"] = False
            vp_old = copy.deepcopy(self.vp)

            self.logging_action = []

            if iteration == 0 and self.optim_state["warmup"]:
                self.logging_action.append("start warm-up")

            # Switch to stochastic entropy towards the end if still
            # deterministic.
            if self.optim_state.get("entropy_switch") and (
                self.function_logger.func_count
                >= self.optim_state.get("entropy_force_switch")
                * self.optim_state.get("max_fun_evals")
            ):
                self.optim_state["entropy_switch"] = False
                self.logging_action.append("entropy switch")


            ## Input warping / reparameterization
            if self.options["incrementalwarpdelay"]:
                WarpDelay = self.options["warpeveryiters"]*np.max([1, self.optim_state["warping_count"]])
            else:
                WarpDelay = self.options["warpeveryiters"]

            doWarping = (self.options.get("warprotoscaling")\
                or self.options.get("warpnonlinear"))\
                and (iteration > 0)\
                and (not self.optim_state["warmup"])\
                and (iteration - self.optim_state["last_warping"] > WarpDelay)\
                and (self.vp.K >= self.options["warpmink"])\
                and (self.iteration_history["rindex"][iteration-1]\
                     < self.options["warptolreliability"])\
                and (self.vp.D > 1)

            if doWarping:
                timer.start_timer("warping")
                vp_tmp, __, __, __ = self.determine_best_vp()
                vp_tmp = copy.deepcopy(vp_tmp)
                # Store variables in case warp needs to be undone:
                # (vp_old copied above)
                optim_state_old = copy.deepcopy(self.optim_state)
                gp_old = copy.deepcopy(gp)
                function_logger_old = copy.deepcopy(self.function_logger)
                elbo_old = elbo
                elbo_sd_old = elbo_sd
                hyp_dict_old = copy.deepcopy(hyp_dict)
                # Compute and apply whitening transform:
                parameter_transformer_warp, self.optim_state, self.function_logger, warp_action = warp_input_vbmc(vp_tmp, self.optim_state, self.function_logger, self.options)

                self.vp, hyp_dict["hyp"] = warp_gpandvp_vbmc(parameter_transformer_warp, self.vp, self)
                # Update the VBMC ParameterTransformer
                self.parameter_transformer = parameter_transformer_warp

                self.logging_action.append(warp_action)
                timer.stop_timer("warping")

                if self.options.get("warpundocheck"):
                    ## Train gp

                    timer.start_timer("gpTrain")

                    gp, Ns_gp, sn2hpd, hyp_dict = train_gp(
                        hyp_dict,
                        self.optim_state,
                        self.function_logger,
                        self.iteration_history,
                        self.options,
                        self.plausible_lower_bounds,
                        self.plausible_upper_bounds,
                    )
                    self.optim_state["sn2hpd"] = sn2hpd

                    timer.stop_timer("gpTrain")

                    ## Optimize variational parameters
                    timer.start_timer("variationalFit")

                    if not self.vp.optimize_mu:
                        # Variational components fixed to training inputs
                        self.vp.mu = gp.X.T
                        Knew = self.vp.mu.shape[1]
                    else:
                        # Update number of variational mixture components
                        Knew = self.vp.K

                    # Decide number of fast/slow optimizations
                    N_fastopts = math.ceil(self.options.eval("nselbo", {"K": self.K}))
                    N_slowops = self.options.get("elbostarts") # Full optimizations.

                    # Run optimization of variational parameters
                    self.vp, varss, pruned = optimize_vp(
                        self.options,
                        self.optim_state,
                        self.vp,
                        gp,
                        N_fastopts,
                        N_slowopts,
                        Knew,
                    )

                    self.optim_state["vpK"] = self.vp.K
                    # Save current entropy
                    self.optim_state["H"] = self.vp.stats["entropy"]

                    # Get real variational posterior (might differ from training posterior)
                    # vp_real = vp.vptrain2real(0, self.options)
                    vp_real = self.vp
                    elbo = vp_real.stats["elbo"]
                    elbo_sd = vp_real.stats["elbo_sd"]

                    timer.stop_timer("variationalFit")

                    # Keep warping only if it substantially improves ELBO
                    # and uncertainty does not blow up too much
                    if (elbo < (elbo_old + self.options["warptolimprovement"]))\
                    or (elbo_sd > (elbo_sd_old * self.options["warptolsdmultiplier"] + self.options["warptolsdbase"])):
                        # Undo input warping:
                        self.vp = vp_old
                        self.gp = gp_old
                        self.optim_state = optim_state_old
                        self.function_logger = function_logger_old
                        hyp_dict = hyp_dict_old

                        # Still keep track of failed warping (failed warp counts twice)
                        self.optim_state["warping_count"] += 2
                        self.optim_state["last_warping"] = self.optim_state["iter"]
                        self.logging_action.append(", undo")


            ## Actively sample new points into the training set
            timer.start_timer("activeSampling")
            self.parameter_transformer = self.vp.parameter_transformer

            if iteration == 0:
                new_funevals = self.options.get("funevalstart")
            else:
                new_funevals = self.options.get("funevalsperiter")

            # Careful with Xn, in MATLAB this condition is > 0
            # due to 1-based indexing.
            if self.function_logger.Xn >= 0:
                self.function_logger.ymax = np.max(
                    self.function_logger.y[self.function_logger.X_flag]
                )

            if self.optim_state.get("skipactivesampling"):
                self.optim_state["skipactivesampling"] = False
            else:
                if (
                    gp is not None
                    and self.options.get("separatesearchgp")
                    and not self.options.get("varactivesample")
                ):
                    # Train a distinct GP for active sampling
                    # Since we are doing iterations from 0 onwards
                    # instead of from 1 onwards, this should be checking
                    # oddness, not evenness.
                    if iteration % 2 == 1:
                        meantemp = self.optim_state.get("gp_meanfun")
                        self.optim_state["gp_meanfun"] = "const"
                        gp_search, Ns_gp, sn2hpd, hyp_dict = train_gp(
                            hyp_dict,
                            self.optim_state,
                            self.function_logger,
                            self.iteration_history,
                            self.options,
                            self.plausible_lower_bounds,
                            self.plausible_upper_bounds,
                        )
                        self.optim_state["sn2hpd"] = sn2hpd
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
                    self.optim_state["hyp_dict"] = hyp_dict
                    (
                        self.function_logger,
                        self.optim_state,
                        self.vp,
                    ) = active_sample(
                        gp_search,
                        new_funevals,
                        self.optim_state,
                        self.function_logger,
                        self.iteration_history,
                        self.vp,
                        self.options,
                    )
                    hyp_dict = self.optim_state["hyp_dict"]

            # Number of training inputs
            self.optim_state["N"] = self.function_logger.Xn
            self.optim_state["n_eff"] = np.sum(
                self.function_logger.nevals[self.function_logger.X_flag]
            )
            assert not np.isnan(self.optim_state["N"])
            assert not np.isnan(self.optim_state["n_eff"])

            timer.stop_timer("activeSampling")

            ## Train gp

            timer.start_timer("gpTrain")

            gp, Ns_gp, sn2hpd, hyp_dict = train_gp(
                hyp_dict,
                self.optim_state,
                self.function_logger,
                self.iteration_history,
                self.options,
                self.plausible_lower_bounds,
                self.plausible_upper_bounds,
            )
            self.optim_state["sn2hpd"] = sn2hpd

            timer.stop_timer("gpTrain")

            # Check if reached stable sampling regime
            if (
                Ns_gp == self.options.get("stablegpsamples")
                and self.optim_state.get("stop_sampling") == 0
            ):
                self.optim_state["stop_sampling"] = self.optim_state.get("N")

            ## Optimize variational parameters
            timer.start_timer("variationalFit")

            if not self.vp.optimize_mu:
                # Variational components fixed to training inputs
                self.vp.mu = gp.X.T
                Knew = self.vp.mu.shape[1]
            else:
                # Update number of variational mixture components
                Knew = update_K(
                    self.optim_state, self.iteration_history, self.options
                )

            # Decide number of fast/slow optimizations
            N_fastopts = math.ceil(self.options.eval("nselbo", {"K": self.K}))

            if self.optim_state.get("recompute_var_post") or (
                self.options.get("alwaysrefitvarpost")
            ):
                # Full optimizations
                N_slowopts = self.options.get("elbostarts")
                self.optim_state["recompute_var_post"] = False
            else:
                # Only incremental change from previous iteration
                N_fastopts = math.ceil(
                    N_fastopts * self.options.get("nselboincr")
                )
                N_slowopts = 1
            # Run optimization of variational parameters
            self.vp, varss, pruned = optimize_vp(
                self.options,
                self.optim_state,
                self.vp,
                gp,
                N_fastopts,
                N_slowopts,
                Knew,
            )

            self.optim_state["vpK"] = self.vp.K
            # Save current entropy
            self.optim_state["H"] = self.vp.stats["entropy"]

            # Get real variational posterior (might differ from training posterior)
            # vp_real = vp.vptrain2real(0, self.options)
            vp_real = self.vp
            elbo = vp_real.stats["elbo"]
            elbo_sd = vp_real.stats["elbo_sd"]

            timer.stop_timer("variationalFit")

            # Finalize iteration

            timer.start_timer("finalize")

            # Compute symmetrized KL-divergence between old and new posteriors
            Nkl = 1e5

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
            fmu, fs2 = gp.predict(gp.X, gp.y, gp.s2, add_noise=False)
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
                wRun = self.options.get("momentsrunweight") ** Nnew
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

            # store current gp in vp
            self.vp.gp = gp

            iteration_values = {
                "iter": iteration,
                "optim_state": self.optim_state,
                "vp": self.vp,
                "elbo": elbo,
                "elbo_sd": elbo_sd,
                "varss": varss,
                "sKL": sKL,
                "sKL_true": sKL_true,
                "gp": gp,
                "gp_hyp_full": gp.get_hyperparameters(as_array=True),
                "Ns_gp": Ns_gp,
                "pruned": pruned,
                "timer": timer,
                "func_count": self.function_logger.func_count,
                "lcbmax": self.optim_state["lcbmax"],
                "n_eff": self.optim_state["n_eff"],
            }

            # Record all useful stats
            self.iteration_history.record_iteration(
                iteration_values,
                iteration,
            )

            # Check warmup
            if (
                self.optim_state.get("iter") > 1
                and self.optim_state.get("stop_gp_sampling") == 0
                and not self.optim_state.get("warmup")
            ):
                if self._is_gp_sampling_finished():
                    self.optim_state[
                        "stop_gp_sampling"
                    ] = self.optim_state.get("N")

            # Check termination conditions
            (
                is_finished,
                termination_message,
                success_flag,
            ) = self._check_termination_conditions()

            # Save stability
            self.vp.stats["stable"] = self.iteration_history["stable"][
                iteration
            ]

            # Check if we are still warming-up
            if self.optim_state.get("warmup") and iteration > 0:
                if self.options.get("recomputelcbmax"):
                    self.optim_state["lcbmax_vec"] = self._recompute_lcbmax().T
                trim_flag = self._check_warmup_end_conditions()
                if trim_flag:
                    self._setup_vbmc_after_warmup()
                    # Re-update GP after trimming
                    gp = reupdate_gp(self.function_logger, gp)
                if not self.optim_state.get("warmup"):
                    self.vp.optimize_mu = self.options.get("variablemeans")
                    self.vp.optimize_weights = self.options.get(
                        "variableweights"
                    )

                    # Switch to main algorithm options
                    # options = options_main
                    # Reset GP hyperparameter covariance
                    # hypstruct.runcov = []
                    hyp_dict["runcov"] = None
                    # Reset VP repository (not used in python)
                    self.optim_state["vp_repo"] = []

                    # Re-get acq info
                    # self.optim_state['acqInfo'] = getAcqInfo(
                    #    options.SearchAcqFcn
                    # )
            # Needs to be below the above block since warmup value can change
            # in _check_warmup_end_conditions
            self.iteration_history.record(
                "warmup", self.optim_state.get("warmup"), iteration
            )

            # Check and update fitness shaping / output warping threshold
            if (
                self.optim_state.get("outwarp_delta") != []
                and self.optim_state.get("R") is not None
                and (
                    self.optim_state.get("R")
                    < self.options.get("warptolreliability")
                )
            ):
                Xrnd, _ = self.vp.sample(N=int(2e4), origflag=False)
                ymu, _ = gp.predict(Xrnd, add_noise=True)
                ydelta = max(
                    [0, self.function_logger.ymax - np.quantile(ymu, 1e-3)]
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
            # Stopped GP sampling this iteration?
            if (
                Ns_gp == self.options["stablegpsamples"]
                and self.iteration_history["Ns_gp"][max(0, iteration - 1)]
                > self.options["stablegpsamples"]
            ):
                if Ns_gp == 0:
                    self.logging_action.append("switch to GP opt")
                else:
                    self.logging_action.append("stable GP sampling")

            if self.options.get("printiterationheader") is None:
                # Default behavior, try to guess based on plotting options:
                reprint_headers = self.options.get("plot")\
                    and iteration > 0\
                    and "inline" in plt.get_backend()
            elif self.options["printiterationheader"]:
                # Re-print every iteration after 0th
                reprint_headers = iteration > 0
            else:
                # Never re-print headers
                reprint_headers = False
            # Reprint the headers if desired:
            if reprint_headers:
                self._log_column_headers()

            if self.optim_state["cache_active"]:
                self.logger.info(
                    display_format.format(
                        iteration,
                        self.function_logger.func_count,
                        self.function_logger.cache_count,
                        elbo,
                        elbo_sd,
                        sKL,
                        self.vp.K,
                        self.optim_state["R"],
                        "".join(self.logging_action),
                    )
                )

            else:
                if (
                    self.optim_state["uncertainty_handling_level"] > 0
                    and self.options.get("maxrepeatedobservations") > 0
                ):
                    self.logger.info(
                        display_format.format(
                            iteration,
                            self.function_logger.func_count,
                            self.optim_state["N"],
                            elbo,
                            elbo_sd,
                            sKL,
                            self.vp.K,
                            self.optim_state["R"],
                            "".join(self.logging_action),
                        )
                    )
                else:
                    self.logger.info(
                        display_format.format(
                            iteration,
                            self.function_logger.func_count,
                            elbo,
                            elbo_sd,
                            sKL,
                            self.vp.K,
                            self.optim_state["R"],
                            "".join(self.logging_action),
                        )
                    )
            self.iteration_history.record(
                "logging_action", self.logging_action, iteration
            )

            # Plot iteration
            if self.options.get("plot"):
                if iteration > 0:
                    previous_gp = self.iteration_history["vp"][
                        iteration - 1
                    ].gp
                    # find points that are new in this iteration
                    # (hacky cause numpy only has 1D set diff)
                    # future fix: active sampling should return the set of
                    # indices of the added points
                    highlight_data = np.array(
                        [
                            i
                            for i, x in enumerate(self.vp.gp.X)
                            if tuple(x) not in set(map(tuple, previous_gp.X))
                        ]
                    )
                else:
                    highlight_data = None

                if len(self.logging_action) > 0:
                    title = "VBMC iteration {} ({})".format(
                        iteration, "".join(self.logging_action)
                    )
                else:
                    title = "VBMC iteration {}".format(iteration)

                self.vp.plot(
                    plot_data=True,
                    highlight_data=highlight_data,
                    plot_vp_centres=True,
                    title=title,
                )
                plt.show()

        # Pick "best" variational solution to return
        self.vp, elbo, elbo_sd, idx_best = self.determine_best_vp()

        # Last variational optimization with large number of components
        self.vp, elbo, elbo_sd, changed_flag = self.finalboost(
            self.vp, self.iteration_history["gp"][idx_best]
        )
        if changed_flag:
            # Recompute symmetrized KL-divergence
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

            if self.options.get("plot"):
                self._log_column_headers()

            if (
                self.optim_state["uncertainty_handling_level"] > 0
                and self.options.get("maxrepeatedobservations") > 0
            ):
                self.logger.info(
                    display_format.format(
                        np.Inf,
                        self.function_logger.func_count,
                        self.optim_state["N"],
                        elbo,
                        elbo_sd,
                        sKL,
                        self.vp.K,
                        self.iteration_history.get("rindex")[idx_best],
                        "finalize",
                    )
                )
            else:
                self.logger.info(
                    display_format.format(
                        np.Inf,
                        self.function_logger.func_count,
                        elbo,
                        elbo_sd,
                        sKL,
                        self.vp.K,
                        self.iteration_history.get("rindex")[idx_best],
                        "finalize",
                    )
                )

        # plot final vp:
        if self.options.get("plot"):
            self.vp.plot(
                plot_data=True,
                highlight_data=None,
                plot_vp_centres=True,
                title="VBMC final ({} iterations)".format(iteration),
            )
            plt.show()

        # Set exit_flag based on stability (check other things in the future)
        if not success_flag:
            if self.vp.stats["stable"]:
                success_flag = True
        else:
            if not self.vp.stats["stable"]:
                success_flag = False

        # Print final message
        self.logger.warning(termination_message)
        self.logger.warning(
            "Estimated ELBO: {:.3f} +/-{:.3f}.".format(elbo, elbo_sd)
        )
        if not success_flag:
            self.logger.warning(
                "Caution: Returned variational solution may have"
                + " not converged."
            )

        result_dict = self._create_result_dict(idx_best, termination_message)

        return (
            copy.deepcopy(self.vp),
            self.vp.stats["elbo"],
            self.vp.stats["elbo_sd"],
            success_flag,
            result_dict,
        )

    # Loop termination:

    def _check_warmup_end_conditions(self):
        """
        Private method to check the warmup end conditions.
        """
        iteration = self.optim_state.get("iter")
        exit_flag = 0

        # First requirement for stopping, no constant improvement of metric
        stable_count_flag = False
        stop_warmup_thresh = self.options.get(
            "stopwarmupthresh"
        ) * self.options.get("funevalsperiter")
        tol_stable_warmup_iters = math.ceil(
            self.options.get("tolstablewarmup")
            / self.options.get("funevalsperiter")
        )

        # MATLAB has +1 on the right side due to different indexing.
        if iteration > tol_stable_warmup_iters:
            # Vector of ELCBO (ignore first two iterations, ELCBO is unreliable)
            elcbo_vec = self.iteration_history.get("elbo") - self.options.get(
                "elcboimproweight"
            ) * self.iteration_history.get("elbo_sd")
            # Here and below the max is one higher in MATLAB.
            max_now = np.amax(
                elcbo_vec[max(3, -tol_stable_warmup_iters + 1) :]
            )
            max_before = np.amax(
                elcbo_vec[3 : max(2, -tol_stable_warmup_iters)], initial=0
            )
            stable_count_flag = (max_now - max_before) < stop_warmup_thresh

        # Vector of maximum lower confidence bounds (LCB) of fcn values
        lcbmax_vec = self.iteration_history.get("lcbmax")[: iteration + 1]

        # Second requirement, also no substantial improvement of max fcn value
        # in recent iters (unless already performing BO-like warmup)
        if self.options.get("warmupcheckmax"):
            idx_last = np.full(lcbmax_vec.shape, False)
            recent_past = iteration - int(
                math.ceil(
                    self.options.get("tolstablewarmup")
                    / self.options.get("funevalsperiter")
                )
                + 1
            )
            idx_last[max(1, recent_past) :] = True
            impro_fcn = max(
                0,
                np.amax(lcbmax_vec[idx_last]) - np.amax(lcbmax_vec[~idx_last]),
            )
        else:
            impro_fcn = 0

        no_recent_improvement_flag = impro_fcn < stop_warmup_thresh

        # Alternative criterion for stopping - no improvement over max fcn value
        max_thresh = np.amax(lcbmax_vec) - self.options.get("tolimprovement")
        idx_1st = np.ravel(np.argwhere(lcbmax_vec > max_thresh))[0]
        yy = self.iteration_history.get("func_count")[: iteration + 1]
        pos = yy[idx_1st]
        currentpos = self.function_logger.func_count
        no_longterm_improvement_flag = (currentpos - pos) > self.options.get(
            "warmupnoimprothreshold"
        )

        if len(self.optim_state.get("data_trim_list")) > 0:
            last_data_trim = self.optim_state.get("data_trim_list")[-1]
        else:
            last_data_trim = -1 * np.Inf

        no_recent_trim_flag = (
            self.optim_state.get("N") - last_data_trim
        ) >= 10

        stop_warmup = (
            stable_count_flag
            and no_recent_improvement_flag
            or no_longterm_improvement_flag
        ) and no_recent_trim_flag

        return stop_warmup

    def _setup_vbmc_after_warmup(self):
        """
        Private method to setup multiple vbmc settings after a the warmup has
        been determined to be ended. The method whether the warmup ending was
        a false alarm and then only prunes.
        """
        iteration = self.optim_state.get("iter")
        if (
            self.iteration_history.get("rindex")[iteration]
            < self.options.get("stopwarmupreliability")
            or len(self.optim_state.get("data_trim_list")) >= 1
        ):
            self.optim_state["warmup"] = False
            self.logging_action.append("end warm-up")
            threshold = self.options.get("warmupkeepthreshold") * (
                len(self.optim_state.get("data_trim_list")) + 1
            )
            self.optim_state["last_warmup"] = iteration

        else:
            # This may be a false alarm; prune and continue
            if self.options.get("warmupkeepthresholdfalsealarm") is None:
                warmup_keep_threshold_false_alarm = self.options.get(
                    "warmupkeepthreshold"
                )
            else:
                warmup_keep_threshold_false_alarm = self.options.get(
                    "warmupkeepthresholdfalsealarm"
                )

            threshold = warmup_keep_threshold_false_alarm * (
                len(self.optim_state.get("data_trim_list")) + 1
            )

            self.optim_state["data_trim_list"] = np.append(
                self.optim_state.get("data_trim_list"),
                [self.optim_state.get("N")],
            )

            self.logging_action.append("trim data")

        # Remove warm-up points from training set unless close to max
        ymax = max(self.function_logger.y_orig[: self.function_logger.Xn + 1])
        n_keep_min = self.D + 1
        idx_keep = (ymax - self.function_logger.y_orig) < threshold
        if np.sum(idx_keep) < n_keep_min:
            y_temp = np.copy(self.function_logger.y_orig)
            y_temp[~np.isfinite(y_temp)] = -np.Inf
            order = np.argsort(y_temp * -1, axis=0)
            idx_keep[
                order[: min(n_keep_min, self.function_logger.Xn) + 1]
            ] = True
        # Note that using idx_keep[:, 0] is necessary since X_flag
        # is a 1D array and idx_keep a 2D array.
        self.function_logger.X_flag = np.logical_and(
            idx_keep[:, 0], self.function_logger.X_flag
        )

        # Skip adaptive sampling for next iteration
        self.optim_state["skipactivesampling"] = self.options.get(
            "skipactivesamplingafterwarmup"
        )

        # Fully recompute variational posterior
        self.optim_state["recompute_var_post"] = True

    def _check_termination_conditions(self):
        """
        Private method to determine the status of termination conditions.

        It also saves the reliability index, ELCBO improvement and stableflag
        to the iteration_history object.
        """
        is_finished_flag = False
        termination_message = ""
        success_flag = True
        output_dict = dict()

        # Maximum number of new function evaluations
        if self.function_logger.func_count >= self.options.get("maxfunevals"):
            is_finished_flag = True
            termination_message = (
                "Inference terminated: reached maximum number "
                + "of function evaluations options.maxfunevals."
            )

        # Maximum number of iterations
        iteration = self.optim_state.get("iter")
        if iteration + 1 >= self.options.get("maxiter"):
            is_finished_flag = True
            termination_message = (
                "Inference terminated: reached maximum number "
                + "of iterations options.maxiter."
            )

        # Quicker stability check for entropy switching
        if self.optim_state.get("entropy_switch"):
            tol_stable_iters = self.options.get("tolstableentropyiters")
        else:
            tol_stable_iters = int(
                math.ceil(
                    self.options.get("tolstablecount")
                    / self.options.get("funevalsperiter")
                )
            )

        rindex, ELCBO_improvement = self._compute_reliability_index(
            tol_stable_iters
        )

        # Store reliability index
        self.iteration_history.record("rindex", rindex, iteration)
        self.iteration_history.record(
            "elcbo_impro", ELCBO_improvement, iteration
        )
        self.optim_state["R"] = rindex

        # Check stability termination condition
        stableflag = False
        if (
            iteration + 1 >= tol_stable_iters
            and rindex < 1
            and ELCBO_improvement < self.options.get("tolimprovement")
        ):
            # Count how many good iters in the recent past (excluding current)
            stable_count = np.sum(
                self.iteration_history.get("rindex")[
                    iteration - tol_stable_iters + 1 : iteration
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
                    self.logging_action.append("entropy switch")
                else:
                    is_finished_flag = True
                    stableflag = True
                    success_flag = False
                    self.logging_action.append("stable")
                    termination_message = (
                        "Inference terminated: variational "
                        + "solution stable for options.tolstablecount"
                        + "fcn evaluations."
                    )

        # Store stability flag
        self.iteration_history.record("stable", stableflag, iteration)

        # Prevent early termination
        if self.function_logger.func_count < self.options.get(
            "minfunevals"
        ) or iteration < self.options.get("miniter"):
            is_finished_flag = False

        return (
            is_finished_flag,
            termination_message,
            success_flag,
        )

    def _compute_reliability_index(self, tol_stable_iters):
        """
        Private function to compute the reliability index.
        """
        iteration_idx = self.optim_state.get("iter")
        # Was < 3 in MATLAB due to different indexing.
        if self.optim_state.get("iter") < 2:
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
                self.iteration_history.get("elbo")[iteration_idx]
                - self.iteration_history.get("elbo")[iteration_idx - 1]
            )
            / tol_sd
        )
        rindex_vec[1] = (
            self.iteration_history.get("elbo_sd")[iteration_idx] / tol_sd
        )
        rindex_vec[2] = self.iteration_history.get("sKL")[
            iteration_idx
        ] / self.options.get("tolskl")

        # Compute average ELCBO improvement per fcn eval in the past few iters
        # TODO: off by one error
        idx0 = int(
            max(
                0,
                self.optim_state.get("iter")
                - math.ceil(0.5 * tol_stable_iters),
            )
        )
        # Remember than upper end of range is exclusive in Python, so +1 is
        # needed.
        xx = self.iteration_history.get("func_count")[idx0 : iteration_idx + 1]
        yy = (
            self.iteration_history.get("elbo")[idx0 : iteration_idx + 1]
            - self.options.get("elcboimproweight")
            * self.iteration_history.get("elbo_sd")[idx0 : iteration_idx + 1]
        )
        # need to casts here to get things to run
        ELCBO_improvement = np.polyfit(
            list(map(float, xx)), list(map(float, yy)), 1
        )[0]
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
        w2 = np.exp(
            -(
                self.iteration_history.get("N")[-1]
                - self.iteration_history.get("N") / 10
            )
        )
        w2 = w2 / np.sum(w2)
        w = 0.5 * w1 + 0.5 * w2
        if np.sum(
            w * self.iteration_history.get("gp_sample_var")
        ) < self.options.get("tolgpvarmcmc"):
            finished_flag = True

        return finished_flag

    def _recompute_lcbmax(self):
        """
        RECOMPUTE_LCBMAX Recompute moving LCB maximum based on current GP.
        """
        # ToDo: Recompute_lcbmax needs to be implemented.
        return np.array([])

    # Finalizing:

    def finalboost(self, vp: VariationalPosterior, gp: gpr.GP):
        """
        Perform a final boost of variational components.

        Parameters
        ----------
        vp : VariationalPosterior
            The VariationalPosterior that should be boosted.
        gp : GaussianProcess
            The corresponding GaussianProcess of the VariationalPosterior.

        Returns
        -------
        vp : VariationalPosterior
            The VariationalPosterior resulting from the final boost.
        elbo : VariationalPosterior
            The ELBO of the VariationalPosterior resulting from the final boost.
        elbo_sd : VariationalPosterior
            The ELBO_SD of the VariationalPosterior resulting from the
            final boost.
        changed_flag : bool
           Indicates if the final boost has taken place or not.
        """

        changed_flag = False

        K_new = max(self.vp.K, self.options.get("minfinalcomponents"))

        # Current entropy samples during variational optimization
        n_sent = self.options.eval("nsent", {"K": K_new})
        n_sent_fast = self.options.eval("nsentfast", {"K": K_new})
        n_sent_fine = self.options.eval("nsentfine", {"K": K_new})

        # Entropy samples for final boost
        if self.options.get("nsentboost") == []:
            n_sent_boost = n_sent
        else:
            n_sent_boost = self.options.eval("nsentboost", {"K": K_new})

        if self.options.get("nsentfastboost") == []:
            n_sent_fast_boost = n_sent_fast
        else:
            n_sent_fast_boost = self.options.eval(
                "nsentfastboost", {"K": K_new}
            )

        if self.options.get("nsentfineboost") == []:
            n_sent_fine_boost = n_sent_fine
        else:
            n_sent_fine_boost = self.options.eval(
                "nsentfineboost", {"K": K_new}
            )

        # Perform final boost?

        do_boost = (
            self.vp.K < self.options.get("minfinalcomponents")
            or n_sent != n_sent_boost
            or n_sent_fine != n_sent_fine_boost
        )

        if do_boost:
            # Last variational optimization with large number of components
            n_fast_opts = math.ceil(self.options.eval("nselbo", {"K": K_new}))

            n_fast_opts = int(
                math.ceil(n_fast_opts * self.options.get("nselboincr"))
            )
            n_slow_opts = 1

            # No pruning of components
            self.options.__setitem__("tolweight", 0, force=True)

            # End warmup
            self.optim_state["warmup"] = False
            self.vp.optimize_mu = self.options.get("variablemeans")
            self.vp.optimize_weights = self.options.get("variableweights")

            self.options.__setitem__("nsent", n_sent_boost, force=True)
            self.options.__setitem__("nsentfast", n_sent_fast_boost, force=True)
            self.options.__setitem__("nsentfine", n_sent_fine_boost, force=True)
            self.options.__setitem__("maxiterstochastic", np.Inf, force=True)
            self.optim_state["entropy_alpha"] = 0

            stable_flag = np.copy(vp.stats["stable"])
            vp, varss, pruned = optimize_vp(
                self.options,
                self.optim_state,
                vp,
                gp,
                n_fast_opts,
                n_slow_opts,
                K_new,
            )
            vp.stats["stable"] = stable_flag
            changed_flag = True
        else:
            vp = self.vp

        elbo = vp.stats["elbo"]
        elbo_sd = vp.stats["elbo_sd"]
        return vp, elbo, elbo_sd, changed_flag

    def determine_best_vp(
        self,
        max_idx: int = None,
        safe_sd: float = 5,
        frac_back: float = 0.25,
        rank_criterion_flag: bool = False,
    ):
        """
        Return the best VariationalPosterior found during the optimization of
        VBMC as well as its ELBO, ELBO_SD and the index of the iteration.

        Parameters
        ----------
        max_idx : int, optional
            Check up to this iteration, by default None which means last iter.
        safe_sd : float, optional
            Penalization for uncertainty, by default 5.
        frac_back : float, optional
            If no past stable iteration, go back up to this fraction of
            iterations, by default 0.25.
        rank_criterion_flag : bool, optional
            If True use new ranking criterion method to pick best solution.
            It finds a solution that combines ELCBO, stability, and recency,
            by default False.

        Returns
        -------
        vp : VariationalPosterior
            The VariationalPosterior found during the optimization of VBMC.
        elbo : float
            The ELBO of the iteration with the best VariationalPosterior.
        elbo_sd : float
            The ELBO_SD of the iteration with the best VariationalPosterior.
        idx_best : int
            The index of the iteration with the best VariationalPosterior.
        """

        # Check up to this iteration (default, last)
        if max_idx is None:
            max_idx = self.iteration_history.get("iter")[-1]

        if self.iteration_history.get("stable")[max_idx]:
            # If the current iteration is stable, return it
            idx_best = max_idx

        else:
            # Otherwise, find best solution according to various criteria

            if rank_criterion_flag:
                # Find solution that combines ELCBO, stability, and recency

                rank = np.zeros((max_idx + 1, 4))
                # Rank by position
                rank[:, 0] = np.arange(1, max_idx + 2)[::-1]

                # Rank by ELCBO
                lnZ_iter = self.iteration_history.get("elbo")[: max_idx + 1]
                lnZsd_iter = self.iteration_history.get("elbo_sd")[
                    : max_idx + 1
                ]
                elcbo = lnZ_iter - safe_sd * lnZsd_iter
                order = elcbo.argsort()[::-1]
                rank[order, 1] = np.arange(1, max_idx + 2)

                # Rank by reliability index
                order = self.iteration_history.get("rindex")[
                    : max_idx + 1
                ].argsort()
                rank[order, 2] = np.arange(1, max_idx + 2)

                # Rank penalty to all non-stable iterations
                rank[:, 3] = max_idx
                rank[
                    self.iteration_history.get("stable")[: max_idx + 1], 3
                ] = 1

                idx_best = np.argmin(np.sum(rank, 1))

            else:
                # Find recent solution with best ELCBO
                laststable = np.argwhere(
                    self.iteration_history.get("stable")[: max_idx + 1] == True
                )

                if len(laststable) == 0:
                    # Go some iterations back if no previous stable iteration
                    idx_start = max(
                        0, int(math.ceil(max_idx - max_idx * frac_back))
                    )
                else:
                    idx_start = np.ravel(laststable)[-1]

                lnZ_iter = self.iteration_history.get("elbo")[
                    idx_start : max_idx + 1
                ]
                lnZsd_iter = self.iteration_history.get("elbo_sd")[
                    idx_start : max_idx + 1
                ]
                elcbo = lnZ_iter - safe_sd * lnZsd_iter
                idx_best = idx_start + np.argmax(elcbo)

        # Return best variational posterior, its ELBO and SD
        vp = self.iteration_history.get("vp")[idx_best]
        elbo = self.iteration_history.get("elbo")[idx_best]
        elbo_sd = self.iteration_history.get("elbo_sd")[idx_best]
        vp.stats["stable"] = self.iteration_history.get("stable")[idx_best]
        return vp, elbo, elbo_sd, idx_best

    def _create_result_dict(self, idx_best: int, termination_message: str):
        """
        Private method to create the result dict.
        """
        output = dict()
        output["function"] = str(self.function_logger.fun)
        if np.all(np.isinf(self.optim_state["lb"])) and np.all(
            np.isinf(self.optim_state["ub"])
        ):
            output["problemtype"] = "unconstrained"
        else:
            output["problemtype"] = "boundconstraints"

        output["iterations"] = self.optim_state["iter"]
        output["funccount"] = self.function_logger.func_count
        output["bestiter"] = idx_best
        output["trainsetsize"] = self.iteration_history["n_eff"][idx_best]
        output["components"] = self.vp.K
        output["rindex"] = self.iteration_history["rindex"][idx_best]
        if self.iteration_history["stable"][idx_best]:
            output["convergencestatus"] = "probable"
        else:
            output["convergencestatus"] = "no"

        output["overhead"] = np.NaN
        output["rngstate"] = "rng"
        output["algorithm"] = "Variational Bayesian Monte Carlo"
        output["version"] = "0.0.1"
        output["message"] = termination_message

        output["elbo"] = self.vp.stats["elbo"]
        output["elbo_sd"] = self.vp.stats["elbo_sd"]

        return output

    def _log_column_headers(self):
        """
        Private method to log column headers for the iteration log.
        """
        # We only want to log the column headers once when writing to a file,
        # but we re-write them to the stream (stdout) when plotting.
        if self.optim_state.get("iter") > 0:
            logger = self.logger.stream_only
        else:
            logger = self.logger

        if self.optim_state["cache_active"]:
            logger.info(
                " Iteration f-count/f-cache    Mean[ELBO]     Std[ELBO]     "
                + "sKL-iter[q]   K[q]  Convergence    Action"
            )
        else:
            if (
                self.optim_state["uncertainty_handling_level"] > 0
                and self.options.get("maxrepeatedobservations") > 0
            ):
                logger.info(
                    " Iteration   f-count (x-count)   Mean[ELBO]     Std[ELBO]"
                    + "     sKL-iter[q]   K[q]  Convergence  Action"
                )
            else:
                logger.info(
                    " Iteration  f-count    Mean[ELBO]    Std[ELBO]    "
                    + "sKL-iter[q]   K[q]  Convergence  Action"
                )

    def _setup_logging_display_format(self):
        """
        Private method to set up the display format for logging the iterations.
        """
        if self.optim_state["cache_active"]:
            display_format = " {:5.0f}     {:5.0f}  /{:5.0f}   {:12.2f}  "
            display_format += (
                "{:12.2f}  {:12.2f}     {:4.0f} {:10.3g}       {}"
            )
        else:
            if (
                self.optim_state["uncertainty_handling_level"] > 0
                and self.options.get("maxrepeatedobservations") > 0
            ):
                display_format = " {:5.0f}       {:5.0f} {:5.0f} {:12.2f}  "
                display_format += (
                    "{:12.2f}  {:12.2f}     {:4.0f} {:10.3g}     "
                )
                display_format += "{}"
            else:
                display_format = " {:5.0f}      {:5.0f}   {:12.2f} {:12.2f} "
                display_format += "{:12.2f}     {:4.0f} {:10.3g}     {}"

        return display_format

    def _init_logger(self, substring=""):
        """
        Private method to initialize the logging object.

        Parameters
        ----------
        substring : str
            A substring to append to the logger name (used to create separate
            logging objects for initialization and optimization, in case
            options change in between). Default "" (empty string).

        Returns
        -------
        logger : logging.Logger
            The main logging interface.
        """
        # set up VBMC logger
        logger = logging.getLogger("VBMC" + substring)
        logger.setLevel(logging.INFO)
        if self.options.get("display") == "off":
            logger.setLevel(logging.WARN)
        elif self.options.get("display") == "iter":
            logger.setLevel(logging.INFO)
        elif self.options.get("display") == "full":
            logger.setLevel(logging.DEBUG)
        # Add a special logger for sending messages only to the default stream:
        logger.stream_only = logging.getLogger("VBMC.stream_only")

        # Options and special handling for writing to a file:

        # If logging for the first time, get write mode from user options
        # (default "a" for append)
        if substring == "_init":
            log_file_mode = self.options.get("logfilemode", "a")
        # On subsequent writes, switch to append mode:
        else:
            log_file_mode = "a"

        # Avoid duplicating a handler for the same log file
        # (remove duplicates, re-add below)
        for handler in logger.handlers:
            if handler.baseFilename == os.path.abspath(
                    self.options.get("logfilename")
            ):
                logger.removeHandler(handler)

        if self.options.get("logfilename")\
           and self.options.get("logfilelevel"):
            file_handler = logging.FileHandler(
                filename=self.options["logfilename"],
                mode=log_file_mode
            )

            # Set file logger level according to string or logging level:
            log_file_level = self.options.get("logfilelevel", logging.INFO)
            if log_file_level == "off":
                file_handler.setLevel(logging.WARN)
            elif log_file_level == "iter":
                file_handler.setLevel(logging.INFO)
            elif log_file_level == "full":
                file_handler.setLevel(logging.DEBUG)
            elif log_file_level in [0, 10, 20, 30, 40, 50]:
                file_handler.setLevel(log_file_level)
            else:
                raise ValueError("Log file logging level is not a recognized" +
                                 "string or logging level.")

            # Add a filter to ignore messages sent to logger.stream_only:
            def log_file_filter(record):
                return record.name != "VBMC.stream_only"
            file_handler.addFilter(log_file_filter)

            logger.addHandler(file_handler)

        return logger
