import copy
import logging
import math
import os
import sys
from textwrap import indent

import gpyreg as gpr
import matplotlib.pyplot as plt
import numpy as np

from pyvbmc.formatting import full_repr, summarize
from pyvbmc.function_logger import FunctionLogger
from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.stats import kl_div_mvn
from pyvbmc.timer import main_timer as timer
from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.whitening import warp_gp_and_vp, warp_input

from .active_sample import active_sample
from .gaussian_process_train import reupdate_gp, train_gp
from .iteration_history import IterationHistory
from .options import Options
from .variational_optimization import optimize_vp, update_K


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
    log_density : callable
        A given target log-posterior or log-likelihood. If ``log_prior`` is
        ``None``, ``log_density`` accepts input ``x`` and returns the value of
        the target log-joint, that is, the unnormalized log-posterior density
        at ``x``. If ``log_prior`` is not ``None``, ``log_density`` should
        return the unnormalized log-likelihood. In either case, if
        ``options["specifytargetnoise"]`` is true, ``log_density`` should
        return a tuple where the first element is the noisy log-density, and
        the second is an estimate of the standard deviation of the noise.
    x0 : np.ndarray, optional
        Starting point for the inference. Ideally ``x0`` is a point in the
        proximity of the mode of the posterior. Default is ``None``.
    lower_bounds, upper_bounds : np.ndarray, optional
        ``lower_bounds`` (`LB`) and ``upper_bounds`` (`UB`) define a set
        of strict lower and upper bounds for the coordinate vector, `x`, so
        that the posterior has support on `LB` < `x` < `UB`.
        If scalars, the bound is replicated in each dimension. Use
        ``None`` for `LB` and `UB` if no bounds exist. Set `LB` [`i`] = -``inf``
        and `UB` [`i`] = ``inf`` if the `i`-th coordinate is unbounded (while
        other coordinates may be bounded). Note that if `LB` and `UB` contain
        unbounded variables, the respective values of `PLB` and `PUB` need to
        be specified (see below), by default ``None``.
    plausible_lower_bounds, plausible_upper_bounds : np.ndarray, optional
        Specifies a set of ``plausible_lower_bounds`` (`PLB`) and
        ``plausible_upper_bounds`` (`PUB`) such that `LB` < `PLB` < `PUB` < `UB`.
        Both `PLB` and `PUB` need to be finite. `PLB` and `PUB` represent a
        "plausible" range, which should denote a region of high posterior
        probability mass. Among other things, the plausible box is used to
        draw initial samples and to set priors over hyperparameters of the
        algorithm. When in doubt, we found that setting `PLB` and `PUB` using
        the topmost ~68% percentile range of the prior (e.g, mean +/- 1 SD
        for a Gaussian prior) works well in many cases (but note that
        additional information might afford a better guess). Both are
        by default ``None``.
    options : dict, optional
        Additional options can be passed as a dict. Please refer to the
        VBMC options page for the default options. If no ``options`` are
        passed, the default options are used.
    log_prior : callable, optional
        An optional separate log-prior function, which should accept a single
        argument `x` and return the log-density of the prior at `x`. If
        ``log_prior`` is not ``None``, the argument ``log_density`` is assumed
        to represent the log-likelihood (otherwise it is assumed to represent
        the log-joint).
    sample_prior : callable, optional
        An optional function which accepts a single argument `n` and returns an
        array of samples from the prior, of shape `(n, D)`, where `D` is the
        problem dimension. Currently unused.

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
    in the PyVBMC documentation:
    https://acerbilab.github.io/pyvbmc/_examples/pyvbmc_example_1.html
    """

    def __init__(
        self,
        log_density: callable,
        x0: np.ndarray = None,
        lower_bounds: np.ndarray = None,
        upper_bounds: np.ndarray = None,
        plausible_lower_bounds: np.ndarray = None,
        plausible_upper_bounds: np.ndarray = None,
        options: dict = None,
        log_prior: callable = None,
        sample_prior: callable = None,
    ):
        # set up root logger (only changes stuff if not initialized yet)
        logging.basicConfig(stream=sys.stdout, format="%(message)s")

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

        if x0.ndim == 1:
            logging.warning("Reshaping x0 to row vector.")
            x0 = x0.reshape((1, -1))
        self.D = x0.shape[1]
        # load basic and advanced options and validate the names
        pyvbmc_path = os.path.dirname(os.path.realpath(__file__))
        basic_path = pyvbmc_path + "/option_configs/basic_vbmc_options.ini"
        self.options = Options(
            basic_path,
            evaluation_parameters={"D": self.D},
            user_options=options,
        )

        advanced_path = (
            pyvbmc_path + "/option_configs/advanced_vbmc_options.ini"
        )
        self.options.load_options_file(
            advanced_path,
            evaluation_parameters={"D": self.D},
        )
        self.options.update_defaults()
        self.options.validate_option_names([basic_path, advanced_path])

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
        ) = self._bounds_check(
            x0,
            lower_bounds,
            upper_bounds,
            plausible_lower_bounds,
            plausible_upper_bounds,
        )

        # starting point
        if not np.all(np.isfinite(self.x0)):
            # print('Initial starting point is invalid or not provided.
            # Starting from center of plausible region.\n');
            self.x0 = 0.5 * (
                self.plausible_lower_bounds + self.plausible_upper_bounds
            )

        # Initialize transformation to unbounded parameters
        self.parameter_transformer = ParameterTransformer(
            self.D,
            self.lower_bounds,
            self.upper_bounds,
            self.plausible_lower_bounds,
            self.plausible_upper_bounds,
            transform_type=self.options["bounded_transform"],
        )

        # Initialize variational posterior
        self.vp = VariationalPosterior(
            D=self.D,
            K=self.options.get("k_warmup"),
            x0=self.x0,
            parameter_transformer=self.parameter_transformer,
        )
        if not self.options.get("warmup"):
            self.vp.optimize_mu = self.options.get("variable_means")
            self.vp.optimize_weights = self.options.get("variable_weights")

        # The underlying Gaussian process which corresponds to current vp
        self.gp = None
        self.hyp_dict = (
            {}
        )  # For storing auxilary info related to gp hyperparameters

        # Optimization of vbmc starts from iteration 0
        self.iteration = -1
        # Whether the optimization has finished
        self.is_finished = False

        self.optim_state = self._init_optim_state()

        # Initialize log-joint
        self.sample_prior = sample_prior
        if callable(log_prior):
            self.log_prior = log_prior
            self.log_likelihood = log_density
            if self.optim_state["uncertainty_handling_level"] == 2:

                def log_joint(theta):
                    log_likelihood, noise_est = log_density(theta)
                    return log_likelihood + log_prior(theta), noise_est

            else:

                def log_joint(theta):
                    return log_density(theta) + log_prior(theta)

        elif log_prior is None:
            log_joint = log_density
        else:
            raise TypeError("`prior` must be a callable or `None`.")
        self.log_joint = log_joint

        self.function_logger = FunctionLogger(
            fun=log_joint,
            D=self.D,
            noise_flag=self.optim_state.get("uncertainty_handling_level") > 0,
            uncertainty_handling_level=self.optim_state.get(
                "uncertainty_handling_level"
            ),
            cache_size=self.options.get("cache_size"),
            parameter_transformer=self.parameter_transformer,
        )

        self.x0 = self.parameter_transformer(self.x0)
        self.random_state = np.random.get_state()
        self.iteration_history = IterationHistory(
            [
                "r_index",
                "elcbo_impro",
                "stable",
                "elbo",
                "vp",
                "warmup",
                "iter",
                "elbo_sd",
                "lcb_max",
                "data_trim_list",
                "gp",
                "gp_hyp_full",
                "Ns_gp",
                "timer",
                "optim_state",
                "sKL",
                "sKL_true",
                "pruned",
                "var_ss",
                "func_count",
                "n_eff",
                "logging_action",
                # For resuming optimization from a specific iteration, mostly
                # useful for debugging
                "function_logger",
                "random_state",
            ]
        )

    def _bounds_check(
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
                    "plausible bounds from starting set X0..."
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
                        "determined from starting set. Using hard upper/lower"
                        " bounds for those instead."
                    )
            else:
                self.logger.warning(
                    "vbmc:pbUnspecified: Plausible lower/upper bounds PLB and"
                    "/or PUB not specified and X0 is not a valid starting set. "
                    "Using hard upper/lower bounds instead."
                )
                if plausible_lower_bounds is None:
                    plausible_lower_bounds = np.copy(lower_bounds)
                if plausible_upper_bounds is None:
                    plausible_upper_bounds = np.copy(upper_bounds)

        # Try to reshape bounds to row vectors
        lower_bounds = np.atleast_1d(lower_bounds)
        upper_bounds = np.atleast_1d(upper_bounds)
        plausible_lower_bounds = np.atleast_1d(plausible_lower_bounds)
        plausible_upper_bounds = np.atleast_1d(plausible_upper_bounds)
        try:
            if lower_bounds.shape != (1, D):
                logging.warning("Reshaping lower bounds to (1, %d).", D)
                lower_bounds = lower_bounds.reshape((1, D))
            if upper_bounds.shape != (1, D):
                logging.warning("Reshaping upper bounds to (1, %d).", D)
                upper_bounds = upper_bounds.reshape((1, D))
            if plausible_lower_bounds.shape != (1, D):
                logging.warning(
                    "Reshaping plausible lower bounds to (1, %d).", D
                )
                plausible_lower_bounds = plausible_lower_bounds.reshape((1, D))
            if plausible_upper_bounds.shape != (1, D):
                logging.warning(
                    "Reshaping plausible upper bounds to (1, %d).", D
                )
                plausible_upper_bounds = plausible_upper_bounds.reshape((1, D))
        except ValueError as exc:
            raise ValueError(
                "Bounds must match problem dimension D=%d.", D
            ) from exc

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

        # Cast all vectors to floats
        # (integer_vars are represented as floats but handled separately).
        if np.issubdtype(x0.dtype, np.integer):
            logging.warning("Casting initial points to floating point.")
            x0 = x0.astype(np.float64)
        if np.issubdtype(lower_bounds.dtype, np.integer):
            logging.warning("Casting lower bounds to floating point.")
            lower_bounds = lower_bounds.astype(np.float64)
        if np.issubdtype(upper_bounds.dtype, np.integer):
            logging.warning("Casting upper bounds to floating point.")
            upper_bounds = upper_bounds.astype(np.float64)
        if np.issubdtype(plausible_lower_bounds.dtype, np.integer):
            logging.warning(
                "Casting plausible lower bounds to floating point."
            )
            plausible_lower_bounds = plausible_lower_bounds.astype(np.float64)
        if np.issubdtype(plausible_upper_bounds.dtype, np.integer):
            logging.warning(
                "Casting plausible upper bounds to floating point."
            )
            plausible_upper_bounds = plausible_upper_bounds.astype(np.float64)

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
                "or numerically too close to the hard bounds LB and UB. "
                "Moving the initial points more inside..."
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
                "and plausible bounds should not be too close. "
                "Moving plausible bounds."
            )
            plausible_lower_bounds = np.maximum(plausible_lower_bounds, LB_eff)
            plausible_upper_bounds = np.minimum(plausible_upper_bounds, UB_eff)

        # Check that all X0 are inside the plausible bounds,
        # move bounds otherwise
        if np.any(x0 <= plausible_lower_bounds) or np.any(
            x0 >= plausible_upper_bounds
        ):
            self.logger.warning(
                "vbmc:InitialPointsOutsidePB. The starting points X0"
                " are not inside the provided plausible bounds PLB and "
                "PUB. Expanding the plausible bounds..."
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
        if np.any(
            (np.isfinite(lower_bounds) & np.isinf(upper_bounds))
            | (np.isinf(lower_bounds) & np.isfinite(upper_bounds))
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
        y_orig = np.array(self.options.get("f_vals")).ravel()
        if len(y_orig) == 0:
            y_orig = np.full([self.x0.shape[0]], np.nan)
        if len(self.x0) != len(y_orig):
            raise ValueError(
                """vbmc:MismatchedStartingInputs The number of
            points in X0 and of their function values as specified in
            self.options.f_vals are not the same."""
            )

        optim_state = {}
        optim_state["cache"] = {}
        optim_state["cache"]["x_orig"] = self.x0
        optim_state["cache"]["y_orig"] = y_orig

        # Does the starting cache contain function values?
        optim_state["cache_active"] = np.any(
            np.isfinite(optim_state.get("cache").get("y_orig"))
        )

        # Integer variables
        optim_state["integer_vars"] = np.full(self.D, False)
        if len(self.options.get("integer_vars")) > 0:
            integeridx = self.options.get("integer_vars") != 0
            optim_state["integer_vars"][integeridx] = True
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
        optim_state["lb_orig"] = self.lower_bounds.copy()
        optim_state["ub_orig"] = self.upper_bounds.copy()
        optim_state["plb_orig"] = self.plausible_lower_bounds.copy()
        optim_state["pub_orig"] = self.plausible_upper_bounds.copy()
        eps_orig = (self.upper_bounds - self.lower_bounds) * self.options.get(
            "tol_bound_x"
        )
        # inf - inf raises warning in numpy, but output is correct
        with np.errstate(invalid="ignore"):
            optim_state["lb_eps_orig"] = self.lower_bounds + eps_orig
            optim_state["ub_eps_orig"] = self.upper_bounds - eps_orig

        # Transform variables (Transform of lower bounds and upper bounds can
        # create warning but we are aware of this and output is correct)
        with np.errstate(divide="ignore"):
            optim_state["lb_tran"] = self.parameter_transformer(
                self.lower_bounds
            )
            optim_state["ub_tran"] = self.parameter_transformer(
                self.upper_bounds
            )
        optim_state["plb_tran"] = self.parameter_transformer(
            self.plausible_lower_bounds
        )
        optim_state["pub_tran"] = self.parameter_transformer(
            self.plausible_upper_bounds
        )

        # Before first iteration
        # Iterations are from 0 onwards in optimize so we should have -1
        # here. In MATLAB this was 0.
        optim_state["iter"] = -1

        # Estimate of GP observation noise around the high posterior
        # density region
        optim_state["sn2_hpd"] = np.inf

        # When was the last warping action performed (number of iterations)
        optim_state["last_warping"] = -np.inf

        # When was the last warping action performed and not undone
        # (number of iterations)
        optim_state["last_successful_warping"] = -np.inf

        # Number of warpings performed
        optim_state["warping_count"] = 0

        # When GP hyperparameter sampling is switched with optimization
        if self.options.get("ns_gp_max") > 0:
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
        if self.options.get("proposal_fcn") is None:
            optim_state["proposal_fcn"] = "@(x)proposal_vbmc"
        else:
            optim_state["proposal_fcn"] = self.options.get("proposal_fcn")

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
        optim_state["vp_K"] = self.options.get("k_warmup")

        # Number of variational components pruned in last iteration
        optim_state["pruned"] = 0

        # Need to switch from deterministic entropy to stochastic entropy
        optim_state["entropy_switch"] = self.options.get("entropy_switch")

        # Only use deterministic entropy if D larger than a fixed number
        if self.D < self.options.get("det_entropy_min_d"):
            optim_state["entropy_switch"] = False

        # Tolerance threshold on GP variance (used by some acquisition fcns)
        optim_state["tol_gp_var"] = self.options.get("tol_gp_var")

        # Copy maximum number of fcn. evaluations,
        # used by some acquisition fcns.
        optim_state["max_fun_evals"] = self.options.get("max_fun_evals")

        # By default, apply variance-based regularization
        # to acquisition functions
        optim_state["variance_regularized_acqfcn"] = True

        # Setup search cache
        optim_state["search_cache"] = []

        # Set uncertainty handling level
        # (0: none; 1: unknown noise level; 2: user-provided noise)
        if self.options.get("specify_target_noise"):
            optim_state["uncertainty_handling_level"] = 2
        elif len(self.options.get("uncertainty_handling")) > 0:
            optim_state["uncertainty_handling_level"] = 1
        else:
            optim_state["uncertainty_handling_level"] = 0

        # Empty hedge struct for acquisition functions
        if self.options.get("acq_hedge"):
            optim_state["hedge"] = []

        # List of points at the end of each iteration
        optim_state["iter_list"] = {}
        optim_state["iter_list"]["u"] = []
        optim_state["iter_list"]["f_val"] = []
        optim_state["iter_list"]["f_sd"] = []
        optim_state["iter_list"]["fhyp"] = []

        # Deterministic entropy approximation lower/upper factor
        optim_state["entropy_alpha"] = self.options.get("det_entropy_alpha")

        # Repository of variational solutions (not used in Python)
        # optim_state["vp_repo"] = []

        # Repeated measurement streak
        optim_state["repeated_observations_streak"] = 0

        # List of data trimming events
        optim_state["data_trim_list"] = []

        # Expanding search bounds
        prange = optim_state.get("pub_tran") - optim_state.get("plb_tran")
        optim_state["lb_search"] = np.maximum(
            optim_state.get("plb_tran")
            - prange * self.options.get("active_search_bound"),
            optim_state.get("lb_tran"),
        )
        optim_state["ub_search"] = np.minimum(
            optim_state.get("pub_tran")
            + prange * self.options.get("active_search_bound"),
            optim_state.get("ub_tran"),
        )

        # Initialize Gaussian process settings
        # Squared exponential kernel with separate length scales
        optim_state["gp_cov_fun"] = 1

        if optim_state.get("uncertainty_handling_level") == 0:
            # Observation noise for stability
            optim_state["gp_noise_fun"] = [1, 0, 0]
        elif optim_state.get("uncertainty_handling_level") == 1:
            # Infer noise
            optim_state["gp_noise_fun"] = [1, 2, 0]
        elif optim_state.get("uncertainty_handling_level") == 2:
            # Provided heteroskedastic noise
            optim_state["gp_noise_fun"] = [1, 1, 0]

        if (
            self.options.get("noise_shaping")
            and optim_state["gp_noise_fun"][1] == 0
        ):
            optim_state["gp_noise_fun"][1] = 1

        optim_state["gp_mean_fun"] = self.options.get("gp_mean_fun")
        valid_gp_mean_funs = [
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

        if not optim_state["gp_mean_fun"] in valid_gp_mean_funs:
            raise ValueError(
                """vbmc:UnknownGPmean:Unknown/unsupported GP mean
            function. Supported mean functions are zero, const,
            egquad, and se"""
            )
        optim_state["int_mean_fun"] = self.options.get("gp_int_mean_fun")
        # more logic here in matlab

        # Starting threshold on y for output warping
        if self.options.get("fitness_shaping"):
            optim_state["out_warp_delta"] = self.options.get(
                "out_warp_thresh_base"
            )
        else:
            optim_state["out_warp_delta"] = []

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
        results : dict
            A dictionary with additional information about the VBMC run.
        """
        # Initialize main logger with potentially new options:
        self.logger = self._init_logger()
        # set up strings for logging of the iteration
        display_format = self._setup_logging_display_format()

        if self.optim_state["uncertainty_handling_level"] > 0:
            self.logger.info(
                "Beginning variational optimization assuming NOISY observations"
                " of the log-joint"
            )
        else:
            self.logger.info(
                "Beginning variational optimization assuming EXACT observations"
                " of the log-joint."
            )

        if self.is_finished:
            self.logger.warning("Continuing optimization from previous state.")
            self.is_finished = False
            self.vp = self.iteration_history["vp"][-1]
            self.optim_state = self.iteration_history["optim_state"][-1]
        self._log_column_headers()
        while not self.is_finished:
            self.iteration += 1
            # Reset timer:
            timer.reset()
            self.optim_state["iter"] = self.iteration
            self.optim_state["redo_roto_scaling"] = False
            vp_old = copy.deepcopy(self.vp)

            self.logging_action = []

            if self.iteration == 0 and self.optim_state["warmup"]:
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
            if self.options["incremental_warp_delay"]:
                WarpDelay = self.options["warp_every_iters"] * np.max(
                    [1, self.optim_state["warping_count"]]
                )
            else:
                WarpDelay = self.options["warp_every_iters"]

            doWarping = (
                (
                    self.options.get("warp_rotoscaling")
                    or self.options.get("warp_nonlinear")
                )
                and (self.iteration > 0)
                and (not self.optim_state["warmup"])
                and (
                    self.iteration - self.optim_state["last_warping"]
                    > WarpDelay
                )
                and (self.vp.K >= self.options["warp_min_k"])
                and (
                    self.iteration_history["r_index"][self.iteration - 1]
                    < self.options["warp_tol_reliability"]
                )
                and (self.vp.D > 1)
            )

            if doWarping:
                timer.start_timer("warping")
                vp_tmp, __, __, __ = self.determine_best_vp()
                vp_tmp = copy.deepcopy(vp_tmp)
                # Store variables in case warp needs to be undone:
                # (vp_old copied above)
                optim_state_old = copy.deepcopy(self.optim_state)
                gp_old = copy.deepcopy(self.gp)
                function_logger_old = copy.deepcopy(self.function_logger)
                elbo_old = self.iteration_history["elbo"][-1]
                elbo_sd_old = self.iteration_history["elbo_sd"][-1]
                hyp_dict_old = copy.deepcopy(self.hyp_dict)
                # Compute and apply whitening transform:
                (
                    parameter_transformer_warp,
                    self.optim_state,
                    self.function_logger,
                    warp_action,
                ) = warp_input(
                    vp_tmp,
                    self.optim_state,
                    self.function_logger,
                    self.options,
                )

                self.vp, self.hyp_dict["hyp"] = warp_gp_and_vp(
                    parameter_transformer_warp, self.gp, self.vp, self
                )

                self.logging_action.append(warp_action)
                timer.stop_timer("warping")

                if self.options.get("warp_undo_check"):
                    ## Train gp

                    timer.start_timer("gp_train")

                    self.gp, Ns_gp, sn2_hpd, self.hyp_dict = train_gp(
                        self.hyp_dict,
                        self.optim_state,
                        self.function_logger,
                        self.iteration_history,
                        self.options,
                        self.optim_state["plb_tran"],
                        self.optim_state["pub_tran"],
                    )
                    self.optim_state["sn2_hpd"] = sn2_hpd

                    timer.stop_timer("gp_train")

                    ## Optimize variational parameters
                    timer.start_timer("variational_fit")

                    if not self.vp.optimize_mu:
                        # Variational components fixed to training inputs
                        self.vp.mu = self.gp.X.T
                        Knew = self.vp.mu.shape[1]
                    else:
                        # Update number of variational mixture components
                        Knew = self.vp.K

                    # Decide number of fast/slow optimizations
                    N_fastopts = math.ceil(
                        self.options.eval("ns_elbo", {"K": self.vp.K})
                    )
                    N_slowopts = self.options.get(
                        "elbo_starts"
                    )  # Full optimizations.

                    # Run optimization of variational parameters
                    self.vp, var_ss, pruned = optimize_vp(
                        self.options,
                        self.optim_state,
                        self.vp,
                        self.gp,
                        N_fastopts,
                        N_slowopts,
                        Knew,
                    )

                    self.optim_state["vp_K"] = self.vp.K
                    # Save current entropy
                    self.optim_state["H"] = self.vp.stats["entropy"]

                    # Get real variational posterior (might differ from training posterior)
                    # vp_real = vp.vptrain2real(0, self.options)
                    vp_real = self.vp
                    elbo = vp_real.stats["elbo"]
                    elbo_sd = vp_real.stats["elbo_sd"]

                    timer.stop_timer("variational_fit")

                    # Keep warping only if it substantially improves ELBO
                    # and uncertainty does not blow up too much
                    if (
                        elbo
                        < (elbo_old + self.options["warp_tol_improvement"])
                    ) or (
                        elbo_sd
                        > (
                            elbo_sd_old
                            * self.options["warp_tol_sd_multiplier"]
                            + self.options["warp_tol_sd_base"]
                        )
                    ):
                        # Undo input warping:
                        self.vp = vp_old
                        self.gp = gp_old
                        self.optim_state = optim_state_old
                        self.function_logger = function_logger_old
                        self.hyp_dict = hyp_dict_old

                        # Still keep track of failed warping (failed warp counts twice)
                        self.optim_state["warping_count"] += 2
                        self.optim_state["last_warping"] = self.optim_state[
                            "iter"
                        ]
                        self.logging_action.append("undo " + warp_action)

            ## Actively sample new points into the training set
            timer.start_timer("active_sampling")
            self.parameter_transformer = self.vp.parameter_transformer
            self.function_logger.parameter_transformer = (
                self.parameter_transformer
            )

            if self.iteration == 0:
                new_funevals = self.options.get("fun_eval_start")
            else:
                new_funevals = self.options.get("fun_evals_per_iter")

            # Careful with Xn, in MATLAB this condition is > 0
            # due to 1-based indexing.
            if self.function_logger.Xn >= 0:
                self.function_logger.y_max = np.max(
                    self.function_logger.y[self.function_logger.X_flag]
                )

            if self.optim_state.get("skip_active_sampling"):
                self.optim_state["skip_active_sampling"] = False
            else:
                if (
                    self.gp is not None
                    and self.options.get("separate_search_gp")
                    and not self.options.get("varactivesample")
                ):
                    # Train a distinct GP for active sampling
                    # Since we are doing iterations from 0 onwards
                    # instead of from 1 onwards, this should be checking
                    # oddness, not evenness.
                    if self.iteration % 2 == 1:
                        meantemp = self.optim_state.get("gp_mean_fun")
                        self.optim_state["gp_mean_fun"] = "const"
                        timer.start_timer("separate_gp_train")
                        gp_search, Ns_gp, sn2_hpd, self.hyp_dict = train_gp(
                            self.hyp_dict,
                            self.optim_state,
                            self.function_logger,
                            self.iteration_history,
                            self.options,
                            self.optim_state["plb_tran"],
                            self.optim_state["pub_tran"],
                        )
                        timer.stop_timer("separate_gp_train")
                        self.optim_state["sn2_hpd"] = sn2_hpd
                        self.optim_state["gp_mean_fun"] = meantemp
                    else:
                        gp_search = self.gp
                else:
                    gp_search = self.gp

                # Perform active sampling
                if self.options.get("varactivesample"):
                    # FIX TIMER HERE IF USING THIS
                    # [optimState,vp,t_active,t_func] =
                    # variationalactivesample_vbmc(optimState,new_funevals,
                    # funwrapper,vp,vp_old,gp_search,options)
                    sys.exit("Function currently not supported")
                else:
                    self.optim_state["hyp_dict"] = self.hyp_dict
                    (
                        self.function_logger,
                        self.optim_state,
                        self.vp,
                        self.gp,
                    ) = active_sample(
                        gp_search,
                        new_funevals,
                        self.optim_state,
                        self.function_logger,
                        self.iteration_history,
                        self.vp,
                        self.options,
                    )
                    self.hyp_dict = self.optim_state["hyp_dict"]

            # Number of training inputs
            self.optim_state["N"] = self.function_logger.Xn + 1
            self.optim_state["n_eff"] = np.sum(
                self.function_logger.n_evals[self.function_logger.X_flag]
            )

            timer.stop_timer("active_sampling")

            ## Train gp

            timer.start_timer("gp_train")

            self.gp, Ns_gp, sn2_hpd, self.hyp_dict = train_gp(
                self.hyp_dict,
                self.optim_state,
                self.function_logger,
                self.iteration_history,
                self.options,
                self.optim_state["plb_tran"],
                self.optim_state["pub_tran"],
            )
            self.optim_state["sn2_hpd"] = sn2_hpd

            timer.stop_timer("gp_train")

            # Check if reached stable sampling regime
            if (
                Ns_gp == self.options.get("stable_gp_samples")
                and self.optim_state.get("stop_sampling") == 0
            ):
                self.optim_state["stop_sampling"] = self.optim_state.get("N")

            ## Optimize variational parameters
            timer.start_timer("variational_fit")

            if not self.vp.optimize_mu:
                # Variational components fixed to training inputs
                self.vp.mu = self.gp.X.T.copy()
                Knew = self.vp.mu.shape[1]
            else:
                # Update number of variational mixture components
                Knew = update_K(
                    self.optim_state, self.iteration_history, self.options
                )

            # Decide number of fast/slow optimizations
            N_fastopts = math.ceil(
                self.options.eval("ns_elbo", {"K": self.vp.K})
            )

            if self.optim_state.get("recompute_var_post") or (
                self.options.get("always_refit_vp")
            ):
                # Full optimizations
                N_slowopts = self.options.get("elbo_starts")
                self.optim_state["recompute_var_post"] = False
            else:
                # Only incremental change from previous iteration
                N_fastopts = math.ceil(
                    N_fastopts * self.options.get("ns_elbo_incr")
                )
                N_slowopts = 1
            # Run optimization of variational parameters
            self.vp, var_ss, pruned = optimize_vp(
                self.options,
                self.optim_state,
                self.vp,
                self.gp,
                N_fastopts,
                N_slowopts,
                Knew,
            )

            self.optim_state["vp_K"] = self.vp.K
            # Save current entropy
            self.optim_state["H"] = self.vp.stats["entropy"]

            # Get real variational posterior (might differ from training posterior)
            # vp_real = vp.vptrain2real(0, self.options)
            vp_real = self.vp
            elbo = vp_real.stats["elbo"]
            elbo_sd = vp_real.stats["elbo_sd"]

            timer.stop_timer("variational_fit")

            # Finalize iteration

            timer.start_timer("finalize")

            # Compute symmetrized KL-divergence between old and new posteriors
            Nkl = 1e5

            sKL = max(
                0,
                0.5
                * np.sum(
                    self.vp.kl_div(
                        vp2=vp_old,
                        N=Nkl,
                        gauss_flag=self.options.get("kl_gauss"),
                    )
                ),
            )

            # Evaluate max LCB of GP prediction on all training inputs
            f_mu, f_s2 = self.gp.predict(
                self.gp.X, self.gp.y, self.gp.s2, add_noise=False
            )
            self.optim_state["lcb_max"] = np.amax(
                f_mu - self.options.get("elcbo_impro_weight") * np.sqrt(f_s2)
            )

            # Compare variational posterior's moments with ground truth
            if (
                self.options.get("true_mean")
                and self.options.get("true_cov")
                and np.all(np.isfinite(self.options.get("true_mean")))
                and np.all(np.isfinite(self.options.get("true_cov")))
            ):
                mubar_orig, sigma_orig = vp_real.moments(1e6, True, True)

                kl = kl_div_mvn(
                    mubar_orig,
                    sigma_orig,
                    self.options.get("true_mean"),
                    self.options.get("true_cov"),
                )
                sKL_true = 0.5 * np.sum(kl)
            else:
                sKL_true = None

            # Record moments in transformed space
            mubar, sigma = self.vp.moments(orig_flag=False, cov_flag=True)
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
                wRun = self.options.get("moments_run_weight") ** Nnew
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
            iteration_values = {
                "iter": self.iteration,
                "vp": self.vp,
                "elbo": elbo,
                "elbo_sd": elbo_sd,
                "var_ss": var_ss,
                "sKL": sKL,
                "sKL_true": sKL_true,
                "gp": self.gp,
                "gp_hyp_full": self.gp.get_hyperparameters(as_array=True),
                "Ns_gp": Ns_gp,
                "pruned": pruned,
                "timer": timer,
                "func_count": self.function_logger.func_count,
                "lcb_max": self.optim_state["lcb_max"],
                "n_eff": self.optim_state["n_eff"],
                "function_logger": self.function_logger,
            }
            # Record all useful stats
            self.iteration_history.record_iteration(
                iteration_values,
                self.iteration,
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
                self.is_finished,
                termination_message,
                success_flag,
            ) = self._check_termination_conditions()

            # Save stability
            self.vp.stats["stable"] = self.iteration_history["stable"][
                self.iteration
            ]

            # Check if we are still warming-up
            if self.optim_state.get("warmup") and self.iteration > 0:
                if self.options.get("recompute_lcb_max"):
                    self.optim_state[
                        "lcb_max_vec"
                    ] = self._recompute_lcb_max().T
                trim_flag = self._check_warmup_end_conditions()
                if trim_flag:
                    self._setup_vbmc_after_warmup()
                    # Re-update GP after trimming
                    self.gp = reupdate_gp(self.function_logger, self.gp)
                if not self.optim_state.get("warmup"):
                    self.vp.optimize_mu = self.options.get("variable_means")
                    self.vp.optimize_weights = self.options.get(
                        "variable_weights"
                    )

                    # Switch to main algorithm options
                    # options = options_main
                    # Reset GP hyperparameter covariance
                    # hypstruct.runcov = []
                    self.hyp_dict["runcov"] = None
                    # Reset VP repository (not used in python)
                    self.optim_state["vp_repo"] = []

                    # Re-get acq info
                    # self.optim_state['acq_info'] = getAcqInfo(
                    #    options.SearchAcqFcn
                    # )
            # Needs to be below the above block since warmup value can change
            # in _check_warmup_end_conditions
            self.iteration_history.record(
                "warmup", self.optim_state.get("warmup"), self.iteration
            )

            # Check and update fitness shaping / output warping threshold
            if (
                self.optim_state.get("out_warp_delta") != []
                and self.optim_state.get("R") is not None
                and (
                    self.optim_state.get("R")
                    < self.options.get("warp_tol_reliability")
                )
            ):
                Xrnd, _ = self.vp.sample(N=int(2e4), orig_flag=False)
                ymu, _ = self.gp.predict(Xrnd, add_noise=True)
                ydelta = max(
                    [0, self.function_logger.y_max - np.quantile(ymu, 1e-3)]
                )
                if (
                    ydelta
                    > self.optim_state.get("out_warp_delta")
                    * self.options.get("out_warp_thresh_tol")
                    and self.optim_state.get("R") is not None
                    and self.optim_state.get("R") < 1
                ):
                    self.optim_state["out_warp_delta"] = self.optim_state.get(
                        "out_warp_delta"
                    ) * self.options.get("out_warp_thresh_mult")

            # Write iteration output
            # Stopped GP sampling this iteration?
            if (
                Ns_gp == self.options["stable_gp_samples"]
                and self.iteration_history["Ns_gp"][max(0, self.iteration - 1)]
                > self.options["stable_gp_samples"]
            ):
                if Ns_gp == 0:
                    self.logging_action.append("switch to GP opt")
                else:
                    self.logging_action.append("stable GP sampling")

            if self.options.get("print_iteration_header") is None:
                # Default behavior, try to guess based on plotting options:
                reprint_headers = (
                    self.options.get("plot")
                    and self.iteration > 0
                    and "inline" in plt.get_backend()
                )
            elif self.options["print_iteration_header"]:
                # Re-print every iteration after 0th
                reprint_headers = self.iteration > 0
            else:
                # Never re-print headers
                reprint_headers = False
            # Reprint the headers if desired:
            if reprint_headers:
                self._log_column_headers()

            if self.optim_state["cache_active"]:
                self.logger.info(
                    display_format.format(
                        self.iteration,
                        self.function_logger.func_count,
                        self.function_logger.cache_count,
                        elbo,
                        elbo_sd,
                        sKL,
                        self.vp.K,
                        self.optim_state["R"],
                        ", ".join(self.logging_action),
                    )
                )

            else:
                if (
                    self.optim_state["uncertainty_handling_level"] > 0
                    and self.options.get("max_repeated_observations") > 0
                ):
                    self.logger.info(
                        display_format.format(
                            self.iteration,
                            self.function_logger.func_count,
                            self.optim_state["N"],
                            elbo,
                            elbo_sd,
                            sKL,
                            self.vp.K,
                            self.optim_state["R"],
                            ", ".join(self.logging_action),
                        )
                    )
                else:
                    self.logger.info(
                        display_format.format(
                            self.iteration,
                            self.function_logger.func_count,
                            elbo,
                            elbo_sd,
                            sKL,
                            self.vp.K,
                            self.optim_state["R"],
                            ", ".join(self.logging_action),
                        )
                    )
            self.iteration_history.record(
                "logging_action", self.logging_action, self.iteration
            )

            # Plot iteration
            if self.options.get("plot"):
                if self.iteration > 0:
                    previous_gp = self.iteration_history["gp"][
                        self.iteration - 1
                    ]
                    # find points that are new in this iteration
                    # (hacky cause numpy only has 1D set diff)
                    # future fix: active sampling should return the set of
                    # indices of the added points
                    highlight_data = np.array(
                        [
                            i
                            for i, x in enumerate(self.gp.X)
                            if tuple(x) not in set(map(tuple, previous_gp.X))
                        ]
                    )
                else:
                    highlight_data = None

                if len(self.logging_action) > 0:
                    title = "VBMC iteration {} ({})".format(
                        self.iteration, ", ".join(self.logging_action)
                    )
                else:
                    title = "VBMC iteration {}".format(self.iteration)

                self.vp.plot(
                    plot_data=True,
                    highlight_data=highlight_data,
                    plot_vp_centres=True,
                    title=title,
                )
                plt.show()

            # Record optim_state and random state
            self.random_state = np.random.get_state()
            self.iteration_history.record_iteration(
                {
                    "optim_state": self.optim_state,
                    "random_state": self.random_state,
                },
                self.iteration,
            )

        # Pick "best" variational solution to return
        self.vp, elbo, elbo_sd, idx_best = self.determine_best_vp()

        if self.options.get("do_final_boost"):
            # Last variational optimization with large number of components
            self.vp, elbo, elbo_sd, changed_flag = self.final_boost(
                self.vp, self.iteration_history["gp"][idx_best]
            )
        else:
            changed_flag = False
        if changed_flag:
            # Recompute symmetrized KL-divergence
            if "vp_old" in locals():
                sKL = max(
                    0,
                    0.5
                    * np.sum(
                        self.vp.kl_div(
                            vp2=vp_old,
                            N=Nkl,
                            gauss_flag=self.options.get("kl_gauss"),
                        )
                    ),
                )
            else:
                sKL = -1  # sKL is undefined

            if self.options.get("plot"):
                self._log_column_headers()

            if (
                self.optim_state["uncertainty_handling_level"] > 0
                and self.options.get("max_repeated_observations") > 0
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
                        self.iteration_history.get("r_index")[idx_best],
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
                        self.iteration_history.get("r_index")[idx_best],
                        "finalize",
                    )
                )

        # plot final vp:
        if self.options.get("plot"):
            self.vp.plot(
                plot_data=True,
                highlight_data=None,
                plot_vp_centres=True,
                title="VBMC final ({} iterations)".format(self.iteration),
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
                " not converged."
            )

        results = self._create_result_dict(
            idx_best, termination_message, success_flag
        )

        return copy.deepcopy(self.vp), results

    def _check_warmup_end_conditions(self):
        """
        Private method to check the warmup end conditions.
        """
        iteration = self.optim_state.get("iter")

        # First requirement for stopping, no constant improvement of metric
        stable_count_flag = False
        stop_warmup_thresh = self.options.get(
            "stop_warmup_thresh"
        ) * self.options.get("fun_evals_per_iter")
        tol_stable_warmup_iters = math.ceil(
            self.options.get("tol_stable_warmup")
            / self.options.get("fun_evals_per_iter")
        )

        # MATLAB has +1 on the right side due to different indexing.
        if iteration > tol_stable_warmup_iters:
            # Vector of ELCBO (ignore first two iterations, ELCBO is unreliable)
            elcbo_vec = self.iteration_history.get("elbo") - self.options.get(
                "elcbo_impro_weight"
            ) * self.iteration_history.get("elbo_sd")
            # NB: Take care with MATLAB "end" indexing and off-by-one errors:
            max_now = np.amax(
                elcbo_vec[max(3, len(elcbo_vec) - tol_stable_warmup_iters) :]
            )
            max_before = np.amax(
                elcbo_vec[2 : max(3, len(elcbo_vec) - tol_stable_warmup_iters)]
            )
            stable_count_flag = (max_now - max_before) < stop_warmup_thresh

        # Vector of maximum lower confidence bounds (LCB) of fcn values
        lcb_max_vec = self.iteration_history.get("lcb_max")[: iteration + 1]

        # Second requirement, also no substantial improvement of max fcn value
        # in recent iters (unless already performing BO-like warmup)
        if self.options.get("warmup_check_max"):
            idx_last = np.full(lcb_max_vec.shape, False)
            recent_past = iteration - int(
                math.ceil(
                    self.options.get("tol_stable_warmup")
                    / self.options.get("fun_evals_per_iter")
                )
                + 1
            )
            idx_last[max(1, recent_past) :] = True
            impro_fcn = max(
                0,
                np.amax(lcb_max_vec[idx_last])
                - np.amax(lcb_max_vec[~idx_last]),
            )
        else:
            impro_fcn = 0

        no_recent_improvement_flag = impro_fcn < stop_warmup_thresh

        # Alternative criterion for stopping - no improvement over max fcn value
        max_thresh = np.amax(lcb_max_vec) - self.options.get("tol_improvement")
        idx_1st = np.ravel(np.argwhere(lcb_max_vec > max_thresh))[0]
        yy = self.iteration_history.get("func_count")[: iteration + 1]
        pos = yy[idx_1st]
        currentpos = self.function_logger.func_count
        no_longterm_improvement_flag = (currentpos - pos) > self.options.get(
            "warmup_no_impro_threshold"
        )

        if len(self.optim_state.get("data_trim_list")) > 0:
            last_data_trim = self.optim_state.get("data_trim_list")[-1]
        else:
            last_data_trim = -1 * np.Inf

        no_recent_trim_flag = (
            self.optim_state.get("N") - last_data_trim
        ) >= 10

        stop_warmup = (
            (stable_count_flag and no_recent_improvement_flag)
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
            self.iteration_history.get("r_index")[iteration]
            < self.options.get("stop_warmup_reliability")
            or len(self.optim_state.get("data_trim_list")) >= 1
        ):
            self.optim_state["warmup"] = False
            self.logging_action.append("end warm-up")
            threshold = self.options.get("warmup_keep_threshold") * (
                len(self.optim_state.get("data_trim_list")) + 1
            )
            self.optim_state["last_warmup"] = iteration

        else:
            # This may be a false alarm; prune and continue
            if self.options.get("warmup_keep_threshold_false_alarm") is None:
                warmup_keep_threshold_false_alarm = self.options.get(
                    "warmup_keep_threshold"
                )
            else:
                warmup_keep_threshold_false_alarm = self.options.get(
                    "warmup_keep_threshold_false_alarm"
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
        y_max = max(self.function_logger.y_orig[: self.function_logger.Xn + 1])
        n_keep_min = self.D + 1
        idx_keep = (y_max - self.function_logger.y_orig) < threshold
        if np.sum(idx_keep) < n_keep_min:
            y_temp = np.copy(self.function_logger.y_orig)
            y_temp[~np.isfinite(y_temp)] = -np.Inf
            order = np.argsort(y_temp * -1, axis=0)
            idx_keep[
                order[: min(n_keep_min, self.function_logger.Xn + 1)]
            ] = True
        # Note that using idx_keep[:, 0] is necessary since X_flag
        # is a 1D array and idx_keep a 2D array.
        self.function_logger.X_flag = np.logical_and(
            idx_keep[:, 0], self.function_logger.X_flag
        )

        # Skip adaptive sampling for next iteration
        self.optim_state["skip_active_sampling"] = self.options.get(
            "skip_active_sampling_after_warmup"
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

        # Maximum number of new function evaluations
        if self.function_logger.func_count >= self.options.get(
            "max_fun_evals"
        ):
            is_finished_flag = True
            termination_message = (
                "Inference terminated: reached maximum number "
                + "of function evaluations options.max_fun_evals."
            )

        # Maximum number of iterations
        iteration = self.optim_state.get("iter")
        if iteration + 1 >= self.options.get("max_iter"):
            is_finished_flag = True
            termination_message = (
                "Inference terminated: reached maximum number "
                + "of iterations options.max_iter."
            )

        # Quicker stability check for entropy switching
        if self.optim_state.get("entropy_switch"):
            tol_stable_iters = self.options.get("tol_stable_entropy_iters")
        else:
            tol_stable_iters = int(
                math.ceil(
                    self.options.get("tol_stable_count")
                    / self.options.get("fun_evals_per_iter")
                )
            )

        r_index, ELCBO_improvement = self._compute_reliability_index(
            tol_stable_iters
        )

        # Store reliability index
        self.iteration_history.record("r_index", r_index, iteration)
        self.iteration_history.record(
            "elcbo_impro", ELCBO_improvement, iteration
        )
        self.optim_state["R"] = r_index

        # Check stability termination condition
        stableflag = False
        if (
            iteration + 1 >= tol_stable_iters
            and r_index < 1
            and ELCBO_improvement < self.options.get("tol_improvement")
        ):
            # Count how many good iters in the recent past (excluding current)
            stable_count = np.sum(
                self.iteration_history.get("r_index")[
                    iteration - tol_stable_iters + 1 : iteration
                ]
                < 1
            )
            # Iteration is stable if almost all recent iterations are stable
            if (
                stable_count
                >= tol_stable_iters
                - np.floor(
                    tol_stable_iters
                    * self.options.get("tol_stable_excpt_frac")
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
                        + "solution stable for options.tol_stable_count "
                        + "fcn evaluations."
                    )

        # Store stability flag
        self.iteration_history.record("stable", stableflag, iteration)

        # Prevent early termination
        if self.function_logger.func_count < self.options.get(
            "min_fun_evals"
        ) or iteration < self.options.get("min_iter"):
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
            r_index = np.Inf
            ELCBO_improvement = np.NaN
            return r_index, ELCBO_improvement

        sn = np.sqrt(self.optim_state.get("sn2_hpd"))
        tol_sn = np.sqrt(sn / self.options.get("tol_sd")) * self.options.get(
            "tol_sd"
        )
        tol_sd = min(
            max(self.options.get("tol_sd"), tol_sn),
            self.options.get("tol_sd") * 10,
        )

        r_index_vec = np.full((3), np.NaN)
        r_index_vec[0] = (
            np.abs(
                self.iteration_history.get("elbo")[iteration_idx]
                - self.iteration_history.get("elbo")[iteration_idx - 1]
            )
            / tol_sd
        )
        r_index_vec[1] = (
            self.iteration_history.get("elbo_sd")[iteration_idx] / tol_sd
        )
        r_index_vec[2] = self.iteration_history.get("sKL")[
            iteration_idx
        ] / self.options.get("tol_skl")

        # Compute average ELCBO improvement per fcn eval in the past few iters
        idx0 = int(
            max(
                0,
                self.optim_state.get("iter")
                - math.ceil(0.5 * tol_stable_iters)
                + 1,
            )
        )
        # Remember than upper end of range is exclusive in Python, so +1 is
        # needed.
        xx = self.iteration_history.get("func_count")[idx0 : iteration_idx + 1]
        yy = (
            self.iteration_history.get("elbo")[idx0 : iteration_idx + 1]
            - self.options.get("elcbo_impro_weight")
            * self.iteration_history.get("elbo_sd")[idx0 : iteration_idx + 1]
        )
        # need to casts here to get things to run
        ELCBO_improvement = np.polyfit(
            list(map(float, xx)), list(map(float, yy)), 1
        )[0]
        return np.mean(r_index_vec), ELCBO_improvement

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
        ) < self.options.get("tol_gp_var_mcmc"):
            finished_flag = True

        return finished_flag

    def _recompute_lcb_max(self):
        """
        RECOMPUTE_LCB_MAX Recompute moving LCB maximum based on current GP.
        """
        # ToDo: Recompute_lcb_max needs to be implemented.
        return np.array([])

    # Finalizing:

    def final_boost(self, vp: VariationalPosterior, gp: gpr.GP):
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

        vp = copy.deepcopy(vp)
        changed_flag = False

        K_new = max(vp.K, self.options.get("min_final_components"))

        # Current entropy samples during variational optimization
        n_sent = self.options.eval("ns_ent", {"K": K_new})
        n_sent_fast = self.options.eval("ns_ent_fast", {"K": K_new})
        n_sent_fine = self.options.eval("ns_ent_fine", {"K": K_new})

        # Entropy samples for final boost
        if self.options.get("ns_ent_boost") == []:
            n_sent_boost = n_sent
        else:
            n_sent_boost = self.options.eval("ns_ent_boost", {"K": K_new})

        if self.options.get("ns_ent_fast_boost") == []:
            n_sent_fast_boost = n_sent_fast
        else:
            n_sent_fast_boost = self.options.eval(
                "ns_ent_fast_boost", {"K": K_new}
            )

        if self.options.get("ns_ent_fine_boost") == []:
            n_sent_fine_boost = n_sent_fine
        else:
            n_sent_fine_boost = self.options.eval(
                "ns_ent_fine_boost", {"K": K_new}
            )

        # Perform final boost?
        do_boost = (
            vp.K < self.options.get("min_final_components")
            or n_sent != n_sent_boost
            or n_sent_fine != n_sent_fine_boost
        )

        if do_boost:
            # Last variational optimization with large number of components
            n_fast_opts = math.ceil(self.options.eval("ns_elbo", {"K": K_new}))

            n_fast_opts = int(
                math.ceil(n_fast_opts * self.options.get("ns_elbo_incr"))
            )
            n_slow_opts = 1

            options = copy.deepcopy(self.options)
            # No pruning of components
            options.__setitem__("tol_weight", 0, force=True)

            # End warmup
            self.optim_state["warmup"] = False
            vp.optimize_mu = options.get("variable_means")
            vp.optimize_weights = options.get("variable_weights")

            options.__setitem__("ns_ent", n_sent_boost, force=True)
            options.__setitem__("ns_ent_fast", n_sent_fast_boost, force=True)
            options.__setitem__("ns_ent_fine", n_sent_fine_boost, force=True)
            options.__setitem__("max_iter_stochastic", np.Inf, force=True)
            self.optim_state["entropy_alpha"] = 0

            stable_flag = np.copy(vp.stats["stable"])
            vp, __, __ = optimize_vp(
                options,
                self.optim_state,
                vp,
                gp,
                n_fast_opts,
                n_slow_opts,
                K_new,
            )
            vp.stats["stable"] = stable_flag
            changed_flag = True

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
                order = self.iteration_history.get("r_index")[
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

    def _create_result_dict(
        self, idx_best: int, termination_message: str, success_flag: bool
    ):
        """
        Private method to create the result dict.
        """
        output = {}
        output["function"] = str(self.function_logger.fun)
        if np.all(np.isinf(self.optim_state["lb_tran"])) and np.all(
            np.isinf(self.optim_state["ub_tran"])
        ):
            output["problem_type"] = "unconstrained"
        else:
            output["problem_type"] = "bounded"

        output["iterations"] = self.optim_state["iter"]
        output["func_count"] = self.function_logger.func_count
        output["best_iter"] = idx_best
        output["train_set_size"] = self.iteration_history["n_eff"][idx_best]
        output["components"] = self.vp.K
        output["r_index"] = self.iteration_history["r_index"][idx_best]
        if self.iteration_history["stable"][idx_best]:
            output["convergence_status"] = "probable"
        else:
            output["convergence_status"] = "no"

        output["overhead"] = np.NaN
        output["rng_state"] = "rng"
        output["algorithm"] = "Variational Bayesian Monte Carlo"
        output["version"] = "0.1.0"
        output["message"] = termination_message

        output["elbo"] = self.vp.stats["elbo"]
        output["elbo_sd"] = self.vp.stats["elbo_sd"]
        output["success_flag"] = success_flag

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
                "sKL-iter[q]   K[q]  Convergence    Action"
            )
        else:
            if (
                self.optim_state["uncertainty_handling_level"] > 0
                and self.options.get("max_repeated_observations") > 0
            ):
                logger.info(
                    " Iteration   f-count (x-count)   Mean[ELBO]     Std[ELBO]"
                    "     sKL-iter[q]   K[q]  Convergence  Action"
                )
            else:
                logger.info(
                    " Iteration  f-count    Mean[ELBO]    Std[ELBO]    "
                    "sKL-iter[q]   K[q]  Convergence  Action"
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
                and self.options.get("max_repeated_observations") > 0
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
            log_file_mode = self.options.get("log_file_mode", "a")
        # On subsequent writes, switch to append mode:
        else:
            log_file_mode = "a"

        # Avoid duplicating a handler for the same log file
        # (remove duplicates, re-add below)
        for handler in logger.handlers:
            if handler.baseFilename == os.path.abspath(
                self.options.get("log_file_name")
            ):
                logger.removeHandler(handler)

        if self.options.get("log_file_name") and self.options.get(
            "log_file_level"
        ):
            file_handler = logging.FileHandler(
                filename=self.options["log_file_name"], mode=log_file_mode
            )

            # Set file logger level according to string or logging level:
            log_file_level = self.options.get("log_file_level", logging.INFO)
            if log_file_level == "off":
                file_handler.setLevel(logging.WARN)
            elif log_file_level == "iter":
                file_handler.setLevel(logging.INFO)
            elif log_file_level == "full":
                file_handler.setLevel(logging.DEBUG)
            elif log_file_level in [0, 10, 20, 30, 40, 50]:
                file_handler.setLevel(log_file_level)
            else:
                raise ValueError(
                    "Log file logging level is not a recognized"
                    + "string or logging level."
                )

            # Add a filter to ignore messages sent to logger.stream_only:
            def log_file_filter(record):
                return record.name != "VBMC.stream_only"

            file_handler.addFilter(log_file_filter)

            logger.addHandler(file_handler)

        return logger

    def __str__(self):
        """Construct a string summary."""

        gp = getattr(getattr(self, "vp", None), "gp", None)
        if gp is not None:
            gp_str = f"gpyreg.{gp}"
        else:
            gp_str = "None"

        return "VBMC:" + indent(
            f"""
dimension = {self.D},
x0: {summarize(self.x0)},
lower bounds: {summarize(self.lower_bounds)},
upper bounds: {summarize(self.upper_bounds)},
plausible lower bounds: {summarize(self.plausible_lower_bounds)},
plausible upper bounds: {summarize(self.plausible_upper_bounds)},
log-density = {getattr(self, "log_likelihood", self.log_joint)},
log-prior = {getattr(self, "log_prior", None)},
prior sampler = {getattr(self, "sample_prior", None)},
variational posterior = {str(getattr(self, "vp", None))},
Gaussian process = {gp_str},
user options = {str(self.options)}""",
            "    ",
        )

    def __repr__(self, arr_size_thresh=10, expand=False):
        """Construct a detailed string summary.

        Parameters
        ----------
        arr_size_thresh : float, optional
            If ``obj`` is an array whose product of dimensions is less than
            ``arr_size_thresh``, print the full array. Otherwise print only the
            shape. Default `10`.
        expand : bool, optional
            If ``expand`` is `False`, then describe any complex child
            attributes of the object by their name and memory location.
            Otherwise, recursively expand the child attributes into their own
            representations. Default `False`.

        Returns
        -------
        string : str
            The string representation of ``self``.
        """
        return full_repr(
            self,
            "VBMC",
            order=[
                "D",
                "x0",
                "lower_bounds",
                "upper_bounds",
                "plausible_lower_bounds",
                "plausible_upper_bounds",
                "log_joint",
                "log_prior",
                "sample_prior",
                "vp",
                "K",
                "gp",
                "parameter_transformer",
                "logger",
                "logging_action",
                "optim_state",
                "options",
            ],
            expand=expand,
            arr_size_thresh=arr_size_thresh,
            exclude=["random_state"],
        )

    def _short_repr(self):
        """Returns abbreviated string representation with memory location.

        Returns
        -------
        string : str
            The abbreviated string representation of the VBMC object.
        """
        return object.__repr__(self)
