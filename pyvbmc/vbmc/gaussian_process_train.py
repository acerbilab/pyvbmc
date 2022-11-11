import math

import gpyreg as gpr
import numpy as np

from pyvbmc.function_logger import FunctionLogger
from pyvbmc.stats import get_hpd

from .iteration_history import IterationHistory
from .options import Options


def train_gp(
    hyp_dict: dict,
    optim_state: dict,
    function_logger: FunctionLogger,
    iteration_history: IterationHistory,
    options: Options,
    plb_tran: np.ndarray,
    pub_tran: np.ndarray,
):
    """
    Train Gaussian process model.

    Parameters
    ==========
    hyp_dict : dict
        Hyperparameter summary statistics dictionary.
        If it does not contain the appropriate keys they will be added
        automatically.
    optim_state : dict
        Optimization state from the VBMC instance we are calling this from.
    function_logger : FunctionLogger
        Function logger from the VBMC instance which we are calling this from.
    iteration_history : IterationHistory
        Iteration history from the VBMC instance we are calling this from.
    options : Options
        Options from the VBMC instance we are calling this from.
    plb_tran : ndarray, shape (1, D)
        Transformed lower plausible bounds, used to set GP hyperparameters.
    pub_tran : ndarray, shape (1, D)
        Transformed upper plausible bounds, used to set GP hyperparameters.

    Returns
    =======
    gp : GP
        The trained GP.
    gp_s_N : int
        The number of samples for fitting.
    sn2_hpd : float
        An estimate of the GP noise variance at high posterior density.
    hyp_dict : dict
        The updated summary statistics.
    """

    # Initialize hyp_dict if empty.
    if "hyp" not in hyp_dict:
        hyp_dict["hyp"] = None
    if "warp" not in hyp_dict:
        hyp_dict["warp"] = None
    if "logp" not in hyp_dict:
        hyp_dict["logp"] = None
    if "full" not in hyp_dict:
        hyp_dict["full"] = None
    if "run_cov" not in hyp_dict:
        hyp_dict["run_cov"] = None

    # Get training dataset.
    x_train, y_train, s2_train, t_train = _get_training_data(function_logger)
    D = x_train.shape[1]

    # Heuristic fitness shaping (unused even in MATLAB)
    # if options.FitnessShaping
    #     [y_train,s2_train] = outputwarp_vbmc(X_train,y_train,s2_train,
    #                                           optimState,options);
    #  end

    # Pick the mean function
    mean_f = _meanfun_name_to_mean_function(optim_state["gp_mean_fun"])

    # Pick the covariance function.
    covariance_f = _cov_identifier_to_covariance_function(
        optim_state["gp_cov_fun"]
    )

    # Pick the noise function.
    const_add = optim_state["gp_noise_fun"][0] == 1
    user_add = optim_state["gp_noise_fun"][1] == 1
    user_scale = optim_state["gp_noise_fun"][1] == 2
    rlod_add = optim_state["gp_noise_fun"][2] == 1
    noise_f = gpr.noise_functions.GaussianNoise(
        constant_add=const_add,
        user_provided_add=user_add,
        scale_user_provided=user_scale,
        rectified_linear_output_dependent_add=rlod_add,
    )

    # Setup a GP.
    gp = gpr.GP(D=D, covariance=covariance_f, mean=mean_f, noise=noise_f)
    # Get number of samples and set priors and bounds.
    gp, hyp0, gp_s_N = _gp_hyp(
        optim_state, options, plb_tran, pub_tran, gp, x_train, y_train
    )
    # Initial GP hyperparameters.
    if hyp_dict["hyp"] is None:
        hyp_dict["hyp"] = hyp0.copy()

    # Get GP training options.
    gp_train = _get_gp_training_options(
        optim_state, iteration_history, options, hyp_dict, gp_s_N
    )

    # In some cases the model can change so be careful.
    if gp_train["widths"] is not None and np.size(
        gp_train["widths"]
    ) != np.size(hyp0):
        gp_train["widths"] = None

    # Build starting points
    # hyp0 = np.empty((0, np.size(hyp_dict["hyp"])))
    hyp0 = np.empty((0, hyp_dict["hyp"].T.shape[0]))
    if gp_train["init_N"] > 0 and optim_state["iter"] > 0:
        # Be very careful with off-by-one errors compared to MATLAB in the
        # range here.
        for i in range(
            math.ceil((np.size(iteration_history["gp"]) + 1) / 2) - 1,
            np.size(iteration_history["gp"]),
        ):
            hyp0 = np.concatenate(
                (
                    hyp0,
                    iteration_history["gp"][i].get_hyperparameters(
                        as_array=True
                    ),
                )
            )
        N0 = hyp0.shape[0]
        if N0 > gp_train["init_N"] / 2:
            hyp0 = hyp0[
                np.random.choice(
                    N0, math.ceil(gp_train["init_N"] / 2), replace=False
                ),
                :,
            ]
    hyp0 = np.concatenate((hyp0, np.atleast_2d(hyp_dict["hyp"])))
    hyp0 = np.unique(hyp0, axis=0)

    # In some cases the model can change so be careful.
    if hyp0.shape[1] != np.size(gp.hyper_priors["mu"]):
        hyp0 = None

    if (
        "hyp_vp" in hyp_dict
        and hyp_dict["hyp_vp"] is not None
        and gp_train["sampler"] == "npv"
    ):
        hyp0 = hyp_dict["hyp_vp"]

    # print(hyp0.shape)
    hyp_dict["hyp"], _, res = gp.fit(
        x_train, y_train, s2_train, hyp0=hyp0, options=gp_train
    )

    if res is not None:
        # Pre-thinning GP hyperparameters
        hyp_dict["full"] = res["samples"]
        hyp_dict["logp"] = res["log_priors"]

        # Missing port: currently not used since we do
        # not support samplers other than slice sampling.
        # if isfield(gpoutput,'hyp_vp')
        #     hypstruct.hyp_vp = gpoutput.hyp_vp;
        # end

        # if isfield(gpoutput,'stepsize')
        #     optimState.gp_mala_step_size = gpoutput.stepsize;
        #     gpoutput.stepsize
        # end

    # Update running average of GP hyperparameter covariance (coarse)
    if hyp_dict["full"] is not None and hyp_dict["full"].shape[1] > 1:
        hyp_cov = np.cov(hyp_dict["full"].T)
        if hyp_dict["run_cov"] is None or options["hyp_run_weight"] == 0:
            hyp_dict["run_cov"] = hyp_cov
        else:
            w = options["hyp_run_weight"] ** options["fun_evals_per_iter"]
            hyp_dict["run_cov"] = (1 - w) * hyp_cov + w * hyp_dict["run_cov"]
    else:
        hyp_dict["run_cov"] = None

    # Missing port: sample for GP for debug (not used)

    # Estimate of GP noise around the top high posterior density region
    # We don't modify optim_state to contain sn2_hpd here.
    sn2_hpd = _estimate_noise(gp)

    return gp, gp_s_N, sn2_hpd, hyp_dict


def _meanfun_name_to_mean_function(name: str):
    """
    Transforms a mean function name to an instance of that mean function.

    Parameters
    ==========
    name : str
        Name of the mean function.

    Returns
    =======
    mean_f : object
        An instance of the specified mean function.

    Raises
    ------
    ValueError
        Raised when the mean function name is unknown.
    """
    if name == "zero":
        mean_f = gpr.mean_functions.ZeroMean()
    elif name == "const":
        mean_f = gpr.mean_functions.ConstantMean()
    elif name == "negquad":
        mean_f = gpr.mean_functions.NegativeQuadratic()
    else:
        raise ValueError("Unknown mean function!")

    return mean_f


def _cov_identifier_to_covariance_function(identifier):
    """
    Transforms a covariance function identifer to an instance of the
    corresponding covariance function.

    Parameters
    ==========
    identifier : object
        Either an integer, or a list such as [3, 3] where the first
        number is the identifier and the further numbers are parameters
        of the covariance function.

    Returns
    =======
    cov_f : object
        An instance of the specified covariance function.

    Raises
    ------
    ValueError
        Raised when the covariance function identifier is unknown.
    """
    if identifier == 1:
        cov_f = gpr.covariance_functions.SquaredExponential()
    elif identifier == 3:
        cov_f = gpr.covariance_functions.Matern(5)
    elif isinstance(identifier, list) and identifier[0] == 3:
        cov_f = gpr.covariance_functions.Matern(identifier[1])
    else:
        raise ValueError("Unknown covariance function")

    return cov_f


def _gp_hyp(
    optim_state: dict,
    options: Options,
    plb_tran: np.ndarray,
    pub_tran: np.ndarray,
    gp: gpr.GP,
    X: np.ndarray,
    y: np.ndarray,
):
    """
    Define bounds, priors and samples for GP hyperparameters.

    Parameters
    ==========
    optim_state : dict
        Optimization state from the VBMC instance we are calling this from.
    options : Options
        Options from the VBMC instance we are calling this from.
    plb_tran : ndarray, shape (1, D)
        Transformed lower plausible bounds, used to set GP hyperparameters.
    pub_tran : ndarray, shape (1, D)
        Transformed upper plausible bounds, used to set GP hyperparameters.
    gp : GP
        Gaussian process for which we are making the bounds,
        priors and so on.
    X : ndarray, shape (N, D)
        Training inputs.
    y : ndarray, shape (N, 1)
        Training targets.

    Returns
    =======
    gp : GP
        The GP with updates priors, bounds and so on.
    hyp0 : ndarray, shape (hyp_N,)
        Initial guess for the hyperparameters.
    gp_s_N : int
        The number of samples for GP fitting.

    Raises
    ------
    TypeError
        Raised if the mean function is not supported by gpyreg.
    """

    # Get high posterior density dataset.
    hpd_X, hpd_y, _, _ = get_hpd(X, y, options["hpd_frac"])
    D = hpd_X.shape[1]
    # s2 = None

    ## Set GP hyperparameter defaults for VBMC.

    cov_bounds_info = gp.covariance.get_bounds_info(hpd_X, hpd_y)
    mean_bounds_info = gp.mean.get_bounds_info(hpd_X, hpd_y)
    noise_bounds_info = gp.noise.get_bounds_info(hpd_X, hpd_y)
    # Missing port: output warping hyperparameters not implemented
    cov_x0 = cov_bounds_info["x0"]
    mean_x0 = mean_bounds_info["x0"]

    noise_x0 = noise_bounds_info["x0"]
    min_noise = options["tol_gp_noise"]
    noise_mult = None
    if optim_state["uncertainty_handling_level"] == 0:
        if options["noise_size"] != []:
            noise_size = max(options["noise_size"], min_noise)
        else:
            noise_size = min_noise
        noise_std = 0.5
    elif optim_state["uncertainty_handling_level"] == 1:
        # This branch is not used and tested at the moment.
        if options["noise_size"] != []:
            noise_mult = max(options["noise_size"], min_noise)
            noise_mult_std = np.log(10) / 2
        else:
            noise_mult = 1
            noise_mult_std = np.log(10)
        noise_size = min_noise
        noise_std = np.log(10)
    elif optim_state["uncertainty_handling_level"] == 2:
        noise_size = min_noise
        noise_std = 0.5
    noise_x0[0] = np.log(noise_size)
    hyp0 = np.concatenate([cov_x0, noise_x0, mean_x0])

    # Missing port: output warping hyperparameters not implemented

    ## Change default bounds and set priors over hyperparameters.

    bounds = gp.get_bounds()
    if options["upper_gp_length_factor"] > 0:
        # Max GP input length scale
        bounds["covariance_log_lengthscale"] = (
            -np.inf,
            np.log(options["upper_gp_length_factor"] * (pub_tran - plb_tran)),
        )
    # Increase minimum noise.
    bounds["noise_log_scale"] = (np.log(min_noise), np.inf)

    # Missing port: we only implement the mean functions that gpyreg supports.
    if isinstance(gp.mean, gpr.mean_functions.ZeroMean):
        pass
    elif isinstance(gp.mean, gpr.mean_functions.ConstantMean):
        # Lower maximum constant mean
        bounds["mean_const"] = (-np.inf, np.min(hpd_y))
    elif isinstance(gp.mean, gpr.mean_functions.NegativeQuadratic):
        if options["gp_quadratic_mean_bound"]:
            delta_y = max(
                options["tol_sd"],
                min(D, np.max(hpd_y) - np.min(hpd_y)),
            )
            bounds["mean_const"] = (-np.inf, np.max(hpd_y) + delta_y)
    else:
        raise TypeError("The mean function is not supported by gpyreg.")

    # Set lower bounds for GP's outputscale and lengthscale
    if isinstance(gp.covariance, gpr.covariance_functions.SquaredExponential):
        bounds["covariance_log_outputscale"] = (
            cov_bounds_info["LB"][D],
            np.nan,
        )
        bounds["covariance_log_lengthscale"] = (
            cov_bounds_info["LB"][:D],
            np.nan,
        )
        # These bounds are wider since cov_bounds_info is based on the
        # high-posterior-density region as opposed to the full data
        # (a smaller set leads to smaller, hence wider, lower bounds)

    # Set priors over hyperparameters (might want to double-check this)
    priors = gp.get_priors()

    # Hyperprior over observation noise
    priors["noise_log_scale"] = (
        "student_t",
        (np.log(noise_size), noise_std, 3),
    )
    if noise_mult is not None:
        priors["noise_provided_log_multiplier"] = (
            "student_t",
            (np.log(noise_mult), noise_mult_std, 3),
        )

    # Missing port: hyperprior over mixture of quadratics mean function

    # Change bounds and hyperprior over output-dependent noise modulation
    # Note: currently this branch is not used.
    if optim_state["gp_noise_fun"][2] == 1:
        bounds["noise_rectified_log_multiplier"] = (
            [np.min(np.min(y), np.max(y) - 20 * D), -np.inf],
            [np.max(y) - 10 * D, np.inf],
        )

        # These two lines were commented out in MATLAB as well.
        # If uncommented add them to the stuff below these two lines
        # where we have np.nan
        # hypprior.mu(Ncov+2) = max(y_hpd) - 10*D;
        # hypprior.sigma(Ncov+2) = 1;

        # Only set the first of the two parameters here.
        priors["noise_rectified_log_multiplier"] = (
            "student_t",
            ([np.nan, np.log(0.01)], [np.nan, np.log(10)], [np.nan, 3]),
        )

    # Missing port: priors and bounds for output warping hyperparameters
    # (not used)

    # VBMC used to have an empirical Bayes prior on some GP hyperparameters,
    # such as input length scales, based on statistics of the GP training
    # inputs. However, this approach could lead to instabilities. From the
    # 2020 paper, we switched to a fixed prior based on the plausible bounds.
    priors["covariance_log_lengthscale"] = (
        "student_t",
        (
            np.log(options["gp_length_prior_mean"] * (pub_tran - plb_tran)),
            options["gp_length_prior_std"],
            3,
        ),
    )

    # Missing port: meanfun == 14 hyperprior case

    # Missing port: output warping priors

    ## Number of GP hyperparameter samples.

    stop_sampling = optim_state["stop_sampling"]

    if stop_sampling == 0:
        # Number of samples
        gp_s_N = options["ns_gp_max"] / np.sqrt(optim_state["N"])

        # Maximum sample cutoff
        if optim_state["warmup"]:
            gp_s_N = np.minimum(gp_s_N, options["ns_gp_max_warmup"])
        else:
            gp_s_N = np.minimum(gp_s_N, options["ns_gp_max_main"])

        # Stop sampling after reaching max number of training points
        if optim_state["N"] >= options["stable_gp_sampling"]:
            stop_sampling = optim_state["N"]

        # Stop sampling after reaching threshold of variational components
        if optim_state["vp_K"] >= options["stable_gp_vp_k"]:
            stop_sampling = optim_state["N"]

    if stop_sampling > 0:
        gp_s_N = options["stable_gp_samples"]

    gp.set_bounds(bounds)
    gp.set_priors(priors)

    return gp, hyp0, round(gp_s_N)


def _get_gp_training_options(
    optim_state: dict,
    iteration_history: IterationHistory,
    options: Options,
    hyp_dict: dict,
    gp_s_N: int,
):
    """
    Get options for training GP hyperparameters.

    Parameters
    ==========
    optim_state : dict
        Optimization state from the VBMC instance we are calling this from.
    iteration_history : IterationHistory
        Iteration history from the VBMC instance we are calling this from.
    options : Options
        Options from the VBMC instance we are calling this from.
    hyp_dict : dict
        Hyperparameter summary statistic dictionary.
    gp_s_N : int
        Number of samples for the GP fitting.

    Returns
    =======
    gp_train : dic
        A dictionary of GP training options.

    Raises
    ------
    ValueError
        Raised if the MCMC sampler for GP hyperparameters is unknown.

    """

    iteration = optim_state["iter"]
    if iteration > 0:
        r_index = iteration_history["r_index"][iteration - 1]
    else:
        r_index = np.inf

    gp_train = {}
    gp_train["thin"] = options["gp_sample_thin"]  # MCMC thinning
    gp_train["init_method"] = options["gp_train_init_method"]
    gp_train["tol_opt"] = options["gp_tol_opt"]
    gp_train["tol_opt_mcmc"] = options["gp_tol_opt_mcmc"]
    gp_train["widths"] = None

    # Get hyperparameter posterior covariance from previous iterations
    hyp_cov = _get_hyp_cov(optim_state, iteration_history, options, hyp_dict)

    # Setup MCMC sampler
    if options["gp_hyp_sampler"] == "slicesample":
        gp_train["sampler"] = "slicesample"
        if options["gp_sample_widths"] > 0 and hyp_cov is not None:
            width_mult = np.maximum(options["gp_sample_widths"], r_index)
            hyp_widths = np.sqrt(np.diag(hyp_cov).T)
            gp_train["widths"] = np.maximum(hyp_widths, 1e-3) * width_mult

    elif options["gp_hyp_sampler"] == "npv":
        gp_train["sampler"] = "npv"

    elif options["gp_hyp_sampler"] == "mala":
        gp_train["sampler"] = "mala"
        if hyp_cov is not None:
            gp_train["widths"] = np.sqrt(np.diag(hyp_cov).T)
        if "gp_mala_step_size" in optim_state:
            gp_train["step_size"] = optim_state["gp_mala_step_size"]

    elif options["gp_hyp_sampler"] == "slicelite":
        gp_train["sampler"] = "slicelite"
        if options["gp_sample_widths"] > 0 and hyp_cov is not None:
            width_mult = np.maximum(options["gp_sample_widths"], r_index)
            hyp_widths = np.sqrt(np.diag(hyp_cov).T)
            gp_train["widths"] = np.maximum(hyp_widths, 1e-3) * width_mult

    elif options["gp_hyp_sampler"] == "splitsample":
        gp_train["sampler"] = "splitsample"
        if options["gp_sample_widths"] > 0 and hyp_cov is not None:
            width_mult = np.maximum(options["gp_sample_widths"], r_index)
            hyp_widths = np.sqrt(np.diag(hyp_cov).T)
            gp_train["widths"] = np.maximum(hyp_widths, 1e-3) * width_mult

    elif options["gp_hyp_sampler"] == "covsample":
        if options["gp_sample_widths"] > 0 and hyp_cov is not None:
            width_mult = np.maximum(options["gp_sample_widths"], r_index)
            if np.all(np.isfinite(width_mult)) and np.all(
                r_index < options["cov_sample_thresh"]
            ):
                hyp_n = hyp_cov.shape[0]
                gp_train["widths"] = (
                    hyp_cov + 1e-6 * np.eye(hyp_n)
                ) * width_mult**2
                gp_train["sampler"] = "covsample"
                gp_train["thin"] *= math.ceil(np.sqrt(hyp_n))
            else:
                hyp_widths = np.sqrt(np.diag(hyp_cov).T)
                gp_train["widths"] = np.maximum(hyp_widths, 1e-3) * width_mult
                gp_train["sampler"] = "slicesample"
        else:
            gp_train["sampler"] = "covsample"

    elif options["gp_hyp_sampler"] == "laplace":
        if optim_state["n_eff"] < 30:
            gp_train["sampler"] = "slicesample"
            if options["gp_sample_widths"] > 0 and hyp_cov is not None:
                width_mult = np.maximum(options["gp_sample_widths"], r_index)
                hyp_widths = np.sqrt(np.diag(hyp_cov).T)
                gp_train["widths"] = np.maximum(hyp_widths, 1e-3) * width_mult
        else:
            gp_train["sampler"] = "laplace"

    else:
        raise ValueError("Unknown MCMC sampler for GP hyperparameters")

    # N-dependent initial training points.
    a = -(options["gp_train_n_init"] - options["gp_train_n_init_final"])
    b = -3 * a
    c = 3 * a
    d = options["gp_train_n_init"]
    x = (optim_state["n_eff"] - options["fun_eval_start"]) / (
        min(options["max_fun_evals"], 1e3) - options["fun_eval_start"]
    )
    f = lambda x_: a * x_**3 + b * x**2 + c * x + d
    init_N = max(round(f(x)), 9)

    # Set other hyperparameter fitting parameters
    if optim_state["recompute_var_post"]:
        gp_train["burn"] = gp_train["thin"] * gp_s_N
        gp_train["init_N"] = init_N
        if gp_s_N > 0:
            gp_train["opts_N"] = 1
        else:
            gp_train["opts_N"] = 2
    else:
        gp_train["burn"] = gp_train["thin"] * 3
        if (
            iteration > 1
            and iteration_history["r_index"][iteration - 1]
            < options["gp_retrain_threshold"]
        ):
            gp_train["init_N"] = 0
            if options["gp_hyp_sampler"] == "slicelite":
                # TODO: gp_retrain_threshold is by default 1, so we get
                #       division by zero. what should the default be?
                gp_train["burn"] = (
                    max(
                        1,
                        math.ceil(
                            gp_train["thin"]
                            * np.log(
                                iteration_history["r_index"][iteration - 1]
                                / np.log(options["gp_retrain_threshold"])
                            )
                        ),
                    )
                    * gp_s_N
                )
                gp_train["thin"] = 1
            if gp_s_N > 0:
                gp_train["opts_N"] = 0
            else:
                gp_train["opts_N"] = 1
        else:
            gp_train["init_N"] = init_N
            if gp_s_N > 0:
                gp_train["opts_N"] = 1
            else:
                gp_train["opts_N"] = 2

    gp_train["n_samples"] = round(gp_s_N)
    gp_train["burn"] = round(gp_train["burn"])

    return gp_train


def _get_hyp_cov(
    optim_state: dict,
    iteration_history: IterationHistory,
    options: Options,
    hyp_dict: dict,
):
    """
    Get hyperparameter posterior covariance.

    Parameters
    ==========
    optim_state : dict
        Optimization state from the VBMC instance we are calling this from.
    iteration_history : IterationHistory
        Iteration history from the VBMC instance we are calling this from.
    options : Options
        Options from the VBMC instance we are calling this from.
    hyp_dict : dict
        Hyperparameter summary statistic dictionary.

    Returns
    =======
    hyp_cov : ndarray, optional
        The hyperparameter posterior covariance if it can be computed.
    """

    if optim_state["iter"] > 0:
        if options["weighted_hyp_cov"]:
            w_list = []
            hyp_list = []
            w = 1
            for i in range(0, optim_state["iter"]):
                if i > 0:
                    # Be careful with off-by-ones compared to MATLAB here
                    diff_mult = max(
                        1,
                        np.log(
                            iteration_history["sKL"][optim_state["iter"] - i]
                            / options["tol_skl"]
                            * options["fun_evals_per_iter"]
                        ),
                    )
                    w *= options["hyp_run_weight"] ** (
                        options["fun_evals_per_iter"] * diff_mult
                    )
                # Check if weight is getting too small.
                if w < options["tol_cov_weight"]:
                    break

                hyp = iteration_history["gp_hyp_full"][
                    optim_state["iter"] - 1 - i
                ]
                hyp_n = hyp.shape[1]
                if len(hyp_list) == 0 or np.shape(hyp_list)[2] == hyp.shape[0]:
                    hyp_list.append(hyp.T)
                    w_list.append(w * np.ones((hyp_n, 1)) / hyp_n)

            w_list = np.concatenate(w_list)
            hyp_list = np.concatenate(hyp_list)

            # Normalize weights
            w_list /= np.sum(w_list, axis=0)
            # Weighted mean
            mu_star = np.sum(hyp_list * w_list, axis=0)

            # Weighted covariance matrix
            hyp_n = np.shape(hyp_list)[1]
            hyp_cov = np.zeros((hyp_n, hyp_n))
            for j in range(0, np.shape(hyp_list)[0]):
                hyp_cov += np.dot(
                    w_list[j],
                    np.dot((hyp_list[j] - mu_star).T, hyp_list[j] - mu_star),
                )

            hyp_cov /= 1 - np.sum(w_list**2)

            return hyp_cov

        return hyp_dict["run_cov"]

    return None


def _get_training_data(function_logger: FunctionLogger):
    """
    Get training data for building GP surrogate.

    Parameters
    ==========
    function_logger : FunctionLogger
        Function logger from the VBMC instance which we are calling this from.

    Returns
    =======
    x_train, ndarray
        Training inputs.
    y_train, ndarray
        Training targets.
    s2_train, ndarray, optional
        Training data noise variance, if noise is used.
    t_train, ndarray
        Array of the times it took to evaluate the function on the training
        data.
    """

    x_train = function_logger.X[function_logger.X_flag, :]
    y_train = function_logger.y[function_logger.X_flag]
    if function_logger.noise_flag:
        s2_train = function_logger.S[function_logger.X_flag] ** 2
    else:
        s2_train = None

    # Missing port: noise_shaping

    t_train = function_logger.fun_eval_time[function_logger.X_flag]

    return x_train, y_train, s2_train, t_train


def _estimate_noise(gp: gpr.GP):
    """Estimate GP observation noise at high posterior density.

    Parameters
    ==========
    gp : GP
        The GP for which to perform the estimate.

    Returns
    =======
    est : float
        The estimate of observation noise.
    """

    hpd_top = 0.2
    N, _ = gp.X.shape

    # Subsample high posterior density dataset
    # Sort by descending order, not ascending.
    order = np.argsort(gp.y, axis=None)[::-1]
    hpd_N = math.ceil(hpd_top * N)
    hpd_X = gp.X[order[0:hpd_N]]
    hpd_y = gp.y[order[0:hpd_N]]

    if gp.s2 is not None:
        hpd_s2 = gp.s2[order[0:hpd_N]]
    else:
        hpd_s2 = None

    cov_N = gp.covariance.hyperparameter_count(gp.D)
    noise_N = gp.noise.hyperparameter_count()
    s_N = np.size(gp.posteriors)

    sn2 = np.zeros((hpd_X.shape[0], s_N))

    for s in range(0, s_N):
        hyp = gp.posteriors[s].hyp[cov_N : cov_N + noise_N]
        sn2[:, s : s + 1] = gp.noise.compute(hyp, hpd_X, hpd_y, hpd_s2)

    return np.median(np.mean(sn2, axis=1))


def reupdate_gp(function_logger: FunctionLogger, gp: gpr.GP):
    """
    Quick posterior reupdate of Gaussian process.

    Parameters
    ==========
    gp : GP
        The GP to update.
    function_logger : FunctionLogger
        Function logger from the VBMC instance which we are calling this from.
    Returns
    =======
    gp : GP
        The updated Gaussian process.
    """

    x_train, y_train, s2_train, t_train = _get_training_data(function_logger)
    gp.X = x_train
    gp.y = y_train
    gp.s2 = s2_train
    # Missing port: gp.t = t_train
    gp.update(compute_posterior=True)

    # Missing port: intmean part

    return gp
