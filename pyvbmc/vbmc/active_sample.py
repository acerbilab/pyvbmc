import copy
import logging
import math

import cma
import gpyreg as gpr
import numpy as np
import importlib

from pyvbmc.acquisition_functions import *
from pyvbmc.function_logger import FunctionLogger
from pyvbmc.stats import get_hpd
from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.vbmc.gaussian_process_train import reupdate_gp, train_gp
from pyvbmc.vbmc.iteration_history import IterationHistory
from pyvbmc.vbmc.variational_optimization import (
    _gplogjoint,
    _negelcbo,
    optimize_vp,
)
from pyvbmc.vbmc.active_importance_sampling import active_importance_sampling

from .options import Options


def active_sample(
    gp: gpr.GP,
    sample_count: int,
    optim_state: dict,
    function_logger: FunctionLogger,
    iteration_history: IterationHistory,
    vp: VariationalPosterior,
    options: Options,
):
    """
    Actively sample points iteratively based on acquisition function.

    Parameters
    ----------
    gp : GaussianProcess
        The GaussianProcess from the VBMC instance this function is called
        from.
    sample_count : int
        The number of samples.
    optim_state : dict
        The optim_state from the VBMC instance this function is called from.
    function_logger : FunctionLogger
        The FunctionLogger from the VBMC instance this function is called from.
    iteration_history : IterationHistory
        The IterationHistory from the VBMC instance this function is called
        from.
    vp : VariationalPosterior
        The VariationalPosterior from the VBMC instance this function is called
        from.
    options : Options
       Options from the VBMC instance this function is called from.

    Returns
    -------
    function_logger : FunctionLogger
        The updated FunctionLogger.
    optim_state : dict
        The updated optim_state.
    vp : VariationalPosterior
        The updated variation posterior.
    """
    # TODO: The timer is missing for now, we have to setup it throught pyvbmc.

    # Logging
    logger = logging.getLogger("ActiveSample")
    logger.setLevel(logging.INFO)
    if options.get("display") == "off":
        logger.setLevel(logging.WARN)
    elif options.get("display") == "iter":
        logger.setLevel(logging.INFO)
    elif options.get("display") == "full":
        logger.setLevel(logging.DEBUG)

    parameter_transformer = function_logger.parameter_transformer

    if gp is None:
        # No GP yet, just use provided points or sample from plausible box.

        # TODO: if the uncertainty_level is 2 the user needs to fill in
        # the cache for the noise S (not just for y) at each x0
        # this is also not implemented in MATLAB yet.

        x0 = optim_state["cache"]["x_orig"]
        provided_sample_count, D = x0.shape

        if provided_sample_count <= sample_count:
            Xs = np.copy(x0)
            ys = np.copy(optim_state["cache"]["y_orig"])

            if provided_sample_count < sample_count:
                pub = optim_state.get("pub")
                plb = optim_state.get("plb")

                if options.get("initdesign") == "plausible":
                    # Uniform random samples in the plausible box
                    # (in transformed space)
                    random_Xs = (
                        np.random.rand(sample_count - provided_sample_count, D)
                        * (pub - plb)
                        + plb
                    )

                elif options.get("initdesign") == "narrow":
                    start_Xs = parameter_transformer(Xs[0])
                    random_Xs = (
                        np.random.rand(sample_count - provided_sample_count, D)
                        - 0.5
                    ) * 0.1 * (pub - plb) + start_Xs
                    random_Xs = np.minimum((np.maximum(random_Xs, plb)), pub)

                else:
                    raise ValueError(
                        "Unknown initial design for VBMC. "
                        "The option 'initdesign' must be 'plausible' or "
                        "'narrow' but was {}.".format(
                            options.get("initdesign")
                        )
                    )

                # Convert back to original space
                random_Xs = parameter_transformer.inverse(random_Xs)
                Xs = np.append(Xs, random_Xs, axis=0)
                ys = np.append(
                    ys,
                    np.full(sample_count - provided_sample_count, np.NaN),
                    axis=0,
                )

            idx_remove = np.full(provided_sample_count, True)

        else:
            # In the MATLAB implementation there is a cluster algorithm being
            # used to pick the best points, but we decided not to implement that
            # yet and just pick the first sample_count points

            Xs = np.copy(x0[:sample_count])
            ys = np.copy(optim_state["cache"]["y_orig"][:sample_count])
            idx_remove = np.full(provided_sample_count, True)
            logger.info(
                "More than sample_count = %s initial points have been "
                "provided, using only the first %s points.",
                sample_count,
                sample_count,
            )

        # Remove points from starting cache
        optim_state["cache"]["x_orig"] = np.delete(
            optim_state["cache"]["x_orig"], idx_remove, 0
        )
        optim_state["cache"]["y_orig"] = np.delete(
            optim_state["cache"]["y_orig"], idx_remove, 0
        )

        Xs = parameter_transformer(Xs)

        for idx in range(sample_count):
            if np.isnan(ys[idx]):  # Function value is not available
                function_logger(Xs[idx])
            else:
                function_logger.add(Xs[idx], ys[idx])

    else:
        # active uncertainty sampling
        SearchAcqFcn = options["searchacqfcn"]

        ### (unused, TODO)
        # Use "hedge" strategy to propose an acquisition function?
        ###

        # Compute time cost (used by some acquisition functions)
        # if optim_state["iter"] > 1:
        #     deltaNeff = max(
        #         1,
        #         iteration_history["optim_state"][optim_state["iter"] - 1][
        #             "n_eff"
        #         ]
        #         - iteration_history["optim_state"][optim_state["iter"] - 2][
        #             "n_eff"
        #         ],
        #     )
        # else:
        #     deltaNeff = iteration_history["optim_state"][0]["n_eff"]

        # time_iter = iteration_history["timer"][optim_state["iter"] - 1]

        # gpTrain_vec = [None] * len(iteration_history["timer"])
        # for i, (_, v) in enumerate(iteration_history["timer"].items()):
        #     gpTrain_vec[i] = v["gpTrain"]

        ###
        # if options.ActiveVariationalSamples > 0 % Unused
        ###

        # Perform GP (and possibly variational) update after each active sample
        active_sample_full_update = (
            options["activesamplevpupdate"] or options["activesamplegpupdate"]
        ) and (
            (
                optim_state["iter"]
                - options["activesamplefullupdatepastwarmup"]
                <= optim_state["last_warmup"]
            )
            or iteration_history["rindex"][-1]
            > options["activesamplefullupdatethreshold"]
        )

        if active_sample_full_update and sample_count > 1:
            # Temporarily change options for local updates
            recompute_var_post_old = optim_state["recompute_var_post"]
            entropy_alpha_old = optim_state["entropy_alpha"]

            options_update = copy.deepcopy(options)
            options_update.__setitem__(
                "gptolopt", options["gptoloptactive"], force=True
            )
            options_update.__setitem__(
                "gptoloptmcmc", options["gptoloptmcmcactive"], force=True
            )
            options_update.__setitem__("tolweight", 0, force=True)
            options_update.__setitem__(
                "nsent", options["nsentactive"], force=True
            )
            options_update.__setitem__(
                "nsentfast", options["nsentfastactive"], force=True
            )
            options_update.__setitem__(
                "nsentfine", options["nsentfineactive"], force=True
            )

            hyp_dict = None
            vp0 = copy.deepcopy(vp)

        ## Active sampling loop (sequentially acquire Ns new points)
        for i in range(sample_count):
            optim_state["N"] = (
                function_logger.Xn + 1
            )  # Number of training inputs
            optim_state["Neff"] = sum(
                function_logger.nevals[function_logger.X_flag]
            )
            ###
            # if options.ActiveVariationalSamples > 0 % Unused
            ###
            ###
            # Nextra = evaloption_vbmc(options.SampleExtraVPMeans,vp.K);
            # if Nextra > 0   % Unused
            ###

            if not options["acqhedge"]:
                # If multiple acquisition functions are provided and not
                # following a "hedge" strategy, pick one at random
                idx_acq = np.random.randint(len(SearchAcqFcn))

            ## Pre-computations for acquisition functions

            # Evaluate noise at each training point
            Ns_gp = np.size(gp.posteriors)
            sn2new = np.zeros((gp.X.shape[0], Ns_gp))

            cov_N = gp.covariance.hyperparameter_count(gp.D)
            noise_N = gp.noise.hyperparameter_count()

            for s in range(Ns_gp):

                hyp_noise = gp.posteriors[s].hyp[cov_N : cov_N + noise_N]
                if hasattr(function_logger, "S"):
                    s2 = (
                        function_logger.S[function_logger.X_flag] ** 2
                    ) * function_logger.nevals[function_logger.X_flag]
                else:
                    s2 = None

                # Missing port: noiseshaping

                sn2new[:, s] = gp.noise.compute(
                    hyp_noise, gp.X, gp.y, s2
                ).reshape(-1,)

            gp.temporary_data["sn2_new"] = sn2new.mean(1)

            # Evaluate GP input length scale (use geometric mean)
            D = gp.D
            ln_ell = np.zeros((D, Ns_gp))
            for s in range(Ns_gp):
                ln_ell[:, s] = gp.posteriors[s].hyp[:D]
            optim_state["gp_length_scale"] = np.exp(ln_ell.mean(1))

            # Rescale GP training inputs by GP length scale
            gp.temporary_data["X_rescaled"] = (
                gp.X / optim_state["gp_length_scale"]
            )

            ### Missing port: line 185-205

            ## Start active search

            # Create fast search set from cache and randomly generated
            X_search, idx_cache = _get_search_points(
                options["nssearch"], optim_state, function_logger, vp, options
            )

            X_search = AbstractAcqFcn._real2int(
                X_search, parameter_transformer, optim_state["integervars"]
            )

            if type(SearchAcqFcn[idx_acq]) == str:
                acq_eval = string_to_acq(SearchAcqFcn[idx_acq])
            else:
                acq_eval = SearchAcqFcn[idx_acq]

            # Prepare for importance sampling based acquistion function
            if acq_eval.acq_info.get("importance_sampling"):
                optim_state[
                    "active_importance_sampling"
                ] = active_importance_sampling(vp, gp, acq_eval, options)

            # Re-evaluate variance of the log joint if requested
            if acq_eval.acq_info.get(
                "compute_varlogjoint"
            ):
                varF = _gplogjoint(vp, gp, 0, 0, 0, 1)[2]
                optim_state["varlogjoint_samples"] = varF

            # Evaluate acquisition function
            acq_fast = acq_eval(X_search, gp, vp, function_logger, optim_state)

            if options["searchcachefrac"] > 0:
                inds = np.argsort(acq_fast)
                optim_state["searchcache"] = X_search[inds]
                idx = inds[0]
            else:
                idx = np.argmin(acq_fast)

            X_acq = X_search[[idx]]
            idx_cache_acq = idx_cache[idx]

            # Remove selected points from search set
            X_search = np.delete(X_search, idx, 0)
            idx_cache = np.delete(idx_cache, idx, 0)

            acq_fun = lambda X: acq_eval(
                X, gp, vp, function_logger, optim_state
            )[0]
            # Additional search via optimization
            if options["searchoptimizer"] != "none":
                if gp.D == 1:
                    # Use Nelder-Mead method for 1D optimization
                    options.__setitem__(
                        "searchoptimizer", "Nelder-Mead", force=True
                    )

                fval_old = acq_fast[idx]
                x0 = X_acq[0, :]

                if (
                    np.isfinite(optim_state["lb_search"]).all()
                    and np.isfinite(optim_state["ub_search"]).all()
                ):
                    lb = np.minimum(x0, optim_state["lb_search"])
                    ub = np.maximum(x0, optim_state["ub_search"])
                else:
                    xrange = gp.X.max(0) - gp.X.min(0)
                    lb = np.minimum(gp.X, x0) - 0.1 * xrange
                    ub = np.maximum(gp.X, x0) + 0.1 * xrange

                if acq_eval.acq_info.get(
                    "log_flag"
                ):
                    tol_fun = 1e-2
                else:
                    tol_fun = max(1e-12, abs(fval_old * 1e-3))

                if options["searchoptimizer"] == "cmaes":

                    if options["searchcmaesvpinit"]:
                        _, Sigma = vp.moments(origflag=False, covflag=True)
                    else:
                        X_hpd = get_hpd(gp.X, gp.y, options["hpdfrac"])[0]
                        Sigma = np.cov(X_hpd, rowvar=False, bias=True)

                    insigma = np.sqrt(np.diag(Sigma))
                    cma_options = {
                        "verbose": -9,
                        "tolfun": tol_fun,
                        "maxfevals": options["searchmaxfunevals"],
                        "bounds": (lb.squeeze(), ub.squeeze()),
                        "seed": np.nan,
                    }

                    res = cma.fmin(
                        acq_fun,
                        x0,
                        np.max(insigma),
                        options=cma_options,
                        noise_handler=cma.NoiseHandler(np.size(x0)),
                    )

                    xsearch_optim, fval_optim = res[:2]
                elif options["searchoptimizer"] == "Nelder-Mead":
                    from scipy.optimize import minimize

                    res = minimize(
                        acq_fun, x0, method="Nelder-Mead", tol=tol_fun
                    )
                    xsearch_optim, fval_optim = res.x, res.fun
                else:
                    raise NotImplementedError("Not implemented yet")

                if fval_optim < fval_old:
                    X_acq[0, :] = AbstractAcqFcn._real2int(
                        xsearch_optim,
                        parameter_transformer,
                        optim_state["integervars"],
                    )
                    idx_cache_acq = np.nan

            # region
            ## Missing port
            # if (
            #     options["uncertaintyhandling"]
            #     and options["maxrepeatedobservations"] > 0
            # ):
            #     if (
            #         optim_state["repeatedobservationsstreak"]
            #         >= options["maxrepeatedobservations"]
            #     ):
            #         # Maximum number of consecutive repeated observations
            #         # (to prevent getting stuck in a wrong belief state)
            #         optim_state["repeatedobservationsstreak"] = 0
            #     else:
            #         from pyvbmc.vbmc.gaussian_process_train import (
            #             _get_training_data,
            #         )

            #         # Re-evaluate acquisition function on training set
            #         X_train = _get_training_data(function_logger)
            #         # Disable variance-based regularization first
            #         oldflag = optim_state["varianceregularizedacqfcn"]
            #         optim_state["varianceregularizedacqfcn"] = False
            #         # Use current cost of GP instead of future cost
            #         old_t_algoperfuneval = optim_state["t_algoperfuneval"]
            #         optim_state["t_algoperfuneval"] = t_base / deltaNeff
            #         acq_train = acq_eval(
            #             X_train, gp, vp, function_logger, optim_state
            #         )
            #         optim_state["VarianceRegularizedAcqFcn"] = oldflag
            #         optim_state["t_algoperfuneval"] = old_t_algoperfuneval

            #         idx_train = np.argmin(acq_train)
            #         acq_train = acq_train[idx_train]

            #         acq_now = acq_eval(
            #             X_acq[0], gp, vp, function_logger, optim_state
            #         )

            #         if acq_train < options["repeatedacqdiscount"]*acq_now:
            #             X_acq[0] = X_train[idx_train]
            #             optim_state["repeatedobservationsstreak"] += 1
            #         else:
            #             optim_state["repeatedobservationsstreak"] = 0
            # endregion

            # Missing port: line 356-361, unused?

            xnew = X_acq
            # See if chosen point comes from starting cache
            idx = idx_cache_acq
            if np.isnan(idx):
                y_orig = np.nan
            else:
                idx = int(idx)
                y_orig = optim_state["cache"]["y_orig"][idx]

            if np.isnan(y_orig):
                # Function value is not available, evaluate
                ynew, _, idx_new = function_logger(xnew)
            else:
                ynew, _, idx_new = function_logger.add(xnew, y_orig)
                # Remove point from starting cache
                optim_state["cache"]["x_orig"] = np.delete(
                    optim_state["cache"]["x_orig"], idx, 0
                )
                optim_state["cache"]["y_orig"] = np.delete(
                    optim_state["cache"]["y_orig"], idx, 0
                )

            if hasattr(function_logger, "S"):
                s2new = function_logger.S[idx_new] ** 2
            else:
                s2new = None

            ## Missing port: line 392-402 in matlab

            if i + 1 < sample_count:
                # If not the last sample, update GP and possibly other things
                # (no need perform updates after the last sample)
                if active_sample_full_update:
                    # If performing full updates with active sampling, the GP
                    # hyperparameters are updated after each acquisition

                    # Quick GP update
                    if hyp_dict is None:
                        hyp_dict = optim_state["hyp_dict"]

                    # Missing port: line 425-432 in matlab (unused)
                    gptmp = None
                    fESS, fESS_thresh = 0, 1
                    if fESS <= fESS_thresh:
                        if options["activesamplegpupdate"]:
                            (
                                gp,
                                __,
                                optim_state["sn2hpd"],
                                optim_state["hyp_dict"],
                            ) = train_gp(
                                hyp_dict,
                                optim_state,
                                function_logger,
                                iteration_history,
                                options_update,
                                optim_state["plb_orig"],
                                optim_state["pub_orig"],
                            )
                        else:
                            if gptmp is None:
                                gp = reupdate_gp(function_logger, gp)
                            else:
                                gp = gptmp

                        if options["activesamplevpupdate"]:
                            # Quick variational optimization

                            # Decide number of fast optimizations
                            N_fastopts = math.ceil(
                                options_update["nselboincr"]
                                * options_update["nselbo"](vp.K)
                            )
                            if options["updaterandomalpha"]:
                                optim_state["entropy_alpha"] = 1 - np.sqrt(
                                    np.random.rand()
                                )

                            vp, _, _ = optimize_vp(
                                options_update,
                                optim_state,
                                vp,
                                gp,
                                N_fastopts,
                                slow_opts_N=1,
                            )

                            if optim_state.get("vp_repo") is not None:
                                np.append(
                                    optim_state["vp_repo"], vp.get_parameters()
                                )
                            else:
                                optim_state["vp_repo"] = vp.get_parameters()
                    else:
                        gp = gptmp
                else:
                    # If NOT performing full updates with active sampling, only
                    # the GP posterior is updated (but not the hyperparameters)

                    # Perform simple rank-1 update if no noise and first sample
                    update1 = (
                        (s2new is None)
                        and function_logger.nevals[idx_new] == 1
                    ) and not options["noiseshaping"]
                    if update1:
                        ynew = np.array([[ynew]])  # (1,1)
                        gp.update(xnew, ynew, compute_posterior=True)
                        # gp.t(end+1) = tnew
                    else:
                        gp = reupdate_gp(function_logger, gp)

            # Check if active search bounds need to be expanded
            delta_search = 0.05 * (
                optim_state["ub_search"] - optim_state["lb_search"]
            )

            # ADD DIFFERENT CHECKS FOR INTEGER VARIABLES!
            idx = np.abs(xnew - optim_state["ub_search"]) < delta_search
            optim_state["lb_search"][idx] = np.maximum(
                optim_state["lb"][idx],
                optim_state["lb_search"][idx] - delta_search[idx],
            )
            idx = np.abs(xnew - optim_state["ub_search"]) < delta_search
            optim_state["ub_search"][idx] = np.minimum(
                optim_state["ub"][idx],
                optim_state["ub_search"][idx] + delta_search[idx],
            )

            # Hard lower/upper bounds on search (unused)
            prange = optim_state["pub"] - optim_state["plb"]
            LB_searchmin = np.maximum(
                optim_state["plb"] - 2 * prange * options["activesearchbound"],
                optim_state["lb"],
            )
            UB_searchmin = np.minimum(
                optim_state["pub"] + 2 * prange * options["activesearchbound"],
                optim_state["ub"],
            )

        if active_sample_full_update and sample_count > 1:
            # Reset optim_state
            optim_state["recompute_var_post"] = recompute_var_post_old
            optim_state["entropy_alpha"] = entropy_alpha_old
            optim_state["hyp_dict"] = hyp_dict

            # If variational posterior has changed, check if old variational
            # posterior is better than current
            theta0 = vp0.get_parameters()
            theta = vp.get_parameters()

            if (np.size(theta0) != np.size(theta)) or (np.any(theta0 != theta)):
                NSentFineK = math.ceil(
                    options["nsentfineactive"](vp0.K) / vp0.K
                )
                elbo0 = -_negelcbo(
                    theta0, gp, vp0, 0.0, NSentFineK, False, True
                )[0]

                if elbo0 > vp.stats["elbo"]:
                    vp = vp0

    return function_logger, optim_state, vp


def _get_search_points(
    number_of_points: int,
    optim_state: dict,
    function_logger: FunctionLogger,
    vp: VariationalPosterior,
    options: Options,
):
    """
    Get search points from starting cache or randomly generated.

    Parameters
    ----------
    number_of_points : int
        The number of points to return.
    optim_state : dict
        The optim_state from the VBMC instance this function is called from.
    function_logger : FunctionLogger
        The FunctionLogger from the VBMC instance this function is called from.
    vp : VariationalPosterior
        The VariationalPosterior from the VBMC instance this function is called
        from.
    options : Options
        Options from the VBMC instance this function is called from.

    Returns
    -------
    search_X : ndarray, shape (number_of_points, D)
        The obtained search points.
    idx_cache : ndarray, shape (number_of_points,)
        The indicies of the search points if coming from the cache.

    Raises
    ------
    ValueError
        When the options lead to more points sampled than requested, that means
        `search_X`.shape[0]` would be greater than `number_of_points``.
    """

    # Take some points from starting cache, if not empty
    x0 = np.copy(optim_state["cache"]["x_orig"])

    lb_search = optim_state.get("lb_search")
    ub_search = optim_state.get("ub_search")

    D = ub_search.shape[1]

    search_X = np.full((0, D), np.NaN)
    idx_cache = np.array([])
    parameter_transformer = function_logger.parameter_transformer

    if x0.size > 0:
        # Fraction of points from cache (if nonempty)
        N_cache = math.ceil(number_of_points * options.get("cachefrac"))

        # idx_cache contains min(n_cache, x0.shape[0]) random indicies
        idx_cache = np.random.permutation(x0.shape[0])[
            : min(N_cache, x0.shape[0])
        ]

        search_X = parameter_transformer(x0[idx_cache])

    # Randomly sample remaining points
    if x0.shape[0] < number_of_points:
        N_random_points = number_of_points - x0.shape[0]
        random_Xs = np.full((0, D), np.NaN)

        N_search_cache = round(
            options.get("searchcachefrac") * N_random_points
        )
        if N_search_cache > 0:  # Take points from search cache
            search_cache = optim_state.get("searchcache")
            random_Xs = np.append(
                random_Xs,
                search_cache[: min(len(search_cache), N_search_cache)],
                axis=0,
            )

        N_heavy = round(options.get("heavytailsearchfrac") * N_random_points)
        if N_heavy > 0:
            heavy_Xs, _ = vp.sample(
                N=N_heavy, origflag=False, balanceflag=True, df=3
            )
            random_Xs = np.append(random_Xs, heavy_Xs, axis=0)

        N_mvn = round(options.get("mvnsearchfrac") * N_random_points)
        if N_mvn > 0:
            mubar, sigmabar = vp.moments(origflag=False, covflag=True)
            mvn_Xs = np.random.multivariate_normal(
                np.ravel(mubar), sigmabar, size=N_mvn
            )
            random_Xs = np.append(random_Xs, mvn_Xs, axis=0)

        N_hpd = round(options.get("hpdsearchfrac") * N_random_points)
        if N_hpd > 0:
            hpd_min = options.get("hpdfrac") / 8
            hpd_max = options.get("hpdfrac")
            hpd_fracs = np.sort(
                np.concatenate(
                    (
                        np.random.uniform(size=4) * (hpd_max - hpd_min)
                        + hpd_min,
                        np.array([hpd_min, hpd_max]),
                    )
                )
            )
            N_hpd_vec = np.diff(
                np.round(np.linspace(0, N_hpd, len(hpd_fracs) + 1))
            )

            X = function_logger.X[function_logger.X_flag]
            y = function_logger.y[function_logger.X_flag]

            for idx in range(len(hpd_fracs)):
                if N_hpd_vec[idx] == 0:
                    continue

                X_hpd, _, _, _ = get_hpd(X, y, hpd_fracs[idx])

                if X_hpd.size == 0:
                    idx_max = np.argmax(y)
                    mubar = X[idx_max]
                    # rowvar is so that each column represents a variable
                    sigmabar = np.cov(X, rowvar=False)
                else:
                    mubar = np.mean(X_hpd, axis=0)
                    # normalize sigmabar by the number of observations
                    # rowvar is so that each column represents a variable
                    sigmabar = np.cov(X_hpd, bias=True, rowvar=False)

                # ensure sigmabar is of shape (D, D)
                if sigmabar.shape != (D, D):
                    sigmabar = np.ones((D, D)) * sigmabar

                hpd_Xs = np.random.multivariate_normal(
                    mubar, sigmabar, size=int(N_hpd_vec[idx])
                )
                random_Xs = np.append(random_Xs, hpd_Xs, axis=0)

        N_box = round(options.get("boxsearchfrac") * N_random_points)
        if N_box > 0:
            X = function_logger.X[function_logger.X_flag]
            X_diam = np.amax(X, axis=0) - np.amin(X, axis=0)
            plb = optim_state.get("plb")
            pub = optim_state.get("pub")

            if np.all(np.isfinite(lb_search)) and np.all(
                np.isfinite(ub_search)
            ):
                box_lb = lb_search
                box_ub = ub_search
            else:
                box_lb = plb - 3 * (pub - plb)
                box_ub = pub + 3 * (pub - plb)

            box_lb = np.maximum(np.amin(X, axis=0) - 0.5 * X_diam, box_lb)
            box_ub = np.minimum(np.amax(X, axis=0) + 0.5 * X_diam, box_ub)

            box_Xs = (
                np.random.standard_normal((N_box, D)) * (box_ub - box_lb)
                + box_lb
            )

            random_Xs = np.append(random_Xs, box_Xs, axis=0)

        # ensure that maximum N_random_points are sampled.
        if N_random_points < random_Xs.shape[0]:
            raise ValueError(
                "A maximum of {} points ".format(N_random_points),
                "should be randomly sampled but {} ".format(
                    random_Xs.shape[0]
                ),
                "were sampled. Please validate the provided options.",
            )

        # remaining samples
        N_vp = max(
            0,
            N_random_points - N_search_cache - N_heavy - N_mvn - N_box - N_hpd,
        )
        if N_vp > 0:
            vp_Xs, _ = vp.sample(N=N_vp, origflag=False, balanceflag=True)
            random_Xs = np.append(random_Xs, vp_Xs, axis=0)

        search_X = np.append(search_X, random_Xs, axis=0)
        idx_cache = np.append(idx_cache, np.full(N_random_points, np.nan))

    # Apply search bounds
    search_X = np.minimum((np.maximum(search_X, lb_search)), ub_search)
    return search_X, idx_cache
