import copy
import logging
import math

import cma
import gpyreg as gpr
import numpy as np

from pyvbmc.acquisition_functions import AbstractAcqFcn
from pyvbmc.acquisition_functions.utilities import string_to_acq
from pyvbmc.function_logger import FunctionLogger
from pyvbmc.stats import get_hpd
from pyvbmc.stats._rounding import round_half_away_from_zero
from pyvbmc.timer import main_timer as timer
from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.vbmc.active_importance_sampling import active_importance_sampling
from pyvbmc.vbmc.gaussian_process_train import reupdate_gp, train_gp
from pyvbmc.vbmc.iteration_history import IterationHistory
from pyvbmc.vbmc.variational_optimization import (
    _gp_log_joint,
    _neg_elcbo,
    optimize_vp,
)

from .options import Options

# Private, process-local seam used by bounded developer experiments.  The
# default path does not install a callback and retains the production search.
_selection_policy_callback = None


def _refresh_training_counts(optim_state, function_logger):
    """Write the counts of the training set into ``optim_state``.

    ``N`` is the number of training inputs and ``n_eff`` the number of
    evaluations over the live rows, where a repeated observation pooled
    into its row counts once more. The GP training reads both.
    """
    optim_state["N"] = function_logger.Xn + 1
    optim_state["n_eff"] = np.sum(
        function_logger.n_evals[function_logger.X_flag]
    )


def _log_search_failure(logger, exc):
    """Report a local acquisition search that raised."""
    logger.warning(
        "Active search failed (%s: %s); using the best candidate of the "
        "search set.",
        type(exc).__name__,
        exc,
    )


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

    Notes
    -----
    Random draws (initial design, acquisition choice, search points, CMA-ES)
    use ``vp.rng``.

    Returns
    -------
    function_logger : FunctionLogger
        The updated FunctionLogger.
    optim_state : dict
        The updated optim_state.
    vp : VariationalPosterior
        The updated VP.
    gp : gpyreg.GaussianProcess
        The updated GP.
    """
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
    rng = vp.rng

    if gp is None:
        # No GP yet, just use provided points or sample from plausible box.

        # TODO: if the uncertainty_level is 2 the user needs to fill in
        # the cache for the noise S (not just for y) at each x0
        # this is also not implemented in MATLAB yet.

        x0 = optim_state["cache"]["x_orig"]
        skip_logger_cache = optim_state["cache"].get("skip_logger")
        if skip_logger_cache is None or skip_logger_cache.shape != (
            x0.shape[0],
        ):
            skip_logger_cache = np.zeros(x0.shape[0], dtype=bool)
            optim_state["cache"].pop("skip_logger", None)
        provided_sample_count, D = x0.shape

        if provided_sample_count <= sample_count:
            Xs = np.copy(x0)
            ys = np.copy(optim_state["cache"]["y_orig"])
            skip_logger = np.copy(skip_logger_cache)

            if provided_sample_count < sample_count:
                pub_tran = optim_state.get("pub_tran")
                plb_tran = optim_state.get("plb_tran")

                if options.get("init_design") == "plausible":
                    # Uniform random samples in the plausible box
                    # (in transformed space)
                    random_Xs = (
                        rng.random((sample_count - provided_sample_count, D))
                        * (pub_tran - plb_tran)
                        + plb_tran
                    )

                elif options.get("init_design") == "narrow":
                    start_Xs = parameter_transformer(Xs[0])
                    random_Xs = (
                        rng.random((sample_count - provided_sample_count, D))
                        - 0.5
                    ) * 0.1 * (pub_tran - plb_tran) + start_Xs
                    random_Xs = np.minimum(
                        (np.maximum(random_Xs, plb_tran)), pub_tran
                    )

                else:
                    raise ValueError(
                        "Unknown initial design for VBMC. "
                        "The option 'init_design' must be 'plausible' or "
                        "'narrow' but was {}.".format(
                            options.get("init_design")
                        )
                    )

                # Convert back to original space
                random_Xs = parameter_transformer.inverse(random_Xs)
                Xs = np.append(Xs, random_Xs, axis=0)
                ys = np.append(
                    ys,
                    np.full(sample_count - provided_sample_count, np.nan),
                    axis=0,
                )
                skip_logger = np.append(
                    skip_logger,
                    np.full(
                        sample_count - provided_sample_count, False, dtype=bool
                    ),
                )

            idx_remove = np.full(provided_sample_count, True)

        else:
            # In the MATLAB implementation there is a cluster algorithm being
            # used to pick the best points, but we decided not to implement that
            # yet and just pick the first sample_count points

            Xs = np.copy(x0[:sample_count])
            ys = np.copy(optim_state["cache"]["y_orig"][:sample_count])
            skip_logger = np.copy(skip_logger_cache[:sample_count])
            # Only the points the initial design consumes leave the cache;
            # the rest stay there with their values, as candidates of the
            # search sieve that are acquired without a target call.
            idx_remove = np.full(provided_sample_count, False)
            idx_remove[:sample_count] = True
            logger.info(
                "More than sample_count=%s initial points have been "
                "provided, using the first %s for the initial design and "
                "keeping the remaining %s in the cache.",
                sample_count,
                sample_count,
                provided_sample_count - sample_count,
            )

        # Remove points from starting cache
        optim_state["cache"]["x_orig"] = np.delete(
            optim_state["cache"]["x_orig"], np.where(idx_remove), 0
        )
        optim_state["cache"]["y_orig"] = np.delete(
            optim_state["cache"]["y_orig"], np.where(idx_remove), 0
        )
        if "skip_logger" in optim_state["cache"]:
            optim_state["cache"]["skip_logger"] = np.delete(
                optim_state["cache"]["skip_logger"], np.where(idx_remove), 0
            )

        Xs = parameter_transformer(Xs)

        if getattr(function_logger, "vectorized_target", False):
            if np.any(~skip_logger):
                function_logger.batch_call(Xs[~skip_logger], ys[~skip_logger])
        else:
            for idx in range(sample_count):
                if skip_logger[idx]:
                    continue
                if np.isnan(ys[idx]):  # Function value is not available
                    function_logger(Xs[idx])
                else:
                    function_logger.add(Xs[idx], ys[idx])
        _refresh_training_counts(optim_state, function_logger)

    else:
        # active uncertainty sampling
        SearchAcqFcn = options["search_acq_fcn"]

        ### (unused, TODO)
        # Use "hedge" strategy to propose an acquisition function?
        ###

        # Compute time cost (used by some acquisition functions)
        # if optim_state["iter"] > 1:
        #     deltaN_eff = max(
        #         1,
        #         iteration_history["optim_state"][optim_state["iter"] - 1][
        #             "n_eff"
        #         ]
        #         - iteration_history["optim_state"][optim_state["iter"] - 2][
        #             "n_eff"
        #         ],
        #     )
        # else:
        #     deltaN_eff = iteration_history["optim_state"][0]["n_eff"]

        # time_iter = iteration_history["timer"][optim_state["iter"] - 1]

        # gpTrain_vec = [None] * len(iteration_history["timer"])
        # for i, (_, v) in enumerate(iteration_history["timer"].items()):
        #     gpTrain_vec[i] = v["gpTrain"]

        ###
        # if options.ActiveVariationalSamples > 0 % Unused
        ###

        # Perform GP (and possibly variational) update after each active sample
        active_sample_full_update = (
            options["active_sample_vp_update"]
            or options["active_sample_gp_update"]
        ) and (
            (
                optim_state["iter"]
                - options["active_sample_full_update_past_warmup"]
                <= optim_state["last_warmup"]
            )
            or iteration_history["r_index"][-1]
            > options["active_sample_full_update_threshold"]
        )

        if active_sample_full_update and sample_count > 1:
            # Temporarily change options for local updates
            recompute_var_post_old = optim_state["recompute_var_post"]
            entropy_alpha_old = optim_state["entropy_alpha"]

            options_update = copy.deepcopy(options)
            options_update.__setitem__(
                "gp_tol_opt", options["gp_tol_opt_active"], force=True
            )
            options_update.__setitem__(
                "gp_tol_opt_mcmc",
                options["gp_tol_opt_mcmc_active"],
                force=True,
            )
            options_update.__setitem__("tol_weight", 0, force=True)
            # A cap on the number of hyperparameter samples of the in-loop
            # refits: `ns_gp_max` is a coefficient (samples = ns_gp_max /
            # sqrt(N)), `ns_gp_max_warmup` and `ns_gp_max_main` the caps on
            # the count, so the option joins the caps. Saved runs predate
            # the option.
            ns_gp_cap = options.get("ns_gp_max_active", np.inf)
            for key in ("ns_gp_max_warmup", "ns_gp_max_main"):
                options_update.__setitem__(
                    key, min(options[key], ns_gp_cap), force=True
                )
            options_update.__setitem__(
                "ns_ent", options["ns_ent_active"], force=True
            )
            options_update.__setitem__(
                "ns_ent_fast", options["ns_ent_fast_active"], force=True
            )
            options_update.__setitem__(
                "ns_ent_fine", options["ns_ent_fine_active"], force=True
            )

            hyp_dict = None
            vp0 = copy.deepcopy(vp)

        ## Active sampling loop (sequentially acquire Ns new points)
        for i in range(sample_count):
            _refresh_training_counts(optim_state, function_logger)
            ###
            # if options.ActiveVariationalSamples > 0 % Unused
            ###
            ###
            # Nextra = evaloption_vbmc(options.SampleExtraVPMeans,vp.K);
            # if Nextra > 0   % Unused
            ###

            if not options["acq_hedge"]:
                # If multiple acquisition functions are provided and not
                # following a "hedge" strategy, pick one at random. The
                # hedge is not ported, and the option is refused at
                # construction, so this is the branch every run takes.
                idx_acq = rng.integers(len(SearchAcqFcn))

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
                    ) * function_logger.n_evals[function_logger.X_flag]
                else:
                    s2 = None

                # Missing port: noise_shaping

                sn2new[:, s] = gp.noise.compute(
                    hyp_noise, gp.X, gp.y, s2
                ).reshape(
                    -1,
                )

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
            selection_policy = _selection_policy_callback
            if selection_policy is not None:
                selection_policy.start(
                    gp=gp,
                    vp=vp,
                    function_logger=function_logger,
                    optim_state=optim_state,
                    options=options,
                )
            cache_rows = {}
            X_search, idx_cache = _get_search_points(
                options["ns_search"],
                optim_state,
                function_logger,
                vp,
                options,
                cache_rows=cache_rows,
            )

            X_search = AbstractAcqFcn._real2int(
                X_search, parameter_transformer, optim_state["integer_vars"]
            )

            # Repeated observations: with observation noise the training
            # inputs join the search set. An exact repeat is pooled into its
            # row by the function logger (precision-weighted), so it sharpens
            # the GP without adding a training point; a cap on consecutive
            # repeats keeps a wrong belief from locking the search onto one
            # input. The chosen repeat skips the local optimizer so that it
            # stays an exact repeat.
            n_train_cand = 0
            X_train_cand = None
            repeat_cap = options["max_repeated_observations"]
            if (
                repeat_cap > 0
                and function_logger.noise_flag
                and optim_state.get("repeated_observations_streak", 0)
                < repeat_cap
            ):
                X_train_cand = function_logger.X[function_logger.X_flag]
                n_train_cand = X_train_cand.shape[0]
                X_search = np.vstack([X_train_cand, X_search])
                idx_cache = np.append(np.full(n_train_cand, np.nan), idx_cache)

            if type(SearchAcqFcn[idx_acq]) == str:
                acq_eval = string_to_acq(SearchAcqFcn[idx_acq])
            else:
                acq_eval = SearchAcqFcn[idx_acq]

            # Prepare for importance sampling based acquistion function
            if acq_eval.acq_info.get("importance_sampling"):
                optim_state[
                    "active_importance_sampling"
                ] = active_importance_sampling(vp, gp, acq_eval, options)

            # Re-evaluate variance of the log joint if requested: per
            # hyperparameter sample, together with the covariance of the
            # components' integrals.
            if acq_eval.acq_info.get("compute_var_log_joint"):
                out = _gp_log_joint(vp, gp, 0, 0, 0, 1, separate_K=True)
                optim_state["var_log_joint_samples"] = out[2]
                optim_state["cov_log_joint_components"] = out[6]

            # Evaluate acquisition function
            acq_fast = acq_eval(X_search, gp, vp, function_logger, optim_state)

            if options["search_cache_frac"] > 0:
                inds = np.argsort(acq_fast)
                # The training inputs at the head of the search set are
                # candidates for a repeated observation of this step
                # alone. They are kept out of the cache: a training input
                # offered again at a later step comes back as an ordinary
                # candidate, without the repeat flag that caps consecutive
                # repeats and keeps the point an exact repeat.
                optim_state["search_cache"] = X_search[
                    inds[inds >= n_train_cand]
                ]
                idx = inds[0]
            else:
                idx = np.argmin(acq_fast)

            policy_selection = None
            if selection_policy is not None:
                policy_selection = selection_policy.select(
                    candidates=X_search,
                    coarse_scores=acq_fast,
                    cache_indices=idx_cache,
                    n_train=n_train_cand,
                    gp=gp,
                    vp=vp,
                    function_logger=function_logger,
                    optim_state=optim_state,
                    options=options,
                )

            X_acq = X_search[[idx]]
            idx_cache_acq = idx_cache[idx]
            repeat_flag = idx < n_train_cand
            if repeat_flag:
                # The stored row itself: the acquisition snaps its input to
                # the integer grid in place, and a row of the initial design
                # was never snapped, so the snapped copy in `X_search` could
                # be a near-duplicate rather than the exact repeat the
                # logger pools.
                X_acq = X_train_cand[[idx]].copy()

            if policy_selection is not None:
                X_acq, idx_cache_acq, repeat_flag = policy_selection
                X_acq = np.asarray(X_acq, dtype=np.float64).reshape(1, gp.D)

            # Remove selected points from search set. Nothing reads either
            # array again: the next step builds both afresh, and the search
            # cache above, where one is kept, was written before the
            # deletion, so it still holds the acquired point unless that
            # point is a training input, which the cache leaves out. The
            # two lines stand where `private/activesample_vbmc.m:242` has
            # them.
            X_search = np.delete(X_search, idx, 0)
            idx_cache = np.delete(idx_cache, idx, 0)

            def acq_fun(X):
                """Acquisition for the search optimizers.

                One point (a 1-D array: the bounded scalar search, or
                CMA-ES's rejection path) returns a float; a list of points
                (one CMA-ES generation) is evaluated in a single batched
                call and returns a list.
                With integer variables the acquisition snaps its input to
                the integer grid in place (`AbstractAcqFcn._real2int`), and
                the pointwise call let that reach CMA-ES's own solution
                arrays through a view; the batched call copies the rows,
                so it writes the snapped rows back.
                """
                if isinstance(X, np.ndarray) and X.ndim == 1:
                    return acq_eval(
                        X[None, :], gp, vp, function_logger, optim_state
                    ).item()
                Xs = np.array(X, dtype=float)
                acq = acq_eval(Xs, gp, vp, function_logger, optim_state)
                if np.any(optim_state.get("integer_vars")):
                    for x, row in zip(X, Xs):
                        x[...] = row
                return acq.tolist()

            # Additional search via optimization
            if (
                policy_selection is None
                and options["search_optimizer"] != "none"
                and not repeat_flag
            ):
                search_optimizer = options["search_optimizer"]
                if gp.D == 1:
                    # A one-dimensional acquisition is minimized over the
                    # whole search interval by a bounded scalar search.
                    search_optimizer = "bounded"

                f_val_old = acq_fast[idx]
                x0 = X_acq[0, :]

                if (
                    np.isfinite(optim_state["lb_search"]).all()
                    and np.isfinite(optim_state["ub_search"]).all()
                ):
                    lb_search = np.minimum(x0, optim_state["lb_search"])
                    ub_search = np.maximum(x0, optim_state["ub_search"])
                else:
                    # One bound per coordinate, from the training inputs
                    # and the starting point taken together.
                    xrange = gp.X.max(0) - gp.X.min(0)
                    X_stacked = np.vstack((gp.X, np.atleast_2d(x0)))
                    lb_search = X_stacked.min(0) - 0.1 * xrange
                    ub_search = X_stacked.max(0) + 0.1 * xrange

                if acq_eval.acq_info.get("log_flag"):
                    tol_fun = 1e-2
                else:
                    tol_fun = max(1e-12, abs(f_val_old * 1e-3))

                # A local search that fails costs one acquisition, not the
                # run: the sieve's best candidate is kept instead.
                xsearch_optim, f_val_optim = x0, np.inf

                if search_optimizer == "cmaes":
                    if options["search_cmaes_vp_init"]:
                        _, Sigma = vp.moments(orig_flag=False, cov_flag=True)
                    else:
                        X_hpd = get_hpd(gp.X, gp.y, options["hpd_frac"])[0]
                        Sigma = np.cov(X_hpd, rowvar=False, bias=True)

                    insigma = np.sqrt(np.diag(Sigma))
                    sigma0 = np.max(insigma)
                    cma_options = {
                        "verbose": -9,
                        "tolfun": tol_fun,
                        "maxfevals": options["search_max_fun_evals"],
                        "bounds": (lb_search.squeeze(), ub_search.squeeze()),
                        # Draw the CMA-ES population from our generator; with
                        # a custom `randn`, cma neither seeds nor uses the
                        # global NumPy state for the population.
                        "seed": np.nan,
                        "randn": lambda *shape: rng.standard_normal(shape),
                    }

                    # Start the search at the per-coordinate standard
                    # deviations `insigma`: `sigma0` is the overall step size
                    # and `CMA_stds` the coordinate scaling, which cma keeps
                    # in a non-adapting `sigma_vec` while `C` starts at the
                    # identity and adapts on top of it. A coordinate scaling
                    # needs every entry positive and finite; otherwise the
                    # search starts isotropic at `sigma0`.
                    if np.all(np.isfinite(insigma)) and np.all(insigma > 0):
                        cma_options["CMA_stds"] = insigma / sigma0

                    # The population of each generation is evaluated in one
                    # call (`parallel_objective`); `ask_and_eval` draws it
                    # with a single `ask` in either mode, so the random
                    # stream is the same as with a pointwise objective.
                    # The GP, the variational posterior and the importance
                    # samples are fixed while the search runs, so the
                    # acquisition is deterministic and the search needs no
                    # noise handling: one generation costs one population.
                    try:
                        res = cma.fmin(
                            acq_fun,
                            x0,
                            sigma0,
                            options=cma_options,
                            parallel_objective=acq_fun,
                        )
                    except Exception as exc:
                        _log_search_failure(logger, exc)
                    else:
                        xsearch_optim, f_val_optim = res[:2]
                elif search_optimizer == "bounded":
                    from scipy.optimize import minimize_scalar

                    def acq_fun_1d(x):
                        return acq_fun(np.atleast_1d(x))

                    try:
                        res = minimize_scalar(
                            acq_fun_1d,
                            method="bounded",
                            bounds=(
                                float(np.ravel(lb_search)[0]),
                                float(np.ravel(ub_search)[0]),
                            ),
                            options={
                                "maxiter": options["search_max_fun_evals"],
                                # The step size at which the CMA-ES search
                                # of the other branch stops.
                                "xatol": 1e-11,
                            },
                        )
                    except Exception as exc:
                        _log_search_failure(logger, exc)
                    else:
                        xsearch_optim = np.atleast_1d(res.x)
                        f_val_optim = res.fun
                else:
                    raise NotImplementedError(
                        "options['search_optimizer'] must be 'cmaes' or "
                        "'none', not "
                        f"{options['search_optimizer']!r}."
                    )

                if f_val_optim < f_val_old:
                    X_acq[0, :] = AbstractAcqFcn._real2int(
                        xsearch_optim,
                        parameter_transformer,
                        optim_state["integer_vars"],
                    )
                    idx_cache_acq = np.nan

            # region
            ## Missing port
            # if (
            #     options["uncertainty_handling"]
            #     and options["max_repeated_observations"] > 0
            # ):
            #     if (
            #         optim_state["repeated_observations_streak"]
            #         >= options["max_repeated_observations"]
            #     ):
            #         # Maximum number of consecutive repeated observations
            #         # (to prevent getting stuck in a wrong belief state)
            #         optim_state["repeated_observations_streak"] = 0
            #     else:
            #         from pyvbmc.vbmc.gaussian_process_train import (
            #             _get_training_data,
            #         )

            #         # Re-evaluate acquisition function on training set
            #         X_train = _get_training_data(function_logger)
            #         # Disable variance-based regularization first
            #         oldflag = optim_state["variance_regularized_acq_fcn"]
            #         optim_state["variance_regularized_acq_fcn"] = False
            #         # Use current cost of GP instead of future cost
            #         old_t_algo_per_fun_eval = optim_state["t_algo_per_fun_eval"]
            #         optim_state["t_algo_per_fun_eval"] = t_base / deltaN_eff
            #         acq_train = acq_eval(
            #             X_train, gp, vp, function_logger, optim_state
            #         )
            #         optim_state["variance_regularized_acq_fcn"] = oldflag
            #         optim_state["t_algo_per_fun_eval"] = old_t_algo_per_fun_eval

            #         idx_train = np.argmin(acq_train)
            #         acq_train = acq_train[idx_train]

            #         acq_now = acq_eval(
            #             X_acq[0], gp, vp, function_logger, optim_state
            #         )

            #         if acq_train < options["repeated_acq_discount"]*acq_now:
            #             X_acq[0] = X_train[idx_train]
            #             optim_state["repeated_observations_streak"] += 1
            #         else:
            #             optim_state["repeated_observations_streak"] = 0
            # endregion

            # Missing port: line 356-361, unused?

            xnew = X_acq
            # See if chosen point comes from starting cache
            idx = idx_cache_acq
            if np.isnan(idx):
                y_orig = np.nan
            else:
                idx = int(idx)
                # The stored value belongs to the point the cache holds.
                # The sieve clips every candidate into the search box and
                # the acquisition snaps integer coordinates to their grid,
                # so a candidate that either of the two moved is a point
                # the target has not been called at: it is evaluated. The
                # candidate is compared, exactly and in the inference space
                # where both act, with the row the sieve made of the cached
                # point before its clip. A new transform of the point is
                # no reference: once a warp has rotated the space the
                # transform is a matrix product, and a row of a product can
                # round differently according to the rows computed with it.
                x_cached = cache_rows.get(idx)
                if x_cached is not None and np.array_equal(x_cached, xnew[0]):
                    y_orig = optim_state["cache"]["y_orig"][idx]
                else:
                    y_orig = np.nan
            if selection_policy is not None:
                selection_policy.finish(
                    selected=xnew,
                    cache_index=idx_cache_acq,
                    repeat=repeat_flag,
                )
            timer.start_timer("fun_time")
            if np.isnan(y_orig):
                # Function value is not available, evaluate
                ynew, _, idx_new = function_logger(xnew)
            else:
                ynew, _, idx_new = function_logger.add(xnew, y_orig)
            if not np.isnan(idx_cache_acq):
                # The acquired point leaves the starting cache, whether
                # its value was stored there or has just been evaluated;
                # a point left behind could be drawn and evaluated again.
                for key in ("x_orig", "y_orig", "skip_logger"):
                    if key in optim_state["cache"]:
                        optim_state["cache"][key] = np.delete(
                            optim_state["cache"][key], idx, 0
                        )
            timer.stop_timer("fun_time")
            # The counts follow each evaluation, as in MATLAB, where the
            # function logger refreshes them (`misc/funlogger_vbmc.m:278-279`):
            # the GP refit below reads them before the next acquisition.
            _refresh_training_counts(optim_state, function_logger)

            if hasattr(function_logger, "S"):
                s2new = function_logger.S[idx_new] ** 2
            else:
                s2new = None

            if repeat_flag:
                optim_state["repeated_observations_streak"] = (
                    optim_state.get("repeated_observations_streak", 0) + 1
                )
            else:
                optim_state["repeated_observations_streak"] = 0

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
                        if options["active_sample_gp_update"]:
                            timer.start_timer("gp_train")
                            (
                                gp,
                                __,
                                optim_state["sn2_hpd"],
                                optim_state["hyp_dict"],
                            ) = train_gp(
                                hyp_dict,
                                optim_state,
                                function_logger,
                                iteration_history,
                                options_update,
                                optim_state["plb_tran"],
                                optim_state["pub_tran"],
                                rng=rng,
                            )
                            timer.stop_timer("gp_train")
                        else:
                            if gptmp is None:
                                gp = reupdate_gp(function_logger, gp)
                            else:
                                gp = gptmp

                        if options["active_sample_vp_update"]:
                            # Quick variational optimization
                            timer.start_timer("variational_fit")
                            # Decide number of fast optimizations
                            N_fastopts = math.ceil(
                                options_update["ns_elbo_incr"]
                                * options_update.eval("ns_elbo", {"K": vp.K})
                            )
                            if options["update_random_alpha"]:
                                optim_state["entropy_alpha"] = 1 - np.sqrt(
                                    rng.random()
                                )

                            vp, _, _ = optimize_vp(
                                options_update,
                                optim_state,
                                vp,
                                gp,
                                N_fastopts,
                                slow_opts_N=1,
                            )

                            # Missing port: variational_init_repo, which
                            # collects the variational parameters reached
                            # here for the sieve of a later variational
                            # optimization to start from.
                            timer.stop_timer("variational_fit")
                    else:
                        gp = gptmp
                else:
                    # If NOT performing full updates with active sampling, only
                    # the GP posterior is updated (but not the hyperparameters)

                    # A first observation at a new input adds one training
                    # row, which the rank-1 update extends the posterior
                    # factors with. A repeat is pooled into an existing row
                    # instead, so the whole posterior is recomputed; so is
                    # it under noise shaping, which rescales the noise of
                    # every training point.
                    timer.start_timer("gp_train")
                    update1 = (
                        function_logger.n_evals[idx_new] == 1
                        and not options["noise_shaping"]
                    )
                    if update1:
                        ynew = np.array([[ynew]])  # (1,1)
                        gp.update(
                            xnew, ynew, s2_new=s2new, compute_posterior=True
                        )
                        # gp.t(end+1) = tnew
                    else:
                        gp = reupdate_gp(function_logger, gp)
                    timer.stop_timer("gp_train")

            # Check if active search bounds need to be expanded
            delta_search = 0.05 * (
                optim_state["ub_search"] - optim_state["lb_search"]
            )

            # ADD DIFFERENT CHECKS FOR INTEGER VARIABLES!
            idx = np.abs(xnew - optim_state["lb_search"]) < delta_search
            optim_state["lb_search"][idx] = np.maximum(
                optim_state["lb_tran"][idx],
                optim_state["lb_search"][idx] - delta_search[idx],
            )
            idx = np.abs(xnew - optim_state["ub_search"]) < delta_search
            optim_state["ub_search"][idx] = np.minimum(
                optim_state["ub_tran"][idx],
                optim_state["ub_search"][idx] + delta_search[idx],
            )

            # Hard lower/upper bounds on search (unused)
            prange = optim_state["pub_tran"] - optim_state["plb_tran"]
            LB_searchmin = np.maximum(
                optim_state["plb_tran"]
                - 2 * prange * options["active_search_bound"],
                optim_state["lb_tran"],
            )
            UB_searchmin = np.minimum(
                optim_state["pub_tran"]
                + 2 * prange * options["active_search_bound"],
                optim_state["ub_tran"],
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

            if (np.size(theta0) != np.size(theta)) or (
                np.any(theta0 != theta)
            ):
                # The two ELBOs are compared on one entropy estimator. The
                # reported ELBO of a one-component posterior takes the exact
                # entropy of a Gaussian (`_eval_full_elcbo`), which
                # `_neg_elcbo` returns when it is given no entropy samples;
                # for more components both sides estimate it by sampling.
                if vp0.K == 1:
                    NSentFineK = 0
                else:
                    NSentFineK = math.ceil(
                        options.eval("ns_ent_fine_active", {"K": vp0.K})
                        / vp0.K
                    )
                elbo0 = -_neg_elcbo(
                    theta0, gp, vp0, 0.0, NSentFineK, False, True
                )[0]

                if elbo0 > vp.stats["elbo"]:
                    vp = vp0

    # A rejected full update restores an isolated deepcopy. Reattach the
    # posterior returned to the live solver to the logger's shared transform.
    vp.parameter_transformer = parameter_transformer
    return function_logger, optim_state, vp, gp


def _get_search_points(
    number_of_points: int,
    optim_state: dict,
    function_logger: FunctionLogger,
    vp: VariationalPosterior,
    options: Options,
    cache_rows: dict = None,
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
    cache_rows : dict, optional
        When given, it receives the rows that the starting cache
        contributes, keyed by their cache index, as the transform into the
        inference space made them and before the clip into the search box.
        The caller tells by them whether a candidate is still the point the
        cache holds.

    Returns
    -------
    search_X : ndarray, shape (number_of_points, D)
        The obtained search points.
    idx_cache : ndarray, shape (number_of_points,)
        The indicies of the search points if coming from the cache.

    Notes
    -----
    The starting cache contributes its own share (``cache_frac``) of the
    points; the rest are drawn from five sources in the order
    ``search_cache_frac``, ``heavy_tail_search_frac``, ``mvn_search_frac``,
    ``hpd_search_frac``, ``box_search_frac``, each taking the rounded share
    its fraction gives it or what the sources before it left, whichever is
    smaller. The search cache gives at most the rows it holds. The
    variational posterior draws the points the five leave.

    Random draws use ``vp.rng``.
    """
    rng = vp.rng

    # Take some points from starting cache, if not empty
    x0 = np.copy(optim_state["cache"]["x_orig"])

    lb_search = optim_state.get("lb_search")
    ub_search = optim_state.get("ub_search")

    D = ub_search.shape[1]

    search_X = np.full((0, D), np.nan)
    idx_cache = np.array([])
    parameter_transformer = function_logger.parameter_transformer

    if x0.size > 0:
        # Fraction of points from cache (if nonempty)
        N_cache = math.ceil(number_of_points * options.get("cache_frac"))

        # idx_cache contains min(n_cache, x0.shape[0]) random indicies
        idx_cache = rng.permutation(x0.shape[0])[: min(N_cache, x0.shape[0])]

        search_X = parameter_transformer(x0[idx_cache])
        if cache_rows is not None:
            cache_rows.update(
                (int(index), np.copy(row))
                for index, row in zip(idx_cache, search_X)
            )

    # Randomly sample the points the cache did not provide
    if search_X.shape[0] < number_of_points:
        N_random_points = number_of_points - search_X.shape[0]
        random_Xs = np.full((0, D), np.nan)

        # What the sources drawn so far have left of the points to draw.
        N_left = N_random_points

        def capped_share(fraction):
            """The rounded share of the points to draw that one source
            takes, capped at what the sources before it left.

            The rounded shares can claim more than the whole even where
            the fractions sum to one, a half going away from zero: three
            quarters of two points are three points. MATLAB has no cap
            and builds a search set larger than it asked for
            (``private/activesample_vbmc.m:627-633``).
            """
            share = round_half_away_from_zero(fraction * N_random_points)
            return int(min(N_left, max(0, share)))

        N_search_cache = capped_share(options.get("search_cache_frac"))
        if N_search_cache > 0:  # Take points from search cache
            # The search cache holds the candidates of the previous step,
            # ranked by acquisition value; it is empty until one has run.
            search_cache = optim_state.get("search_cache")
            if search_cache is None:
                search_cache = np.full((0, D), np.nan)
            else:
                search_cache = np.reshape(search_cache, (-1, D))
            N_search_cache = min(N_search_cache, search_cache.shape[0])
            random_Xs = np.append(
                random_Xs,
                search_cache[:N_search_cache],
                axis=0,
            )
        N_left -= N_search_cache

        N_heavy = capped_share(options.get("heavy_tail_search_frac"))
        if N_heavy > 0:
            heavy_Xs, _ = vp.sample(
                N=N_heavy, orig_flag=False, balance_flag=True, df=3
            )
            random_Xs = np.append(random_Xs, heavy_Xs, axis=0)
        N_left -= N_heavy

        N_mvn = capped_share(options.get("mvn_search_frac"))
        if N_mvn > 0:
            mubar, sigmabar = vp.moments(orig_flag=False, cov_flag=True)
            mvn_Xs = rng.multivariate_normal(
                np.ravel(mubar), sigmabar, size=N_mvn
            )
            random_Xs = np.append(random_Xs, mvn_Xs, axis=0)
        N_left -= N_mvn

        N_hpd = capped_share(options.get("hpd_search_frac"))
        if N_hpd > 0:
            hpd_min = options.get("hpd_frac") / 8
            hpd_max = options.get("hpd_frac")
            hpd_fracs = np.sort(
                np.concatenate(
                    (
                        rng.uniform(size=4) * (hpd_max - hpd_min) + hpd_min,
                        np.array([hpd_min, hpd_max]),
                    )
                )
            )
            N_hpd_vec = np.diff(
                round_half_away_from_zero(
                    np.linspace(0, N_hpd, len(hpd_fracs) + 1)
                )
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

                hpd_Xs = rng.multivariate_normal(
                    mubar, sigmabar, size=int(N_hpd_vec[idx])
                )
                random_Xs = np.append(random_Xs, hpd_Xs, axis=0)
        N_left -= N_hpd

        N_box = capped_share(options.get("box_search_frac"))
        if N_box > 0:
            X = function_logger.X[function_logger.X_flag]
            X_diam = np.amax(X, axis=0) - np.amin(X, axis=0)
            plb_tran = optim_state.get("plb_tran")
            pub_tran = optim_state.get("pub_tran")

            if np.all(np.isfinite(lb_search)) and np.all(
                np.isfinite(ub_search)
            ):
                box_lb = lb_search
                box_ub = ub_search
            else:
                box_lb = plb_tran - 3 * (pub_tran - plb_tran)
                box_ub = pub_tran + 3 * (pub_tran - plb_tran)

            box_lb = np.maximum(np.amin(X, axis=0) - 0.5 * X_diam, box_lb)
            box_ub = np.minimum(np.amax(X, axis=0) + 0.5 * X_diam, box_ub)

            box_Xs = rng.random((N_box, D)) * (box_ub - box_lb) + box_lb

            random_Xs = np.append(random_Xs, box_Xs, axis=0)
        N_left -= N_box

        # remaining samples
        N_vp = N_left
        if N_vp > 0:
            vp_Xs, _ = vp.sample(N=N_vp, orig_flag=False, balance_flag=True)
            random_Xs = np.append(random_Xs, vp_Xs, axis=0)

        search_X = np.append(search_X, random_Xs, axis=0)
        idx_cache = np.append(idx_cache, np.full(N_random_points, np.nan))

    # Apply search bounds
    search_X = np.minimum((np.maximum(search_X, lb_search)), ub_search)
    return search_X, idx_cache
