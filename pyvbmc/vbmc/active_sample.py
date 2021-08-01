import logging
import math

import numpy as np
from pyvbmc.function_logger import FunctionLogger
from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.stats import get_hpd
from .options import Options


def active_sample(
    gp,
    sample_count: int,
    optim_state: dict,
    function_logger: FunctionLogger,
    parameter_transformer: ParameterTransformer,
    vp: VariationalPosterior,
    options: Options,
):
    """
    Actively sample points iteratively based on acquisition function.

    Parameters
    ----------
    gp : GaussianProcess
        The GaussianProcess from the VBMC instance this function is called from.
    sample_count : int
        The number of samples.
    optim_state : dict
        The optim_state from the VBMC instance this function is called from.
    function_logger : FunctionLogger
        The FunctionLogger from the VBMC instance this function is called from.
    parameter_transformer : ParameterTransformer
        The ParameterTransformer from the VBMC instance this function is called
        from.
    vp : VariationalPosterior
        The VariationalPosterior from the VBMC instance this function is called
        from.
    options : Options
       Options from the VBMC instance this function is called from.

    Returns
    -------
    function_logger : FunctionLogger
        The FunctionLogger from the VBMC instance this function is called from.
    optim_state : dict
        The optim_state from the VBMC instance this function is called from.
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
                        np.random.standard_normal(
                            (sample_count - provided_sample_count, D)
                        )
                        * (pub - plb)
                        + plb
                    )

                elif options.get("initdesign") == "narrow":
                    start_Xs = parameter_transformer(Xs[0])
                    random_Xs = (
                        np.random.standard_normal(
                            (sample_count - provided_sample_count, D)
                        )
                        - 0.5 * 0.1 * (pub - plb)
                        + start_Xs
                    )
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
        optim_state["cache"]["x_orig"][idx_remove] = None
        optim_state["cache"]["y_orig"][idx_remove] = None

        Xs = parameter_transformer(Xs)

        for idx in range(sample_count):
            if np.isnan(ys[idx]):  # Function value is not available
                function_logger(Xs[idx])
            else:
                function_logger.add(Xs[idx], ys[idx])

    else:
        # active uncertainty sampling
        pass

    return function_logger, optim_state


def _get_search_points(
    number_of_points: int,
    optim_state: dict,
    options: Options,
    parameter_transformer: ParameterTransformer,
    function_logger: FunctionLogger,
    vp: VariationalPosterior,
):
    """
    Get search points from starting cache or randomly generated.

    Parameters
    ----------
    number_of_points : int
        The number of points to return.
    optim_state : dict
        The optim_state from the VBMC instance this function is called from.
    options : Options
        Options from the VBMC instance this function is called from.
    parameter_transformer : ParameterTransformer
        The ParameterTransformer from the VBMC instance this function is called
        from.
    function_logger : FunctionLogger
        The FunctionLogger from the VBMC instance this function is called from.
    vp : VariationalPosterior
        The VariationalPosterior from the VBMC instance this function is called
        from.

    Returns
    -------
    search_X : ndarray, shape (number_of_points, D)
        The obtained search points.
    idx_cache : ndarray, shape (number_of_points,)
        The indicies of the search points if coming from the cache.
    """

    # Take some points from starting cache, if not empty
    x0 = np.copy(optim_state["cache"]["x_orig"])

    lb_search = optim_state.get("LB_search")
    ub_search = optim_state.get("UB_search")

    D = ub_search.shape[1]

    search_X = np.full((0, D), np.NaN)
    idx_cache = np.array([])

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

                X_hpd, _, _ = get_hpd(X, y, hpd_fracs[idx])

                if X_hpd.size == 0:
                    idx_max = np.argmax(y)
                    mubar = X[idx_max]
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

        # remaining samples
        N_vp = max(
            0,
            N_random_points - N_search_cache - N_heavy - N_mvn - N_box - N_hpd,
        )
        if N_vp > 0:
            vp_Xs, _ = vp.sample(N=N_vp, origflag=False, balanceflag=True)
            random_Xs = np.append(random_Xs, vp_Xs, axis=0)

        search_X = np.append(search_X, random_Xs, axis=0)
        idx_cache = np.append(idx_cache, np.full(N_random_points, np.NaN))

    # Apply search bounds
    search_X = np.minimum((np.maximum(search_X, lb_search)), ub_search)
    return search_X, idx_cache
