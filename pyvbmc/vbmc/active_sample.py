import logging
import math

import numpy as np
from pyvbmc.function_logger import FunctionLogger
from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.variational_posterior import VariationalPosterior

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

        # remaining samples
        N_vp = max(0, N_random_points - N_search_cache - N_heavy - N_mvn)
        if N_vp > 0:
            vp_Xs, _ = vp.sample(N=N_vp, origflag=False, balanceflag=True)
            random_Xs = np.append(random_Xs, vp_Xs, axis=0)

        search_X = np.append(search_X, random_Xs, axis=0)
        idx_cache = np.append(idx_cache, np.full(N_random_points, np.NaN))

    # Apply search bounds
    search_X = np.minimum((np.maximum(search_X, lb_search)), ub_search)
    return search_X, idx_cache
