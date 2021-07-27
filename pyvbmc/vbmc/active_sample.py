import logging

import numpy as np
from pyvbmc.function_logger import FunctionLogger
from pyvbmc.parameter_transformer import ParameterTransformer

from .options import Options


def active_sample(
    gp,
    sample_count: int,
    optim_state: dict,
    function_logger: FunctionLogger,
    parameter_transformer: ParameterTransformer,
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
