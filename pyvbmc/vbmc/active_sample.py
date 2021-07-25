import numpy as np
from pyvbmc.function_logger import FunctionLogger
from pyvbmc.parameter_transformer import ParameterTransformer

from .options import Options


def active_sample(
    gp,
    sample_count: int,
    optim_state,
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
    function_logger : FunctionLogger
        The FunctionLogger from the VBMC instance this function is called from.
    parameter_transformer : ParameterTransformer
        The ParameterTransformer from the VBMC instance this function is called
        from.
    sample_count : int
        The number of samples to return.
    options : Options
       Options from the VBMC instance this function is called from.

    Returns
    -------
    nd.array
        ? samples
    """

    if gp is None:
        # Initial sample design (provided or random box).
        x0 = optim_state["cache"]["x_orig"]
        provided_sample_count, D = x0.shape

        if provided_sample_count <= sample_count:
            Xs = np.copy(x0)
            ys = np.copy(optim_state["cache"]["y_orig"])

            idx_remove = np.full(provided_sample_count, True)

        # Remove points from starting cache
        optim_state["cache"]["x_orig"][idx_remove] = None
        optim_state["cache"]["y_orig"][idx_remove] = None

        Xs = parameter_transformer(Xs)

    for idx in range(sample_count):
        if np.isnan(ys[idx]):  # Function value is not available
            function_logger(Xs[idx])
        else:
            function_logger.add(Xs[idx], ys[idx])

    return function_logger, optim_state
