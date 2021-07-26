import sys

import numpy as np
from pyvbmc.function_logger import FunctionLogger
from pyvbmc.timer import Timer

from .options import Options


def active_sample(
    gp,
    sample_count: int,
    optim_state,
    function_logger: FunctionLogger,
    options: Options,
):
    """
    active_sample Actively sample points iteratively based on acquisition function.
    Parameters
    ----------
    function_logger : FunctionLogger
        the FunctionLogger of the function to sample from
    sample_count : int
        the number of samples
    options : Options
        the vbmc algorithm options
    Returns
    -------
    nd.array
        ? samples
    """
    timer = Timer()
    function_time = 0
    timer.start_timer("active_sampling")

    if gp is None:
        # Initial sample design (provided or random box).
        x0 = optim_state["cache"]["x_orig"]
        provided_sample_count, dimension_count = x0.shape

        if provided_sample_count <= sample_count:
            Xs = np.copy(x0)
            ys = np.copy(optim_state["cache"]["y_orig"])

            if provided_sample_count < sample_count:
                if options["initdesign"] == "plausible":
                    # Uniform random samples in the plausible box (in transformed space)
                    window = optim_state["pub"] - optim_state["plb"]
                    rnd_tmp = np.random.rand(
                        sample_count - provided_sample_count, dimension_count
                    )
                    Xrnd = window * rnd_tmp + optim_state["plb"]
                elif options["initdesign"] == "narrow":

                    # dummy fill with 1
                    Xrnd = np.full(
                        (
                            (sample_count - provided_sample_count),
                            dimension_count,
                        ),
                        1,
                    )
                else:
                    sys.exit("Unknown initial design for VBMC")

            # convert Xrnd back to original space
            Xs = np.append(Xs, Xrnd, axis=0)
            ys = np.append(
                ys,
                np.full((sample_count - provided_sample_count,), np.nan),
                axis=0,
            )
        else:
            # select best points by clustering
            # original R-Code:
            # cluster starting points
            # From each cluster, take points with higher density in original space

            # dummy implementation
            Xs = Xs[:sample_count]
            ys = ys[:sample_count]

        # TODO: Remove points from starting cache
        # optim_state["cache"]["x_orig"][idx_remove, :] = []
        # optim_state["cache"]["y_orig"][idx_remove] = []

        # warp Xs points

        timer.start_timer("timer_func")
        for sample_idx in range(sample_count):
            # timer

            if np.isnan(ys[sample_idx]):
                # value is not available, evaluate it
                ys[sample_idx], _, _ = function_logger(Xs[sample_idx])
            else:
                function_logger.add(Xs[sample_idx], ys[sample_idx])

        timer.stop_timer("timer_func")
        function_time += timer.get_duration("timer_func")

    else:
        # Active uncertainty sampling

        # dummy implementation
        window = optim_state["pub"] - optim_state["plb"]
        rnd_tmp = np.random.rand(sample_count, window.shape[1])
        Xs = window * rnd_tmp + optim_state["plb"]
        for sample_idx in range(sample_count):
            _, _, _ = function_logger(Xs[sample_idx])

    timer.stop_timer("active_sampling")
    active_sampling_time = timer.get_duration("active_sampling")
    # handle funEvals timer
