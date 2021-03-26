import sys

import numpy as np

from ..function_logger import FunctionLogger
from ..timer import Timer
from .options_vbmc import OptionsVBMC


def active_sample(
    x0: np.array,
    function_logger: FunctionLogger,
    sample_count: int,
    options: OptionsVBMC,
):
    """
    active_sample Actively sample points iteratively based on acquisition function.

    Parameters
    ----------
    x0 : np.array
        given points?
    function_logger : FunctionLogger
        the FunctionLogger of the function to sample from
    sample_count : int
        the number of samples
    options : OptionsVBMC
        the vbmc algorithm options

    Returns
    -------
    nd.array
        ? samples
    """
    timer = Timer()
    function_time = 0
    timer.start_timer("active_sampling")

    # if GP is None
    if True:
        """
        Initial sample design (provided or random box).
        """
        provided_sample_count, dimension_count = x0.shape

        if provided_sample_count <= sample_count:
            Xs = np.copy(x0)
            ys = np.copy(function_logger.y_orig)

            if provided_sample_count < sample_count:
                if options.InitDesign == "pausible":
                    # Uniform random samples in the plausible box (in transformed space)

                    # dummy fill with 1
                    Xrnd = np.fill(
                        (
                            (sample_count - provided_sample_count),
                            dimension_count,
                        ),
                        1,
                    )
                elif options.InitDesign == "narrow":

                    # dummy fill with 1
                    Xrnd = np.fill(
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
                np.empty(sample_count - provided_sample_count) * np.nan,
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

        # Remove points from starting cache

        # warp Xs points

        timer.start_timer("timer_func")
        for sample_idx in range(sample_count):
            # timer

            if ys[sample_idx] is np.NaN:
                # value is not available, evaluate it
                ys[sample_idx], _, _ = function_logger(Xs[sample_idx])
            else:
                function_logger.add(Xs[sample_idx], ys[sample_idx])

        timer.stop_timer("timer_func")
        function_time += timer.get_duration("timer_func")

    else:

        """
        Active uncertainty sampling
        """

        return None

    timer.stop_timer("active_sampling")
    active_sampling_time = timer.get_duration("active_sampling")
    # handle funEvals timer
