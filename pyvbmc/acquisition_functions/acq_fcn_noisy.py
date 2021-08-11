import sys

import gpyreg as gpr
import numpy as np
from pyvbmc.function_logger import FunctionLogger
from pyvbmc.variational_posterior import VariationalPosterior

from .abstract_acq_fcn import AbstractAcqFcn


class AcqFcnNoisy(AbstractAcqFcn):
    """
    Acquisition function for noisy prospective uncertainty search.
    """

    def _compute_acquisition_function(
        self,
        Xs: np.ndarray,
        vp: VariationalPosterior,
        gp: gpr.GP,
        function_logger: FunctionLogger,
        optim_state: dict,
        f_mu: np.ndarray,
        f_s2: np.ndarray,
        f_bar: np.ndarray,
        var_tot: np.ndarray,
    ):
        """
        Compute the value of the acquisition function.
        """
        # Xs is in *transformed* coordinates

        # Probability density of variational posterior at test points
        realmin = sys.float_info.min
        p = np.ravel(np.maximum(vp.pdf(Xs, origflag=False), realmin))

        # Estimate observation noise at test points from nearest neighbor
        # unravel_index as the indicies are 1D otherwise
        pos = np.unravel_index(
            np.argmin(
                super()._sq_dist(
                    Xs / optim_state.get("gp_length_scale"),
                    gp.temporary_data.get("X_rescaled"),
                ),
                axis=1,
            ),
            gp.temporary_data.get("sn2_new").shape,
        )
        sn2 = gp.temporary_data.get("sn2_new")[pos]

        z = function_logger.ymax

        # Prospective uncertainty search corrected for noisy observations
        acq = -var_tot * (1 - sn2 / (var_tot + sn2)) * np.exp(f_bar - z) * p
        return acq

