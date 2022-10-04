import sys

import gpyreg as gpr
import numpy as np

from pyvbmc.function_logger import FunctionLogger
from pyvbmc.variational_posterior import VariationalPosterior

from .abstract_acq_fcn import AbstractAcqFcn


class AcqFcnLog(AbstractAcqFcn):
    """
    Acquisition function for prospective uncertainty search (log-valued).
    """

    def __init__(self):
        super().__init__()
        self.acq_info["log_flag"] = True

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
        log_p = np.ravel(
            np.maximum(
                vp.pdf(Xs, orig_flag=False, log_flag=True), np.log(realmin)
            )
        )

        # Log prospective uncertainty search
        z = function_logger.y_max
        acq = -(np.log(var_tot) + f_bar - z + log_p)

        return acq
