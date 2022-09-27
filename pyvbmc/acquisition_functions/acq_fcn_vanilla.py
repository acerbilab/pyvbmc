import sys

import gpyreg as gpr
import numpy as np

from pyvbmc.function_logger import FunctionLogger
from pyvbmc.variational_posterior import VariationalPosterior

from .abstract_acq_fcn import AbstractAcqFcn


class AcqFcnVanilla(AbstractAcqFcn):
    """
    Acquisition function via vanilla uncertainty sampling.
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
        p = np.ravel(np.maximum(vp.pdf(Xs, orig_flag=False), realmin))

        # Uncertainty search
        acq = -var_tot * p**2

        return acq
