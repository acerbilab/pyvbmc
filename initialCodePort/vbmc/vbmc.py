import math
import sys

import numpy as np
from entropy import entlb_vbmc, entub_vbmc
from timer import Timer
from variational_posterior import VariationalPosterior
from .options import Options


class VBMC(object):
    """
    The VBMC algorithm class
    """

    def __init__(
        self,
        fun: callable,
        x0: np.ndarray = None,
        lower_bounds: np.ndarray = None,
        upper_bounds: np.ndarray = None,
        plausible_lower_bounds: np.ndarray = None,
        plausible_upper_bounds: np.ndarray = None,
        user_options: dict = None,
    ):
        # Initialize variables and algorithm structures
        if x0 is None:
            if (
                plausible_lower_bounds is None
                or plausible_upper_bounds is None
            ):
                raise ValueError(
                    """vbmc:UnknownDims If no starting point is
                 provided, PLB and PUB need to be specified."""
                )
            else:
                x0 = np.full((plausible_lower_bounds.shape), np.NaN)

        self.D = x0.shape[1]

        # Check/fix boundaries and starting points
        self._boundscheck(
            fun,
            x0,
            lower_bounds,
            upper_bounds,
            plausible_lower_bounds,
            plausible_upper_bounds,
        )

    def _boundscheck(
        self,
        fun: callable,
        x0: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        plausible_lower_bounds: np.ndarray,
        plausible_upper_bounds: np.ndarray,
    ):
        """
        Private function to do the initial check of the VBMC bounds.

        Parameters
        ----------
        fun : callable
            [description]
        x0 : np.ndarray
            [description]
        lower_bounds, upper_bounds  : np.ndarray
            [description]
        plausible_lower_bounds, plausible_upper_bounds : np.ndarray
            [description]
        """    
        pass

    def algorithm(self, fun, x0, LB, UB, PLB, PUB, options):
        """
        This is a perliminary version of the VBMC loop in order to identify possible objects
        """
        pass

    def __1acqhedge_vbmc(self, action, hedge, stats, options):
        """
        ACQPORTFOLIO Evaluate and update portfolio of acquisition functions. (unused)
        """
        pass

    def __1getAcqInfo(self, SearchAcqFcn):
        """
        GETACQINFO Get information from acquisition function(s)
        """
        pass

    def __1gpreupdate(self, gp, optimState, options):
        """
        GPREUPDATE Quick posterior reupdate of Gaussian process
        """
        pass

    # GP Training

    def __2gptrain_vbmc(self, hypstruct, optimState, stats, options):
        """
        GPTRAIN_VBMC Train Gaussian process model.
        """
        # return [gp,hypstruct,Ns_gp,optimState]
        pass

    # Variational optimization / training of variational posterior:

    def __3updateK(self, optimState, stats, options):
        """
        UPDATEK Update number of variational mixture components.
        """
        pass

    def __3vpoptimize_vbmc(
        self, Nfastopts, Nslowopts, vp, gp, K, optimState, options, prnt
    ):
        """
        VPOPTIMIZE Optimize variational posterior.
        """
        pass

    # Loop termination:

    def __4vbmc_warmup(self, optimState, stats, action, options):
        """
        check if warmup ends
        """
        pass

    def __4vbmc_termination(self, optimState, action, stats, options):
        """
        Compute stability index and check termination conditions.
        """
        pass

    def __4recompute_lcbmax(self, gp, optimState, stats, options):
        """
        RECOMPUTE_LCBMAX Recompute moving LCB maximum based on current GP.
        """
        pass

    # Finalizing:

    def __5finalboost_vbmc(self, vp, idx_best, optimState, stats, options):
        """
        FINALBOOST_VBMC Final boost of variational components.
        """
        pass

    def __5best_vbmc(
        self, stats, idx, SafeSD, FracBack, RankCriterion, RealFlag
    ):
        """
        VBMC_BEST Return best variational posterior from stats structure.
        """
        pass

    def acqhedge_vbmc(self):
        pass
