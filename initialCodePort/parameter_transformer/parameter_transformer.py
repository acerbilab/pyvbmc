import sys

import numpy as np


class ParameterTransformer:
    def __init__(
        self,
        nvars: int,
        lower_bound: np.ndarray = None,
        upper_bound: np.ndarray = None,
        plausible_lower_bound: np.ndarray = None,
        plausible_upper_bound: np.ndarray = None,
    ):
        """
        __init__ [summary]

        Parameters
        ----------
        nvars : int
            [description]
        lower_bound : np.ndarray, optional
            [description], by default None
        upper_bound : np.ndarray, optional
            [description], by default None
        plausible_lower_bound : np.ndarray, optional
            [description], by default None
        plausible_upper_bound : np.ndarray, optional
            [description], by default None
        """
        # Empty LB and UB are Infs
        if lower_bound is None:
            lower_bound = np.ones(nvars) * -np.inf
        if upper_bound is None:
            upper_bound = np.ones(nvars) * np.inf

        # Empty plausible bounds equal hard bounds
        if plausible_lower_bound is None:
            plausible_lower_bound = np.copy(lower_bound)
        if plausible_upper_bound is None:
            plausible_upper_bound = np.copy(upper_bound)

        # Convert scalar inputs to row vectors (I do not think it is necessary)

        if (
            not np.all(lower_bound <= plausible_lower_bound)
            and np.all(plausible_lower_bound < plausible_upper_bound)
            and np.all(plausible_upper_bound <= upper_bound)
        ):
            sys.error(
                "Variable bounds should be LB <= PLB < PUB <= UB for all variables."
            )

        # Transform to log coordinates
        self.lower_bound_orig = lower_bound
        self.upper_bound_orig = upper_bound

        self.type = np.zeros((nvars))
        for i in range(nvars):
            if np.isfinite(lower_bound[i]) and np.isinf(upper_bound[i]):
                self.type[i] = 1
            if np.isinf(lower_bound[i]) and np.isfinite(upper_bound[i]):
                self.type[i] = 2
            if (
                np.isfinite(lower_bound[i])
                and np.isfinite(upper_bound[i])
                and lower_bound[i] < upper_bound[i]
            ):
                self.type[i] = 3

        # Centering (at the end of the transform)
        self.mu = np.zeros(nvars)
        self.delta = np.ones(nvars)

        # Get transformed PLB and ULB
        plausible_lower_bound = self.direct_transform(plausible_lower_bound)
        plausible_upper_bound = self.direct_transform(plausible_upper_bound)

        # Center in transformed space
        for i in range(nvars):
             if np.isfinite(plausible_lower_bound[i]) and np.isfinite(
                 plausible_upper_bound[i]
             ):
                 self.mu[i] = 0.5 * (
                     plausible_lower_bound[i] + plausible_upper_bound[i]
                 )
                 self.delta[i] = (
                     plausible_lower_bound[i] - plausible_upper_bound[i]
                 )

    def direct_transform(self, constrained_variables: np.ndarray):
        """
        direct_transform performs direct transform of constrained variables X into unconstrained variables Y

        Parameters
        ----------
        constrained_variables : nd.array
            a N x NVARS array, where N is the number of input data and NVARS is the number of dimensions.

        Returns
        -------
        nd.array
            the variables transformed to unconstrained variables
        """

        unconstrained_variables = np.copy(constrained_variables)

        # Unbounded scalars (possibly center and rescale)
        mask = self.type == 0
        if np.any(mask):
            unconstrained_variables[mask] = (
                constrained_variables[mask] - self.mu[mask]
            ) / self.delta[mask]

        # Lower and upper bounded scalars
        mask = self.type == 3
        if np.any(mask):
            z = (
                constrained_variables[mask]
                - self.lower_bound_orig / self.upper_bound_orig
                - self.lower_bound_orig
            )
            unconstrained_variables[mask] = np.log(z / (1 - z))
            unconstrained_variables[mask] = (
                unconstrained_variables[mask] - self.mu[mask]
            ) / self.delta[mask]

        self.R_mat = None
        scale = None
        # rotate output
        if self.R_mat is not None:
            unconstrained_variables = unconstrained_variables * self.R_mat
        # rescale input
        if scale is not None:
            print(scale)

        return unconstrained_variables

    def inverse_transform(self, unconstrained_variables: np.ndarray):
        # performs inverse transform of unconstrained variables Y into constrained variables X.
        constrained_variables = 3
        return constrained_variables

    def log_jacobian(self):
        pass