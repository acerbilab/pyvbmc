import sys

import numpy as np


class ParameterTransformer:
    def __init__(
        self,
        nvars: int,
        lower_bounds: np.ndarray = None,
        upper_bounds: np.ndarray = None,
        plausible_lower_bounds: np.ndarray = None,
        plausible_upper_bounds: np.ndarray = None,
    ):
        """
        __init__ [summary]

        Parameters
        ----------
        nvars : int
            [description]
        lower_bounds : np.ndarray, optional
            [description], by default None
        upper_bounds : np.ndarray, optional
            [description], by default None
        plausible_lower_bounds : np.ndarray, optional
            [description], by default None
        plausible_upper_bounds : np.ndarray, optional
            [description], by default None
        """
        # Empty LB and UB are Infs
        if lower_bounds is None:
            lower_bounds = np.ones((1, nvars)) * -np.inf
        if upper_bounds is None:
            upper_bounds = np.ones((1, nvars)) * np.inf

        # Empty plausible bounds equal hard bounds
        if plausible_lower_bounds is None:
            plausible_lower_bounds = np.copy(lower_bounds)
        if plausible_upper_bounds is None:
            plausible_upper_bounds = np.copy(upper_bounds)

        # Convert scalar inputs to row vectors (I do not think it is necessary)

        if (
            not np.all(lower_bounds <= plausible_lower_bounds)
            and np.all(plausible_lower_bounds < plausible_upper_bounds)
            and np.all(plausible_upper_bounds <= upper_bounds)
        ):
            sys.error(
                "Variable bounds should be LB <= PLB < PUB <= UB for all variables."
            )

        # Transform to log coordinates
        self.lower_bounds_orig = lower_bounds
        self.upper_bounds_orig = upper_bounds

        self.type = np.zeros((nvars))
        for i in range(nvars):
            if (
                np.isfinite(lower_bounds[:, i])
                and np.isfinite(upper_bounds[:, i])
                and lower_bounds[:, i] < upper_bounds[:, i]
            ):
                self.type[i] = 3

        # Centering (at the end of the transform)
        self.mu = np.zeros(nvars)
        self.delta = np.ones(nvars)

        # Get transformed PLB and ULB
        plausible_lower_bounds = self.direct_transform(plausible_lower_bounds)
        plausible_upper_bounds = self.direct_transform(plausible_upper_bounds)

        # Center in transformed space
        for i in range(nvars):
            if np.isfinite(plausible_lower_bounds[:, i]) and np.isfinite(
                plausible_upper_bounds[:, i]
            ):
                self.mu[i] = 0.5 * (
                    plausible_lower_bounds[:, i] + plausible_upper_bounds[:, i]
                )
                self.delta[i] = (
                    plausible_lower_bounds[:, i] - plausible_upper_bounds[:, i]
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
            unconstrained_variables[:, mask] = (
                constrained_variables[:, mask] - self.mu[mask]
            ) / self.delta[mask]

        # Lower and upper bounded scalars
        mask = self.type == 3
        if np.any(mask):
            z = (constrained_variables[:, mask] - self.lower_bounds_orig) / (
                self.upper_bounds_orig - self.lower_bounds_orig
            )
            unconstrained_variables[:, mask] = np.log(np.divide(z, (1 - z)))
            unconstrained_variables[:, mask] = (
                unconstrained_variables[:, mask] - self.mu[mask]
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
        self.R_mat = None
        scale = None
        # rotate output (copy array before)
        if self.R_mat is not None:
            unconstrained_variables = unconstrained_variables * self.R_mat
        # rescale input
        if scale is not None:
            print(scale)

        constrained_variables = np.copy(unconstrained_variables)

        # Unbounded scalars (possibly unscale and uncenter)
        mask = self.type == 0
        if np.any(mask):
            constrained_variables[:, mask] = (
                unconstrained_variables[:, mask] * self.delta[mask]
                + self.mu[mask]
            )

        # Lower and upper bounded scalars
        mask = self.type == 3
        if np.any(mask):
            constrained_variables[:, mask] = (
                unconstrained_variables[:, mask] * self.delta[mask]
                + self.mu[mask]
            )
            constrained_variables[:, mask] = self.lower_bounds_orig[
                :, mask
            ] + (
                (
                    self.upper_bounds_orig[:, mask]
                    - self.lower_bounds_orig[:, mask]
                )
                * (1 / (1 + np.exp(-constrained_variables[:, mask])))
            )

        # Force to stay within bounds
        mask = np.isfinite(self.lower_bounds_orig)[0]
        constrained_variables[:, mask] = np.maximum(
            constrained_variables[:, mask], self.lower_bounds_orig[:, mask]
        )

        mask = np.isfinite(self.upper_bounds_orig)[0]
        constrained_variables[:, mask] = np.minimum(
            constrained_variables[:, mask], self.upper_bounds_orig[:, mask]
        )
        return constrained_variables

    def log_jacobian(self):
        pass