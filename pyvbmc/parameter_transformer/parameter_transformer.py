import numpy as np
from pyvbmc.decorators import handle_0D_1D_input


class ParameterTransformer:
    """
    A class used to enable transforming of variables from unconstrained to
    constrained space and vice versa.

    Parameters
    ----------
    D : int
        The number of dimensions of the spaces.
    lower_bounds : np.ndarray, optional
        The lower_bound (LB) of the space. LB and UB define a set of strict
        lower and upper bounds coordinate vector, by default None.
    upper_bounds : np.ndarray, optional
        The upper_bounds (UB) of the space. LB and UB define a set of strict
        lower and upper bounds coordinate vector, by default None.
    plausible_lower_bounds : np.ndarray, optional
        The plausible_lower_bound (PLB) such that LB < PLB < PUB < UB.
        PLB and PUB represent a "plausible" range, by default None.
    plausible_upper_bounds : np.ndarray, optional
        The plausible_upper_bound (PUB) such that LB < PLB < PUB < UB.
        PLB and PUB represent a "plausible" range, by default None.
    """

    def __init__(
        self,
        D: int,
        lower_bounds: np.ndarray = None,
        upper_bounds: np.ndarray = None,
        plausible_lower_bounds: np.ndarray = None,
        plausible_upper_bounds: np.ndarray = None,
    ):
        # Empty LB and UB are Infs
        if lower_bounds is None:
            lower_bounds = np.ones((1, D)) * -np.inf
        if upper_bounds is None:
            upper_bounds = np.ones((1, D)) * np.inf

        # Empty plausible bounds equal hard bounds
        if plausible_lower_bounds is None:
            plausible_lower_bounds = np.copy(lower_bounds)
        if plausible_upper_bounds is None:
            plausible_upper_bounds = np.copy(upper_bounds)

        # Convert scalar inputs to row vectors (I do not think it is necessary)

        if not (
            np.all(lower_bounds <= plausible_lower_bounds)
            and np.all(plausible_lower_bounds < plausible_upper_bounds)
            and np.all(plausible_upper_bounds <= upper_bounds)
        ):
            raise ValueError(
                """Variable bounds should be LB <= PLB < PUB <= UB
                for all variables."""
            )

        # Transform to log coordinates
        self.lb_orig = lower_bounds
        self.ub_orig = upper_bounds

        self.type = np.zeros((D))
        for i in range(D):
            if (
                np.isfinite(lower_bounds[:, i])
                and np.isfinite(upper_bounds[:, i])
                and lower_bounds[:, i] < upper_bounds[:, i]
            ):
                self.type[i] = 3

        # Centering (at the end of the transform)
        self.mu = np.zeros(D)
        self.delta = np.ones(D)

        # Get transformed PLB and ULB
        if not (
            np.all(plausible_lower_bounds == self.lb_orig)
            and np.all(plausible_upper_bounds == self.ub_orig)
        ):
            plausible_lower_bounds = self.__call__(plausible_lower_bounds)
            plausible_upper_bounds = self.__call__(plausible_upper_bounds)

            # Center in transformed space
            for i in range(D):
                if np.isfinite(plausible_lower_bounds[:, i]) and np.isfinite(
                    plausible_upper_bounds[:, i]
                ):
                    self.mu[i] = 0.5 * (
                        plausible_lower_bounds[:, i]
                        + plausible_upper_bounds[:, i]
                    )
                    self.delta[i] = (
                        plausible_upper_bounds[:, i]
                        - plausible_lower_bounds[:, i]
                    )

    @handle_0D_1D_input(patched_kwargs=["x"], patched_argpos=[0])
    def __call__(self, x: np.ndarray):
        """
        Performs direct transform of original variables X into
        unconstrained variables U.

        Parameters
        ----------
        x : np.ndarray
            A N x D array, where N is the number of input data
            and D is the number of dimensions

        Returns
        -------
        u : np.ndarray
            The variables transformed to unconstrained variables.
        """

        u = np.copy(x)

        # Unbounded scalars (possibly center and rescale)
        mask = self.type == 0
        if np.any(mask):
            u[:, mask] = (x[:, mask] - self.mu[mask]) / self.delta[mask]

        # Lower and upper bounded scalars
        mask = self.type == 3
        if np.any(mask):
            z = (x[:, mask] - self.lb_orig) / (self.ub_orig - self.lb_orig)

            # prevent divide by zero
            u_temp = np.zeros(x[:, mask].shape)
            u_temp[z == 0] = -np.inf
            u_temp[z == 1] = np.inf

            u_temp[u_temp == 0] = np.log(z[u_temp == 0] / (1 - z[u_temp == 0]))
            u[:, mask] = u_temp

            u[:, mask] = (u[:, mask] - self.mu[mask]) / self.delta[mask]

        # # rotate output
        # if self.R_mat is not None:
        #     u = u * self.R_mat
        # # rescale input
        # if scale is not None:
        #     print(scale)

        return u

    @handle_0D_1D_input(patched_kwargs=["u"], patched_argpos=[0])
    def inverse(self, u: np.ndarray):
        """
        Performs inverse transform of unconstrained variables u
        into variables x in the original space

        Parameters
        ----------
        u : np.ndarray
            The unconstrained variables that will be transformed.

        Returns
        -------
        x : np.ndarray
            The original variables which result of the transformation.
        """
        # # rotate input (copy array before)
        # if self.R_mat is not None:
        #     u = u * self.R_mat
        # # rescale input
        # if scale is not None:
        #     print(scale)

        x = np.copy(u)

        # Unbounded scalars (possibly unscale and uncenter)
        mask = self.type == 0
        if np.any(mask):
            x[:, mask] = u[:, mask] * self.delta[mask] + self.mu[mask]

        # Lower and upper bounded scalars
        mask = self.type == 3
        if np.any(mask):
            x[:, mask] = u[:, mask] * self.delta[mask] + self.mu[mask]
            x[:, mask] = self.lb_orig[:, mask] + (
                (self.ub_orig[:, mask] - self.lb_orig[:, mask])
                * (1 / (1 + np.exp(-x[:, mask])))
            )

        # Force to stay within bounds
        mask = np.isfinite(self.lb_orig)[0]
        x[:, mask] = np.maximum(x[:, mask], self.lb_orig[:, mask])

        mask = np.isfinite(self.ub_orig)[0]
        x[:, mask] = np.minimum(x[:, mask], self.ub_orig[:, mask])
        return x

    @handle_0D_1D_input(
        patched_kwargs=["u"], patched_argpos=[0], return_scalar=True
    )
    def log_abs_det_jacobian(self, u: np.ndarray):
        r"""
        log_abs_det_jacobian returns the log absolute value of the determinant
        of the Jacobian of the parameter transformation evaluated at U, that is
        log \|D \du(g^-1(u))\|

        Parameters
        ----------
        u : np.ndarray
            The points where the log determinant of the Jacobian should be
            evaluated (in transformed space).

        Returns
        -------
        p : np.ndarray
            The log absolute determinant of the Jacobian.
        """
        u_c = np.copy(u)

        # # rotate input (copy array before)
        # if self.R_mat is not None:
        #     u_c = u_c * self.R_mat
        # # rescale input
        # if scale is not None:
        #     print(scale)

        p = np.zeros(u_c.shape)

        # Unbounded scalars
        mask = self.type == 0
        if np.any(mask):
            p[:, mask] = np.log(self.delta[mask])[np.newaxis]

        # Lower and upper bounded scalars
        mask = self.type == 3
        if np.any(mask):
            u_c[:, mask] = u_c[:, mask] * self.delta[mask] + self.mu[mask]
            z = -np.log1p(np.exp(-u_c[:, mask]))
            p[:, mask] = (
                np.log(self.ub_orig - self.lb_orig) - u_c[:, mask] + 2 * z
            )
            p[:, mask] = p[:, mask] + np.log(self.delta[mask])

        # Scale transform
        # if scale is not None:
        #     p + np.log(scale)
        p = np.sum(p, axis=1)
        return p
