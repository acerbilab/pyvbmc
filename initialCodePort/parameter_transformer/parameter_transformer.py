import numpy as np


def _handle_1D_input(method):
    """
    _handle_1D_input a decorator to handle 1D inputs
    in several ParameterTransformer methods

    Parameters
    ----------
    method : method
        the method wrapped by the decorator
    """

    def wrapper(self, *args, **kwargs):
        possible_kwargs = np.array(["x", "u"])
        kwargs_present = np.array([k in kwargs for k in possible_kwargs])

        if np.any(kwargs_present):
            # for keyword arguments
            kwarg = possible_kwargs[kwargs_present][0]
            input_array = kwargs.get(kwarg)
            input_dims = np.ndim(input_array)
            if input_dims == 1:
                kwargs[kwarg] = np.reshape(
                    input_array, (1, input_array.shape[0])
                )

        elif len(args) > 0:
            # for positional arguments
            input_array = args[0]
            input_dims = np.ndim(input_array)
            if input_dims == 1:
                arg_list = list(args)
                arg_list[0] = np.reshape(
                    input_array, (1, input_array.shape[0])
                )
                args = tuple(arg_list)

        res = method(self, *args, **kwargs)

        # return value 1D or scalar when 1D input
        if np.ndim(res) != 0 and input_dims == 1:
            if method.__name__ == "log_abs_det_jacobian":
                return res.flatten()[0]
            else:
                return res.flatten()
        else:
            return res

    return wrapper


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
            the number of dimensions
        lower_bounds : np.ndarray, optional
            lower_bound (LB) LB and UB define a set of strict
            lower and upper bounds coordinate vector, by default None
        upper_bounds : np.ndarray, optional
            upper_bounds (UB) LB and UB define a set of strict
            lower and upper bounds coordinate vector, by default None
        plausible_lower_bounds : np.ndarray, optional
            plausible_lower_bound (PLB) such that LB < PLB < PUB < UB.
            PLB and PUB represent a "plausible" range, by default None
        plausible_upper_bounds : np.ndarray, optional
            plausible_upper_bound (PUB) such that LB < PLB < PUB < UB.
            PLB and PUB represent a "plausible" range, by default None
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

        if not (
            np.all(lower_bounds <= plausible_lower_bounds)
            and np.all(plausible_lower_bounds < plausible_upper_bounds)
            and np.all(plausible_upper_bounds <= upper_bounds)
        ):
            raise ValueError(
                "Variable bounds should be LB <= PLB < PUB <= UB for all variables."
            )

        # Transform to log coordinates
        self.lb_orig = lower_bounds
        self.ub_orig = upper_bounds

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
        if not (
            np.all(plausible_lower_bounds == self.lb_orig)
            and np.all(plausible_upper_bounds == self.ub_orig)
        ):
            plausible_lower_bounds = self.__call__(plausible_lower_bounds)
            plausible_upper_bounds = self.__call__(plausible_upper_bounds)

            # Center in transformed space
            for i in range(nvars):
                if np.isfinite(plausible_lower_bounds[:, i]) and np.isfinite(
                    plausible_upper_bounds[:, i]
                ):
                    self.mu[i] = 0.5 * (
                        plausible_lower_bounds[:, i]
                        + plausible_upper_bounds[:, i]
                    )
                    self.delta[i] = (
                        plausible_lower_bounds[:, i]
                        - plausible_upper_bounds[:, i]
                    )

    @_handle_1D_input
    def __call__(self, x: np.ndarray):
        """
        __call__ performs direct transform of constrained variables X into unconstrained variables Y

        Parameters
        ----------
        x : nd.array
            a N x NVARS array, where N is the number of input data and NVARS is the number of dimensions.

        Returns
        -------
        nd.array
            the variables transformed to unconstrained variables
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
            u[:, mask] = np.log(np.divide(z, (1 - z)))
            u[:, mask] = (u[:, mask] - self.mu[mask]) / self.delta[mask]

        # # rotate output
        # if self.R_mat is not None:
        #     u = u * self.R_mat
        # # rescale input
        # if scale is not None:
        #     print(scale)

        return u

    @_handle_1D_input
    def inverse(self, u: np.ndarray):
        """
        inverse performs inverse transform of
        unconstrained variables u into constrained variables x.

        Parameters
        ----------
        u : np.ndarray
            unconstrained variables

        Returns
        -------
        np.ndarray
            constrained variables
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

    @_handle_1D_input
    def log_abs_det_jacobian(self, u: np.ndarray):
        """
        log_abs_det_jacobian returns log probability term for the
        original log pdf evaluated at f^{-1}(Y).

        Parameters
        ----------
        u : np.ndarray
            [description]

        Returns
        -------
        np.array
            log probability term
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
