import numpy as np
from scipy.special import erfc, erfcinv

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
    bounded_transform_type : str, optional
        A string indicating the type of transform for bounded variables: one of
        ["logit", ("norminv" || "probit"), "student4"]. Default "logit".
    """

    def __init__(
        self,
        D: int,
        lower_bounds: np.ndarray = None,
        upper_bounds: np.ndarray = None,
        plausible_lower_bounds: np.ndarray = None,
        plausible_upper_bounds: np.ndarray = None,
        scale: np.ndarray = None,
        rotation_matrix: np.ndarray = None,
        transform_type="logit",
    ):
        self.scale = scale
        self.R_mat = rotation_matrix

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

        # Select and validate the type of transform:
        transform_types = {
            "logit": 3,
            "norminv": 12,
            "probit": 12,
            "student4": 13,
        }
        if type(transform_type) == str:
            try:
                bounded_type = transform_types[transform_type]
            except KeyError:
                raise ValueError(
                    f"Unrecognized bounded transform {transform_type}."
                )
        else:
            if transform_type not in transform_types.values():
                raise ValueError(
                    f"Unrecognized bounded transform {transform_type}."
                )
            bounded_type = transform_type

        # Setup bounded transforms:
        self.bounded_types = [bounded_type]
        self._bounded_transforms = {}
        self._set_bounded_transforms()

        self.type = np.zeros((D))
        for i in range(D):
            if (
                np.isfinite(lower_bounds[:, i])
                and np.isfinite(upper_bounds[:, i])
                and lower_bounds[:, i] < upper_bounds[:, i]
            ):
                self.type[i] = bounded_type

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
        for t in self.bounded_types:
            mask = self.type == t
            if np.any(mask):
                u[:, mask] = self._bounded_transforms[t]["direct"](x, mask)

        # Rotoscale whitening:
        # Rotate and rescale points in transformed space.
        if self.R_mat is not None:
            u = u @ self.R_mat
        if self.scale is not None:
            u = u / self.scale

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

        x = np.copy(u)

        # Rotoscale whitening:
        # Undo rescaling and rotation.
        if self.scale is not None:
            x = x * self.scale
        if self.R_mat is not None:
            x = x @ np.transpose(self.R_mat)

        xNew = np.copy(x)

        # Unbounded scalars (possibly unscale and uncenter)
        mask = self.type == 0
        if np.any(mask):
            xNew[:, mask] = x[:, mask] * self.delta[mask] + self.mu[mask]

        # Lower and upper bounded scalars
        for t in self.bounded_types:
            mask = self.type == t
            if np.any(mask):
                xNew[:, mask] = self._bounded_transforms[t]["inverse"](x, mask)

        return xNew

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

        # Rotoscale whitening:
        # Undo rescaling and rotation.
        if self.scale is not None:
            u_c = u_c * self.scale
        if self.R_mat is not None:
            u_c = u_c @ np.transpose(self.R_mat)

        p = np.zeros(u_c.shape)

        # Unbounded scalars
        mask = self.type == 0
        if np.any(mask):
            p[:, mask] = np.log(self.delta[mask])[np.newaxis]

        # Lower and upper bounded scalars
        for t in self.bounded_types:
            mask = self.type == t
            if np.any(mask):
                p[:, mask] = self._bounded_transforms[t]["jacobian"](u_c, mask)

        # Whitening/rotoscaling density correction:
        if self.scale is not None:
            p = p + np.log(self.scale)

        p = np.sum(p, axis=1)
        return p

    def _set_bounded_transforms(self):
        r"""Initialize functions for bounded transform(s) by type.

        Stores the resulting callables in a dictionary
        ``self._bounded_transforms[t][case]``, where ``t`` is the integer name
        of the transform type, and ``case`` is one of ``["direct", "inverse",
        "jacobian"]`` for the corresponding function.
        """
        for t in self.bounded_types:
            self._bounded_transforms[t] = {}
            if t == 3:
                # logit: default transform

                def bounded_transform(x, mask):
                    return _center(
                        _logit(
                            _to_unit_interval(
                                x[:, mask],
                                self.lb_orig[:, mask],
                                self.ub_orig[:, mask],
                            )
                        ),
                        self.mu[mask],
                        self.delta[mask],
                    )

                self._bounded_transforms[t]["direct"] = bounded_transform

                def bounded_inverse(u, mask):
                    return _from_unit_interval(
                        _inverse_logit(
                            _uncenter(
                                u[:, mask], self.mu[mask], self.delta[mask]
                            )
                        ),
                        self.lb_orig[:, mask],
                        self.ub_orig[:, mask],
                    )

                self._bounded_transforms[t]["inverse"] = bounded_inverse

                def bounded_jacobian(u, mask):
                    j1 = np.log(self.ub_orig[:, mask] - self.lb_orig[:, mask])
                    y = _uncenter(u[:, mask], self.mu[mask], self.delta[mask])
                    z = -np.log1p(np.exp(-y))
                    j2 = -y + 2 * z
                    j3 = np.log(self.delta[mask])
                    return j1 + j2 + j3

                self._bounded_transforms[t]["jacobian"] = bounded_jacobian

            elif t == 12:
                # norminv: normal CDF (probit) transform

                def bounded_transform(x, mask):
                    return _center(
                        _norminv(
                            _to_unit_interval(
                                x[:, mask],
                                self.lb_orig[:, mask],
                                self.ub_orig[:, mask],
                            )
                        ),
                        self.mu[mask],
                        self.delta[mask],
                    )

                self._bounded_transforms[t]["direct"] = bounded_transform

                def bounded_inverse(u, mask):
                    return _from_unit_interval(
                        _inverse_norminv(
                            _uncenter(
                                u[:, mask], self.mu[mask], self.delta[mask]
                            )
                        ),
                        self.lb_orig[:, mask],
                        self.ub_orig[:, mask],
                    )

                self._bounded_transforms[t]["inverse"] = bounded_inverse

                def bounded_jacobian(u, mask):
                    j1 = np.log(self.ub_orig[:, mask] - self.lb_orig[:, mask])
                    y = _uncenter(u[:, mask], self.mu[mask], self.delta[mask])
                    j2 = -0.5 * np.log(2 * np.pi) - 0.5 * y**2
                    j3 = np.log(self.delta[mask])
                    return j1 + j2 + j3

                self._bounded_transforms[t]["jacobian"] = bounded_jacobian

            elif t == 13:
                # student4: Student's T with nu=4 CDF transform

                def bounded_transform(x, mask):
                    return _center(
                        _student4(
                            _to_unit_interval(
                                x[:, mask],
                                self.lb_orig[:, mask],
                                self.ub_orig[:, mask],
                            )
                        ),
                        self.mu[mask],
                        self.delta[mask],
                    )

                self._bounded_transforms[t]["direct"] = bounded_transform

                def bounded_inverse(u, mask):
                    return _from_unit_interval(
                        _inverse_student4(
                            _uncenter(
                                u[:, mask], self.mu[mask], self.delta[mask]
                            )
                        ),
                        self.lb_orig[:, mask],
                        self.ub_orig[:, mask],
                    )

                self._bounded_transforms[t]["inverse"] = bounded_inverse

                def bounded_jacobian(u, mask):
                    j1 = np.log(self.ub_orig[:, mask] - self.lb_orig[:, mask])
                    y = _uncenter(u[:, mask], self.mu[mask], self.delta[mask])
                    j2 = np.log(3 / 8) - (5 / 2) * np.log1p(y**2 / 4)
                    j3 = np.log(self.delta[mask])
                    return j1 + j2 + j3

                self._bounded_transforms[t]["jacobian"] = bounded_jacobian

            else:
                raise NotImplementedError

    def __str__(self):
        """Print a string summary."""
        transform_names = {
            0: "unbounded",
            3: "logit",
            12: "norminv",
            12: "probit",
            13: "student4",
        }
        transforms = [transform_names[number] for number in self.type]
        return f"""ParameterTransformer:
    D = {self.lb_orig.shape[1]},
    lower bounds = {self.lb_orig},
    upper bounds = {self.ub_orig},
    transform type(s) = {transforms}."""

    def __repr__(self):
        """Print a detailed string representation."""
        return f"""ParameterTransformer:
    D = {self.lb_orig.shape[1]},
    self.lb_orig = {self.lb_orig},
    self.ub_orig = {self.ub_orig},
    self.type = {self.type},
    self.mu = {self.mu},
    self.delta = {self.delta},
    self.scale = {self.scale},
    self.R_mat = {self.R_mat}."""


def _to_unit_interval(x, lb, ub, safe=True):
    z = (x - lb) / (ub - lb)
    if safe:  # Nudge points away from boundary
        mask = (z == 0) & (x != lb)
        if np.any(mask):
            z[mask] = np.nextafter(0, np.inf)
        mask = (z == 1) & (x != ub)
        if np.any(mask):
            z[mask] = np.nextafter(1, -np.inf)
    return z


def _from_unit_interval(z, lb, ub, safe=True):
    y = z * (ub - lb) + lb
    if safe:  # Nudge points away from boundary
        y = np.maximum(y, np.nextafter(lb, np.inf))
        y = np.minimum(y, np.nextafter(ub, -np.inf))
    return y


def _center(u, mu, delta):
    return (u - mu) / delta


def _uncenter(v, mu, delta):
    return v * delta + mu


def _logit(z):
    # prevent divide by zero
    u = np.zeros(z.shape)
    u[z == 0] = -np.inf
    u[z == 1] = np.inf

    u[u == 0] = np.log(z[u == 0] / (1 - z[u == 0]))
    return u


def _inverse_logit(u):
    return 1 / (1 + np.exp(-u))


def _norminv(z):
    return -np.sqrt(2) * erfcinv(2 * z)


def _inverse_norminv(u):
    return 0.5 * erfc(-u / np.sqrt(2))


def _student4(z):
    aa = np.sqrt(4 * z * (1 - z))
    q = np.cos(np.arccos(aa) / 3) / aa
    return np.sign(z - 0.5) * (2 * np.sqrt(q - 1))


def _inverse_student4(u):
    t2 = u**2
    return 0.5 + (3 / 8) * (u / np.sqrt(1 + t2 / 4)) * (
        1 - t2 / (1 + t2 / 4) / 12
    )
