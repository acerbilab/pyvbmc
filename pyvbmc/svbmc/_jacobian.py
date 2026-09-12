"""Deterministic component expectations of parameter-transform Jacobians."""

import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.special import roots_hermitenorm

_GH_ORDERS = (32, 64, 128, 256, 512, 1024, 2048)
_GH_TOL = 1e-8


def _gaussian_marginals(vp):
    """Return means and variances before the coordinate-wise transform."""
    transform = vp.parameter_transformer
    scale = (
        np.ones(vp.D, dtype=np.float64)
        if transform.scale is None
        else np.asarray(transform.scale, dtype=np.float64).reshape(vp.D)
    )
    rotation = (
        np.eye(vp.D, dtype=np.float64)
        if transform.R_mat is None
        else np.asarray(transform.R_mat, dtype=np.float64)
    )

    means = np.asarray(vp.mu, dtype=np.float64)
    stds = np.asarray(vp.lambd, dtype=np.float64) * np.asarray(
        vp.sigma, dtype=np.float64
    )
    scaled_means = scale[:, None] * means
    scaled_variances = (scale[:, None] * stds) ** 2
    return rotation @ scaled_means, rotation**2 @ scaled_variances, scale


def _log_density_term(kind, points):
    """Log density of the CDF used by a bounded transform."""
    if kind == 3:
        return -np.logaddexp(0.0, points) - np.logaddexp(0.0, -points)
    if kind == 13:
        return np.log(3.0 / 8.0) - 2.5 * np.log1p(points**2 / 4.0)
    raise NotImplementedError(f"Unsupported bounded transform type {kind}.")


def _quadrature_terms(kind, means, stds, order):
    """Gauss-Hermite expectations for scalar Gaussian marginals."""
    nodes, weights = roots_hermitenorm(order)
    points = means[..., None] + stds[..., None] * nodes
    values = _log_density_term(kind, points)
    return np.sum(values * weights, axis=-1) / np.sqrt(2.0 * np.pi)


def _adaptive_term(kind, mean, std, tolerance):
    """Adaptive standard-normal integral and its aggregate error estimate."""
    if std == 0.0:
        return float(_log_density_term(kind, mean)), 0.0

    transition = -mean / std
    width = (1.0 if kind == 3 else 2.0) / std
    # Resolve both the standard-normal mass and the transform's nonlinear
    # transition, including when that transition is in a tail.
    splits = [
        -8.0,
        0.0,
        8.0,
        transition - width,
        transition,
        transition + width,
    ]
    splits = sorted(set(point for point in splits if np.isfinite(point)))
    bounds = [-np.inf, *splits, np.inf]
    interval_tolerance = tolerance / (4.0 * (len(bounds) - 1))

    def integrand(z):
        value = _log_density_term(kind, mean + std * z)
        return value * np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)

    value = 0.0
    error = 0.0
    for lower, upper in zip(bounds[:-1], bounds[1:]):
        part, part_error = quad(
            integrand,
            lower,
            upper,
            epsabs=interval_tolerance,
            epsrel=0.0,
            limit=200,
        )
        value += part
        error += part_error
    return value, error


def expected_log_jacobian(vp):
    """Compute each component's expected inverse-transform log-Jacobian.

    The expectation is under a component's diagonal Gaussian in the VP's
    transformed coordinates. Affine and probit contributions are analytic.
    Logit and Student-4 contributions use a deterministic one-dimensional
    Gauss-Hermite ladder. Marginals that do not converge on the finite ladder
    use adaptive scalar integration with an explicit aggregate error check.

    Parameters
    ----------
    vp : VariationalPosterior
        Posterior whose component expectations are required.

    Returns
    -------
    np.ndarray, shape (K,)
        Float64 expected log absolute Jacobians, one per component.

    Raises
    ------
    RuntimeError
        If deterministic quadrature cannot certify the requested accuracy
        for one or more components.
    NotImplementedError
        If the transformer contains an unknown coordinate transform.
    """
    transform = vp.parameter_transformer
    means, variances, scale = _gaussian_marginals(vp)
    delta = np.asarray(transform.delta, dtype=np.float64).reshape(vp.D)
    center = np.asarray(transform.mu, dtype=np.float64).reshape(vp.D)
    transform_types = np.asarray(transform.type).reshape(vp.D)
    lower = np.asarray(transform.lb_orig, dtype=np.float64).reshape(vp.D)
    upper = np.asarray(transform.ub_orig, dtype=np.float64).reshape(vp.D)

    result = np.full(
        int(vp.K), np.sum(np.log(scale), dtype=np.float64), dtype=np.float64
    )

    unbounded = transform_types == 0
    if np.any(unbounded):
        result += np.sum(np.log(delta[unbounded]), dtype=np.float64)

    bounded = ~unbounded
    if np.any(bounded):
        result += np.sum(
            np.log(upper[bounded] - lower[bounded]) + np.log(delta[bounded]),
            dtype=np.float64,
        )

    y_means = delta[:, None] * means + center[:, None]
    y_variances = delta[:, None] ** 2 * variances

    probit = transform_types == 12
    if np.any(probit):
        result += np.sum(
            -0.5 * np.log(2.0 * np.pi)
            - 0.5 * (y_means[probit] ** 2 + y_variances[probit]),
            axis=0,
        )

    nonlinear = np.isin(transform_types, (3, 13))
    unknown = bounded & ~probit & ~nonlinear
    if np.any(unknown):
        codes = np.unique(transform_types[unknown]).tolist()
        raise NotImplementedError(
            f"Unsupported bounded transform type(s): {codes}."
        )

    if np.any(nonlinear):
        nonlinear_means = y_means[nonlinear]
        nonlinear_stds = np.sqrt(y_variances[nonlinear])
        nonlinear_types = transform_types[nonlinear]
        previous = None
        current = None
        converged = None
        for order in _GH_ORDERS:
            current = np.empty_like(nonlinear_means, dtype=np.float64)
            for kind in (3, 13):
                mask = nonlinear_types == kind
                if np.any(mask):
                    current[mask] = _quadrature_terms(
                        kind,
                        nonlinear_means[mask],
                        nonlinear_stds[mask],
                        order,
                    )
            if previous is not None:
                component_error = np.sum(np.abs(current - previous), axis=0)
                converged = np.isfinite(component_error) & (
                    component_error <= _GH_TOL
                )
                if np.all(converged):
                    result += np.sum(current, axis=0)
                    return np.asarray(result, dtype=np.float64)
            previous = current

        gh_failed = np.flatnonzero(~converged)
        fallback_failed = []
        with warnings.catch_warnings():
            warnings.simplefilter("error", IntegrationWarning)
            for component in gh_failed:
                coordinate_tolerance = _GH_TOL / nonlinear_means.shape[0]
                component_error = 0.0
                try:
                    for coordinate, kind in enumerate(nonlinear_types):
                        value, error = _adaptive_term(
                            kind,
                            nonlinear_means[coordinate, component],
                            nonlinear_stds[coordinate, component],
                            coordinate_tolerance,
                        )
                        current[coordinate, component] = value
                        component_error += error
                except IntegrationWarning:
                    fallback_failed.append(int(component))
                    continue
                if not (
                    np.all(np.isfinite(current[:, component]))
                    and np.isfinite(component_error)
                    and component_error <= _GH_TOL
                ):
                    fallback_failed.append(int(component))

        if fallback_failed:
            raise RuntimeError(
                "Gauss-Hermite quadrature did not converge by "
                f"{_GH_ORDERS[-1]} nodes and adaptive quadrature did not "
                "certify 1e-8 nats for component(s) "
                f"{fallback_failed}."
            )
        result += np.sum(current, axis=0)
        return np.asarray(result, dtype=np.float64)

    return np.asarray(result, dtype=np.float64)
