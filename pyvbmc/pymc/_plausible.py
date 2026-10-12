"""Numerical helpers for PyMC target initialization.

This module deliberately depends only on NumPy and SciPy. PyMC graph handling
and setup accounting belong to :mod:`pyvbmc.pymc._target`.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

MODE_IMPROVEMENT_TOL = 1e-3
MODE_FTOL = 1e-14
MODE_GTOL = 1e-8
PRIOR_LOCATION_Q = 0.005


class _SearchStop(Exception):
    pass


def _vectors(*values):
    arrays = tuple(
        np.asarray(value, dtype=np.float64).reshape(-1) for value in values
    )
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("Coordinate arrays must have matching shapes.")
    return arrays


def move_inside(x0, lb, ub, fraction=2e-3):
    """Move an automatically generated point inside finite hard bounds."""
    x0, lb, ub = _vectors(x0, lb, ub)
    if not 0 < fraction < 0.5:
        raise ValueError(
            "fraction must be strictly between zero and one half."
        )
    result = x0.copy()
    finite = np.isfinite(lb) & np.isfinite(ub)
    width = np.zeros_like(lb)
    width[finite] = ub[finite] - lb[finite]
    if np.any(finite & ~(width > 0)):
        raise ValueError("Finite hard bounds must have positive width.")
    lower = np.empty_like(lb)
    upper = np.empty_like(ub)
    lower[finite] = lb[finite] + fraction * width[finite]
    upper[finite] = ub[finite] - fraction * width[finite]
    result[finite] = np.clip(result[finite], lower[finite], upper[finite])
    moved = result != x0
    return result, moved


def search_mode(value_and_grad, x_start, lb, ub, max_calls):
    """Maximize a value-and-gradient callable under a strict call cap.

    Returns ``(x_best, X, y, n_calls, cap_reached)``. ``X`` and ``y``
    contain finite observations only, while ``n_calls`` counts every attempted
    invocation, including nonfinite line-search trials.
    """
    x_start, lb, ub = _vectors(x_start, lb, ub)
    if isinstance(max_calls, (bool, np.bool_)) or not isinstance(
        max_calls, (int, np.integer)
    ):
        raise ValueError("max_calls must be a positive integer.")
    max_calls = int(max_calls)
    if max_calls <= 0:
        raise ValueError("max_calls must be a positive integer.")

    bounds = [
        (
            float(lower) if np.isfinite(lower) else None,
            float(upper) if np.isfinite(upper) else None,
        )
        for lower, upper in zip(lb, ub)
    ]
    if np.any(lb >= ub):
        raise ValueError(
            "Each lower bound must be smaller than its upper bound."
        )
    points = []
    values = []
    accepted_values = []
    n_calls = 0
    stopped_for_cap = False
    stopped_for_improvement = False

    def objective(x):
        nonlocal n_calls, stopped_for_cap
        if n_calls >= max_calls:
            stopped_for_cap = True
            raise _SearchStop("cap")
        n_calls += 1
        value, gradient = value_and_grad(np.asarray(x, dtype=np.float64))
        value_array = np.asarray(value)
        if value_array.size != 1:
            raise ValueError("value_and_grad must return a scalar value.")
        value = float(value_array.reshape(-1)[0])
        gradient = np.asarray(gradient, dtype=np.float64).reshape(
            x_start.shape
        )
        if np.isfinite(value):
            points.append(np.asarray(x, dtype=np.float64).copy())
            values.append(value)
        return -value, -gradient

    def callback(intermediate):
        nonlocal stopped_for_improvement
        x = getattr(intermediate, "x", intermediate)
        matching = [
            value
            for point, value in zip(reversed(points), reversed(values))
            if np.array_equal(point, x)
        ]
        if not matching:
            return
        accepted_values.append(matching[0])
        if len(accepted_values) < 2:
            return
        improvement = accepted_values[-1] - accepted_values[-2]
        if 0 <= improvement <= MODE_IMPROVEMENT_TOL:
            stopped_for_improvement = True
            raise _SearchStop("absolute improvement")

    try:
        minimize(
            objective,
            x_start,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            callback=callback,
            options={
                "ftol": MODE_FTOL,
                "gtol": MODE_GTOL,
                "maxfun": max_calls,
                "maxiter": max_calls,
                "maxls": 40,
            },
        )
    except _SearchStop:
        pass

    if not values:
        raise ValueError("Mode search did not obtain a finite log density.")
    best = int(np.argmax(values))
    X = np.asarray(points, dtype=np.float64).reshape(-1, x_start.size)
    y = np.asarray(values, dtype=np.float64)
    cap_reached = stopped_for_cap or (
        n_calls >= max_calls and not stopped_for_improvement
    )
    return X[best].copy(), X, y, n_calls, cap_reached


def marginal_sd(hessian):
    """Return marginal SDs only when the full negative Hessian is PD."""
    hessian = np.asarray(hessian, dtype=np.float64)
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("hessian must be a square matrix.")
    D = hessian.shape[0]
    precision = -(hessian + hessian.T) / 2
    try:
        np.linalg.cholesky(precision)
        covariance = np.linalg.solve(precision, np.eye(D, dtype=np.float64))
        variances = np.diag(covariance)
        sd = np.full(D, np.nan, dtype=np.float64)
        positive = np.isfinite(variances) & (variances > 0)
        sd[positive] = np.sqrt(variances[positive])
        usable = np.isfinite(sd) & (sd > 0)
    except np.linalg.LinAlgError:
        sd = np.full(D, np.nan, dtype=np.float64)
        usable = np.zeros(D, dtype=bool)
    return sd, usable


def quantile_box(draws, quantiles=(0.05, 0.95)):
    """Return coordinate-wise quantiles of prior draws."""
    draws = np.asarray(draws, dtype=np.float64)
    if draws.ndim != 2 or draws.shape[0] == 0:
        raise ValueError("draws must have shape (N, D) with N positive.")
    if not np.all(np.isfinite(draws)):
        raise ValueError("draws must contain only finite values.")
    lower_q, upper_q = quantiles
    if not 0 <= lower_q < upper_q <= 1:
        raise ValueError("quantiles must be ordered probabilities in [0, 1].")
    lower, upper = np.quantile(draws, [lower_q, upper_q], axis=0)
    return np.asarray(lower, dtype=np.float64), np.asarray(
        upper, dtype=np.float64
    )


def location_outside_prior(x0, draws, q=PRIOR_LOCATION_Q):
    """Flag coordinates outside a central prior interval without moving them."""
    x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
    if not 0 <= q < 0.5:
        raise ValueError("q must be at least zero and less than one half.")
    lower, upper = quantile_box(draws, quantiles=(q, 1 - q))
    if lower.shape != x0.shape:
        raise ValueError(
            "draws and x0 must have the same coordinate dimension."
        )
    return (x0 < lower) | (x0 > upper)


def clip_inside(plb, pub, lb, ub, x0, margin=0.01):
    """Clip an automatic plausible box inside finite hard bounds."""
    plb, pub, lb, ub, x0 = _vectors(plb, pub, lb, ub, x0)
    if not 0 < margin < 0.5:
        raise ValueError("margin must be strictly between zero and one half.")
    result_lower = plb.copy()
    result_upper = pub.copy()
    finite = np.isfinite(lb) & np.isfinite(ub)
    width = np.zeros_like(lb)
    width[finite] = ub[finite] - lb[finite]
    if np.any(finite & ~(width > 0)):
        raise ValueError("Finite hard bounds must have positive width.")

    safe_lower = np.zeros_like(lb)
    safe_upper = np.zeros_like(ub)
    inner_lower = lb[finite] + margin * width[finite]
    inner_upper = ub[finite] - margin * width[finite]
    safe_lower[finite] = np.where(
        inner_lower >= x0[finite],
        (lb[finite] + x0[finite]) / 2,
        inner_lower,
    )
    safe_upper[finite] = np.where(
        inner_upper <= x0[finite],
        (ub[finite] + x0[finite]) / 2,
        inner_upper,
    )
    touched = finite & (
        (result_lower < safe_lower) | (result_upper > safe_upper)
    )
    result_lower[finite] = np.maximum(result_lower[finite], safe_lower[finite])
    result_upper[finite] = np.minimum(result_upper[finite], safe_upper[finite])
    bad = finite & (
        (result_upper <= result_lower)
        | (result_lower >= x0)
        | (result_upper <= x0)
    )
    result_lower[bad] = safe_lower[bad]
    result_upper[bad] = safe_upper[bad]
    return result_lower, result_upper, touched | bad


def laplace_box(
    x0,
    hessian,
    lb,
    ub,
    prior_draws,
    k=3.0,
    check_location=True,
):
    """Construct a Laplace plausible box with prior-width fallbacks."""
    x0, lb, ub = _vectors(x0, lb, ub)
    if not np.isfinite(k) or k <= 0:
        raise ValueError("k must be finite and positive.")

    hessian_value = hessian()
    if hessian_value is None:
        sd = np.full(x0.size, np.nan, dtype=np.float64)
        usable = np.zeros(x0.size, dtype=bool)
    else:
        sd, usable = marginal_sd(hessian_value)
        if sd.shape != x0.shape:
            raise ValueError("hessian and x0 dimensions do not match.")
    curvature_mask = ~usable

    draws = None

    def get_draws():
        nonlocal draws
        if draws is None:
            draws = np.asarray(prior_draws(), dtype=np.float64)
            if draws.ndim != 2 or draws.shape[1:] != (x0.size,):
                raise ValueError("prior draws must have shape (N, D).")
        return draws

    if np.any(curvature_mask):
        prior_lower, prior_upper = quantile_box(get_draws())
        prior_sd = (prior_upper - prior_lower) / (2 * k)
        if np.any(~np.isfinite(prior_sd) | (prior_sd <= 0)):
            raise ValueError(
                "Prior fallback widths must be finite and positive."
            )
        sd[curvature_mask] = prior_sd[curvature_mask]

    if check_location:
        location_mask = location_outside_prior(x0, get_draws())
    else:
        location_mask = np.zeros(x0.size, dtype=bool)

    plb = x0 - k * sd
    pub = x0 + k * sd
    plb, pub, clipped_mask = clip_inside(plb, pub, lb, ub, x0)
    return (
        x0.copy(),
        plb,
        pub,
        curvature_mask,
        location_mask,
        clipped_mask,
    )


__all__ = [
    "MODE_FTOL",
    "MODE_GTOL",
    "MODE_IMPROVEMENT_TOL",
    "PRIOR_LOCATION_Q",
    "clip_inside",
    "laplace_box",
    "location_outside_prior",
    "marginal_sd",
    "move_inside",
    "quantile_box",
    "search_mode",
]
