"""Shared normalization of VBMC hard and plausible bounds."""

import logging
import sys

import numpy as np


def _effective_bounds(lower_bounds, upper_bounds):
    """Return hard bounds moved inward by VBMC's numerical margin."""
    bounds_range = upper_bounds - lower_bounds
    bounds_range[np.isinf(bounds_range)] = 1e3
    scale_factor = 1e-3
    realmin = sys.float_info.min
    lower_effective = lower_bounds + scale_factor * bounds_range
    lower_effective[np.abs(lower_bounds) <= realmin] = (
        scale_factor * bounds_range[np.abs(lower_bounds) <= realmin]
    )
    upper_effective = upper_bounds - scale_factor * bounds_range
    upper_effective[np.abs(upper_bounds) <= realmin] = (
        -scale_factor * bounds_range[np.abs(upper_bounds) <= realmin]
    )
    # Infinities stay the same
    lower_effective[np.isinf(lower_bounds)] = lower_bounds[
        np.isinf(lower_bounds)
    ]
    upper_effective[np.isinf(upper_bounds)] = upper_bounds[
        np.isinf(upper_bounds)
    ]

    if np.any(lower_effective >= upper_effective):
        raise ValueError(
            "vbmc:StrictBoundsTooClose: Hard bounds LB and UB\n"
            "                are numerically too close. Make them more "
            "separate."
        )

    return lower_effective, upper_effective


def _expand_scalar_bound(bound, D):
    """Replicate a bound given as a single value across the D variables."""
    if bound is None or bound.size != 1:
        return bound
    return np.full((1, D), bound.reshape(-1)[0])


def _normalize_bounds(
    x0,
    lower_bounds,
    upper_bounds,
    plausible_lower_bounds=None,
    plausible_upper_bounds=None,
    *,
    logger,
):
    """Validate and normalize one VBMC starting set and its bounds."""
    # Work on detached arrays: the permissive bound repairs below must
    # not modify arrays owned by the caller.
    x0 = np.array(x0, copy=True)
    lower_bounds = np.array(lower_bounds, copy=True)
    upper_bounds = np.array(upper_bounds, copy=True)
    if plausible_lower_bounds is not None:
        plausible_lower_bounds = np.array(plausible_lower_bounds, copy=True)
    if plausible_upper_bounds is not None:
        plausible_upper_bounds = np.array(plausible_upper_bounds, copy=True)

    supplied = [
        x0,
        lower_bounds,
        upper_bounds,
        plausible_lower_bounds,
        plausible_upper_bounds,
    ]
    if any(
        value is not None and np.any(np.invert(np.isreal(value)))
        for value in supplied
    ):
        raise ValueError(
            "All input vectors (x0, lower_bounds, upper_bounds,\n"
            "                 plausible_lower_bounds, "
            "plausible_upper_bounds), if specified,\n"
            "                 need to be real valued."
        )
    integer_inputs = [
        value is not None and np.issubdtype(value.dtype, np.integer)
        for value in supplied
    ]
    x0 = x0.astype(np.float64, copy=False)
    lower_bounds = lower_bounds.astype(np.float64, copy=False)
    upper_bounds = upper_bounds.astype(np.float64, copy=False)
    if plausible_lower_bounds is not None:
        plausible_lower_bounds = plausible_lower_bounds.astype(
            np.float64, copy=False
        )
    if plausible_upper_bounds is not None:
        plausible_upper_bounds = plausible_upper_bounds.astype(
            np.float64, copy=False
        )

    N0, D = x0.shape

    # A bound given as a single value holds for every variable.
    lower_bounds = _expand_scalar_bound(lower_bounds, D)
    upper_bounds = _expand_scalar_bound(upper_bounds, D)
    plausible_lower_bounds = _expand_scalar_bound(plausible_lower_bounds, D)
    plausible_upper_bounds = _expand_scalar_bound(plausible_upper_bounds, D)

    if plausible_lower_bounds is None or plausible_upper_bounds is None:
        if N0 > 1:
            logger.warning(
                "PLB and/or PUB not specified. Estimating"
                "plausible bounds from starting set X0..."
            )
            width = x0.max(0) - x0.min(0)
            if plausible_lower_bounds is None:
                plausible_lower_bounds = x0.min(0) - width / N0
                plausible_lower_bounds = np.maximum(
                    plausible_lower_bounds, lower_bounds
                )
            if plausible_upper_bounds is None:
                plausible_upper_bounds = x0.max(0) + width / N0
                plausible_upper_bounds = np.minimum(
                    plausible_upper_bounds, upper_bounds
                )

            idx = plausible_lower_bounds == plausible_upper_bounds
            if np.any(idx):
                plausible_lower_bounds[idx] = lower_bounds[idx]
                plausible_upper_bounds[idx] = upper_bounds[idx]
                logger.warning(
                    "vbmc:pbInitFailed: Some plausible bounds could not be "
                    "determined from starting set. Using hard upper/lower"
                    " bounds for those instead."
                )
        else:
            logger.warning(
                "vbmc:pbUnspecified: Plausible lower/upper bounds PLB and"
                "/or PUB not specified and X0 is not a valid starting set. "
                "Using hard upper/lower bounds instead."
            )
            if plausible_lower_bounds is None:
                plausible_lower_bounds = np.copy(lower_bounds)
                integer_inputs[3] = integer_inputs[1]
            if plausible_upper_bounds is None:
                plausible_upper_bounds = np.copy(upper_bounds)
                integer_inputs[4] = integer_inputs[2]

    # Try to reshape bounds to row vectors
    lower_bounds = np.atleast_1d(lower_bounds)
    upper_bounds = np.atleast_1d(upper_bounds)
    plausible_lower_bounds = np.atleast_1d(plausible_lower_bounds)
    plausible_upper_bounds = np.atleast_1d(plausible_upper_bounds)
    try:
        if lower_bounds.shape != (1, D):
            logging.warning("Reshaping lower bounds to (1, %d).", D)
            lower_bounds = lower_bounds.reshape((1, D))
        if upper_bounds.shape != (1, D):
            logging.warning("Reshaping upper bounds to (1, %d).", D)
            upper_bounds = upper_bounds.reshape((1, D))
        if plausible_lower_bounds.shape != (1, D):
            logging.warning("Reshaping plausible lower bounds to (1, %d).", D)
            plausible_lower_bounds = plausible_lower_bounds.reshape((1, D))
        if plausible_upper_bounds.shape != (1, D):
            logging.warning("Reshaping plausible upper bounds to (1, %d).", D)
            plausible_upper_bounds = plausible_upper_bounds.reshape((1, D))
    except ValueError as exc:
        raise ValueError(
            f"Bounds must match problem dimension D={D}."
        ) from exc

    # check that plausible bounds are finite
    if np.any(np.invert(np.isfinite(plausible_lower_bounds))) or np.any(
        np.invert(np.isfinite(plausible_upper_bounds))
    ):
        raise ValueError(
            "Plausible interval bounds PLB and PUB need to be finite."
        )

    # Preserve warnings for integer inputs after validation and widening.
    if integer_inputs[0]:
        logging.warning("Casting initial points to floating point.")
    if integer_inputs[1]:
        logging.warning("Casting lower bounds to floating point.")
    if integer_inputs[2]:
        logging.warning("Casting upper bounds to floating point.")
    if integer_inputs[3]:
        logging.warning("Casting plausible lower bounds to floating point.")
    if integer_inputs[4]:
        logging.warning("Casting plausible upper bounds to floating point.")

    # Fixed variables (all bounds equal) are not supported
    fixidx = (
        (lower_bounds == upper_bounds)
        & (upper_bounds == plausible_lower_bounds)
        & (plausible_lower_bounds == plausible_upper_bounds)
    )
    if np.any(fixidx):
        raise ValueError(
            "vbmc:FixedVariables VBMC does not support fixed\n"
            "            variables. Lower and upper bounds should be "
            "different."
        )

    # Test that plausible bounds are different
    if np.any(plausible_lower_bounds == plausible_upper_bounds):
        raise ValueError(
            "vbmc:MatchingPB:For all variables,\n"
            "            plausible lower and upper bounds need to be "
            "distinct."
        )

    # Check that all X0 are inside the bounds
    if np.any(x0 < lower_bounds) or np.any(x0 > upper_bounds):
        raise ValueError(
            "vbmc:InitialPointsNotInsideBounds: The starting\n"
            "            points X0 are not inside the provided hard bounds "
            "LB and UB."
        )

    lower_effective, upper_effective = _effective_bounds(
        lower_bounds, upper_bounds
    )

    # Fix when provided X0 are almost on the bounds -- move them inside
    if np.any(x0 < lower_effective) or np.any(x0 > upper_effective):
        logger.warning(
            "vbmc:InitialPointsTooClosePB: The starting points X0 are on "
            "or numerically too close to the hard bounds LB and UB. "
            "Moving the initial points more inside..."
        )
        x0 = np.maximum((np.minimum(x0, upper_effective)), lower_effective)

    # Test order of bounds (permissive)
    ordidx = (
        (lower_bounds <= plausible_lower_bounds)
        & (plausible_lower_bounds < plausible_upper_bounds)
        & (plausible_upper_bounds <= upper_bounds)
    )
    if np.any(np.invert(ordidx)):
        raise ValueError(
            "vbmc:StrictBounds: For each variable, hard and\n"
            "            plausible bounds should respect the ordering "
            "LB < PLB < PUB < UB."
        )

    # Test that plausible bounds are reasonably separated from hard bounds
    if np.any(lower_effective > plausible_lower_bounds) or np.any(
        plausible_upper_bounds > upper_effective
    ):
        logger.warning(
            "vbmc:TooCloseBounds: For each variable, hard "
            "and plausible bounds should not be too close. "
            "Moving plausible bounds."
        )
        plausible_lower_bounds = np.maximum(
            plausible_lower_bounds, lower_effective
        )
        plausible_upper_bounds = np.minimum(
            plausible_upper_bounds, upper_effective
        )

    # Check that all X0 are inside the plausible bounds,
    # move bounds otherwise
    if np.any(x0 <= plausible_lower_bounds) or np.any(
        x0 >= plausible_upper_bounds
    ):
        logger.warning(
            "vbmc:InitialPointsOutsidePB. The starting points X0"
            " are not inside the provided plausible bounds PLB and "
            "PUB. Expanding the plausible bounds..."
        )
        plausible_lower_bounds = np.minimum(plausible_lower_bounds, x0.min(0))
        plausible_upper_bounds = np.maximum(plausible_upper_bounds, x0.max(0))

    # Test order of bounds
    ordidx = (
        (lower_bounds < plausible_lower_bounds)
        & (plausible_lower_bounds < plausible_upper_bounds)
        & (plausible_upper_bounds < upper_bounds)
    )
    if np.any(np.invert(ordidx)):
        raise ValueError(
            "vbmc:StrictBounds: For each variable, hard and\n"
            "            plausible bounds should respect the ordering "
            "LB < PLB < PUB < UB."
        )

    # Check that variables are either bounded or unbounded
    # (not half-bounded)
    if np.any(
        (np.isfinite(lower_bounds) & np.isinf(upper_bounds))
        | (np.isinf(lower_bounds) & np.isfinite(upper_bounds))
    ):
        raise ValueError(
            "vbmc:HalfBounds: Each variable needs to be unbounded or\n"
            "            bounded. Variables bounded only below/above are "
            "not supported."
        )

    return (
        x0,
        lower_bounds,
        upper_bounds,
        plausible_lower_bounds,
        plausible_upper_bounds,
    )
