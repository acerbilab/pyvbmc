"""Compatibility helpers for the optional PyMC target adapter.

This module must remain importable without PyMC or PyTensor installed.
Imports of either dependency therefore happen only inside functions.
"""

from __future__ import annotations

import copy

import numpy as np

TESTED_RANGE = "PyMC 6.3.2, PyTensor 3.3.1–3.3.2 and ArviZ 1.3.0"


class UnsupportedModel(ValueError):
    """Raised when a PyMC model is outside the adapter's supported scope."""


def import_pymc():
    """Import and return the optional PyMC modules used by the adapter."""
    try:
        import pymc as pm
        import pytensor
        import pytensor.tensor as pt
    except ModuleNotFoundError as exc:
        missing = (exc.name or "").split(".", 1)[0]
        if missing in {"pymc", "pytensor"}:
            raise ImportError(
                "PyMC targets require the pymc extra; install "
                "pyvbmc[pymc] on Python 3.12+."
            ) from exc
        raise
    return pm, pytensor, pt


def _version_error(pm, capability: str, exc=None) -> ImportError:
    version = getattr(pm, "__version__", "unknown")
    error = ImportError(
        f"PyMC {version} does not provide the {capability} required by "
        f"PyVBMC. This adapter was tested with {TESTED_RANGE}."
    )
    if exc is not None:
        error.__cause__ = exc
    return error


def transform_classes() -> dict[str, type]:
    """Return the exact transform classes supported by the adapter."""
    pm, _, _ = import_pymc()
    try:
        from pymc.distributions.transforms import (
            Interval,
            IntervalTransform,
            LogOddsTransform,
            LogTransform,
        )
    except (AttributeError, ImportError) as exc:
        raise _version_error(pm, "supported transform classes", exc)
    return {
        "log": LogTransform,
        "logodds": LogOddsTransform,
        "interval": Interval,
        "interval_base": IntervalTransform,
    }


def default_transform(op, rv):
    """Return PyMC's registered default transform for a random variable."""
    pm, _, _ = import_pymc()
    try:
        from pymc.distributions.transforms import _default_transform
    except (AttributeError, ImportError) as exc:
        raise _version_error(pm, "default-transform registry", exc)
    return _default_transform(op, rv)


def remove_value_transforms(model, vars):
    """Remove selected transforms while preserving supported initial values."""
    pm, _, _ = import_pymc()
    try:
        from pymc.model.transform import remove_value_transforms as remove
    except (AttributeError, ImportError) as exc:
        raise _version_error(pm, "value-transform removal", exc)

    saved = dict(model.rvs_to_initial_values)
    by_name = {rv.name: copy.deepcopy(value) for rv, value in saved.items()}
    try:
        for rv in saved:
            model.rvs_to_initial_values[rv] = None
        try:
            transformed = remove(model, vars=vars)
        except (AttributeError, KeyError, TypeError) as exc:
            raise _version_error(pm, "value-transform removal", exc)
    finally:
        model.rvs_to_initial_values.update(saved)

    try:
        for name, value in by_name.items():
            transformed.set_initval(transformed[name], value)
    except (AttributeError, KeyError) as exc:
        raise _version_error(pm, "initial-value reconstruction", exc)
    return transformed


def clone_model(model):
    """Clone a PyMC model while preserving supported initial values.

    PyMC's ordinary model clone delegates directly to its graph round trip,
    which does not carry nondefault initial values.  Clearing and restoring
    them around that round trip also makes this helper suitable for target
    ``deepcopy`` implementations.
    """
    pm, _, _ = import_pymc()
    try:
        from pymc.model.fgraph import clone_model as clone
    except (AttributeError, ImportError) as exc:
        raise _version_error(pm, "model graph cloning", exc)

    saved = dict(model.rvs_to_initial_values)
    by_name = {rv.name: copy.deepcopy(value) for rv, value in saved.items()}
    try:
        for rv in saved:
            model.rvs_to_initial_values[rv] = None
        try:
            cloned = clone(model)
        except (AttributeError, KeyError, TypeError) as exc:
            raise _version_error(pm, "model graph cloning", exc)
    finally:
        model.rvs_to_initial_values.update(saved)

    try:
        for name, value in by_name.items():
            cloned.set_initval(cloned[name], value)
    except (AttributeError, KeyError) as exc:
        raise _version_error(pm, "initial-value reconstruction", exc)
    return cloned


def check_model(model) -> None:
    """Check the undocumented PyMC model capabilities used by the adapter."""
    pm, _, _ = import_pymc()
    required = (
        "free_RVs",
        "observed_RVs",
        "deterministics",
        "potentials",
        "data_vars",
        "rvs_to_values",
        "rvs_to_transforms",
        "rvs_to_initial_values",
        "named_vars_to_dims",
        "dim_lengths",
        "coords",
        "set_initval",
    )
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise _version_error(
            pm,
            "PyMC Model attributes " + ", ".join(missing),
        )


def _expit(value):
    value = np.asarray(value, dtype=np.float64)
    result = np.empty_like(value)
    nonnegative = value >= 0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-value[nonnegative]))
    exp_value = np.exp(value[~nonnegative])
    result[~nonnegative] = exp_value / (1.0 + exp_value)
    return result


def closed_form(kind, value, lower=None, upper=None):
    """Evaluate a supported transform's backward map with NumPy."""
    value = np.asarray(value, dtype=np.float64)
    if kind == "log":
        return np.exp(value)
    if kind == "logodds":
        return _expit(value)
    if kind not in {"interval", "interval_base"}:
        raise ValueError(f"Unknown transform kind {kind!r}.")

    lower = -np.inf if lower is None else lower
    upper = np.inf if upper is None else upper
    value, lower, upper = np.broadcast_arrays(
        value,
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
    )
    lower_finite = np.isfinite(lower)
    upper_finite = np.isfinite(upper)
    if np.any(~lower_finite & ~upper_finite):
        raise ValueError(
            "Every interval-transform coordinate needs a finite bound."
        )

    result = np.empty_like(value)
    two_sided = lower_finite & upper_finite
    lower_only = lower_finite & ~upper_finite
    upper_only = ~lower_finite & upper_finite
    result[two_sided] = lower[two_sided] + (
        upper[two_sided] - lower[two_sided]
    ) * _expit(value[two_sided])
    result[lower_only] = lower[lower_only] + np.exp(value[lower_only])
    result[upper_only] = upper[upper_only] - np.exp(value[upper_only])
    return result


def real_line_support_types() -> frozenset[type]:
    """Return exact operator types with documented whole-real support.

    The registry is intentionally narrow. PyMC documents Normal, StudentT,
    Cauchy, Laplace, Logistic, Gumbel and SkewNormal on the real line,
    MvNormal on :math:`R^k`, and MatrixNormal on :math:`R^{m x n}`.
    Improper flat distributions and user-defined operators are excluded.
    """
    pm, _, _ = import_pymc()
    try:
        from pymc.distributions.continuous import SkewNormalRV
        from pymc.distributions.multivariate import MatrixNormalRV
        from pytensor.tensor.random.basic import (
            CauchyRV,
            GumbelRV,
            LaplaceRV,
            LogisticRV,
            MvNormalRV,
            NormalRV,
            StudentTRV,
        )
    except (AttributeError, ImportError) as exc:
        raise _version_error(pm, "real-line distribution operators", exc)
    return frozenset(
        {
            NormalRV,
            StudentTRV,
            CauchyRV,
            LaplaceRV,
            LogisticRV,
            GumbelRV,
            MvNormalRV,
            SkewNormalRV,
            MatrixNormalRV,
        }
    )


def is_known_real_line(op) -> bool:
    """Return whether ``op`` has a recognized whole-real support."""
    return type(op) in real_line_support_types()


def snapshot_model(model):
    """Return a fixed numerical model and the source free-variable order."""
    from pyvbmc.pymc._snapshot import snapshot_model as make_snapshot

    return make_snapshot(model)


__all__ = [
    "TESTED_RANGE",
    "UnsupportedModel",
    "check_model",
    "clone_model",
    "closed_form",
    "default_transform",
    "import_pymc",
    "is_known_real_line",
    "real_line_support_types",
    "remove_value_transforms",
    "snapshot_model",
    "transform_classes",
]
