"""Isolated objective variants for the bounded Phase 6 eta experiment.

This developer-only module builds private copies of
``pyvbmc.vbmc.variational_optimization``.  The copies keep the production
optimizer and all of its helpers, while changing only the two lines needed
to distinguish the experimental eta-bound treatments.

The experiment is pinned to the validated ``03650a2`` source.  A source
digest and exact replacement counts make drift fail loudly instead of
silently producing a fourth treatment.
"""

from __future__ import annotations

import hashlib
import types
from pathlib import Path

import numpy as np

ARMS = ("A", "B", "C")
ARM_LABELS = {
    "A": "no_eta_bound",
    "B": "matlab_raw_eta",
    "C": "current_relative_eta",
}

SOURCE_REVISION = "03650a226634525dfeb3d293f1540582094603ef"
SOURCE_SHA256 = (
    "118ca558acc7034368d5f8a1c31c8d40c3a73eabda25b3251cebfd27ace0280e"
)

_SOURCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "pyvbmc"
    / "vbmc"
    / "variational_optimization.py"
)
_ETA_VIEW_LINE = "        vp.eta = theta[-K:]\n"
_ETA_COPY_LINE = "        vp.eta = theta[-K:].copy()\n"
_VARIANTS: dict[str, types.ModuleType] = {}


def _normalize_arm(arm: str) -> str:
    """Return the canonical single-letter arm name."""
    candidate = str(arm).strip()
    upper = candidate.upper()
    if upper in ARMS:
        return upper
    for key, label in ARM_LABELS.items():
        if candidate.lower() == label:
            return key
    choices = ", ".join(f"{key} ({ARM_LABELS[key]})" for key in ARMS)
    raise ValueError(f"unknown eta-bound arm {arm!r}; choose {choices}")


def _read_pinned_source() -> str:
    """Read and validate the exact production source used by the experiment."""
    source = _SOURCE_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if digest != SOURCE_SHA256:
        raise RuntimeError(
            "variational_optimization.py differs from the validated Phase 6 "
            f"source at {SOURCE_REVISION}: expected {SOURCE_SHA256}, "
            f"got {digest}"
        )
    if source.count(_ETA_VIEW_LINE) != 1:
        raise RuntimeError(
            "the eta-view assignment is absent or ambiguous in the pinned "
            "source"
        )
    return source


def _install_no_eta_bound_wrapper(module: types.ModuleType) -> None:
    """Replace only the eta slice of ``_vp_bound_loss`` with no penalty."""
    module.__dict__["_vp_bound_loss_with_eta"] = module._vp_bound_loss
    wrapper_source = """
def _vp_bound_loss(
    vp,
    theta,
    theta_bnd,
    tol_con=1e-3,
    compute_grad=True,
):
    if not vp.optimize_weights:
        return _vp_bound_loss_with_eta(
            vp, theta, theta_bnd, tol_con=tol_con, compute_grad=compute_grad
        )
    bounds_without_eta = theta_bnd.copy()
    bounds_without_eta["lb"] = np.array(theta_bnd["lb"], copy=True)
    bounds_without_eta["ub"] = np.array(theta_bnd["ub"], copy=True)
    bounds_without_eta["lb"][-vp.K:] = -np.inf
    bounds_without_eta["ub"][-vp.K:] = np.inf
    return _vp_bound_loss_with_eta(
        vp,
        theta,
        bounds_without_eta,
        tol_con=tol_con,
        compute_grad=compute_grad,
    )
"""
    exec(compile(wrapper_source, str(_SOURCE_PATH), "exec"), module.__dict__)


def _build_variant(arm: str) -> types.ModuleType:
    source = _read_pinned_source()
    if arm in ("A", "B"):
        source = source.replace(_ETA_VIEW_LINE, _ETA_COPY_LINE, 1)

    name = f"pyvbmc.vbmc._eta_bound_variant_{arm.lower()}"
    module = types.ModuleType(name)
    module.__file__ = str(_SOURCE_PATH)
    module.__package__ = "pyvbmc.vbmc"
    exec(compile(source, str(_SOURCE_PATH), "exec"), module.__dict__)
    module.ETA_BOUND_ARM = arm
    module.ETA_BOUND_LABEL = ARM_LABELS[arm]
    module.ETA_BOUND_SOURCE_REVISION = SOURCE_REVISION

    if arm == "A":
        _install_no_eta_bound_wrapper(module)
    return module


def get_variant(arm: str) -> types.ModuleType:
    """Return an isolated module containing one eta-bound objective variant.

    ``A`` omits only eta from the soft-bound loss, ``B`` applies the MATLAB
    raw-eta bounds and consistent derivative, and ``C`` is an exact isolated
    copy of current production behavior, including caller-theta mutation.
    Module globals are private to each arm, so a replay harness may instrument
    helpers such as ``_sieve`` without patching the shared production module.
    """
    canonical = _normalize_arm(arm)
    if canonical not in _VARIANTS:
        _VARIANTS[canonical] = _build_variant(canonical)
    return _VARIANTS[canonical]


def _soft_bound_terms(x, lb, ub, tol_con):
    """Pure component-level equivalent of production ``_soft_bound_loss``."""
    x = np.asarray(x, dtype=float)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    scale = (ub - lb) * tol_con
    loss = np.zeros(x.shape, dtype=float)
    gradient = np.zeros(x.shape, dtype=float)

    lower = x < lb
    if np.any(lower):
        loss[lower] = 0.5 * ((lb[lower] - x[lower]) / scale[lower]) ** 2
        gradient[lower] = (x[lower] - lb[lower]) / scale[lower] ** 2
    upper = x > ub
    if np.any(upper):
        loss[upper] = 0.5 * ((x[upper] - ub[upper]) / scale[upper]) ** 2
        gradient[upper] = (x[upper] - ub[upper]) / scale[upper] ** 2
    return loss, gradient


def _pack_bound_parameters(theta, vp):
    """Pack theta exactly as production ``_vp_bound_loss`` does."""
    theta = np.asarray(theta)
    position = 0
    blocks = []

    if vp.optimize_mu:
        mu = theta[position : position + vp.D * vp.K]
        position += vp.D * vp.K
        blocks.append(mu.ravel())

    if vp.optimize_sigma:
        ln_sigma = theta[position : position + vp.K]
        position += vp.K
    else:
        ln_sigma = np.log(np.asarray(vp.sigma).ravel())

    if vp.optimize_lambd:
        ln_lambd = theta[position : position + vp.D]
        position += vp.D
    else:
        ln_lambd = np.log(np.asarray(vp.lambd).ravel())

    if vp.optimize_sigma or vp.optimize_lambd:
        ln_scale = ln_lambd.reshape(-1, 1) + ln_sigma.reshape(1, -1)
        blocks.append(ln_scale.ravel(order="F"))

    eta = np.empty(0, dtype=theta.dtype)
    if vp.optimize_weights:
        eta = np.array(theta[-vp.K :], copy=True)
        blocks.append(eta)

    if blocks:
        packed = np.concatenate(blocks)
    else:
        packed = np.empty(0, dtype=theta.dtype)
    return packed, eta


def _fold_bound_gradient(dpacked, theta, vp):
    """Fold packed scale gradients back to raw VP parameter blocks."""
    dpacked = np.asarray(dpacked)
    gradient = np.zeros(np.asarray(theta).shape, dtype=dpacked.dtype)
    packed_position = 0
    theta_position = 0

    if vp.optimize_mu:
        size = vp.D * vp.K
        gradient[theta_position : theta_position + size] = dpacked[
            packed_position : packed_position + size
        ]
        packed_position += size
        theta_position += size

    if vp.optimize_sigma or vp.optimize_lambd:
        size = vp.D * vp.K
        dscale = dpacked[packed_position : packed_position + size].reshape(
            (vp.D, vp.K), order="F"
        )
        packed_position += size
        if vp.optimize_sigma:
            gradient[theta_position : theta_position + vp.K] = np.sum(
                dscale, axis=0
            )
            theta_position += vp.K
        if vp.optimize_lambd:
            gradient[theta_position : theta_position + vp.D] = np.sum(
                dscale, axis=1
            )
            theta_position += vp.D

    if vp.optimize_weights:
        gradient[-vp.K :] = dpacked[-vp.K :]
    return gradient


def _stable_weights(raw_eta):
    relative_eta = raw_eta - np.max(raw_eta)
    exp_eta = np.exp(relative_eta)
    return relative_eta, exp_eta / np.sum(exp_eta)


def penalty_terms(theta, bounds, vp, arm: str) -> dict:
    """Decompose the arm's bound and fixed small-weight penalties.

    The helper is observational: it does not draw random numbers or mutate
    ``theta``, ``bounds``, or ``vp``.  For arm C, ``bound_gradient`` is the
    gradient currently supplied to the optimizer, while
    ``correct_bound_gradient`` includes the derivative of
    ``eta - max(eta)`` whenever the maximum is unique.  At an active tied
    maximum the transform is nondifferentiable; the corrected eta block is
    reported as NaN and ``correct_gradient_defined`` is false.
    """
    canonical = _normalize_arm(arm)
    theta_array = np.asarray(theta)
    if theta_array.ndim != 1:
        raise ValueError("theta must be a one-dimensional parameter vector")

    packed, raw_eta = _pack_bound_parameters(theta_array, vp)
    lb = np.asarray(bounds["lb"])
    ub = np.asarray(bounds["ub"])
    if packed.shape != lb.shape or packed.shape != ub.shape:
        raise ValueError(
            "bounds must have the same packed shape as the active VP "
            "parameters"
        )
    effective = np.array(packed, copy=True)
    eta_start = packed.size
    if vp.optimize_weights:
        eta_start -= vp.K
        if canonical == "C":
            effective[eta_start:] -= np.max(effective[eta_start:])

    component_loss, packed_gradient = _soft_bound_terms(
        effective, lb, ub, bounds["tol_con"]
    )
    if canonical == "A" and vp.optimize_weights:
        component_loss[eta_start:] = 0.0
        packed_gradient[eta_start:] = 0.0

    supplied_gradient = _fold_bound_gradient(packed_gradient, theta_array, vp)
    correct_packed_gradient = np.array(packed_gradient, copy=True)
    correct_gradient_defined = True
    max_tie_count = 0
    if canonical == "C" and vp.optimize_weights:
        eta_gradient = correct_packed_gradient[eta_start:]
        maxima = np.flatnonzero(raw_eta == np.max(raw_eta))
        max_tie_count = int(maxima.size)
        if maxima.size == 1:
            eta_gradient[maxima[0]] -= np.sum(eta_gradient)
        elif np.any(eta_gradient != 0.0):
            eta_gradient[:] = np.nan
            correct_gradient_defined = False
    correct_gradient = _fold_bound_gradient(
        correct_packed_gradient, theta_array, vp
    )

    if vp.optimize_weights:
        effective_eta = effective[eta_start:].copy()
        eta_lb = lb[eta_start:].copy()
        eta_ub = ub[eta_start:].copy()
        lower_violation = np.maximum(eta_lb - effective_eta, 0.0)
        upper_violation = np.maximum(effective_eta - eta_ub, 0.0)
        eta_loss_components = component_loss[eta_start:].copy()
    else:
        effective_eta = np.empty(0, dtype=float)
        eta_lb = np.empty(0, dtype=float)
        eta_ub = np.empty(0, dtype=float)
        lower_violation = np.empty(0, dtype=float)
        upper_violation = np.empty(0, dtype=float)
        eta_loss_components = np.empty(0, dtype=float)

    small_weight_loss = 0.0
    small_weight_gradient = np.zeros(theta_array.shape, dtype=float)
    weights = np.empty(0, dtype=float)
    if vp.optimize_weights:
        _, weights = _stable_weights(raw_eta)
        threshold = bounds["weight_threshold"]
        weight_penalty = bounds["weight_penalty"]
        small_weight_loss = float(
            np.sum(
                weights * (weights < threshold)
                + threshold * (weights >= threshold)
            )
            * weight_penalty
        )
        weight_gradient = weight_penalty * (weights < threshold)
        jacobian = np.diag(weights) - np.outer(weights, weights)
        small_weight_gradient[-vp.K :] = jacobian @ weight_gradient

    all_violations = np.concatenate((lower_violation, upper_violation))
    violation_max = (
        float(np.max(all_violations)) if all_violations.size else 0.0
    )
    return {
        "arm": canonical,
        "label": ARM_LABELS[canonical],
        "bound_loss": float(np.sum(component_loss)),
        "bound_gradient": supplied_gradient,
        "correct_bound_gradient": correct_gradient,
        "correct_gradient_defined": correct_gradient_defined,
        "non_eta_loss": float(np.sum(component_loss[:eta_start])),
        "eta_loss": float(np.sum(eta_loss_components)),
        "eta_loss_components": eta_loss_components,
        "raw_eta": raw_eta.copy(),
        "effective_eta": effective_eta,
        "eta_lb": eta_lb,
        "eta_ub": eta_ub,
        "eta_lower_violation": lower_violation,
        "eta_upper_violation": upper_violation,
        "eta_violation_count": int(
            np.count_nonzero(lower_violation)
            + np.count_nonzero(upper_violation)
        ),
        "eta_violation_max": violation_max,
        "eta_max_tie_count": max_tie_count,
        "weights": weights,
        "small_weight_loss": small_weight_loss,
        "small_weight_gradient": small_weight_gradient,
    }
