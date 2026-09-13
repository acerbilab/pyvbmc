"""Focused checks for the standard-VIQR sum over importance points."""

import copy
import math
from decimal import Decimal, localcontext

import numpy as np
import pytest

import pyvbmc.acquisition_functions.acq_fcn_viqr as viqr_module
from pyvbmc.acquisition_functions import AcqFcnVIQR
from pyvbmc.acquisition_functions.acq_fcn_viqr import _log_viqr_sum

from .test_acq_fcn_viqr_losses import _brute_force, _setup


def test_log_viqr_sum_small_tiny_and_zero_rows():
    """The direct identity retains tiny terms lost by ``1 - exp(-2a)``."""
    tiny = Decimal("1e-300")
    a = np.array(
        [
            [0.0, 0.25, 1.0],
            [0.0, float(tiny), 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    actual = _log_viqr_sum(a)
    moderate = math.log(
        2.0 * math.fsum(math.sinh(float(value)) for value in a[0])
    )
    with localcontext() as context:
        context.prec = 80
        tiny_expected = float((Decimal(2) * tiny).ln())

    assert actual.dtype == np.float64
    assert actual[0] == pytest.approx(moderate, rel=1e-15, abs=0.0)
    assert actual[1] == pytest.approx(tiny_expected, rel=0.0, abs=2e-13)
    assert actual[2] == -np.inf


def test_log_viqr_sum_guard_boundary_and_mixed_rows(monkeypatch):
    """Only wholly safe rows reach ``sinh``; large rows stay log-space."""
    n_terms = 4
    direct_limit = np.log(np.finfo(np.float64).max) - np.log(2 * n_terms)
    above_limit = np.nextafter(direct_limit, np.inf)
    a = np.array(
        [
            np.full(n_terms, direct_limit),
            [above_limit, 0.0, 0.0, 0.0],
            [1.0, 2.0, 1000.0, 3.0],
        ]
    )

    original_sinh = viqr_module.np.sinh
    direct_inputs = []

    def checked_sinh(values):
        direct_inputs.append(values.copy())
        assert np.all(values <= direct_limit)
        return original_sinh(values)

    monkeypatch.setattr(viqr_module.np, "sinh", checked_sinh)
    with np.errstate(all="ignore"):
        actual = _log_viqr_sum(a)

    assert len(direct_inputs) == 1
    assert direct_inputs[0].shape == (1, n_terms)
    expected = np.array([direct_limit + np.log(n_terms), above_limit, 1000.0])
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-13)


def test_log_viqr_sum_unsafe_arguments_retain_fallback_results():
    """Negative and nonfinite rows keep the former unsupported behavior."""
    a = np.array(
        [
            [-1.0, 2.0],
            [np.inf, 1.0],
            [np.nan, 1.0],
        ]
    )
    with np.errstate(all="ignore"):
        actual = _log_viqr_sum(a)
    assert np.all(np.isnan(actual))


@pytest.mark.parametrize("ns_gp", [1, 2])
@pytest.mark.parametrize("quantile", [0.6, 0.9])
def test_public_viqr_matches_definition_without_mutating_state(
    ns_gp, quantile
):
    """Batch and scalar calls match the full-covariance VIQR definition."""
    gp, vp, optim_state, X_eval = _setup(ns_gp)
    acq_fcn = AcqFcnVIQR(quantile=quantile)
    s2_a, tau2 = _brute_force(acq_fcn, gp, optim_state, X_eval)
    s_pred = np.sqrt(np.maximum(s2_a[None, :, :] - tau2, 0.0))
    per_sample = np.sum(2.0 * np.sinh(acq_fcn.u * s_pred), axis=1)
    expected = np.log(np.mean(per_sample, axis=1))

    active_is = optim_state["active_importance_sampling"]
    arrays_before = {
        key: value.copy()
        for key, value in active_is.items()
        if isinstance(value, np.ndarray)
    }
    rng_before = copy.deepcopy(vp.rng.bit_generator.state)
    X_before = X_eval.copy()

    batch = acq_fcn(X_eval, gp, vp, None, optim_state)
    pointwise = np.array(
        [acq_fcn(x, gp, vp, None, optim_state)[0] for x in X_eval]
    )

    assert batch.shape == (X_eval.shape[0],)
    assert batch.dtype == np.float64
    np.testing.assert_allclose(batch, expected, rtol=1e-8, atol=0.0)
    np.testing.assert_allclose(batch, pointwise, rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(X_eval, X_before)
    assert vp.rng.bit_generator.state == rng_before
    for key, before in arrays_before.items():
        np.testing.assert_array_equal(active_is[key], before)

    optim_state["lb_eps_orig"] = -1.5
    optim_state["ub_eps_orig"] = 1.5
    bounded = acq_fcn(X_eval, gp, vp, None, optim_state)
    assert np.all(np.isfinite(bounded[:2]))
    assert np.all(bounded[2:] == np.inf)
    assert vp.rng.bit_generator.state == rng_before
    for key, before in arrays_before.items():
        np.testing.assert_array_equal(active_is[key], before)
