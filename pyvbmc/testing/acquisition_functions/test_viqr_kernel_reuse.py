"""Compatibility and lifetime checks for VIQR prediction-kernel reuse."""

import copy
import types

import gpyreg as gpr
import numpy as np
import pytest

import pyvbmc.acquisition_functions.acq_fcn_viqr as viqr_module
from pyvbmc.acquisition_functions import AcqFcnVIQR

from .test_acq_fcn_viqr_losses import _setup


def _active_arrays(optim_state):
    return {
        key: value.copy()
        for key, value in optim_state["active_importance_sampling"].items()
        if isinstance(value, np.ndarray)
    }


def _assert_active_arrays_unchanged(optim_state, before):
    active_is = optim_state["active_importance_sampling"]
    for key, expected in before.items():
        np.testing.assert_array_equal(active_is[key], expected)


def _as_non_cholesky(gp, optim_state):
    """Convert equivalent posterior factors and the VIQR helper matrices."""
    active_is = optim_state["active_importance_sampling"]
    for s, posterior in enumerate(gp.posteriors):
        assert posterior.L_chol
        scaled_inverse = posterior.sW * np.linalg.inv(posterior.L)
        posterior.L = -(scaled_inverse @ scaled_inverse.T)
        posterior.L_chol = False
        active_is["C_tmp"][s] = posterior.L @ active_is["K_Xa_X"][s].T


@pytest.mark.parametrize(
    ("ns_gp", "candidate_count", "quantile"),
    [(1, 1, 0.6), (2, 6, 0.75), (2, 8192, 0.9)],
)
def test_fast_and_fallback_are_equivalent_and_immutable(
    monkeypatch, ns_gp, candidate_count, quantile
):
    """The reused path matches ordinary VIQR through sieve-sized batches."""
    gp, vp, optim_state, X_base = _setup(ns_gp)
    rng = np.random.default_rng(916 + candidate_count)
    if candidate_count <= X_base.shape[0]:
        X_eval = X_base[:candidate_count].copy()
    else:
        X_eval = rng.uniform(-4.0, 4.0, size=(candidate_count, gp.D))
    acq = AcqFcnVIQR(quantile=quantile)
    X_before = X_eval.copy()
    state_before = _active_arrays(optim_state)
    rng_before = copy.deepcopy(vp.rng.bit_generator.state)
    acq_state_before = copy.deepcopy(vars(acq))
    gp_keys_before = set(vars(gp))

    fast = acq(X_eval, gp, vp, None, optim_state)
    monkeypatch.setattr(viqr_module, "_VIQR_KERNEL_CACHE_MAX_BYTES", 0)
    fallback = acq(X_eval, gp, vp, None, optim_state)

    np.testing.assert_array_equal(fast, fallback)
    assert fast.shape == (candidate_count,)
    assert fast.dtype == np.float64
    np.testing.assert_array_equal(X_eval, X_before)
    _assert_active_arrays_unchanged(optim_state, state_before)
    assert vp.rng.bit_generator.state == rng_before
    assert vars(acq) == acq_state_before
    assert set(vars(gp)) == gp_keys_before
    assert "cross_covariance" not in vars(gp)
    assert "context" not in vars(acq)


@pytest.mark.parametrize(
    ("offset", "uses_context"), [(-1, False), (0, True), (1, True)]
)
def test_payload_cap_edges(monkeypatch, offset, uses_context):
    """A payload at the cap is admitted; one byte over it falls back."""
    gp, __, __, X_eval = _setup(2)
    X_eval = X_eval[:3]
    payload = 8 * len(gp.X) * len(X_eval) * len(gp.posteriors)
    monkeypatch.setattr(
        viqr_module, "_VIQR_KERNEL_CACHE_MAX_BYTES", payload + offset
    )

    __, __, context = AcqFcnVIQR()._predict_with_context(X_eval, gp)

    assert (context is not None) is uses_context
    if context is not None:
        assert isinstance(context, tuple)
        assert len(context) == len(gp.posteriors)
        assert all(K.shape == (len(gp.X), len(X_eval)) for K in context)


def test_payload_size_uses_python_integer_arithmetic(monkeypatch):
    """Huge logical shapes reject reuse without fixed-width overflow."""
    gp, __, __, __ = _setup(2)
    monkeypatch.setattr(
        viqr_module, "_VIQR_KERNEL_CACHE_MAX_BYTES", 2**63 - 1
    )

    class ShapeOnly:
        shape = (2**62, gp.D)

    assert not AcqFcnVIQR()._can_reuse_prediction_kernel(ShapeOnly(), gp)


@pytest.mark.parametrize("ns_gp", [1, 2])
def test_fast_and_fallback_match_both_factor_representations(
    monkeypatch, ns_gp
):
    gp, vp, optim_state, X_eval = _setup(ns_gp)
    _as_non_cholesky(gp, optim_state)
    assert all(not posterior.L_chol for posterior in gp.posteriors)
    acq = AcqFcnVIQR()

    fast = acq(X_eval, gp, vp, None, optim_state)
    monkeypatch.setattr(viqr_module, "_VIQR_KERNEL_CACHE_MAX_BYTES", 0)
    fallback = acq(X_eval, gp, vp, None, optim_state)

    np.testing.assert_array_equal(fast, fallback)


def test_reduction_loss_and_custom_viqr_subclass_fall_back():
    gp, vp, optim_state, X_eval = _setup(2)

    acq = AcqFcnVIQR(loss="iqr_reduction")
    assert acq._predict_with_context(X_eval, gp)[2] is None

    calls = []

    class CustomVIQR(AcqFcnVIQR):
        def _compute_acquisition_function(self, *args):
            calls.append(args[0].copy())
            return super()._compute_acquisition_function(*args)

    custom = CustomVIQR()
    assert custom._predict_with_context(X_eval, gp)[2] is None
    actual = custom(X_eval, gp, vp, None, optim_state)
    assert calls and actual.shape == (len(X_eval),)


@pytest.mark.parametrize("scope", ["class", "instance"])
def test_gp_prediction_overrides_are_called_on_fallback(monkeypatch, scope):
    gp, vp, optim_state, X_eval = _setup(1)
    original = viqr_module._ORIGINAL_GP_PREDICT
    calls = []

    def overridden(self, *args, **kwargs):
        calls.append(kwargs.copy())
        return original(self, *args, **kwargs)

    if scope == "class":
        monkeypatch.setattr(gpr.GP, "predict", overridden)
    else:
        monkeypatch.setattr(
            gp, "predict", types.MethodType(overridden, gp), raising=False
        )

    acq = AcqFcnVIQR()
    assert acq._predict_with_context(X_eval, gp)[2] is None
    actual = acq(X_eval, gp, vp, None, optim_state)

    assert actual.shape == (len(X_eval),)
    assert calls
    assert all("return_cross_covariance" not in kwargs for kwargs in calls)


@pytest.mark.parametrize("scope", ["class", "instance", "subclass"])
def test_kernel_compute_overrides_are_called_on_fallback(monkeypatch, scope):
    gp, vp, optim_state, X_eval = _setup(1)
    original = viqr_module._ORIGINAL_SE_COMPUTE
    calls = []

    def overridden(self, *args, **kwargs):
        calls.append((args, kwargs.copy()))
        return original(self, *args, **kwargs)

    if scope == "class":
        monkeypatch.setattr(
            gpr.covariance_functions.SquaredExponential,
            "compute",
            overridden,
        )
    elif scope == "instance":
        monkeypatch.setattr(
            gp.covariance,
            "compute",
            types.MethodType(overridden, gp.covariance),
            raising=False,
        )
    else:

        class CustomSquaredExponential(
            gpr.covariance_functions.SquaredExponential
        ):
            compute = overridden

        gp.covariance = CustomSquaredExponential()

    acq = AcqFcnVIQR()
    assert acq._predict_with_context(X_eval, gp)[2] is None
    actual = acq(X_eval, gp, vp, None, optim_state)

    assert actual.shape == (len(X_eval),)
    assert calls


@pytest.mark.parametrize("scope", ["class", "instance"])
def test_exact_viqr_acquisition_override_is_called(monkeypatch, scope):
    gp, vp, optim_state, X_eval = _setup(1)
    acq = AcqFcnVIQR()
    calls = []

    def overridden(self, Xs, *args):
        calls.append(Xs.copy())
        return np.full(Xs.shape[0], 17.0)

    if scope == "class":
        monkeypatch.setattr(
            AcqFcnVIQR, "_compute_acquisition_function", overridden
        )
    else:
        monkeypatch.setattr(
            acq,
            "_compute_acquisition_function",
            types.MethodType(overridden, acq),
        )

    assert acq._predict_with_context(X_eval, gp)[2] is None
    actual = acq(X_eval, gp, vp, None, optim_state)
    np.testing.assert_array_equal(actual, np.full(len(X_eval), 17.0))
    assert len(calls) == 1


def test_legacy_viqr_without_loss_uses_kernel_context():
    gp, __, __, X_eval = _setup(1)
    acq = AcqFcnVIQR()
    del acq.loss

    __, __, context = acq._predict_with_context(X_eval, gp)

    assert context is not None
    assert len(context) == 1


def test_integer_mapping_bounds_and_penalty_match_fallback(monkeypatch):
    gp, vp, optim_state, __ = _setup(2)
    optim_state["integer_vars"] = np.array([True, False])
    optim_state["lb_eps_orig"] = -1.5
    optim_state["ub_eps_orig"] = 1.5
    optim_state["variance_regularized_acq_fcn"] = True
    optim_state["tol_gp_var"] = 2.0
    X_eval = np.array([[0.49, 0.2], [1.51, -0.5], [-0.51, 2.0]])
    acq = AcqFcnVIQR()

    fast_input = X_eval.copy()
    fast = acq(fast_input, gp, vp, None, optim_state)
    monkeypatch.setattr(viqr_module, "_VIQR_KERNEL_CACHE_MAX_BYTES", 0)
    fallback_input = X_eval.copy()
    fallback = acq(fallback_input, gp, vp, None, optim_state)

    np.testing.assert_array_equal(fast, fallback)
    np.testing.assert_array_equal(fast_input, fallback_input)
    rounded = vp.parameter_transformer.inverse(fast_input)
    np.testing.assert_array_equal(rounded[:, 0], np.around(rounded[:, 0]))
    assert np.any(np.isfinite(fast))
    assert np.any(np.isinf(fast))


def test_repeated_calls_use_current_candidates_and_gp_state():
    gp, __, __, X_eval = _setup(2)
    acq = AcqFcnVIQR()

    __, __, first = acq._predict_with_context(X_eval[:1], gp)
    gp.X = gp.X.copy()
    gp.X[0] += 0.25
    __, __, second = acq._predict_with_context(X_eval[:3], gp)

    assert all(K.shape == (len(gp.X), 1) for K in first)
    assert all(K.shape == (len(gp.X), 3) for K in second)
    assert all(not np.shares_memory(a, b) for a, b in zip(first, second))
    cov_N = gp.covariance.hyperparameter_count(gp.D)
    for s, K in enumerate(second):
        expected = gp.covariance.compute(
            gp.posteriors[s].hyp[:cov_N], gp.X, X_eval[:3]
        )
        np.testing.assert_array_equal(K, expected)
    assert "cross_covariance" not in vars(gp)
    assert "context" not in vars(acq)


def test_untrained_and_empty_states_do_not_request_context():
    gp, __, __, X_eval = _setup(1)
    acq = AcqFcnVIQR()
    gp.y = None
    assert acq._predict_with_context(X_eval, gp)[2] is None

    gp, __, __, __ = _setup(1)
    assert not acq._can_reuse_prediction_kernel(np.empty((0, gp.D)), gp)


def test_context_is_not_retained_when_acquisition_raises(monkeypatch):
    gp, vp, optim_state, X_eval = _setup(1)
    acq = AcqFcnVIQR()
    seen = []

    def raising(self, *args, cross_covariance=None, **kwargs):
        seen.append(tuple(K.shape for K in cross_covariance))
        raise RuntimeError("deliberate acquisition failure")

    monkeypatch.setattr(acq, "_compute_viqr", types.MethodType(raising, acq))
    acq_keys = set(vars(acq))
    gp_keys = set(vars(gp))
    state_keys = set(optim_state)

    with pytest.raises(RuntimeError, match="deliberate acquisition failure"):
        acq(X_eval, gp, vp, None, optim_state)

    assert seen == [((len(gp.X), len(X_eval)),)]
    assert set(vars(acq)) == acq_keys
    assert set(vars(gp)) == gp_keys
    assert set(optim_state) == state_keys
    assert "context" not in vars(acq)
    assert "cross_covariance" not in vars(gp)
