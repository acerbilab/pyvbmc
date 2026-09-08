"""The reduction losses of ``AcqFcnVIQR`` against exact identities.

The look-ahead variance reduction ``tau2`` that all members of the family
share is checked against a brute-force computation from the full GP
posterior covariance (``gp.predict_full``), and the ``"iqr_reduction"``
member against the algebraic identity that ties it to the default VIQR on
the same importance samples.
"""

import os

import gpyreg as gpr
import numpy as np
import pytest

from pyvbmc.acquisition_functions import AcqFcnVIQR
from pyvbmc.acquisition_functions.utilities import string_to_acq
from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.vbmc import active_importance_sampling
from pyvbmc.vbmc.options import Options

from ._regularization import check_variance_regularization

REDUCTIONS = ("iqr_reduction", "var_reduction", "sd_reduction")


def _setup(ns_gp, seed=0, D=2, N=25, Na=40):
    """A noisy GP with ``ns_gp`` hyperparameter samples, a two-component
    VP and the importance-sampling state of the VIQR family."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, size=(N, D))
    y = -0.5 * np.sum(X**2, axis=1, keepdims=True) + rng.normal(size=(N, 1))
    s2 = rng.uniform(0.5, 2.0, size=(N, 1))
    base = [
        0.0,
        0.2,  # log ell
        0.5,  # log sf
        -1.0,  # log base noise sd
        -(D / 2) * np.log(2 * np.pi),  # mean function maximum
        0.0,
        0.0,  # mean function location
        0.5,
        0.5,  # log mean function scale
    ]
    hyp = np.array(
        [base, np.array(base) + 0.1 * rng.normal(size=len(base))][:ns_gp]
    )
    gp = gpr.GP(
        D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(
            constant_add=True, user_provided_add=True
        ),
    )
    gp.update(X_new=X, y_new=y, s2_new=s2, hyp=hyp)

    vp = VariationalPosterior(D, 2)
    vp.mu = np.array([[-0.5, 0.7], [0.3, -0.4]])
    vp.sigma = np.array([[0.8, 1.2]])
    vp.w = np.array([[0.4, 0.6]])
    vp.rng = np.random.default_rng(seed + 1)

    optim_state = {
        "integer_vars": None,
        "lb_eps_orig": -np.inf,
        "ub_eps_orig": np.inf,
    }
    ln_ell = np.column_stack([p.hyp[:D] for p in gp.posteriors])
    optim_state["gp_length_scale"] = np.exp(ln_ell.mean(1))
    gp.temporary_data["X_rescaled"] = gp.X / optim_state["gp_length_scale"]
    cov_N = gp.covariance.hyperparameter_count(D)
    noise_N = gp.noise.hyperparameter_count()
    sn2_new = np.column_stack(
        [
            gp.noise.compute(
                p.hyp[cov_N : cov_N + noise_N], gp.X, gp.y, s2
            ).reshape(-1)
            for p in gp.posteriors
        ]
    )
    gp.temporary_data["sn2_new"] = sn2_new.mean(1)

    vbmc_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "..", "vbmc"
    )
    basic_path = os.path.join(
        vbmc_path, "option_configs", "basic_vbmc_options.ini"
    )
    advanced_path = os.path.join(
        vbmc_path, "option_configs", "advanced_vbmc_options.ini"
    )
    options = Options(
        basic_path,
        evaluation_parameters={"D": D},
        user_options={"active_importance_sampling_mcmc_samples": Na},
    )
    options.load_options_file(advanced_path, evaluation_parameters={"D": D})
    optim_state["active_importance_sampling"] = active_importance_sampling(
        vp, gp, AcqFcnVIQR(), options
    )
    X_eval = np.array([[0.0, 0.0], [1.0, -1.0], [-2.5, 2.0], [4.0, 4.0]])
    return gp, vp, optim_state, X_eval


def _brute_force(acq_fcn, gp, optim_state, X_eval):
    """Per candidate and hyperparameter sample: the current variance at the
    importance points and the look-ahead reduction, from the full
    posterior covariance."""
    Xa = optim_state["active_importance_sampling"]["X"]
    Na = Xa.shape[0]
    sn2 = acq_fcn._estimate_observation_noise(X_eval, gp, optim_state)
    ns_gp = len(gp.posteriors)
    s2_a = np.empty((Na, ns_gp))
    tau2 = np.empty((X_eval.shape[0], Na, ns_gp))
    for i, x in enumerate(X_eval):
        __, cov = gp.predict_full(np.vstack([Xa, x[None, :]]))
        for s in range(ns_gp):
            c = cov[:, :, s]
            s2_a[:, s] = np.diag(c)[:Na]
            tau2[i, :, s] = c[:Na, -1] ** 2 / (c[-1, -1] + sn2[i])
    return s2_a, tau2


@pytest.mark.parametrize("ns_gp", [1, 2])
def test_stored_importance_variance_matches_full_covariance(ns_gp):
    gp, vp, optim_state, X_eval = _setup(ns_gp)
    s2_a, __ = _brute_force(AcqFcnVIQR(), gp, optim_state, X_eval)
    assert np.allclose(
        optim_state["active_importance_sampling"]["f_s2"], s2_a, rtol=1e-10
    )


@pytest.mark.parametrize("ns_gp", [1, 2])
@pytest.mark.parametrize("loss", REDUCTIONS)
def test_reduction_losses_match_brute_force(ns_gp, loss):
    gp, vp, optim_state, X_eval = _setup(ns_gp)
    acq_fcn = AcqFcnVIQR(loss=loss)
    s2_a, tau2 = _brute_force(acq_fcn, gp, optim_state, X_eval)
    s_a = np.sqrt(s2_a)[None, :, :]
    s_pred = np.sqrt(np.maximum(s2_a[None, :, :] - tau2, 0.0))
    u = acq_fcn.u
    if loss == "var_reduction":
        term = tau2
    elif loss == "sd_reduction":
        term = s_a - s_pred
    else:
        term = np.sinh(u * s_a) - np.sinh(u * s_pred)
    # Uniform weights, normalized over samples and hyperparameters; the
    # reductions are averaged over the hyperparameter samples.
    w = np.exp(optim_state["active_importance_sampling"]["ln_weights"])
    assert np.allclose(w, 1 / w.size)
    expected = -np.log(np.mean(np.sum(term * w.T[None, :, :], axis=1), 1))
    actual = acq_fcn(X_eval, gp, vp, None, optim_state)
    assert actual.shape == (X_eval.shape[0],)
    assert np.all(np.isfinite(actual))
    assert np.allclose(actual, expected, rtol=1e-8, atol=0.0)


def test_iqr_reduction_identity_with_viqr():
    """exp(-a_red) * Na + exp(a_iqr) / 2 = sum_a sinh(u s_a), at Ns = 1.

    VIQR carries the factor 2 of the interquantile range (``2 sinh``); the
    reduction acquisition is a weighted mean of the reductions.
    """
    gp, vp, optim_state, X_eval = _setup(1)
    viqr = AcqFcnVIQR()
    red = AcqFcnVIQR(loss="iqr_reduction")
    a_iqr = viqr(X_eval, gp, vp, None, optim_state)
    a_red = red(X_eval, gp, vp, None, optim_state)
    Na = optim_state["active_importance_sampling"]["X"].shape[0]
    s_a = np.sqrt(optim_state["active_importance_sampling"]["f_s2"][:, 0])
    total = np.sum(np.sinh(viqr.u * s_a))
    assert np.allclose(
        np.exp(-a_red) * Na + np.exp(a_iqr) / 2, total, rtol=1e-8, atol=0.0
    )
    # And the same minimizer over the candidate set.
    assert np.argmin(a_iqr) == np.argmin(a_red)


@pytest.mark.parametrize("loss", REDUCTIONS)
def test_no_reduction_is_worst(loss):
    """A candidate far from the importance points reduces nothing: the
    acquisition is +inf there, the worst value for a minimizer."""
    gp, vp, optim_state, X_eval = _setup(1)
    far = np.array([[60.0, -60.0]])
    acq_fcn = AcqFcnVIQR(loss=loss)
    # Neighbouring points keep the far one from being clamped by bounds.
    a = acq_fcn(np.vstack([X_eval[:1], far]), gp, vp, None, optim_state)
    assert np.isfinite(a[0])
    assert a[1] == np.inf


@pytest.mark.parametrize("ns_gp", [1, 2])
@pytest.mark.parametrize("M", [1, 3])
@pytest.mark.parametrize("loss", REDUCTIONS)
def test_reduction_variance_regularization_batch_matches_pointwise(
    ns_gp, M, loss
):
    check_variance_regularization(AcqFcnVIQR(loss=loss), ns_gp, M)


def test_loss_argument():
    assert AcqFcnVIQR().loss == "iqr"
    assert AcqFcnVIQR().acq_info["loss"] == "iqr"
    for loss in REDUCTIONS:
        acq = string_to_acq(f'AcqFcnVIQR(loss="{loss}")')
        assert acq.loss == loss
        assert acq.acq_info["loss"] == loss
        acq = string_to_acq(f'AcqFcnVIQR(0.6, "{loss}")')
        assert acq.loss == loss
    with pytest.raises(ValueError):
        AcqFcnVIQR(loss="iqr_squared")


def test_instance_without_loss_attribute_is_viqr():
    """Instances pickled before the argument existed."""
    gp, vp, optim_state, X_eval = _setup(1)
    fresh = AcqFcnVIQR()
    legacy = AcqFcnVIQR()
    del legacy.loss
    a_fresh = fresh(X_eval, gp, vp, None, optim_state)
    a_legacy = legacy(X_eval, gp, vp, None, optim_state)
    np.testing.assert_array_equal(a_fresh, a_legacy)
