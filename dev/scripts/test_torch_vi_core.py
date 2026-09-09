"""Focused gates for the Stage 4 Torch variational-objective prototype.

Run explicitly; this file is outside the package's default pytest discovery.
"""

from __future__ import annotations

import copy
import itertools
import math
import sys
from pathlib import Path

import gpyreg as gpr
import numpy as np
import pytest
import scipy.linalg as spla

torch = pytest.importorskip("torch")

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from torch_vi_core import (  # noqa: E402
    EntropyAnchors,
    TorchBounds,
    TorchGP,
    TorchVPTemplate,
    decode_theta,
    entropy_lower_bound,
    entropy_mc,
    gp_log_joint,
    neg_elcbo,
    parameter_count,
    soft_bound_loss,
    vp_bound_loss,
)

from pyvbmc.entropy import entlb_vbmc, entmc_vbmc  # noqa: E402
from pyvbmc.testing.oracles._oracles import compare, format_rows  # noqa: E402
from pyvbmc.testing.oracles._state import (  # noqa: E402
    build_state,
    load_snapshot,
    snapshot_names,
)
from pyvbmc.variational_posterior import VariationalPosterior  # noqa: E402
from pyvbmc.vbmc.variational_optimization import (  # noqa: E402
    _gp_log_joint,
    _neg_elcbo,
    _soft_bound_loss,
    _vp_bound_loss,
)

FIXTURES = ROOT / "pyvbmc" / "testing" / "oracles" / "fixtures"
NAMES = snapshot_names(FIXTURES)
SEED = 20260904
SOLVE_RTOL = 1e-6
SOLVE_ATOL = 1e-10
VAR_RTOL = 1e-3
VAR_ATOL = 1e-8
EXACT_RTOL = 1e-10
EXACT_ATOL = 1e-12


@pytest.fixture(scope="module")
def snapshots():
    assert NAMES
    return {name: load_snapshot(FIXTURES / name) for name in NAMES}


def _prepared_vp(state):
    vp = copy.deepcopy(state["vp"])
    theta = vp.get_parameters(raw_flag=True)
    return vp, theta


def _numpy_vp(template, theta):
    vp = copy.deepcopy(template)
    vp.set_parameters(np.asarray(theta), raw_flag=True)
    if vp.optimize_weights:
        eta = np.asarray(theta[-vp.K :]).copy()
        eta -= np.max(eta)
        vp.eta = eta.reshape(1, -1)
    return vp


def _epsilon(seed, K, Ns, D):
    return np.random.default_rng(seed).standard_normal(
        (K, int(math.ceil(Ns / 2)), D)
    )


def _tensor_array(value):
    return value.detach().cpu().numpy()


def _assert_close(actual, expected, *, rtol, atol):
    output = (
        _tensor_array(actual) if isinstance(actual, torch.Tensor) else actual
    )
    rows = compare(
        {"value": np.asarray(expected)},
        {"value": np.asarray(output)},
        rtol,
        atol,
    )
    assert rows[0][3], "\n" + format_rows(rows)


def _torch_grad(fn, theta):
    theta_t = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    value = fn(theta_t)
    grad = torch.autograd.grad(value, theta_t)[0]
    return value.detach(), grad.detach()


def _central_difference(fn, x, step=2e-6):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    for i in range(x.size):
        hi = step * max(1.0, abs(x[i]))
        xp, xm = x.copy(), x.copy()
        xp[i] += hi
        xm[i] -= hi
        out[i] = (fn(xp) - fn(xm)) / (2.0 * hi)
    return out


@pytest.mark.parametrize("name", NAMES)
def test_all_oracle_states_gp_entropy_and_objective(snapshots, name):
    """All eight committed states gate every load-bearing core output."""
    state = build_state(snapshots[name])
    vp, theta = _prepared_vp(state)
    tvp = TorchVPTemplate.from_vp(vp)
    tgp = TorchGP.from_gp(state["gp"])
    flags = (
        vp.optimize_mu,
        vp.optimize_sigma,
        vp.optimize_lambd,
        vp.optimize_weights,
    )

    # Averaged and per-hyperparameter GP expectations and gradients.
    vp_np = _numpy_vp(vp, theta)
    G_np, dG_np, *_ = _gp_log_joint(
        vp_np, state["gp"], flags, True, True, False
    )
    G_t, dG_t = _torch_grad(
        lambda th: gp_log_joint(decode_theta(th, tvp), tgp)[0], theta
    )
    _assert_close(G_t, G_np, rtol=SOLVE_RTOL, atol=SOLVE_ATOL)
    _assert_close(dG_t, dG_np, rtol=SOLVE_RTOL, atol=SOLVE_ATOL)

    Gs_np, dGs_np, *_ = _gp_log_joint(
        vp_np, state["gp"], flags, False, True, False
    )
    theta_t = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    jac = torch.autograd.functional.jacobian(
        lambda th: gp_log_joint(decode_theta(th, tvp), tgp, avg_flag=False)[0],
        theta_t,
    )
    Gs_t = gp_log_joint(decode_theta(theta_t, tvp), tgp, avg_flag=False)[0]
    _assert_close(Gs_t, Gs_np, rtol=SOLVE_RTOL, atol=SOLVE_ATOL)
    jacobian_blocks = jac if tgp.Ns == 1 else jac.transpose(0, 1)
    _assert_close(jacobian_blocks, dGs_np, rtol=SOLVE_RTOL, atol=SOLVE_ATOL)

    Gv_np, _, var_np, _, varss_np, I_np, J_np = _gp_log_joint(
        vp_np, state["gp"], False, True, True, True, True
    )
    decoded = decode_theta(torch.tensor(theta, dtype=torch.float64), tvp)
    diagnostics = {}
    Gv_t, var_t, varss_t, I_t, J_t = gp_log_joint(
        decoded,
        tgp,
        compute_var=True,
        separate_K=True,
        diagnostics=diagnostics,
    )
    _assert_close(Gv_t, Gv_np, rtol=SOLVE_RTOL, atol=SOLVE_ATOL)
    _assert_close(I_t, I_np, rtol=SOLVE_RTOL, atol=SOLVE_ATOL)
    _assert_close(var_t, var_np, rtol=VAR_RTOL, atol=VAR_ATOL)
    _assert_close(varss_t, varss_np, rtol=VAR_RTOL, atol=VAR_ATOL)
    _assert_close(J_t, J_np, rtol=VAR_RTOL, atol=VAR_ATOL)
    assert torch.equal(diagnostics["J_raw"], J_t)
    assert torch.all(
        diagnostics["J_diag_clamped"] >= torch.finfo(torch.float64).eps
    )
    assert torch.all(
        diagnostics["varG_clamped"] >= torch.finfo(torch.float64).eps
    )
    assert torch.equal(
        diagnostics["J_diag_clamped_mask"],
        diagnostics["J_diag_raw"] < torch.finfo(torch.float64).eps,
    )
    assert torch.equal(
        diagnostics["varG_clamped_mask"],
        diagnostics["varG_preclamp"] < torch.finfo(torch.float64).eps,
    )

    # Deterministic entropy and the baseline pathwise MC estimator.
    H0_np, dH0_np = entlb_vbmc(vp_np, flags, True)
    H0_t, dH0_t = _torch_grad(
        lambda th: entropy_lower_bound(decode_theta(th, tvp)), theta
    )
    _assert_close(H0_t, H0_np, rtol=EXACT_RTOL, atol=EXACT_ATOL)
    _assert_close(dH0_t, dH0_np, rtol=EXACT_RTOL, atol=EXACT_ATOL)

    Ns = int(math.ceil(state["options"].eval("ns_ent", {"K": vp.K}) / vp.K))
    eps = _epsilon(SEED, vp.K, Ns, vp.D)
    H_np, dH_np = entmc_vbmc(
        vp_np, Ns, flags, True, rng=np.random.default_rng(SEED)
    )
    H_t, dH_t = _torch_grad(
        lambda th: entropy_mc(decode_theta(th, tvp), Ns, eps), theta
    )
    _assert_close(H_t, H_np, rtol=EXACT_RTOL, atol=EXACT_ATOL)
    _assert_close(dH_t, dH_np, rtol=EXACT_RTOL, atol=EXACT_ATOL)

    # The bounded optimizer objective, including its capped weight penalty.
    vp_for_bounds = copy.deepcopy(vp)
    theta_bnd = vp_for_bounds.get_bounds(state["gp"].X, state["options"], vp.K)
    np_obj_vp = copy.deepcopy(vp)
    np_obj_vp.rng = np.random.default_rng(SEED)
    expected = _neg_elcbo(
        theta.copy(),
        state["gp"],
        np_obj_vp,
        0.0,
        Ns,
        True,
        False,
        theta_bnd,
    )
    actual = neg_elcbo(
        theta,
        tgp,
        tvp,
        0.0,
        Ns,
        True,
        False,
        TorchBounds.from_dict(theta_bnd),
        False,
        epsilon=eps,
    )
    for i in (0, 1, 2):
        _assert_close(actual[i], expected[i], rtol=SOLVE_RTOL, atol=SOLVE_ATOL)
    _assert_close(actual[3], expected[3], rtol=EXACT_RTOL, atol=EXACT_ATOL)

    # Value-only full variance and per-component pieces at the same bounded
    # sample count (fine scoring uses the same kernel with a larger epsilon).
    np_full_vp = copy.deepcopy(vp)
    np_full_vp.rng = np.random.default_rng(SEED)
    expected_full = _neg_elcbo(
        theta.copy(),
        state["gp"],
        np_full_vp,
        0.0,
        Ns,
        False,
        True,
        None,
        0.0,
        True,
    )
    actual_full = neg_elcbo(
        theta,
        tgp,
        tvp,
        0.0,
        Ns,
        False,
        True,
        None,
        True,
        epsilon=eps,
    )
    solve_indices = (0, 2, 9)
    variance_indices = (4, 6, 7, 8, 10)
    for i in solve_indices:
        _assert_close(
            actual_full[i], expected_full[i], rtol=SOLVE_RTOL, atol=SOLVE_ATOL
        )
    _assert_close(
        actual_full[3], expected_full[3], rtol=EXACT_RTOL, atol=EXACT_ATOL
    )
    for i in variance_indices:
        if actual_full[i] is None or expected_full[i] is None:
            assert actual_full[i] is expected_full[i] is None
        else:
            _assert_close(
                actual_full[i], expected_full[i], rtol=VAR_RTOL, atol=VAR_ATOL
            )


def test_decode_all_masks_fortran_order_and_no_mutation():
    rng = np.random.default_rng(10)
    D, K = 3, 2
    original = VariationalPosterior(D, K, rng=1)
    original.mu = rng.normal(size=(D, K))
    original.sigma = np.exp(rng.normal(size=(1, K)))
    original.lambd = np.exp(rng.normal(size=(D, 1)))
    eta = rng.normal(size=(1, K))
    original.eta = eta
    original.w = np.exp(eta) / np.exp(eta).sum()

    for flags in itertools.product((False, True), repeat=4):
        base = copy.deepcopy(original)
        (
            base.optimize_mu,
            base.optimize_sigma,
            base.optimize_lambd,
            base.optimize_weights,
        ) = flags
        theta = base.get_parameters()
        before = theta.copy()
        expected = _numpy_vp(base, theta)
        template = TorchVPTemplate.from_vp(base)
        decoded = decode_theta(
            torch.tensor(theta, dtype=torch.float64), template
        )
        assert parameter_count(template) == theta.size
        assert np.array_equal(theta, before)
        _assert_close(
            decoded.mu,
            np.asarray(expected.mu).reshape(D, K),
            rtol=0,
            atol=1e-14,
        )
        _assert_close(
            decoded.sigma,
            np.asarray(expected.sigma).reshape(K),
            rtol=0,
            atol=1e-14,
        )
        _assert_close(
            decoded.lambd,
            np.asarray(expected.lambd).reshape(D),
            rtol=0,
            atol=1e-14,
        )
        _assert_close(
            decoded.w, np.asarray(expected.w).reshape(K), rtol=0, atol=1e-14
        )

    # Explicitly pin Fortran order with non-symmetric values.
    vp = VariationalPosterior(D, K, rng=1)
    vp.optimize_sigma = vp.optimize_lambd = vp.optimize_weights = False
    theta = np.arange(1, D * K + 1, dtype=float)
    decoded = decode_theta(theta, TorchVPTemplate.from_vp(vp))
    np.testing.assert_array_equal(
        _tensor_array(decoded.mu), np.reshape(theta, (D, K), order="F")
    )


def test_gauge_and_eta_shifts_preserve_values_and_gradients():
    D, K = 3, 2
    vp = VariationalPosterior(D, K, rng=1)
    vp.mu = np.arange(D * K, dtype=float).reshape(D, K) / 7
    vp.sigma = np.array([[0.7, 1.3]])
    vp.lambd = np.array([[0.8], [1.1], [1.4]])
    theta = vp.get_parameters()
    template = TorchVPTemplate.from_vp(vp)

    def objective(th):
        return entropy_lower_bound(decode_theta(th, template))

    value, grad = _torch_grad(objective, theta)
    shifted = theta.copy()
    s0 = D * K
    shifted[s0 : s0 + K] += 3.0
    shifted[s0 + K : s0 + K + D] -= 3.0
    shifted[-K:] += 20.0
    value2, grad2 = _torch_grad(objective, shifted)
    _assert_close(value2, value, rtol=0, atol=2e-13)
    _assert_close(grad2, grad, rtol=0, atol=2e-12)


def test_deterministic_entropy_central_finite_difference():
    D, K = 2, 3
    rng = np.random.default_rng(4)
    vp = VariationalPosterior(D, K, rng=1)
    vp.mu = rng.normal(size=(D, K))
    vp.sigma = np.exp(rng.normal(scale=0.3, size=(1, K)))
    vp.lambd = np.exp(rng.normal(scale=0.2, size=(D, 1)))
    vp.eta = rng.normal(size=(1, K))
    vp.w = np.exp(vp.eta) / np.exp(vp.eta).sum()
    theta = vp.get_parameters()
    template = TorchVPTemplate.from_vp(vp)
    _, grad = _torch_grad(
        lambda th: entropy_lower_bound(decode_theta(th, template)), theta
    )
    fd = _central_difference(
        lambda th: float(
            entropy_lower_bound(decode_theta(th, template)).detach().cpu()
        ),
        theta,
    )
    np.testing.assert_allclose(_tensor_array(grad), fd, rtol=2e-5, atol=2e-7)


def test_mc_surrogate_finite_difference_uses_frozen_anchors():
    D, K, Ns = 2, 3, 40
    rng = np.random.default_rng(5)
    vp = VariationalPosterior(D, K, rng=1)
    vp.mu = rng.normal(size=(D, K))
    vp.sigma = np.exp(rng.normal(scale=0.2, size=(1, K)))
    vp.lambd = np.exp(rng.normal(scale=0.2, size=(D, 1)))
    vp.eta = rng.normal(size=(1, K))
    vp.w = np.exp(vp.eta) / np.exp(vp.eta).sum()
    theta = vp.get_parameters()
    template = TorchVPTemplate.from_vp(vp)
    eps = _epsilon(6, K, Ns, D)
    theta_t = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    base_state = decode_theta(theta_t, template)
    anchors = EntropyAnchors.from_state(base_state)
    H = entropy_mc(base_state, Ns, eps, anchors=anchors)
    grad = torch.autograd.grad(H, theta_t)[0]

    fd = _central_difference(
        lambda th: float(
            entropy_mc(decode_theta(th, template), Ns, eps, anchors=anchors)
            .detach()
            .cpu()
        ),
        theta,
    )
    np.testing.assert_allclose(_tensor_array(grad), fd, rtol=2e-5, atol=2e-7)


def test_mc_checkpoint_and_chunk_bound_preserve_value_and_gradient():
    """Checkpoint recomputation must retain each loop chunk's own j offset."""
    D, K, Ns = 2, 3, 40
    rng = np.random.default_rng(15)
    vp = VariationalPosterior(D, K, rng=1)
    vp.mu = rng.normal(size=(D, K))
    vp.sigma = np.exp(rng.normal(scale=0.2, size=(1, K)))
    vp.lambd = np.exp(rng.normal(scale=0.2, size=(D, 1)))
    vp.eta = rng.normal(size=(1, K))
    vp.w = np.exp(vp.eta) / np.exp(vp.eta).sum()
    theta = vp.get_parameters()
    template = TorchVPTemplate.from_vp(vp)
    eps = _epsilon(16, K, Ns, D)

    def evaluate(checkpoint_chunks):
        return _torch_grad(
            lambda th: entropy_mc(
                decode_theta(th, template),
                Ns,
                eps,
                max_tensor_elements=100,
                checkpoint_chunks=checkpoint_chunks,
            ),
            theta,
        )

    value, grad = evaluate(False)
    checked_value, checked_grad = evaluate(True)
    _assert_close(checked_value, value, rtol=0, atol=1e-14)
    _assert_close(checked_grad, grad, rtol=0, atol=1e-13)


def test_neg_elcbo_is_pure_and_does_not_consult_rng(snapshots):
    state = build_state(
        snapshots["normal_D2_warmup"], rng=np.random.default_rng(3)
    )
    vp, theta = _prepared_vp(state)
    theta_before = theta.copy()
    arrays_before = {
        name: np.array(getattr(vp, name), copy=True)
        for name in ("mu", "sigma", "lambd", "w", "eta")
    }
    rng_before = copy.deepcopy(vp.rng.bit_generator.state)
    bounds = vp.get_bounds(state["gp"].X, state["options"], vp.K)
    bounds_before = copy.deepcopy(bounds)
    Ns = 12
    eps = _epsilon(4, vp.K, Ns, vp.D)
    neg_elcbo(
        theta,
        TorchGP.from_gp(state["gp"]),
        vp,
        Ns=Ns,
        theta_bnd=bounds,
        epsilon=eps,
    )
    assert np.array_equal(theta, theta_before)
    for name, expected in arrays_before.items():
        assert np.array_equal(getattr(vp, name), expected)
    assert vp.rng.bit_generator.state == rng_before
    for key, expected in bounds_before.items():
        if isinstance(expected, np.ndarray):
            assert np.array_equal(bounds[key], expected)
        else:
            assert bounds[key] == expected


def test_soft_bounds_ties_infinities_and_scale_folding():
    x = torch.tensor(
        [-1.0, 0.0, 1.0, 4.0], dtype=torch.float64, requires_grad=True
    )
    lb = np.array([-1.0, -1.0, -2.0, -np.inf])
    ub = np.array([2.0, 1.0, 1.0, np.inf])
    loss = soft_bound_loss(x, lb, ub, tol_con=0.1)
    grad = torch.autograd.grad(loss, x)[0]
    assert loss.item() == 0.0
    assert torch.count_nonzero(grad).item() == 0

    D, K = 3, 2
    vp = VariationalPosterior(D, K, rng=1)
    X = np.linspace(-1.0, 1.0, 30).reshape(10, D)
    from pyvbmc.vbmc.options import Options

    options = Options(
        "option_configs/basic_vbmc_options.ini", evaluation_parameters={"D": D}
    )
    options.load_options_file(
        "option_configs/advanced_vbmc_options.ini",
        evaluation_parameters={"D": D},
    )
    bnd = vp.get_bounds(X, options, K)
    theta = vp.get_parameters()
    theta[D * K + 1] = bnd["ub"][D * K + D] + 1.0
    theta_t = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    t_loss = vp_bound_loss(theta_t, TorchVPTemplate.from_vp(vp), bnd)
    t_grad = torch.autograd.grad(t_loss, theta_t)[0]
    n_loss, n_grad = _vp_bound_loss(vp, theta, bnd)
    _assert_close(t_loss, n_loss, rtol=EXACT_RTOL, atol=EXACT_ATOL)
    _assert_close(t_grad, n_grad, rtol=EXACT_RTOL, atol=EXACT_ATOL)
    assert torch.count_nonzero(t_grad[-K:]).item() == 0


def test_soft_bound_loss_matches_numpy_outside_and_at_ties():
    lb = np.array([-10.0, -10.0, -10.0, -10.0])
    ub = np.array([10.0, 10.0, 10.0, 10.0])
    x_np = np.array([10.5, -10.7, -10.0, 10.0])
    expected, expected_grad = _soft_bound_loss(x_np, lb, ub, compute_grad=True)
    x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    actual = soft_bound_loss(x, lb, ub)
    actual_grad = torch.autograd.grad(actual, x)[0]
    _assert_close(actual, expected, rtol=0, atol=1e-12)
    _assert_close(actual_grad, expected_grad, rtol=0, atol=1e-12)


def test_epsilon_contract_d1_k1_and_extreme_mixture():
    # D=1, K=1 pins scalar shapes and exact antithetic construction.
    vp1 = VariationalPosterior(1, 1, rng=1)
    theta1 = vp1.get_parameters()
    template1 = TorchVPTemplate.from_vp(vp1)
    eps1 = _epsilon(7, 1, 3, 1)
    H1, dH1 = _torch_grad(
        lambda th: entropy_mc(decode_theta(th, template1), 3, eps1), theta1
    )
    np_vp1 = _numpy_vp(vp1, theta1)
    H1_np, dH1_np = entmc_vbmc(np_vp1, 3, rng=np.random.default_rng(7))
    assert H1.ndim == 0 and dH1.shape == (theta1.size,)
    _assert_close(H1, H1_np, rtol=EXACT_RTOL, atol=EXACT_ATOL)
    _assert_close(dH1, dH1_np, rtol=EXACT_RTOL, atol=EXACT_ATOL)
    with pytest.raises(ValueError, match="raw-half shape"):
        entropy_mc(decode_theta(theta1, template1), 3, np.zeros((1, 3, 1)))
    with pytest.raises(ValueError, match="epsilon is required"):
        entropy_mc(decode_theta(theta1, template1), 3, None)

    # Extreme but finite width/weight ratios retain current direct arithmetic.
    D, K, Ns = 2, 3, 20
    vp = VariationalPosterior(D, K, rng=1)
    vp.mu = np.array([[-20.0, 0.0, 20.0], [0.0, 0.0, 0.0]])
    vp.sigma = np.exp(np.array([[-8.0, 0.0, 8.0]]))
    vp.lambd = np.array([[0.6], [1.4]])
    vp.eta = np.array([[-20.0, 0.0, -8.0]])
    vp.w = np.exp(vp.eta) / np.exp(vp.eta).sum()
    theta = vp.get_parameters()
    template = TorchVPTemplate.from_vp(vp)
    eps = _epsilon(8, K, Ns, D)
    np_vp = _numpy_vp(vp, theta)
    H_np, dH_np = entmc_vbmc(np_vp, Ns, rng=np.random.default_rng(8))
    H_t, dH_t = _torch_grad(
        lambda th: entropy_mc(decode_theta(th, template), Ns, eps), theta
    )
    assert np.array_equal(np.isfinite(_tensor_array(H_t)), np.isfinite(H_np))
    assert np.array_equal(np.isfinite(_tensor_array(dH_t)), np.isfinite(dH_np))
    _assert_close(H_t, H_np, rtol=EXACT_RTOL, atol=EXACT_ATOL)
    _assert_close(dH_t, dH_np, rtol=EXACT_RTOL, atol=EXACT_ATOL)


def test_zero_mean_validation_and_invalid_models():
    D, N = 2, 8
    rng = np.random.default_rng(12)
    X = rng.normal(size=(N, D))
    y = rng.normal(size=(N, 1))

    def fitted(mean, covariance=None):
        gp = gpr.GP(
            D=D,
            covariance=covariance
            or gpr.covariance_functions.SquaredExponential(),
            mean=mean,
            noise=gpr.noise_functions.GaussianNoise(constant_add=True),
        )
        count = (
            gp.covariance.hyperparameter_count(D)
            + gp.noise.hyperparameter_count()
            + gp.mean.hyperparameter_count(D)
        )
        hyp = np.zeros((1, count))
        hyp[:, gp.covariance.hyperparameter_count(D)] = -2.0
        gp.update(X_new=X, y_new=y, hyp=hyp)
        return gp

    zero_gp = fitted(gpr.mean_functions.ZeroMean())
    tgp = TorchGP.from_gp(zero_gp)
    assert tgp.mean_kind == "zero" and tgp.xm is None
    vp = VariationalPosterior(D, 2, rng=1)
    theta = vp.get_parameters()
    G_np = _gp_log_joint(vp, zero_gp, False)[0]
    G_t = gp_log_joint(decode_theta(theta, TorchVPTemplate.from_vp(vp)), tgp)[
        0
    ]
    _assert_close(G_t, G_np, rtol=SOLVE_RTOL, atol=SOLVE_ATOL)

    with pytest.raises(ValueError, match="NegativeQuadratic and ZeroMean"):
        TorchGP.from_gp(fitted(gpr.mean_functions.ConstantMean()))
    with pytest.raises(ValueError, match="SquaredExponential"):
        TorchGP.from_gp(
            fitted(
                gpr.mean_functions.ZeroMean(),
                gpr.covariance_functions.Matern(degree=3),
            )
        )


def test_consistent_non_cholesky_factor_representation(snapshots):
    state = build_state(snapshots["normal_D2_warmup"])
    gp_chol = state["gp"]
    assert all(post.L_chol for post in gp_chol.posteriors)
    gp_inverse = copy.deepcopy(gp_chol)
    N = gp_chol.X.shape[0]
    for post in gp_inverse.posteriors:
        sn2_eff = 1.0 / np.asarray(post.sW).reshape(-1)[0] ** 2
        normalized_inverse = spla.cho_solve(
            (np.asarray(post.L), False), np.eye(N), check_finite=False
        )
        post.L = -normalized_inverse / sn2_eff
        post.L_chol = False

    vp, theta = _prepared_vp(state)
    decoded = decode_theta(theta, TorchVPTemplate.from_vp(vp))
    chol = gp_log_joint(
        decoded, TorchGP.from_gp(gp_chol), compute_var=True, separate_K=True
    )
    inverse = gp_log_joint(
        decoded, TorchGP.from_gp(gp_inverse), compute_var=True, separate_K=True
    )
    _assert_close(inverse[0], chol[0], rtol=1e-10, atol=1e-10)
    _assert_close(inverse[1], chol[1], rtol=1e-10, atol=1e-10)
    _assert_close(inverse[4], chol[4], rtol=1e-10, atol=1e-10)


def test_unsupported_variance_combinations(snapshots):
    state = build_state(snapshots["normal_D2_warmup"])
    vp, theta = _prepared_vp(state)
    tgp = TorchGP.from_gp(state["gp"])
    tvp = TorchVPTemplate.from_vp(vp)
    with pytest.raises(NotImplementedError, match="Diagonal approximation"):
        neg_elcbo(theta, tgp, tvp, compute_grad=False, compute_var=2)
    with pytest.raises(NotImplementedError, match="variance"):
        neg_elcbo(theta, tgp, tvp, compute_grad=True, compute_var=True)
    with pytest.raises(ValueError, match="per-component"):
        neg_elcbo(theta, tgp, tvp, compute_grad=True, separate_K=True)


def test_weight_penalty_exact_tie_has_zero_derivative(snapshots):
    state = build_state(snapshots["normal_D2_warmup"])
    state["vp"].optimize_weights = True
    vp, theta = _prepared_vp(state)
    assert vp.K == 2
    theta[-2:] = 0.0
    base = vp.get_bounds(state["gp"].X, state["options"], vp.K)
    tied = copy.deepcopy(base)
    tied["weight_threshold"] = 0.5
    tied["weight_penalty"] = 0.1
    unpenalized = copy.deepcopy(tied)
    unpenalized["weight_penalty"] = 0.0
    tgp = TorchGP.from_gp(state["gp"])
    tvp = TorchVPTemplate.from_vp(vp)
    F, dF, *_ = neg_elcbo(
        theta, tgp, tvp, theta_bnd=tied, Ns=0, compute_var=False
    )
    F0, dF0, *_ = neg_elcbo(
        theta, tgp, tvp, theta_bnd=unpenalized, Ns=0, compute_var=False
    )
    assert torch.isclose(F - F0, F.new_tensor(0.1))
    torch.testing.assert_close(dF, dF0, rtol=0, atol=1e-13)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_dtype_device_smoke(snapshots):
    state = build_state(snapshots["normal_D2_warmup"])
    vp, theta = _prepared_vp(state)
    tgp = TorchGP.from_gp(state["gp"], "cuda")
    tvp = TorchVPTemplate.from_vp(vp, "cuda")
    result = neg_elcbo(theta, tgp, tvp, Ns=0, compute_var=False)
    for value in result:
        if isinstance(value, torch.Tensor):
            assert value.dtype == torch.float64 and value.device.type == "cuda"
