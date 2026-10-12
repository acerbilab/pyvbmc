"""Additional numerical-reliability gates for the Stage 4 Torch prototype.

These tests are intentionally explicit developer tests.  They exercise fixed
GP posterior factors only; in particular, the cancellation stress below is
not evidence about the reliability of fitting or factorizing a future Torch
GP implementation.
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import gpyreg as gpr
import numpy as np
import pytest

torch = pytest.importorskip("torch")

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from torch_vi_core import (  # noqa: E402
    TorchGP,
    TorchVP,
    TorchVPTemplate,
    decode_theta,
    entropy_lower_bound,
    entropy_mc,
    gp_log_joint,
    neg_elcbo,
)
from torch_vi_fixtures import load_workload  # noqa: E402

from pyvbmc.entropy import entlb_vbmc, entmc_vbmc  # noqa: E402
from pyvbmc.testing.oracles._oracles import compare, format_rows  # noqa: E402
from pyvbmc.variational_posterior import VariationalPosterior  # noqa: E402
from pyvbmc.vbmc.variational_optimization import _gp_log_joint  # noqa: E402

EXACT_RTOL = 1e-10
EXACT_ATOL = 1e-12
SOLVE_RTOL = 1e-6
SOLVE_ATOL = 1e-10
VAR_RTOL = 1e-3
VAR_ATOL = 1e-8


def _array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _assert_oracle_close(
    actual, expected, *, rtol, atol, label="value", context=None
):
    rows = compare(
        {label: np.asarray(expected)},
        {label: _array(actual)},
        rtol,
        atol,
    )
    suffix = "" if context is None else f"\ncontext: {context}"
    assert rows[0][3], "\n" + format_rows(rows) + suffix


def _epsilon(seed, K, Ns, D):
    return np.random.default_rng(seed).standard_normal(
        (K, int(math.ceil(Ns / 2)), D)
    )


def _central_difference(fn, theta, relative_step=2e-6):
    theta = np.asarray(theta, dtype=float)
    gradient = np.empty_like(theta)
    for index in range(theta.size):
        step = relative_step * max(1.0, abs(theta[index]))
        plus = theta.copy()
        minus = theta.copy()
        plus[index] += step
        minus[index] -= step
        gradient[index] = (fn(plus) - fn(minus)) / (2.0 * step)
    return gradient


def _fresh_vp_from_theta(template, theta):
    vp = copy.deepcopy(template)
    vp.set_parameters(np.asarray(theta), raw_flag=True)
    if vp.optimize_weights:
        eta = np.asarray(theta[-vp.K :], dtype=float).copy()
        eta -= np.max(eta)
        vp.eta = eta.reshape(1, -1)
    return vp


def _physical_state(mu, sigma, lambd, w):
    D, K = mu.shape
    return TorchVP(
        D=D,
        K=K,
        mu=mu,
        sigma=sigma,
        lambd=lambd,
        w=w,
        eta=torch.log(w.detach()),
        optimize_mu=True,
        optimize_sigma=True,
        optimize_lambd=True,
        optimize_weights=True,
    )


def _fixed_gp(D=2, N=12, Ns=2, *, stress=False):
    """Build posterior factors at supplied hyperparameters; never fit a GP."""
    rng = np.random.default_rng(20260909 + D + N + Ns)
    if stress:
        if D != 2 or N != 8:
            raise ValueError("the cancellation recipe is fixed at D=2, N=8")
        centres = np.array(
            [[-1.0, -0.4], [-0.25, 0.7], [0.35, -0.8], [1.1, 0.25]]
        )
        offsets = np.array([[1e-8, -1e-8], [-1e-8, 1e-8]])
        X = (centres[:, None, :] + offsets[None, :, :]).reshape(N, D)
    else:
        X = rng.uniform(-1.5, 1.5, size=(N, D))
    y = (-0.4 * np.sum(X**2, axis=1) + 0.1 * np.sum(X, axis=1)).reshape(
        -1, 1
    )
    covariance = gpr.covariance_functions.SquaredExponential()
    mean = gpr.mean_functions.NegativeQuadratic()
    noise = gpr.noise_functions.GaussianNoise(constant_add=True)
    gp = gpr.GP(D=D, covariance=covariance, mean=mean, noise=noise)
    cov_N = covariance.hyperparameter_count(D)
    noise_N = noise.hyperparameter_count()
    mean_N = mean.hyperparameter_count(D)
    hyp = np.zeros((Ns, cov_N + noise_N + mean_N), dtype=np.float64)
    for sample in range(Ns):
        shift = 0.03 * (sample - (Ns - 1) / 2)
        hyp[sample, :D] = np.log(0.9 + 0.1 * np.arange(D)) + shift
        hyp[sample, D] = -0.1 + shift
        # Variance is exp(2*hyp_noise).  The stress value is below gpyreg's
        # 1e-6 factor-representation threshold, but remains large enough for
        # a meaningful, reproducible inverse of the near-coincident system.
        hyp[sample, cov_N] = (-7.05 if stress else -0.8) + shift
        mean_start = cov_N + noise_N
        hyp[sample, mean_start] = float(np.max(y))
        hyp[sample, mean_start + 1 : mean_start + 1 + D] = 0.0
        hyp[sample, mean_start + 1 + D :] = np.log(2.5 + 0.1 * np.arange(D))
    gp.update(X_new=X, y_new=y, hyp=hyp)

    K = 3
    vp = VariationalPosterior(D, K, rng=1)
    vp.mu = np.array(X[np.linspace(0, N - 1, K, dtype=int)].T, copy=True)
    vp.sigma = np.array([[0.45, 0.7, 1.05]], dtype=np.float64)
    if K != vp.sigma.shape[1]:
        raise RuntimeError("fixed VP recipe and K disagree")
    vp.lambd = np.linspace(0.8, 1.2, D, dtype=np.float64).reshape(D, 1)
    vp.w = np.array([[0.2, 0.5, 0.3]], dtype=np.float64)
    vp.eta = np.log(vp.w)
    return gp, vp


def test_mc_physical_coordinate_gradient_matches_current_estimator():
    """Gate raw mu, sigma, lambda and weight gradients on common epsilon."""
    D, K, Ns, seed = 3, 4, 80, 902
    rng = np.random.default_rng(901)
    mu_np = rng.normal(scale=0.7, size=(D, K))
    sigma_np = np.exp(rng.normal(scale=0.25, size=K))
    lambd_np = np.exp(rng.normal(scale=0.2, size=D))
    eta_np = rng.normal(scale=0.5, size=K)
    w_np = np.exp(eta_np) / np.sum(np.exp(eta_np))

    vp = VariationalPosterior(D, K, rng=1)
    vp.mu = mu_np.copy()
    vp.sigma = sigma_np.reshape(1, K)
    vp.lambd = lambd_np.reshape(D, 1)
    vp.w = w_np.reshape(1, K)
    vp.eta = eta_np.reshape(1, K)
    H_np, dH_np = entmc_vbmc(
        vp,
        Ns,
        grad_flags=(True, True, True, True),
        jacobian_flag=False,
        rng=np.random.default_rng(seed),
    )

    mu = torch.tensor(mu_np, dtype=torch.float64, requires_grad=True)
    sigma = torch.tensor(sigma_np, dtype=torch.float64, requires_grad=True)
    lambd = torch.tensor(lambd_np, dtype=torch.float64, requires_grad=True)
    w = torch.tensor(w_np, dtype=torch.float64, requires_grad=True)
    state = _physical_state(mu, sigma, lambd, w)
    H = entropy_mc(state, Ns, _epsilon(seed, K, Ns, D))
    gradients = torch.autograd.grad(H, (mu, sigma, lambd, w))
    dH = torch.cat(
        (
            gradients[0].transpose(0, 1).reshape(-1),
            gradients[1],
            gradients[2],
            gradients[3],
        )
    )
    _assert_oracle_close(H, H_np, rtol=EXACT_RTOL, atol=EXACT_ATOL, label="H")
    _assert_oracle_close(
        dH,
        dH_np,
        rtol=EXACT_RTOL,
        atol=EXACT_ATOL,
        label="dH_physical",
    )


def test_k1_high_sample_entropy_and_pathwise_moments():
    """For K=1 every sampled value and pathwise width gradient is analytic."""
    D, K, Ns = 4, 1, 20000
    mu_np = np.array([[0.2], [-0.3], [0.5], [1.1]])
    sigma_np = np.array([0.75])
    lambd_np = np.array([0.7, 0.9, 1.2, 1.4])
    w_np = np.array([1.0])
    eps_half = _epsilon(903, K, Ns, D)
    eps = np.concatenate((eps_half, -eps_half), axis=1)
    mean_eps2 = np.mean(eps**2, axis=(0, 1))
    H_gaussian = (
        0.5 * D * (1.0 + np.log(2.0 * np.pi))
        + D * np.log(sigma_np[0])
        + np.sum(np.log(lambd_np))
    )
    H_sampled = H_gaussian + 0.5 * (np.sum(mean_eps2) - D)

    mu = torch.tensor(mu_np, dtype=torch.float64, requires_grad=True)
    sigma = torch.tensor(sigma_np, dtype=torch.float64, requires_grad=True)
    lambd = torch.tensor(lambd_np, dtype=torch.float64, requires_grad=True)
    w = torch.tensor(w_np, dtype=torch.float64, requires_grad=True)
    H = entropy_mc(_physical_state(mu, sigma, lambd, w), Ns, eps_half)
    dmu, dsigma, dlambd, dw = torch.autograd.grad(H, (mu, sigma, lambd, w))
    _assert_oracle_close(
        H, H_sampled, rtol=1e-12, atol=1e-12, label="H_sampled"
    )
    np.testing.assert_allclose(_array(dmu), 0.0, rtol=0.0, atol=2e-14)
    np.testing.assert_allclose(
        _array(dsigma),
        np.array([np.sum(mean_eps2) / sigma_np[0]]),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _array(dlambd), mean_eps2 / lambd_np, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        _array(dw), np.array([H_sampled - 1.0]), rtol=1e-12, atol=1e-12
    )
    assert abs(float(H.detach()) - H_gaussian) < 0.03


def test_coincident_components_have_known_entropy_values():
    D, K, Ns = 2, 3, 100
    mu_np = np.tile(np.array([[0.4], [-0.2]]), (1, K))
    sigma_np = np.full(K, 0.8)
    lambd_np = np.array([0.75, 1.25])
    w_np = np.array([0.15, 0.35, 0.5])
    eps_half = _epsilon(904, K, Ns, D)
    eps = np.concatenate((eps_half, -eps_half), axis=1)
    per_component_eps2 = np.mean(np.sum(eps**2, axis=2), axis=1)
    H_gaussian = (
        0.5 * D * (1.0 + np.log(2.0 * np.pi))
        + D * np.log(sigma_np[0])
        + np.sum(np.log(lambd_np))
    )
    expected_mc = H_gaussian + 0.5 * (np.dot(w_np, per_component_eps2) - D)
    expected_lower = (
        0.5 * D * np.log(4.0 * np.pi)
        + D * np.log(sigma_np[0])
        + np.sum(np.log(lambd_np))
    )
    state = _physical_state(
        torch.tensor(mu_np, dtype=torch.float64),
        torch.tensor(sigma_np, dtype=torch.float64),
        torch.tensor(lambd_np, dtype=torch.float64),
        torch.tensor(w_np, dtype=torch.float64),
    )
    _assert_oracle_close(
        entropy_mc(state, Ns, eps_half),
        expected_mc,
        rtol=1e-12,
        atol=1e-12,
        label="coincident_mc",
    )
    _assert_oracle_close(
        entropy_lower_bound(state),
        expected_lower,
        rtol=1e-12,
        atol=1e-12,
        label="coincident_lower_bound",
    )


def test_deterministic_gp_and_objective_gradients_by_central_difference():
    gp, vp = _fixed_gp()
    theta = vp.get_parameters(raw_flag=True)
    template = TorchVPTemplate.from_vp(vp)
    torch_gp = TorchGP.from_gp(gp)

    theta_t = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    G = gp_log_joint(decode_theta(theta_t, template), torch_gp)[0]
    dG = torch.autograd.grad(G, theta_t)[0]
    dG_fd = _central_difference(
        lambda th: float(
            gp_log_joint(decode_theta(th, template), torch_gp)[0]
        ),
        theta,
    )
    np.testing.assert_allclose(_array(dG), dG_fd, rtol=2e-5, atol=2e-7)

    F, dF, *_ = neg_elcbo(
        theta,
        torch_gp,
        template,
        beta=0.0,
        Ns=0,
        compute_grad=True,
        compute_var=False,
    )
    dF_fd = _central_difference(
        lambda th: float(
            neg_elcbo(
                th,
                torch_gp,
                template,
                beta=0.0,
                Ns=0,
                compute_grad=False,
                compute_var=False,
            )[0]
        ),
        theta,
    )
    assert torch.isfinite(F)
    np.testing.assert_allclose(_array(dF), dF_fd, rtol=2e-5, atol=2e-7)


def _stress_metrics(gp):
    pairwise = gp.X[:, None, :] - gp.X[None, :, :]
    distances = np.sqrt(np.sum(pairwise**2, axis=2))
    distances[distances == 0.0] = np.inf
    cov_N = gp.covariance.hyperparameter_count(gp.D)
    condition_numbers = []
    residuals = []
    noise_variances = []
    for post in gp.posteriors:
        K = gp.covariance.compute(post.hyp[:cov_N], gp.X)
        sn2 = 1.0 / np.asarray(post.sW).reshape(-1)[0] ** 2
        A = K + sn2 * np.eye(K.shape[0])
        condition_numbers.append(float(np.linalg.cond(A)))
        noise_variances.append(float(sn2))
        if post.L_chol:
            rebuilt = sn2 * np.asarray(post.L).T @ np.asarray(post.L)
            residuals.append(
                float(np.linalg.norm(rebuilt - A) / np.linalg.norm(A))
            )
        else:
            inverse = -np.asarray(post.L)
            residuals.append(
                float(np.linalg.norm(A @ inverse - np.eye(A.shape[0])))
            )
    return {
        "min_pair_distance": float(np.min(distances)),
        "noise_variances": noise_variances,
        "condition_numbers": condition_numbers,
        "factor_residuals": residuals,
        "factor_forms": [
            "chol" if post.L_chol else "inverse" for post in gp.posteriors
        ],
        "scope": "fixed gpyreg factors; no GP fitting exercised",
    }


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_near_coincident_tiny_noise_fixed_factor_cancellation(
    device, record_property
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    gp, vp = _fixed_gp(D=2, N=8, Ns=2, stress=True)
    metrics = _stress_metrics(gp)
    for key, value in metrics.items():
        record_property(f"fixed_factor_stress_{key}", repr(value))
    assert metrics["min_pair_distance"] < 3e-8, metrics
    assert max(metrics["condition_numbers"]) > 1e6, metrics
    assert all(not post.L_chol for post in gp.posteriors), metrics

    theta = vp.get_parameters(raw_flag=True)
    vp_np = _fresh_vp_from_theta(vp, theta)
    G_np, dG_np, *_ = _gp_log_joint(vp_np, gp, True, True, True, False)
    _, _, var_np, _, varss_np, I_np, J_np = _gp_log_joint(
        vp_np, gp, False, True, True, True, True
    )
    template = TorchVPTemplate.from_vp(vp, device)
    torch_gp = TorchGP.from_gp(gp, device)
    theta_t = torch.tensor(
        theta, dtype=torch.float64, device=device, requires_grad=True
    )
    decoded = decode_theta(theta_t, template)
    G = gp_log_joint(decoded, torch_gp)[0]
    dG = torch.autograd.grad(G, theta_t)[0]
    diagnostics = {}
    Gv, var, varss, I, J = gp_log_joint(
        decode_theta(
            torch.tensor(theta, dtype=torch.float64, device=device), template
        ),
        torch_gp,
        compute_var=True,
        separate_K=True,
        diagnostics=diagnostics,
    )
    for label, actual, expected in (
        ("G", G, G_np),
        ("G_var_call", Gv, G_np),
        ("dG", dG, dG_np),
        ("I_sk", I, I_np),
    ):
        _assert_oracle_close(
            actual,
            expected,
            rtol=SOLVE_RTOL,
            atol=SOLVE_ATOL,
            label=label,
            context=metrics,
        )
    for label, actual, expected in (
        ("varG", var, var_np),
        ("var_ss", varss, varss_np),
        ("J_sjk", J, J_np),
    ):
        _assert_oracle_close(
            actual,
            expected,
            rtol=VAR_RTOL,
            atol=VAR_ATOL,
            label=label,
            context=metrics,
        )

    w = vp_np.w.reshape(-1)
    eps = np.finfo(float).eps
    J_clamped_np = np.array(J_np, copy=True)
    diagonal = np.arange(vp.K)
    J_clamped_np[:, diagonal, diagonal] = np.maximum(
        eps, J_np[:, diagonal, diagonal]
    )
    raw_weighted_np = np.einsum("sjk,j,k->s", J_np, w, w)
    preclamp_np = np.einsum("sjk,j,k->s", J_clamped_np, w, w)
    clamped_np = np.maximum(preclamp_np, eps)
    for label, expected in (
        ("varG_unclamped_J", raw_weighted_np),
        ("varG_preclamp", preclamp_np),
        ("varG_clamped", clamped_np),
    ):
        _assert_oracle_close(
            diagnostics[label],
            expected,
            rtol=VAR_RTOL,
            atol=VAR_ATOL,
            label=label,
            context=metrics,
        )
    assert np.all(np.isfinite(_array(diagnostics["J_raw"]))), metrics
    assert np.all(_array(diagnostics["varG_clamped"]) >= eps), metrics


def test_single_width_active_sequential_setter_is_stateful_but_fresh_matches():
    D, K = 3, 2
    vp = VariationalPosterior(D, K, rng=1)
    vp.sigma = np.array([[0.6, 1.4]])
    vp.lambd = np.ones((D, 1))
    vp.optimize_mu = False
    vp.optimize_sigma = False
    vp.optimize_lambd = True
    vp.optimize_weights = False
    theta = np.log(np.array([0.55, 1.0, 1.8]))

    sequential = copy.deepcopy(vp)
    sequential.set_parameters(theta)
    first_sigma = sequential.sigma.copy()
    sequential.set_parameters(theta)
    second_sigma = sequential.sigma.copy()
    assert not np.allclose(second_sigma, first_sigma)

    fresh = copy.deepcopy(vp)
    fresh.set_parameters(theta)
    template = TorchVPTemplate.from_vp(vp)
    decoded1 = decode_theta(theta, template)
    decoded2 = decode_theta(theta, template)
    np.testing.assert_allclose(_array(decoded1.sigma), fresh.sigma.reshape(-1))
    np.testing.assert_allclose(_array(decoded1.lambd), fresh.lambd.reshape(-1))
    torch.testing.assert_close(decoded1.sigma, decoded2.sigma)
    torch.testing.assert_close(decoded1.lambd, decoded2.lambd)

    H_np, dH_np = entlb_vbmc(
        fresh,
        grad_flags=(False, False, True, False),
        jacobian_flag=True,
    )
    theta_t = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    H = entropy_lower_bound(decode_theta(theta_t, template))
    dH = torch.autograd.grad(H, theta_t)[0]
    _assert_oracle_close(
        H, H_np, rtol=EXACT_RTOL, atol=EXACT_ATOL, label="fresh_H"
    )
    _assert_oracle_close(
        dH,
        dH_np,
        rtol=EXACT_RTOL,
        atol=EXACT_ATOL,
        label="fresh_dH",
    )


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_boost_single_d15_k50_ns1_fixed_shape_kernel(device):
    """Exercise the planned largest complete-fit GP shape without fitting."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    state = load_workload("boost_single", seed=1701)
    gp = state["gp"]
    D, K = 15, 50
    assert gp.X.shape == (750, D)
    assert len(gp.posteriors) == 1
    vp = VariationalPosterior(D, K, rng=1)
    pick = np.linspace(0, gp.X.shape[0] - 1, K, dtype=int)
    vp.mu = np.array(gp.X[pick].T, dtype=np.float64, copy=True)
    vp.sigma = np.linspace(0.25, 0.65, K, dtype=np.float64).reshape(1, K)
    vp.lambd = np.linspace(0.8, 1.2, D, dtype=np.float64).reshape(D, 1)
    vp.w = np.linspace(1.0, 2.0, K, dtype=np.float64).reshape(1, K)
    vp.w /= np.sum(vp.w)
    vp.eta = np.log(vp.w)
    theta = vp.get_parameters(raw_flag=True)
    vp_np = _fresh_vp_from_theta(vp, theta)  # refresh eta from theta

    G_np, dG_np, *_ = _gp_log_joint(vp_np, gp, True, True, True, False)
    Gs_np, _, _, _, _, I_np, _ = _gp_log_joint(
        vp_np, gp, False, False, True, False, True
    )
    template = TorchVPTemplate.from_vp(vp, device)
    torch_gp = TorchGP.from_gp(gp, device)
    theta_t = torch.tensor(
        theta, dtype=torch.float64, device=device, requires_grad=True
    )
    G = gp_log_joint(decode_theta(theta_t, template), torch_gp)[0]
    dG = torch.autograd.grad(G, theta_t)[0]
    Gs, _, _, I, _ = gp_log_joint(
        decode_theta(
            torch.tensor(theta, dtype=torch.float64, device=device), template
        ),
        torch_gp,
        avg_flag=False,
        separate_K=True,
    )
    _assert_oracle_close(
        G, G_np, rtol=SOLVE_RTOL, atol=SOLVE_ATOL, label="boost_G"
    )
    _assert_oracle_close(
        dG, dG_np, rtol=SOLVE_RTOL, atol=SOLVE_ATOL, label="boost_dG"
    )
    _assert_oracle_close(
        Gs, Gs_np, rtol=SOLVE_RTOL, atol=SOLVE_ATOL, label="boost_G_sample"
    )
    _assert_oracle_close(
        I, I_np, rtol=SOLVE_RTOL, atol=SOLVE_ATOL, label="boost_I_sk"
    )
