"""Float64 PyTorch prototype of the PyVBMC variational objective.

This module is deliberately confined to ``dev/scripts``.  It transfers a
fitted gpyreg GP once, decodes variational parameters without mutating the
template posterior, and keeps all objective arithmetic on one Torch device.
The implementation follows the current NumPy formulas; it is an experiment,
not a production backend.

The Monte Carlo entropy deserves special mention.  Its forward value is the
ordinary sampled ``-log q`` entropy.  PyVBMC's gradient, however, is the
pathwise estimator: density locations and scales do not contribute score
terms, while sample locations, the outer component weights, and the weights
inside the mixture density remain live.  :func:`entropy_mc` reproduces that
estimator by detaching only the density location/scale anchors.  A maintainer
changing the density must preserve this split explicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

_FLOAT_DTYPE = torch.float64
_MAX_MC_TENSOR_ELEMENTS = 2**16


def as_float64_tensor(value: Any, device: Any = None) -> torch.Tensor:
    """Copy/convert ``value`` to a float64 tensor on ``device``.

    Tensor inputs already on the requested device retain their autograd
    connection.  NumPy inputs are copied so the supposedly frozen prototype
    state cannot alias a caller-owned array.
    """
    if isinstance(value, torch.Tensor):
        return value.to(dtype=_FLOAT_DTYPE, device=device or value.device)
    return torch.tensor(value, dtype=_FLOAT_DTYPE, device=device)


def _require_float64(tensor: torch.Tensor, name: str) -> None:
    if tensor.dtype != _FLOAT_DTYPE:
        raise TypeError(f"{name} must be torch.float64, got {tensor.dtype}.")


@dataclass(frozen=True)
class TorchGP:
    """Immutable tensor view of the fitted GP state used by VI."""

    D: int
    N: int
    Ns: int
    cov_N: int
    noise_N: int
    mean_kind: str
    X: torch.Tensor
    hyp: torch.Tensor
    ell: torch.Tensor
    ln_sf2: torch.Tensor
    sum_lnell: torch.Tensor
    m0: torch.Tensor
    xm: torch.Tensor | None
    inv_omega2: torch.Tensor | None
    alpha: torch.Tensor
    L: torch.Tensor
    sn2_eff: torch.Tensor
    L_chol: tuple[bool, ...]

    @property
    def device(self) -> torch.device:
        return self.X.device

    @classmethod
    def from_gp(cls, gp: Any, device: Any = "cpu") -> "TorchGP":
        """Extract and transfer every fixed GP array exactly once.

        Only the SE-ARD covariance with ``NegativeQuadratic`` (production) or
        ``ZeroMean`` (focused model-layout check) is accepted.
        """
        from gpyreg.covariance_functions import SquaredExponential
        from gpyreg.mean_functions import NegativeQuadratic, ZeroMean

        if not isinstance(gp.covariance, SquaredExponential):
            raise ValueError(
                "Torch VI supports only gpyreg SquaredExponential ARD covariance."
            )
        if type(gp.covariance) is not SquaredExponential:
            raise ValueError(
                "Torch VI requires the non-isotropic SquaredExponential layout."
            )
        if isinstance(gp.mean, NegativeQuadratic):
            mean_kind = "negative_quadratic"
        elif isinstance(gp.mean, ZeroMean):
            mean_kind = "zero"
        else:
            raise ValueError(
                "Torch VI supports only NegativeQuadratic and ZeroMean layouts."
            )

        X_np = np.asarray(gp.X)
        if X_np.ndim != 2:
            raise ValueError(
                f"gp.X must be two-dimensional, got {X_np.shape}."
            )
        N, D = (int(v) for v in X_np.shape)
        posts = tuple(gp.posteriors)
        if not posts:
            raise ValueError("gp.posteriors must contain at least one sample.")
        Ns = len(posts)
        cov_N = int(gp.covariance.hyperparameter_count(D))
        noise_N = int(gp.noise.hyperparameter_count())
        hyp_np = np.stack([np.asarray(post.hyp).reshape(-1) for post in posts])
        expected_hyp = (
            cov_N + noise_N + (0 if mean_kind == "zero" else 1 + 2 * D)
        )
        if hyp_np.shape != (Ns, expected_hyp):
            raise ValueError(
                f"unsupported GP hyperparameter layout {hyp_np.shape}; "
                f"expected {(Ns, expected_hyp)}."
            )
        alpha_np = np.stack(
            [np.asarray(post.alpha).reshape(-1) for post in posts]
        )
        L_np = np.stack([np.asarray(post.L) for post in posts])
        if alpha_np.shape != (Ns, N) or L_np.shape != (Ns, N, N):
            raise ValueError(
                "GP posterior factor shapes do not match X: "
                f"alpha={alpha_np.shape}, L={L_np.shape}, X={X_np.shape}."
            )
        sn2_eff_np = np.asarray(
            [1.0 / np.asarray(post.sW).reshape(-1)[0] ** 2 for post in posts]
        )
        L_chol = tuple(bool(post.L_chol) for post in posts)

        X = as_float64_tensor(X_np, device)
        hyp = as_float64_tensor(hyp_np, device)
        alpha = as_float64_tensor(alpha_np, device)
        L = as_float64_tensor(L_np, device)
        sn2_eff = as_float64_tensor(sn2_eff_np, device)
        ell = torch.exp(hyp[:, :D])
        ln_sf2 = 2.0 * hyp[:, D]
        sum_lnell = torch.sum(hyp[:, :D], dim=1)
        if mean_kind == "zero":
            m0 = torch.zeros(Ns, dtype=_FLOAT_DTYPE, device=X.device)
            xm = None
            inv_omega2 = None
        else:
            mean_start = cov_N + noise_N
            m0 = hyp[:, mean_start]
            xm = hyp[:, mean_start + 1 : mean_start + D + 1]
            omega = torch.exp(hyp[:, mean_start + D + 1 :])
            inv_omega2 = 1.0 / omega**2

        return cls(
            D,
            N,
            Ns,
            cov_N,
            noise_N,
            mean_kind,
            X,
            hyp,
            ell,
            ln_sf2,
            sum_lnell,
            m0,
            xm,
            inv_omega2,
            alpha,
            L,
            sn2_eff,
            L_chol,
        )


@dataclass(frozen=True)
class TorchVPTemplate:
    """Fixed VP data and active-parameter flags cached on one device."""

    D: int
    K: int
    mu: torch.Tensor
    sigma: torch.Tensor
    lambd: torch.Tensor
    w: torch.Tensor
    eta: torch.Tensor
    optimize_mu: bool
    optimize_sigma: bool
    optimize_lambd: bool
    optimize_weights: bool

    @property
    def device(self) -> torch.device:
        return self.mu.device

    @classmethod
    def from_vp(cls, vp: Any, device: Any = "cpu") -> "TorchVPTemplate":
        D, K = int(vp.D), int(vp.K)
        mu = as_float64_tensor(np.asarray(vp.mu).reshape(D, K), device)
        sigma = as_float64_tensor(np.asarray(vp.sigma).reshape(K), device)
        lambd = as_float64_tensor(np.asarray(vp.lambd).reshape(D), device)
        w = as_float64_tensor(np.asarray(vp.w).reshape(K), device)
        eta = as_float64_tensor(np.asarray(vp.eta).reshape(K), device)
        return cls(
            D,
            K,
            mu,
            sigma,
            lambd,
            w,
            eta,
            bool(vp.optimize_mu),
            bool(vp.optimize_sigma),
            bool(vp.optimize_lambd),
            bool(vp.optimize_weights),
        )


@dataclass(frozen=True)
class TorchVP:
    """Decoded physical variational state."""

    D: int
    K: int
    mu: torch.Tensor
    sigma: torch.Tensor
    lambd: torch.Tensor
    w: torch.Tensor
    eta: torch.Tensor
    optimize_mu: bool
    optimize_sigma: bool
    optimize_lambd: bool
    optimize_weights: bool

    @property
    def device(self) -> torch.device:
        return self.mu.device


@dataclass(frozen=True)
class TorchBounds:
    """Device-resident copy of the current soft-bound metadata."""

    lb: torch.Tensor
    ub: torch.Tensor
    tol_con: float
    weight_threshold: float | None
    weight_penalty: float | None

    @classmethod
    def from_dict(
        cls, bounds: dict[str, Any], device: Any = "cpu"
    ) -> "TorchBounds":
        return cls(
            as_float64_tensor(bounds["lb"], device),
            as_float64_tensor(bounds["ub"], device),
            float(bounds["tol_con"]),
            (
                None
                if "weight_threshold" not in bounds
                else float(bounds["weight_threshold"])
            ),
            (
                None
                if "weight_penalty" not in bounds
                else float(bounds["weight_penalty"])
            ),
        )


@dataclass(frozen=True)
class EntropyAnchors:
    """Detached density locations/scales for the MC pathwise surrogate."""

    mu: torch.Tensor
    sigma: torch.Tensor
    lambd: torch.Tensor

    @classmethod
    def from_state(cls, state: TorchVP) -> "EntropyAnchors":
        return cls(
            state.mu.detach(), state.sigma.detach(), state.lambd.detach()
        )


def _template(vp: Any, device: Any) -> TorchVPTemplate:
    if isinstance(vp, TorchVPTemplate):
        if torch.device(device) != vp.device:
            raise ValueError(
                f"VP template is on {vp.device}, requested device is {device}."
            )
        return vp
    return TorchVPTemplate.from_vp(vp, device)


def parameter_count(vp: Any) -> int:
    """Number of active optimizer coordinates in a VP/template."""
    return (
        int(vp.optimize_mu) * int(vp.D) * int(vp.K)
        + int(vp.optimize_sigma) * int(vp.K)
        + int(vp.optimize_lambd) * int(vp.D)
        + int(vp.optimize_weights) * int(vp.K)
    )


def decode_theta(theta: Any, vp: Any, device: Any = None) -> TorchVP:
    """Pure equivalent of ``VariationalPosterior.set_parameters``.

    Means use the current Fortran packing.  The RMS lambda gauge is applied
    with a detached normalizer.  This is deliberate: the current hand
    gradients use unprojected physical log-gradients, including unusual
    partial active masks.  All ordinary objective terms depend only on
    ``sigma_k * lambd_d``, so detaching changes no standard all-width-active
    derivative and preserves the established convention for partial masks.
    """
    inferred_device = (
        theta.device if isinstance(theta, torch.Tensor) else device
    )
    template = _template(vp, inferred_device or "cpu")
    theta = as_float64_tensor(theta, template.device)
    _require_float64(theta, "theta")
    if theta.ndim != 1:
        raise ValueError(
            f"theta must be one-dimensional, got {tuple(theta.shape)}."
        )
    expected = parameter_count(template)
    if theta.numel() != expected:
        raise ValueError(
            f"theta has {theta.numel()} entries; expected {expected}."
        )

    D, K = template.D, template.K
    start = 0
    if template.optimize_mu:
        mu = theta[start : start + D * K].reshape(K, D).transpose(0, 1)
        start += D * K
    else:
        mu = template.mu
    if template.optimize_sigma:
        sigma = torch.exp(theta[start : start + K])
        start += K
    else:
        sigma = template.sigma
    if template.optimize_lambd:
        lambd = torch.exp(theta[start : start + D])
        start += D
    else:
        lambd = template.lambd
    if template.optimize_weights:
        eta_raw = theta[start : start + K]
        eta = eta_raw - torch.amax(eta_raw)
        exp_eta = torch.exp(eta)
        w = exp_eta / torch.sum(exp_eta)
        start += K
    else:
        eta = template.eta
        w = template.w
    if start != expected:
        raise RuntimeError("internal theta decoder offset mismatch")

    gauge = torch.sqrt(torch.sum(lambd**2) / D).detach()
    lambd = lambd / gauge
    sigma = sigma * gauge
    return TorchVP(
        D,
        K,
        mu,
        sigma,
        lambd,
        w,
        eta,
        template.optimize_mu,
        template.optimize_sigma,
        template.optimize_lambd,
        template.optimize_weights,
    )


def entropy_lower_bound(state: TorchVP) -> torch.Tensor:
    """Current deterministic Jensen entropy lower bound (forward graph)."""
    D, K = state.D, state.K
    mu_t = state.mu.transpose(0, 1)
    sigma, lambd, w = state.sigma, state.lambd, state.w
    if K == 1:
        return (
            0.5 * D * (1.0 + math.log(2.0 * math.pi))
            + D * torch.sum(torch.log(sigma))
            + torch.sum(torch.log(lambd))
        )

    sumsigma2 = sigma[:, None] ** 2 + sigma[None, :] ** 2
    sumsigma = torch.sqrt(sumsigma2)
    nconst = 1.0 / (2.0 * math.pi) ** (D / 2.0) / torch.prod(lambd)
    delta = (mu_t[:, None, :] - mu_t[None, :, :]) / (
        sumsigma[..., None] * lambd
    )
    d2 = torch.sum(delta**2, dim=2)
    gamma = nconst / sumsigma**D * torch.exp(-0.5 * d2)
    gammasum = torch.sum(w * gamma, dim=1)
    return -torch.sum(w * torch.log(gammasum))


def _epsilon_tensor(
    epsilon: Any, state: TorchVP, Ns: int
) -> tuple[torch.Tensor, int]:
    if epsilon is None:
        raise ValueError(
            "epsilon is required for MC entropy; draw raw half-antithetic "
            "samples outside the objective."
        )
    half = int(math.ceil(int(Ns) / 2))
    eps_half = as_float64_tensor(epsilon, state.device)
    expected = (state.K, half, state.D)
    if tuple(eps_half.shape) != expected:
        raise ValueError(
            f"epsilon must have raw-half shape {expected}, got "
            f"{tuple(eps_half.shape)}."
        )
    return torch.cat((eps_half, -eps_half), dim=1), 2 * half


def entropy_mc(
    state: TorchVP,
    Ns: int,
    epsilon: Any,
    *,
    anchors: EntropyAnchors | None = None,
    max_tensor_elements: int = _MAX_MC_TENSOR_ELEMENTS,
    checkpoint_chunks: bool = False,
) -> torch.Tensor:
    """Sampled entropy with PyVBMC's pathwise surrogate gradient.

    ``epsilon`` is the raw NumPy-compatible half draw with shape
    ``(K, ceil(Ns / 2), D)``.  Antithetic samples are constructed here.  For
    finite-difference checks, pass one ``EntropyAnchors`` instance created at
    the unperturbed state to every perturbed call; recomputing anchors tests
    the derivative of sampled entropy, which is a different estimator.

    Gradient-bearing chunks can use non-reentrant activation checkpointing.
    This retains the explicit epsilon and small state tensors but recomputes
    the ``components x samples x D x K`` density slab during backward,
    bounding saved activation memory as well as the forward temporary.  It is
    off by default because approved optimizer shapes retain only tens of MB
    and checkpointing every small chunk adds substantial eager overhead; fine
    scoring already runs without a graph.  The benchmark can measure the
    explicit checkpointed mode for larger follow-up shapes.
    """
    if int(Ns) <= 0:
        raise ValueError("Ns must be positive for MC entropy.")
    if max_tensor_elements <= 0:
        raise ValueError("max_tensor_elements must be positive.")
    epsilon_full, Ns_even = _epsilon_tensor(epsilon, state, Ns)
    anchors = EntropyAnchors.from_state(state) if anchors is None else anchors
    if not (
        anchors.mu.device == state.device
        and anchors.sigma.device == state.device
        and anchors.lambd.device == state.device
    ):
        raise ValueError("entropy anchors and VP state must share one device.")

    D, K = state.D, state.K
    anchor_scale = anchors.lambd[:, None] * anchors.sigma[None, :]
    nconst = 1.0 / (2.0 * math.pi) ** (D / 2.0) / torch.prod(anchors.lambd)
    nf = nconst / anchors.sigma**D
    per_component = Ns_even * D * K
    group = max(1, min(K, max_tensor_elements // max(1, per_component)))
    step = (
        Ns_even
        if per_component <= max_tensor_elements
        else max(1, max_tensor_elements // max(1, D * K))
    )
    sums = [state.mu.new_zeros(()) for _ in range(K)]

    def chunk_sum(
        mu: torch.Tensor,
        sigma: torch.Tensor,
        lambd: torch.Tensor,
        w: torch.Tensor,
        eps_b: torch.Tensor,
        j0: int,
    ) -> torch.Tensor:
        width = eps_b.shape[0]
        x_b = (
            eps_b * lambd * sigma[j0 : j0 + width, None, None]
            + mu.transpose(0, 1)[j0 : j0 + width, None, :]
        )
        delta = (x_b[..., None] - anchors.mu) / anchor_scale
        d2 = torch.einsum("jndk,jndk->jnk", delta, delta)
        E = torch.exp(-0.5 * d2)
        q = E @ (w * nf)
        return torch.sum(torch.log(q), dim=1)

    use_checkpoint = checkpoint_chunks and torch.is_grad_enabled()
    for j0 in range(0, K, group):
        j1 = min(K, j0 + group)
        for n0 in range(0, Ns_even, step):
            eps_b = epsilon_full[j0:j1, n0 : min(Ns_even, n0 + step)]
            if use_checkpoint:
                values = checkpoint(
                    lambda mu, sigma, lambd, w, eps, chunk_j0=j0: chunk_sum(
                        mu, sigma, lambd, w, eps, chunk_j0
                    ),
                    state.mu,
                    state.sigma,
                    state.lambd,
                    state.w,
                    eps_b,
                    use_reentrant=False,
                )
            else:
                values = chunk_sum(
                    state.mu,
                    state.sigma,
                    state.lambd,
                    state.w,
                    eps_b,
                    j0,
                )
            for local, j in enumerate(range(j0, j1)):
                sums[j] = sums[j] + values[local]
    sum_logq = torch.stack(sums)
    return -torch.sum(state.w * sum_logq) / Ns_even


def gp_log_joint(
    state: TorchVP,
    gp: TorchGP,
    *,
    avg_flag: bool = True,
    compute_var: bool = False,
    separate_K: bool = False,
    diagnostics: dict[str, torch.Tensor] | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    """Forward GP integrals: ``G, varG, var_ss, I_sk, J_sjk``.

    ``G`` remains connected to the VP tensors for ordinary autograd.  Full
    variance is value-only in the prototype, matching the current supported
    combinations.  ``J_sjk`` is the raw, lower-triangle-mirrored quantity;
    its diagonal and the final weighted variance are clamped only in the
    variance contraction, exactly as in NumPy.  If ``diagnostics`` is supplied,
    it is filled with detached per-hyperparameter raw/clamped intermediates;
    the default timing path creates none of them beyond the necessary
    contractions.
    """
    if compute_var == 2:
        raise NotImplementedError(
            "Diagonal approximation of GP log-joint variance not implemented."
        )
    if compute_var and any(
        tensor.requires_grad
        for tensor in (state.mu, state.sigma, state.lambd, state.w)
    ):
        raise NotImplementedError(
            "Computation of gradient of log joint variance is currently "
            "available only for diagonal approximation of the variance."
        )
    if state.D != gp.D:
        raise ValueError(f"VP dimension {state.D} does not match GP {gp.D}.")
    if state.device != gp.device:
        raise ValueError(f"VP is on {state.device}, GP is on {gp.device}.")
    D, K, Ns = state.D, state.K, gp.Ns
    mu_t = state.mu.transpose(0, 1)
    sigma, lambd, w = state.sigma, state.lambd, state.w

    tau = torch.sqrt(
        sigma[None, :, None] ** 2 * lambd[None, None, :] ** 2
        + gp.ell[:, None, :] ** 2
    )
    lnnf = (
        gp.ln_sf2[:, None]
        + gp.sum_lnell[:, None]
        - torch.sum(torch.log(tau), dim=2)
    )
    Xt = mu_t[:, :, None] - gp.X.transpose(0, 1)[None, :, :]
    delta = Xt[None, :, :, :] / tau[:, :, :, None]
    dsq_sum = torch.einsum("skdn,skdn->skn", delta, delta)
    z = torch.exp(lnnf[:, :, None] - 0.5 * dsq_sum)
    zalpha = (z @ gp.alpha[:, :, None])[:, :, 0]
    I_sk = zalpha + gp.m0[:, None]
    if gp.mean_kind == "negative_quadratic":
        assert gp.xm is not None and gp.inv_omega2 is not None
        sig2lam2 = sigma[None, :, None] ** 2 * lambd[None, None, :] ** 2
        xm = gp.xm[:, None, :]
        nu = -0.5 * torch.sum(
            gp.inv_omega2[:, None, :]
            * (mu_t[None] ** 2 + sig2lam2 - 2.0 * mu_t[None] * xm + xm**2),
            dim=2,
        )
        I_sk = I_sk + nu
    G = torch.sum(w[None, :] * I_sk, dim=1)

    J_sjk = None
    varG = None
    if compute_var:
        sig2 = sigma**2
        tau_jk = torch.sqrt(
            (sig2[:, None] + sig2[None, :])[None, :, :, None]
            * lambd[None, None, None, :] ** 2
            + gp.ell[:, None, None, :] ** 2
        )
        lnnf_jk = (
            gp.ln_sf2[:, None, None]
            + gp.sum_lnell[:, None, None]
            - torch.sum(torch.log(tau_jk), dim=3)
        )
        delta_jk = (mu_t[:, None, :] - mu_t[None, :, :])[
            None, :, :, :
        ] / tau_jk
        J = torch.exp(lnnf_jk - 0.5 * torch.sum(delta_jk**2, dim=3))
        corrections = []
        for s in range(Ns):
            z_s = z[s]
            if gp.L_chol[s]:
                inner = torch.linalg.solve_triangular(
                    gp.L[s].transpose(0, 1), z_s.transpose(0, 1), upper=False
                )
                Y = (
                    torch.linalg.solve_triangular(gp.L[s], inner, upper=True)
                    / gp.sn2_eff[s]
                )
                corrections.append(-(z_s @ Y))
            else:
                corrections.append(z_s @ gp.L[s] @ z_s.transpose(0, 1))
        J = J + torch.stack(corrections)
        lower = torch.tril(J)
        J = (
            lower
            + lower.transpose(1, 2)
            - torch.diag_embed(torch.diagonal(lower, dim1=1, dim2=2))
        )
        diag = torch.diagonal(J, dim1=1, dim2=2)
        eps = torch.finfo(_FLOAT_DTYPE).eps
        J_clamped = J + torch.diag_embed(torch.clamp_min(diag, eps) - diag)
        varG_preclamp = torch.einsum("sjk,j,k->s", J_clamped, w, w)
        varG = torch.clamp_min(varG_preclamp, eps)
        if diagnostics is not None:
            diagnostics.update(
                {
                    "J_raw": J.detach(),
                    "J_diag_raw": diag.detach(),
                    "J_diag_clamped": torch.diagonal(
                        J_clamped, dim1=1, dim2=2
                    ).detach(),
                    "varG_unclamped_J": torch.einsum(
                        "sjk,j,k->s", J, w, w
                    ).detach(),
                    "varG_preclamp": varG_preclamp.detach(),
                    "varG_clamped": varG.detach(),
                    "J_diag_clamped_mask": (diag < eps).detach(),
                    "varG_clamped_mask": (varG_preclamp < eps).detach(),
                }
            )
        J_sjk = J if separate_K else None

    var_ss = state.mu.new_zeros(())
    if Ns > 1 and avg_flag:
        G_bar = torch.sum(G) / Ns
        if compute_var:
            assert varG is not None
            varG_ss = torch.sum((G - G_bar) ** 2) / (Ns - 1)
            var_ss = varG_ss + torch.std(varG, correction=1)
            varG = torch.sum(varG) / Ns + varG_ss
        G = G_bar
    if Ns == 1:
        G = G[0]
        if compute_var:
            assert varG is not None
            varG = varG[0]
    return G, varG, var_ss, I_sk, J_sjk


def soft_bound_loss(
    x: torch.Tensor,
    slb: Any,
    sub: Any,
    tol_con: float = 1e-3,
) -> torch.Tensor:
    """Quadratic soft-bound loss with zero derivative at exact ties."""
    slb = as_float64_tensor(slb, x.device)
    sub = as_float64_tensor(sub, x.device)
    if x.shape != slb.shape or x.shape != sub.shape:
        raise ValueError(
            f"bound shapes must equal x: {tuple(x.shape)}, "
            f"{tuple(slb.shape)}, {tuple(sub.shape)}."
        )
    valid = torch.isfinite(slb) & torch.isfinite(sub) & (sub > slb)
    safe_lb = torch.where(valid, slb, torch.zeros_like(slb))
    safe_ub = torch.where(valid, sub, torch.ones_like(sub))
    ell = (safe_ub - safe_lb) * float(tol_con)
    lower = valid & (x < slb)
    upper = valid & (x > sub)
    dl = torch.where(lower, (safe_lb - x) / ell, torch.zeros_like(x))
    du = torch.where(upper, (x - safe_ub) / ell, torch.zeros_like(x))
    return 0.5 * torch.sum(dl**2 + du**2)


def _bounds(bounds: Any, device: Any) -> TorchBounds:
    if isinstance(bounds, TorchBounds):
        if bounds.lb.device != torch.device(device):
            raise ValueError("bounds and theta must share one device.")
        return bounds
    return TorchBounds.from_dict(bounds, device)


def vp_bound_loss(
    theta: torch.Tensor,
    vp: Any,
    theta_bnd: TorchBounds | dict[str, Any],
    tol_con: float = 1e-3,
) -> torch.Tensor:
    """Current bound loss in optimizer coordinates, including scale folds."""
    template = _template(vp, theta.device)
    bnd = _bounds(theta_bnd, theta.device)
    D, K = template.D, template.K
    start = 0
    pieces = []
    if template.optimize_mu:
        pieces.append(theta[: D * K])
        start = D * K
    if template.optimize_sigma:
        ln_sigma = theta[start : start + K]
        start += K
    else:
        ln_sigma = torch.log(template.sigma)
    if template.optimize_lambd:
        ln_lambd = theta[start : start + D]
        start += D
    else:
        ln_lambd = torch.log(template.lambd)
    if template.optimize_sigma or template.optimize_lambd:
        ln_scale = ln_lambd[:, None] + ln_sigma[None, :]
        pieces.append(ln_scale.transpose(0, 1).reshape(-1))
    if template.optimize_weights:
        pieces.append(theta[start : start + K])
        start += K
    if start != theta.numel():
        raise ValueError("theta length does not match VP active flags.")
    theta_ext = torch.cat(pieces) if pieces else theta.new_empty((0,))
    if theta_ext.shape != bnd.lb.shape or theta_ext.shape != bnd.ub.shape:
        raise ValueError(
            f"extended theta has {theta_ext.numel()} entries but bounds have "
            f"{bnd.lb.numel()}."
        )
    lb, ub = bnd.lb, bnd.ub
    if template.optimize_weights:
        free = torch.full(
            (K,), float("inf"), dtype=_FLOAT_DTYPE, device=theta.device
        )
        lb = torch.cat((lb[:-K], -free))
        ub = torch.cat((ub[:-K], free))
    return soft_bound_loss(theta_ext, lb, ub, tol_con)


def _neg_elcbo_forward(
    theta: torch.Tensor,
    gp: TorchGP,
    vp: TorchVPTemplate,
    beta: float,
    Ns: int,
    compute_var: bool,
    theta_bnd: TorchBounds | None,
    separate_K: bool,
    epsilon: Any,
) -> tuple[Any, ...]:
    state = decode_theta(theta, vp)
    G, varG, varG_ss, I_sk, J_sjk = gp_log_joint(
        state,
        gp,
        avg_flag=True,
        compute_var=compute_var,
        separate_K=separate_K,
    )
    H = (
        entropy_mc(state, Ns, epsilon)
        if Ns > 0
        else entropy_lower_bound(state)
    )
    F = -G - H
    zero = F.new_zeros(())
    varH = zero
    varF = varG + varH if compute_var else zero
    if beta != 0.0:
        F = F + beta * torch.sqrt(varF)
    if theta_bnd is not None:
        F = F + vp_bound_loss(theta, vp, theta_bnd, tol_con=theta_bnd.tol_con)
        if vp.optimize_weights:
            if (
                theta_bnd.weight_threshold is None
                or theta_bnd.weight_penalty is None
            ):
                raise ValueError(
                    "weight bounds lack threshold/penalty metadata."
                )
            threshold = F.new_tensor(theta_bnd.weight_threshold)
            capped = torch.where(state.w < threshold, state.w, threshold)
            F = F + theta_bnd.weight_penalty * torch.sum(capped)
    if separate_K:
        return (
            F,
            None,
            G,
            H,
            varF,
            None,
            varG_ss,
            varG if compute_var else zero,
            varH,
            I_sk,
            J_sjk,
        )
    return F, None, G, H, varF


def neg_elcbo(
    theta: Any,
    torch_gp: TorchGP,
    vp: Any,
    beta: float = 0.0,
    Ns: int = 0,
    compute_grad: bool = True,
    compute_var: bool | int | None = None,
    theta_bnd: TorchBounds | dict[str, Any] | None = None,
    separate_K: bool = False,
    epsilon: Any = None,
) -> tuple[Any, ...]:
    """Pure tensor counterpart of ``_neg_elcbo``.

    Returns five items normally and eleven with ``separate_K=True``, matching
    the baseline order.  Values are float64 tensors on ``torch_gp.device``;
    unavailable gradients/pair integrals are ``None``.  Supplied MC epsilon is
    the raw half-antithetic draw and no RNG is consulted by this function.
    """
    beta = float(beta)
    if not math.isfinite(beta):
        beta = 0.0
    if compute_var is None:
        compute_var = beta != 0.0
    if compute_var == 2:
        raise NotImplementedError(
            "Diagonal approximation of GP log-joint variance not implemented."
        )
    compute_var = bool(compute_var)
    if separate_K and compute_grad:
        raise ValueError(
            "Computing variational gradients and per-component results together."
        )
    if compute_grad and compute_var:
        raise NotImplementedError(
            "Computation of gradient of log joint variance is currently "
            "available only for diagonal approximation of the variance."
        )
    if compute_grad and beta != 0.0:
        raise NotImplementedError(
            "Computation of the gradient of ELBO with full variance not supported"
        )

    vp_template = _template(vp, torch_gp.device)
    theta_tensor = as_float64_tensor(theta, torch_gp.device)
    _require_float64(theta_tensor, "theta")
    theta_work = (
        theta_tensor.detach().clone().requires_grad_(bool(compute_grad))
    )
    bounds = None if theta_bnd is None else _bounds(theta_bnd, torch_gp.device)
    context = torch.enable_grad() if compute_grad else torch.no_grad()
    with context:
        result = _neg_elcbo_forward(
            theta_work,
            torch_gp,
            vp_template,
            beta,
            int(Ns),
            compute_var,
            bounds,
            separate_K,
            epsilon,
        )
        if compute_grad:
            F = result[0]
            if theta_work.numel() == 0:
                dF = theta_work.clone()
            else:
                dF = torch.autograd.grad(F, theta_work, create_graph=False)[0]
            result = tuple(
                value.detach() if isinstance(value, torch.Tensor) else value
                for value in result
            )
            result = (result[0], dF.detach(), *result[2:])
    return result


__all__ = [
    "EntropyAnchors",
    "TorchBounds",
    "TorchGP",
    "TorchVP",
    "TorchVPTemplate",
    "as_float64_tensor",
    "decode_theta",
    "entropy_lower_bound",
    "entropy_mc",
    "gp_log_joint",
    "neg_elcbo",
    "parameter_count",
    "soft_bound_loss",
    "vp_bound_loss",
]
