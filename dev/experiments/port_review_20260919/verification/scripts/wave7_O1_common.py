"""Shared helpers for the wave-7 verification of slice O1 (scratch only)."""

import copy
import subprocess

import gpyreg as gpr
import numpy as np

import pyvbmc
from pyvbmc.entropy import entlb_vbmc
from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.vbmc.variational_optimization import (
    _gp_log_joint,
    _soft_bound_loss,
)

WT = "C:/Users/luigi/Documents/GitHub/pyvbmc-wave7"
GPYREG = "C:/Users/luigi/Documents/GitHub/gpyreg"

OPTS = {
    "tol_con_loss": 0.01,
    "tol_weight": 1e-2,
    "weight_penalty": 0.1,
    "tol_length": 1e-6,
}


def banner():
    print("pyvbmc.__file__:", pyvbmc.__file__, flush=True)
    print("gpyreg.__file__:", gpr.__file__, flush=True)
    for path in (WT, GPYREG):
        rev = subprocess.run(
            ["git", "-C", path, "log", "-1", "--format=%h %s"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        print(f"  {path}: {rev}", flush=True)


MEANS = {
    "zero": gpr.mean_functions.ZeroMean,
    "const": gpr.mean_functions.ConstantMean,
    "negquad": gpr.mean_functions.NegativeQuadratic,
}


def build_gp(D, N=20, Ns=2, seed=0, mean="negquad", ln_sn=-1.5):
    """GP on a noisy Gaussian log density, hyperparameters drawn at random."""
    rng = np.random.default_rng(seed)
    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=MEANS[mean](),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    X = rng.uniform(-2.5, 2.5, size=(N, D))
    y = -0.5 * np.sum((X - 0.3) ** 2 / 1.5, axis=1)
    y = y + 0.05 * rng.standard_normal(N)
    cov_N = gp.covariance.hyperparameter_count(D)
    noise_N = gp.noise.hyperparameter_count()
    mean_N = gp.mean.hyperparameter_count(D)
    hyp = np.zeros((Ns, cov_N + noise_N + mean_N))
    hyp[:, :D] = np.log(1.2) + 0.2 * rng.standard_normal((Ns, D))
    hyp[:, D] = np.log(2.0) + 0.2 * rng.standard_normal(Ns)
    hyp[:, cov_N : cov_N + noise_N] = ln_sn
    if mean_N >= 1:
        hyp[:, cov_N + noise_N] = 0.3 * rng.standard_normal(Ns)
    if mean_N > 1:
        hyp[
            :, cov_N + noise_N + 1 : cov_N + noise_N + 1 + D
        ] = 0.3 + 0.2 * rng.standard_normal((Ns, D))
        hyp[:, cov_N + noise_N + 1 + D :] = np.log(1.3) + 0.2 * (
            rng.standard_normal((Ns, D))
        )
    gp.update(X_new=X, y_new=y.reshape(-1, 1), hyp=hyp)
    return gp


def central_fd(f, x, h=1e-5):
    """Fourth-order central differences."""
    g = np.zeros_like(x, dtype=float)
    for i in range(x.size):
        e = np.zeros_like(x, dtype=float)
        e[i] = h
        g[i] = (-f(x + 2 * e) + 8 * f(x + e) - 8 * f(x - e) + f(x - 2 * e)) / (
            12 * h
        )
    return g


def matlab_negelcbo(theta, vp_in, gp, thetabnd, compute_grad=True):
    """Transcription of misc/negelcbo_vbmc.m (Ns = 0, beta = 0, no variance)
    with misc/vpbndloss.m, for the flags stored on vp_in.

    vp is passed by value (deep copy); THETA is assigned without any
    rescaling (negelcbo_vbmc.m:33-48). The eta entries of the soft-bound loss
    are left out, as PyVBMC does by deliberate change (sheet, P6, "The eta
    soft bound was removed"), so that only the sigma/lambda handling is
    compared. G and H come from PyVBMC's _gp_log_joint and entlb_vbmc, whose
    agreement with gplogjoint.m / entlb_vbmc.m is established elsewhere.
    """
    vp = copy.deepcopy(vp_in)
    D, K = vp.D, vp.K
    theta = np.asarray(theta, float)
    if vp.optimize_mu:
        vp.mu = theta[: D * K].reshape((D, K), order="F").copy()
        i0 = D * K
    else:
        i0 = 0
    if vp.optimize_sigma:
        vp.sigma = np.exp(theta[i0 : i0 + K]).reshape(1, -1)
        i0 += K
    if vp.optimize_lambd:
        vp.lambd = np.exp(theta[i0 : i0 + D]).reshape(-1, 1)
    if vp.optimize_weights:
        eta = theta[-K:].copy()
        vp.eta = eta.reshape(1, -1)
        w = np.exp(eta - eta.max())
        vp.w = (w / w.sum()).reshape(1, -1)
    flags = (
        (
            vp.optimize_mu,
            vp.optimize_sigma,
            vp.optimize_lambd,
            vp.optimize_weights,
        )
        if compute_grad
        else (False,) * 4
    )
    G, dG, *_ = _gp_log_joint(vp, gp, flags, 1, 1, 0)
    H, dH = entlb_vbmc(vp, flags, 1)
    F = -G - H
    dF = (-dG - dH) if compute_grad else None
    # vpbndloss.m
    i0 = 0
    if vp.optimize_mu:
        mu = theta[: D * K]
        i0 = D * K
    else:
        mu = vp.mu.ravel(order="F")
    if vp.optimize_sigma:
        lnsigma = theta[i0 : i0 + K]
        i0 += K
    else:
        lnsigma = np.log(vp.sigma.ravel())
    if vp.optimize_lambd:
        lnlambda = theta[i0 : i0 + D]
    else:
        lnlambda = np.log(vp.lambd.ravel())
    lnscale = lnlambda[:, None] + lnsigma[None, :]  # (D, K)
    ext, lb, ub = [], [], []
    nmu = D * K if vp.optimize_mu else 0
    nsc = D * K if (vp.optimize_sigma or vp.optimize_lambd) else 0
    if vp.optimize_mu:
        ext.append(mu)
    if nsc:
        ext.append(lnscale.ravel(order="F"))
    ext = np.concatenate(ext) if ext else np.zeros(0)
    lb = np.asarray(thetabnd["lb"], float)[: nmu + nsc]
    ub = np.asarray(thetabnd["ub"], float)[: nmu + nsc]
    L, dLext = _soft_bound_loss(ext, lb, ub, thetabnd["tol_con"], True)
    F += L
    if compute_grad:
        dL = []
        if vp.optimize_mu:
            dL.append(dLext[:nmu])
        if nsc:
            dls = dLext[nmu : nmu + nsc].reshape((D, K), order="F")
            if vp.optimize_sigma:
                dL.append(dls.sum(axis=0))
            if vp.optimize_lambd:
                dL.append(dls.sum(axis=1))
        if vp.optimize_weights:
            dL.append(np.zeros(K))
        dF = dF + np.concatenate(dL)
    if vp.optimize_weights:
        thr = thetabnd["weight_threshold"]
        w = vp.w.ravel()
        F += (
            np.sum(w * (w < thr) + thr * (w >= thr))
            * thetabnd["weight_penalty"]
        )
        if compute_grad:
            wg = thetabnd["weight_penalty"] * (w < thr)
            e = np.exp(vp.eta.ravel())
            J = -np.outer(e, e) / e.sum() ** 2 + np.diag(e / e.sum())
            dF[-K:] += J @ wg
    return (F, dF) if compute_grad else F
