"""P3-10: the first question, checked against my own MATLAB transcription.

Converts the stored noisy oracle state into an uncertainty-level-1 state
(noise function [1 2 0], `S = 1/sqrt(n_evals)`) and compares, on the same
inputs:

  * `sn2_new` and `gp_length_scale` as `active_sample.py:327-363` builds
    them, against a transcription of `private/activesample_vbmc.m:159-183`
    with `gplite/gplite_noisefun.m:176-194` inlined;
  * the four pointwise acquisitions and VIQR/IMIQR against transcriptions
    of `acq/acqf_vbmc.m`, `acqflog_vbmc.m`, `acqus_vbmc.m`,
    `acqfsn2_vbmc.m`, `acqviqr_vbmc.m`, `acqimiqr_vbmc.m` and
    `acqwrapper_vbmc.m`, written here from the MATLAB source.

Unlike the wave-4 reviewer's check, the inserted log-multiplier is NOT
zero: with `exp(h1) = 1` level 1 and level 2 compute the same number on a
state whose `S^2 * n_evals` is 1, so a mishandled multiplier would be
invisible. Both settings are run.
"""
import sys

import numpy as np
from scipy.linalg import solve_triangular
from scipy.spatial.distance import cdist
from scipy.stats import norm

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from wave4_P3_common import banner, load  # noqa: E402

banner()

import gpyreg as gpr  # noqa: E402

from pyvbmc.acquisition_functions import (  # noqa: E402
    AcqFcn,
    AcqFcnIMIQR,
    AcqFcnLog,
    AcqFcnNoisy,
    AcqFcnVanilla,
    AcqFcnVIQR,
)
from pyvbmc.testing.oracles._oracles import (  # noqa: E402
    prepare_gp_for_acq,
    prepare_importance_sampling,
)

REALMAX = np.finfo(np.float64).max
REALMIN = np.finfo(np.float64).tiny
SEED = 20260904


# ---------------------------------------------------------------- MATLAB
def sq_dist(a, b):
    """The `sq_dist` subfunction inlined in every acq/*.m file."""
    n, m = a.shape[0], b.shape[0]
    mu = (m / (n + m)) * np.mean(b, 0) + (n / (n + m)) * np.mean(a, 0)
    a = a - mu
    b = b - mu
    C = (a * a).sum(1)[:, None] + ((b * b).sum(1)[None, :] - 2 * a @ b.T)
    return np.maximum(C, 0)


def noisefun(hyp, X, noise_fun, y, s2):
    """gplite/gplite_noisefun.m:176-194, the three switches."""
    N = X.shape[0]
    i = 0
    if noise_fun[0] == 1:
        sn2 = np.full((N, 1), np.exp(2 * hyp[i]))
        i += 1
    else:
        sn2 = np.full((N, 1), np.spacing(1.0))
    if noise_fun[1] == 1:
        sn2 = sn2 + s2
    elif noise_fun[1] == 2:
        sn2 = sn2 + np.exp(hyp[i]) * s2
        i += 1
    return sn2


def activesample_noise(gp, S, n_evals, has_S, noise_fun):
    """private/activesample_vbmc.m:159-183."""
    Ns = len(gp.posteriors)
    cov_N = gp.covariance.hyperparameter_count(gp.D)
    noise_N = gp.noise.hyperparameter_count()
    sn2new = np.zeros((gp.X.shape[0], Ns))
    for s in range(Ns):
        h = gp.posteriors[s].hyp[cov_N : cov_N + noise_N]
        s2 = (S**2) * n_evals if has_S else np.zeros((gp.X.shape[0], 1))
        sn2new[:, s] = noisefun(h, gp.X, noise_fun, gp.y, s2).ravel()
    ln_ell = np.zeros((gp.D, Ns))
    for s in range(Ns):
        ln_ell[:, s] = gp.posteriors[s].hyp[: gp.D]
    ell = np.exp(ln_ell.mean(1))
    return sn2new.mean(1), ell, gp.X / ell


def ml_sn2(Xs, ell_scale, X_rescaled, sn2_new):
    pos = np.argmin(sq_dist(Xs / ell_scale, X_rescaled), axis=1)
    return sn2_new[pos][:, None]


def ml_acqf(p, fbar, vtot, z, **kw):
    return -vtot * np.exp(fbar - z) * p


def ml_acqflog(p, fbar, vtot, z, **kw):
    return -(np.log(vtot) + fbar - z + np.log(p))


def ml_acqus(p, fbar, vtot, z, **kw):
    return -vtot * p**2


def ml_acqfsn2(p, fbar, vtot, z, sn2=None, **kw):
    return -vtot * (1 - sn2 / (vtot + sn2)) * np.exp(fbar - z) * p


def ml_acqviqr(Xs, gp, ais, fs2, sn2, u):
    Nx, D = Xs.shape
    Ns = fs2.shape[1]
    ys2 = fs2 + sn2
    Xa = ais["X"]
    acq = np.zeros((Nx, Ns))
    for s in range(Ns):
        hyp = gp.posteriors[s].hyp
        ell = np.exp(hyp[0:D])
        sf2 = np.exp(2 * hyp[D])
        Xs_ell = Xs / ell
        Ks = sf2 * np.exp(-sq_dist(gp.X / ell, Xs_ell) / 2)  # (N,Nx)
        Ka = sf2 * np.exp(-sq_dist(Xs_ell, Xa / ell) / 2)  # (Nx,Na)
        Ctmp = ais["C_tmp"][s, :, :]
        C = Ka - Ks.T @ Ctmp if gp.posteriors[s].L_chol else Ka + Ks.T @ Ctmp
        tau2 = C**2 / ys2[:, s][:, None]
        s_pred = np.sqrt(np.maximum(ais["f_s2"][:, s][None, :] - tau2, 0))
        with np.errstate(divide="ignore", invalid="ignore"):
            zz = u * s_pred + np.log1p(-np.exp(-2 * u * s_pred))
            lnmax = np.max(zz, axis=1)
            acq[:, s] = np.log(np.sum(np.exp(zz - lnmax[:, None]), 1)) + lnmax
    if Ns > 1:
        M = np.max(acq, axis=1)
        acq = M + np.log(np.sum(np.exp(acq - M[:, None]), 1) / Ns)
        acq = acq[:, None]
    return acq


def ml_acqimiqr(Xs, gp, ais, fs2, sn2, u):
    Nx, D = Xs.shape
    Ns = fs2.shape[1]
    ys2 = fs2 + sn2
    multi = ais["X"].ndim == 3
    acq = np.zeros((Nx, Ns))
    for s in range(Ns):
        hyp = gp.posteriors[s].hyp
        L = gp.posteriors[s].L
        sn2_eff = 1 / gp.posteriors[s].sW[0] ** 2
        Xa = ais["X"][s] if multi else ais["X"]
        ell = np.exp(hyp[0:D])
        sf2 = np.exp(2 * hyp[D])
        Ks = sf2 * np.exp(-sq_dist(gp.X / ell, Xs / ell) / 2)  # (N,Nx)
        Ka = sf2 * np.exp(-sq_dist(Xa / ell, Xs / ell) / 2)  # (Na,Nx)
        Kax = ais["K_Xa_X"][s, :, :]
        if gp.posteriors[s].L_chol:
            C = (
                Ka.T
                - Ks.T
                @ solve_triangular(
                    L,
                    solve_triangular(L, Kax.T, trans=1, check_finite=False),
                    check_finite=False,
                )
                / sn2_eff
            )
        else:
            C = Ka.T + Ks.T @ (L @ Kax.T)
        tau2 = C**2 / ys2[:, s][:, None]
        s_pred = np.sqrt(np.maximum(ais["f_s2"][:, s][None, :] - tau2, 0))
        lnw = ais["ln_weights"][s, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            zz = lnw + u * s_pred + np.log1p(-np.exp(-2 * u * s_pred))
            lnmax = np.max(zz, axis=1)
            acq[:, s] = np.log(np.sum(np.exp(zz - lnmax[:, None]), 1)) + lnmax
    if Ns > 1:
        M = np.max(acq, axis=1)
        acq = M + np.log(np.sum(np.exp(acq - M[:, None]), 1) / Ns)
        acq = acq[:, None]
    return acq


def ml_wrapper(acq, vtot, log_flag, os_, vp, Xs):
    """acq/acqwrapper_vbmc.m:34-52 with transpose_flag = 0."""
    acq = np.asarray(acq, float).reshape(-1, 1)
    if os_.get(
        "variance_regularized_acqfcn",
        os_.get("variance_regularized_acq_fcn", False),
    ):
        tol = os_["tol_gp_var"]
        idx = (vtot < tol).ravel()
        if np.any(idx):
            if log_flag:
                acq[idx, 0] += tol / vtot[idx, 0] - 1
            else:
                acq[idx, 0] *= np.exp(-(tol / vtot[idx, 0] - 1))
    acq = np.where(np.isnan(acq), -REALMAX, np.maximum(acq, -REALMAX))
    X_orig = vp.parameter_transformer.inverse(Xs)
    bad = np.any(X_orig < os_["lb_eps_orig"], 1) | np.any(
        X_orig > os_["ub_eps_orig"], 1
    )
    acq[bad, 0] = np.inf
    return acq.ravel()


# ---------------------------------------------------------------- state
def level1_state(log_mult):
    st = load("rosenbrock_D2_noise1_viqr", seed=SEED)
    gp0, vp, logger, os_ = (
        st["gp"],
        st["vp"],
        st["logger"],
        st["optim_state"],
    )
    live = logger.X_flag
    logger.uncertainty_handling_level = 1
    logger.S[live] = 1.0 / np.sqrt(logger.n_evals[live])
    os_["uncertainty_handling_level"] = 1
    os_["gp_noise_fun"] = [1, 2, 0]
    gp = gpr.GP(
        D=gp0.D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(
            constant_add=True, user_provided_add=True, scale_user_provided=True
        ),
    )
    gp.update(
        X_new=logger.X[live],
        y_new=logger.y[live],
        s2_new=logger.S[live] ** 2,
        compute_posterior=False,
    )
    cov_N = gp.covariance.hyperparameter_count(gp.D)
    old = np.array([p.hyp for p in gp0.posteriors])
    hyp = np.concatenate(
        [
            old[:, : cov_N + 1],
            np.full((old.shape[0], 1), log_mult),
            old[:, cov_N + 1 :],
        ],
        1,
    )
    gp.update(hyp=hyp, compute_posterior=True)
    st["gp"] = gp
    return st


for log_mult in (0.0, np.log(2.3)):
    st = level1_state(log_mult)
    gp, vp, logger, os_ = (
        st["gp"],
        st["vp"],
        st["logger"],
        st["optim_state"],
    )
    Xs = np.array(st["cand"]["Xs"], dtype=float)
    prepare_gp_for_acq(gp, logger, os_)
    live = logger.X_flag
    sn2_ml, ell_ml, Xr_ml = activesample_noise(
        gp, logger.S[live], logger.n_evals[live], True, [1, 2, 0]
    )
    print(
        f"\n=== level 1, exp(h1) = {np.exp(log_mult):.3f} ===\n"
        f"sn2_new: python {gp.temporary_data['sn2_new'][:3]} "
        f"max|py-ml| {np.abs(gp.temporary_data['sn2_new'] - sn2_ml).max():.3e}"
        f"  bit-identical "
        f"{np.array_equal(gp.temporary_data['sn2_new'], sn2_ml)}"
    )
    print(
        "gp_length_scale bit-identical",
        np.array_equal(os_["gp_length_scale"], ell_ml),
        " X_rescaled bit-identical",
        np.array_equal(gp.temporary_data["X_rescaled"], Xr_ml),
    )

    z = logger.y_max
    for cls in (
        AcqFcn,
        AcqFcnLog,
        AcqFcnVanilla,
        AcqFcnNoisy,
        AcqFcnVIQR,
        AcqFcnIMIQR,
    ):
        acq = cls()
        if acq.acq_info.get("importance_sampling"):
            prepare_importance_sampling(st, acq, SEED)
        py = acq(Xs.copy(), gp, vp, logger, os_)

        # The MATLAB side, on the same (already integer-snapped) inputs.
        f_mu, f_s2 = gp.predict(x_star=Xs, separate_samples=True)
        Ns = f_mu.shape[1]
        fbar = np.sum(f_mu, 1, keepdims=True) / Ns
        vbar = np.sum(f_s2, 1, keepdims=True) / Ns
        vf = (
            np.sum((f_mu - fbar) ** 2, 1, keepdims=True) / (Ns - 1)
            if Ns > 1
            else 0
        )
        vtot = vf + vbar
        p = np.maximum(vp.pdf(Xs, orig_flag=False), REALMIN).reshape(-1, 1)
        sn2 = ml_sn2(Xs, ell_ml, Xr_ml, sn2_ml)
        u = norm.ppf(0.75)
        if cls is AcqFcn:
            raw = ml_acqf(p, fbar, vtot, z)
        elif cls is AcqFcnLog:
            raw = ml_acqflog(p, fbar, vtot, z)
        elif cls is AcqFcnVanilla:
            raw = ml_acqus(p, fbar, vtot, z)
        elif cls is AcqFcnNoisy:
            raw = ml_acqfsn2(p, fbar, vtot, z, sn2=sn2)
        elif cls is AcqFcnVIQR:
            raw = ml_acqviqr(
                Xs, gp, os_["active_importance_sampling"], f_s2, sn2, u
            )
        else:
            raw = ml_acqimiqr(
                Xs, gp, os_["active_importance_sampling"], f_s2, sn2, u
            )
        ml = ml_wrapper(
            raw, vtot, acq.acq_info.get("log_flag", False), os_, vp, Xs
        )
        fin = np.isfinite(py) & np.isfinite(ml)
        d = np.abs(py - ml)[fin]
        rel = d / np.maximum(np.abs(ml[fin]), 1e-300)
        print(
            f"  {cls.__name__:14s} bit-identical "
            f"{np.array_equal(py, ml)!s:5s} maxabs {d.max():.3e} "
            f"maxrel {rel.max():.3e} argmin py {np.argmin(py)} "
            f"ml {np.argmin(ml)}"
        )
