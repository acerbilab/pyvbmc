"""O1 first question: the configurations the shipped FD tests leave out.

For zero, constant and negative-quadratic means, D in {1, 2}, K in {1, 2, 3},
Ns in {1, 3} GP hyperparameter samples:
  1. dG of _gp_log_joint against fourth-order central differences, VP built
     from raw theta with no gauge fix (jacobian_flag on);
  2. dF of _neg_elcbo (deterministic entropy) with soft bounds active and,
     for K >= 2, one weight below the weight-penalty threshold, against FD
     through a fresh posterior per evaluation;
  3. the per-component values I_sk against an independent quadrature of the
     GP posterior mean (gpyreg predict, separate samples) over each
     component: a dense grid for D = 1, Gauss-Hermite 80x80 for D = 2.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from wave7_O1_common import OPTS, banner, build_gp, central_fd

from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.vbmc.variational_optimization import _gp_log_joint, _neg_elcbo

banner()


def vp_raw(theta, D, K):
    vp = VariationalPosterior(D, K, rng=0)
    vp.mu = theta[: D * K].reshape((D, K), order="F").copy()
    vp.sigma = np.exp(theta[D * K : D * K + K]).reshape(1, -1)
    vp.lambd = np.exp(theta[D * K + K : D * K + K + D]).reshape(-1, 1)
    eta = theta[-K:] - np.max(theta[-K:])
    vp.eta = eta.reshape(1, -1)
    vp.w = (np.exp(eta) / np.sum(np.exp(eta))).reshape(1, -1)
    return vp


def raw_theta(D, K, rng, small_w):
    mu = rng.uniform(-1.0, 1.0, size=D * K)
    ln_sigma = np.log(0.4 + 0.5 * rng.random(K))
    ln_lambd = np.log(0.7 + 0.6 * rng.random(D))
    eta = 0.4 * rng.standard_normal(K)
    if small_w and K >= 2:
        eta[0] = eta[1:].max() - 3.5
    return np.concatenate([mu, ln_sigma, ln_lambd, eta])


def quad_I(gp, mu_k, s_k):
    """E_{N(mu_k, diag s_k^2)}[m_s(x)] for each hyperparameter sample s."""
    D = mu_k.size
    if D == 1:
        x = np.linspace(mu_k[0] - 12 * s_k[0], mu_k[0] + 12 * s_k[0], 20001)
        m, _ = gp.predict(x[:, None], separate_samples=True)
        p = np.exp(-0.5 * ((x - mu_k[0]) / s_k[0]) ** 2) / (
            np.sqrt(2 * np.pi) * s_k[0]
        )
        return np.trapezoid(m * p[:, None], x, axis=0)
    t, wt = np.polynomial.hermite_e.hermegauss(80)
    wt = wt / wt.sum()
    T1, T2 = np.meshgrid(t, t, indexing="ij")
    W = np.outer(wt, wt).ravel()
    X = np.column_stack(
        [mu_k[0] + s_k[0] * T1.ravel(), mu_k[1] + s_k[1] * T2.ravel()]
    )
    m, _ = gp.predict(X, separate_samples=True)
    return W @ m


worst = {"dG": 0.0, "dF": 0.0, "I": 0.0}
rng = np.random.default_rng(2026)
rows = []
for mean in ("zero", "const", "negquad"):
    for D in (1, 2):
        gp_by_Ns = {
            Ns: build_gp(D, N=18, Ns=Ns, seed=100 + D * 10 + Ns, mean=mean)
            for Ns in (1, 3)
        }
        for K in (1, 2, 3):
            for Ns in (1, 3):
                gp = gp_by_Ns[Ns]
                theta = raw_theta(D, K, rng, small_w=True)

                # 1. dG
                def fG(th):
                    return _gp_log_joint(vp_raw(th, D, K), gp, False)[0]

                dG = _gp_log_joint(vp_raw(theta, D, K), gp, True)[1]
                fdG = central_fd(fG, theta, 1e-5)
                eG = np.max(np.abs(dG - fdG)) / max(1.0, np.max(np.abs(fdG)))

                # 2. dF with bounds and weight penalty active
                vpb = VariationalPosterior(D, K, rng=0)
                bnd = vpb.get_bounds(gp.X, OPTS, K)
                th = theta.copy()
                th[0] = bnd["lb"][0] - 0.3  # a mean below its soft bound
                th[
                    D * K
                ] = 1.9  # ln sigma_1 large: a log scale above its bound

                def fF(t):
                    v = VariationalPosterior(D, K, rng=0)
                    return _neg_elcbo(t, gp, v, 0.0, 0, False, False, bnd)[0]

                vF = VariationalPosterior(D, K, rng=0)
                F, dF = _neg_elcbo(th, gp, vF, 0.0, 0, True, False, bnd)[:2]
                w = vF.w.ravel()
                thr = bnd["weight_threshold"]
                fdF = central_fd(fF, th, 1e-5)
                eF = np.max(np.abs(dF - fdF)) / max(1.0, np.max(np.abs(fdF)))
                pen_active = bool(np.any(w < thr))

                # 3. I_sk against quadrature
                vpv = vp_raw(theta, D, K)
                I_sk = _gp_log_joint(vpv, gp, False, True, True, False, True)[
                    5
                ]
                s = vpv.sigma.ravel()[None, :] * vpv.lambd.ravel()[:, None]
                I_ref = np.column_stack(
                    [quad_I(gp, vpv.mu[:, k], s[:, k]) for k in range(K)]
                )  # (Ns, K)
                eI = np.max(np.abs(I_sk - I_ref)) / max(
                    1.0, np.max(np.abs(I_ref))
                )

                worst["dG"] = max(worst["dG"], eG)
                worst["dF"] = max(worst["dF"], eF)
                worst["I"] = max(worst["I"], eI)
                rows.append(
                    f"{mean:7s} D={D} K={K} Ns={Ns} | dG {eG:.1e} | dF {eF:.1e}"
                    f" (min w {w.min():.3f}, thr {thr:.3f}, penalty active "
                    f"{pen_active}) | I_sk vs quad {eI:.1e}"
                )
                print(rows[-1], flush=True)

print("\nworst relative errors:", {k: f"{v:.2e}" for k, v in worst.items()})

# The penalty gradient in isolation, K = 2, one weight below threshold:
D, K = 2, 2
gp = build_gp(D, N=18, Ns=3, seed=7)
bnd = VariationalPosterior(D, K).get_bounds(gp.X, OPTS, K)
th = raw_theta(D, K, np.random.default_rng(5), small_w=True)
nopen = dict(bnd)
nopen["weight_penalty"] = 0.0


def f_pen(t):
    a = _neg_elcbo(
        t, gp, VariationalPosterior(D, K), 0.0, 0, False, False, bnd
    )[0]
    b = _neg_elcbo(
        t, gp, VariationalPosterior(D, K), 0.0, 0, False, False, nopen
    )[0]
    return a - b


da = _neg_elcbo(th, gp, VariationalPosterior(D, K), 0.0, 0, True, False, bnd)[
    1
]
db = _neg_elcbo(
    th, gp, VariationalPosterior(D, K), 0.0, 0, True, False, nopen
)[1]
print(
    "penalty-only gradient, eta block: analytic",
    (da - db)[-K:],
    " FD",
    central_fd(f_pen, th, 1e-5)[-K:],
)
