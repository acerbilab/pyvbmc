"""Settles the two claims about the factorization scale in `GP.quad`:

(a) `quad` recomputes `sl = min(sn2) * sn2_mult` from the current training
    noise (gaussian_process.py:2193-2196) instead of reading the stored
    `Posterior.sl`, so after a rank-one update that appends a point with a
    smaller total noise the integral variance collapses to machine epsilon;
(b) `gplite_quad.m:66-67` normalizes by `exp(2*hyp(Ncov+1)) * sn2_mult`, the
    constant noise hyperparameter alone, which parts from the scale `L`
    carries under heteroskedastic noise; gpyreg's value is the right one.

The reference in both cases is an independent grid quadrature of the latent
predictive covariance against the same Gaussian measure.

Run: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python G2_2_quad_scale.py
Needs `matlab_gplite_g2.py` next to it.
"""

import os
import sys

import numpy as np
import scipy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gpyreg as gpr
import wave6_G2_matlab_gplite_g2 as M

print("gpyreg:", gpr.__file__)
np.set_printoptions(precision=12, linewidth=160)

D = 1
n_cov, n_mean = D + 1, 1 + 2 * D


def make(params, X, y, s2, hyp_noise_vals):
    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(
            constant_add=params[0] == 1,
            user_provided_add=params[1] > 0,
            scale_user_provided=params[1] == 2,
        ),
    )
    n_noise = gp.noise.hyperparameter_count()
    h = np.zeros(n_cov + n_noise + n_mean)
    h[0:D] = np.log(0.4)
    h[D] = np.log(1.5)
    h[n_cov : n_cov + n_noise] = hyp_noise_vals
    h[n_cov + n_noise] = 0.3
    h[n_cov + n_noise + 1 : n_cov + n_noise + 1 + D] = 0.0
    h[n_cov + n_noise + 1 + D :] = np.log(1.5)
    gp.update(X_new=X, y_new=y, s2_new=s2, hyp=h[None, :])
    return gp


def grid_truth(gp, mu, sigma, n=4001, span=8.0):
    """int f(x) N(x; mu, sigma^2) dx : mean and variance from predict_full."""
    g = np.linspace(mu - span * sigma, mu + span * sigma, n).reshape(-1, 1)
    w = np.exp(-0.5 * ((g - mu) / sigma) ** 2).ravel()
    w /= np.trapezoid(w, g.ravel())
    m, C = gp.predict_full(g, add_noise=False)
    dx = g.ravel()
    mean = np.trapezoid(w * m[:, 0], dx)
    # var = w' C w  as a double trapezoid
    inner = np.trapezoid(w[None, :] * C[:, :, 0], dx, axis=1)
    var = np.trapezoid(w * inner, dx)
    return mean, var


rng = np.random.default_rng(4)
X = np.sort(rng.uniform(-1.5, 1.5, size=(14, 1)), axis=0)
y = np.sin(3 * X) + 0.05 * rng.standard_normal((14, 1))

print("\n=== A. heteroskedastic noise, fresh posterior: gpyreg vs MATLAB ===")
s2_tr = np.full((14, 1), 0.05)
s2_tr[0] = 0.3
gp = make((1, 1, 0), X, y, s2_tr, [np.log(1e-2)])
post = gp.posteriors[0]
sn2 = gp.noise.compute(post.hyp[n_cov : n_cov + 1], X, y, s2_tr)
print(
    "  stored sl =",
    post.sl,
    " min(sn2)*mult =",
    float(np.min(sn2) * post.sn2_mult),
    " exp(2*h_noise)*mult =",
    float(np.exp(2 * post.hyp[n_cov]) * post.sn2_mult),
)
mposts = [M.core_posterior(post.hyp, X, y, (1, 1, 0), 4, s2_tr)]
for mu, sig in [(0.0, 0.5), (0.4, 0.3), (-0.7, 0.6)]:
    Fg, Vg = gp.quad(np.array([[mu]]), np.array([[sig]]), compute_var=True)
    Fm, Vm = M.quad(
        mposts, X, y, (1, 1, 0), 4, [[mu]], [[sig]], compute_var=True
    )
    mt, vt = grid_truth(gp, mu, sig)
    print(
        f"  mu={mu:+.1f} sigma={sig:.1f}:"
        f" mean gpyreg={Fg[0,0]:.12f} MATLAB={Fm[0,0]:.12f}"
        f" grid={mt:.12f}"
    )
    print(
        f"      var  gpyreg={Vg[0,0]:.12e} MATLAB={Vm[0,0]:.12e}"
        f" grid={vt:.12e}"
    )

print("\n=== B. constant noise only: the two scales coincide ===")
gp2 = make((1, 0, 0), X, y, None, [np.log(0.1)])
post2 = gp2.posteriors[0]
mposts2 = [M.core_posterior(post2.hyp, X, y, (1, 0, 0), 4, None)]
for mu, sig in [(0.0, 0.5), (0.4, 0.3)]:
    Fg, Vg = gp2.quad(np.array([[mu]]), np.array([[sig]]), compute_var=True)
    Fm, Vm = M.quad(
        mposts2, X, y, (1, 0, 0), 4, [[mu]], [[sig]], compute_var=True
    )
    mt, vt = grid_truth(gp2, mu, sig)
    print(
        f"  mu={mu:+.1f}: var gpyreg={Vg[0,0]:.12e}"
        f" MATLAB={Vm[0,0]:.12e} grid={vt:.12e}"
    )

print("\n=== C. after a rank-one update with a lower-noise point ===")
s2_tr3 = np.full((14, 1), 0.05)
gp3 = make((1, 1, 0), X, y, s2_tr3, [np.log(1e-3)])
p3 = gp3.posteriors[0]
print("  before: stored sl =", p3.sl)
x_new = np.array([[0.9]])
y_new = np.array([[np.sin(2.7)]])
gp3.update(X_new=x_new, y_new=y_new, s2_new=np.array([[0.0]]))
p3 = gp3.posteriors[0]
sn2_now = gp3.noise.compute(p3.hyp[n_cov : n_cov + 1], gp3.X, gp3.y, gp3.s2)
print(
    "  after : stored sl =",
    p3.sl,
    " recomputed min(sn2)*mult =",
    float(np.min(sn2_now) * p3.sn2_mult),
    " ratio =",
    float(p3.sl / (np.min(sn2_now) * p3.sn2_mult)),
)
# A GP recomputed from scratch on the same data and hyperparameters
gp3b = make((1, 1, 0), gp3.X, gp3.y, gp3.s2, [np.log(1e-3)])
gp3b.update(hyp=p3.hyp[None, :])
print("  recomputed-from-scratch stored sl =", gp3b.posteriors[0].sl)
print(
    "  |alpha(rank-one) - alpha(full)| =",
    np.max(np.abs(p3.alpha - gp3b.posteriors[0].alpha)),
)
for mu, sig in [(0.0, 0.5), (0.4, 0.3), (-0.7, 0.6)]:
    Fa, Va = gp3.quad(np.array([[mu]]), np.array([[sig]]), compute_var=True)
    Fb, Vb = gp3b.quad(np.array([[mu]]), np.array([[sig]]), compute_var=True)
    mt, vt = grid_truth(gp3, mu, sig)
    print(
        f"  mu={mu:+.1f}: mean rank-one={Fa[0,0]:.12f} full={Fb[0,0]:.12f}"
        f" grid={mt:.12f}"
    )
    print(
        f"      var  rank-one={Va[0,0]:.12e} full={Vb[0,0]:.12e}"
        f" grid={vt:.12e}"
    )

print("\n=== D. the same quad on the rank-one GP with the stored sl ===")
# Recompute quad's variance by hand, with the stored sl in place of the
# recomputed one, to show that only the scale is wrong.
p = gp3.posteriors[0]
hyp = p.hyp
Xall = gp3.X
ell = np.exp(hyp[0:D])
ln_sf2 = 2 * hyp[D]
sum_lnell = np.sum(hyp[0:D])
for mu, sig in [(0.0, 0.5), (0.4, 0.3), (-0.7, 0.6)]:
    mu_a = np.array([[mu]])
    sig_a = np.array([[sig]])
    tau = np.sqrt(sig_a**2 + ell**2)
    lnnf = ln_sf2 + sum_lnell - np.sum(np.log(tau), 1)
    sd2 = ((mu_a[:, 0] - Xall[:, 0].reshape(-1, 1)).T / tau[:, 0:1]) ** 2
    z = np.exp(lnnf.reshape(-1, 1) - 0.5 * sd2)
    tau_kk = np.sqrt(2 * sig_a**2 + ell**2)
    nf_kk = np.exp(ln_sf2 + sum_lnell - np.sum(np.log(tau_kk), 1))
    tmp = sp.linalg.solve_triangular(p.L, z.T, trans=1)
    invKzk = sp.linalg.solve_triangular(p.L, tmp, trans=0) / p.sl
    J = nf_kk - np.sum(z * invKzk.T, 1)
    _, vt = grid_truth(gp3, mu, sig)
    print(
        f"  mu={mu:+.1f}: J_kk with STORED sl = {J[0]:.12e}"
        f"   grid = {vt:.12e}"
    )
