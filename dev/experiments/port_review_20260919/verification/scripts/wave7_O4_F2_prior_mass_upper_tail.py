"""O4 F2: the prior mass inside the bounds, cdf(ub) - cdf(lb), rounds to 0
when both bounds lie far in the upper tail. Gaussian, Student's t and the
upper branches of the two smooth-box CDFs; the mirrored lower-tail cases;
what the fit then does; and the masses of the priors of a PyVBMC-built GP."""

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import scipy as sp
from gpyreg.f_min_fill import smoothbox_cdf, smoothbox_student_t_cdf
from wave7_O4_common import attach, banner, make_gp

banner()
warnings.simplefilter("ignore")

rng = np.random.default_rng(2)
N, D = 12, 1
X = rng.uniform(-1, 1, (N, D))
y = (np.sin(2 * X[:, 0])).reshape(-1, 1)


def gp_with(prior, lo, hi):
    gp = make_gp(D, "const", (1, 0, 0))
    attach(gp, X, y)
    b = gp.get_recommended_bounds()
    b["mean_const"] = (np.array([lo]), np.array([hi]))
    gp.set_bounds(b)
    pri = {k: None for k in b}
    pri["mean_const"] = prior
    gp.set_priors(pri)
    return gp


def true_mass(prior, lo, hi):
    """Mass from the far tail: survival functions above the centre, CDFs below."""
    kind, p = prior
    if hi <= 0:
        return true_mass(_mirror(prior), -hi, -lo)
    if kind == "gaussian":
        mu, s = p
        return sp.stats.norm.sf(lo, mu, s) - sp.stats.norm.sf(hi, mu, s)
    if kind == "student_t":
        mu, s, nu = p
        return sp.stats.t.sf(lo, nu, mu, s) - sp.stats.t.sf(hi, nu, mu, s)
    if kind == "smoothbox":
        a, b_, s = p
        C = 1 + (b_ - a) / (s * np.sqrt(2 * np.pi))
        return (sp.stats.norm.sf(lo, b_, s) - sp.stats.norm.sf(hi, b_, s)) / C
    if kind == "smoothbox_student_t":
        a, b_, s, nu = p
        c = np.exp(
            sp.special.gammaln((nu + 1) / 2) - sp.special.gammaln(nu / 2)
        ) / (s * np.sqrt(nu * np.pi))
        C = 1 + (b_ - a) * c
        return (
            sp.stats.t.sf(lo, nu, b_, s) - sp.stats.t.sf(hi, nu, b_, s)
        ) / C


def _mirror(prior):
    kind, p = prior
    if kind in ("gaussian", "student_t"):
        return (kind, (-p[0],) + tuple(p[1:]))
    return (kind, (-p[1], -p[0]) + tuple(p[2:]))


cases = [
    (("gaussian", (0.0, 1.0)), 9.0, 10.0),
    (("gaussian", (0.0, 1.0)), -10.0, -9.0),
    (("gaussian", (0.0, 1.0)), 8.2, 9.0),
    (("gaussian", (0.0, 1.0)), 8.3, 9.0),
    (("gaussian", (0.0, 1.0)), 8.0, np.inf),
    (("student_t", (0.0, 1.0, 3)), 1e6, 2e6),
    (("student_t", (0.0, 1.0, 3)), -2e6, -1e6),
    (("smoothbox", (0.0, 1.0, 1.0)), 10.0, 11.0),
    (("smoothbox", (0.0, 1.0, 1.0)), -10.0, -9.0),
    (("smoothbox_student_t", (0.0, 1.0, 1.0, 3)), 1e6, 2e6),
]
print(
    "prior                                    bounds             stored mass   true mass     log_posterior(mid)"
)
for prior, lo, hi in cases:
    gp = gp_with(prior, lo, hi)
    stored = gp.normalization_constants[-1]
    tm = true_mass(prior, lo, hi)
    mid = lo + 0.5 if np.isinf(hi) else 0.5 * (lo + hi)
    hyp = np.array([0.0, np.log(np.std(y)), np.log(0.1), mid])
    lp = gp.log_posterior(hyp)
    print(
        f"{str(prior):40s} [{lo:9.3g},{hi:9.3g}]  {stored:.4e}  {tm:.4e}  {lp}"
    )

print("\nfit with the Gaussian prior on mean_const, bounds [9, 10]:")
gp = gp_with(("gaussian", (0.0, 1.0)), 9.0, 10.0)
try:
    hyp, res, _ = gp.fit(
        options={"n_samples": 0, "init_N": 64, "opts_N": 2},
        rng=np.random.default_rng(0),
    )
    print(
        f"   optimizer nit={res.nit} fun={res.fun} msg={res.message!r} hyp={hyp[0]}"
    )
    h_res = hyp[0]

    def my_obj(h):
        lz, dlz = gp.log_likelihood(h, compute_grad=True)
        lp = sp.stats.norm.logpdf(h[-1], 0.0, 1.0)
        dlz = dlz.copy()
        dlz[-1] += -h[-1]
        return -(lz + lp), -dlz

    best = None
    for start in (h_res, np.array([0.0, np.log(np.std(y)), np.log(0.1), 9.5])):
        r2 = sp.optimize.minimize(
            my_obj,
            start,
            jac=True,
            method="L-BFGS-B",
            bounds=list(zip(gp.lower_bounds, gp.upper_bounds)),
        )
        if best is None or r2.fun < best.fun:
            best = r2
    print(
        f"   finite objective (log lik + log N(m;0,1)) at the fit's result: {-my_obj(h_res)[0]:.4f}; "
        f"its optimum over the same box: {-best.fun:.4f} at {np.round(best.x, 3)}"
    )
except Exception as e:  # noqa: BLE001
    print(f"   fit raised {type(e).__name__}: {e}")
gp = gp_with(("gaussian", (0.0, 1.0)), -10.0, -9.0)
hyp, res, _ = gp.fit(
    options={"n_samples": 0, "init_N": 64, "opts_N": 2},
    rng=np.random.default_rng(0),
)
print(
    f"mirrored [-10, -9]: optimizer nit={res.nit} fun={res.fun:.4f} msg={res.message!r}"
)

print("\nPyVBMC-built GPs: masses of their priors inside the filled bounds")
import gpyreg as gpr

from pyvbmc import VBMC
from pyvbmc.vbmc.gaussian_process_train import (
    _cov_identifier_to_covariance_function,
    _gp_hyp,
    _meanfun_name_to_mean_function,
)

for level in (0, 1, 2):
    for Dd in (1, 3):
        f = (
            (lambda x: (-0.5 * np.sum(x**2), 0.3))
            if level == 2
            else (lambda x: -0.5 * np.sum(x**2))
        )
        opts = {"display": "off"}
        if level == 1:
            opts["uncertainty_handling"] = True
        if level == 2:
            opts["specify_target_noise"] = True
        vb = VBMC(
            f,
            np.zeros((1, Dd)),
            np.full((1, Dd), -np.inf),
            np.full((1, Dd), np.inf),
            -np.ones((1, Dd)),
            np.ones((1, Dd)),
            opts,
        )
        os_ = vb.optim_state
        nf = os_["gp_noise_fun"]
        g = gpr.GP(
            D=Dd,
            covariance=_cov_identifier_to_covariance_function(
                os_["gp_cov_fun"]
            ),
            mean=_meanfun_name_to_mean_function(os_["gp_mean_fun"]),
            noise=gpr.noise_functions.GaussianNoise(
                constant_add=nf[0] == 1,
                user_provided_add=nf[1] > 0,
                scale_user_provided=nf[1] == 2,
            ),
        )
        r = np.random.default_rng(level + 10 * Dd)
        Xp = r.uniform(-1, 1, (20, Dd))
        yp = (-0.5 * np.sum(Xp**2, 1)).reshape(-1, 1)
        s2 = (
            None
            if level == 0
            else np.full((20, 1), 1.0 if level == 1 else 0.09)
        )
        os_["N"], os_["stop_sampling"] = 20, 0
        g, _, _ = _gp_hyp(
            os_, vb.options, os_["plb_tran"], os_["pub_tran"], g, Xp, yp
        )
        Xc, yc, sc = g._convert_shapes(Xp, yp, s2)
        g.X, g.y, g.s2 = Xc, yc, sc
        g.set_bounds(g.get_recommended_bounds(g.lower_bounds, g.upper_bounds))
        # masses as fit sees them (df of the Student's t priors is 3, set explicitly)
        nc = g.normalization_constants
        has = np.isfinite(g.hyper_priors["mu"])
        print(
            f"  L{level} D={Dd}: masses of the coordinates with a prior: {np.round(nc[has], 4)}"
        )
