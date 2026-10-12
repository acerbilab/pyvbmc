"""Settles whether the smooth-box normalization constant of
``GP.__compute_log_priors`` is subscripted with the mask it multiplies.

``gpyreg/gaussian_process.py:1479`` builds ``masks["C_sb"]`` with one entry
per smooth-box coordinate; lines 1541-1551 then multiply it by
``sigma[sb_idx_b | sb_idx_a]`` and ``sigma[sb_idx_btw]``, which hold one
entry per smooth-box coordinate *on that side of the box*.  Lines
1579-1601 do the same with ``C_sb_t``.  (G1 internal F2.)

The script evaluates ``log_posterior - log_likelihood`` against an
independent evaluation of the smooth-box and smooth-box-Student-t log
densities written from their definitions, for D = 1..4 and for every
placement of the coordinates relative to (a, b).  There is no MATLAB
counterpart: ``gplite_hypprior.m`` has no smooth-box family.
"""

import gpyreg as gpr
import numpy as np
import scipy as sp

print("gpyreg:", gpr.__file__)


def sb_logpdf(x, sigma, a, b):
    """log pdf of the smooth box, from the definition."""
    C = 1.0 + (b - a) / (sigma * np.sqrt(2 * np.pi))
    h = 1.0 / (C * sigma * np.sqrt(2 * np.pi))
    if x < a:
        return np.log(h) - 0.5 * ((x - a) / sigma) ** 2
    if x <= b:
        return np.log(h)
    return np.log(h) - 0.5 * ((x - b) / sigma) ** 2


def sb_t_logpdf(x, df, sigma, a, b):
    """log pdf of the smooth-box Student t, from the definition."""
    c = sp.special.gamma(0.5 * (df + 1)) / (
        sp.special.gamma(0.5 * df) * sigma * np.sqrt(df * np.pi)
    )
    C = 1.0 + (b - a) * c
    lognum = (
        sp.special.gammaln(0.5 * (df + 1))
        - sp.special.gammaln(0.5 * df)
        - 0.5 * np.log(np.pi * df)
        - np.log(C * sigma)
    )
    if x < a:
        z2 = ((x - a) / sigma) ** 2
        return lognum - 0.5 * (df + 1) * np.log1p(z2 / df)
    if x <= b:
        return lognum
    z2 = ((x - b) / sigma) ** 2
    return lognum - 0.5 * (df + 1) * np.log1p(z2 / df)


def build(D, prior, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2.0, 2.0, size=(16, D))
    y = np.sum(np.sin(X), axis=1, keepdims=True) + 1.0
    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.ConstantMean(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    gp.X, gp.y = X, y
    lb = np.full(D + 3, -10.0)
    ub = np.full(D + 3, 10.0)
    gp.set_bounds(gp.bounds_to_dict(lb, ub))
    gp.set_priors(
        {
            "covariance_log_lengthscale": prior,
            "covariance_log_outputscale": None,
            "noise_log_scale": None,
            "mean_const": None,
        }
    )
    return gp


A, B, SIG, DF = -1.0, 1.0, 0.7, 4.0

for kind in ("smoothbox", "smoothbox_student_t"):
    print(
        f"\n=== {kind} on covariance_log_lengthscale (a={A}, b={B}, "
        f"sigma={SIG}" + (f", df={DF}" if "student" in kind else "") + ") ==="
    )
    prior = (
        (kind, (A, B, SIG)) if kind == "smoothbox" else (kind, (A, B, SIG, DF))
    )
    lpdf = (
        (lambda x: sb_logpdf(x, SIG, A, B))
        if kind == "smoothbox"
        else (lambda x: sb_t_logpdf(x, DF, SIG, A, B))
    )
    for D in (1, 2, 3, 4):
        gp = build(D, prior)
        placements = {
            "all on the plateau": np.zeros(D),
            "all below a": np.full(D, -3.0),
            "all above b": np.full(D, 3.0),
            "split below a / plateau": np.array([-3.0] + [0.0] * (D - 1)),
            "split below a / above b": np.array([-3.0] + [3.0] * (D - 1)),
        }
        for label, ell in placements.items():
            hyp = np.concatenate([ell, [0.0, np.log(0.1), 0.0]])
            want = sum(lpdf(v) for v in ell)
            try:
                got = gp.log_posterior(hyp) - gp.log_likelihood(hyp)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"  D={D} {label:26s} RAISED {type(exc).__name__}: {exc}"
                )
                continue
            flag = "OK " if abs(got - want) < 1e-9 else "BAD"
            print(
                f"  D={D} {label:26s} {flag} code={got:.6f} "
                f"definition={want:.6f} ratio={got / want if want else float('nan'):.4f}"
            )
