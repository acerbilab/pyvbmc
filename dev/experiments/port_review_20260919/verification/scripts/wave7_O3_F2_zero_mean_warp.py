"""O3 F2: warp_gp_and_vp on a zero-mean, a constant-mean and a
negative-quadratic GP, on one constructed state (D=3, two bounded
coordinates and one unbounded, probit, the default transform).

Settles: (a) the zero-mean GP raises ValueError in warp_gp_and_vp;
(b) for the constant mean, the warped m0 equals m0 + C, where C is the
constant change of log|J| of the affine rotoscaling warp, which is also the
shift of every stored log joint in the logger; (c) what the MATLAB branch
`case 0` of misc/warp_gpandvp_vbmc.m would index for each MATLAB mean code,
from a transcription of its index arithmetic and of gplite_meanfun.m's
hyperparameter counts (read from the source, not run).
"""

import gpyreg as gpr
import numpy as np

import pyvbmc
from pyvbmc.function_logger import FunctionLogger
from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.whitening import warp_gp_and_vp, warp_input

print("pyvbmc:", pyvbmc.__file__)
print("gpyreg:", gpr.__file__)

D = 3
lb = np.array([[-2.0, 0.0, -np.inf]])
ub = np.array([[4.0, 5.0, np.inf]])
plb = np.array([[-1.0, 1.0, -2.0]])
pub = np.array([[3.0, 4.0, 3.0]])


class _Owner:
    """The two attributes of VBMC that warp_gp_and_vp reads."""

    def __init__(self, optim_state):
        self.optim_state = optim_state
        self.D = D


def target(x):
    x = np.atleast_2d(x)
    m = np.array([1.0, 2.0, 0.5])
    P = np.linalg.inv(
        np.array([[1.0, 0.5, 0.2], [0.5, 1.2, -0.3], [0.2, -0.3, 1.5]])
    )
    d = x - m
    return float(-0.5 * (d @ P @ d.T).item())


def state(mean_name, seed=3):
    rng = np.random.default_rng(seed)
    pt = ParameterTransformer(D, lb, ub, plb, pub, transform_type="probit")
    fl = FunctionLogger(target, D, False, 0, parameter_transformer=pt)
    Lc = np.array([[0.4, 0, 0], [0.3, 0.25, 0], [-0.1, 0.15, 0.3]])
    for u in rng.standard_normal((25, D)) @ Lc.T:
        fl(u)
    K = 5
    vp = VariationalPosterior(
        D,
        K,
        x0=np.zeros((1, D)),
        parameter_transformer=pt,
        rng=np.random.default_rng(seed + 10),
    )
    vp.mu = (rng.standard_normal((K, D)) @ Lc.T).T
    vp.sigma = rng.uniform(0.1, 0.2, (1, K))
    lam = rng.uniform(0.6, 1.4, (D, 1))
    vp.lambd = lam / np.sqrt(np.sum(lam**2) / D)
    vp.w = np.full((1, K), 1.0 / K)
    vp.eta = np.log(vp.w)
    mean = {
        "zero": gpr.mean_functions.ZeroMean(),
        "const": gpr.mean_functions.ConstantMean(),
        "negquad": gpr.mean_functions.NegativeQuadratic(),
    }[mean_name]
    gp = gpr.GP(
        D,
        gpr.covariance_functions.SquaredExponential(),
        mean,
        gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    hyp = [np.log([0.6, 0.5, 0.8]), [np.log(2.0)], [np.log(1e-3)]]
    if mean_name == "const":
        hyp.append([-1.5])
    if mean_name == "negquad":
        hyp += [[0.7], [0.1, 0.0, -0.1], np.log([0.5, 0.6, 0.9])]
    hyp = np.concatenate(hyp)[None, :]
    gp.update(
        X_new=fl.X[fl.X_flag],
        y_new=fl.y[fl.X_flag],
        hyp=hyp,
        compute_posterior=True,
    )
    plb_t, pub_t = pt(plb), pt(pub)
    pr = pub_t - plb_t
    optim_state = {
        "N": 25,
        "plb_orig": plb,
        "pub_orig": pub,
        "lb_search": plb_t - 2 * pr,
        "ub_search": pub_t + 2 * pr,
        "search_cache": None,
        "iter": 12,
        "warping_count": 0,
    }
    return pt, fl, vp, gp, optim_state


options = {
    "warp_nonlinear": False,
    "warp_rotoscaling": True,
    "warp_roto_corr_thresh": 0.05,
    "warp_cov_reg": 0,
}

for name in ("zero", "const", "negquad"):
    pt, fl, vp, gp, os_ = state(name)
    pt_new, os_new, fl_new, _ = warp_input(vp, os_, fl, options)
    n = fl.Xn + 1
    dy = fl_new.y[:n, 0] - fl.y[:n, 0]
    print(
        f"\n[{name}] logger y shift: mean {dy.mean():.12f}, "
        f"spread {dy.max() - dy.min():.2e}"
    )
    try:
        vp_new, hyp_w = warp_gp_and_vp(pt_new, gp, vp, _Owner(os_new))
    except Exception as exc:  # the finding
        print(f"[{name}] warp_gp_and_vp raised {type(exc).__name__}: {exc}")
        continue
    hyp = gp.posteriors[0].hyp
    Ncov, Nnoise = D + 1, 1
    if name in ("const", "negquad"):
        m0, m0w = hyp[Ncov + Nnoise], hyp_w[0, Ncov + Nnoise]
        print(
            f"[{name}] m0' - m0 = {m0w - m0:.12f}; minus the logger shift: "
            f"{(m0w - m0) - dy.mean():.2e}"
        )

# MATLAB index arithmetic of misc/warp_gpandvp_vbmc.m `case 0` (reads
# hyp(Ncov+Nnoise+1)), for the mean codes of gplite_meanfun.m.
print("\nMATLAB: gp.meanfun is numeric (gplite_post.m:120-121,")
print("gplite_meanfun.m:57-62, 347): 0 zero, 1 const, 4 negquad.")
Ncov, Nnoise = D + 1, 1  # SE-ARD; noisefun [1 0]
for code, name, Nmean in (
    (0, "zero", 0),
    (1, "const", 1),
    (4, "negquad", 2 * D + 1),
):
    Nhyp = Ncov + Nnoise + Nmean
    branch = {0: "case 0 ('Warp constant mean')", 4: "case 4"}.get(
        code, "otherwise -> error"
    )
    read = Ncov + Nnoise + 1
    verdict = (
        "index out of range (MATLAB error), unless Noutwarp > 0"
        if code == 0 and read > Nhyp
        else ("in range" if code != 1 else "not reached")
    )
    print(
        f"  '{name}' (code {code}): numel(hyp) = {Nhyp}, branch {branch}; "
        f"case-0 read of hyp({read}): {verdict}"
    )
