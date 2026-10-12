"""P3-3: the candidate noise averaged over GP hyperparameter samples.

Settles: (a) that `active_sample.py:351` (`sn2new.mean(1)`) is what
`private/activesample_vbmc.m:174` (`gp.sn2new = mean(sn2new,2)`) does, so
the averaging is shared with MATLAB; (b) how wide the per-sample spread of
the candidate noise is at uncertainty level 1, where the noise multiplier
is an inferred hyperparameter. One small `train_gp` fit, 26 points, D = 2.
"""
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from wave4_P3_common import banner  # noqa: E402

banner()

from pyvbmc.function_logger import FunctionLogger  # noqa: E402
from pyvbmc.parameter_transformer import ParameterTransformer  # noqa: E402
from pyvbmc.testing.oracles._state import build_options  # noqa: E402
from pyvbmc.vbmc.gaussian_process_train import train_gp  # noqa: E402
from pyvbmc.vbmc.iteration_history import IterationHistory  # noqa: E402

rng = np.random.default_rng(7)
D, sigma = 2, 1.5
pt = ParameterTransformer(D)
fl = FunctionLogger(
    lambda x: float(
        -0.5 * np.sum(np.atleast_1d(x) ** 2) + sigma * rng.standard_normal()
    ),
    D,
    True,
    1,
    parameter_transformer=pt,
)
X = rng.normal(scale=1.5, size=(26, D))
for i in range(X.shape[0]):
    fl(X[i])
fl(X[0])
fl(X[0])
fl(X[5])

options = build_options({"uncertainty_handling": True}, D)
optim_state = dict(
    gp_mean_fun="negquad",
    gp_cov_fun=1,
    gp_noise_fun=[1, 2, 0],
    uncertainty_handling_level=1,
    N=fl.Xn + 1,
    n_eff=float(np.sum(fl.n_evals[fl.X_flag])),
    iter=0,
    warmup=True,
    stop_sampling=0,
    vp_K=2,
    recompute_var_post=True,
    max_fun_evals=options["max_fun_evals"],
)
ih = IterationHistory(["r_index", "gp", "gp_hyp_full", "sKL"])
plb = np.full((1, D), -3.0)
pub = np.full((1, D), 3.0)
gp, gp_s_N, sn2_hpd, _ = train_gp(
    {}, optim_state, fl, ih, options, plb, pub, rng=np.random.default_rng(3)
)
Ns = len(gp.posteriors)
cov_N = gp.covariance.hyperparameter_count(D)
noise_N = gp.noise.hyperparameter_count()
print("Ns =", Ns, "noise_N =", noise_N)
h0 = np.array([gp.posteriors[s].hyp[cov_N] for s in range(Ns)])
h1 = np.array([gp.posteriors[s].hyp[cov_N + 1] for s in range(Ns)])
print("exp(2*h0) per sample:", np.round(np.exp(2 * h0), 6))
print(
    "exp(h1)   per sample:",
    np.round(np.exp(h1), 4),
    " (target variance sigma^2 =",
    sigma**2,
    ")",
)

s2 = (fl.S[fl.X_flag] ** 2) * fl.n_evals[fl.X_flag]
print("s2 = S^2 * n_evals, unique values:", np.unique(np.round(s2, 12)))
sn2new = np.zeros((gp.X.shape[0], Ns))
for s in range(Ns):
    hn = gp.posteriors[s].hyp[cov_N : cov_N + noise_N]
    sn2new[:, s] = np.asarray(gp.noise.compute(hn, gp.X, gp.y, s2)).reshape(-1)
print("per-sample sn2_new at training row 0:", np.round(sn2new[0], 4))
print(
    "spread across samples: min %.4f max %.4f ratio %.2f mean %.4f"
    % (
        sn2new[0].min(),
        sn2new[0].max(),
        sn2new[0].max() / sn2new[0].min(),
        sn2new[0].mean(),
    )
)
print(
    "sn2_new equal at every training point (homoskedastic at level 1):",
    bool(np.allclose(sn2new.mean(1), sn2new.mean(1)[0], rtol=0, atol=0)),
)
# What the GP itself assigns to its own (pooled) training rows.
sn2_rows = np.stack(
    [
        np.asarray(
            gp.noise.compute(
                gp.posteriors[s].hyp[cov_N : cov_N + noise_N],
                gp.X,
                gp.y,
                gp.s2,
            )
        ).ravel()
        for s in range(Ns)
    ],
    1,
).mean(1)
n_ev = fl.n_evals[fl.X_flag].ravel()
for i in np.argsort(-n_ev)[:3]:
    print(
        f"  row {i}: n_evals={int(n_ev[i])} pooled-row noise="
        f"{sn2_rows[i]:.4f} sn2_new={sn2new[i].mean():.4f} "
        f"ratio={sn2new[i].mean()/sn2_rows[i]:.3f}"
    )
