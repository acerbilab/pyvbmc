"""P3-7: the clamped `s_pred` and the unclamped `d` of `_log_iqr_reduction`.

Settles whether `tau2 > f_s2_a` can occur on a realistic state, where
`s_pred` is clamped to zero while `d = tau2 / (s_a + s_pred)` is not, so
that `d` no longer equals `s_a - s_pred`. Recomputes the per-sample `tau2`
of `AcqFcnVIQR._compute_viqr` on the stored noisy oracle state, with the
`iqr_reduction` loss, and compares it with the importance points' own
posterior variance.
"""
import sys

import numpy as np
from scipy.spatial.distance import cdist

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from wave4_P3_common import banner, load  # noqa: E402

banner()

from pyvbmc.acquisition_functions import AcqFcnVIQR  # noqa: E402
from pyvbmc.testing.oracles._oracles import (  # noqa: E402
    prepare_gp_for_acq,
    prepare_importance_sampling,
)

SEED = 20260904
st = load("rosenbrock_D2_noise1_viqr", seed=SEED)
gp, vp, logger, os_ = st["gp"], st["vp"], st["logger"], st["optim_state"]
Xs = np.array(st["cand"]["Xs"], dtype=float)
prepare_gp_for_acq(gp, logger, os_)
acq = AcqFcnVIQR(loss="iqr_reduction")
prepare_importance_sampling(st, acq, SEED)
out = acq(Xs.copy(), gp, vp, logger, os_)
print(
    "iqr_reduction acquisition: finite",
    int(np.sum(np.isfinite(out))),
    "of",
    out.size,
    " min",
    np.nanmin(out),
    " max",
    np.nanmax(out),
)

# Recompute tau2 exactly as `_compute_viqr` does, per hyperparameter sample.
Xs_int = acq._real2int(
    Xs.copy(), vp.parameter_transformer, os_.get("integer_vars")
)
f_mu, f_s2 = gp.predict(x_star=Xs_int, separate_samples=True)
sn2 = acq._estimate_observation_noise(Xs_int, gp, os_)
y_s2 = f_s2 + sn2.reshape(-1, 1)
ais = os_["active_importance_sampling"]
Xa = ais["X"]
D = gp.D
cov_N = gp.covariance.hyperparameter_count(D)
worst = 0.0
n_exceed = 0
n_total = 0
for s in range(f_mu.shape[1]):
    hyp = gp.posteriors[s].hyp[0:cov_N]
    ell = np.exp(hyp[0:D])
    sf2 = np.exp(2 * hyp[D])
    Xs_ell = Xs_int / ell
    K_Xs_X = sf2 * np.exp(-cdist(Xs_ell, gp.X / ell, "sqeuclidean") / 2)
    K_Xs_Xa = sf2 * np.exp(-cdist(Xs_ell, Xa / ell, "sqeuclidean") / 2)
    C_tmp = ais["C_tmp"][s, :, :]
    if gp.posteriors[s].L_chol:
        C = K_Xs_Xa - K_Xs_X @ C_tmp
    else:
        C = K_Xs_Xa + K_Xs_X @ C_tmp
    tau2 = C**2 / y_s2[:, s].reshape(-1, 1)
    f_s2_a = ais["f_s2"][:, s]
    ratio = tau2 / f_s2_a  # (Nx, Na)
    n_total += ratio.size
    n_exceed += int(np.sum(tau2 > f_s2_a))
    worst = max(worst, float(np.max(ratio)))
    if s == 0:
        print(
            "sample 0: f_s2_a min %.3e max %.3e ; sn2 %.6f ; "
            "f_s2(candidate) min %.3e max %.3e"
            % (
                f_s2_a.min(),
                f_s2_a.max(),
                float(np.unique(sn2)[0]),
                f_s2[:, 0].min(),
                f_s2[:, 0].max(),
            )
        )
print(
    f"entries with tau2 > f_s2_a: {n_exceed} of {n_total}; "
    f"max tau2/f_s2_a = {worst:.12f}"
)
print(
    "upper bound from the noise alone, max_s s2*/(s2*+sn2) = "
    f"{float(np.max(f_s2 / (f_s2 + sn2.reshape(-1,1)))):.12f}"
)

# What the two expressions do when the clip is active, by construction.
u = acq.u
for eps in (1e-16, 1e-12, 1e-6, 1e-2):
    s_a2 = 1.0
    tau2 = s_a2 * (1.0 + eps)
    s_a = np.sqrt(s_a2)
    s_pred = np.sqrt(max(s_a2 - tau2, 0.0))
    d = tau2 / (s_a + s_pred)
    print(
        f"  tau2/f_s2_a = 1+{eps:g}: s_a-s_pred = {float(s_a - s_pred):.17g}, "
        f"d = {float(d):.17g}, relative excess "
        f"{float((d - (s_a - s_pred)) / (s_a - s_pred)):.3e}"
    )
