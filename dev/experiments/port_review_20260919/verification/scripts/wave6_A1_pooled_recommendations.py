"""W6-A1, how far gpyreg's pooled bound recommendations are from gplite's
per-column ones on the training sets of PyVBMC runs.

For every stored oracle state: the spread of the training inputs per
dimension, on the whole training set (from which `GP.fit` fills the bounds
and the plausible bounds that `_gp_hyp` leaves NaN) and on the
high-posterior-density subset (from which `_gp_hyp` takes `x0`); the
recommendations of gpyreg as it is and with the statistics taken per column
(`wave6_per_column_patch.py`); and where the hyperparameters that the stored
fit arrived at lie with respect to both.

Run from anywhere with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1.
"""

import sys
from pathlib import Path

import gpyreg
import numpy as np

import pyvbmc
from pyvbmc.stats import get_hpd
from pyvbmc.testing.oracles._state import (
    build_state,
    load_snapshot,
    snapshot_names,
)
from pyvbmc.vbmc.gaussian_process_train import _get_training_data

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wave6_per_column_patch as patch  # noqa: E402

print("pyvbmc:", pyvbmc.__file__)
print("gpyreg:", gpyreg.__file__, flush=True)
np.set_printoptions(precision=3, suppress=True, linewidth=120)

fixtures = (
    Path(pyvbmc.__file__).resolve().parent / "testing" / "oracles" / "fixtures"
)


def recommendations(gp, X, y):
    return gp.covariance.get_bounds_info(X, y), gp.mean.get_bounds_info(X, y)


worst = {"ell_x0": 0.0, "omega_x0": 0.0, "sd_ratio": 0.0}
for name in snapshot_names(fixtures):
    state = build_state(load_snapshot(fixtures / name))
    gp = state["gp"]
    X, y, _, _ = _get_training_data(state["logger"])
    hpd_X, hpd_y, _, _ = get_hpd(X, y, state["options"]["hpd_frac"])
    N, D = X.shape
    cov_N = gp.covariance.hyperparameter_count(D)
    noise_N = gp.noise.hyperparameter_count()
    H = gp.get_hyperparameters(as_array=True)
    ell = H[:, :D]
    xm = H[:, cov_N + noise_N + 1 : cov_N + noise_N + 1 + D]
    omega = H[:, cov_N + noise_N + 1 + D :]

    patch.restore()
    cov_p_hpd, mean_p_hpd = recommendations(gp, hpd_X, hpd_y)
    _, mean_p = recommendations(gp, X, y)
    patch.install()
    cov_c_hpd, mean_c_hpd = recommendations(gp, hpd_X, hpd_y)
    _, mean_c = recommendations(gp, X, y)
    patch.restore()

    sd, sd_hpd = np.std(X, axis=0, ddof=1), np.std(hpd_X, axis=0, ddof=1)
    print(f"\n=== {name}: D={D}, N={N}, N_hpd={hpd_X.shape[0]}, Ns={len(H)}")
    print("mean function:", type(gp.mean).__name__)
    print("SD of X per dimension, whole set:", sd)
    print("SD of X per dimension, HPD set:  ", sd_hpd)
    print(
        f"largest over smallest SD: whole {sd.max() / sd.min():.2f},"
        f" HPD {sd_hpd.max() / sd_hpd.min():.2f}"
    )

    # x0, from the HPD subset, as `_gp_hyp` takes it.
    d_ell = cov_p_hpd["x0"][:D] - cov_c_hpd["x0"][:D]
    print("x0 log length scale, pooled:    ", cov_p_hpd["x0"][:D])
    print("x0 log length scale, per column:", cov_c_hpd["x0"][:D])
    print(
        "  pooled minus per column:      ",
        d_ell,
        f"(a factor of up to {np.exp(np.abs(d_ell).max()):.2f})",
    )
    print("  fitted (mean over samples):   ", ell.mean(axis=0))
    print(
        "  bounds agree bit for bit:",
        all(
            np.array_equal(cov_p_hpd[k][:D], cov_c_hpd[k][:D])
            for k in ("LB", "UB", "PLB", "PUB")
        ),
    )
    if mean_p_hpd["x0"].size > 1:
        s_xm, s_om = slice(1, 1 + D), slice(1 + D, 1 + 2 * D)
        d_om = mean_p_hpd["x0"][s_om] - mean_c_hpd["x0"][s_om]
        print("x0 mean location, pooled:       ", mean_p_hpd["x0"][s_xm])
        print("x0 mean location, per column:   ", mean_c_hpd["x0"][s_xm])
        print("  fitted (mean over samples):   ", xm.mean(axis=0))
        print("x0 log mean scale, pooled:      ", mean_p_hpd["x0"][s_om])
        print("x0 log mean scale, per column:  ", mean_c_hpd["x0"][s_om])
        print("  fitted (mean over samples):   ", omega.mean(axis=0))

        # The bounds that `GP.fit` fills, from the whole training set.
        for key in ("LB", "UB", "PLB", "PUB"):
            print(
                f"{key:>3} location pooled / per column:",
                mean_p[key][s_xm],
                "/",
                mean_c[key][s_xm],
            )
        for key in ("LB", "UB", "PLB", "PUB"):
            print(
                f"{key:>3} log scale pooled / per column:",
                mean_p[key][s_om],
                "/",
                mean_c[key][s_om],
            )
        out_xm = (xm < mean_c["LB"][s_xm]) | (xm > mean_c["UB"][s_xm])
        out_om = (omega < mean_c["LB"][s_om]) | (omega > mean_c["UB"][s_om])
        print(
            "stored samples outside the per-column hard bounds:",
            f"location {out_xm.any(axis=1).sum()} of {len(H)},",
            f"log scale {out_om.any(axis=1).sum()} of {len(H)}",
        )
        pl_xm = (xm < mean_c["PLB"][s_xm]) | (xm > mean_c["PUB"][s_xm])
        pl_om = (omega < mean_c["PLB"][s_om]) | (omega > mean_c["PUB"][s_om])
        pp_xm = (xm < mean_p["PLB"][s_xm]) | (xm > mean_p["PUB"][s_xm])
        pp_om = (omega < mean_p["PLB"][s_om]) | (omega > mean_p["PUB"][s_om])
        print(
            "stored samples outside the plausible box, pooled / per column:",
            f"location {pp_xm.any(axis=1).sum()} / {pl_xm.any(axis=1).sum()},",
            f"log scale {pp_om.any(axis=1).sum()} / {pl_om.any(axis=1).sum()}",
        )
        worst["omega_x0"] = max(worst["omega_x0"], np.abs(d_om).max())
    worst["ell_x0"] = max(worst["ell_x0"], np.abs(d_ell).max())
    worst["sd_ratio"] = max(worst["sd_ratio"], sd_hpd.max() / sd_hpd.min())

print("\nlargest over the states:", worst)
