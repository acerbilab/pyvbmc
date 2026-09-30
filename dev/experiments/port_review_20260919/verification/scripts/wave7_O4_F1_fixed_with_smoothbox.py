"""O4 F1, the report's claim that a prior of any family on the fixed
coordinate overwrites the NaN: the smooth-box branches write the gradient
only for coordinates below or above the box, so a fixed coordinate whose
value lies inside its box keeps the NaN. Checked for each family, with the
fixed value inside and outside a box."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from wave7_O4_common import attach, banner, make_gp

banner()
rng = np.random.default_rng(11)
N, D = 20, 2
X = rng.uniform(-1, 1, (N, D))
y = (-0.5 * np.sum(X**2, 1) + 0.02 * rng.standard_normal(N)).reshape(-1, 1)
FIX = np.log(1e-2)
NOISE = D + 1
priors = {
    "gaussian": ("gaussian", (FIX, 1.0)),
    "student_t": ("student_t", (FIX, 1.0, 3)),
    "smoothbox, value inside": ("smoothbox", (FIX - 1, FIX + 1, 0.5)),
    "smoothbox, value below": ("smoothbox", (FIX + 1, FIX + 2, 0.5)),
    "smoothbox_student_t, value inside": (
        "smoothbox_student_t",
        (FIX - 1, FIX + 1, 0.5, 3),
    ),
    "smoothbox_student_t, value above": (
        "smoothbox_student_t",
        (FIX - 2, FIX - 1, 0.5, 3),
    ),
}
for label, pr in priors.items():
    gp = make_gp(D, "negquad", (1, 0, 0))
    attach(gp, X, y)
    b = gp.get_recommended_bounds()
    b["noise_log_scale"] = (np.array([FIX]), np.array([FIX]))
    gp.set_bounds(b)
    pri = {k: None for k in b}
    pri["noise_log_scale"] = pr
    gp.set_priors(pri)
    hyp = np.concatenate(
        [
            gp.covariance.get_bounds_info(gp.X, gp.y)["x0"],
            gp.noise.get_bounds_info(gp.X, gp.y)["x0"],
            gp.mean.get_bounds_info(gp.X, gp.y)["x0"],
        ]
    )
    hyp[NOISE] = FIX
    _, g = gp._GP__gp_obj_fun(hyp, True, False)
    hyp_fit, res, _ = gp.fit(
        options={"n_samples": 0, "init_N": 128, "opts_N": 2},
        rng=np.random.default_rng(0),
    )
    print(
        f"prior on the fixed coordinate = {label:36s}: fit-objective gradient entry = {g[NOISE]}; "
        f"fit: nit={res.nit}, log_lik={gp.log_likelihood(hyp_fit[0]):.4f}"
    )
