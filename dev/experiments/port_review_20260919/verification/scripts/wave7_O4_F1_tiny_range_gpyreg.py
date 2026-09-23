"""O4 F1, the gpyreg-alone trigger through the recommended bounds: targets
whose range is below 1e-6 collapse the noise's hard pair (the case of row
W6-38); with a noise without a prior and a prior elsewhere, the fit
objective's gradient is NaN there. Compared with the same set without any
prior, and with a range just above 1e-6."""

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from wave7_O4_common import attach, banner, make_gp

banner()
warnings.simplefilter("ignore")

rng = np.random.default_rng(4)
N, D = 15, 1
X = rng.uniform(-1, 1, (N, D))
base = np.sin(3 * X[:, 0])
for rng_y in (5e-7, 5e-6):
    y = (
        2.0 + rng_y * (base - base.min()) / (base.max() - base.min())
    ).reshape(-1, 1)
    for prior in ("none", "lengthscale"):
        gp = make_gp(D, "const", (1, 0, 0))
        attach(gp, X, y)
        pri = None
        if prior == "lengthscale":
            b = gp.get_bounds()
            pri = {k: None for k in b}
            pri["covariance_log_lengthscale"] = ("gaussian", (0.0, 1.0))
        gp.set_priors(pri)
        hyp, res, _ = gp.fit(
            options={"n_samples": 0, "init_N": 128, "opts_N": 2},
            rng=np.random.default_rng(0),
        )
        lb, ub = gp.lower_bounds, gp.upper_bounds
        eq = np.where(lb == ub)[0]
        _, g = gp._GP__gp_obj_fun(hyp[0], True, False)
        print(
            f"range={rng_y:.0e} prior={prior:11s}: equal pairs at {eq.tolist()} "
            f"({lb[eq]}), NaN grad at {np.where(np.isnan(g))[0].tolist()}, "
            f"optimizer nit={res.nit} msg={res.message!r}, "
            f"log_lik={gp.log_likelihood(hyp[0]):.4f}",
            flush=True,
        )
