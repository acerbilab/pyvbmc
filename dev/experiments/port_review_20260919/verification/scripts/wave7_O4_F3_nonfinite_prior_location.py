"""O4 F3: `set_priors` validates only `sigma`; a location that is not
finite (infinite or NaN `mu`, an infinite or NaN end of a smooth box) is
accepted. log_posterior with and without filled bounds, beside a
transcription of `gplite_hypprior.m` (non-finite mu -> uniform), and what a
fit does with one such prior."""

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from wave7_O4_common import attach, banner, make_gp

banner()
warnings.simplefilter("ignore")

rng = np.random.default_rng(3)
N, D = 12, 1
X = rng.uniform(-1, 1, (N, D))
y = (np.sin(2 * X[:, 0])).reshape(-1, 1)
hyp = np.array([0.0, np.log(np.std(y)), np.log(0.1), 0.2])

cases = [
    ("gaussian", (np.inf, 1.0)),
    ("gaussian", (-np.inf, 1.0)),
    ("gaussian", (np.nan, 1.0)),
    ("student_t", (np.inf, 1.0, 3)),
    ("smoothbox", (0.0, np.inf, 1.0)),
    ("smoothbox", (-np.inf, 0.0, 1.0)),
    ("smoothbox", (np.nan, 1.0, 1.0)),
    ("smoothbox_student_t", (0.0, np.inf, 1.0, 3)),
    ("gaussian", (0.0, np.inf)),  # control: refused
]


def matlab_hypprior_lp(kind, p, h):
    """gplite_hypprior.m for one coordinate: uniform when mu or sigma is
    not finite (line 35); MATLAB has no smooth-box family."""
    if kind == "gaussian":
        mu, s = p
    elif kind == "student_t":
        mu, s, _ = p
    else:
        return "no counterpart"
    if not (np.isfinite(mu) and np.isfinite(s)):
        return 0.0
    return "finite prior"


for kind, p in cases:
    gp = make_gp(D, "const", (1, 0, 0))
    attach(gp, X, y)
    pri = {k: None for k in gp.get_bounds()}
    pri["mean_const"] = (kind, p)
    try:
        gp.set_priors(pri)
    except ValueError as e:
        print(
            f"{kind:20s} {str(p):22s}: refused by set_priors ({str(e)[:60]}...)"
        )
        continue
    lp_unset = gp.log_posterior(hyp)
    gp.set_bounds(gp.get_recommended_bounds())
    lp_set = gp.log_posterior(hyp)
    nc = gp.normalization_constants[-1]
    print(
        f"{kind:20s} {str(p):22s}: accepted; log_posterior bounds unset={lp_unset}, "
        f"bounds filled={lp_set} (mass={nc}); log_likelihood={gp.log_likelihood(hyp):.4f}; "
        f"gplite_hypprior lp of this coordinate: {matlab_hypprior_lp(kind, p, hyp[-1])}"
    )

print("\nfit with ('smoothbox', (0, inf, 1)) on mean_const, n_samples=0:")
gp = make_gp(D, "const", (1, 0, 0))
attach(gp, X, y)
pri = {k: None for k in gp.get_bounds()}
pri["mean_const"] = ("smoothbox", (0.0, np.inf, 1.0))
gp.set_priors(pri)
try:
    h, res, _ = gp.fit(
        options={"n_samples": 0, "init_N": 64, "opts_N": 2},
        rng=np.random.default_rng(0),
    )
    print(
        f"   returned: nit={res.nit} fun={res.fun} msg={res.message!r} hyp={h[0]}, "
        f"log_lik there={gp.log_likelihood(h[0]):.4f}"
    )
except Exception as e:  # noqa: BLE001
    print(f"   fit raised {type(e).__name__}: {e}")
gp = make_gp(D, "const", (1, 0, 0))
attach(gp, X, y)
h, res, _ = gp.fit(
    options={"n_samples": 0, "init_N": 64, "opts_N": 2},
    rng=np.random.default_rng(0),
)
print(f"   control without priors: log_lik={gp.log_likelihood(h[0]):.4f}")
