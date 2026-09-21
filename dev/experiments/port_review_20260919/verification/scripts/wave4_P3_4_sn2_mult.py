"""P3-4: `sn2_new` and the Cholesky-retry multiplier `sn2_mult`.

Settles: (a) that a posterior with `sn2_mult > 1` is constructible, so the
path is not hypothetical; (b) that `gp.noise.compute` (what `active_sample`
stores as `sn2_new`) carries no `sn2_mult`, while `predict(add_noise=True)`
and the rank-one `GP.update` do. MATLAB's `private/activesample_vbmc.m:172`
calls `gplite_noisefun` the same way, and `gplite_pred.m:121` /
`gplite_post.m:208` apply the multiplier the same way, so the omission is
the same on both sides.
"""
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from wave4_P3_common import banner  # noqa: E402

banner()

import gpyreg as gpr  # noqa: E402

D = 2
rng = np.random.default_rng(0)
# Near-duplicate rows plus a tiny constant noise force the low-noise
# (non-Cholesky) branch to retry.
base = rng.normal(size=(6, D))
X = np.vstack([base, base + 1e-9, base - 1e-9])
y = np.sum(X**2, axis=1, keepdims=True) * -0.5

gp = gpr.GP(
    D=D,
    covariance=gpr.covariance_functions.SquaredExponential(),
    mean=gpr.mean_functions.NegativeQuadratic(),
    noise=gpr.noise_functions.GaussianNoise(constant_add=True),
)
gp.update(X_new=X, y_new=y, compute_posterior=False)
cov_N = gp.covariance.hyperparameter_count(D)
noise_N = gp.noise.hyperparameter_count()
mean_N = gp.mean.hyperparameter_count(D)

for log_sn in (-9.0, -12.0, -15.0, -18.0):
    hyp = np.zeros((1, cov_N + noise_N + mean_N))
    hyp[0, :D] = 1.5  # log length scales
    hyp[0, D] = 0.0  # log output scale
    hyp[0, cov_N] = log_sn  # log noise SD
    hyp[0, cov_N + noise_N] = 0.0  # mean const
    hyp[0, cov_N + noise_N + 1 : cov_N + noise_N + 1 + D] = 0.0  # mean loc
    hyp[0, cov_N + noise_N + 1 + D :] = 1.0  # mean log scales
    try:
        gp.set_hyperparameters(hyp, compute_posterior=True)
    except Exception as exc:  # pragma: no cover - diagnostic
        print(f"log_sn={log_sn}: set_hyperparameters raised {exc!r}")
        continue
    post = gp.posteriors[0]
    x_star = np.array([[0.3, -0.2], [1.0, 1.0]])
    _, s2_latent = gp.predict(x_star)
    _, s2_noisy = gp.predict(x_star, add_noise=True)
    sn2_compute = np.asarray(
        gp.noise.compute(post.hyp[cov_N : cov_N + noise_N], x_star, None, None)
    ).ravel()
    delta = (s2_noisy - s2_latent).ravel()
    print(
        f"log_sn={log_sn:6.1f} L_chol={post.L_chol} "
        f"sn2_mult={post.sn2_mult} "
        f"noise.compute={sn2_compute} "
        f"predict(add_noise)-predict()={delta} "
        f"ratio={delta / np.broadcast_to(sn2_compute, delta.shape)}"
    )
