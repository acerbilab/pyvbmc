"""Settles the hyperprior type masks of gpyreg's ``GP.__prior_masks``.

Two claims under verification, both about
``gpyreg/gaussian_process.py:1437-1460``:

  (a) ``df == 0 | ~np.isfinite(df)`` (lines 1441 and 1457) mis-parses in
      Python, so a non-finite ``df`` falls out of every mask, while
      ``gplite_hypprior.m:36`` gives it a Gaussian prior
      (G1 internal F1 / G1 comparison F3);
  (b) ``u_idx = ~isfinite(mu) & ~isfinite(sigma)`` (line 1453) requires
      both to be non-finite where ``gplite_hypprior.m:35`` uses ``|``, so
      the flat prior documented in ``gplite_nlZ.m:13`` (sigma = Inf)
      makes gpyreg's log posterior NaN (G1 comparison F4).

It prints the parse, the mask table as written / as intended / as MATLAB's
``gplite_hypprior.m`` computes it, and the log posterior of a small GP for
each configuration.  Run from anywhere.
"""

import gpyreg as gpr
import numpy as np

print("gpyreg:", gpr.__file__)


def matlab_masks(mu, sigma, df):
    """Transcription of gplite_hypprior.m:35-37 (read at vbmc 396d649).

    MATLAB's `==` binds tighter than `|`, so `(df == 0 | ~isfinite(df))`
    there is `(df == 0) | ~isfinite(df)`.  MATLAB has no smooth-box family
    and no fixed-prior mask.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.abs(np.asarray(sigma, dtype=float))
    df = np.asarray(df, dtype=float)
    uidx = ~np.isfinite(mu) | ~np.isfinite(sigma)
    gidx = ~uidx & ((df == 0) | ~np.isfinite(df)) & np.isfinite(sigma)
    tidx = ~uidx & (df > 0) & np.isfinite(df)
    return {"uniform": uidx, "gaussian": gidx, "student_t": tidx}


def small_gp(priors, D=1, n=12, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2.0, 2.0, size=(n, D))
    y = np.sum(np.sin(X), axis=1, keepdims=True) + 1.0
    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.ConstantMean(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    gp.X, gp.y = X, y
    gp.set_bounds(None)
    gp.set_priors(priors)
    return gp


print("\n=== 1. the parse ===")
df = np.array([0.0, 3.0, 7.0, np.inf, np.nan])
as_written = df == 0 | ~np.isfinite(df)
as_intended = (df == 0) | ~np.isfinite(df)
print("df            ", df)
print("0 | ~isfinite ", 0 | ~np.isfinite(df))
print("as written    ", as_written)
print("as intended   ", as_intended)
print(
    "MATLAB gidx-half (df==0)|~isfinite:",
    matlab_masks(np.zeros(5), np.ones(5), df)["gaussian"],
)

print("\n=== 2. the masks on a 1-D GP, one hyperparameter at a time ===")
gp = small_gp(
    {
        "covariance_log_lengthscale": None,
        "covariance_log_outputscale": None,
        "noise_log_scale": None,
        "mean_const": None,
    }
)
names = ["ell", "sf", "sn", "m0"]
cases = [
    ("gaussian via set_priors (df=0)", 0.0, 1.0, 0.0),
    ("df = inf, finite mu/sigma", 0.0, 1.0, np.inf),
    ("df = nan, finite mu/sigma", 0.0, 1.0, np.nan),
    ("sigma = inf (MATLAB's flat prior)", 0.0, np.inf, 3.0),
    ("mu = nan, sigma finite", np.nan, 1.0, 3.0),
    ("student_t df=3", 0.0, 1.0, 3.0),
]
for label, mu_i, sig_i, df_i in cases:
    # write the entry of hyperparameter 3 (mean_const) directly
    gp.hyper_priors["mu"] = np.array([np.nan, np.nan, np.nan, mu_i])
    gp.hyper_priors["sigma"] = np.array([np.nan, np.nan, np.nan, sig_i])
    gp.hyper_priors["df"] = np.array([np.nan, np.nan, np.nan, df_i])
    gp.hyper_priors["a"] = np.full(4, np.nan)
    gp.hyper_priors["b"] = np.full(4, np.nan)
    gp._prior_cache = None
    gp.no_prior = False
    gp._GP__recompute_normalization_constants()
    masks = gp._GP__prior_masks()
    # u_idx is not kept in the cache; recompute it as line 1453 writes it.
    masks = dict(masks)
    masks["u_idx"] = ~np.isfinite(gp.hyper_priors["mu"]) & ~np.isfinite(
        np.abs(gp.hyper_priors["sigma"])
    )
    active = [
        k
        for k in ("u_idx", "sb_idx", "sb_t_idx", "g_idx", "t_idx")
        if masks[k][3]
    ]
    ml = matlab_masks([mu_i], [sig_i], [df_i])
    ml_active = [k for k, v in ml.items() if v[0]]
    hyp = np.array([0.0, 0.0, np.log(0.1), 0.0])
    ll = gp.log_likelihood(hyp)
    lp = gp.log_posterior(hyp)
    print(
        f"  {label:36s} python={active or ['<none>']:} "
        f"matlab={ml_active} "
        f"log_post-log_lik={lp - ll:+.10g} norm_const={gp.normalization_constants[3]:.6g}"
    )

print("\n=== 3. the size of the dropped Gaussian term ===")
gp2 = small_gp(
    {
        "covariance_log_lengthscale": None,
        "covariance_log_outputscale": None,
        "noise_log_scale": None,
        "mean_const": ("gaussian", (0.0, 1.0)),
    }
)
hyp = np.array([0.0, 0.0, np.log(0.1), 0.0])
print(
    "  df = 0   : log_post - log_lik =",
    gp2.log_posterior(hyp) - gp2.log_likelihood(hyp),
)
print("  -0.5*log(2*pi) =", -0.5 * np.log(2 * np.pi))
gp2.hyper_priors["df"][3] = np.inf
gp2._prior_cache = None
gp2._GP__recompute_normalization_constants()
print(
    "  df = inf : log_post - log_lik =",
    gp2.log_posterior(hyp) - gp2.log_likelihood(hyp),
)
print(
    "  (MATLAB would give the Gaussian term for df = Inf: gplite_nlZ.m:12-13)"
)

print("\n=== 4. student_t with sigma = inf: the NaN of comparison F4 ===")
gp3 = small_gp(
    {
        "covariance_log_lengthscale": None,
        "covariance_log_outputscale": None,
        "noise_log_scale": None,
        "mean_const": ("student_t", (0.0, np.inf, 3.0)),
    }
)
print("  log_likelihood =", gp3.log_likelihood(hyp))
print("  log_posterior  =", gp3.log_posterior(hyp))
print("  normalization_constants =", gp3.normalization_constants)
masks = gp3._GP__prior_masks()
print(
    "  python masks on mean_const:",
    {
        "u_idx": bool(
            (
                ~np.isfinite(gp3.hyper_priors["mu"])
                & ~np.isfinite(np.abs(gp3.hyper_priors["sigma"]))
            )[3]
        ),
        "g_idx": bool(masks["g_idx"][3]),
        "t_idx": bool(masks["t_idx"][3]),
    },
)
print(
    "  MATLAB masks:",
    {k: bool(v[0]) for k, v in matlab_masks([0.0], [np.inf], [3.0]).items()},
)

print("\n=== 4b. the same with the bounds filled (finite) ===")
gp3b = small_gp(
    {
        "covariance_log_lengthscale": None,
        "covariance_log_outputscale": None,
        "noise_log_scale": None,
        "mean_const": None,
    }
)
gp3b.set_bounds(gp3b.get_recommended_bounds())
gp3b.set_priors(
    {
        "covariance_log_lengthscale": None,
        "covariance_log_outputscale": None,
        "noise_log_scale": None,
        "mean_const": ("student_t", (0.0, np.inf, 3.0)),
    }
)
print("  bounds mean_const:", gp3b.lower_bounds[3], gp3b.upper_bounds[3])
print("  normalization_constants =", gp3b.normalization_constants)
print("  log_likelihood =", gp3b.log_likelihood(hyp))
print("  log_posterior  =", gp3b.log_posterior(hyp))
print("  -> with unset (nan) bounds the constant stays 1 and the posterior")
print("     is -inf; with finite bounds the constant is 0 and it is NaN.")

print(
    "\n=== 5. f_min_fill's branch with sigma = inf (comparison F4, second half) ==="
)
from gpyreg.f_min_fill import f_min_fill

hprior = {
    "mu": np.array([0.0]),
    "sigma": np.array([np.inf]),
    "df": np.array([3.0]),
    "a": np.array([np.nan]),
    "b": np.array([np.nan]),
}
X, yv = f_min_fill(
    lambda h: float(np.sum(h**2)),
    np.zeros((1, 1)),
    np.array([-5.0]),
    np.array([5.0]),
    np.array([-1.0]),
    np.array([1.0]),
    hprior,
    8,
    "rand",
    rng=np.random.default_rng(0),
)
print("  design column (python, `and` at f_min_fill.py:119):", X.ravel())
print("  number of NaN rows:", int(np.sum(~np.isfinite(X))))
print("  MATLAB fminfill.m:73 uses `||`, so this column would be uuinv over")
print("  [LB,PLB,PUB,UB] = [-5,-1,1,5] instead.")
