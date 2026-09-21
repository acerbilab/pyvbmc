"""Settles the smaller claims of both G1 reports about gpyreg's public API
and about one PyVBMC call site.

  1. the dict form of ``log_likelihood`` / ``log_posterior``
     (``gaussian_process.py:1674-1675``, ``:1706-1707``) -- internal F6;
  2. ``set_priors`` / ``set_bounds`` accept an unknown hyperparameter name
     silently, against the documented ``ValueError`` (``:506-508``)
                                                        -- internal M-D;
  3. ``get_recommended_bounds`` with a tuple (``:390-393``), and the
     ``ValueError`` message of the ``upper_bounds`` branch (``:385``)
                                          -- internal M-E / comparison M11;
  4. ``sigma`` is not validated, and ``__prior_masks`` uses ``abs(sigma)``
     (``:1430``) where ``f_min_fill.py:169-177`` uses the raw value
                                                        -- internal M-F;
  5. ``sp.special.gamma`` in the smooth-box-t normalizers overflows
     (``:1485-1488``, ``f_min_fill.py:307``, ``:370``) -- internal M-L;
  6. ``update`` on a GP whose hyperparameters were never set
                                                        -- internal M-M;
  7. ``get_priors`` on a multi-element smooth-box block (``:466``)
                                                       -- comparison M4;
  8. the truncation renormalization ``lp -= masks["log_norm"]`` (``:1649``)
     and the ``f_idx`` fixed prior (``:1519-1523``) are Python-only
                                                       -- comparison M5;
  9. ``f_min_fill``'s docstring names ``init_method`` for the parameter
     ``design`` (``f_min_fill.py:22``, ``:49``)         -- comparison M10;
 10. ``hyp_start`` is taken from SciPy as it is (``:1326``) where
     ``gplite_train.m:304-306`` re-clamps and pins fixed coordinates
                                                        -- comparison M1;
 11. ``gaussian_process_train.py:178`` assigns ``res["log_priors"]`` to
     ``hyp_dict["logp"]``                               -- comparison M2.

Run from anywhere.
"""

import inspect

import gpyreg as gpr
import numpy as np
import scipy as sp
from gpyreg.f_min_fill import f_min_fill, smoothbox_student_t_cdf

print("gpyreg:", gpr.__file__)


def toy(D=2, n=14, seed=31, mean=None):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2.0, 2.0, size=(n, D))
    y = np.sum(1.0 + np.sin(X), axis=1, keepdims=True)
    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=mean or gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    gp.X, gp.y = X, y
    gp.set_bounds(None)
    gp.set_priors({k: None for k in gp.get_priors()})
    return gp


print("\n=== 1. the dict form of log_likelihood / log_posterior (F6) ===")
gp = toy()
hyp = np.concatenate([np.zeros(3), [np.log(0.1)], np.zeros(5)])
gp.update(hyp=hyp.reshape(1, -1))
print("  array form  log_likelihood =", gp.log_likelihood(hyp))
d = gp.hyperparameters_to_dict(hyp)[0]
print(
    "  hyperparameters_from_dict(d).shape =",
    gp.hyperparameters_from_dict(d).shape,
)
for name in ("log_likelihood", "log_posterior"):
    for grad in (False, True):
        try:
            out = getattr(gp, name)(d, compute_grad=grad)
            print(f"  {name}(dict, compute_grad={grad}) -> {out}")
        except Exception as exc:  # noqa: BLE001
            print(
                f"  {name}(dict, compute_grad={grad}) -> "
                f"{type(exc).__name__}: {exc}"
            )

print("\n=== 2. an unknown hyperparameter name (M-D) ===")
gp = toy()
priors = {k: None for k in gp.get_priors()}
priors["not_a_hyperparameter"] = ("gaussian", (0.0, 1.0))
try:
    gp.set_priors(priors)
    print("  set_priors with an unknown name -> accepted silently")
except Exception as exc:  # noqa: BLE001
    print(f"  set_priors -> {type(exc).__name__}: {exc}")
bounds = gp.get_bounds()
bounds["not_a_hyperparameter"] = (-1.0, 1.0)
try:
    gp.set_bounds(bounds)
    print("  set_bounds with an unknown name -> accepted silently")
except Exception as exc:  # noqa: BLE001
    print(f"  set_bounds -> {type(exc).__name__}: {exc}")

print("\n=== 3. get_recommended_bounds (M-E / M11) ===")
gp = toy()
n = gp.lower_bounds.size
try:
    gp.get_recommended_bounds(tuple(np.full(n, np.nan)), "recommended")
    print("  tuple lower_bounds -> accepted")
except Exception as exc:  # noqa: BLE001
    print(f"  tuple lower_bounds -> {type(exc).__name__}: {exc}")
try:
    gp.get_recommended_bounds("recommended", "nonsense")
except Exception as exc:  # noqa: BLE001
    print(f"  upper_bounds='nonsense' -> {type(exc).__name__}: {exc}")
lb = np.full(n, 1.0)
ub = np.full(n, -1.0)
out = gp.get_recommended_bounds(lb, ub)
print(
    "  inverted bounds (lb=1, ub=-1) repaired silently to:",
    {k: v for k, v in list(out.items())[:2]},
)

print("\n=== 4. sigma is not validated (M-F) ===")
gp = toy()
p = {k: None for k in gp.get_priors()}
p["mean_const"] = ("gaussian", (0.0, -2.0))
gp.set_priors(p)
print(
    "  set_priors with sigma = -2 -> accepted;"
    f" stored sigma = {gp.hyper_priors['sigma'][4]}"
)
masks = gp._GP__prior_masks()
print("  __prior_masks sigma (abs applied) =", masks["sigma"][4])
hpr = {k: gp.hyper_priors[k].copy() for k in ("mu", "sigma", "df", "a", "b")}
X, _ = f_min_fill(
    lambda h: 0.0,
    np.zeros((1, n)),
    np.full(n, -5.0),
    np.full(n, 5.0),
    np.full(n, -1.0),
    np.full(n, 1.0),
    hpr,
    6,
    "rand",
    rng=np.random.default_rng(0),
)
print("  f_min_fill column 4 (raw sigma used):", np.round(X[:, 4], 4))

print("\n=== 5. gamma overflow for a large df (M-L) ===")
for df in (3.0, 100.0, 300.0, 340.0, 343.0, 400.0):
    c = sp.special.gamma(0.5 * (df + 1)) / (
        sp.special.gamma(0.5 * df) * 1.0 * np.sqrt(df * np.pi)
    )
    print(
        f"  df={df:6.1f}  c = {c!r}   smoothbox_t_cdf(0) = "
        f"{smoothbox_student_t_cdf(0.0, df, 1.0, -1.0, 1.0)!r}"
    )

print("\n=== 6. update on a GP with no hyperparameters (M-M) ===")
gp = gpr.GP(
    D=2,
    covariance=gpr.covariance_functions.SquaredExponential(),
    mean=gpr.mean_functions.NegativeQuadratic(),
    noise=gpr.noise_functions.GaussianNoise(constant_add=True),
)
rng = np.random.default_rng(1)
Xn = rng.uniform(-2, 2, size=(8, 2))
yn = np.sum(1 + np.sin(Xn), axis=1, keepdims=True)
try:
    gp.update(X_new=Xn, y_new=yn)
    print("  update -> completed")
except Exception as exc:  # noqa: BLE001
    print(f"  update -> {type(exc).__name__}: {exc}")

print("\n=== 7. get_priors on a multi-element smooth-box block (M4) ===")
gp = toy()
p = {k: None for k in gp.get_priors()}
p["covariance_log_lengthscale"] = ("smoothbox", (-1.0, 1.0, 0.7))
gp.set_priors(p)
try:
    out = gp.get_priors()
    print("  get_priors ->", out["covariance_log_lengthscale"])
except Exception as exc:  # noqa: BLE001
    print(f"  get_priors -> {type(exc).__name__}: {exc}")
p["covariance_log_lengthscale"] = (
    "smoothbox_student_t",
    (-1.0, 1.0, 0.7, 4.0),
)
gp.set_priors(p)
try:
    out = gp.get_priors()
    print("  smoothbox_student_t -> ", out["covariance_log_lengthscale"])
except Exception as exc:  # noqa: BLE001
    print(f"  smoothbox_student_t -> {type(exc).__name__}: {exc}")

print("\n=== 8. log_norm is constant in hyp; the f_idx fixed prior (M5) ===")
gp = toy()
p = {k: None for k in gp.get_priors()}
p["mean_const"] = ("student_t", (0.0, 1.0, 3.0))
gp.set_priors(p)
gp.set_bounds(gp.get_recommended_bounds())
gp.set_priors(p)
masks = gp._GP__prior_masks()
print(
    "  log_norm =",
    masks["log_norm"],
    " normalization_constants =",
    np.round(gp.normalization_constants, 6),
)
h1 = np.concatenate([np.zeros(3), [np.log(0.1)], np.zeros(5)])
h2 = h1.copy()
h2[4] = 0.3
lp1 = gp._GP__compute_log_priors(h1, False)
lp2 = gp._GP__compute_log_priors(h2, False)
print(
    f"  lp(h1) = {lp1:.10g}, lp(h2) = {lp2:.10g};"
    f" difference of the two log_norm offsets = 0 by construction"
)
# the fixed prior
lb, ub = gp.lower_bounds.copy(), gp.upper_bounds.copy()
lb[4] = ub[4] = 0.0
gp.set_bounds(gp.bounds_to_dict(lb, ub))
gp.set_priors(p)
print(
    "  LB[4] == UB[4] == 0: lp at hyp[4]=0 ->",
    gp._GP__compute_log_priors(h1, False),
    " at hyp[4]=0.3 ->",
    gp._GP__compute_log_priors(h2, False),
)
print("  gplite_hypprior.m has no counterpart for either")

print("\n=== 9. f_min_fill's parameter name (M10) ===")
print("  signature:", inspect.signature(f_min_fill))
doc = f_min_fill.__doc__
print(
    "  docstring names 'init_method':",
    "init_method" in doc,
    "; names 'design':",
    "\n    design" in doc,
)

print("\n=== 10. hyp_start and the bound clamp (M1) ===")
print("  scipy's minimize keeps x inside the box and returns exactly c for")
res = sp.optimize.minimize(
    lambda x: (float((x[0] - 5.0) ** 2), np.array([2 * (x[0] - 5.0)])),
    x0=np.array([2.0]),
    jac=True,
    bounds=[(2.0, 2.0)],
)
print(f"  a (c, c) bound: res.x = {res.x}  (MATLAB sets c - eps(c))")
res2 = sp.optimize.minimize(
    lambda x: (float((x[0] - 5.0) ** 2), np.array([2 * (x[0] - 5.0)])),
    x0=np.array([0.0]),
    jac=True,
    bounds=[(-1.0, 1.0)],
)
print(f"  a bound-hugging optimum: res.x = {res2.x} (inside [-1, 1])")

print("\n=== 11. hyp_dict['logp'] (comparison M2) ===")
from gpyreg.slice_sample import SliceSampler

s = SliceSampler(
    lambda x: -0.5 * float(np.sum(np.asarray(x) ** 2)),
    np.array([0.0, 0.0]),
    widths=[1.0, 1.0],
    options={"display": "off", "diagnostics": False},
    rng=np.random.default_rng(0),
)
res = s.sample(6, burn=3)
print("  res['log_priors'] =", res["log_priors"])
print("  res['f_vals'].ravel() =", np.round(res["f_vals"].ravel(), 6))
print(
    "  gaussian_process_train.py:178 assigns res['log_priors'];"
    " gptrain_vbmc.m:66 stores gpoutput.logp = the thinned f_vals"
)
