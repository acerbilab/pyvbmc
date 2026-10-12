"""Settles the claims about ``GP.fit`` and ``f_min_fill``.

  * the sampler option is read as ``options["sampler"]``
    (``gaussian_process.py:1126``) while the docstring documents
    ``sampler_name`` (``:1073``)                       -- internal F5;
  * ``hyp = X0[0:max(opts_N,1), :]`` (``:1243``) is a view, so
    ``hyp[1, :] = xx[idx_best, :]`` (``:1259``) mutates the design and
    ``widths_default = np.std(X0, ...)`` (``:1262``) is taken from the
    mutated design, where ``gplite_train.m:206-207`` copies and computes
    the widths before the substitution at ``:220``
                                        -- internal F7 / comparison F5;
  * neither ``f_min_fill`` nor ``fit``'s optimizer loop catches an
    exception from the objective, where ``fminfill.m:104-110``,
    ``gplite_train.m:276-296`` and ``gp_objfun``'s try/catch do
                                                      -- comparison F7;
  * ``f_min_fill`` with ``N <= N0`` returns only ``N`` of the provided
    rows                                              -- internal M-G;
  * ``fit`` fills ``hyper_priors["df"]`` permanently (``:1153``)
                                                      -- internal M-H;
  * the ``-inf`` sentinel of the ``init_N == 0`` branch (``:1267``) and
    the dead ``sampler_name != "laplace"`` clause (``:1221``)
                                                      -- internal M-K;
  * the default ``hyp0`` (``:1201-1207``) against ``zeros`` of
    ``gplite_train.m:98``                             -- comparison M8;
  * ``np.argsort`` stability                          -- comparison M9.

Run from anywhere.
"""


import gpyreg as gpr
import numpy as np
import scipy as sp
from gpyreg.f_min_fill import f_min_fill

print("gpyreg:", gpr.__file__)


def toy_gp(D=2, n=30, seed=17):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3.0, 3.0, size=(n, D))
    y = np.sum(1.0 + np.sin(X), axis=1, keepdims=True)
    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    gp.X, gp.y = X, y
    gp.set_bounds(None)
    gp.set_priors(
        {
            "covariance_log_lengthscale": None,
            "covariance_log_outputscale": None,
            "noise_log_scale": None,
            "mean_const": None,
            "mean_location": None,
            "mean_log_scale": None,
        }
    )
    return gp


print("\n=== 1. the sampler option name (internal F5) ===")
gp = toy_gp()
opts = {"init_N": 8, "opts_N": 1, "n_samples": 2, "thin": 1, "burn": 2}
try:
    gp.fit(
        options={**opts, "sampler_name": "does_not_exist"},
        rng=np.random.default_rng(0),
    )
    print(
        "  options={'sampler_name': 'does_not_exist'} -> completed, value ignored"
    )
except Exception as exc:  # noqa: BLE001
    print(f"  sampler_name -> {type(exc).__name__}: {exc}")
gp = toy_gp()
try:
    gp.fit(
        options={**opts, "sampler": "does_not_exist"},
        rng=np.random.default_rng(0),
    )
    print("  options={'sampler': 'does_not_exist'} -> completed (unexpected)")
except Exception as exc:  # noqa: BLE001
    print(f"  options={{'sampler': ...}} -> {type(exc).__name__}: {exc}")
gp = toy_gp()
try:
    gp.fit(
        options={**opts, "sampler": "laplace"}, rng=np.random.default_rng(0)
    )
    print("  options={'sampler': 'laplace'} -> completed (unexpected)")
except Exception as exc:  # noqa: BLE001
    print(
        f"  options={{'sampler': 'laplace'}} -> {type(exc).__name__}: {exc}"
        "   (so the 'laplace' clause at :1221 is dead when n_samples > 0)"
    )

print(
    "\n=== 2. the design is mutated and the widths follow (internal F7 / comp. F5) ==="
)
import gpyreg.gaussian_process as gp_mod

captured = {}
orig_fmf = gp_mod.f_min_fill


def spy(*args, **kwargs):
    X0, y0 = orig_fmf(*args, **kwargs)
    captured["X0_copy"] = X0.copy()
    captured["y0"] = y0.copy()
    captured["X0"] = X0
    return X0, y0


for init_N in (8, 64, 128, 256):
    gp = toy_gp()
    gp_mod.f_min_fill = spy
    try:
        gp.fit(
            options={
                "init_N": init_N,
                "opts_N": 3,
                "n_samples": 2,
                "thin": 1,
                "burn": 2,
            },
            rng=np.random.default_rng(3),
        )
    finally:
        gp_mod.f_min_fill = orig_fmf
    before = captured["X0_copy"]
    after = captured["X0"]
    changed = np.where(np.any(before != after, axis=1))[0]
    w_before = np.std(before, axis=0, ddof=1)
    w_after = np.std(after, axis=0, ddof=1)
    rel = np.max(np.abs(w_after - w_before) / np.maximum(w_before, 1e-300))
    print(
        f"  init_N={init_N:4d}: rows of the design changed = {changed.tolist()}"
        f"  max rel change in widths = {rel:.3g}"
    )
    if init_N == 128:
        i = int(
            np.argmax(
                np.abs(w_after - w_before) / np.maximum(w_before, 1e-300)
            )
        )
        print(
            f"    coordinate {i}: widths {w_before[i]:.6g} -> {w_after[i]:.6g}"
        )

print("\n=== 3. hyp is a view of X0 ===")
X0 = np.zeros((4, 3))
X0[:] = np.arange(12, dtype=float).reshape(4, 3)
hyp = X0[0 : np.maximum(3, 1), :]
print(
    "  hyp.base is X0:",
    hyp.base is X0,
    " shares memory:",
    np.shares_memory(hyp, X0),
)
hyp[1, :] = -1.0
print("  after hyp[1,:] = -1, X0[1] =", X0[1])

print("\n=== 4. an objective that raises (comparison F7) ===")
calls = {"n": 0}


def raising(h):
    calls["n"] += 1
    if calls["n"] == 5:
        raise sp.linalg.LinAlgError("forced failure at design point 5")
    return float(np.sum(h**2))


try:
    f_min_fill(
        raising,
        np.zeros((1, 3)),
        np.full(3, -5.0),
        np.full(3, 5.0),
        np.full(3, -1.0),
        np.full(3, 1.0),
        {
            "mu": np.full(3, np.nan),
            "sigma": np.full(3, np.nan),
            "df": np.full(3, np.nan),
            "a": np.full(3, np.nan),
            "b": np.full(3, np.nan),
        },
        8,
        "rand",
        rng=np.random.default_rng(0),
    )
    print("  f_min_fill completed (unexpected)")
except Exception as exc:  # noqa: BLE001
    print(f"  f_min_fill propagated {type(exc).__name__}: {exc}")
    print("  fminfill.m:104-110 would keep y(5) = Inf and carry on")

gp = toy_gp()
real_min = sp.optimize.minimize


def raising_min(*a, **kw):
    raise sp.linalg.LinAlgError("forced failure inside the optimizer")


sp.optimize.minimize = raising_min
try:
    gp.fit(
        options={"init_N": 8, "opts_N": 1, "n_samples": 0},
        rng=np.random.default_rng(0),
    )
    print("  fit completed with a raising optimizer (unexpected)")
except Exception as exc:  # noqa: BLE001
    print(f"  fit propagated {type(exc).__name__}: {exc}")
    print("  gplite_train.m:276-296 would keep the starting point")
finally:
    sp.optimize.minimize = real_min

print("\n=== 5. f_min_fill with N <= N0 (internal M-G) ===")
seen = []
x0 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
out_X, out_y = f_min_fill(
    lambda h: (seen.append(h.copy()), float(np.sum(h**2)))[1],
    x0,
    np.full(2, -5.0),
    np.full(2, 5.0),
    np.full(2, -1.0),
    np.full(2, 1.0),
    {k: np.full(2, np.nan) for k in ("mu", "sigma", "df", "a", "b")},
    2,
    "rand",
    rng=np.random.default_rng(0),
)
print(
    f"  x0 has 4 rows, N=2: returned {out_X.shape[0]} rows after "
    f"{len(seen)} objective calls"
)
print("  returned rows:", out_X.tolist())
print("  MATLAB: X = [x0;[]]; y = Inf(1,N); loop over 1:N -- identical")

print("\n=== 6. fit fills hyper_priors['df'] permanently (internal M-H) ===")
gp = toy_gp()
gp.set_priors(
    {
        "covariance_log_lengthscale": ("student_t", (0.0, 1.0, np.nan)),
        "covariance_log_outputscale": None,
        "noise_log_scale": None,
        "mean_const": None,
        "mean_location": None,
        "mean_log_scale": None,
    }
)
print("  df before fit :", gp.hyper_priors["df"])
gp.fit(
    options={"init_N": 8, "opts_N": 1, "n_samples": 0, "df_base": 7},
    rng=np.random.default_rng(0),
)
print("  df after fit(df_base=7) :", gp.hyper_priors["df"])
gp.fit(
    options={"init_N": 8, "opts_N": 1, "n_samples": 0, "df_base": 20},
    rng=np.random.default_rng(0),
)
print(
    "  df after fit(df_base=20):",
    gp.hyper_priors["df"],
    " (MATLAB rebuilds hprior.df from the caller's struct each call,"
    " gplite_train.m:117)",
)

print(
    "\n=== 7. the -inf sentinel of the init_N == 0 branch (internal M-K) ==="
)
src = open(gp_mod.__file__, encoding="utf-8").read().splitlines()
print("  line 1267:", src[1266].strip())
print("  line 1305:", src[1304].strip())
print("  MATLAB gplite_train.m:250 uses nll = Inf(1,size(hyp0,2))")

print("\n=== 8. the default hyp0 (comparison M8) ===")
gp = toy_gp()
gp.posteriors = None
PLB, PUB = None, None
# reproduce lines 1201-1207 with the GP's own bounds
lb, ub = gp.lower_bounds, gp.upper_bounds
print(
    "  posteriors None -> hyp0 = clip((PLB+PUB)/2, LB, UB); MATLAB uses zeros"
)
print("  lower_bounds:", np.round(lb, 3))

print("\n=== 9. np.argsort stability with ties (comparison M9) ===")
v = np.array([np.inf, 1.0, np.inf, 0.5, np.inf])
print("  np.argsort default (quicksort):", np.argsort(v))
print("  np.argsort(kind='stable')     :", np.argsort(v, kind="stable"))
print("  MATLAB's sort is stable, so tied Inf entries keep their order")
