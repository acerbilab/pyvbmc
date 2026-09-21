"""P9-4. Do NaN and infinite constructor arguments pass the argument checks
of every family, and what object do they produce?

Settles: the table of consequences (construction, density, sample) for NaN
and infinite arguments in UniformBox, Trapezoidal, SplineTrapezoidal and
SmoothBox; the NaN-pivot case where `sample` is uniform on [a, b] while the
density is zero everywhere; and what the MATLAB functions do with the same
arguments.
"""

import matlab_transcription as M
import numpy as np

import pyvbmc
from pyvbmc.priors import SmoothBox, SplineTrapezoidal, Trapezoidal, UniformBox

print("pyvbmc.__file__ =", pyvbmc.__file__)
print()

inf = np.inf
nan = np.nan
rng = np.random.default_rng(1)

cases = [
    ("UniformBox(nan, 1)", UniformBox, (nan, 1.0)),
    ("UniformBox(0, nan)", UniformBox, (0.0, nan)),
    ("UniformBox(-inf, inf)", UniformBox, (-inf, inf)),
    ("UniformBox(0, inf)", UniformBox, (0.0, inf)),
    ("Trapezoidal(0, nan, .75, 1)", Trapezoidal, (0.0, nan, 0.75, 1.0)),
    ("Trapezoidal(-inf, 0, 1, inf)", Trapezoidal, (-inf, 0.0, 1.0, inf)),
    (
        "SplineTrapezoidal(0, nan, .75, 1)",
        SplineTrapezoidal,
        (0.0, nan, 0.75, 1.0),
    ),
    (
        "SplineTrapezoidal(-inf, 0, 1, inf)",
        SplineTrapezoidal,
        (-inf, 0.0, 1.0, inf),
    ),
    ("SmoothBox(0, 1, nan)", SmoothBox, (0.0, 1.0, nan)),
    ("SmoothBox(0, 1, inf)", SmoothBox, (0.0, 1.0, inf)),
    ("SmoothBox(-inf, inf, 1)", SmoothBox, (-inf, inf, 1.0)),
]

grid = np.array([[-5.0], [0.1], [0.5], [0.9], [5.0]])
print(
    f"{'constructor':36s} {'built?':7s} density at "
    "[-5, .1, .5, .9, 5]   -> sample(6)"
)
for name, cls, args in cases:
    try:
        p = cls(*args)
    except Exception as e:
        print(f"{name:36s} refused  {type(e).__name__}: {e}")
        continue
    with np.errstate(all="ignore"):
        try:
            d = p.pdf(grid, keepdims=False)
            dstr = np.array2string(d, precision=3)
        except Exception as e:
            dstr = f"pdf raised {type(e).__name__}"
        try:
            s = p.sample(6, rng=rng)
            sstr = np.array2string(s.ravel(), precision=3)
        except Exception as e:
            sstr = f"sample raised {type(e).__name__}: {e}"
    print(f"{name:36s} built    {dstr:34s} -> {sstr}")
print()

print("--- the NaN-pivot case: sample vs density, 20000 draws ---")
for cls in (Trapezoidal, SplineTrapezoidal):
    p = cls(0.0, nan, 0.75, 1.0)
    s = p.sample(20000, rng=np.random.default_rng(3)).ravel()
    with np.errstate(all="ignore"):
        dens = p.pdf(np.linspace(0.01, 0.99, 9).reshape(-1, 1), keepdims=False)
    print(
        f"{cls.__name__:20s} sample min/max/mean = "
        f"{s.min():.4f}/{s.max():.4f}/{s.mean():.4f}; "
        f"density on (0,1) = {np.unique(dens)}"
    )
    hist, edges = np.histogram(s, bins=5, range=(0, 1), density=True)
    print(
        f"{'':20s} histogram density on 5 bins = "
        f"{np.array2string(hist, precision=3)}  (U(0,1) would be 1)"
    )
print()

print("--- MATLAB with the same arguments ---")
with np.errstate(all="ignore"):
    print("munifboxlogpdf(x, nan, 1)      ->", end=" ")
    try:
        print(M.munifboxlogpdf(grid, nan, 1.0))
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
    print("munifboxlogpdf(x, -inf, inf)   ->", end=" ")
    try:
        print(M.munifboxlogpdf(grid, -inf, inf))
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
    print(
        "mtrapezlogpdf(x, 0, nan, .75, 1) ->",
        M.mtrapezlogpdf(grid, 0.0, nan, 0.75, 1.0),
    )
    print(
        "msplinetrapezlogpdf(x, 0, nan, .75, 1) ->",
        M.msplinetrapezlogpdf(grid, 0.0, nan, 0.75, 1.0),
    )
    print("msmoothboxlogpdf(x, 0, 1, nan) ->", end=" ")
    try:
        print(M.msmoothboxlogpdf(grid, 0.0, 1.0, nan))
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
    print("msmoothboxlogpdf(x, 0, 1, inf) ->", end=" ")
    try:
        print(M.msmoothboxlogpdf(grid, 0.0, 1.0, inf))
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
print()
print("MATLAB checks read from the sources:")
print("  munifboxlogpdf.m:44  any(a(:) >= b(:))     -> OrderError")
print("  msmoothboxlogpdf.m:26 any(sigma(:) <= 0)   -> NonPositiveSigma")
print("  msmoothboxlogpdf.m:46 any(a(:) >= b(:))    -> OrderError")
print("  mtrapezlogpdf.m / msplinetrapezlogpdf.m: no order check at all")
print("  All of these comparisons are False for NaN, as in Python.")
print()

print("--- MATLAB rejection sampler with a NaN pivot (mtrapezrnd) ---")
r = M.mtrapezrnd(
    np.array([[0.0]]),
    np.array([[nan]]),
    np.array([[0.75]]),
    np.array([[1.0]]),
    20000,
    np.random.default_rng(4),
)
print(
    "mtrapezrnd(0, nan, .75, 1, 20000): min/max/mean =",
    f"{r.min():.4f}/{r.max():.4f}/{r.mean():.4f}",
)
hist, _ = np.histogram(r.ravel(), bins=5, range=(0, 1), density=True)
print("  histogram density =", np.array2string(hist, precision=3))
