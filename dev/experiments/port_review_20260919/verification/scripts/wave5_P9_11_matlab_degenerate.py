"""P9-11. What do the MATLAB trapezoid functions compute at the degenerate
pivots that PyVBMC refuses (u == a, u == v, v == b), and what would the two
rejection samplers do there?

Settles: the MATLAB log-density values at those limits, their
normalization by quadrature, the PyVBMC refusals, and the behavior of both
sides' rejection samplers at the same arguments.
"""

import matlab_transcription as M
import numpy as np
from scipy.integrate import quad

import pyvbmc
from pyvbmc.priors import SplineTrapezoidal, Trapezoidal

print("pyvbmc.__file__ =", pyvbmc.__file__)
print()

print("--- PyVBMC refusals ---")
for cls in (Trapezoidal, SplineTrapezoidal):
    for name, args in (
        ("u == a", (0.0, 0.0, 0.75, 1.0)),
        ("u == v", (0.0, 0.5, 0.5, 1.0)),
        ("v == b", (0.0, 0.25, 1.0, 1.0)),
        ("a == b", (1.0, 1.0, 1.0, 1.0)),
    ):
        try:
            cls(*args)
            print(f"{cls.__name__:20s} {name:8s} {args} -> built")
        except Exception as e:
            print(
                f"{cls.__name__:20s} {name:8s} {args} -> "
                f"{type(e).__name__}: {e}"
            )
print()

print("--- MATLAB values at the same arguments ---")
with np.errstate(all="ignore"):
    print(
        "mtrapezlogpdf(0.5, 0, .5, .5, 1)   =",
        M.mtrapezlogpdf([[0.5]], 0.0, 0.5, 0.5, 1.0),
        "   log 2 =",
        np.log(2.0),
    )
    print(
        "mtrapezlogpdf(0.5, 0, .25, 1, 1)   =",
        M.mtrapezlogpdf([[0.5]], 0.0, 0.25, 1.0, 1.0),
        "   log(1/0.875) =",
        np.log(1 / 0.875),
    )
    print(
        "mtrapezlogpdf(0.5, 0, 0, .75, 1)   =",
        M.mtrapezlogpdf([[0.5]], 0.0, 0.0, 0.75, 1.0),
        " (u == a)",
    )
    print(
        "mtrapezlogpdf(0.9, 0, 0, .75, 1)   =",
        M.mtrapezlogpdf([[0.9]], 0.0, 0.0, 0.75, 1.0),
        " (u == a, right ramp)",
    )
    print(
        "msplinetrapezlogpdf(0.5, 0, .5, .5, 1) =",
        M.msplinetrapezlogpdf([[0.5]], 0.0, 0.5, 0.5, 1.0),
    )
    print(
        "msplinetrapezlogpdf(0.5, 0, 0, .75, 1) =",
        M.msplinetrapezlogpdf([[0.5]], 0.0, 0.0, 0.75, 1.0),
        " (u == a)",
    )
    print(
        "msplinetrapezlogpdf(0.5, 0, .25, 1, 1) =",
        M.msplinetrapezlogpdf([[0.5]], 0.0, 0.25, 1.0, 1.0),
        " (v == b)",
    )
print()

print("--- are the MATLAB degenerate densities normalized? (quadrature) ---")


def integ(f, args, lo, hi):
    with np.errstate(all="ignore"):
        val, _ = quad(
            lambda t: float(np.exp(f([[t]], *args))[0]), lo, hi, limit=400
        )
    return val


rows = [
    ("mtrapez u==v (triangular)", M.mtrapezlogpdf, (0.0, 0.5, 0.5, 1.0)),
    ("mtrapez v==b", M.mtrapezlogpdf, (0.0, 0.25, 1.0, 1.0)),
    ("mtrapez u==a", M.mtrapezlogpdf, (0.0, 0.0, 0.75, 1.0)),
    ("msplinetrapez u==v", M.msplinetrapezlogpdf, (0.0, 0.5, 0.5, 1.0)),
    ("msplinetrapez v==b", M.msplinetrapezlogpdf, (0.0, 0.25, 1.0, 1.0)),
    ("msplinetrapez u==a", M.msplinetrapezlogpdf, (0.0, 0.0, 0.75, 1.0)),
]
for name, f, args in rows:
    print(f"{name:28s} integral over [0,1] = {integ(f, args, 0.0, 1.0)!r}")
print()

print("--- the rejection samplers at u == v and v == b ---")
print("(both sides draw x0 = 0.5*(u+v) and reject against y_max = pdf(x0))")
for name, args in (
    ("u == v (triangular)", (0.0, 0.5, 0.5, 1.0)),
    ("v == b", (0.0, 0.25, 1.0, 1.0)),
    ("u == a", (0.0, 0.0, 0.75, 1.0)),
):
    a, u, v, b = args
    x0 = 0.5 * (u + v)
    with np.errstate(all="ignore"):
        ymax = float(M.mtrapezpdf([[x0]], a, u, v, b)[0])
    print(f"  mtrapez {name:20s}: x0 = {x0}, y_max = pdf(x0) = {ymax!r}")
    if np.isfinite(ymax) and ymax > 0:
        r = M.mtrapezrnd(
            np.array([[a]]),
            np.array([[u]]),
            np.array([[v]]),
            np.array([[b]]),
            40000,
            np.random.default_rng(7),
        ).ravel()
        hist, edges = np.histogram(r, bins=10, range=(0, 1), density=True)
        mid = 0.5 * (edges[:-1] + edges[1:])
        with np.errstate(all="ignore"):
            truth = M.mtrapezpdf(mid.reshape(-1, 1), a, u, v, b)
        print(
            f"     histogram rmse against the density = "
            f"{float(np.sqrt(np.mean((hist - truth) ** 2))):.4f}"
        )
    else:
        print("     y_max is not a usable envelope height")
