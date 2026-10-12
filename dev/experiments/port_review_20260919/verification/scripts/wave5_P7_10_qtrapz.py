"""P7-10: qtrapz.m against scipy.integrate.trapezoid at unit spacing.

Settles whether the two rules differ in endpoint handling, as the
known-differences sheet's entry says, by formula and on random vectors,
and checks that mtv and vbmc_mtv.m apply the spacing the same way.
"""

import numpy as np
from scipy.integrate import trapezoid

import pyvbmc

print("pyvbmc.__file__ =", pyvbmc.__file__)


def qtrapz(y):
    """shared/qtrapz.m, the vector case (n = 1, dim = the non-singleton one).

    z = sum(y) - 0.5*(y(1) + y(end)).
    """
    y = np.asarray(y, dtype=float).ravel()
    return np.sum(y) - 0.5 * (y[0] + y[-1])


rng = np.random.default_rng(31415)
print("\n== identical formula? ==")
for n in [2, 3, 5, 1000, 8192]:
    y = rng.standard_normal(n)
    a = qtrapz(y)
    b = trapezoid(y)
    print(f"  n={n:>5}: qtrapz={a!r}  trapezoid={b!r}  diff={a - b!r}")

print("\n== on a positive density-like vector (as mtv uses it) ==")
x = np.linspace(-6, 6, 8192)
y = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
dx = x[1] - x[0]
print("  qtrapz(y)*dx    =", qtrapz(y) * dx)
print("  trapezoid(y)*dx =", trapezoid(y) * dx)
print("  difference      =", qtrapz(y) * dx - trapezoid(y) * dx)
print("  scipy trapezoid with dx= given:", trapezoid(y, dx=dx))

print("\n== the algebra ==")
print(
    "  sum(y) - (y0+yn)/2 = sum_i (y_i + y_{i+1})/2 for a uniform grid:",
    np.allclose(qtrapz(y), np.sum(0.5 * (y[:-1] + y[1:])), rtol=0, atol=0),
    " bitwise:",
    qtrapz(y) == np.sum(0.5 * (y[:-1] + y[1:])),
)

print("\n== how the spacing is applied ==")
print(
    "  PyVBMC mtv: trapezoid(yy1) * (x1mesh[1] - x1mesh[0])"
    "   [variational_posterior.py:1379, :1383]"
)
print(
    "  PyVBMC mtv: 0.5 * trapezoid(f(xx_range)) * (xx_range[1]-xx_range[0])"
    "   [:1406-1408]"
)
print(
    "  MATLAB     : qtrapz(yy1)*(x1mesh(2)-x1mesh(1))   [vbmc_mtv.m:68, :71]"
)
print(
    "  MATLAB     : 0.5*qtrapz(f(xx_range))*(xx_range(2)-xx_range(1))"
    "   [vbmc_mtv.m:77]"
)
