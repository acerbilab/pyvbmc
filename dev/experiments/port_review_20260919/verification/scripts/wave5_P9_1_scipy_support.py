"""P9-1. Does `SciPy.support()` report the unshifted support of a frozen
univariate distribution, and does `Product` copy those numbers?

Settles: (a) what SciPy itself stores in `.a`/`.b` of a frozen distribution,
by reading `rv_frozen.__init__` from the installed SciPy and by measuring;
(b) the values `SciPy.support()` returns for shifted/scaled distributions;
(c) that `Product.__init__` copies them into `Product.a`/`.b`;
(d) what the package's own unit-integral test helper computes when it
integrates over a wrong support box.
"""

import inspect

import numpy as np
import scipy
import scipy.stats as sps
from scipy.integrate import nquad

import pyvbmc
from pyvbmc.priors import Product, SciPy, UniformBox

print("pyvbmc.__file__ =", pyvbmc.__file__)
print("numpy", np.__version__, "scipy", scipy.__version__)
print()

# (a) SciPy's own code for a frozen distribution's .a/.b
src = inspect.getsource(sps._distn_infrastructure.rv_frozen.__init__)
print("--- scipy rv_frozen.__init__ ---")
print(src)
print("file:", inspect.getsourcefile(sps._distn_infrastructure.rv_frozen))

print("--- frozen .a/.b vs frozen.support() ---")
cases = {
    "uniform(loc=2, scale=3)": sps.uniform(loc=2, scale=3),
    "beta(2,3,loc=-1,scale=4)": sps.beta(2, 3, loc=-1, scale=4),
    "beta(2,1.5,loc=1,scale=4)": sps.beta(2, 1.5, loc=1, scale=4),
    "expon(loc=5)": sps.expon(loc=5),
    "gamma(2,loc=3,scale=2)": sps.gamma(2, loc=3, scale=2),
    "lognorm(1.0,loc=4)": sps.lognorm(1.0, loc=4),
    "norm()": sps.norm(),
    "norm(loc=3, scale=2)": sps.norm(loc=3, scale=2),
    "lognorm(1.0)": sps.lognorm(1.0),
    "beta(2,1.5)": sps.beta(2, 1.5),
    "t(3)": sps.t(3),
}
for name, dist in cases.items():
    p = SciPy(dist)
    lb, ub = p.support()
    print(
        f"{name:28s} .a/.b = ({dist.a:g}, {dist.b:g})  "
        f"frozen.support() = {tuple(float(s) for s in dist.support())}  "
        f"ppf(0)/ppf(1) = ({dist.ppf(0):g}, {dist.ppf(1):g})  "
        f"SciPy.support() = ({lb[0]:g}, {ub[0]:g})"
    )
print()

# (c) Product copies them
prod = Product([sps.uniform(loc=2, scale=3), sps.uniform(loc=-5, scale=1)])
print("--- Product of two shifted uniforms ---")
print("Product.a =", prod.a, " Product.b =", prod.b)
print("Product.support() =", prod.support())
print(prod)
print()
prod2 = Product([sps.uniform(loc=2, scale=3), UniformBox(0, 1)])
print(
    "Product([uniform(loc=2,scale=3), UniformBox(0,1)]).a,.b =",
    prod2.a,
    prod2.b,
)
print()

# (d) the package's own unit-integral helper, over the reported support
print("--- integral of the density over the reported support ---")
for name, dist in [
    ("SciPy(uniform(loc=2, scale=3))", sps.uniform(loc=2, scale=3)),
    ("SciPy(norm())", sps.norm()),
]:
    p = SciPy(dist)
    lb, ub = p.support()
    val, err = nquad(lambda *x: p.pdf(np.array(x)).item(), list(zip(lb, ub)))
    print(f"{name:32s} support=({lb[0]:g},{ub[0]:g})  integral = {val!r}")

p = Product([sps.uniform(loc=2, scale=3), UniformBox(0, 1)])
lb, ub = p.support()
val, err = nquad(lambda *x: p.pdf(np.array(x)).item(), list(zip(lb, ub)))
print(
    f"{'Product(shifted unif, UniformBox)':32s} "
    f"support={list(zip(lb, ub))}  integral = {val!r}"
)

# The sample() of a shifted SciPy prior against its reported support box
rng = np.random.default_rng(0)
s = SciPy(sps.uniform(loc=2, scale=3)).sample(1000, rng=rng)
print(
    "sample range of SciPy(uniform(loc=2,scale=3)):",
    float(s.min()),
    float(s.max()),
)
