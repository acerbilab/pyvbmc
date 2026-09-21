"""P9-8. Does `sample_prior` alone reach `_init_log_joint`'s prior branch and
pass the PyMC-target guard?

Settles: that `convert_to_prior(sample_prior=f)` alone returns a
`UserFunction` whose `log_pdf` is None; that `_init_log_joint` enters the
prior branch for `sample_prior` alone and then falls through to the plain
log-joint; and what a constructed `VBMC` then holds.  The PyMC half is a
separate run in the PyMC environment.
"""

import inspect

import numpy as np

import pyvbmc
from pyvbmc import VBMC
from pyvbmc.priors import convert_to_prior

print("pyvbmc.__file__ =", pyvbmc.__file__)
print()

f = lambda n: np.zeros((n, 2))
p = convert_to_prior(None, None, f, 2)
print(
    "convert_to_prior(sample_prior=f, D=2) ->",
    type(p).__name__,
    "| log_pdf is None:",
    p.log_pdf is None,
    "| D =",
    p.D,
    "| sample is f:",
    p.sample is f,
)
print()

print("--- the guard and the branch, read from the present code ---")
src = inspect.getsource(VBMC._resolve_pymc_target)
i = src.find("if prior is not None")
print("   vbmc.py guard:", " ".join(src[i : i + 160].split()))
src2 = inspect.getsource(VBMC._init_log_joint)
print(
    "   _init_log_joint branch:",
    " ".join(src2[src2.find("if (") : src2.find("if (") + 150].split()),
)
print()

print("--- a constructed VBMC with sample_prior only ---")
lb, ub = np.array([[-5.0, -5.0]]), np.array([[5.0, 5.0]])
plb, pub = np.array([[-1.0, -1.0]]), np.array([[1.0, 1.0]])
x0 = np.array([[0.0, 0.0]])
ll = lambda t: -0.5 * float(np.sum(np.asarray(t) ** 2))
v = VBMC(ll, x0, lb, ub, plb, pub, sample_prior=f, options={"display": "off"})
print(
    "vbmc.prior =",
    type(v.prior).__name__,
    "| prior.log_pdf is None:",
    v.prior.log_pdf is None,
)
print("vbmc.log_likelihood is None:", v.log_likelihood is None)
print(
    "log_joint(x0) =",
    v.log_joint(x0.ravel()),
    " raw log_density(x0) =",
    ll(x0.ravel()),
)
print(
    "=> log_density is used verbatim as the log joint:",
    v.log_joint(x0.ravel()) == ll(x0.ravel()),
)
print()
print("__str__ line about the prior:")
for line in str(v).splitlines():
    if "prior" in line.lower() or "log-density" in line.lower():
        print("   ", line.strip())
