"""W5, a prior whose support differs from the hard bounds of the problem.

`VBMC._init_log_joint` checks the dimension of a prior and nothing else;
`Prior.support()` has no reader in `vbmc.py`. This script settles what a user
sees in the two directions:

1. a prior narrower than the hard bounds (`UniformBox(0, 1)` on the box
   `[0, 10]^2`, plausible bounds inside the prior's support): the log joint
   at a point of the box outside the support, and a short run at the default
   options (the record is the exception and the number of evaluations made
   before it);
2. a prior wider than the hard bounds (`UniformBox(-100, 100)` on the same
   box): the log joint inside the box, which carries the prior's own
   normalization, `-2 log 200`, where a prior truncated to the box and
   normalized again would give `-2 log 10`.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import logging

import numpy as np

import pyvbmc
from pyvbmc import VBMC
from pyvbmc.priors import UniformBox

print("pyvbmc:", pyvbmc.__file__, flush=True)
logging.disable(logging.CRITICAL)

calls = []


def log_likelihood(x):
    x = np.atleast_2d(x)
    calls.append(x.copy())
    return float(-0.5 * np.sum((x - 0.5) ** 2) / 0.2**2)


D = 2
lb, ub = np.zeros((1, D)), np.full((1, D), 10.0)


def build(prior, plb, pub, x0):
    calls.clear()
    return VBMC(
        log_likelihood,
        np.full((1, D), x0),
        lb,
        ub,
        np.full((1, D), plb),
        np.full((1, D), pub),
        options={"display": "off", "max_fun_evals": 60},
        prior=prior,
        seed=1,
    )


print("\n--- 1. prior narrower than the hard bounds", flush=True)
vbmc = build(UniformBox(0.0, 1.0, D=D), 0.2, 0.8, 0.5)
print("   support of the prior:", vbmc.prior.support())
with np.errstate(all="ignore"):
    print(
        "   log joint at (0.5, 0.5):", vbmc.log_joint(np.array([[0.5, 0.5]]))
    )
    print(
        "   log joint at (5, 5)    :", vbmc.log_joint(np.array([[5.0, 5.0]]))
    )
n_before = len(calls)  # the two calls of the log joint above
try:
    vbmc.optimize()
    print("   the run completed,", len(calls), "evaluations")
except Exception as exc:  # the record is the exception itself
    inside = sum(bool(np.all((c >= 0) & (c <= 1))) for c in calls)
    print(
        f"   the run raised at evaluation {len(calls) - n_before} ("
        f"{inside - 1} of its evaluations "
        f"inside the prior's support):"
    )
    print(f"   {type(exc).__name__}: {str(exc)[:400]}")

print("\n--- 2. prior wider than the hard bounds", flush=True)
vbmc = build(UniformBox(-100.0, 100.0, D=D), 0.2, 0.8, 0.5)
x = np.array([[0.5, 0.5]])
print(
    "   log joint - log likelihood at (0.5, 0.5):",
    float(np.ravel(vbmc.log_joint(x))[0]) - log_likelihood(x),
)
print(
    "   -2 log 200 =", -2 * np.log(200.0), "  -2 log 10 =", -2 * np.log(10.0)
)
