"""Stack posteriors saved elsewhere, optimize, sample, save the stack.
Needs Torch.

Usage: python wave2_xver_stack.py <directory>

Part of the two-interpreter experiment behind W2-30 of `../wave2.md`: a
posterior of a bounded problem saved under one minor version of Python and
used under another. Each interpreter needs PyVBMC's dependencies and, with
`PYTHONPATH`, this checkout and the gpyreg checkout; every step runs in a
process of its own, because the failing ones end the interpreter.
"""

import faulthandler
import logging
import sys

faulthandler.enable()
import dill
import numpy as np

logging.disable(logging.CRITICAL)
from pyvbmc import SVBMC
from pyvbmc.variational_posterior import VariationalPosterior

out = sys.argv[1]
vps = [
    VariationalPosterior.load(f"{out}/run{seed}_vp.pkl") for seed in (1, 2, 3)
]
stacked = SVBMC(vps, seed=0)
stacked.optimize(max_steps=30, n_samples=10)
x = np.asarray(stacked.sample(2000))
print(
    "Python",
    sys.version.split()[0],
    "| stacked",
    len(vps),
    "posteriors | sample mean",
    np.round(x.mean(0), 3),
    "| elbo",
    round(float(stacked.elbo), 4),
)
with open(f"{out}/stack.pkl", "wb") as f:
    dill.dump(stacked, f)
data = open(f"{out}/stack.pkl", "rb").read()
print(
    "stack saved,",
    len(data),
    "bytes | holds functions by value:",
    b"_create_function" in data,
)
