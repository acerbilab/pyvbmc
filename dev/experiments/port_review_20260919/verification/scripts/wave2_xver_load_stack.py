"""Load a stack saved by another interpreter and sample from it. Needs
Torch.

Usage: python wave2_xver_load_stack.py <directory>

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
import pyvbmc

out = sys.argv[1]
with open(f"{out}/stack.pkl", "rb") as f:
    stacked = dill.load(f)
x = np.asarray(stacked.sample(2000))
print(
    "Python",
    sys.version.split()[0],
    "| stack loaded | sample mean",
    np.round(x.mean(0), 3),
    "| elbo",
    round(float(stacked.elbo), 4),
)
