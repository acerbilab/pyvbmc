import sys
import traceback

import numpy as np

sys.path.insert(0, r"C:/Users/luigi/Documents/GitHub/pyvbmc")
import pymc as pm
from pytensor.gradient import NullTypeGradError
from pytensor.graph.utils import MethodNotDefined

print("MethodNotDefined bases:", MethodNotDefined.__mro__, flush=True)
print("NullTypeGradError bases:", NullTypeGradError.__mro__, flush=True)
print(
    "issubclass(MethodNotDefined, NotImplementedError):",
    issubclass(MethodNotDefined, NotImplementedError),
    flush=True,
)

from pyvbmc.pymc import PyMCTarget

with pm.Model() as m:
    pm.Rice("x", nu=1.0, sigma=1.0)
try:
    PyMCTarget(m, seed=0)
except Exception:
    tb = traceback.format_exc()
    print(tb[-1800:], flush=True)
