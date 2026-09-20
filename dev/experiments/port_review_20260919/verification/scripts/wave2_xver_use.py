"""One step (load, sample, pdf, resave, vbmc_load, vbmc_sample,
vbmc_resave) on a file saved by another interpreter.

Usage: python wave2_xver_use.py <step> <file> <directory>

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
import numpy as np

logging.disable(logging.CRITICAL)
from pyvbmc import VBMC
from pyvbmc.variational_posterior import VariationalPosterior

step, path, out = sys.argv[1], sys.argv[2], sys.argv[3]
if step.startswith("vbmc"):
    obj = VBMC.load(path)
    if step == "vbmc_load":
        print("loaded, iterations", obj.iteration + 1)
    elif step == "vbmc_sample":
        x, _ = obj.vp.sample(2000)
        print("sample mean", np.round(x.mean(0), 3))
    elif step == "vbmc_resave":
        obj.save(out + "/resaved_vbmc.pkl", overwrite=True)
        print("saved again")
else:
    vp = VariationalPosterior.load(path)
    if step == "load":
        print("loaded, K", vp.K)
    elif step == "sample":
        x, _ = vp.sample(2000)
        print("sample mean", np.round(x.mean(0), 3))
    elif step == "pdf":
        print("pdf", np.round(vp.pdf(np.full((1, vp.D), 3.0)).ravel(), 5))
    elif step == "resave":
        vp.save(out + "/resaved_vp.pkl", overwrite=True)
        print("saved again")
