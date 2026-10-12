"""Which functions does dill pickle by value (F1) or by reference (F2) when
it saves a VBMC instance, its variational posterior alone, or a stored VP?"""

import io
import logging
import re
import sys

import dill
import dill.detect
import numpy as np

logging.disable(logging.CRITICAL)
from pyvbmc import VBMC
from pyvbmc.variational_posterior import VariationalPosterior


def traced(obj, label):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger("dill")
    logging.disable(logging.NOTSET)
    logger.addHandler(handler)
    dill.detect.trace(True)
    try:
        data = dill.dumps(obj)
    finally:
        dill.detect.trace(False)
        logger.removeHandler(handler)
        logging.disable(logging.CRITICAL)
    lines = stream.getvalue().splitlines()
    by_value = sorted(
        {
            re.sub(r" at 0x[0-9A-Fa-f]+", "", l.split("F1:")[1].strip())
            for l in lines
            if "F1:" in l
        }
    )
    code = sum(1 for l in lines if re.search(r"\bCo:", l))
    print(
        f"{label}: {len(data)} bytes; functions by value: {len(by_value)}; code objects: {code}"
    )
    for name in by_value:
        print("     F1", name[:110])


print("Python", sys.version.split()[0], "dill", dill.__version__)
v = VBMC.load("pyvbmc/testing/vbmc/test_vbmc_save_static.pkl")
traced(v, "whole VBMC instance (static fixture)")
traced(v.vp, "its variational posterior alone")
traced(v.iteration_history["vp"][3], "a recorded variational posterior")
vp_file = VariationalPosterior.load(
    "pyvbmc/testing/variational_posterior/test_vp_save_static.pkl"
)
traced(vp_file, "VP loaded from test_vp_save_static.pkl")

fresh = VBMC(
    lambda x: -0.5 * np.sum(x**2),
    np.zeros((1, 2)),
    np.full((1, 2), -np.inf),
    np.full((1, 2), np.inf),
    np.full((1, 2), -1.0),
    np.full((1, 2), 1.0),
    options={"display": "off", "max_iter": 2},
    seed=1,
)
vp, _ = fresh.optimize()
traced(vp, "VP returned by a fresh run")
