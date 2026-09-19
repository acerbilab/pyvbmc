import logging
import sys

import numpy as np

sys.path.insert(0, r"C:/Users/luigi/Documents/GitHub/pyvbmc")
logging.disable(logging.WARNING)
from pyvbmc.pymc import PyMCTarget
from pyvbmc.testing.pymc import models as M

spec = [s for s in M.accepted_models() if s["name"] == "one_sided"][0]
t = PyMCTarget(
    spec["model"],
    plausible_bounds={"t": (1.5, 4.0), "v": (-4.0, -1.0)},
    start={"t": 2.0, "v": -2.0},
    seed=8,
)
print("support", t.support, flush=True)
print("kept", t.kept, "coordinate_names", t.coordinate_names, flush=True)
print("plb", t.plb, "pub", t.pub, flush=True)
# forward-map the model plausible bounds by hand
print("t: log(1.5-1)=", np.log(0.5), "log(4-1)=", np.log(3.0), flush=True)
print(
    "v: log(-0.5-(-4))=",
    np.log(3.5),
    "log(-0.5-(-1))=",
    np.log(0.5),
    flush=True,
)
print(
    "x0",
    t.x0,
    "expected",
    np.log(2.0 - 1.0),
    np.log(-0.5 - (-2.0)),
    flush=True,
)
# monotonicity direction of the kept maps
for name, probes in (("t", [1.2, 2.0, 5.0]), ("v", [-5.0, -2.0, -0.6])):
    xs = [
        t.from_model_variables(
            {"t": p if name == "t" else 2.0, "v": p if name == "v" else -2.0}
        )
        for p in probes
    ]
    print(
        name,
        "model",
        probes,
        "-> coords",
        [float(x[0 if name == "t" else 1]) for x in xs],
        flush=True,
    )
# to_model_variables at extreme coordinates
for c in (-50.0, 0.0, 50.0, 500.0, 800.0):
    try:
        print(
            "coord",
            c,
            "->",
            t.to_model_variables(np.array([[c, c]])),
            flush=True,
        )
    except Exception as e:
        print("coord", c, "->", type(e).__name__, str(e)[:80], flush=True)
