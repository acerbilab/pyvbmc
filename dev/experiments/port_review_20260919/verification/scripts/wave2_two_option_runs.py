"""Two short optimize() runs with a non-default option each.

Wave-2 findings: the forced entropy switch (all four reports) and the
separate search GP (P1a comparison F6). Each run is capped at a few
iterations; the point is whether the loop survives the option at all.
"""

import traceback

import numpy as np

import pyvbmc
from pyvbmc import VBMC

print("pyvbmc:", pyvbmc.__file__, flush=True)


def run(D, options, label):
    print(f"\n=== {label}: D = {D}, options = {options} ===", flush=True)

    def f(x):
        return -0.5 * np.sum(np.atleast_2d(x) ** 2)

    v = VBMC(
        f,
        np.zeros((1, D)),
        -10 * np.ones((1, D)),
        10 * np.ones((1, D)),
        -np.ones((1, D)),
        np.ones((1, D)),
        options=dict(display="off", **options),
        seed=3,
    )
    try:
        _, results = v.optimize()
        print("completed:", results["message"], flush=True)
        print("iterations recorded:", len(v.iteration_history["iter"]))
    except Exception as err:
        tb = traceback.extract_tb(err.__traceback__)
        where = [
            f"{fr.filename.split('pyvbmc')[-1]}:{fr.lineno} {fr.name}"
            for fr in tb
            if "pyvbmc" in fr.filename
        ]
        print(f"RAISED {type(err).__name__}: {err}", flush=True)
        print("  at:", " <- ".join(reversed(where[-3:])), flush=True)
        recorded = v.iteration_history["iter"]
        print(
            "  iterations recorded before the failure:",
            0 if recorded is None else len(recorded),
            flush=True,
        )


run(5, {"entropy_switch": True, "max_iter": 3}, "entropy switch")
run(
    4,
    {"entropy_switch": True, "max_iter": 3},
    "entropy switch below det_entropy_min_d (control)",
)
run(2, {"separate_search_gp": True, "max_iter": 4}, "separate search GP")
