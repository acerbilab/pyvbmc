"""Does the final boost run with variable_means=False?

Found by a fix agent of wave 2 (`../../fixes/wave2_agent_A.md`). With the
means of the variational components fixed, `final_boost` asks
`optimize_vp` for `max(vp.K, min_final_components)` components, and the
candidates the sieve builds keep the posterior's `vp.K` means beside
weights and scales for the larger number, so the expected log joint cannot
broadcast them. `misc/finalboost_vbmc.m:6` with `misc/vbinit_vbmc.m:132-136`
reads the same way. A run whose posterior already holds as many components
as `min_final_components`, which is the usual state of a fixed-means run
after warm-up, where the means are the training inputs, is not affected.

Two four-iteration runs at `D = 2`, which end during warm-up with two
components. Before commit `73d2a81` the first raises in the boost and the
second, without a boost, completes; since that commit the boost places the
components at the training inputs and both complete.
"""
import logging
import traceback

import numpy as np

import pyvbmc
from pyvbmc import VBMC

logging.disable(logging.CRITICAL)
print("pyvbmc:", pyvbmc.__file__, flush=True)


def run(options, label):
    D = 2
    v = VBMC(
        lambda x: -0.5 * np.sum(np.atleast_2d(x) ** 2),
        np.zeros((1, D)),
        np.full((1, D), -np.inf),
        np.full((1, D), np.inf),
        np.full((1, D), -1.0),
        np.full((1, D), 1.0),
        options=dict(display="off", max_iter=4, **options),
        seed=3,
    )
    try:
        vp, results = v.optimize()
        print(f"{label}: completed, K = {vp.K}, mu {vp.mu.shape}", flush=True)
    except Exception as err:
        tb = traceback.extract_tb(err.__traceback__)
        where = [
            f"{fr.filename.split('pyvbmc')[-1]}:{fr.lineno} {fr.name}"
            for fr in tb
            if "pyvbmc" in fr.filename
        ]
        print(f"{label}: RAISED {type(err).__name__}: {err}", flush=True)
        print("   at:", " <- ".join(reversed(where[-4:])), flush=True)
        print("   iterations recorded:", len(v.iteration_history["iter"]))
        print("   vp.K", v.vp.K, "mu", v.vp.mu.shape, "gp.X", v.gp.X.shape)


run({"variable_means": False}, "variable_means=False, boost on")
run(
    {"variable_means": False, "do_final_boost": False},
    "variable_means=False, boost off",
)
