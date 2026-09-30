"""W7-17: a capped VBMC run whose starting points share one coordinate.

With at least `fun_eval_start` starting points, the initial design is those
points; if they share a coordinate, gpyreg's recommended length-scale bounds
for it are (-inf, -inf), and the first GP fit is expected to fail. A control
run with generic starting points shows the setup otherwise runs.
"""
import traceback

import gpyreg
import numpy as np

import pyvbmc
from pyvbmc import VBMC

print("pyvbmc:", pyvbmc.__file__)
print("gpyreg:", gpyreg.__file__)
D = 2


def log_density(x):
    x = np.atleast_2d(x)
    return float(-0.5 * np.sum(((x - [0.2, -0.3]) / [1.0, 0.7]) ** 2))


def run(label, x0):
    options = {"max_iter": 3, "display": "off", "plot": False}
    vbmc = VBMC(
        log_density,
        x0,
        np.full((1, D), -np.inf),
        np.full((1, D), np.inf),
        np.full((1, D), -3.0),
        np.full((1, D), 3.0),
        options=options,
        seed=20260923,
    )
    print(
        f"== {label}: {x0.shape[0]} starting points, fun_eval_start "
        f"{vbmc.options.get('fun_eval_start')}, spread per coordinate "
        f"{np.ptp(x0, axis=0)}",
        flush=True,
    )
    try:
        vp, results = vbmc.optimize()
        print(
            f"   completed: {results['iterations']} iterations, "
            f"{results['func_count']} evaluations, elbo {results['elbo']:.4f}",
            flush=True,
        )
    except Exception as err:
        print(f"   raised {type(err).__name__}: {err}", flush=True)
        tb = traceback.extract_tb(err.__traceback__)
        for frame in tb[-6:]:
            print(
                f"     {frame.filename.split('GitHub')[-1]}:{frame.lineno} "
                f"in {frame.name}",
                flush=True,
            )


rng = np.random.default_rng(1)
shared = np.column_stack([np.linspace(-2, 2, 10), np.full(10, 0.5)])
shared12 = np.column_stack([np.linspace(-2, 2, 12), np.full(12, 0.5)])
generic = rng.uniform(-2, 2, size=(10, D))
run("shared coordinate, 10 points", shared)
run("shared coordinate, 12 points", shared12)
run("control, 10 generic points", generic)
