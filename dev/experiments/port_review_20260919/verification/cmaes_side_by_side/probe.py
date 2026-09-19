"""Checks before the grid: CMA_stds semantics, capture sanity, run timing."""

import time

import cma
import numpy as np
from harness import FIX, RUN_OF, capture, run_once

from pyvbmc.testing.oracles._state import snapshot_names

P = lambda *a: print(*a, flush=True)

P("cma version:", cma.__version__)

# --- 1. what CMA_stds means for the initial sampling distribution ---------
insig = np.array([1.0, 0.01, 0.3])
x0 = np.zeros(3)
s0 = float(insig.max())
rng = np.random.default_rng(0)
es = cma.CMAEvolutionStrategy(
    x0,
    s0,
    {
        "verbose": -9,
        "seed": np.nan,
        "randn": lambda *sh: rng.standard_normal(sh),
        "CMA_stds": insig / s0,
    },
)
X = np.array(es.ask(200000))
P("CMA_stds on : per-coordinate SD of the initial population", X.std(0))
P("             target insigma                               ", insig)
rng = np.random.default_rng(0)
es = cma.CMAEvolutionStrategy(
    x0,
    s0,
    {
        "verbose": -9,
        "seed": np.nan,
        "randn": lambda *sh: rng.standard_normal(sh),
    },
)
X = np.array(es.ask(200000))
P("CMA_stds off: per-coordinate SD of the initial population", X.std(0))
P(
    "             sigma_vec kind:",
    type(es.sigma_vec).__name__,
    "CMA_diagonal_decoding =",
    es.opts["CMA_diagonal_decoding"],
)
P("")

# --- 2. capture every state ----------------------------------------------
names = snapshot_names(FIX)
P("states:", names)
caps = {}
for n in names:
    t0 = time.perf_counter()
    c = capture(n)
    caps[n] = c
    ins = c["insigma"]
    P(
        f"{n:28s} D={c['D']} K={c['K']:2d} N={c['n_train']:3d} Ns={c['Ns']} "
        f"noisy={c['noisy']} acq={c['acq']} "
        f"aniso={ins.max()/ins.min():8.3f} sigma0={c['sigma0']:.4g} "
        f"maxfev={c['cma_options']['maxfevals']} tolfun={c['cma_options']['tolfun']:.3g} "
        f"f(x0)={c['f_x0']:.6g} intvars={np.any(c['integer_vars'])} "
        f"[capture {time.perf_counter()-t0:.1f}s]"
    )
    P("     insigma =", np.array2string(ins, precision=4))
P("")

# --- 3. one run of each of the four optimizer configurations -------------
for n in names:
    c = caps[n]
    for stds, nonoise in [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ]:
        t0 = time.perf_counter()
        r = run_once(c, 1, stds, nonoise)
        fb = float(c["acq_fun"](r["bestever"]))
        fm = float(c["acq_fun"](r["matlab"]))
        P(
            f"{n:28s} stds={int(stds)} nonoise={int(nonoise)} "
            f"evals={r['n_evals']:5d} iters={r['n_iters']:4d} "
            f"wall={r['wall']:6.2f}s f_best={fb:.8g} f_matlab={fm:.8g} "
            f"aligned={r['aligned']} mean_used={r['mean_used']} stop={r['stop']}"
        )
