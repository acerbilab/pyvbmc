"""W4-2: a run with an integer variable, at the default options otherwise.

`AbstractAcqFcn._real2int` indexes its input as `X[:, integer_vars]`, and
`active_sample` hands it the one-dimensional result of the local search
(`active_sample.py:624`) when that search improves on the sieve's best
candidate. This script settles whether a run with `integer_vars` set
completes: first the bare call on a one-dimensional and on a two-dimensional
point, then `VBMC.optimize()` on a two-dimensional target whose first
variable is an integer, with the default search optimizer and, for
comparison, with `search_optimizer="none"`, which skips the call.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import traceback

import gpyreg
import numpy as np

import pyvbmc
from pyvbmc import VBMC
from pyvbmc.acquisition_functions import AbstractAcqFcn
from pyvbmc.parameter_transformer import ParameterTransformer

print("pyvbmc:", pyvbmc.__file__, flush=True)
print("gpyreg:", gpyreg.__file__, flush=True)

lb = np.array([[-10.5, -10.0]])
ub = np.array([[10.5, 10.0]])
plb = np.array([[-4.5, -3.0]])
pub = np.array([[4.5, 3.0]])
mask = np.array([True, False])

print("\n--- the bare call", flush=True)
pt = ParameterTransformer(2, lb, ub, plb, pub)
x2 = pt(np.array([[1.3, 0.4]]))
print("2-D input:", pt.inverse(AbstractAcqFcn._real2int(x2.copy(), pt, mask)))
try:
    AbstractAcqFcn._real2int(x2[0].copy(), pt, mask)
    print("1-D input: returned")
except Exception as exc:  # the record is the exception itself
    print(f"1-D input: {type(exc).__name__}: {exc}")


def target(x):
    x = np.atleast_2d(x)
    return float(
        -0.5 * ((x[0, 0] - 2.0) / 2.0) ** 2 - 0.5 * (x[0, 1] / 1.5) ** 2
    )


def run(label, extra):
    print(f"\n--- optimize(), {label}", flush=True)
    options = {
        "integer_vars": mask,
        "max_fun_evals": 40,
        "display": "off",
        "plot": False,
    }
    options.update(extra)
    vbmc = VBMC(
        target,
        np.array([[1.0, 0.5]]),
        lb,
        ub,
        plb,
        pub,
        options=options,
        seed=20260920,
    )
    try:
        vp, results = vbmc.optimize()
    except Exception:
        tb = traceback.format_exc().strip().splitlines()
        print("raised:", tb[-1])
        frames = [ln.strip() for ln in tb if ln.strip().startswith("File")]
        for ln in frames[-3:]:
            print("   ", ln)
        logger = vbmc.function_logger
        print(
            "    evaluations made before the exception:",
            int(np.sum(logger.n_evals[: logger.Xn + 1])),
        )
        return
    X = vbmc.function_logger.X_orig[vbmc.function_logger.X_flag]
    on_grid = np.abs(X[:, 0] - np.round(X[:, 0])) < 1e-9
    print(
        f"completed: {results['func_count']} evaluations, "
        f"elbo={results['elbo']:.4f}; evaluated points with an integer "
        f"first coordinate: {int(on_grid.sum())} of {X.shape[0]}"
    )


run("default search optimizer", {})
run('search_optimizer="none"', {"search_optimizer": "none"})
