"""W6-A3, a training set whose targets are all equal.

The recommended bounds of the output scale of the kernel come from the range
of the targets, `height = max(y) - min(y)`: `log(height) + log(tol)` and
`log(10 * height)` (`covariance_functions.py:453-454`, as
`gplite_covfun.m:130-131`), and the upper bound of the noise from the same
range (`noise_functions.py:131`, as `gplite_noisefun.m:104`). With equal
targets both are `-inf`. Part 1 fits gpyreg on such a set. Part 2 runs PyVBMC,
capped at a few evaluations past the initial design, on a constant log joint
and on one that is constant over the initial design alone, with hard bounds
and without: the GP is trained on the log joint of the transformed space,
which under a bounded transform carries the log Jacobian and is not constant.

Run from anywhere with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1.
"""

import logging
import traceback

import gpyreg
import numpy as np

import pyvbmc
from pyvbmc import VBMC

print("pyvbmc:", pyvbmc.__file__)
print("gpyreg:", gpyreg.__file__, flush=True)
logging.disable(logging.CRITICAL)


def report(label, call):
    try:
        call()
        print(f"{label}: completed")
    except Exception as err:  # the check is which exception, and from where
        frames = traceback.extract_tb(err.__traceback__)
        where = " <- ".join(
            f"{f.filename.replace(chr(92), '/').split('/')[-1]}:{f.lineno}"
            for f in reversed(frames[-4:])
        )
        print(f"{label}: {type(err).__name__}: {err}")
        print(f"   raised at {where}", flush=True)


# Part 1: gpyreg alone.
rng = np.random.default_rng(0)
X = rng.uniform(-1, 1, (12, 2))
for label, y in (
    ("equal targets", np.full((12, 1), 1.3)),
    ("two values", np.where(X[:, :1] > 0, 1.3, 1.2)),
):
    gp = gpyreg.GP(
        D=2,
        covariance=gpyreg.covariance_functions.SquaredExponential(),
        mean=gpyreg.mean_functions.NegativeQuadratic(),
        noise=gpyreg.noise_functions.GaussianNoise(constant_add=True),
    )
    info = gp.covariance.get_bounds_info(X, y)
    print(
        f"\ngpyreg, {label}: output scale LB {info['LB'][2]},"
        f" UB {info['UB'][2]}; noise UB"
        f" {gp.noise.get_bounds_info(X, y)['UB'][0]}"
    )
    report(
        f"gpyreg fit, {label}",
        lambda: gp.fit(
            X,
            y,
            options={"n_samples": 2, "init_N": 64},
            rng=np.random.default_rng(1),
        ),
    )

# Part 2: PyVBMC.
D = 2
x0 = np.zeros((1, D))
plausible = (np.full((1, D), -1.0), np.full((1, D), 1.0))
HARD = {
    "hard bounds": (np.full((1, D), -5.0), np.full((1, D), 5.0)),
    "no hard bounds": (np.full((1, D), -np.inf), np.full((1, D), np.inf)),
}


def flat_then_normal():
    calls = {"n": 0}

    def fun(x):
        calls["n"] += 1
        if calls["n"] <= 10:  # the ten points of the initial design
            return -3.0
        return float(-0.5 * np.sum(np.asarray(x) ** 2))

    return fun


print()
for bounds_label, hard in HARD.items():
    for label, make in (
        ("constant log joint", lambda: (lambda x: 0.0)),
        ("log joint constant over the initial design", flat_then_normal),
    ):
        vbmc = VBMC(
            make(),
            x0,
            *hard,
            *plausible,
            options={"max_fun_evals": 25, "display": "off"},
            seed=3,
        )
        report(f"PyVBMC, {bounds_label}, {label}", vbmc.optimize)
