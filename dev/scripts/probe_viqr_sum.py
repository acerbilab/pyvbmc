"""Time alternative evaluations of VIQR's existing sum on an oracle state.

This bounded diagnostic rebuilds a stored GP and draws candidates; it never
fits a model or calls a target. The alternatives operate on saved intermediate
arrays and do not modify the acquisition implementation. Run from the repo::

    .venv/Scripts/python.exe dev/scripts/probe_viqr_sum.py --out <file.json>
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import gpyreg
import numpy as np
import scipy
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pyvbmc.acquisition_functions import AcqFcnVIQR
from pyvbmc.testing.oracles._oracles import (
    prepare_gp_for_acq,
    prepare_importance_sampling,
)
from pyvbmc.testing.oracles._state import build_state, load_snapshot


def original(a):
    """The production log-sinh followed by log-sum-exp, with a = u*s."""
    z = a + np.log1p(-np.exp(-2 * a))
    m = np.max(z, axis=1)
    m[m == -np.inf] = 0.0
    return m + np.log(np.sum(np.exp(z - m[:, None]), axis=1))


def direct(a):
    """Unscaled identity; only timed when all arguments are in [0, 300]."""
    return np.log(np.sum(np.sinh(a), axis=1)) + np.log(2.0)


def scaled(a):
    """Scaled identity for finite nonnegative arguments, including zeros."""
    m = np.max(a, axis=1, keepdims=True)
    return m[:, 0] + np.log(
        np.sum(np.exp(a - m) * (-np.expm1(-2 * a)), axis=1)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fixture = (
        ROOT / "pyvbmc/testing/oracles/fixtures/rosenbrock_D2_noise1_viqr"
    )
    state = build_state(load_snapshot(fixture))
    gp, vp, optim = state["gp"], state["vp"], state["optim_state"]
    acq = AcqFcnVIQR()
    prepare_gp_for_acq(gp, state["logger"], optim)
    prepare_importance_sampling(state, acq, 20260913)
    xs = vp.sample(8192, orig_flag=False)[0]
    importance = optim["active_importance_sampling"]
    _, fs2 = gp.predict(xs, separate_samples=True)
    ys2 = fs2 + acq._estimate_observation_noise(xs, gp, optim)[:, None]
    arrays = []
    for s, posterior in enumerate(gp.posteriors):
        ell = np.exp(posterior.hyp[: gp.D])
        sf2 = np.exp(2 * posterior.hyp[gp.D])
        kx = sf2 * np.exp(-cdist(xs / ell, gp.X / ell, "sqeuclidean") / 2)
        ka = sf2 * np.exp(
            -cdist(xs / ell, importance["X"] / ell, "sqeuclidean") / 2
        )
        correction = kx @ importance["C_tmp"][s]
        c = ka - correction if posterior.L_chol else ka + correction
        tau2 = c**2 / ys2[:, s, None]
        arrays.append(
            acq.u * np.sqrt(np.maximum(importance["f_s2"][:, s] - tau2, 0))
        )
    assert all(np.all((a >= 0) & (a <= 300)) for a in arrays)
    methods = {"original": original, "direct": direct, "scaled": scaled}
    reference = [original(a) for a in arrays]
    errors = {
        name: max(
            float(np.max(np.abs(fn(a) - r))) for a, r in zip(arrays, reference)
        )
        for name, fn in methods.items()
    }
    timings = {name: [] for name in methods}
    # Warm all methods, then rotate order to reduce timing-order bias.
    for fn in methods.values():
        for a in arrays:
            fn(a)
    names = list(methods)
    for repeat in range(9):
        order = names[repeat % 3 :] + names[: repeat % 3]
        for name in order:
            start = time.perf_counter()
            for a in arrays:
                methods[name](a)
            timings[name].append(1000 * (time.perf_counter() - start))
    result = {
        "purpose": "Standard VIQR sum only; no complete optimized acquisition or run timing",
        "fixture": fixture.name,
        "fixture_sha256": {
            ext: hashlib.sha256(
                fixture.with_suffix(ext).read_bytes()
            ).hexdigest()
            for ext in (".json", ".npz")
        },
        "probe_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "seed": 20260913,
        "shape": {
            "D": gp.D,
            "N": len(gp.X),
            "Ns": len(arrays),
            "Nc": len(xs),
            "Na": 100,
        },
        "argument_range": [
            float(min(a.min() for a in arrays)),
            float(max(a.max() for a in arrays)),
        ],
        "max_abs_error_vs_original": errors,
        "milliseconds": timings,
        "median_milliseconds": {
            name: float(np.median(t)) for name, t in timings.items()
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "gpyreg_path": gpyreg.__file__,
            "blas_threads": 1,
        },
        "limitations": "One early noisy D2 state; direct path needs an overflow guard; edge cases and public-call performance remain to be validated.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "shape",
                    "argument_range",
                    "max_abs_error_vs_original",
                    "median_milliseconds",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
