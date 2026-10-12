"""Shared helpers for the wave-4 slice-P4 verification scripts.

Loads a stored oracle state (``pyvbmc/testing/oracles/fixtures``) and
prints the imported checkouts, so that every script's log records which
package it ran against.
"""

import os
import sys
from pathlib import Path

REPO = Path(str(__import__("pathlib").Path(__file__).resolve().parents[5]))
sys.path.insert(0, str(REPO))

import gpyreg as gpr  # noqa: E402
import numpy as np  # noqa: E402

import pyvbmc  # noqa: E402


def banner():
    print("pyvbmc.__file__ =", pyvbmc.__file__)
    print("gpyreg.__file__ =", gpr.__file__)
    print(
        "threads:",
        {
            k: os.environ.get(k)
            for k in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
    )
    print()


def load(name, seed=0, fun=None):
    from pyvbmc.testing.oracles._state import build_state, load_snapshot

    path = REPO / "pyvbmc" / "testing" / "oracles" / "fixtures" / name
    snap = load_snapshot(path)
    return build_state(snap, fun=fun, rng=np.random.default_rng(seed))
