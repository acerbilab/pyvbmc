"""Shared helpers for the wave-4 P3 verification scripts.

Loads a stored oracle snapshot and rebuilds it through the oracle
helpers, and prints the imported checkouts once per process.
"""
import os

import numpy as np

FIXTURES = os.path.join(
    str(__import__("pathlib").Path(__file__).resolve().parents[5]),
    "pyvbmc",
    "testing",
    "oracles",
    "fixtures",
)


def banner():
    import gpyreg

    import pyvbmc

    print("pyvbmc.__file__ =", pyvbmc.__file__)
    print("gpyreg.__file__ =", gpyreg.__file__)


def load(name, seed=20260904):
    from pyvbmc.testing.oracles._state import build_state, load_snapshot

    snap = load_snapshot(os.path.join(FIXTURES, name))
    return build_state(snap, rng=np.random.default_rng(seed))
