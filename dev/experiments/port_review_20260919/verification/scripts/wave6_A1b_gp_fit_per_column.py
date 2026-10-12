"""W6-A1b, what gplite's per-column bound recommendations move in one GP fit.

Runs the `gp_fit` oracle, one `train_gp` call from a stored state under a
seeded generator, on every stored oracle state twice: with gpyreg as it is,
which on the machine that generated the fixtures reproduces the stored
reference bit for bit and so shows that the harness runs the recorded fit,
and with the statistics of the bound recommendations taken per column
(`wave6_per_column_patch.py`). The same seed drives both, so a difference is
the recommendations' doing.

Run from anywhere with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1.
"""

import sys
import time
from pathlib import Path

import gpyreg
import numpy as np

import pyvbmc
from pyvbmc.testing.oracles._oracles import ORACLES
from pyvbmc.testing.oracles._state import (
    build_state,
    load_snapshot,
    snapshot_names,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wave6_per_column_patch as patch  # noqa: E402

print("pyvbmc:", pyvbmc.__file__)
print("gpyreg:", gpyreg.__file__, flush=True)
np.set_printoptions(precision=3, suppress=True, linewidth=120)

fixtures = (
    Path(pyvbmc.__file__).resolve().parent / "testing" / "oracles" / "fixtures"
)
oracle = ORACLES["gp_fit"]

for name in snapshot_names(fixtures):
    snap = load_snapshot(fixtures / name)
    state = build_state(snap)
    if not oracle.applies(state):
        print(f"\n=== {name}: the oracle does not apply", flush=True)
        continue
    D = state["gp"].D
    cov_N = state["gp"].covariance.hyperparameter_count(D)
    noise_N = state["gp"].noise.hyperparameter_count()
    blocks = {
        "log length scale": slice(0, D),
        "log output scale": slice(D, cov_N),
        "noise": slice(cov_N, cov_N + noise_N),
        "mean constant": slice(cov_N + noise_N, cov_N + noise_N + 1),
        "mean location": slice(cov_N + noise_N + 1, cov_N + noise_N + 1 + D),
        "log mean scale": slice(cov_N + noise_N + 1 + D, None),
    }

    patch.restore()
    t0 = time.time()
    pooled = oracle(build_state(snap))
    t_pooled = time.time() - t0
    patch.install()
    per_column = oracle(build_state(snap))
    patch.restore()

    ref = state["ref"].get("gp_fit")
    print(
        f"\n=== {name}: D={D}, samples {pooled['hyp'].shape[0]},"
        f" {t_pooled:.1f} s a fit"
    )
    if ref is not None:
        same = all(
            np.array_equal(np.asarray(ref[k]), pooled[k]) for k in pooled
        )
        print(
            "gpyreg as it is reproduces the stored reference bit for bit:",
            same,
        )
    if pooled["hyp"].shape != per_column["hyp"].shape:
        print(
            "number of samples differs:",
            pooled["hyp"].shape,
            per_column["hyp"].shape,
        )
        continue
    print(
        "identical with per-column recommendations:",
        np.array_equal(pooled["hyp"], per_column["hyp"]),
    )
    for label, block in blocks.items():
        p, c = pooled["hyp"][:, block], per_column["hyp"][:, block]
        if p.size == 0:
            continue
        print(
            f"  {label:>17}: mean over samples pooled {p.mean(axis=0)},"
            f" per column {c.mean(axis=0)};"
            f" SD over samples pooled {p.std(axis=0)},"
            f" per column {c.std(axis=0)}"
        )
    print(
        "  sn2_hpd pooled / per column:",
        pooled["sn2_hpd"],
        per_column["sn2_hpd"],
        flush=True,
    )
