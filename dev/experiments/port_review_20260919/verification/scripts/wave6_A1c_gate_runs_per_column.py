"""W6-A1c, what gplite's per-column bound recommendations move in a run.

Records the four short seeded runs of `wave2_fixpass_gate_runs.py` with the
statistics of gpyreg's bound recommendations taken per column, in this
process alone (`wave6_per_column_patch.py`), and compares the record with one
made by the packages as they are. The runs are a fingerprint of the
trajectory and no measure of accuracy.

Usage:
    python -u wave6_A1c_gate_runs_per_column.py --out per_column.npz \
        --baseline <record of the packages as they are>

Run with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1.
"""

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

import gpyreg
import numpy as np

import pyvbmc

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wave6_per_column_patch as patch  # noqa: E402

print("pyvbmc:", pyvbmc.__file__)
print("gpyreg:", gpyreg.__file__, flush=True)

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--baseline", required=True)
args = parser.parse_args()

patch.install()
print("per-column bound recommendations installed", flush=True)

gate_path = Path(__file__).with_name("wave2_fixpass_gate_runs.py")
spec = importlib.util.spec_from_file_location("gate_runs", gate_path)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

logging.disable(logging.CRITICAL)  # as the gate script's own `main` does
arrays = {}
for name, run in gate.build_runs().items():
    arrays.update(gate.record(name, run))
    print(f"recorded {name}", flush=True)
np.savez(args.out, **arrays)
print(f"saved {args.out}", flush=True)
gate.compare(args.baseline, args.out)
