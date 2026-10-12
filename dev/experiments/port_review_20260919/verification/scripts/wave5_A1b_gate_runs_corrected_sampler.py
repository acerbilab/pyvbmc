"""W5, what the corrected balanced draw moves in a run.

Records the four short seeded runs of `wave2_fixpass_gate_runs.py` with
`VariationalPosterior.sample` replaced, in this process alone, by the same
method with `np.sum` in the remainder weights of the balanced draw
(`variational_posterior.py:643`), and compares the record with one made by
the package as it is. The corrected line leaves the number of random draws
alone, so a record that differs shows a candidate, or a Monte Carlo moment,
that the corrected weights changed and a decision that followed from it.

Usage:
    python -u wave5_A1b_gate_runs_corrected_sampler.py --out corrected.npz \
        --baseline <record of the package as it is>

Run with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1.
"""

import argparse
import importlib.util
import inspect
import logging
import textwrap
from pathlib import Path

import numpy as np

import pyvbmc
import pyvbmc.variational_posterior.variational_posterior as vp_module
from pyvbmc.variational_posterior import VariationalPosterior

print("pyvbmc:", pyvbmc.__file__, flush=True)

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--baseline", required=True)
args = parser.parse_args()

source = textwrap.dedent(inspect.getsource(VariationalPosterior.sample))
old = "repeats_extra - sum(w_extra)"
assert source.count(old) == 1, "the line is not as the finding describes it"
namespace = dict(vars(vp_module))
exec(source.replace(old, "repeats_extra - np.sum(w_extra)"), namespace)
VariationalPosterior.sample = namespace["sample"]
print(
    "VariationalPosterior.sample replaced by the corrected method", flush=True
)

gate_path = Path(__file__).with_name("wave2_fixpass_gate_runs.py")
spec = importlib.util.spec_from_file_location("gate_runs", gate_path)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

logging.disable(logging.CRITICAL)  # as the gate script's own `main` does
arrays = {}
for name, run in gate.build_runs().items():
    arrays.update(gate.record(name, run))
np.savez(args.out, **arrays)
print(f"saved {args.out}", flush=True)
gate.compare(args.baseline, args.out)
