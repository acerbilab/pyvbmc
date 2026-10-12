"""W4-7: does the rewritten proposal density change any number?

`active_sample_proposal_pdf` gives weight zero to a point at which every
component of the proposal has density zero, where it raised, and computes
the log-sum-exp of the other points on the rows inside the support alone.
The function is reached by the first step of `active_importance_sampling`
for an acquisition such as IMIQR; VIQR samples from the variational
posterior and never calls it, so the `acq_AcqFcnVIQR` oracle says nothing
about it, and `acq_AcqFcnIMIQR` moved with the fix of W4-1.

This script runs that first step on the stored noisy state of the oracles
with the module of `f8735567`, read from git, and with the module of the
checkout, same seed, the MCMC step switched off so that W4-1 stays out of
it, and compares every returned array bit for bit.

Run with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1, from a
checkout whose history holds `f8735567`.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import copy
import importlib
import subprocess
import types
from pathlib import Path

import numpy as np

import pyvbmc
from pyvbmc.acquisition_functions import AcqFcnIMIQR
from pyvbmc.testing.oracles import _state

print("pyvbmc:", pyvbmc.__file__, flush=True)

repo = Path(pyvbmc.__file__).parent.parent
old_source = subprocess.run(
    ["git", "show", "f8735567:pyvbmc/vbmc/active_importance_sampling.py"],
    capture_output=True,
    text=True,
    encoding="utf-8",
    cwd=repo,
    check=True,
).stdout
new = importlib.import_module("pyvbmc.vbmc.active_importance_sampling")
old = types.ModuleType("ais_f8735567")
exec(compile(old_source, "ais_f8735567.py", "exec"), old.__dict__)
assert "in_support" in Path(new.__file__).read_text(encoding="utf-8")
assert "in_support" not in old_source

fixtures = Path(pyvbmc.__file__).parent / "testing" / "oracles" / "fixtures"
state = _state.build_state(
    _state.load_snapshot(fixtures / "rosenbrock_D2_noise1_viqr")
)
vp, gp = state["vp"], state["gp"]
options = copy.deepcopy(state["options"])
options.__setitem__("active_importance_sampling_mcmc_samples", 0, force=True)

all_same = True
for seed in range(5):
    out = {}
    for label, module in (("f8735567", old), ("checkout", new)):
        vp.rng = np.random.default_rng(seed)
        out[label] = module.active_importance_sampling(
            vp, gp, AcqFcnIMIQR(), options
        )
    same = {
        key: bool(
            np.array_equal(
                out["f8735567"][key], out["checkout"][key], equal_nan=True
            )
        )
        for key in out["f8735567"]
    }
    all_same = all_same and all(same.values())
    print(f"seed {seed}: bit-identical {same}", flush=True)
print("every array bit-identical over the five seeds:", all_same)
