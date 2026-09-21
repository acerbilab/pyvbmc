"""W4-1: the resampling weights of the MCMC step of the importance sampling.

`active_importance_sampling` (the branch IMIQR reaches) picks the starting
point of each slice-sampling chain by importance sampling-resampling:
`rng.choice(a=len(weights), p=weights)`. This script runs the function as it
is on the stored noisy state of the oracles, records the `p` handed to
`choice` for every GP hyperparameter sample, and sets beside it the weights
that `private/activeimportancesampling_vbmc.m:206-208` defines from the same
inputs: `w = exp(lnw - max(lnw,[],2))` with `lnw` a row, the maximum taken
over the importance samples.

It settles whether the weights PyVBMC draws from are uniform, and how far
from uniform MATLAB's are on a realistic state.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import copy
from pathlib import Path

import gpyreg
import numpy as np

import pyvbmc
from pyvbmc.acquisition_functions import AcqFcnIMIQR
from pyvbmc.testing.oracles import _state
from pyvbmc.vbmc.active_importance_sampling import active_importance_sampling

print("pyvbmc:", pyvbmc.__file__, flush=True)
print("gpyreg:", gpyreg.__file__, flush=True)


class RecordingGenerator(np.random.Generator):
    """A Generator that keeps the `p` of every `choice` call."""

    def __init__(self, bit_generator):
        super().__init__(bit_generator)
        self.recorded_p = []

    def choice(self, *args, **kwargs):
        if kwargs.get("p") is not None and kwargs.get("replace") is False:
            self.recorded_p.append(np.array(kwargs["p"], copy=True))
        return super().choice(*args, **kwargs)


fixtures = Path(pyvbmc.__file__).parent / "testing" / "oracles" / "fixtures"
snap = _state.load_snapshot(fixtures / "rosenbrock_D2_noise1_viqr")
state = _state.build_state(snap)
vp, gp, options = state["vp"], state["gp"], state["options"]
print(
    f"state: D={gp.D}, N={gp.X.shape[0]}, Ns_gp={len(gp.posteriors)}, "
    f"K={vp.K}; mcmc_samples="
    f"{options['active_importance_sampling_mcmc_samples']}, "
    f"vp_samples={options['active_importance_sampling_vp_samples']}, "
    f"box_samples={options['active_importance_sampling_box_samples']}",
    flush=True,
)

acq = AcqFcnIMIQR()
rng = RecordingGenerator(np.random.PCG64(20260920))
vp._rng = rng  # the setter would rebuild a plain Generator from it
assert vp.rng is rng
active_importance_sampling(vp, gp, acq, options)

print(f"\n`choice` was called with p= {len(rng.recorded_p)} time(s)")
for s, p in enumerate(rng.recorded_p):
    print(
        f"  hyperparameter sample {s}: len={p.size}, min={p.min():.6g}, "
        f"max={p.max():.6g}, all equal 1/len: "
        f"{bool(np.all(p == 1.0 / p.size))}"
    )

# MATLAB's weights from the same inputs: the step-1 state (the function
# with the MCMC step switched off, same seed, so the same proposal draws).
options_no_mcmc = copy.deepcopy(options)
options_no_mcmc.__setitem__(
    "active_importance_sampling_mcmc_samples", 0, force=True
)
vp._rng = np.random.Generator(np.random.PCG64(20260920))
old = active_importance_sampling(vp, gp, acq, options_no_mcmc)
print("\nMATLAB's rule on the step-1 state (max over the samples):")
gp1 = copy.deepcopy(gp)
for s in range(len(gp.posteriors)):
    gp1.posteriors = np.array([gp.posteriors[s]])
    f_mu, f_s2 = gp1.predict(old["X"], separate_samples=True)
    lnw = old["ln_weights"][s, :] + acq.is_log_added(
        f_mu=f_mu, f_s2=f_s2
    ).reshape(-1)
    shape_in_python = (
        old["ln_weights"][s, :].reshape(-1, 1)
        + acq.is_log_added(f_mu=f_mu, f_s2=f_s2)
    ).shape
    w = np.exp(lnw - np.max(lnw))
    w = w / w.sum()
    ess = 1.0 / np.sum(w**2)
    print(
        f"  sample {s}: PyVBMC's ln_weights has shape {shape_in_python}; "
        f"MATLAB's w: max={w.max():.4f}, ESS={ess:.1f} of {w.size}, "
        f"weights below 1e-6: {int(np.sum(w < 1e-6))}, "
        f"-inf log weights: {int(np.sum(~np.isfinite(lnw)))}"
    )
