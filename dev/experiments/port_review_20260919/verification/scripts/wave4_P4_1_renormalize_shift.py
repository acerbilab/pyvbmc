"""P4-1: does ``renormalize_weights`` change any decision?

Settles (a) whether the single log-sum-exp that
``active_importance_sampling.py:329`` subtracts from the whole
``(Ns_gp, Na)`` array of log weights is a constant for every consumer, so
that no candidate ranking, no local-search decision and no tolerance
changes; and (b) what each row of the MCMC branch's weights estimates,
by printing the per-row totals with and without the shift.

Consumers exercised: ``AcqFcnIMIQR`` (the only reader of ``ln_weights``
in the value of the acquisition), ``AcqFcnVIQR(loss="iqr_reduction")``
(the only VIQR reader), the variance regularization of
``AbstractAcqFcn.__call__`` and the ``f_val_optim < f_val_old``
comparison of the local search.
"""

import copy

import numpy as np
from wave4_P4_common import banner, load

banner()

from pyvbmc.acquisition_functions import AcqFcnIMIQR, AcqFcnVIQR  # noqa: E402
from pyvbmc.vbmc.active_importance_sampling import (  # noqa: E402
    active_importance_sampling,
    renormalize_weights,
)

st = load("rosenbrock_D2_noise1_viqr", seed=7)
vp, gp, opts, ostate = st["vp"], st["gp"], st["options"], st["optim_state"]
logger = st["logger"]
print("Ns_gp =", len(gp.posteriors), " N_train =", gp.X.shape[0], " D =", gp.D)
print(
    "uncertainty_handling_level =",
    ostate["uncertainty_handling_level"],
    " gp_noise_fun =",
    ostate["gp_noise_fun"],
)
print(
    "mcmc_samples option =",
    opts["active_importance_sampling_mcmc_samples"],
    " vp_samples =",
    opts["active_importance_sampling_vp_samples"],
    " box_samples =",
    opts["active_importance_sampling_box_samples"],
)
print()

# ---------------------------------------------------------------- IMIQR path
acq = AcqFcnIMIQR()
vp.rng = np.random.default_rng(11)
active_is = active_importance_sampling(vp, gp, acq, opts)
W = active_is["ln_weights"]
print("MCMC branch: ln_weights shape", W.shape)
print("  whole-array sum of exp(ln_w) =", np.exp(W).sum())
print("  per-row sums of exp(ln_w)    =", np.exp(W).sum(axis=1))
print(
    "  ratio max/min of the row sums =",
    np.exp(W).sum(axis=1).max() / np.exp(W).sum(axis=1).min(),
)
# Undo the shift to get MATLAB's raw lnw = lny - logp.
# renormalize_weights subtracts M + log sum exp(lnw - M); recover it.
shift = None
raw = None
# Recompute the raw weights by re-running with the renormalization removed:
# replicate `renormalize_weights` inverse via the known identity.
# (log sum exp of the returned array is 0 by construction.)
print("  logsumexp of the returned array =", np.log(np.exp(W).sum()))
print()

# Now: is the acquisition shifted by exactly the constant?
cand = np.asarray(st["cand"]["Xs"], dtype=np.float64)
if cand.ndim == 1:
    cand = cand.reshape(-1, gp.D)
print("candidate set:", cand.shape)

# Need the acquisition's own precomputations (gp.temporary_data etc.)
from pyvbmc.testing.oracles._oracles import prepare_gp_for_acq  # noqa: E402

prepare_gp_for_acq(gp, logger, ostate)
ostate["active_importance_sampling"] = active_is

c = 3.7  # arbitrary constant shift of the log weights
base = acq(cand, gp, vp, logger, ostate).copy()

is_shift = copy.deepcopy(active_is)
is_shift["ln_weights"] = active_is["ln_weights"] + c
ostate["active_importance_sampling"] = is_shift
shifted = acq(cand, gp, vp, logger, ostate).copy()

d = shifted - base
fin = np.isfinite(d)
print("IMIQR: acq(shifted) - acq(base):")
print("  finite entries:", fin.sum(), "of", d.size)
print("  min, max =", d[fin].min(), d[fin].max())
print("  max |d - c| =", np.abs(d[fin] - c).max())
print("  argmin unchanged:", np.argmin(base) == np.argmin(shifted))
print(
    "  ranking unchanged:",
    np.array_equal(
        np.argsort(base, kind="stable"), np.argsort(shifted, kind="stable")
    ),
)
print(
    "  variance_regularized_acq_fcn =",
    ostate.get(
        "variance_regularized_acq_fcn",
        ostate.get("variance_regularized_acqfcn"),
    ),
    " log_flag =",
    acq.acq_info.get("log_flag"),
)
print()

# --------------------------------------------- VIQR with the reduction loss
viqr = AcqFcnVIQR(loss="iqr_reduction")
vp2 = copy.deepcopy(vp)
vp2.rng = np.random.default_rng(11)
active_is_v = active_importance_sampling(vp2, gp, viqr, opts)
print("VIQR (only_vp path): ln_weights shape", active_is_v["ln_weights"].shape)
print(
    "  unique values of ln_weights:",
    np.unique(active_is_v["ln_weights"]),
    " (-log(Ns_gp*Na) =",
    -np.log(active_is_v["ln_weights"].size),
    ")",
)
ostate["active_importance_sampling"] = active_is_v
base_v = viqr(cand, gp, vp, logger, ostate).copy()
is_v_shift = copy.deepcopy(active_is_v)
is_v_shift["ln_weights"] = active_is_v["ln_weights"] + c
ostate["active_importance_sampling"] = is_v_shift
shift_v = viqr(cand, gp, vp, logger, ostate).copy()
dv = shift_v - base_v
finv = np.isfinite(dv)
print("  acq(shifted) - acq(base): min,max =", dv[finv].min(), dv[finv].max())
print("  max |d + c| =", np.abs(dv[finv] + c).max())
print(
    "  ranking unchanged:",
    np.array_equal(
        np.argsort(base_v, kind="stable"), np.argsort(shift_v, kind="stable")
    ),
)
print()

# ------------------------------------------- the ISR branch (no MCMC step)
opts2 = copy.deepcopy(opts)
opts2.__setitem__("active_importance_sampling_mcmc_samples", 0, force=True)
vp3 = copy.deepcopy(vp)
vp3.rng = np.random.default_rng(11)
is_isr = active_importance_sampling(vp3, gp, AcqFcnIMIQR(), opts2)
print(
    "ISR branch (mcmc_samples=0): ln_weights shape", is_isr["ln_weights"].shape
)
print("  per-row sums of exp(ln_w) =", np.exp(is_isr["ln_weights"]).sum(1))
print("  whole-array sum =", np.exp(is_isr["ln_weights"]).sum())

# ---------------------------------------- the size of the constant shift
# Record renormalize_weights' input without touching the repository:
# rebind the module attribute for the duration of one call.
import importlib  # noqa: E402

# The subpackage exports the function under the module's own name, so the
# module object has to be fetched explicitly (AGENTS.md records the trap).
ais = importlib.import_module("pyvbmc.vbmc.active_importance_sampling")

captured = {}
_orig = ais.renormalize_weights


def _spy(ln_w):
    M = np.amax(ln_w)
    C = M + np.log(np.sum(np.exp(ln_w - M)))
    captured["C"] = C
    return ln_w - C


print()
ais.renormalize_weights = _spy
try:
    for name, a in (("IMIQR", AcqFcnIMIQR()), ("VIQR", AcqFcnVIQR())):
        v = copy.deepcopy(vp)
        v.rng = np.random.default_rng(11)
        active_importance_sampling(v, gp, a, opts)
        print(
            f"  {name}: the subtracted constant is {captured['C']:.4f} "
            f"nats; MATLAB's weights (and, for IMIQR, its acquisition "
            f"value) are larger by that amount"
        )
finally:
    ais.renormalize_weights = _orig
