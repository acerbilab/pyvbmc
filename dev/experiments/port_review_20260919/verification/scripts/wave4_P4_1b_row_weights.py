"""P4-1b: what each row of the MCMC branch's importance weights estimates.

In the MCMC branch each row ``s`` holds ``lnw = ln_y - log_p`` for points
drawn by a chain whose target is the *unnormalized* density
``q~_s = exp(f_s) * 2 sinh(u s_s)``.  So

    sum_a exp(lnw[s, a])  ~  N * I_s / Z_s ,
    I_s = int exp(f_s),    Z_s = int exp(f_s) 2 sinh(u s_s) ,

with ``Z_s`` the *current* integrated interquantile range of
hyperparameter sample ``s`` -- the very quantity the acquisition
measures after adding a point.  This script estimates ``I_s`` and
``Z_s`` from the ISR samples of the same call (which are drawn from a
known normalized proposal) and checks that the ratio of two rows'
weight totals matches the ratio of ``I_s / Z_s``.  Both PyVBMC and
MATLAB form the rows this way; PyVBMC's extra global log-sum-exp is one
scalar and cancels from every ratio.
"""

import copy

import numpy as np
from wave4_P4_common import banner, load

banner()

from pyvbmc.acquisition_functions import AcqFcnIMIQR  # noqa: E402
from pyvbmc.vbmc import active_importance_sampling as ais_mod  # noqa: E402
from pyvbmc.vbmc.active_importance_sampling import (  # noqa: E402
    active_importance_sampling,
)

st = load("rosenbrock_D2_noise1_viqr", seed=7)
vp, gp, opts = st["vp"], st["gp"], st["options"]
acq = AcqFcnIMIQR()

# The ISR branch alone, to estimate I_s and Z_s from a normalized proposal.
opts0 = copy.deepcopy(opts)
opts0.__setitem__("active_importance_sampling_mcmc_samples", 0, force=True)
vp0 = copy.deepcopy(vp)
vp0.rng = np.random.default_rng(101)
isr = active_importance_sampling(vp0, gp, acq, opts0)
# undo the global renormalization of the ISR weights: the shift is the
# array's own log-sum-exp, which the function drives to 0.
lnw_isr = isr["ln_weights"]  # shifted; ratios and per-row shapes intact
f_s2 = isr["f_s2"]  # (Na, Ns)
add = acq.is_log_added(f_s2=f_s2).T  # (Ns, Na)
Ns, Na = lnw_isr.shape

# Ihat_s propto mean_a exp(lnw_isr[s,a]); Zhat_s propto mean_a
# exp(lnw_isr[s,a] + add[s,a]).  The unknown common shift cancels in I/Z.
from scipy.special import logsumexp  # noqa: E402

ln_I = logsumexp(lnw_isr, axis=1) - np.log(Na)
ln_Z = logsumexp(lnw_isr + add, axis=1) - np.log(Na)
ln_ratio_pred = ln_I - ln_Z

# The MCMC branch on the same state.
vp1 = copy.deepcopy(vp)
vp1.rng = np.random.default_rng(101)
mcmc = active_importance_sampling(vp1, gp, acq, opts)
ln_row = logsumexp(mcmc["ln_weights"], axis=1)

print("per hyperparameter sample s:")
print("  s |  log Ihat  |  log Zhat  | log(I/Z) | log row total | difference")
for s in range(Ns):
    print(
        f"  {s} | {ln_I[s]:10.3f} | {ln_Z[s]:10.3f} | "
        f"{ln_ratio_pred[s]:8.3f} | {ln_row[s]:13.3f} | "
        f"{ln_row[s] - ln_ratio_pred[s]:9.3f}"
    )
d = ln_row - ln_ratio_pred
print()
print(
    "  spread of log(row total) across s:", float(ln_row.max() - ln_row.min())
)
print(
    "  spread of log(I/Z)       across s:",
    float(ln_ratio_pred.max() - ln_ratio_pred.min()),
)
print(
    "  spread of the difference (should be ~0 if the model is right):",
    float(d.max() - d.min()),
)
print()
print("A uniform average over hyperparameter samples of the per-sample")
print("integral would need every row total equal; both PyVBMC and MATLAB")
print("leave the factor I_s / Z_s in, and the global shift cancels here.")
print()
print("Outcome: the identity above is the expectation, not the realized")
print("value. With 100 correlated draws per row the totals scatter far")
print("beyond the estimated I_s / Z_s spread, so on this state most of the")
print("inequality between rows is chain noise. Either way the rows are not")
print("comparable, and one scalar shift neither causes nor removes that.")
