"""P4-3: the retained ``mcmc_importance_sampling`` branch.

Settles (i) whether the branch at
``active_importance_sampling.py:86-126`` runs at all when a custom
acquisition requests it, by forcing the flag on a shipped VIQR instance
and lowering the fESS gate; (ii) what a ``SliceSampler`` built with a
1-D start would return for the branch's thinning
(``burn_in = 0``, ``thin = 1``, then ``Xa[-Na:]``); and (iii) what the
branch stores as the weights.
"""

import copy

import numpy as np
from wave4_P4_common import banner, load

banner()

from pyvbmc.acquisition_functions import AcqFcnIMIQR, AcqFcnVIQR  # noqa: E402
from pyvbmc.vbmc.active_importance_sampling import (  # noqa: E402
    active_importance_sampling,
    get_mcmc_opts,
)

st = load("normal_D2_singlesample", seed=5)
vp, gp, opts = st["vp"], st["gp"], st["options"]

for name, acq in (("VIQR", AcqFcnVIQR()), ("IMIQR", AcqFcnIMIQR())):
    a = copy.deepcopy(acq)
    a.acq_info["mcmc_importance_sampling"] = True
    a.acq_info["variational_importance_sampling"] = True  # step-0 branch
    o = copy.deepcopy(opts)
    o.__setitem__("active_importance_sampling_fess_thresh", 1.0, force=True)
    v = copy.deepcopy(vp)
    v.rng = np.random.default_rng(5)
    try:
        out = active_importance_sampling(v, gp, a, o)
        print(
            f"{name}: branch completed, ln_weights shape",
            out["ln_weights"].shape,
        )
    except Exception as exc:
        print(f"{name}: raised {type(exc).__name__}: {exc}")

print()
print("fess_thresh default =", opts["active_importance_sampling_fess_thresh"])
print("mcmc_thin default   =", opts["active_importance_sampling_mcmc_thin"])
print(
    "branch's sampler opts (Nmcmc = Na*thin):",
    get_mcmc_opts(200)[0],
    "thin/burn_in passed by the branch: 1 / 0",
)

# (ii) the hand-made thinning keeps the LAST Na consecutive draws.
print()
print("Branch's own thinning: thin=1, burn_in=0, then Xa[-Na:, :].")
print("With Na=200 and thin=2 the sampler draws 400 states with no burn-in;")
print("Xa[-200:] are states 201..400 of one chain, consecutive.")

# (iii) the weights the branch stores
Na, Ns = 4, 1
f_s2 = np.full((Na, Ns), 0.25)
f_mu = np.full((Na, Ns), -2.0)
Xa = np.zeros((Na, 2))
print()
for name, a in (("VIQR", AcqFcnVIQR()), ("IMIQR", AcqFcnIMIQR())):
    print(
        name,
        "is_log_base (the branch's ln_y) =",
        a.is_log_base(Xa, f_mu=f_mu, f_s2=f_s2).ravel(),
    )
print("branch line :125 stores ln_y.T only; no '- log_p' term.")
