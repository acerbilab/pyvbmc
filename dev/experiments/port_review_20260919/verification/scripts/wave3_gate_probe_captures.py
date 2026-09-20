"""What the fit of each authentic GP-history capture does: the length of its
history, the size of its space-filling design and of its pool of past
samples, and where its best starting point comes from."""

import sys

import numpy as np

sys.path.insert(0, "dev/scripts")
import gpyreg as gpr  # noqa: E402
from make_oracle_fixtures import (  # noqa: E402
    GP_HISTORY_FIXTURES,
    load_snapshot,
)

import pyvbmc  # noqa: E402
from pyvbmc.testing.oracles import _gp_fit_history as H  # noqa: E402
from pyvbmc.vbmc import gaussian_process_train as G  # noqa: E402

print("pyvbmc:", pyvbmc.__file__, flush=True)

for name in ("early_sampled", "later_changing_ns", "noisy_nonuniform_weights"):
    snap = load_snapshot(GP_HISTORY_FIXTURES / name)
    args = H.build_inputs(snap["pre"])
    hyp_dict, optim_state, logger, history, options = args[:5]
    n = int(np.size(history["gp"]))
    seen = {}
    real_fit = gpr.GP.fit

    def spy(self, X, y, s2, hyp0=None, options=None, rng=None, **kw):
        seen["init_N"] = options["init_N"]
        seen["opts_N"] = options["opts_N"]
        seen["n_hyp0"] = None if hyp0 is None else int(np.shape(hyp0)[0])
        seen["LB"] = self.lower_bounds.copy()
        seen["UB"] = self.upper_bounds.copy()
        return real_fit(
            self, X, y, s2, hyp0=hyp0, options=options, rng=rng, **kw
        )

    gpr.GP.fit = spy
    try:
        G.train_gp(*args)
    finally:
        gpr.GP.fit = real_fit
    print(
        f"{name}: iter {optim_state['iter']}, recorded GPs n = {n} "
        f"({'even: window differs' if n % 2 == 0 and n else 'odd: window the same'}), "
        f"init_N = {seen['init_N']}, opts_N = {seen['opts_N']}, starting "
        f"vectors handed to the fit = {seen['n_hyp0']}, recompute_var_post = "
        f"{optim_state.get('recompute_var_post')}",
        flush=True,
    )
