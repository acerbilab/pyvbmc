"""Finding 9 (P1b comparison F5): `AcqFcn` against `AcqFcnLog`.

Evaluates both acquisition forms on the stored oracle candidate sets
(read-only) and reports: the argmin of each, whether the two rank the
candidates identically (Spearman-style comparison of the orderings
restricted to the finite, non-saturated entries), how many candidates the
plain form returns as exactly 0 (the underflow of
`var_tot * exp(f_bar - z) * p`), and how many as +inf (the hard-bound
penalty, which both share).
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
import pyvbmc
from pyvbmc import acquisition_functions as acqs
from pyvbmc.testing.oracles import _oracles, _state

print("pyvbmc:", pyvbmc.__file__)

fixtures = REPO / "pyvbmc" / "testing" / "oracles" / "fixtures"
names = _state.snapshot_names(fixtures)
print("fixtures:", names)

for name in names:
    snap = _state.load_snapshot(fixtures / name)
    out = {}
    for cls in ("AcqFcn", "AcqFcnLog"):
        state = _state.build_state(snap, rng=np.random.default_rng(0))
        gp, optim_state = state["gp"], state["optim_state"]
        _oracles.prepare_gp_for_acq(gp, state["logger"], optim_state)
        acq = getattr(acqs, cls)()
        out[cls] = acq(
            np.array(state["cand"]["Xs"]),
            gp,
            state["vp"],
            state["logger"],
            optim_state,
        )
    a, b = out["AcqFcn"], out["AcqFcnLog"]
    finite = np.isfinite(a) & np.isfinite(b)
    zeros = int(np.sum(a == 0.0))
    infs = int(np.sum(np.isinf(a)))
    # Ranking agreement on the entries where the plain form has not
    # saturated at 0 (its worst value).
    live = finite & (a != 0.0)
    order_a = np.argsort(a[live], kind="stable")
    order_b = np.argsort(b[live], kind="stable")
    same_order = np.array_equal(order_a, order_b)
    print(
        f"\n{name}: n={a.size}"
        f"\n  AcqFcn    argmin={int(np.argmin(a))} min={a.min():.6e}"
        f"  exact zeros={zeros}  +inf={infs}"
        f"\n  AcqFcnLog argmin={int(np.argmin(b))} min={b.min():.6e}"
        f"  +inf={int(np.sum(np.isinf(b)))}"
        f"\n  identical order on the {int(live.sum())} non-saturated finite entries: {same_order}"
    )
    if zeros:
        print(
            f"  -> the plain form ties {zeros} candidates at its worst value 0;"
            f" the log form spreads them over "
            f"[{b[a == 0.0].min():.3e}, {b[a == 0.0].max():.3e}]"
        )
