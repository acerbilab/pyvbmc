"""P4-2: ``fess(vp, gp, N)`` with an integer third argument.

Settles whether the documented scalar form of
``active_importance_sampling.fess`` (``X : np.ndarray(N, D) or int``,
default ``X=100``) runs, and what it raises if not.  Also checks the
array form against a hand computation of the fractional effective sample
size, so that the failure is localized to the scalar branch.
"""

import numpy as np
from wave4_P4_common import banner, load

banner()

from pyvbmc.vbmc.active_importance_sampling import fess  # noqa: E402

st = load("normal_D2_singlesample", seed=3)
vp, gp = st["vp"], st["gp"]
vp.rng = np.random.default_rng(3)

# scalar / default form
for arg in ("default", 100, 25):
    try:
        if arg == "default":
            out = fess(vp, gp)
        else:
            out = fess(vp, gp, arg)
        print(f"fess(vp, gp, {arg}) -> {out}")
    except Exception as exc:
        print(f"fess(vp, gp, {arg}) raised {type(exc).__name__}: {exc}")

# array form, plus a hand computation
X, _ = vp.sample(40, orig_flag=False)
got = fess(vp, gp, X)
f_bar, _ = gp.predict(X)
f_bar = f_bar.ravel()
lp = vp.pdf(X, orig_flag=False, log_flag=True).ravel()
lw = f_bar - lp
w = np.exp(lw - lw.max())
w = w / w.sum()
hand = (1 / np.sum(w**2)) / X.shape[0]
print("fess(vp, gp, X) =", got, " hand =", hand, " diff =", abs(got - hand))

# the only live call site in the package passes arrays
print()
print("live call site: active_importance_sampling.py:88  fess(vp, f_mu, Xa)")
print("stubbed MATLAB gate: active_sample.py:736-737")
