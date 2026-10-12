"""O1 F2: the default of compute_var in _neg_elcbo; O1 F3: the docstrings.
F2: Python's default computes the variance only for beta != 0. The MATLAB
default (negelcbo_vbmc.m:16) also computes it when varF is requested
(nargout > 4), but with its own default compute_grad = nargout > 1 such a
call reaches gplogjoint.m:25-29, which raises for a variance with gradients;
only an explicit compute_grad = 0 lets MATLAB compute it. This script shows
the Python side of each of the three calls."""
import inspect
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from wave7_O1_common import banner, build_gp

from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.vbmc import variational_optimization as vo

banner()
D, K = 2, 2
gp = build_gp(D, N=20, Ns=3, seed=1)
vp = VariationalPosterior(D, K, rng=0)
vp.mu = np.array([[0.1, 0.8], [0.0, -0.4]])
vp.sigma = np.array([[0.5, 0.3]])
theta = vp.get_parameters()
out = vo._neg_elcbo(theta, gp, VariationalPosterior(D, K))
print(
    "(i)  _neg_elcbo(theta, gp, vp): len(out) =",
    len(out),
    " varF =",
    out[4],
    " dF is None:",
    out[1] is None,
)
out = vo._neg_elcbo(theta, gp, VariationalPosterior(D, K), compute_grad=False)
print("(ii) compute_grad=False:          varF =", out[4])
out = vo._neg_elcbo(
    theta, gp, VariationalPosterior(D, K), compute_grad=False, compute_var=True
)
print("(iii) compute_grad=False, compute_var=True: varF =", out[4])
try:
    vo._neg_elcbo(theta, gp, VariationalPosterior(D, K), compute_var=True)
except NotImplementedError as e:
    print(
        "(iv) compute_var=True with the default compute_grad=True raises:",
        repr(e)[:90],
    )
print("\nF3: signature:", inspect.signature(vo._neg_elcbo))
doc = vo._neg_elcbo.__doc__
for key in ("Ns : int", "entropy_alpha", "compute_var :"):
    i = doc.find(key)
    print("  _neg_elcbo doc:", " ".join(doc[i : i + 95].split()))
d2 = vo._initialize_full_elcbo.__doc__
for key in ("D : int", "Ns : int"):
    i = d2.find(key)
    print("  _initialize_full_elcbo doc:", " ".join(d2[i : i + 70].split()))
src = inspect.getsource(vo.optimize_vp)
i = src.find("elbo_stats = _initialize_full_elcbo")
print(
    "  call site:",
    src[src.find("theta_N =") : src.find("theta_N =") + 60].strip(),
    "|",
    src[src.find("Ns = np.size") : src.find("Ns = np.size") + 30].strip(),
    "|",
    src[i : i + 70].strip(),
)
