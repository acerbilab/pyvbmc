"""First question: entlb_vbmc on the MATLAB fixture (D = 4, K = 3, with the
Jacobian): the value and every gradient entry against MATLAB's."""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from wave7_O2_common import banner

import pyvbmc
from pyvbmc.entropy import entlb_vbmc
from pyvbmc.variational_posterior import VariationalPosterior

banner()
path = (
    Path(pyvbmc.__file__).parent / "testing" / "entropy" / "entropy-test.npz"
)
with np.load(path, allow_pickle=False) as fx:
    D, K = fx["D"].item(), fx["K"].item()
    vp = VariationalPosterior(D, K, rng=0)
    vp.w = fx["vp_w"].astype(float)
    vp.mu = fx["vp_mu"].astype(float)
    vp.sigma = fx["vp_sigma"].astype(float)
    vp.lambd = fx["vp_lambd"].astype(float)
    vp.eta = fx["vp_eta"].astype(float)
    Hl, dHl, jac = (
        fx["Hl"].item(),
        fx["dHl"].squeeze(),
        fx["jacobian_flag"].item(),
    )
H, dH = entlb_vbmc(vp, jacobian_flag=jac)
print(
    f"jacobian_flag={jac}: |H - Hl| = {abs(H - Hl):.1e}; max |dH - dHl| = "
    f"{np.max(np.abs(dH - dHl)):.1e}; max |dHl| = {np.max(np.abs(dHl)):.2f}",
    flush=True,
)
