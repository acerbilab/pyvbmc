"""Test note: test_entmc_vbmc_matlab's tolerance on the mu block.

The test asserts np.allclose(dH, dHm, rtol=0.01, atol=0.01). For each of
the 12 mu entries, the admitted band is 0.01 + 0.01 |dHm_i|. Prints the
MATLAB mu entries, which of them a zero mu block would still pass, the
seed-to-seed SD of PyVBMC's estimate over 24 seeds at the fixture's Ns, and
z-scores of PyVBMC's mean against the MATLAB estimate (whose own MC error
is taken equal to one PyVBMC SD).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from wave7_O2_common import banner

import pyvbmc
from pyvbmc.entropy import entmc_vbmc
from pyvbmc.variational_posterior import VariationalPosterior

banner()
path = (
    Path(pyvbmc.__file__).parent / "testing" / "entropy" / "entropy-test.npz"
)
with np.load(path, allow_pickle=False) as fx:
    D, K, Ns = fx["D"].item(), fx["K"].item(), fx["Ns"].item()
    vp = VariationalPosterior(D, K, rng=0)
    vp.w = fx["vp_w"].astype(float)
    vp.mu = fx["vp_mu"].astype(float)
    vp.sigma = fx["vp_sigma"].astype(float)
    vp.lambd = fx["vp_lambd"].astype(float)
    vp.eta = fx["vp_eta"].astype(float)
    Hm, dHm = fx["H"].item(), fx["dH"].squeeze()
    jac = fx["jacobian_flag"].item()
print(f"D={D} K={K} Ns={Ns} jacobian_flag={jac}", flush=True)
mu_m = dHm[: D * K]
band = 0.01 + 0.01 * np.abs(mu_m)
print("MATLAB mu entries:", np.round(mu_m, 5), flush=True)
print("max |mu entry| =", np.max(np.abs(mu_m)), flush=True)
print(
    "entries a zero mu block would pass:",
    int(np.sum(np.abs(mu_m) <= band)),
    "of",
    D * K,
    flush=True,
)

Hs, dHs = [], []
for seed in range(24):
    H, dH = entmc_vbmc(vp, Ns, jacobian_flag=jac, rng=seed)
    Hs.append(H)
    dHs.append(dH)
Hs, dHs = np.array(Hs), np.array(dHs)
sd = dHs.std(axis=0, ddof=1)
print(
    "PyVBMC seed-to-seed SD of the mu entries: min %.1e max %.1e"
    % (sd[: D * K].min(), sd[: D * K].max()),
    flush=True,
)
z = (dHs.mean(0) - dHm) / (sd * np.sqrt(1 + 1 / len(Hs)))
zH = (Hs.mean() - Hm) / (Hs.std(ddof=1) * np.sqrt(1 + 1 / len(Hs)))
print(
    "z of the 22 gradient entries: max |z| = %.2f; z of H = %.2f"
    % (np.max(np.abs(z)), zH),
    flush=True,
)
print("z by block: mu %s" % np.round(z[: D * K], 2), flush=True)
print(
    "           sigma %s lambda %s w %s"
    % (
        np.round(z[D * K : D * K + K], 2),
        np.round(z[D * K + K : D * K + K + D], 2),
        np.round(z[D * K + K + D :], 2),
    ),
    flush=True,
)
