"""Test note: do the blocks agree across grad_flags subsets?

For both entropies, every one of the 16 subsets of grad_flags, with and
without the Jacobian, against the call with all flags set: H and each
returned block compared bit for bit (entmc with the same seed), including
a production-sized entmc call whose canonical blocks split samples.
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from wave7_O2_common import banner

from pyvbmc.entropy import entlb_vbmc, entmc_vbmc
from pyvbmc.variational_posterior import VariationalPosterior

banner()


def make(D, K, seed):
    rs = np.random.default_rng(seed)
    vp = VariationalPosterior(D, K, rng=0)
    vp.mu = rs.normal(size=(D, K))
    vp.sigma = rs.uniform(0.4, 1.2, (1, K))
    vp.lambd = rs.uniform(0.5, 1.5, (D, 1))
    eta = rs.normal(size=(1, K))
    vp.eta = eta - eta.max()
    vp.w = np.exp(vp.eta) / np.exp(vp.eta).sum()
    return vp


def blocks(dH, flags, D, K):
    sizes = [D * K, K, D, K]
    out, i = [], 0
    for f, s in zip(flags, sizes):
        if f:
            out.append(dH[i : i + s])
            i += s
        else:
            out.append(None)
    return out


for name, fn, D, K, Ns in [
    ("entlb", lambda vp, fl, jac: entlb_vbmc(vp, fl, jac), 4, 5, None),
    (
        "entmc",
        lambda vp, fl, jac: entmc_vbmc(vp, 60, fl, jac, rng=9),
        4,
        5,
        60,
    ),
    (
        "entmc production size",
        lambda vp, fl, jac: entmc_vbmc(
            vp, int(np.ceil(100 * 30 ** (2 / 3))), fl, jac, rng=9
        ),
        8,
        30,
        None,
    ),
]:
    vp = make(D, K, 3)
    bad = 0
    for jac in (False, True):
        H0, dH0 = fn(vp, (True,) * 4, jac)
        full = blocks(dH0, (True,) * 4, D, K)
        for fl in itertools.product((False, True), repeat=4):
            H, dH = fn(vp, fl, jac)
            if not np.array_equal(H, H0):
                bad += 1
            for b, f0 in zip(blocks(dH, fl, D, K), full):
                if b is not None and not np.array_equal(b, f0):
                    bad += 1
    print(
        f"{name}: 32 calls, mismatches (H or block, bitwise): {bad}",
        flush=True,
    )
