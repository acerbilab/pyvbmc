"""Settles P8c F6: the inverted low-correlation comparison and NaN.

MATLAB (``misc/warp_input_vbmc.m:52-56``) keeps ``abs(vp_corr) > thresh``
and zeroes the complement, so an entry whose correlation is NaN is zeroed.
PyVBMC (``whitening.py:154-159``) zeroes ``abs(vp_corr) <= thresh``, so a
NaN entry survives into the regularization and the SVD.  This script shows
which covariance shapes make the two maskings differ, and what
``vp.moments`` can actually produce.
"""

import numpy as np

import pyvbmc
from pyvbmc.variational_posterior import VariationalPosterior

print("pyvbmc.__file__ =", pyvbmc.__file__)
print()

THRESH = 0.05


def both(cov):
    with np.errstate(all="ignore"):
        corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
        py = cov.copy()
        py[np.abs(corr) <= THRESH] = 0  # PyVBMC
        ml = cov.copy()
        ml[~(np.abs(corr) > THRESH)] = 0  # MATLAB
    return corr, py, ml


cases = {
    "ordinary PSD": np.array(
        [[1.0, 0.5, 0.01], [0.5, 1.0, 0.02], [0.01, 0.02, 1.0]]
    ),
    "zero diagonal entry": np.array(
        [[1.0, 0.5, 0.3], [0.5, 1.0, 0.02], [0.3, 0.02, 0.0]]
    ),
    "negative diagonal entry": np.array(
        [[1.0, 0.5, 0.3], [0.5, 1.0, 0.02], [0.3, 0.02, -1.0]]
    ),
    "non-finite diagonal entry": np.array(
        [[1.0, 0.5, np.inf], [0.5, 1.0, 0.02], [np.inf, 0.02, np.inf]]
    ),
}
for name, cov in cases.items():
    corr, py, ml = both(cov)
    same = np.array_equal(
        np.nan_to_num(py, nan=-999), np.nan_to_num(ml, nan=-999)
    )
    print(f"--- {name}: maskings agree = {same}")
    if not same:
        print("   corr =\n", corr)
        print("   PyVBMC ->\n", py)
        print("   MATLAB ->\n", ml)
print()

print("=== what vp.moments(cov_flag=True) can produce ===")
vp = VariationalPosterior(D=2, K=2)
vp.w = np.array([[0.5, 0.5]])
vp.mu = np.array([[0.0, 1.0], [0.0, 1.0]])
vp.sigma = np.array([[1.0, 1.0]])
vp.lambd = np.array([[1.0], [1.0]])
__, cov = vp.moments(orig_flag=False, cov_flag=True)
print(
    "  a well-formed posterior gives a PSD covariance:\n",
    cov,
    "\n  eigenvalues:",
    np.linalg.eigvalsh(cov),
)
print(
    "  a zero/negative/non-finite diagonal needs sigma or lambda to be"
    " zero or non-finite, which the parameterization excludes"
    " (sigma and lambda are stored positive)."
)
