"""Wave 3, A-6: does the warp's correlation threshold leave an indefinite
covariance on realistic posteriors?

`warp_input` (`whitening.py:142-182`, as `misc/warp_input_vbmc.m:46-73`)
zeroes the entries of the posterior covariance whose correlation is at most
`warp_roto_corr_thresh` (0.05) and takes the SVD of what is left as its
eigendecomposition. A thresholded covariance need not be positive
semi-definite from `D = 3` on; the SVD then returns `|lambda|`.

This script takes the posterior each run of the population campaigns ended
with before its final boost (the `pre` entry of every `*.boost.pkl`; one per
run, `D` from 2 to 15), builds the covariance exactly as `warp_input` does,
applies the threshold and reports the smallest eigenvalue. For an indefinite
case it also computes the transform the code would install and the variances
the posterior attains under it (1 means whitened). Nothing is optimized and
no target is evaluated.
"""

import collections
import glob
import sys

import dill
import numpy as np

import pyvbmc

sys.path.insert(0, "dev/scripts")
print("pyvbmc:", pyvbmc.__file__, flush=True)

THRESH = 0.05
paths = sorted(glob.glob("dev/scripts/runs/population_*/results/*.boost.pkl"))
print("boost captures:", len(paths), flush=True)

by_problem = collections.defaultdict(lambda: [0, 0, 0, 0])
margins = collections.defaultdict(list)
cases = []
failed = 0
for i, p in enumerate(paths):
    name = p.replace("\\", "/").split("/")[-1].replace(".boost.pkl", "")
    problem = name.rsplit("_seed", 1)[0]
    try:
        with open(p, "rb") as fh:
            vp = dill.load(fh)["pre"]
        __, cov = vp.moments(orig_flag=False, cov_flag=True)
    except Exception as exc:  # a capture this checkout cannot read
        failed += 1
        if failed <= 3:
            print("cannot read", name, repr(exc)[:120], flush=True)
        continue
    pt = vp.parameter_transformer
    R = np.eye(vp.D) if pt.R_mat is None else pt.R_mat
    s = np.ones(vp.D) if pt.scale is None else pt.scale
    cov = R @ np.diag(s) @ cov @ np.diag(s) @ R.T
    cov = np.diag(pt.delta) @ cov @ np.diag(pt.delta)
    corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
    mask = np.abs(corr) <= THRESH
    cov_t = cov.copy()
    cov_t[mask] = 0
    eig = np.linalg.eigvalsh(cov_t)
    eig_full = np.linalg.eigvalsh(cov)
    rec = by_problem[problem]
    rec[0] += 1
    rec[1] += int(mask.sum() > 0)
    rec[3] = vp.D
    # How much of the smallest eigenvalue the threshold removes: 1 means
    # none, 0 means the matrix became singular, below 0 indefinite.
    margins[problem].append(eig[0] / eig_full[0])
    if eig[0] < -1e-12 * eig[-1]:
        rec[2] += 1
        U, sv, __ = np.linalg.svd(cov_t)
        scale = np.sqrt(sv + np.finfo(np.float64).eps)
        W = np.diag(1 / scale) @ U.T
        attained = np.diag(W @ cov @ W.T)
        cases.append(
            (
                name,
                vp.D,
                vp.K,
                int(mask.sum() // 2),
                eig[0] / eig[-1],
                eig_full[0] / eig_full[-1],
                attained.min(),
                attained.max(),
            )
        )
    if (i + 1) % 100 == 0:
        print(f"... {i + 1} read", flush=True)

print(
    "\nproblem: D, posteriors, with any entry zeroed, indefinite", flush=True
)
n = z = bad = 0
for problem in sorted(by_problem):
    c, anyz, ind, D = by_problem[problem]
    n, z, bad = n + c, z + anyz, bad + ind
    print(
        f"{problem}: D={D} n={c} zeroed={anyz} indefinite={ind} "
        f"smallest eigenvalue after/before the threshold: min "
        f"{min(margins[problem]):.3f}, median "
        f"{np.median(margins[problem]):.3f}",
        flush=True,
    )
print(
    f"\ntotal: {n} posteriors, {z} with an entry zeroed, {bad} indefinite "
    f"after the threshold; {failed} captures unreadable",
    flush=True,
)
print(
    "\nindefinite cases: run, D, K, zeroed pairs, min/max eigenvalue after "
    "the threshold, min/max eigenvalue before it, attained variances "
    "[min, max]",
    flush=True,
)
for c in sorted(cases, key=lambda c: c[4]):
    print(
        f"{c[0]} D={c[1]} K={c[2]} zeroed={c[3]} after={c[4]:.3e} "
        f"before={c[5]:.3e} attained=[{c[6]:.3g}, {c[7]:.3g}]",
        flush=True,
    )
