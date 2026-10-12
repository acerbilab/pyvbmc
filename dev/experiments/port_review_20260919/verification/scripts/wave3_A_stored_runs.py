"""Wave 3, A-2, A-3, A-4 and A-6 measured on the stored complete runs.

Reads the 21 stored runs (no target is evaluated, nothing is optimized) and
reports, per run and in total:

A-2  how many main GP-training calls collect the historical starting points
     (`init_N > 0`) with an even history length `n`, where Python's window
     `range(ceil((n+1)/2)-1, n)` is one recorded GP short of MATLAB's
     `ceil(n/2):n` (`misc/gptrain_vbmc.m:39`), and how many rows that GP holds;
A-3  in how many of those calls the pool is subsampled (`N0 > init_N/2`) with
     an odd `init_N`, where Python keeps `ceil(init_N/2)` and MATLAB
     `floor(Ninit/2)` (`misc/gptrain_vbmc.m:44`);
A-4  the rows of the recorded `gp_hyp_full` block beside the number of
     hyperparameter samples of the recorded GP (equal: the thinned set) and
     what MATLAB records, `Ns * Thin` (`gplite/gplite_train.m:314`, `:482`);
A-6  for every recorded posterior that meets the warp's own conditions
     (`D > 1`, `K >= warp_min_k`, after warm-up), whether zeroing the
     covariance entries with `|corr| <= warp_roto_corr_thresh` leaves a
     matrix with a negative eigenvalue, and the variances the resulting
     transform then attains, computed as `whitening.py:142-182` does.

`init_N` is recomputed from the recorded `n_eff` by the formula of
`gaussian_process_train.py:620-639`; whether the call kept it or set it to
zero is decided as at `:651-656` from the recorded reliability index, taking
`recompute_var_post` as False (it is not recorded per iteration), so the
A-2 and A-3 counts are slightly low.
"""

import glob
import math
import sys

import dill
import numpy as np

import pyvbmc

sys.path.insert(0, "dev/scripts")
print("pyvbmc:", pyvbmc.__file__, flush=True)

paths = sorted(
    glob.glob("dev/scripts/runs/gp_box_20260916/*/*.vbmc.pkl")
    + glob.glob("dev/scripts/runs/svbmc_pool_20260913/pool/*.vbmc.pkl")
)
print("stored runs:", len(paths), flush=True)

tot = dict(calls=0, collect=0, even=0, subsample=0, odd_sub=0, vps=0, indef=0)
worst = []
full_ratio = []

for p in paths:
    name = p.replace("\\", "/").split("/")[-1].replace(".vbmc.pkl", "")
    with open(p, "rb") as fh:
        v = dill.load(fh)
    h = v.iteration_history
    opt = v.options
    n_iter = len(h["iter"])
    D = v.D

    # --- A-2 / A-3 -------------------------------------------------------
    a = -(opt["gp_train_n_init"] - opt["gp_train_n_init_final"])
    b, c, d = -3 * a, 3 * a, opt["gp_train_n_init"]
    limit = min(v.optim_state.get("max_fun_evals", opt["max_fun_evals"]), 1e3)
    span = limit - opt["fun_eval_start"]
    rows = [np.atleast_2d(h["gp_hyp_full"][i]).shape[0] for i in range(n_iter)]
    r = dict(calls=0, collect=0, even=0, subsample=0, odd_sub=0)
    missing_rows = []
    for it in range(1, n_iter):
        r["calls"] += 1
        x = (h["n_eff"][it] - opt["fun_eval_start"]) / span
        init_N = max(round(a * x**3 + b * x**2 + c * x + d), 9)
        if it > 1 and h["r_index"][it - 1] < opt["gp_retrain_threshold"]:
            init_N = 0
        if init_N <= 0:
            continue
        r["collect"] += 1
        n = it
        py = list(range(math.ceil((n + 1) / 2) - 1, n))
        ml = list(range(math.ceil(n / 2) - 1, n))
        if py != ml:
            r["even"] += 1
            missing_rows.append(rows[ml[0]])
        N0 = sum(rows[i] for i in py)
        if N0 > init_N / 2:
            r["subsample"] += 1
            if init_N % 2 == 1:
                r["odd_sub"] += 1
    for k in r:
        tot[k] += r[k]

    # --- A-4 -------------------------------------------------------------
    thin = opt["gp_sample_thin"]
    sampled = [
        (rows[i], int(h["Ns_gp"][i]))
        for i in range(n_iter)
        if h["Ns_gp"][i] and h["Ns_gp"][i] > 0
    ]
    same = sum(1 for rr, ns in sampled if rr == ns)
    full_ratio.append((len(sampled), same))

    # --- A-6 -------------------------------------------------------------
    thresh = opt["warp_roto_corr_thresh"]
    n_vp = n_indef = 0
    for t in range(n_iter):
        vp = h["vp"][t]
        if vp is None or vp.D < 2 or vp.K < opt["warp_min_k"]:
            continue
        if h["warmup"][t]:
            continue
        __, cov = vp.moments(orig_flag=False, cov_flag=True)
        pt = vp.parameter_transformer
        R = np.eye(vp.D) if pt.R_mat is None else pt.R_mat
        s = np.ones(vp.D) if pt.scale is None else pt.scale
        cov = R @ np.diag(s) @ cov @ np.diag(s) @ R.T
        cov = np.diag(pt.delta) @ cov @ np.diag(pt.delta)
        corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
        cov_t = cov.copy()
        mask = np.abs(corr) <= thresh
        cov_t[mask] = 0
        eig = np.linalg.eigvalsh(cov_t)
        n_vp += 1
        if eig[0] < -1e-12 * eig[-1]:
            n_indef += 1
            U, sv, __ = np.linalg.svd(cov_t)
            scale = np.sqrt(sv + np.finfo(np.float64).eps)
            W = np.diag(1 / scale) @ U.T
            attained = np.diag(W @ cov @ W.T)
            worst.append(
                (
                    name,
                    t,
                    vp.D,
                    vp.K,
                    int(mask.sum() // 2),
                    eig[0] / eig[-1],
                    attained.min(),
                    attained.max(),
                )
            )
    tot["vps"] += n_vp
    tot["indef"] += n_indef

    print(
        f"{name}: D={D} iters={n_iter} | A-2/3: calls={r['calls']} "
        f"collect={r['collect']} window-differs={r['even']} "
        f"(rows of the missing GP: {missing_rows[:6]}"
        f"{'...' if len(missing_rows) > 6 else ''}) subsampled="
        f"{r['subsample']} odd-and-subsampled={r['odd_sub']} | A-4: "
        f"sampling iterations={len(sampled)}, recorded rows == Ns_gp in "
        f"{same} (MATLAB records Ns*{thin}) | A-6: posteriors={n_vp} "
        f"indefinite-after-threshold={n_indef}",
        flush=True,
    )

print("\n== totals ==", flush=True)
print(tot, flush=True)
print(
    "A-4: sampling iterations",
    sum(a for a, _ in full_ratio),
    "with recorded rows == Ns_gp in",
    sum(b for _, b in full_ratio),
    flush=True,
)
print(
    "\n== A-6: posteriors whose thresholded covariance is indefinite ==",
    flush=True,
)
print(
    "run, iter, D, K, zeroed pairs, min eig / max eig, attained variance "
    "min, max (1 = whitened)",
    flush=True,
)
for w in sorted(worst, key=lambda w: w[5]):
    print(
        f"{w[0]} it={w[1]} D={w[2]} K={w[3]} zeroed={w[4]} "
        f"ratio={w[5]:.3e} attained=[{w[6]:.3g}, {w[7]:.3g}]",
        flush=True,
    )
