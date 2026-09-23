"""W7-2: what the stale hyp_dict of a kept-warp iteration changes for a run
resumed there.

A seeded run on a correlated Gaussian in D = 3 is run to its end and saved.
At an iteration k that kept a warp it is resumed three ways and each resume
is run to its end:
  A. the fixed code: the record at k holds the hyperparameters fitted after
     the warp;
  B. the defect: the record's hyp_dict replaced by that of k - 1, which is
     what the code before the fix stored at k (the hyperparameters of the
     iteration before, in the space before the warp);
  C. a control: the fixed code resumed at an ordinary iteration j, to see
     whether a resume replays the uninterrupted run at all.
"""
import copy
import sys

import gpyreg
import numpy as np

import pyvbmc
from pyvbmc import VBMC

print("pyvbmc:", pyvbmc.__file__)
print("gpyreg:", gpyreg.__file__)
out = sys.argv[1]
D = 3
rho = 0.8
Sigma = np.array([[1.0, rho, rho**2], [rho, 1.0, rho], [rho**2, rho, 1.0]])
Sigma = Sigma * np.array([1.0, 0.6, 1.5])[:, None] * np.array([1.0, 0.6, 1.5])
P = np.linalg.inv(Sigma)
lnZ = 0.5 * np.linalg.slogdet(2 * np.pi * Sigma)[1]


def log_density(x):
    x = np.atleast_2d(x)
    return -0.5 * float((x @ P @ x.T).item())


def kl_to_truth(vp):
    mu, cov = vp.moments(orig_flag=True, cov_flag=True)
    mu = mu.ravel()
    return 0.5 * (
        np.trace(np.linalg.solve(Sigma, cov))
        + mu @ P @ mu
        - D
        + np.linalg.slogdet(Sigma)[1]
        - np.linalg.slogdet(cov)[1]
    )


def summary(label, vbmc, vp, results):
    elbo = np.array(vbmc.iteration_history["elbo"], dtype=float)
    print(
        f"{label}: {results['iterations']} iterations, "
        f"{results['func_count']} evaluations, elbo {results['elbo']:.5f} "
        f"(elbo - lnZ {results['elbo'] - lnZ:+.5f}), KL(vp || truth) "
        f"{kl_to_truth(vp):.5f}",
        flush=True,
    )
    return elbo


def kept_warps(vbmc):
    return [
        i
        for i, a in enumerate(vbmc.iteration_history["logging_action"])
        if "rotoscale" in a and "undo rotoscale" not in a
    ]


for seed in (20260923, 1, 2, 3):
    vbmc = VBMC(
        log_density,
        np.zeros((1, D)),
        np.full((1, D), -np.inf),
        np.full((1, D), np.inf),
        np.full((1, D), -3.0),
        np.full((1, D), 3.0),
        options={"display": "off", "plot": False},
        seed=seed,
    )
    vp, results = vbmc.optimize()
    kept = kept_warps(vbmc)
    print(f"seed {seed}: kept warps at iterations {kept}", flush=True)
    if kept and kept[0] + 2 < results["iterations"]:
        break
elbo_full = summary("uninterrupted", vbmc, vp, results)
k = kept[0]
rec = vbmc.iteration_history["optim_state"]
print(
    f"record at k={k}: hyp equals get_gp(k): "
    f"{np.array_equal(rec[k]['hyp_dict']['hyp'], vbmc.get_gp(k).get_hyperparameters(as_array=True))}; "
    f"record at k-1 shape {np.shape(rec[k-1]['hyp_dict']['hyp'])} vs "
    f"get_gp(k) {np.shape(vbmc.get_gp(k).get_hyperparameters(as_array=True))}",
    flush=True,
)
path = f"{out}/w7_2_run.pkl"
vbmc.save(path, overwrite=True)


def resume(iteration, stale=False):
    v = VBMC.load(path, iteration=iteration, set_random_state=True)
    if stale:
        old = copy.deepcopy(
            v.iteration_history["optim_state"][iteration - 1]["hyp_dict"]
        )
        v.hyp_dict = old
        v.optim_state["hyp_dict"] = v.hyp_dict
    p, r = v.optimize()
    return v, p, r


j = k - 2
vc, pc, rc = resume(j)
elbo_c = summary(f"C: resumed at the ordinary iteration {j}", vc, pc, rc)
va, pa, ra = resume(k)
elbo_a = summary(f"A: resumed at the kept warp {k}, fixed record", va, pa, ra)
vb, pb, rb = resume(k, stale=True)
elbo_b = summary(f"B: resumed at the kept warp {k}, stale record", vb, pb, rb)


def first_diff(a, b):
    n = min(len(a), len(b))
    d = [i for i in range(n) if not (a[i] == b[i])]
    return d[0] if d else None


print(
    f"C vs uninterrupted: first differing ELBO at iteration "
    f"{first_diff(elbo_full, elbo_c)} (resumed at {j})"
)
print(
    f"A vs uninterrupted: first differing ELBO at iteration "
    f"{first_diff(elbo_full, elbo_a)} (resumed at {k})"
)
print(
    f"B vs A: first differing ELBO at iteration {first_diff(elbo_a, elbo_b)}"
)
ga = va.iteration_history["gp_hyp_full"]
gb = vb.iteration_history["gp_hyp_full"]
if len(ga) > k + 1 and len(gb) > k + 1:
    print(
        f"GP hyperparameter samples at k+1: A {np.shape(ga[k+1])}, "
        f"B {np.shape(gb[k+1])}, max |mean A - mean B| "
        f"{np.max(np.abs(np.mean(ga[k+1], axis=1) - np.mean(gb[k+1], axis=1))):.4g}"
    )
kl_ab = pa.kl_div(vp2=pb, gauss_flag=True)
print(f"final posteriors A vs B: KL (Gaussian approx.) {np.round(kl_ab, 5)}")
