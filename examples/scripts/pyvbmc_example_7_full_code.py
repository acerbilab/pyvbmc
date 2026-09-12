import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scs
from scipy.special import logsumexp

from pyvbmc import SVBMC, VBMC
from pyvbmc.svbmc.utils import find_init_bounds, overlay_corner_plot

D = 2
modes = np.array([[-2.5, -2.5], [2.5, 2.5]])
mode_sd = 0.8


def log_joint(x):
    x = np.atleast_2d(x)
    log_comps = np.stack(
        [scs.norm.logpdf(x, m, mode_sd).sum(axis=1) for m in modes], axis=1
    )
    return logsumexp(log_comps, axis=1) + np.log(0.5)


def sample_truth(n, rng):
    which = rng.integers(len(modes), size=n)
    return modes[which] + mode_sd * rng.standard_normal((n, D))


LB = np.full((1, D), -10.0)  # hard bounds
UB = np.full((1, D), 10.0)
PLB = np.full((1, D), -5.0)  # plausible bounds
PUB = np.full((1, D), 5.0)


n_runs = 4
rng = np.random.default_rng(0)
sample_lb, sample_ub = find_init_bounds(LB, UB, PLB, PUB)

vps = []
for i in range(n_runs):
    x0 = rng.uniform(sample_lb, sample_ub, size=(1, D))
    vbmc = VBMC(
        log_joint, x0, LB, UB, PLB, PUB, options={"display": "off"}, seed=i
    )
    vp, results = vbmc.optimize()
    vps.append(vp)
    print(
        f"Run {i}: x0 = {np.round(x0.ravel(), 2)}, "
        f"ELBO = {vp.stats['elbo']:.3f} +/- {vp.stats['elbo_sd']:.3f}, "
        f"converged: {vp.stats['stable']}"
    )


stacked = SVBMC(vps, seed=0)
stacked.optimize(n_samples_final=100)

print("Headline stacked ELBO:", round(stacked.elbo, 3))
print("SD of the raw estimate: ", round(stacked.elbo_sd, 3))
print("Headline method:        ", stacked.elbo_details["headline_method"])
print("Raw ELBO:               ", round(stacked.elbo_details["raw"], 3))
print("Noise status source:    ", stacked.noise_status_source)

# Total weight assigned to each retained run:
offsets = np.concatenate([[0], np.cumsum(stacked.K)])
for m in range(stacked.M):
    share = stacked.w[0, offsets[m] : offsets[m + 1]].sum()
    print(f"Run {m}: total weight {share:.3f}")


n_samples = 5000
samples = [sample_truth(n_samples, rng)]
labels = ["Ground truth"]
for vp in stacked.vp_list:  # the runs that passed the filters
    samples.append(vp.sample(n_samples)[0])
    labels.append(f"Run {vps.index(vp)}")
samples.append(stacked.sample(n_samples))
labels.append("Stacked")

fig = overlay_corner_plot(samples, labels=labels, figsize=(7, 7), bins=40)


fig = stacked.plot(n_samples=n_samples, figsize=(6, 6), bins=40)
