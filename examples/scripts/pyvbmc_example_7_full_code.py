import sys

if "google.colab" in sys.modules:  # Colab lacks PyVBMC: install it
    get_ipython().run_line_magic("pip", 'install --upgrade "pyvbmc[torch]"')


import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scs
from scipy.special import logsumexp

from pyvbmc import SVBMC, VBMC
from pyvbmc.svbmc.utils import find_init_bounds, overlay_corner_plot

D = 2
modes = np.array([[-3.5, -3.5], [-3.5, 3.5], [3.5, -3.5], [3.5, 3.5]])
mode_sd = 0.6


def log_joint(x):
    x = np.atleast_2d(x)
    log_comps = np.stack(
        [scs.norm.logpdf(x, m, mode_sd).sum(axis=1) for m in modes], axis=1
    )
    return logsumexp(log_comps, axis=1) - np.log(len(modes))


def sample_truth(n, rng):
    which = rng.integers(len(modes), size=n)
    return modes[which] + mode_sd * rng.standard_normal((n, D))


def mode_shares(samples):
    # The share of the samples within three SDs of each mode.
    dist = np.linalg.norm(samples[:, None, :] - modes, axis=2)
    return np.mean(dist < 3 * mode_sd, axis=0)


LB = np.full((1, D), -10.0)  # hard bounds
UB = np.full((1, D), 10.0)
PLB = np.full((1, D), -5.0)  # plausible bounds
PUB = np.full((1, D), 5.0)


n_runs = 6
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

print("Share of samples within three SDs of each mode:")
print(" " * 12, "  ".join(f"({m[0]:+.1f}, {m[1]:+.1f})" for m in modes))
for label, s in zip(labels, samples):
    print(f"{label:>12}", "  ".join(f"{p:^12.2f}" for p in mode_shares(s)))


best = int(np.argmax([vp.stats["elbo"] for vp in vps]))
fig = overlay_corner_plot(
    [samples[0], vps[best].sample(n_samples)[0], samples[-1]],
    labels=["Ground truth", f"Run {best} (highest ELBO)", "Stacked"],
    figsize=(7, 7),
    bins=40,
)


fig = stacked.plot(n_samples=n_samples, figsize=(6, 6), bins=40)


stacked.save("stacked_posterior.pkl", overwrite=True)

restored = SVBMC.load("stacked_posterior.pkl")
print("Same weights:      ", np.array_equal(restored.w, stacked.w))
print("Same headline ELBO:", restored.elbo == stacked.elbo)
