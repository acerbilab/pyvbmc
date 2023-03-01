import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scs

from pyvbmc import VBMC
from pyvbmc.priors import SmoothBox, SplineTrapezoidal, Trapezoidal, UniformBox

figsize = (8, 4)


lb = -3
ub = 3
plb = -2
pub = 2
x = np.linspace(lb - 1, ub + 1, 1000)

prior = UniformBox(lb, ub)

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=figsize)
ax1.plot(x, prior.pdf(x))
ax1.set_xlim(-4, 4)
ax1.set_ylim(0, 0.25)
ax1.set_xlabel("x0")
ax1.set_ylabel("prior pdf")
ax2.plot(x, prior.log_pdf(x))
ax2.set_ylim(-20, 0)
ax2.set_xlabel("x0")
ax2.set_ylabel("prior log-pdf")
plt.suptitle("Uniform-box prior")
fig.tight_layout()
# (Note that the log-pdf is not plotted where it takes values of -infinity.)


prior = Trapezoidal(lb, plb, pub, ub)

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=figsize)
ax1.plot(x, prior.pdf(x))
ax1.set_xlim(-4, 4)
ax1.set_ylim(0, 0.25)
ax1.set_xlabel("x0")
ax1.set_ylabel("prior pdf")
ax2.plot(x, prior.log_pdf(x))
ax2.set_ylim(-20, 0)
ax2.set_xlabel("x0")
ax2.set_ylabel("prior log-pdf")
plt.suptitle("Trapezoidal prior")
fig.tight_layout()


prior = SplineTrapezoidal(lb, plb, pub, ub)

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=figsize)
ax1.plot(x, prior.pdf(x))
ax1.set_xlim(-4, 4)
ax1.set_ylim(0, 0.25)
ax1.set_xlabel("x0")
ax1.set_ylabel("prior pdf")
ax2.plot(x, prior.log_pdf(x))
ax2.set_ylim(-20, 0)
ax2.set_xlabel("x0")
ax2.set_ylabel("prior log-pdf")
plt.suptitle("Smoothed trapezoidal prior")
fig.tight_layout()


lb = -np.inf
ub = np.inf
plb = -2
pub = 2
# We recommend setting the scale as a fraction of the plausible range.
# For example `scale` set to 4/10 of the plausible range assigns ~50%
# (marginal) probability to the plateau of the distribution.
# Also similar fractions (e.g., half of the range) would be reasonable.
# Do not set `scale` too small with respect to the plausible range, as it
# might cause issues.
p_range = pub - plb
scale = 0.4 * p_range

prior = SmoothBox(plb, pub, scale)

x = np.linspace(plb - 2 * p_range, pub + 2 * p_range, 1000)
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=figsize)
ax1.plot(x, prior.pdf(x))
ax1.set_xlim(x[0], x[-1])
ax1.set_ylim(0, 0.25)
ax1.set_xlabel("x0")
ax1.set_ylabel("prior pdf")
ax2.plot(x, prior.log_pdf(x))
ax2.set_ylim(-20, 0)
ax2.set_xlabel("x0")
ax2.set_ylabel("prior log-pdf")
plt.suptitle("Smooth-box prior (unbounded parameters)")
fig.tight_layout()


D = 2  # Still in 2-D
lb = np.zeros((1, D))
ub = 10 * np.ones((1, D))
plb = 0.1 * np.ones((1, D))
pub = 3 * np.ones((1, D))

# Define the prior and the log-likelihood
prior = SplineTrapezoidal(lb, plb, pub, ub)


def log_likelihood(theta):
    """D-dimensional Rosenbrock's banana function."""
    theta = np.atleast_2d(theta)

    x, y = theta[:, :-1], theta[:, 1:]
    return -np.sum((x**2 - y) ** 2 + (x - 1) ** 2 / 100, axis=1)


x0 = np.ones((1, D))
np.random.seed(42)
vbmc = VBMC(log_likelihood, x0, lb, ub, plb, pub, prior=prior)
# vbmc = VBMC(log_likelihood, x0, lb, ub, plb, pub, log_prior=prior.log_pdf)  # equivalently
# vbmc = VBMC(lambda x: log_likelihood(x) + prior.log_pdf(x), x0, lb, ub, plb, pub)  # equivalently
vp, results = vbmc.optimize()
vp.plot()


D = 2  # Still in 2-D
lb = np.full((1, D), -np.inf)
ub = np.full((1, D), np.inf)
plb = -3 * np.ones((1, D))
pub = 3 * np.ones((1, D))

# Define the prior as a multivariate normal (scipy.stats distribution)
scale = 0.5 * (pub - plb).flatten()
prior = scs.multivariate_normal(mean=np.zeros(D), cov=scale**2)
# prior = [scs.norm(scale=scale[0]),scs.norm(scale=scale[1])] # equivalently

x0 = np.zeros((1, D))
np.random.seed(42)
vbmc = VBMC(log_likelihood, x0, lb, ub, plb, pub, prior=prior)
vp, results = vbmc.optimize()
vp.plot()
