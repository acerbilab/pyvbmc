import numpy as np
import scipy.stats as scs

from pyvbmc import VBMC

D = 2  # We consider a 2-D problem


def log_likelihood(theta):
    """D-dimensional Rosenbrock's banana function."""
    theta = np.atleast_2d(theta)

    x, y = theta[:, :-1], theta[:, 1:]
    return -np.sum((x**2 - y) ** 2 + (x - 1) ** 2 / 100, axis=1)


# In general, `log_likelihood` would depend on the data and *your* particular model


prior_mu = np.zeros((1, D))
prior_std = 3 * np.ones((1, D))


def log_prior(x):
    """Independent normal prior."""
    return np.sum(scs.norm.logpdf(x, prior_mu, prior_std))


def log_joint(x):
    """log-density of the joint distribution."""
    return log_likelihood(x) + log_prior(x)


LB = np.full((1, D), -np.inf)  # Lower bounds
UB = np.full((1, D), np.inf)  # Upper bounds


PLB = prior_mu - prior_std  # Plausible lower bounds
PUB = prior_mu + prior_std  # Plausible upper bounds

# Alternatively, you could set the plausible bounds using the quantiles:
# PLB = scs.norm.ppf(0.1587, prior_mu, prior_std)
# PUB = scs.norm.ppf(0.8413, prior_mu, prior_std)


x0 = np.copy(prior_mu)

# Alternatively, we could use a random sample from inside the plausible box:
# x0 = PLB + np.random.uniform(size=(1,D))*(PUB - PLB)


vbmc = VBMC(log_joint, x0, LB, UB, PLB, PUB)


vp, results = vbmc.optimize()


lml_true = -2.272  # ground truth, which we know for this toy scenario

print("The true log model evidence is:", lml_true)
print("The obtained ELBO is:", format(results["elbo"], ".3f"))
print("The obtained ELBO_SD is:", format(results["elbo_sd"], ".3f"))


# First, generate a large number of samples from the variational posterior:
n_samples = int(3e5)
Xs, _ = vp.sample(n_samples)

# Easily compute statistics such as moments, credible intervals, etc.
post_mean = np.mean(Xs, axis=0)  # Posterior mean
post_cov = np.cov(Xs.T)  # Posterior covariance matrix
print("The approximate posterior mean is:", post_mean)
print("The approximate posterior covariance matrix is:\n", post_cov)


vp.plot()
