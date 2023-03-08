import numpy as np
import scipy.stats as scs
from scipy.optimize import minimize

from pyvbmc import VBMC
from pyvbmc.formatting import format_dict

D = 4  # A four-dimensional problem
prior_mu = np.zeros(D)
prior_var = 3 * np.ones(D)


def log_prior(theta):
    """Multivariate normal prior on theta."""
    cov = np.diag(prior_var)
    return scs.multivariate_normal(prior_mu, cov).logpdf(theta)


def log_likelihood(theta, data=np.ones(D)):
    """D-dimensional Rosenbrock's banana function."""
    # In this simple demo the data just translates the parameters:
    theta = np.atleast_2d(theta)
    theta = theta + data

    x, y = theta[:, :-1], theta[:, 1:]
    return -np.sum((x**2 - y) ** 2 + (x - 1) ** 2 / 100, axis=1)


def log_joint(theta, data=np.ones(D)):
    """log-density of the joint distribution."""
    return log_likelihood(theta, data) + log_prior(theta)


LB = np.full(D, -np.inf)  # Lower bounds
UB = np.full(D, np.inf)  # Upper bounds
PLB = np.full(D, prior_mu - np.sqrt(prior_var))  # Plausible lower bounds
PUB = np.full(D, prior_mu + np.sqrt(prior_var))  # Plausible upper bounds


np.random.seed(41)
x0 = np.random.uniform(PLB, PUB)  # Random point inside plausible box
x0 = minimize(
    lambda t: -log_joint(t),
    x0,
    bounds=[
        (-np.inf, np.inf),
        (-np.inf, np.inf),
        (-np.inf, np.inf),
        (-np.inf, np.inf),
    ],
).x
np.random.seed(42)


# Limit number of function evaluations
options = {
    "max_fun_evals": 10 * D,
}
# We can specify either the log-joint, or the log-likelihood and log-prior.
# In other words, the following lines are equivalent:
vbmc = VBMC(
    log_likelihood,
    x0,
    LB,
    UB,
    PLB,
    PUB,
    options=options,
    log_prior=log_prior,
)
# vbmc = VBMC(
#     log_joint,
#     x0, LB, UB, PLB, PUB, options=options,
# )


vp, results = vbmc.optimize()


print(results["success_flag"])


print(format_dict(results))


# Here we specify `overwrite=True`, since we don't care about overwriting our
# test file. By default, `overwrite=False` and PyVBMC will raise an error if
# the file already exists.
vbmc.save("vbmc_test_save.pkl", overwrite=True)
# We could also save just the final variational posterior:
# vbmc.vp.save("vp_test_save.pkl")


new_options = {
    "max_fun_evals": 50 * (D + 2),
}
vbmc = VBMC.load(
    "vbmc_test_save.pkl",
    new_options=new_options,
    iteration=None,  # the default: start from the last stored iteration.
    set_random_state=False,  # the default: don't modify the random state
    # (can be set to True for reproducibility).
)
vp, results = vbmc.optimize()


print(format_dict(results))


vbmc.save("vbmc_test_save.pkl", overwrite=True)
vbmc = VBMC.load("vbmc_test_save.pkl")

samples, components = vbmc.vp.sample(5)
# `samples` are samples drawn from the variational posterior.
# `components` are the index of the mixture components each
#  sample was drawn from.
print(samples)
print(components)
