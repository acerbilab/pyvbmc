import dill
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


with open("../vbmc_test_save.pkl", "wb") as f:
    dill.dump(vbmc, f)

with open("../vbmc_test_save.pkl", "rb") as f:
    vbmc = dill.load(f)


iteration = (
    vbmc.iteration
)  # continue from specified iteration, here it's the last iteration
assert 1 <= iteration <= vbmc.iteration

## Set states for VBMC right before the specified iteration
vbmc.gp = vbmc.iteration_history["gp"][iteration - 1]
vbmc.vp = vbmc.iteration_history["vp"][iteration - 1]
vbmc.function_logger = vbmc.iteration_history["function_logger"][iteration - 1]
vbmc.optim_state = vbmc.iteration_history["optim_state"][iteration - 1]
vbmc.hyp_dict = vbmc.optim_state["hyp_dict"]
vbmc.iteration = iteration - 1

for k, v in vbmc.iteration_history.items():
    try:
        vbmc.iteration_history[k] = vbmc.iteration_history[k][:iteration]
    except TypeError:
        pass
# (Optionally) Set random state for reproducibility, note that it can only
# reproduce exactly the same optimization process when VBMC's options are not updated
random_state = vbmc.iteration_history["random_state"][iteration - 1]
np.random.set_state(random_state)


options = {
    "max_fun_evals": 50 * (D + 2),
}
vbmc.options.is_initialized = (
    False  # Temporarily set to False for updating options
)
vbmc.options.update(options)
vbmc.options.is_initialized = True

vbmc.is_finished = False
vp, results = vbmc.optimize()


print(format_dict(results))


with open("../vbmc_test_save.pkl", "wb") as f:
    dill.dump(vbmc, f)

with open("../vbmc_test_save.pkl", "rb") as f:
    vbmc = dill.load(f)

samples, components = vbmc.vp.sample(5)
# `samples` are samples drawn from the variational posterior.
# `components` are the index of the mixture components each
#  sample was drawn from.
print(samples)
print(components)
