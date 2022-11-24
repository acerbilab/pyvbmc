import dill
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scs

from pyvbmc import VBMC

D = 2  # We'll use a 2-D problem, again for speed
prior_mu = np.zeros(D)
prior_var = 3 * np.ones(D)
LB = np.full((1, D), -np.inf)  # Lower bounds
UB = np.full((1, D), np.inf)  # Upper bounds
PLB = np.full((1, D), prior_mu - np.sqrt(prior_var))  # Plausible lower bounds
PUB = np.full((1, D), prior_mu + np.sqrt(prior_var))  # Plausible upper


def log_prior(theta):
    """Multivariate normal prior on theta, same as before."""
    cov = np.diag(prior_var)
    return scs.multivariate_normal(prior_mu, cov).logpdf(theta)


# log-likelihood (Rosenbrock)
def log_likelihood(theta):
    """D-dimensional Rosenbrock's banana function."""
    theta = np.atleast_2d(theta)
    n, D = theta.shape

    # Standard deviation of synthetic noise:
    noise_sd = np.sqrt(1.0 + 0.5 * np.linalg.norm(theta) ** 2)

    # Rosenbrock likelihood:
    x, y = theta[:, :-1], theta[:, 1:]
    base_density = -np.sum((x**2 - y) ** 2 + (x - 1) ** 2 / 100, axis=1)

    noisy_estimate = base_density + noise_sd * np.random.normal(size=(n, 1))
    return noisy_estimate, noise_sd


# Full model:
def log_joint(theta, data=np.ones(D)):
    """log-density of the joint distribution."""
    log_p = log_prior(theta)
    log_l, noise_est = log_likelihood(theta)
    # For the joint, we have to add log densities and carry-through the noise estimate.
    return log_p + log_l, noise_est


x0 = np.zeros((1, D))  # Initial point


options = {"specify_target_noise": True}
vbmc = VBMC(
    log_joint,
    x0,
    LB,
    UB,
    PLB,
    PUB,
    options=options,
)


np.random.seed(42)
vp, results = vbmc.optimize()


with open("../noise_free_vp.pkl", "rb") as f:
    noise_free_vp = dill.load(f)
# KL divergence between this VP and the noise-free VP:
print(vbmc.vp.kl_div(vp2=noise_free_vp))


vp.plot(title="Noisy VP")


noise_free_vp.plot(title="Noise-Free VP")
