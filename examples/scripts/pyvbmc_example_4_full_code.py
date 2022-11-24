import dill
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scs
from scipy.optimize import minimize

from pyvbmc import VBMC

D = 2  # We'll use a 2-D problem for a quicker demonstration
prior_mu = np.zeros(D)
prior_var = 3 * np.ones(D)


def log_prior(theta):
    """Multivariate normal prior on theta."""
    cov = np.diag(prior_var)
    return scs.multivariate_normal(prior_mu, cov).logpdf(theta)


# log-likelihood (Rosenbrock)
def log_likelihood(theta):
    """D-dimensional Rosenbrock's banana function."""
    theta = np.atleast_2d(theta)

    x, y = theta[:, :-1], theta[:, 1:]
    return -np.sum((x**2 - y) ** 2 + (x - 1) ** 2 / 100, axis=1)


# Full model:
def log_joint(theta, data=np.ones(D)):
    """log-density of the joint distribution."""
    return log_likelihood(theta) + log_prior(theta)


LB = np.full((1, D), -np.inf)  # Lower bounds
UB = np.full((1, D), np.inf)  # Upper bounds
PLB = np.full((1, D), prior_mu - np.sqrt(prior_var))  # Plausible lower bounds
PUB = np.full((1, D), prior_mu + np.sqrt(prior_var))  # Plausible upper bounds
options = {
    "max_fun_evals": 50 * D  # Slightly reduced from 50 * (D + 2), for speed
}


np.random.seed(42)
n_runs = 3
vps, elbos, elbo_sds, success_flags, result_dicts = [], [], [], [], []
for i in range(n_runs):
    print(f"PyVBMC run {i} of {n_runs}")

    # Determine initial point x0:
    if i == 0:
        x0 = prior_mu  # First run, start from prior mean
    else:
        x0 = np.random.uniform(PLB, PUB)  # Other runs, randomize
    # Preliminary maximum a posteriori (MAP) estimation:
    x0 = minimize(
        lambda t: -log_joint(t),
        x0.ravel(),  # minimize expects 1-D input for initial point x0
        bounds=[(-np.inf, np.inf), (-np.inf, np.inf)],
    ).x

    # Run PyVBMC:
    vbmc = VBMC(
        log_joint,
        x0,
        LB,
        UB,
        PLB,
        PUB,
        options=options,
    )
    vp, results = vbmc.optimize()

    # Record the results:
    vps.append(vp)
    elbos.append(results["elbo"])
    elbo_sds.append(results["elbo_sd"])
    success_flags.append(results["success_flag"])
    result_dicts.append(results)


print(elbos)


kl_matrix = np.zeros((n_runs, n_runs))
for i in range(n_runs):
    for j in range(i, n_runs):
        # The `kldiv` method computes the divergence in both directions:
        kl_ij, kl_ji = vps[i].kl_div(vp2=vps[j])
        kl_matrix[i, j] = kl_ij
        kl_matrix[j, i] = kl_ji


print(kl_matrix)


for i, vp in enumerate(vps):
    samples, __ = vp.sample(1000)
    plt.scatter(
        samples[:, 0],
        samples[:, 1],
        alpha=1 / len(vps),
        marker=".",
        label=f"VP {i}",
    )
plt.title("""Variational Posterior Samples""")
plt.xlabel("x0")
plt.ylabel("x1")
plt.legend()


print(success_flags)


beta_lcb = 3.0  # Standard confidence parameter (in standard deviations)
# beta_lcb = 5.0  # This is more conservative
elcbos = np.array(elbos) - beta_lcb * np.array(elbo_sds)
idx_best = np.argmax(elcbos)
print(idx_best)


with open("../noise_free_vp.pkl", "wb") as f:
    dill.dump(vps[idx_best], f)
