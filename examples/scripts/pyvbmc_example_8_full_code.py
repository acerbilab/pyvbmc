import sys

if "google.colab" in sys.modules:  # Colab lacks PyVBMC: install it
    get_ipython().run_line_magic("pip", 'install --upgrade "pyvbmc[pymc]"')


import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import scipy.stats as scs
from scipy.integrate import quad

from pyvbmc import VBMC, PyMCTarget

SEED = 7  # Fix the seed, so that the results below are reproducible


rng = np.random.default_rng(SEED)
n_observations = 30
predictor = np.linspace(-1.0, 1.0, n_observations)
design = np.column_stack([np.ones(n_observations), predictor])
beta_true = np.array([0.5, 1.0])
sigma_true = 0.7
observed = design @ beta_true + rng.normal(
    0.0, sigma_true, size=n_observations
)

coords = {
    "coef": ["intercept", "slope"],
    "observation": np.arange(n_observations),
}
with pm.Model(coords=coords) as model:
    beta = pm.Normal("beta", mu=0.0, sigma=5.0, dims="coef")
    sigma = pm.HalfNormal("sigma", sigma=2.0)
    mu = pm.Deterministic("mu", design @ beta, dims="observation")
    pm.Normal(
        "y",
        mu=mu,
        sigma=sigma,
        observed=observed,
        dims="observation",
    )


target = PyMCTarget(model, seed=SEED)
print(target)


vbmc = VBMC(target, seed=SEED)
vp, results = vbmc.optimize()


def log_evidence_given_sigma(s):
    """log p(y | sigma), with the coefficients integrated out."""
    cov = s**2 * np.eye(n_observations) + 5.0**2 * design @ design.T
    return scs.multivariate_normal(np.zeros(n_observations), cov).logpdf(
        observed
    )


# Factor out exp(shift), so that the integrand is of order one:
shift = log_evidence_given_sigma(sigma_true)
integral, _ = quad(
    lambda s: np.exp(log_evidence_given_sigma(s) - shift)
    * scs.halfnorm(scale=2.0).pdf(s),
    0,
    np.inf,
)
lml_true = shift + np.log(integral)

print("The true log model evidence is:", format(lml_true, ".3f"))
print("The obtained ELBO is:", format(results["elbo"], ".3f"))
print("The obtained ELBO_SD is:", format(results["elbo_sd"], ".3f"))


posterior_data = target.to_arviz(vp, n_samples=4000)
vbmc_summary = az.summary(
    posterior_data,
    var_names=["beta", "sigma"],
    kind="stats",  # MCMC diagnostics do not apply to independent draws
    round_to=3,
)
vbmc_summary


posterior_data = pm.compute_deterministics(
    posterior_data,
    model=model,
    var_names=["mu"],
    extend_dataset=True,  # keep beta and sigma beside mu
    progressbar=False,
)
posterior_predictive = pm.sample_posterior_predictive(
    posterior_data,
    model=model,
    var_names=["y"],
    random_seed=SEED,
    progressbar=False,
)
az.plot_ppc_interval(
    posterior_predictive,
    var_names=["y"],
    point_estimate="mean",
    ci_probs=(0.5, 0.9),
)


with model:
    nuts_data = pm.sample(
        chains=4, cores=1, random_seed=SEED, progressbar=False
    )

nuts_summary = az.summary(
    nuts_data, var_names=["beta", "sigma"], kind="stats", round_to=3
)
pd.concat({"PyVBMC": vbmc_summary, "NUTS": nuts_summary}, axis=1)


n_gradients = int(nuts_data.sample_stats["n_steps"].sum())
print(
    "PyVBMC evaluations of the log density:",
    results["evaluation_budget"]["used"],
)
print("NUTS evaluations of the log density and its gradient:", n_gradients)
