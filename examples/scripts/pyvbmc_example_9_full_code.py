import sys

if "google.colab" in sys.modules:  # Colab lacks PyVBMC: install it
    get_ipython().run_line_magic(
        "pip", 'install --upgrade "pyvbmc[torch,arviz]"'
    )


import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import expit

from pyvbmc import VBMC
from pyvbmc.priors import UniformBox

SEED = 9  # Fix the seed, so that the results below are reproducible


guess_rate = 0.5
lapse_rate = 0.02
stimulus = np.linspace(-3.0, 3.0, 11)
n_trials = np.full(stimulus.size, 60)
threshold_true, slope_true = 0.3, 1.8
probability_true = guess_rate + (1 - guess_rate - lapse_rate) * expit(
    slope_true * (stimulus - threshold_true)
)
rng = np.random.default_rng(SEED)
n_correct = rng.binomial(n_trials, probability_true)

fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.scatter(stimulus, n_correct / n_trials, label="Observed proportion")
ax.plot(
    stimulus, probability_true, "--", color="0.4", label="Generating curve"
)
ax.set(xlabel="Stimulus level", ylabel="Probability correct", ylim=(0.4, 1.02))
ax.legend()
fig.tight_layout()
plt.show()


stimulus_t = torch.as_tensor(stimulus, dtype=torch.float64)
n_trials_t = torch.as_tensor(n_trials, dtype=torch.float64)
n_correct_t = torch.as_tensor(n_correct, dtype=torch.float64)


def response_probability_t(theta, levels):
    """Probability correct at each level, for each row [threshold, slope]."""
    threshold = theta[:, 0:1]
    slope = theta[:, 1:2]
    return guess_rate + (1 - guess_rate - lapse_rate) * torch.sigmoid(
        slope * (levels[None, :] - threshold)
    )


def torch_log_likelihood(x):
    """Binomial log-likelihood of the data, for each row of x."""
    with torch.no_grad():
        theta = torch.as_tensor(x, dtype=torch.float64)
        probabilities = response_probability_t(theta, stimulus_t)
        values = (
            torch.distributions.Binomial(
                total_count=n_trials_t, probs=probabilities
            )
            .log_prob(n_correct_t)
            .sum(dim=1)
        )
    return values.numpy()


# Two parameter vectors in, two log-likelihoods out:
check_points = np.array([[0.0, 1.0], [0.3, 1.8]])
torch_log_likelihood(check_points)


lower_bounds = np.array([[-3.0, 0.2]])
upper_bounds = np.array([[3.0, 6.0]])
plausible_lower = np.array([[-1.0, 0.5]])
plausible_upper = np.array([[1.0, 3.0]])
x0 = np.array([[0.0, 1.0]])
prior = UniformBox(lower_bounds, upper_bounds)

vbmc = VBMC(
    torch_log_likelihood,
    x0,
    lower_bounds,
    upper_bounds,
    plausible_lower,
    plausible_upper,
    prior=prior,
    options={"vectorized_target": True},
    seed=SEED,
)
vp, results = vbmc.optimize()


posterior_t = vp.to_torch()
torch.manual_seed(SEED)  # Sampling uses Torch's random generator
parameter_draws_t = posterior_t.sample((2000,))

levels_t = torch.linspace(-3.0, 3.0, 200, dtype=torch.float64)
probability_draws_t = response_probability_t(parameter_draws_t, levels_t)
bands = torch.quantile(
    probability_draws_t,
    torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64),
    dim=0,
).numpy()

fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.fill_between(
    levels_t.numpy(),
    bands[0],
    bands[2],
    alpha=0.25,
    label="90% credible interval",
)
ax.plot(levels_t.numpy(), bands[1], label="Posterior median")
ax.scatter(
    stimulus,
    n_correct / n_trials,
    color="black",
    s=25,
    label="Observed proportion",
)
ax.plot(
    stimulus, probability_true, "--", color="0.5", label="Generating curve"
)
ax.set(xlabel="Stimulus level", ylabel="Probability correct", ylim=(0.4, 1.02))
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()


point_t = parameter_draws_t[0].clone().requires_grad_(True)
log_density_t = posterior_t.log_prob(point_t)
gradient_t = torch.autograd.grad(log_density_t, point_t)[0]
print("Point [threshold, slope]:", point_t.detach().numpy())
print("Gradient of the log density:", gradient_t.numpy())


posterior_data = vp.to_arviz(n_samples=2000, var_names=["threshold", "slope"])
az.summary(
    posterior_data,
    kind="stats",  # MCMC diagnostics do not apply to independent draws
    round_to=3,
)


import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.scipy.stats import binom

stimulus_j = jnp.asarray(stimulus, dtype=jnp.float64)
n_trials_j = jnp.asarray(n_trials, dtype=jnp.float64)
n_correct_j = jnp.asarray(n_correct, dtype=jnp.float64)


@jax.jit
def jax_log_likelihood_values(theta):
    threshold = theta[:, 0:1]
    slope = theta[:, 1:2]
    probabilities = guess_rate + (
        1 - guess_rate - lapse_rate
    ) * jax.nn.sigmoid(slope * (stimulus_j[None, :] - threshold))
    return binom.logpmf(n_correct_j, n_trials_j, probabilities).sum(axis=1)


def jax_log_likelihood(x):
    """Binomial log-likelihood of the data, for each row of x."""
    values = jax_log_likelihood_values(jnp.asarray(x, dtype=jnp.float64))
    return np.asarray(jax.device_get(values), dtype=np.float64)


print("JAX:  ", jax_log_likelihood(check_points))
print("Torch:", torch_log_likelihood(check_points))


vbmc_jax = VBMC(
    jax_log_likelihood,
    x0,
    lower_bounds,
    upper_bounds,
    plausible_lower,
    plausible_upper,
    prior=prior,
    options={"vectorized_target": True, "display": "off"},
    seed=SEED,
)
vp_jax, results_jax = vbmc_jax.optimize()

print("ELBO with the Torch likelihood:", format(results["elbo"], ".3f"))
print("ELBO with the JAX likelihood:  ", format(results_jax["elbo"], ".3f"))
