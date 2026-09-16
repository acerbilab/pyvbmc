import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import expit

from pyvbmc import VBMC
from pyvbmc.priors import UniformBox

SEED = 9


guess_rate = 0.5
lapse_rate = 0.02
stimulus = np.linspace(-3.0, 3.0, 11)
n_trials = np.full(stimulus.size, 60)
threshold_true, slope_true = 0.3, 1.8
probability_true = guess_rate + (1 - guess_rate - lapse_rate) * expit(
    slope_true * (stimulus - threshold_true)
)
data_rng = np.random.default_rng(SEED)
n_correct = data_rng.binomial(n_trials, probability_true)

fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.scatter(stimulus, n_correct / n_trials, label="Observed proportion")
ax.plot(
    stimulus, probability_true, "--", color="0.4", label="Generating curve"
)
ax.set(xlabel="Stimulus level", ylabel="Probability correct", ylim=(0.4, 1.02))
ax.legend()
fig.tight_layout()
plt.show()


device = torch.device("cpu")
stimulus_t = torch.as_tensor(stimulus, dtype=torch.float64, device=device)
trials_t = torch.as_tensor(n_trials, dtype=torch.float64, device=device)
correct_t = torch.as_tensor(n_correct, dtype=torch.float64, device=device)


def response_probability_t(theta, levels):
    threshold = theta[:, 0:1]
    slope = theta[:, 1:2]
    return guess_rate + (1 - guess_rate - lapse_rate) * torch.sigmoid(
        slope * (levels[None, :] - threshold)
    )


def torch_log_likelihood(x):
    with torch.no_grad():
        theta = torch.as_tensor(x, dtype=torch.float64, device=device)
        probabilities = response_probability_t(theta, stimulus_t)
        values = (
            torch.distributions.Binomial(
                total_count=trials_t, probs=probabilities
            )
            .log_prob(correct_t)
            .sum(dim=1)
        )
    return values.detach().cpu().numpy()


check_points = np.array([[0.0, 1.0], [0.3, 1.8], [-0.5, 3.0]])
print("Batch log-likelihoods:", torch_log_likelihood(check_points))
print("Single-row output shape:", torch_log_likelihood(check_points[:1]).shape)


lower_bounds = np.array([-3.0, 0.2])
upper_bounds = np.array([3.0, 6.0])
plausible_lower = np.array([-1.0, 0.5])
plausible_upper = np.array([1.0, 3.0])
x0 = np.array([0.0, 1.0])
prior = UniformBox(lower_bounds, upper_bounds)
options = {"vectorized_target": True, "display": "off"}

vbmc = VBMC(
    torch_log_likelihood,
    x0,
    lower_bounds,
    upper_bounds,
    plausible_lower,
    plausible_upper,
    prior=prior,
    options=options,
    seed=SEED,
)
vp, results = vbmc.optimize()
print("Convergence status:", results["convergence_status"])
print("Target evaluations:", results["func_count"])
print(f"ELBO: {results['elbo']:.3f} +/- {results['elbo_sd']:.3f}")


posterior_t = vp.to_torch()
torch.manual_seed(SEED)
parameter_draws_t = posterior_t.sample((2000,))
print("Posterior sample shape:", tuple(parameter_draws_t.shape))

# The exported density agrees with the NumPy posterior at the same points.
np.testing.assert_allclose(
    posterior_t.log_prob(parameter_draws_t[:10]).detach().numpy(),
    vp.pdf(parameter_draws_t[:10].numpy(), log_flag=True).ravel(),
    rtol=1e-10,
    atol=1e-10,
)

grid_t = torch.linspace(-3.0, 3.0, 200, dtype=torch.float64)
probability_draws_t = response_probability_t(parameter_draws_t, grid_t)
bands = torch.quantile(
    probability_draws_t,
    torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64),
    dim=0,
).numpy()

fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.fill_between(
    grid_t.numpy(),
    bands[0],
    bands[2],
    alpha=0.25,
    label="90% pointwise credible interval",
)
ax.plot(grid_t.numpy(), bands[1], label="Posterior median probability")
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


point_t = parameter_draws_t[0].detach().clone().requires_grad_(True)
log_density_t = posterior_t.log_prob(point_t)
gradient_t = torch.autograd.grad(log_density_t, point_t)[0]
print("Point [threshold, slope]:", point_t.detach().numpy())
print("Log-density gradient:", gradient_t.detach().numpy())


import arviz as az

posterior_data = vp.to_arviz(n_samples=2000, var_names=["threshold", "slope"])
print(az.summary(posterior_data, kind="stats", round_to=3))


import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.scipy.stats import binom

stimulus_j = jnp.asarray(stimulus, dtype=jnp.float64)
trials_j = jnp.asarray(n_trials, dtype=jnp.float64)
correct_j = jnp.asarray(n_correct, dtype=jnp.float64)


@jax.jit
def jax_log_likelihood_values(theta):
    threshold = theta[:, 0:1]
    slope = theta[:, 1:2]
    probabilities = guess_rate + (
        1 - guess_rate - lapse_rate
    ) * jax.nn.sigmoid(slope * (stimulus_j[None, :] - threshold))
    return binom.logpmf(correct_j, trials_j, probabilities).sum(axis=1)


def jax_log_likelihood(x):
    values = jax_log_likelihood_values(jnp.asarray(x, dtype=jnp.float64))
    return np.asarray(jax.device_get(values), dtype=np.float64)


for points in (check_points, check_points[:1]):
    np.testing.assert_allclose(
        jax_log_likelihood(points),
        torch_log_likelihood(points),
        rtol=1e-10,
        atol=1e-10,
    )
print("Torch and JAX likelihoods agree for batch and single-row inputs.")
