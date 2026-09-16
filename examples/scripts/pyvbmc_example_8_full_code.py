import arviz as az
import numpy as np
import pymc as pm

from pyvbmc import VBMC, PyMCTarget

SEED = 7


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


total_budget = 150
vbmc = VBMC(
    target,
    options={"max_fun_evals": total_budget, "display": "off"},
    seed=SEED,
)
vp, results = vbmc.optimize()

budget = results["evaluation_budget"]
print(f"Convergence status: {results['convergence_status']}")
print(
    "Function-equivalent budget: "
    f"{budget['used']} used of {budget['limit']} "
    f"({budget['initialization']} setup + {budget['new_calls']} fresh)"
)
print(
    "Reused setup observations: "
    f"{results.get('precomputed_observations', 0)}"
)


posterior_data = target.to_arviz(vp, n_samples=1000)
vbmc_summary = az.summary(
    posterior_data,
    var_names=["beta", "sigma"],
    kind="stats",
    round_to=3,
)
vbmc_summary


posterior_data = pm.compute_deterministics(
    posterior_data,
    model=target.model,
    var_names=["mu"],
    extend_dataset=True,
    progressbar=False,
)
assert {"beta", "sigma", "mu"} <= set(posterior_data.posterior.data_vars)
posterior_data.posterior["mu"]


posterior_predictive = pm.sample_posterior_predictive(
    posterior_data,
    model=target.model,
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
        draws=500,
        tune=500,
        chains=2,
        cores=1,
        target_accept=0.9,
        random_seed=SEED,
        progressbar=False,
    )

az.summary(
    nuts_data,
    var_names=["beta", "sigma"],
    kind="all",
    round_to=3,
)


# Use a common sample axis for the independent draws and NUTS chains.
forest_data = {
    label: az.extract(data, var_names=["beta", "sigma"]).reset_index(
        "sample", drop=True
    )
    for label, data in {"PyVBMC": posterior_data, "NUTS": nuts_data}.items()
}
forest = az.plot_forest(
    forest_data,
    var_names=["beta", "sigma"],
    sample_dims="sample",
    combined=True,
    point_estimate="mean",
    ci_probs=(0.5, 0.9),
)
forest.add_legend("model", title="method")
