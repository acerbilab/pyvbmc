"""Small PyMC models and hand-written densities for adapter tests."""

import numpy as np


def _log_normal(x, mean, sd):
    x = np.asarray(x, dtype=np.float64)
    return -0.5 * np.log(2 * np.pi) - np.log(sd) - 0.5 * ((x - mean) / sd) ** 2


def _log_halfnormal(x, sd):
    x = np.asarray(x, dtype=np.float64)
    return np.where(
        x > 0,
        0.5 * np.log(2 / np.pi) - np.log(sd) - 0.5 * (x / sd) ** 2,
        -np.inf,
    )


def scalar_model(rng):
    import pymc as pm

    y = rng.normal(1.2, 1.5, size=20)
    with pm.Model() as model:
        mu = pm.Normal("mu", 0.0, 3.0)
        pm.Normal("y", mu, 1.5, observed=y)

    def hand(point):
        mu_value = float(point["mu"])
        return float(
            _log_normal(mu_value, 0.0, 3.0)
            + np.sum(_log_normal(y, mu_value, 1.5))
        )

    precision = 1 / 9.0 + len(y) / 1.5**2
    return {
        "name": "scalar",
        "model": model,
        "hand": hand,
        "truth_mean": float(np.sum(y) / 1.5**2 / precision),
        "truth_sd": float(np.sqrt(1 / precision)),
    }


def positive_model(rng):
    import pymc as pm

    y = rng.normal(0.0, 0.8, size=15)
    with pm.Model() as model:
        sigma = pm.HalfNormal("sigma", 2.0)
        pm.Normal("y", 0.0, sigma, observed=y)

    def hand(point):
        sigma_value = float(point["sigma"])
        if sigma_value <= 0:
            return -np.inf
        return float(
            _log_halfnormal(sigma_value, 2.0)
            + np.sum(_log_normal(y, 0.0, sigma_value))
        )

    return {"name": "positive", "model": model, "hand": hand}


def vector_model(rng):
    import pymc as pm

    n = 30
    X = np.column_stack((np.ones(n), rng.normal(size=n)))
    y = X @ np.array([0.5, -1.0]) + rng.normal(0.0, 0.7, size=n)
    with pm.Model(coords={"coef": ["intercept", "slope"]}) as model:
        beta = pm.Normal("beta", 0.0, 5.0, dims="coef")
        sigma = pm.HalfNormal("sigma", 2.0)
        mu = pm.Deterministic("mu", X @ beta)
        pm.Normal("y", mu, sigma, observed=y)

    def hand(point):
        beta_value = np.asarray(point["beta"], dtype=np.float64)
        sigma_value = float(point["sigma"])
        if sigma_value <= 0:
            return -np.inf
        return float(
            np.sum(_log_normal(beta_value, 0.0, 5.0))
            + _log_halfnormal(sigma_value, 2.0)
            + np.sum(_log_normal(y, X @ beta_value, sigma_value))
        )

    return {
        "name": "vector",
        "model": model,
        "hand": hand,
        "X": X,
        "y": y,
    }


def bounded_model(rng):
    import pymc as pm
    from scipy.special import betaln

    lower = np.array([-1.0, 0.0])
    upper = np.array([2.0, 1.0])
    y = rng.normal(1.5, 1.0, size=12)
    with pm.Model() as model:
        p = pm.Beta("p", 2.0, 2.0)
        u = pm.Uniform("u", lower=lower, upper=upper, shape=2)
        pm.Normal("y", p + u[0] + u[1], 1.0, observed=y)

    def hand(point):
        p_value = float(point["p"])
        u_value = np.asarray(point["u"], dtype=np.float64)
        if (
            not 0 < p_value < 1
            or np.any(u_value <= lower)
            or np.any(u_value >= upper)
        ):
            return -np.inf
        return float(
            np.log(p_value)
            + np.log1p(-p_value)
            - betaln(2.0, 2.0)
            - np.sum(np.log(upper - lower))
            + np.sum(_log_normal(y, p_value + u_value.sum(), 1.0))
        )

    return {"name": "bounded", "model": model, "hand": hand}


def one_sided_model(rng):
    import pymc as pm
    from scipy.stats import norm

    y_t = rng.normal(1.8, 1.0, size=8)
    y_v = rng.normal(-1.2, 1.0, size=8)
    with pm.Model() as model:
        t = pm.TruncatedNormal("t", 0.0, 2.0, lower=1.0)
        v = pm.TruncatedNormal("v", 0.0, 2.0, upper=-0.5)
        pm.Normal("y_t", t, 1.0, observed=y_t)
        pm.Normal("y_v", v, 1.0, observed=y_v)
    norm_t = float(norm.logsf(1.0, 0.0, 2.0))
    norm_v = float(norm.logcdf(-0.5, 0.0, 2.0))

    def hand(point):
        t_value = float(point["t"])
        v_value = float(point["v"])
        if t_value <= 1.0 or v_value >= -0.5:
            return -np.inf
        return float(
            _log_normal(t_value, 0.0, 2.0)
            - norm_t
            + np.sum(_log_normal(y_t, t_value, 1.0))
            + _log_normal(v_value, 0.0, 2.0)
            - norm_v
            + np.sum(_log_normal(y_v, v_value, 1.0))
        )

    return {"name": "one_sided", "model": model, "hand": hand}


def accepted_models(seed=20260916):
    builders = (
        scalar_model,
        positive_model,
        vector_model,
        bounded_model,
        one_sided_model,
    )
    return [
        builder(np.random.default_rng([seed, index]))
        for index, builder in enumerate(builders)
    ]


def mixed_order_model():
    """Model whose partial transform removal changes PyMC's RV order."""
    import pymc as pm

    with pm.Model(coords={"coef": ["intercept", "slope"]}) as model:
        pm.Normal("beta", shape=2, dims="coef")
        pm.HalfNormal("sigma", 2.0)
        pm.TruncatedNormal("t", 0.0, 2.0, lower=1.0)
        pm.Beta("p", 2.0, 2.0)
    return model


def schools_model(centered):
    import pymc as pm

    y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
    sd = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
    with pm.Model() as model:
        mu = pm.Normal("mu", 0.0, 5.0)
        tau = pm.HalfCauchy("tau", 5.0)
        if centered:
            theta = pm.Normal("theta", mu, tau, shape=8)
        else:
            z = pm.Normal("z", 0.0, 1.0, shape=8)
            theta = pm.Deterministic("theta", mu + tau * z)
        pm.Normal("y", theta, sd, observed=y)
    return model


def no_gradient_model():
    import pymc as pm
    import pytensor.tensor as pt
    from pytensor.compile.ops import as_op

    @as_op(itypes=[pt.dscalar], otypes=[pt.dscalar])
    def likelihood(x):
        return np.asarray(-0.5 * ((x - 1.2) / 0.3) ** 2)

    with pm.Model() as model:
        x = pm.Normal("x", 0.0, 3.0)
        pm.Potential("likelihood", likelihood(x))
    return model


def undefined_gradient_model():
    """Model whose potential uses a scalar operator without a gradient."""
    import pymc as pm
    from pytensor.scalar.basic import UnaryScalarOp, upgrade_to_float
    from pytensor.tensor.elemwise import Elemwise

    class _Quadratic(UnaryScalarOp):
        def impl(self, x):
            return -0.5 * ((x - 1.2) / 0.3) ** 2

    quadratic = Elemwise(_Quadratic(upgrade_to_float, name="quadratic"))

    with pm.Model() as model:
        x = pm.Normal("x", 0.0, 3.0)
        pm.Potential("likelihood", quadratic(x))
    return model


def undefined_hessian_model():
    """Model with a first derivative but no second derivative.

    The Rice log density calls an exponentially scaled Bessel function whose
    derivative is another such function, and that second operator carries no
    derivative rule of its own.
    """
    import pymc as pm

    with pm.Model() as model:
        pm.Rice("x", nu=1.0, sigma=1.0)
    return model


def rejected_models(seed=771):
    import pymc as pm

    rng = np.random.default_rng(seed)
    models = {}
    with pm.Model() as model:
        k = pm.Poisson("k", 3.0)
        pm.Normal("y", k, 1.0, observed=rng.normal(3.0, 1.0, size=5))
    models["discrete"] = model
    with pm.Model() as model:
        w = pm.Dirichlet("w", np.ones(3))
        pm.Multinomial("counts", 20, w, observed=np.array([5, 7, 8]))
    models["simplex"] = model
    with pm.Model() as model:
        s = pm.HalfNormal("s", 2.0, default_transform=None)
        pm.Normal("y", 0.0, s, observed=rng.normal(size=5))
    models["suppressed"] = model
    with pm.Model() as model:
        mu = pm.Normal("mu", 0.0, 1.0)
        h = pm.Uniform("h", lower=mu, upper=mu + 1.0)
        pm.Normal("y", h, 1.0, observed=rng.normal(size=5))
    models["random_bounds"] = model
    with pm.Model() as model:
        upper = pm.Truncated(
            "upper",
            pm.StudentT.dist(nu=5.0, mu=2.0, sigma=0.5),
            lower=1.0,
            upper=3.0,
            initval=2.0,
        )
        pm.Uniform("symbolic_bound", 0.0, upper, initval=0.5)
    models["symbolic_random_bounds"] = model
    with pm.Model() as model:
        pm.Normal(
            "ordered",
            0.0,
            1.0,
            shape=3,
            transform=pm.distributions.transforms.ordered,
        )
    models["ordered"] = model
    with pm.Model() as model:
        pm.ZeroSumNormal("zero_sum", sigma=1.0, shape=(2, 3))
    models["zero_sum"] = model
    with pm.Model() as model:
        pm.Uniform(
            "mixed",
            lower=np.array([0.0, -np.inf]),
            upper=np.array([1.0, 2.0]),
            shape=2,
        )
    models["mixed"] = model
    return models


def nonfinite_mode_model():
    import pymc as pm
    import pytensor.tensor as pt

    with pm.Model() as model:
        x = pm.Normal("x", 0.0, 1.0)
        pm.Potential("invalid_near_mode", pt.log(x - 10.0))
    return model


def stochastic_initial_model():
    import pymc as pm

    with pm.Model() as model:
        pm.Normal("x", 0.0, 1.0, initval="prior")
    return model
