from pathlib import Path

import dill
import numpy as np
import pytest
import scipy.stats as scs

from pyvbmc import VBMC
from pyvbmc.priors import SciPy

base_path = Path(__file__).parent


def test_vbmc_save_dynamic():
    """Test saving an arbitrary VBMC instance."""

    D = np.random.choice(range(1, 21))
    lb = np.random.uniform(size=(1, D))
    plb = lb + 0.1 + np.random.uniform(size=(1, D))
    pub = plb + 1.0 + np.random.uniform(size=(1, D))
    ub = pub + 0.1 + np.random.uniform(size=(1, D))
    x0 = np.random.uniform(plb, pub)
    x_random = np.random.uniform(lb, ub, size=(100, D))

    # Test saving lambda expression likelihood and "complicated" prior object
    def log_likelihood(x):
        return np.sum(x**2 + x + 1)

    prior = SciPy._generic(D)
    vbmc = VBMC(log_likelihood, x0, lb, ub, plb, pub, prior=prior)
    vbmc.save(base_path.joinpath("test_vbmc_save_dynamic.pkl"), overwrite=True)

    y_random = vbmc.log_joint(x_random)
    data = {
        "D": D,
        "lb": lb,
        "plb": plb,
        "pub": pub,
        "ub": ub,
        "x0": x0,
        "x_random": x_random,
        "y_random": y_random,
    }
    with open(
        base_path.joinpath("test_vbmc_save_dynamic_data.pkl"), "wb"
    ) as f:
        dill.dump(data, f)


def test_vbmc_load_dynamic():
    """Test loading the object saved above."""
    vbmc = VBMC.load(base_path.joinpath("test_vbmc_save_dynamic"))
    with open(
        base_path.joinpath("test_vbmc_save_dynamic_data.pkl"), "rb"
    ) as f:
        data = dill.load(f)

    assert vbmc.D == data["D"]
    assert np.all(vbmc.lower_bounds == data["lb"])
    assert np.all(vbmc.plausible_lower_bounds == data["plb"])
    assert np.all(vbmc.plausible_upper_bounds == data["pub"])
    assert np.all(vbmc.upper_bounds == data["ub"])
    assert np.allclose(vbmc.parameter_transformer.inverse(vbmc.x0), data["x0"])
    assert np.all(vbmc.log_joint(data["x_random"]) == data["y_random"])


def test_vbmc_load_static():
    """Test loading VBMC object which has already been optimized."""
    D = 4
    prior_mu = np.zeros(D)
    prior_var = 3 * np.ones(D)
    LB = np.full(D, -np.inf)  # Lower bounds
    PLB = np.full(D, prior_mu - np.sqrt(prior_var))  # Plausible lower bounds
    PUB = np.full(D, prior_mu + np.sqrt(prior_var))  # Plausible upper bounds
    UB = np.full(D, np.inf)  # Upper bounds

    random_state = np.random.get_state()

    vbmc = VBMC.load(base_path.joinpath("test_vbmc_save_static.pkl"))
    assert vbmc.D == D
    assert np.all(np.equal(vbmc.lower_bounds, LB))
    assert np.all(np.equal(vbmc.plausible_lower_bounds, PLB))
    assert np.all(np.equal(vbmc.plausible_upper_bounds, PUB))
    assert np.all(np.equal(vbmc.upper_bounds, UB))
    assert vbmc.options["max_iter"] == 300
    for i, val in enumerate(random_state):
        assert np.all(np.random.get_state()[i] == val)
    assert vbmc.options["max_fun_evals"] == 40
    assert vbmc.iteration == 6
    assert vbmc.options["vectorized_target"] is False
    assert vbmc.function_logger.vectorized_target is False

    vbmc = VBMC.load(
        base_path.joinpath("test_vbmc_save_static.pkl"),
        iteration=0,
    )
    assert vbmc.D == D
    assert np.all(np.equal(vbmc.lower_bounds, LB))
    assert np.all(np.equal(vbmc.plausible_lower_bounds, PLB))
    assert np.all(np.equal(vbmc.plausible_upper_bounds, PUB))
    assert np.all(np.equal(vbmc.upper_bounds, UB))
    assert vbmc.options["max_iter"] == 300
    for i, val in enumerate(random_state):
        assert np.all(np.random.get_state()[i] == val)
    assert vbmc.options["max_fun_evals"] == 40
    assert vbmc.iteration == 0
    # The instance is set up to continue, so its GP carries the posterior
    # factors the iteration history does not store, and is a copy of the
    # record rather than the record itself.
    assert vbmc.gp.posteriors[0].alpha is not None
    assert vbmc.gp is not vbmc.iteration_history["gp"][0]

    vbmc = VBMC.load(
        base_path.joinpath("test_vbmc_save_static.pkl"),
        new_options={"max_fun_evals": 42},
        iteration=0,
    )
    assert vbmc.D == D
    assert np.all(np.equal(vbmc.lower_bounds, LB))
    assert np.all(np.equal(vbmc.plausible_lower_bounds, PLB))
    assert np.all(np.equal(vbmc.plausible_upper_bounds, PUB))
    assert np.all(np.equal(vbmc.upper_bounds, UB))
    assert vbmc.options["max_iter"] == 300
    for i, val in enumerate(random_state):
        assert np.all(np.random.get_state()[i] == val)
    assert vbmc.options["max_fun_evals"] == 42
    assert vbmc.iteration == 0

    vbmc = VBMC.load(
        base_path.joinpath("test_vbmc_save_static.pkl"),
        new_options={"max_fun_evals": 42},
        iteration=0,
        set_random_state=True,
    )
    assert vbmc.D == D
    assert np.all(np.equal(vbmc.lower_bounds, LB))
    assert np.all(np.equal(vbmc.plausible_lower_bounds, PLB))
    assert np.all(np.equal(vbmc.plausible_upper_bounds, PUB))
    assert np.all(np.equal(vbmc.upper_bounds, UB))
    assert vbmc.options["max_iter"] == 300
    assert not np.all(np.random.get_state()[1] == random_state[1])
    assert vbmc.options["max_fun_evals"] == 42
    assert vbmc.iteration == 0


def test_vbmc_save_load_error_handling():
    vbmc = VBMC.load(base_path.joinpath("test_vbmc_save_static.pkl"))
    with pytest.raises(FileExistsError) as err:
        vbmc.save(base_path.joinpath("test_vbmc_save_static.pkl"))
    with pytest.raises(OSError) as err:
        vbmc.save("/this/path/does/not/exist.pkl")
    with pytest.raises(OSError) as err:
        vbmc = VBMC.load("/this/path/does/not/exist.pkl")
    with pytest.raises(ValueError) as err:
        vbmc = VBMC.load(
            base_path.joinpath("test_vbmc_save_static.pkl"), iteration=10
        )
    assert (
        "Specified iteration (10) should be >= 0 and <= last stored iteration (6)."
        in err.value.args[0]
    )
    with pytest.raises(ValueError) as err:
        vbmc = VBMC.load(
            base_path.joinpath("test_vbmc_save_static.pkl"), iteration=-1
        )
    assert (
        "Specified iteration (-1) should be >= 0 and <= last stored iteration (6)."
        in err.value.args[0]
    )


def _save_load_vbmc(target, tmp_path, *, vectorized):
    D = 2
    vbmc = VBMC(
        target,
        np.zeros((1, D)),
        np.full((1, D), -np.inf),
        np.full((1, D), np.inf),
        np.full((1, D), -1.0),
        np.full((1, D), 1.0),
        prior=SciPy._generic(D),
        options={"vectorized_target": vectorized},
    )
    path = tmp_path / ("vectorized" if vectorized else "scalar")
    vbmc.save(path)
    return path


def test_vbmc_save_load_vectorized_mode_and_override(tmp_path):
    def dual_target(x):
        if x.ndim == 1:
            return np.sum(x)
        return np.sum(x, axis=1)

    path = _save_load_vbmc(dual_target, tmp_path, vectorized=True)
    loaded = VBMC.load(path)
    assert loaded.options["vectorized_target"] is True
    assert loaded.function_logger.vectorized_target is True
    values, _, _ = loaded.function_logger.batch_call(np.zeros((2, loaded.D)))
    assert values.shape == (2,)

    scalar = VBMC.load(path, new_options={"vectorized_target": False})
    assert scalar.options["vectorized_target"] is False
    assert scalar.function_logger.vectorized_target is False
    value, _, _ = scalar.function_logger(np.zeros(scalar.D))
    assert np.isfinite(value)


def test_vbmc_load_rebuilds_vectorized_prior_wrapper(tmp_path):
    def dual_target(x):
        if x.ndim == 1:
            return np.sum(x)
        return np.sum(x, axis=1)

    path = _save_load_vbmc(dual_target, tmp_path, vectorized=False)
    loaded = VBMC.load(path, new_options={"vectorized_target": True})

    assert loaded.options["vectorized_target"] is True
    assert loaded.function_logger.vectorized_target is True
    values, _, _ = loaded.function_logger.batch_call(np.zeros((2, loaded.D)))
    assert values.shape == (2,)


def test_vbmc_load_defaults_missing_vectorized_flags_to_false(tmp_path):
    path = _save_load_vbmc(np.sum, tmp_path, vectorized=False)
    with open(path.with_suffix(".pkl"), "rb") as file:
        vbmc = dill.load(file)
    del vbmc.options["vectorized_target"]
    del vbmc.function_logger.vectorized_target
    vbmc.save(path, overwrite=True)

    loaded = VBMC.load(path)

    assert loaded.options["vectorized_target"] is False
    assert loaded.function_logger.vectorized_target is False


def test_vbmc_load_mode_override_requires_likelihood_provenance(tmp_path):
    path = _save_load_vbmc(np.sum, tmp_path, vectorized=False)
    with open(path.with_suffix(".pkl"), "rb") as file:
        vbmc = dill.load(file)
    vbmc.log_likelihood = None
    vbmc.save(path, overwrite=True)

    with pytest.raises(ValueError, match="original likelihood is unavailable"):
        VBMC.load(path, new_options={"vectorized_target": True})
