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
