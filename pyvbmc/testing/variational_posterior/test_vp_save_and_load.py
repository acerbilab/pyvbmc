from pathlib import Path

import dill
import numpy as np
import scipy.stats as scs
from pytest import raises

from pyvbmc.priors import SciPy
from pyvbmc.variational_posterior import VariationalPosterior

base_path = Path(__file__).parent


def test_vp_save_dynamic():
    """Test saving an arbitrary VariationalPosterior instance."""

    D = np.random.choice(range(1, 21))
    vp = VariationalPosterior(D)
    vp.sigma = np.ones((1, 2))
    vp.lambd = np.array(np.arange(1, D + 1)).reshape((D, 1))
    vp.save(base_path.joinpath("test_vp_save_dynamic.pkl"), overwrite=True)


def test_vp_load_dynamic():
    """Test loading the object saved above."""
    vp = VariationalPosterior.load(base_path.joinpath("test_vp_save_dynamic"))

    samples = vp.sample(100000)[0]
    for d in range(vp.D):
        assert np.isclose(d + 1, np.std(samples[:, d]), rtol=1e-2)


def test_vp_load_static():
    """Test loading VariationalPosterior object which has already been fit."""
    vp = VariationalPosterior.load(base_path.joinpath("test_vp_save_static"))
    assert vp.D == 2
    assert np.allclose(
        vp.mu,
        np.array(
            [
                [
                    0.20990155,
                    -0.32485617,
                    0.15434171,
                    0.45233729,
                    -0.15668311,
                    -0.1793005,
                    0.2427951,
                    -0.40967251,
                    0.14198521,
                    0.23391621,
                    0.38135332,
                    0.31212378,
                    -0.47160819,
                    -0.08918451,
                    0.28012249,
                    -0.316014,
                    0.18095631,
                    0.21487277,
                    -0.48769093,
                    0.32718892,
                    -0.28620862,
                    0.34553241,
                    0.26756161,
                    -0.2133796,
                    0.22070358,
                    0.30751985,
                    -0.05068146,
                    -0.49338015,
                    0.00190115,
                    0.03877403,
                    -0.1798939,
                    0.07455512,
                    0.27982491,
                    0.28114294,
                    0.14654626,
                    0.25004056,
                    -0.34846141,
                    0.06809136,
                    -0.36080548,
                    0.14595948,
                    -0.20216944,
                    -0.23940817,
                    -0.06560868,
                    -0.19496327,
                    -0.23289291,
                    0.33350488,
                    0.30273095,
                    -0.0050368,
                    -0.03969364,
                    0.15792473,
                ],
                [
                    0.0879098,
                    0.45372082,
                    0.17966125,
                    0.70664151,
                    -0.02344269,
                    0.12610371,
                    0.25115909,
                    0.5448011,
                    0.05966203,
                    -0.00233701,
                    0.42820515,
                    0.1900226,
                    0.76913012,
                    0.05242408,
                    0.16545057,
                    0.4266044,
                    -0.01658707,
                    0.1677153,
                    0.80117657,
                    0.23009211,
                    0.2372513,
                    0.19791618,
                    0.38439837,
                    0.22146568,
                    -0.08393632,
                    0.46856025,
                    -0.10851975,
                    0.79685866,
                    0.16268645,
                    0.17219586,
                    0.00370612,
                    -0.15460775,
                    0.25004612,
                    0.29880957,
                    0.0819508,
                    0.22355102,
                    0.42766117,
                    0.03572637,
                    0.32463783,
                    -0.06868017,
                    0.13280438,
                    0.21807099,
                    0.04962052,
                    0.11763449,
                    0.05816395,
                    0.50194697,
                    0.19322664,
                    0.12493376,
                    -0.08861651,
                    -0.04242627,
                ],
            ]
        ),
    )
    # Ensure callables are pickled correctly:
    x_random = np.random.normal(size=(100, vp.D))
    assert np.allclose(
        x_random,
        vp.parameter_transformer.inverse(vp.parameter_transformer(x_random)),
    )
    samples, components = vp.sample(100)


def test_vp_save_load_error_handling():
    vp = VariationalPosterior.load(base_path.joinpath("test_vp_save_static"))
    with raises(FileExistsError) as err:
        vp.save(base_path.joinpath("test_vp_save_static"))
    with raises(OSError) as err:
        vp.save("/this/path/does/not/exist.pkl")
    with raises(OSError) as err:
        vp = VariationalPosterior.load("/this/path/does/not/exist.pkl")
