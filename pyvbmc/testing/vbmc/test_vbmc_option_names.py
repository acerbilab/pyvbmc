"""An option name that PyVBMC does not declare is refused, whichever way
it was supplied: in the ``options`` dictionary, in the file of
``options_path``, or in the ``new_options`` of :py:meth:`VBMC.load`. The
same holds for an option value that construction refuses."""

import numpy as np
import pytest

from pyvbmc import VBMC

D = 2


def log_joint(x):
    return -0.5 * np.sum(x**2)


def _vbmc_of_dimension(dimension, **kwargs):
    return VBMC(
        log_joint,
        np.zeros((1, dimension)),
        np.full((1, dimension), -10.0),
        np.full((1, dimension), 10.0),
        np.full((1, dimension), -1.0),
        np.full((1, dimension), 1.0),
        **kwargs,
    )


def _vbmc(**kwargs):
    return _vbmc_of_dimension(D, **kwargs)


def test_unknown_name_in_the_options_dictionary_raises():
    with pytest.raises(ValueError) as execinfo:
        _vbmc(options={"max_fun_eval": 123})
    assert "The option max_fun_eval does not exist." in execinfo.value.args[0]


def test_unknown_name_in_an_options_file_raises(tmp_path):
    path = tmp_path.joinpath("user_options.ini")
    path.write_text("[UserOptions]\n# a misspelt budget\nmax_fun_eval = 123\n")
    with pytest.raises(ValueError) as execinfo:
        _vbmc(options_path=path)
    assert "The option max_fun_eval does not exist." in execinfo.value.args[0]


def test_declared_name_in_an_options_file_is_accepted(tmp_path):
    path = tmp_path.joinpath("user_options.ini")
    path.write_text("[UserOptions]\n# the budget\nmax_fun_evals = 123\n")
    vbmc = _vbmc(options_path=path)
    assert vbmc.options["max_fun_evals"] == 123


def test_unknown_name_in_new_options_raises(tmp_path):
    saved = tmp_path.joinpath("run.pkl")
    _vbmc().save(saved)
    with pytest.raises(ValueError) as execinfo:
        VBMC.load(saved, new_options={"max_fun_eval": 999})
    assert "The option max_fun_eval does not exist." in execinfo.value.args[0]


def test_declared_name_in_new_options_is_accepted(tmp_path):
    saved = tmp_path.joinpath("run.pkl")
    _vbmc().save(saved)
    vbmc = VBMC.load(saved, new_options={"max_fun_evals": 999})
    assert vbmc.options["max_fun_evals"] == 999


@pytest.mark.parametrize(
    "new_options, error, message",
    [
        ({"gp_hyp_sampler": "covsample"}, NotImplementedError, "slicesample"),
        ({"noise_shaping": True}, NotImplementedError, "noise_shaping"),
        ({"search_acq_fcn": "AcqFcnLog()"}, ValueError, "search_acq_fcn"),
        (
            {"search_optimizer": "Nelder-Mead"},
            ValueError,
            "search_optimizer",
        ),
        ({"search_optimizer": "bounded"}, ValueError, "search_optimizer"),
    ],
)
def test_new_options_are_checked_as_at_construction(
    tmp_path, new_options, error, message
):
    """A value that construction refuses is refused by ``load`` as well,
    with the same message, where the run would otherwise fail at the first
    use of the option, after that iteration's evaluations of the target."""
    with pytest.raises(error) as at_construction:
        _vbmc(options=new_options)
    assert message in at_construction.value.args[0]

    saved = tmp_path.joinpath("run.pkl")
    _vbmc().save(saved)
    with pytest.raises(error) as at_load:
        VBMC.load(saved, new_options=new_options)
    assert at_load.value.args[0] == at_construction.value.args[0]


@pytest.mark.parametrize("value", ["Nelder-Mead", "bounded", "fmincon", ""])
def test_a_search_optimizer_outside_the_two_values_is_refused(value):
    """``search_optimizer`` names one of the two local searches of the
    acquisition; every other value is refused, the internal ``"bounded"``
    of the one-dimensional search among them."""
    with pytest.raises(ValueError) as execinfo:
        _vbmc(options={"search_optimizer": value})
    message = execinfo.value.args[0]
    assert "search_optimizer" in message
    assert "'cmaes'" in message and "'none'" in message


@pytest.mark.parametrize("value", ["cmaes", "none"])
def test_the_two_search_optimizers_are_accepted(value):
    assert (
        _vbmc(options={"search_optimizer": value}).options["search_optimizer"]
        == value
    )


def _saved_carrying_nelder_mead(tmp_path, dimension, name):
    """A file whose stored options hold the value that release 1.0.4 wrote
    into every one-dimensional run."""
    vbmc = _vbmc_of_dimension(dimension)
    vbmc.options.__setitem__("search_optimizer", "Nelder-Mead", force=True)
    saved = tmp_path.joinpath(name)
    vbmc.save(saved)
    return saved


def test_a_one_dimensional_run_that_stored_nelder_mead_loads(tmp_path):
    """With one variable the option has no effect, the acquisition being
    searched by a bounded scalar method, so the stored value stands for the
    default and the file loads with it."""
    saved = _saved_carrying_nelder_mead(tmp_path, 1, "one.pkl")
    assert VBMC.load(saved).options["search_optimizer"] == "cmaes"


def test_a_wider_run_that_stored_nelder_mead_is_refused(tmp_path):
    """With more than one variable the value was the caller's own, and it
    named a search PyVBMC does not run."""
    saved = _saved_carrying_nelder_mead(tmp_path, 2, "two.pkl")
    with pytest.raises(ValueError) as execinfo:
        VBMC.load(saved)
    assert "new_options" in execinfo.value.args[0]


def test_a_stored_nelder_mead_that_new_options_replaces_loads(tmp_path):
    saved = _saved_carrying_nelder_mead(tmp_path, 2, "two.pkl")
    vbmc = VBMC.load(saved, new_options={"search_optimizer": "cmaes"})
    assert vbmc.options["search_optimizer"] == "cmaes"
