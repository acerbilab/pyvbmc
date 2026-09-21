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


def _vbmc(**kwargs):
    return VBMC(
        log_joint,
        np.zeros((1, D)),
        np.full((1, D), -10.0),
        np.full((1, D), 10.0),
        np.full((1, D), -1.0),
        np.full((1, D), 1.0),
        **kwargs,
    )


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
