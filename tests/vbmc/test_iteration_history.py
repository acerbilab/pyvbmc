import numpy as np
import pytest
from pyvbmc.vbmc import IterationHistory


def test_iteration_history_record_existing():
    iteration_history = IterationHistory(["rindex"])
    iteration_history["rindex"] = np.zeros(2)
    iteration_history.record("rindex", 1e-5, 0)
    assert iteration_history.get("rindex")[0] == 1e-5
    iteration_history.record("rindex", 1e-7, 1)
    assert iteration_history.get("rindex")[1] == 1e-7


def test_iteration_history_record_non_long_enough():
    # consecutive records
    iteration_history = IterationHistory(["rindex", "ELCBO_improvement"])
    iteration_history["rindex"] = np.zeros(1)
    iteration_history.record("rindex", 1e-5, 0)
    assert iteration_history.get("rindex")[0] == 1e-5
    iteration_history.record("rindex", 1e-7, 1)
    assert iteration_history.get("rindex")[1] == 1e-7

    # record with gap
    iteration_history["ELCBO_improvement"] = np.zeros(1)
    iteration_history.record("ELCBO_improvement", 1e-5, 0)
    assert iteration_history.get("ELCBO_improvement")[0] == 1e-5
    iteration_history.record("ELCBO_improvement", 1e-7, 2)
    assert iteration_history.get("ELCBO_improvement")[1] is None
    assert iteration_history.get("ELCBO_improvement")[2] == 1e-7


def test_iteration_history_record_key_not_existing():
    # consecutive records
    iteration_history = IterationHistory([])
    with pytest.raises(ValueError) as execinfo:
        iteration_history.record("rindex", 0.5, 0)
    assert "The key has not been specified" in execinfo.value.args[0]


def test_iteration_lower_than_0():
    iteration_history = IterationHistory(["rindex"])
    with pytest.raises(ValueError) as execinfo:
        iteration_history.record("rindex", 0.5, -1)
    assert "The iteration must be >= 0." in execinfo.value.args[0]


def test_str():
    iteration_history = IterationHistory(["rindex"])
    iteration_history.record("rindex", 0.5, 0)
    assert "rindex: [0.5]" in iteration_history.__str__()


def test_len():
    iteration_history = IterationHistory(["rindex"])
    iteration_history.record("rindex", 0.5, 0)
    assert len(iteration_history) == 1


def test_del():
    iteration_history = IterationHistory(["rindex"])
    iteration_history.record("rindex", 0.5, 0)
    iteration_history.pop("rindex")
    assert len(iteration_history) == 0
    assert "rindex" not in iteration_history


def test_iteration_history_set_item_deepcopy():
    iteration_history = IterationHistory(["foo", "foo2"])
    foo = {"name": "foo"}
    iteration_history["foo"] = foo
    assert iteration_history.get("foo") == foo
    foo["name"] = "foo2"
    iteration_history["foo2"] = foo
    assert iteration_history.get("foo") != foo
    assert iteration_history.get("foo2") == foo


def test_iteration_history_record_iteration():
    """
    Extensive testing to make sure that the deepcopy works.
    """
    iteration_history = IterationHistory(["gp", "elbo_sd"])
    iteration_key_values = dict()
    iteration_key_values["gp"] = {"name": "gp"}
    iteration_key_values["elbo_sd"] = 4
    iteration_history.record_iteration(
        iteration_key_values,
        0,
    )

    assert iteration_history.get("elbo_sd")[0] == 4
    assert iteration_history.get("gp")[0] == iteration_key_values.get("gp")

    iteration_key_values["gp"]["name"] = "gp2"
    iteration_key_values["elbo_sd"] = 2
    iteration_history.record_iteration(
        iteration_key_values,
        1,
    )
    # first iteration
    assert iteration_history.get("elbo_sd")[0] == 4
    assert iteration_history.get("gp")[0] != iteration_key_values.get("gp")

    # second iteration
    assert iteration_history.get("elbo_sd")[1] == 2
    assert iteration_history.get("gp")[1] == iteration_key_values.get("gp")


def test_iteration_history_record_iteration_iteration_below_0():
    """
    Extensive testing to make sure that the deepcopy works.
    """
    iteration_history = IterationHistory(["gp", "elbo_sd"])
    iteration_key_values = dict()
    iteration_key_values["gp"] = {"name": "gp"}
    iteration_key_values["elbo_sd"] = 4
    with pytest.raises(ValueError) as execinfo:
        iteration_history.record_iteration(
            iteration_key_values,
            -1,
        )
    assert "The iteration must be >= 0." in execinfo.value.args[0]


def test_iteration_history_record_iteration_key_not_existing():
    iteration_history = IterationHistory([])
    iteration_key_values = dict()
    iteration_key_values["gp"] = {"name": "gp"}
    with pytest.raises(ValueError) as execinfo:
        iteration_history.record_iteration(
            iteration_key_values,
            1,
        )
    assert "The key has not been specified" in execinfo.value.args[0]


def test_iteration__setitem___key_not_existing():
    iteration_history = IterationHistory([])
    with pytest.raises(ValueError) as execinfo:
        iteration_history["foo"] = None
    assert "The key has not been specified" in execinfo.value.args[0]
