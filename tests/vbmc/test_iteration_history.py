import numpy as np
import pytest
from pyvbmc.vbmc import IterationHistory


def test_iteration_history_record_existing():
    iteration_history = IterationHistory()
    iteration_history["rindex"] = np.zeros(2)
    iteration_history.record("rindex", 1e-5, 0)
    assert iteration_history.get("rindex")[0] == 1e-5
    iteration_history.record("rindex", 1e-7, 1)
    assert iteration_history.get("rindex")[1] == 1e-7


def test_iteration_history_record_non_long_enough():
    # consecutive records
    iteration_history = IterationHistory()
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
    assert np.isnan(iteration_history.get("ELCBO_improvement")[1])
    assert iteration_history.get("ELCBO_improvement")[2] == 1e-7


def test_iteration_history_record_key_not_existing():
    # consecutive records
    iteration_history = IterationHistory()
    iteration_history.record("rindex", 1e-5, 0)
    assert iteration_history.get("rindex")[0] == 1e-5
    iteration_history.record("rindex", 1e-7, 1)
    assert iteration_history.get("rindex")[1] == 1e-7

    # record with gap
    iteration_history.record("ELCBO_improvement", 1e-5, 0)
    assert iteration_history.get("ELCBO_improvement")[0] == 1e-5
    iteration_history.record("ELCBO_improvement", 1e-7, 2)
    assert np.isnan(iteration_history.get("ELCBO_improvement")[1])
    assert iteration_history.get("ELCBO_improvement")[2] == 1e-7

def test_iteration_lower_than_0():
    iteration_history = IterationHistory()
    with pytest.raises(ValueError) as execinfo:
        iteration_history.record("rindex", 0.5, -1)
    assert "The iteration must be >= 0." in execinfo.value.args[0]

def test_str():
    iteration_history = IterationHistory()
    iteration_history.record("rindex", 0.5, 0)
    assert "rindex: [0.5]" in iteration_history.__str__()


def test_len():
    iteration_history = IterationHistory()
    iteration_history.record("rindex", 0.5, 0)
    assert len(iteration_history) == 1


def test_del():
    iteration_history = IterationHistory()
    iteration_history.record("rindex", 0.5, 0)
    iteration_history.pop("rindex")
    assert len(iteration_history) == 0
    assert "rindex" not in iteration_history
