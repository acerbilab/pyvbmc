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
    assert iteration_history.get("ELCBO_improvement")[1] is None
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
    assert iteration_history.get("ELCBO_improvement")[1] is None
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


def test_iteration_history_record_iteration():
    iteration_history = IterationHistory()
    optim_state = {"name": "optimstate"}
    vp = {"name": "vp"}
    elbo = 1000
    elbo_sd = 4
    varss = list()
    sKL = 3333
    sKL_true = 0.1
    gp = {"name": "gp"}
    Ns_gp = 3000
    pruned = 3
    timer = {"name": "timer"}
    iteration = 0
    iteration_history.record_iteration(
        optim_state,
        vp,
        elbo,
        elbo_sd,
        varss,
        sKL,
        sKL_true,
        gp,
        Ns_gp,
        pruned,
        timer,
        iteration,
    )

    assert iteration_history.get("optim_state")[iteration] == optim_state
    assert iteration_history.get("vp")[iteration] == vp
    assert iteration_history.get("elbo")[iteration] == elbo
    assert iteration_history.get("elbo_sd")[iteration] == elbo_sd
    assert iteration_history.get("sKL")[iteration] == sKL
    assert iteration_history.get("sKL_true")[iteration] == sKL_true
    assert iteration_history.get("gp")[iteration] == gp
    assert iteration_history.get("Ns_gp")[iteration] == Ns_gp
    assert iteration_history.get("pruned")[iteration] == pruned
    assert iteration_history.get("timer")[iteration] == timer

    optim_state["name"] = "optimstate2"
    vp["name"] =  "vp2"
    gp["name"] =  "gp2"
    timer["name"] =  "timer2"
    sKL = 3332
    iteration2 = 1
    iteration_history.record_iteration(
        optim_state,
        vp,
        elbo,
        elbo_sd,
        varss,
        sKL,
        sKL_true,
        gp,
        Ns_gp,
        pruned,
        timer,
        iteration2,
    )
    # first iteration
    assert iteration_history.get("optim_state")[iteration] != optim_state
    assert iteration_history.get("vp")[iteration] != vp
    assert iteration_history.get("gp")[iteration] != gp
    assert iteration_history.get("timer")[iteration] != timer
    assert iteration_history.get("timer")[iteration] != sKL

    # second iteration
    assert iteration_history.get("optim_state")[iteration2] == optim_state
    assert iteration_history.get("vp")[iteration2] == vp
    assert iteration_history.get("elbo")[iteration2] == elbo
    assert iteration_history.get("elbo_sd")[iteration2] == elbo_sd
    assert iteration_history.get("sKL")[iteration2] == sKL
    assert iteration_history.get("sKL_true")[iteration2] == sKL_true
    assert iteration_history.get("gp")[iteration2] == gp
    assert iteration_history.get("Ns_gp")[iteration2] == Ns_gp
    assert iteration_history.get("pruned")[iteration2] == pruned
    assert iteration_history.get("timer")[iteration2] == timer