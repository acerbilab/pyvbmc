"""Detection of changes to a PyMC target after a run was set up."""

import logging

import numpy as np
import pytest

pm = pytest.importorskip("pymc")
pytest.importorskip("arviz_base")
import pytensor
import pytensor.tensor as pt

from pyvbmc import VBMC
from pyvbmc.pymc import PyMCTarget

# A refused optimize() returns at once; the cap bounds the run otherwise.
OPTIONS = {"display": "off", "max_iter": 1}


def _loop_target(weights):
    # PyTensor interns loop bodies by the values of their arrays: a test
    # whose loop read the same values as another's would read that test's
    # array. Each test passes values of its own.
    with pm.Model() as model:
        scale = pm.HalfNormal("scale")
        # The loop's function captures the array, which PyTensor keeps in
        # the loop's inner graph, where the target does not copy it.
        total = pytensor.scan(
            lambda i, accumulated, s: accumulated
            + pt.as_tensor_variable(weights)[i] * s,
            sequences=pt.arange(len(weights)),
            outputs_info=pt.zeros(()),
            non_sequences=[scale],
            return_updates=False,
        )[-1]
        pm.Normal("y", total, 1.0, observed=np.array(2.0))
    return PyMCTarget(
        model,
        start={"scale": 1.0},
        plausible_bounds={"scale": (0.3, 3.0)},
        seed=923,
    )


def _warnings(caplog):
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "PyMC target of this run has changed" in record.getMessage()
    ]


def test_unchanged_target_saves_and_loads_without_warning(tmp_path, caplog):
    vbmc = VBMC(_loop_target(np.array([1.0, 2.0, 3.0])), options=OPTIONS)
    vbmc.save(tmp_path / "run")
    loaded = VBMC.load(tmp_path / "run")

    assert loaded._pymc_target_changed is None
    assert vbmc._pymc_target_change() is None
    assert not _warnings(caplog)


def test_changed_loop_array_warns_and_refuses_to_continue(tmp_path, caplog):
    weights = np.array([1.25, 2.5, 3.75])
    vbmc = VBMC(_loop_target(weights), options=OPTIONS, seed=924)
    weights += 3.0

    with pytest.raises(RuntimeError, match=r"optimize\(\) is refused"):
        vbmc.optimize()
    vbmc.save(tmp_path / "run")
    (saved,) = _warnings(caplog)
    assert "pytensor.scan" in saved

    caplog.clear()
    loaded = VBMC.load(tmp_path / "run")
    (warning,) = _warnings(caplog)
    assert "pytensor.scan" in warning
    assert "starting point" in warning
    assert "remain valid" in warning
    with pytest.raises(RuntimeError, match=r"optimize\(\) is refused"):
        loaded.optimize()


def test_load_compares_the_density_with_the_recorded_value(tmp_path, caplog):
    with pm.Model() as model:
        mu = pm.Normal("mu", 0.0, 2.0)
        pm.Normal("y", mu, 1.0, observed=np.array([0.3, -0.1, 0.4]))
    target = PyMCTarget(
        model, start={"mu": 0.0}, plausible_bounds={"mu": (-1.0, 1.0)}
    )
    vbmc = VBMC(target, options=OPTIONS, seed=925)
    # As if the run had recorded another density, which a change that
    # leaves no trace in the target's arrays would also produce.
    X_setup, y_setup = target.setup_evaluations
    target.setup_evaluations = (X_setup, y_setup + 1.0)
    vbmc.save(tmp_path / "run")
    assert not _warnings(caplog)

    loaded = VBMC.load(tmp_path / "run")
    (warning,) = _warnings(caplog)
    assert "starting point" in warning
    assert "pytensor.scan" not in warning
    with pytest.raises(RuntimeError, match=r"optimize\(\) is refused"):
        loaded.optimize()
