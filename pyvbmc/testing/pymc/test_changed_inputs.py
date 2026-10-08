"""Detection of changes to a PyMC target after a run was set up."""

import logging

import numpy as np
import pytest

pm = pytest.importorskip("pymc")
pytest.importorskip("arviz_base")
import pytensor
import pytensor.tensor as pt
from pytensor.compile.builders import OpFromGraph
from pytensor.tensor.blockwise import Blockwise

from pyvbmc import VBMC
from pyvbmc.pymc import PyMCTarget

# A refused optimize() returns at once; the cap bounds the run otherwise.
OPTIONS = {"display": "off", "max_iter": 1}
REFUSED = r"optimize\(\) is refused"


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


def _plain_target():
    with pm.Model() as model:
        mu = pm.Normal("mu", 0.0, 2.0)
        pm.Normal("y", mu, 1.0, observed=np.array([0.3, -0.1, 0.4]))
    return PyMCTarget(
        model, start={"mu": 0.0}, plausible_bounds={"mu": (-1.0, 1.0)}
    )


def _warnings(caplog):
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "PyMC target of this run has changed" in record.getMessage()
    ]


def test_unchanged_target_saves_and_loads_without_warning(tmp_path, caplog):
    vbmc = VBMC(
        _loop_target(np.array([1.0, 2.0, 3.0])), options=OPTIONS, seed=926
    )
    vbmc.save(tmp_path / "run")
    loaded = VBMC.load(tmp_path / "run")

    assert loaded._pymc_target_changed is None
    assert vbmc._pymc_target_change() is None
    assert not _warnings(caplog)


def test_changed_loop_array_warns_and_refuses_to_continue(tmp_path, caplog):
    weights = np.array([1.25, 2.5, 3.75])
    vbmc = VBMC(_loop_target(weights), options=OPTIONS, seed=924)
    weights += 3.0

    with pytest.raises(RuntimeError, match=REFUSED):
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
    with pytest.raises(RuntimeError, match=REFUSED):
        loaded.optimize()


def test_changed_array_in_a_vectorized_operation_is_detected():
    covariates = np.array([[1.07, 0.5], [1.0, 1.0], [1.0, 2.0]])
    row = pt.dvector("row")
    # A Blockwise operation applies an operation with an inner graph, here
    # an OpFromGraph that captures the array, across a batch.
    product = Blockwise(
        OpFromGraph([row], [pt.as_tensor_variable(covariates) @ row]),
        signature="(n)->(m)",
    )
    with pm.Model() as model:
        beta = pm.Normal("beta", 0.0, 2.0, shape=(2, 2))
        pm.Normal("y", product(beta), 1.0, observed=np.zeros((2, 3)))
    target = PyMCTarget(
        model,
        start={"beta": np.full((2, 2), 0.4)},
        plausible_bounds={
            "beta": (np.full((2, 2), -2.0), np.full((2, 2), 2.0))
        },
        seed=928,
    )
    vbmc = VBMC(target, options=OPTIONS, seed=929)
    assert vbmc._pymc_target_change() is None

    covariates += 5.0
    assert "OpFromGraph" in vbmc._pymc_target_change()
    with pytest.raises(RuntimeError, match=REFUSED):
        vbmc.optimize()


def test_only_loading_evaluates_the_target_once(tmp_path, monkeypatch):
    weights = np.array([2.25, 0.5, 1.75])
    vbmc = VBMC(_loop_target(weights), options=OPTIONS, seed=930)
    calls = []
    log_joint = PyMCTarget.log_joint

    def counted(self, x):
        calls.append(x)
        return log_joint(self, x)

    monkeypatch.setattr(PyMCTarget, "log_joint", counted)
    weights += 1.0
    vbmc.save(tmp_path / "run")
    with pytest.raises(RuntimeError, match=REFUSED):
        vbmc.optimize()
    assert not calls

    loaded = VBMC.load(tmp_path / "run")
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match=REFUSED):
        loaded.optimize()
    assert len(calls) == 1


def test_load_compares_the_density_with_the_recorded_value(tmp_path, caplog):
    target = _plain_target()
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
    with pytest.raises(RuntimeError, match=REFUSED):
        loaded.optimize()


def test_load_warns_when_the_target_fails_to_evaluate(
    tmp_path, caplog, monkeypatch
):
    vbmc = VBMC(_plain_target(), options=OPTIONS, seed=927)
    vbmc.save(tmp_path / "run")

    def failing(self, x):
        raise IndexError("index out of bounds")

    monkeypatch.setattr(PyMCTarget, "log_joint", failing)
    loaded = VBMC.load(tmp_path / "run")
    (warning,) = _warnings(caplog)
    assert "raised IndexError: index out of bounds" in warning
    with pytest.raises(RuntimeError, match=REFUSED):
        loaded.optimize()


def test_target_without_a_digest_is_checked_by_its_density(tmp_path, caplog):
    target = _plain_target()
    # As a target built before the digest existed.
    del target._inner_digest
    vbmc = VBMC(target, options=OPTIONS, seed=931)
    assert vbmc._pymc_target_change() is None
    vbmc.save(tmp_path / "run")

    loaded = VBMC.load(tmp_path / "run")
    assert loaded._pymc_target_changed is None
    assert not _warnings(caplog)
