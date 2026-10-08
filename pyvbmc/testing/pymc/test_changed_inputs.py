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
OPTIONS = {"display": "off", "min_iter": 0, "max_iter": 1}
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
        and "PyMC target of this run no longer matches" in record.getMessage()
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
    # Restoring the array lets the run continue.
    weights -= 3.0
    assert vbmc._pymc_target_change() is None

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


def test_changed_array_moved_out_of_its_loop_is_detected(tmp_path, caplog):
    covariates = np.array([[1.0, 0.75], [0.25, 1.5]])
    with pm.Model() as model:
        scale = pm.HalfNormal("scale")
        # The term does not change between iterations, so compiling moves it,
        # and the array it reads, out of the loop. At the starting point,
        # scale = 1, the term is zero whatever the array holds, so only the
        # digest can show the change.
        total = pytensor.scan(
            lambda i, accumulated, s: accumulated
            + pt.dot(
                pt.as_tensor_variable(covariates),
                pt.stack([s - 1.0, s - 1.0]),
            )[i % 2],
            sequences=pt.arange(3),
            outputs_info=pt.zeros(()),
            non_sequences=[scale],
            return_updates=False,
        )[-1]
        pm.Normal("y", total, 1.0, observed=np.array(2.0))
    target = PyMCTarget(
        model,
        start={"scale": 1.0},
        plausible_bounds={"scale": (0.3, 3.0)},
        seed=932,
    )
    vbmc = VBMC(target, options=OPTIONS, seed=933)
    covariates += 3.0

    assert "pytensor.scan" in vbmc._pymc_target_change()
    with pytest.raises(RuntimeError, match=REFUSED):
        vbmc.optimize()
    vbmc.save(tmp_path / "run")
    assert _warnings(caplog)
    loaded = VBMC.load(tmp_path / "run")
    assert "pytensor.scan" in loaded._pymc_target_changed
    assert "starting point" not in loaded._pymc_target_changed


def test_swapped_inner_arrays_are_detected_with_unchanged_start(
    tmp_path, caplog
):
    first = np.array([1.13, 2.27])
    second = np.array([3.39, 4.41])
    x = pt.dscalar("x")
    mean = OpFromGraph(
        [x],
        [
            pt.dot(first, pt.stack([x, x**2]))
            + pt.dot(second, pt.stack([x**3, x**4]))
        ],
    )
    with pm.Model() as model:
        beta = pm.Normal("beta", 0.0, 2.0)
        pm.Normal("y", mean(beta), 1.0, observed=np.array(0.0))
    target = PyMCTarget(
        model,
        start={"beta": 0.0},
        plausible_bounds={"beta": (-2.0, 2.0)},
        seed=934,
    )
    vbmc = VBMC(target, options=OPTIONS, seed=935)
    start_value = target.log_joint(target.x0)
    probe = np.array([0.5])
    probe_value = target.log_joint(probe)

    # Both arrays are still present, with their roles exchanged. The density
    # at zero cannot reveal the change because every term vanishes there.
    original = first.copy()
    first[:] = second
    second[:] = original
    assert target.log_joint(target.x0) == start_value
    assert "OpFromGraph" in vbmc._pymc_target_change()
    with pytest.raises(RuntimeError, match=REFUSED):
        vbmc.optimize()
    vbmc.save(tmp_path / "swapped")
    assert _warnings(caplog)

    caplog.clear()
    loaded = VBMC.load(tmp_path / "swapped")
    assert loaded.target.log_joint(loaded.target.x0) == start_value
    assert loaded.target.log_joint(probe) != pytest.approx(probe_value)
    assert "OpFromGraph" in loaded._pymc_target_changed
    assert "starting point" not in loaded._pymc_target_changed
    assert _warnings(caplog)
    with pytest.raises(RuntimeError, match=REFUSED):
        loaded.optimize()


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
    # As a target pickled before the digest existed.
    del target._constants_digest
    vbmc = VBMC(target, options=OPTIONS, seed=931)
    assert vbmc._pymc_target_change() is None
    X_setup, y_setup = target.setup_evaluations
    target.setup_evaluations = (X_setup, y_setup + 1.0)
    vbmc.save(tmp_path / "run")
    assert not _warnings(caplog)

    loaded = VBMC.load(tmp_path / "run")
    assert "starting point" in loaded._pymc_target_changed
    assert "pytensor.scan" not in loaded._pymc_target_changed
