"""Saving and loading an :class:`~pyvbmc.svbmc.SVBMC` object.

The stacks are optimized with cheap settings, as in ``test_svbmc.py``:
what is checked is that the file restores the whole object, not the
quality of the stack.
"""

import copy
import logging
import sys

import matplotlib
import numpy as np
import pytest

pytest.importorskip("torch")

from pyvbmc.calibration.profile import CalibrationProfile  # noqa: E402
from pyvbmc.parameter_transformer import ParameterTransformer  # noqa: E402
from pyvbmc.svbmc import SVBMC  # noqa: E402
from pyvbmc.testing.svbmc._fixtures import load_group  # noqa: E402
from pyvbmc.variational_posterior import VariationalPosterior  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def quiet_logger():
    """Progress messages are checked in ``test_svbmc_filters.py``."""
    logger = logging.getLogger("SVBMC")
    previous = logger.level
    logger.setLevel(logging.WARNING)
    yield
    logger.setLevel(previous)


@pytest.fixture(scope="module")
def d2_vps():
    # A bounded problem, whose parameter transformers hold bounded
    # transforms: functions that a pickle would otherwise hold by value.
    return load_group("bounded_D2", rng=0)[0]


def _stack(vp_list, seed=0):
    """A stacked posterior optimized with cheap settings."""
    stacked = SVBMC(vp_list, seed=seed)
    stacked.optimize(n_samples=5, max_steps=3, n_samples_final=5)
    return stacked


def _assert_same(restored, original, where):
    """Assert that ``restored`` holds what ``original`` holds, recursively.

    The stack and its posteriors are compared attribute by attribute, so
    an attribute that a later version gives them is compared too. The
    transformer and the calibration profile compare with their own ``==``,
    the transformer's leaving out the bounded transforms that it rebuilds
    on loading. The generators compare by state, and the logger is the
    process's own.
    """
    if isinstance(original, logging.Logger):
        assert restored is original, where
    elif isinstance(original, np.random.Generator):
        assert type(restored) is type(original), where
        assert (
            restored.bit_generator.state == original.bit_generator.state
        ), where
    elif isinstance(original, (ParameterTransformer, CalibrationProfile)):
        assert type(restored) is type(original), where
        assert restored == original, where
    elif isinstance(original, (SVBMC, VariationalPosterior)):
        assert type(restored) is type(original), where
        assert vars(restored).keys() == vars(original).keys(), where
        for name, value in vars(original).items():
            _assert_same(getattr(restored, name), value, f"{where}.{name}")
    elif isinstance(original, np.ndarray):
        assert type(restored) is np.ndarray, where
        assert restored.dtype == original.dtype, where
        np.testing.assert_array_equal(restored, original, err_msg=where)
    elif isinstance(original, dict):
        assert type(restored) is dict, where
        assert restored.keys() == original.keys(), where
        for key, value in original.items():
            _assert_same(restored[key], value, f"{where}[{key!r}]")
    elif isinstance(original, (list, tuple)):
        assert type(restored) is type(original), where
        assert len(restored) == len(original), where
        for i, value in enumerate(original):
            _assert_same(restored[i], value, f"{where}[{i}]")
    else:
        assert type(restored) is type(original), where
        np.testing.assert_equal(restored, original, err_msg=where)


@pytest.mark.parametrize("group", ["bounded_D2", "corr_D3"])
def test_an_optimized_stack_round_trips(group, tmp_path):
    """The loaded object holds everything the saved one holds and draws
    what the saved one draws next. ``corr_D3`` stacks an unwarped run with
    two warped ones, each with its own rotation."""
    stacked = _stack(load_group(group, rng=0)[0])
    path = tmp_path / "stack.pkl"
    stacked.save(path)

    restored = SVBMC.load(path)
    _assert_same(restored, stacked, "svbmc")
    np.testing.assert_array_equal(restored.sample(64), stacked.sample(64))
    np.testing.assert_array_equal(
        restored.sample(64, balance_flag=True),
        stacked.sample(64, balance_flag=True),
    )


def test_a_loaded_stack_optimizes_as_the_saved_one(d2_vps, tmp_path):
    """A stack saved before :meth:`optimize` and loaded gives the result
    that the saved one gives: the file holds the starting weights, the
    corrected expected log-joints and the generator that the optimization
    reads."""
    stacked = SVBMC(d2_vps, seed=4)
    path = tmp_path / "stack.pkl"
    stacked.save(path)
    restored = SVBMC.load(path)
    assert restored.elbo is None and restored.elbo_details is None

    for s in (stacked, restored):
        s.optimize(n_samples=5, max_steps=3, n_samples_final=5)
    np.testing.assert_array_equal(restored.w, stacked.w)
    assert restored.elbo == stacked.elbo
    assert restored.elbo_sd == stacked.elbo_sd
    assert restored.entropy == stacked.entropy
    assert restored.elbo_details == stacked.elbo_details


def test_the_file_holds_no_function(d2_vps, tmp_path):
    """A saved stack can be loaded under another Python version than the
    one that wrote it: the file carries no function by value (dill marks
    one with ``_create_function``), and so no bytecode of any Python
    version."""
    path = tmp_path / "stack.pkl"
    _stack(d2_vps).save(path)

    data = path.read_bytes()
    assert b"_create_function" not in data
    assert b"_create_code" not in data


def test_file_name_and_overwrite_guard(d2_vps, tmp_path):
    stacked = _stack(d2_vps)

    stacked.save(tmp_path / "stack")
    assert (tmp_path / "stack.pkl").is_file()
    assert SVBMC.load(tmp_path / "stack").elbo == stacked.elbo

    with pytest.raises(FileExistsError):
        stacked.save(tmp_path / "stack.pkl")
    stacked.w = np.full_like(stacked.w, 1.0 / stacked.w.size)
    stacked.save(tmp_path / "stack.pkl", overwrite=True)
    np.testing.assert_array_equal(
        SVBMC.load(tmp_path / "stack.pkl").w, stacked.w
    )

    with pytest.raises(OSError):
        stacked.save(tmp_path / "missing" / "stack.pkl")
    with pytest.raises(OSError):
        SVBMC.load(tmp_path / "missing.pkl")


def test_load_refuses_another_object(d2_vps, tmp_path):
    path = tmp_path / "posterior.pkl"
    d2_vps[0].save(path)
    with pytest.raises(
        TypeError,
        match=r"variational_posterior\.VariationalPosterior, not "
        r"pyvbmc\.svbmc\.svbmc\.SVBMC",
    ):
        SVBMC.load(path)


def test_load_migrates_the_retained_posteriors(d2_vps, tmp_path):
    """A retained posterior saved before calibration profiles existed gets
    the profile that ``VariationalPosterior.load`` would give it."""
    stacked = SVBMC(d2_vps, seed=0)
    legacy = stacked.vp_list[1] = copy.deepcopy(stacked.vp_list[1])
    for name in (
        "_calibration_profile",
        "_calibration_request",
        "_calibration_hint_emitted",
    ):
        del legacy.__dict__[name]
    path = tmp_path / "stack.pkl"
    stacked.save(path)

    restored = SVBMC.load(path).vp_list[1]
    assert restored.__dict__["_calibration_profile"].source == "legacy"
    assert restored.__dict__["_calibration_request"] is None
    assert restored.__dict__["_calibration_hint_emitted"] is False


def test_a_loaded_stack_samples_without_torch(d2_vps, tmp_path, monkeypatch):
    """Only the methods that estimate the ELBO need torch, so a saved stack
    can be used where torch is not installed."""
    path = tmp_path / "stack.pkl"
    _stack(d2_vps).save(path)

    # An entry of None makes `import torch` raise ImportError.
    monkeypatch.setitem(sys.modules, "torch", None)
    restored = SVBMC.load(path)
    X = restored.sample(100)
    assert X.shape == (100, 2) and np.all(np.isfinite(X))

    from matplotlib import pyplot as plt

    previous = matplotlib.get_backend()
    plt.switch_backend("Agg")
    try:
        plt.close(restored.plot(n_samples=200))
    finally:
        plt.switch_backend(previous)

    with pytest.raises(ImportError, match=r"pyvbmc\[torch\]"):
        restored.optimize(n_samples=5, max_steps=3, n_samples_final=5)


def test_load_sets_up_the_logger(d2_vps, tmp_path):
    """The file names the logger but does not hold its level: a loaded
    stack shows its progress as a newly constructed one does, unless a
    level has been set."""
    path = tmp_path / "stack.pkl"
    SVBMC(d2_vps, seed=0).save(path)

    logger = logging.getLogger("SVBMC")
    previous = logger.level
    try:
        logger.setLevel(logging.NOTSET)
        assert SVBMC.load(path).logger.level == logging.INFO
        logger.setLevel(logging.ERROR)
        assert SVBMC.load(path).logger.level == logging.ERROR
    finally:
        logger.setLevel(previous)
