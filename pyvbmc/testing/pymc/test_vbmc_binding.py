"""Direct ``PyMCTarget`` binding to :class:`pyvbmc.VBMC`."""

import copy

import numpy as np
import pytest

from pyvbmc import VBMC
from pyvbmc.testing._dtype import non_float64_leaves


class _FakeTarget:
    """Small adapter-shaped object used without importing PyMC."""

    def __init__(self):
        self.D = 2
        self.x0 = np.array([[0.0, 0.0]], dtype=np.float64)
        self.lb = np.array([[-2.0, -3.0]], dtype=np.float64)
        self.ub = np.array([[2.0, 3.0]], dtype=np.float64)
        self.plb = np.array([[-1.0, -1.5]], dtype=np.float64)
        self.pub = np.array([[1.0, 1.5]], dtype=np.float64)
        self.setup_evaluations = (
            np.array([[0.0, 0.0], [0.5, -0.5]], dtype=np.float64),
            np.array([0.0, -0.5], dtype=np.float64),
        )
        self.setup_cost = 3
        self.calls = 0
        # A backend-owned non-float64 leaf must remain behind the bound
        # method boundary in the VBMC numerical-state walk.
        self.backend_buffer = np.ones(2, dtype=np.float32)

    def log_joint(self, x):
        self.calls += 1
        return -float(np.sum(np.asarray(x, dtype=np.float64) ** 2))


@pytest.fixture
def target(monkeypatch):
    import pyvbmc.pymc._target as target_module

    monkeypatch.setattr(target_module, "PyMCTarget", _FakeTarget)
    return _FakeTarget()


def _options(**overrides):
    options = {
        "display": "off",
        "fun_eval_start": 4,
        "max_fun_evals": 20,
        "min_iter": 0,
    }
    options.update(overrides)
    return options


def _manual(target, *, seed=12, **kwargs):
    return VBMC(
        target.log_joint,
        target.x0,
        target.lb,
        target.ub,
        target.plb,
        target.pub,
        options=_options(),
        seed=seed,
        precomputed_evaluations=target.setup_evaluations,
        initialization_cost=target.setup_cost,
        **kwargs,
    )


def _assert_same_initial_state(left, right):
    for name in (
        "x0",
        "lower_bounds",
        "upper_bounds",
        "plausible_lower_bounds",
        "plausible_upper_bounds",
    ):
        assert np.array_equal(getattr(left, name), getattr(right, name))
    for name in ("mu", "sigma", "lambd", "w", "eta"):
        assert np.array_equal(getattr(left.vp, name), getattr(right.vp, name))
    assert left.rng.bit_generator.state == right.rng.bit_generator.state
    assert left.initialization_cost == right.initialization_cost
    assert left._effective_max_fun_evals == right._effective_max_fun_evals
    left_rows = left.function_logger.X_flag
    right_rows = right.function_logger.X_flag
    assert np.array_equal(
        left.function_logger.X_orig[left_rows],
        right.function_logger.X_orig[right_rows],
    )
    assert np.array_equal(
        left.function_logger.y_orig[left_rows],
        right.function_logger.y_orig[right_rows],
    )


def test_direct_target_matches_manually_unpacked_form(target):
    direct = VBMC(target, options=_options(), seed=12)
    manual = _manual(target)

    _assert_same_initial_state(direct, manual)
    assert direct.target is target
    assert manual.target is None
    assert direct.log_joint.__self__ is target
    assert direct.function_logger.fun.__self__ is target
    assert target.calls == 0


def test_direct_target_overrides_and_reuse_controls(target):
    x0 = np.array([[0.25, -0.25]])
    plb = np.array([[-0.75, -1.0]])
    pub = np.array([[0.75, 1.0]])
    direct = VBMC(
        target,
        x0=x0,
        plausible_lower_bounds=plb,
        plausible_upper_bounds=pub,
        options=_options(),
        seed=4,
        precomputed_evaluations=None,
        initialization_cost=0,
    )

    assert np.array_equal(direct.parameter_transformer.inverse(direct.x0), x0)
    assert np.array_equal(direct.plausible_lower_bounds, plb)
    assert np.array_equal(direct.plausible_upper_bounds, pub)
    assert direct.precomputed_evaluations is None
    assert direct.precomputed_observation_count == 0
    assert direct.initialization_cost == 0
    assert direct._effective_max_fun_evals == 20
    assert target.calls == 0


def test_callable_target_subclass_keeps_adapter_defaults(target):
    class CallableTarget(_FakeTarget):
        __call__ = _FakeTarget.log_joint

    callable_target = CallableTarget()
    direct = VBMC(callable_target, options=_options(), seed=12)
    assert direct.target is callable_target
    assert direct.initialization_cost == callable_target.setup_cost
    assert direct.precomputed_observation_count == 2
    assert callable_target.calls == 0


def test_disabling_reuse_retains_target_setup_cost(target):
    direct = VBMC(
        target,
        options=_options(),
        seed=4,
        precomputed_evaluations=None,
    )
    assert direct.precomputed_evaluations is None
    assert direct.precomputed_observation_count == 0
    assert direct.initialization_cost == target.setup_cost
    assert direct._effective_max_fun_evals == 20 - target.setup_cost
    assert np.count_nonzero(direct.function_logger.X_flag) == 0
    assert target.calls == 0


def test_zero_setup_cost_retains_target_observations(target):
    direct = VBMC(
        target,
        options=_options(),
        seed=4,
        initialization_cost=0,
    )
    for actual, expected in zip(
        direct.precomputed_evaluations, target.setup_evaluations
    ):
        np.testing.assert_array_equal(actual, expected)
    assert direct.precomputed_observation_count == 2
    assert direct.initialization_cost == 0
    assert direct._effective_max_fun_evals == 20
    assert np.count_nonzero(direct.function_logger.X_flag) == 2
    assert target.calls == 0


def test_direct_target_hard_bounds_must_match_support(target):
    accepted = VBMC(
        target,
        lower_bounds=target.lb.ravel().astype(np.int64),
        upper_bounds=target.ub.ravel().astype(np.int64),
        options=_options(),
        seed=2,
    )
    assert np.array_equal(accepted.lower_bounds, target.lb)
    assert np.array_equal(accepted.upper_bounds, target.ub)

    with pytest.raises(ValueError, match="lower_bounds.*model support"):
        VBMC(
            target,
            lower_bounds=np.array([[-2.1, -3.0]]),
            options=_options(),
        )
    with pytest.raises(ValueError, match="upper_bounds.*model support"):
        VBMC(
            target,
            upper_bounds=np.array([[2.0, 3.1]]),
            options=_options(),
        )
    assert target.calls == 0


@pytest.mark.parametrize(
    "keyword,value",
    [
        ("prior", object()),
        ("log_prior", lambda x: 0.0),
    ],
)
def test_direct_target_rejects_separate_prior_inputs(target, keyword, value):
    with pytest.raises(ValueError, match="already includes.*prior"):
        VBMC(target, options=_options(), **{keyword: value})
    assert target.calls == 0


def test_direct_target_accepts_sample_prior(target):
    sample_prior = lambda n: np.zeros((n, 2))
    vbmc = VBMC(
        target,
        options=_options(),
        sample_prior=sample_prior,
        seed=2,
    )
    assert vbmc.target is target
    assert vbmc.prior.log_pdf is None
    assert vbmc.prior.sample is sample_prior
    assert target.calls == 0


def test_direct_target_is_reusable_and_target_property_is_read_only(target):
    before = copy.deepcopy(target.__dict__)
    first = VBMC(target, options=_options(), seed=8)
    second = VBMC(target, options=_options(), seed=8)

    _assert_same_initial_state(first, second)
    for name, value in before.items():
        if isinstance(value, np.ndarray):
            assert np.array_equal(target.__dict__[name], value)
        elif isinstance(value, tuple):
            for actual, expected in zip(target.__dict__[name], value):
                assert np.array_equal(actual, expected)
        else:
            assert target.__dict__[name] == value
    with pytest.raises(AttributeError):
        first.target = target


def test_direct_target_rejects_vectorized_target_option(target):
    with pytest.raises(
        ValueError, match="does not support.*vectorized_target"
    ):
        VBMC(target, options=_options(vectorized_target=True))
    assert target.calls == 0


def test_target_association_survives_logger_copy_and_save_load(
    target, tmp_path
):
    vbmc = VBMC(target, options=_options(), seed=7)
    logger_copy = copy.deepcopy(vbmc.function_logger)
    assert logger_copy.fun.__self__ is target

    offenders, _ = non_float64_leaves(vbmc)
    assert not any("backend_buffer" in path for path, _ in offenders)

    path = tmp_path / "pymc_target"
    vbmc.save(path)
    loaded = VBMC.load(path)
    assert isinstance(loaded.target, _FakeTarget)
    assert loaded.log_joint.__self__ is loaded.target
    assert loaded.function_logger.fun.__self__ is loaded.target


def test_ordinary_and_legacy_instances_have_no_target(tmp_path):
    ordinary = VBMC(
        np.sum,
        np.array([[0.0]]),
        np.array([[-2.0]]),
        np.array([[2.0]]),
        np.array([[-1.0]]),
        np.array([[1.0]]),
        options=_options(),
        seed=3,
    )
    assert ordinary.target is None
    ordinary.__dict__.pop("_uses_pymc_target", None)
    path = tmp_path / "legacy"
    ordinary.save(path)
    assert VBMC.load(path).target is None
