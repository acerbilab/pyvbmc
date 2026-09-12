"""Statistical reporting and runtime-guidance contracts for S-VBMC.

The posteriors are deliberately synthetic.  These tests exercise the final
evaluation and uncertainty algebra without adding another VBMC optimization
run or depending on convergence of the S-VBMC weight optimizer.
"""

import copy
import importlib
import logging
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pyvbmc.svbmc import SVBMC  # noqa: E402
from pyvbmc.svbmc import _runtime_tips  # noqa: E402
from pyvbmc.svbmc._tip_catalog import TIPS  # noqa: E402
from pyvbmc.testing.svbmc._fixtures import load_vp  # noqa: E402
from pyvbmc.testing.svbmc.test_svbmc_filters import make_vp  # noqa: E402


def _run(*, weights=(1.0,), I=None, J=None, Ns=1, seed=0):
    """Return a one-dimensional fitted-looking posterior with exact stats."""
    weights = np.asarray(weights, dtype=np.float64)
    K = weights.size
    vp = make_vp(D=1, K=K, Ns=Ns, seed=seed)
    vp.w = (weights / weights.sum()).reshape(1, K)
    vp.mu = np.linspace(-0.5, 0.5, K, dtype=np.float64).reshape(1, K)
    vp.sigma = np.ones((1, K), dtype=np.float64)
    vp.lambd = np.ones((1, 1), dtype=np.float64)
    if I is None:
        I = np.zeros((Ns, K), dtype=np.float64)
    if J is None:
        J = np.zeros((Ns, K, K), dtype=np.float64)
    vp.stats["I_sk"] = np.asarray(I, dtype=np.float64)
    vp.stats["J_sjk"] = np.asarray(J, dtype=np.float64)
    vp.stats["elbo"] = float(np.mean(vp.stats["I_sk"]))
    return vp


@pytest.fixture(autouse=True)
def isolate_logger_and_tip_scheduler():
    """Keep this module independent of process-global logging and tip state."""
    logger = logging.getLogger("SVBMC")
    previous_level = logger.level
    logger.setLevel(logging.WARNING)

    saved = (
        _runtime_tips._RNG.getstate(),
        None if _runtime_tips._ORDER is None else list(_runtime_tips._ORDER),
        set(_runtime_tips._SEEN),
        _runtime_tips._ELIGIBLE_STARTS,
    )
    _runtime_tips._RNG.seed(1729)
    _runtime_tips._ORDER = None
    _runtime_tips._SEEN = set()
    _runtime_tips._ELIGIBLE_STARTS = 0
    yield
    logger.setLevel(previous_level)
    _runtime_tips._RNG.setstate(saved[0])
    _runtime_tips._ORDER = saved[1]
    _runtime_tips._SEEN = saved[2]
    _runtime_tips._ELIGIBLE_STARTS = saved[3]


def _fixed_final_evaluation(monkeypatch, stacked, selected, entropy=0.0):
    """Replace optimization and sampling while retaining reporting logic."""
    calls = []

    def maximize(**kwargs):
        return (
            torch.as_tensor(selected, dtype=torch.float64),
            torch.tensor(-999.0, dtype=torch.float64),
            torch.tensor(-888.0, dtype=torch.float64),
        )

    def evaluate(w, n_samples, *, compute_variance=False):
        calls.append(
            (
                np.asarray(w.detach().cpu(), dtype=np.float64).ravel().copy(),
                n_samples,
                compute_variance,
            )
        )
        return (
            torch.tensor(float(entropy), dtype=torch.float64),
            stacked._jacobian_corrections.copy(),
            0.09 if compute_variance else None,
        )

    monkeypatch.setattr(stacked, "maximize_ELBO", maximize)
    monkeypatch.setattr(stacked, "_stacked_entropy", evaluate)
    return calls


def test_gp_uncertainty_is_total_blockwise_covariance():
    """Within-run off-diagonals and hyperparameter spread both contribute."""
    I_a = np.array([[1.0, 4.0], [3.0, 0.0], [2.0, 5.0]])
    J_a = np.array(
        [
            [[0.40, 0.12], [0.12, 0.90]],
            [[0.30, -0.08], [-0.08, 0.60]],
            [[0.50, 0.20], [0.20, 0.70]],
        ]
    )
    I_b = np.array([[7.0]])  # Ns=1: no between-hyperparameter term.
    J_b = np.array([[[0.25]]])
    stacked = SVBMC(
        [
            _run(weights=(0.4, 0.6), I=I_a, J=J_a, Ns=3, seed=1),
            _run(I=I_b, J=J_b, Ns=1, seed=2),
        ],
        M_min=2,
        noisy=False,
        seed=3,
    )

    w = np.array([0.20, 0.30, 0.50])
    wa, wb = w[:2], w[2:]
    expected_a = np.mean([wa @ covariance @ wa for covariance in J_a])
    expected_a += np.var(I_a @ wa, ddof=1)
    expected_b = wb @ J_b[0] @ wb
    expected = expected_a + expected_b

    assert stacked._expected_log_joint_variance(w) == pytest.approx(expected)
    # A run with zero global mass contributes exactly zero, including its
    # diagonal numerical floor.
    zero_a = stacked._expected_log_joint_variance([0.0, 0.0, 1.0])
    assert zero_a == pytest.approx(J_b[0, 0, 0])
    diagonal_only = np.mean(
        [np.dot(wa**2, np.diag(covariance)) for covariance in J_a]
    )
    diagonal_only += np.var(I_a @ wa, ddof=1) + expected_b
    assert expected != pytest.approx(diagonal_only)


def test_entropy_variance_matches_fixed_stratified_draws():
    """Check the estimator against normal log-density algebra."""
    stacked = SVBMC(
        [_run(weights=(0.25, 0.75), Ns=1)], M_min=1, noisy=False, seed=4
    )
    sigma = 1.7
    stacked.vp_list[0].mu[:] = 0.3
    stacked.vp_list[0].sigma[:] = sigma
    z_by_component = (
        np.array([[-2.0], [-0.5], [0.25], [1.0]]),
        np.array([[-1.5], [0.0], [0.75], [2.5]]),
    )

    class FixedNormalDraws:
        def __init__(self, draws):
            self.draws = iter(draws)

        def standard_normal(self, shape):
            result = next(self.draws).copy()
            assert result.shape == shape
            return result

    stacked.rng = FixedNormalDraws(z_by_component)
    weights = np.array([0.25, 0.75])
    H, _, variance = stacked._stacked_entropy(
        torch.as_tensor(weights, dtype=torch.float64),
        4,
        compute_variance=True,
    )

    normalizer = 0.5 * np.log(2.0 * np.pi * sigma**2)
    losses = [0.5 * np.ravel(z) ** 2 for z in z_by_component]
    expected_H = normalizer + sum(
        weight * np.mean(loss) for weight, loss in zip(weights, losses)
    )
    expected_variance = sum(
        weight**2 * np.var(loss, ddof=1) / 4
        for weight, loss in zip(weights, losses)
    )
    assert H.item() == pytest.approx(expected_H)
    assert variance == pytest.approx(expected_variance)


def test_final_report_uses_fresh_count_and_returned_weights(monkeypatch):
    stacked = SVBMC(
        [_run(I=[[2.0]], seed=1), _run(I=[[8.0]], seed=2)],
        M_min=2,
        noisy=False,
        seed=5,
    )
    stacked.I_corrected = np.array([[2.0, 8.0]])
    stacked.E_corrected = np.array([2.0, 8.0])
    calls = _fixed_final_evaluation(
        monkeypatch, stacked, selected=[0.75, 0.25], entropy=1.5
    )

    stacked.optimize(n_samples=3, max_steps=1, n_samples_final=17)

    np.testing.assert_array_equal(stacked.w, [[0.75, 0.25]])
    np.testing.assert_array_equal(calls[0][0], [0.75, 0.25])
    assert calls[0][1:] == (17, True)
    np.testing.assert_array_equal(calls[1][0], stacked._naive_weights)
    assert calls[1][1:] == (17, False)
    assert stacked.entropy == 1.5
    assert stacked.elbo_details["raw"] == pytest.approx(5.0)
    assert stacked.elbo_details["entropy_sd"] == pytest.approx(0.3)
    assert stacked.elbo_details["raw_sd"] == stacked.elbo_sd


def test_naive_mode_reuses_final_evaluation_and_original_weights(monkeypatch):
    runs = [
        _run(weights=(0.8, 0.2), Ns=1, seed=1),
        _run(weights=(0.1, 0.9), Ns=1, seed=2),
    ]
    stacked = SVBMC(runs, M_min=2, noisy=False, seed=6)
    expected = np.array([0.4, 0.1, 0.05, 0.45])
    calls = []

    def evaluate(w, n_samples, *, compute_variance=False):
        calls.append((np.asarray(w).ravel().copy(), compute_variance))
        value = float(len(calls))
        return torch.tensor(value), stacked._jacobian_corrections.copy(), 0.0

    monkeypatch.setattr(stacked, "_stacked_entropy", evaluate)
    stacked.w = np.array([[0.0, 0.0, 1.0, 0.0]])
    stacked.optimize(version="ns", n_samples=2, max_steps=1, n_samples_final=3)

    np.testing.assert_array_equal(stacked.w.ravel(), expected)
    assert [variance for _, variance in calls] == [False, True]
    np.testing.assert_array_equal(calls[0][0], expected)
    np.testing.assert_array_equal(calls[1][0], expected)
    assert stacked.elbo_details["raw"] == stacked.elbo_details["naive"]

    stacked.w = np.array([[1.0, 0.0, 0.0, 0.0]])
    stacked.optimize(version="ns", n_samples=2, max_steps=1, n_samples_final=3)
    np.testing.assert_array_equal(stacked.w.ravel(), expected)
    assert stacked.elbo_details["raw"] == stacked.elbo_details["naive"]


@pytest.mark.parametrize(
    "noisy, method, expected_elbo, expected_cap",
    [
        (True, "capped_I_median", 6.0, 5.0),
        (False, "raw", 11.0, 0.0),
    ],
)
def test_cap_is_headline_only_for_noisy_stacks(
    monkeypatch, caplog, noisy, method, expected_elbo, expected_cap
):
    stacked = SVBMC(
        [_run(I=[[0.0]], seed=1), _run(I=[[10.0]], seed=2)],
        M_min=2,
        noisy=noisy,
        show_tips=False,
        seed=7,
    )
    stacked.I_corrected = np.array([[0.0, 10.0]])
    stacked.E_corrected = np.array([0.0, 10.0])
    _fixed_final_evaluation(
        monkeypatch, stacked, selected=[0.0, 1.0], entropy=1.0
    )
    caplog.set_level(logging.INFO, logger="SVBMC")

    stacked.optimize(n_samples=2, max_steps=1, n_samples_final=3)

    details = stacked.elbo_details
    assert stacked.elbo == pytest.approx(expected_elbo)
    assert details["raw"] == pytest.approx(11.0)
    assert details["capped_I_median"] == pytest.approx(6.0)
    assert details["capped_E_median"] == pytest.approx(6.0)
    assert details["headline_method"] == method
    assert details["cap_amount"] == pytest.approx(expected_cap)
    assert details["noisy"] is noisy
    assert details["gp_sd"] == pytest.approx(np.sqrt(np.spacing(1)))
    assert stacked.elbo_sd == pytest.approx(
        np.hypot(details["entropy_sd"], details["gp_sd"])
    )
    cap_messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "SVBMC" and "capped by" in record.getMessage()
    ]
    assert bool(cap_messages) is noisy


def test_noise_metadata_precedes_proxy_and_only_retained_runs_count():
    recorded_quiet = _run(seed=1)
    recorded_quiet.stats.update(uncertainty_handling_level=0, elbo_sd=100.0)
    inferred_noisy = _run(seed=2)
    inferred_noisy.stats["elbo_sd"] = 0.10001
    dropped = _run(seed=3)
    dropped.stats["stable"] = False
    dropped.stats.pop("elbo_sd")

    stacked = SVBMC([recorded_quiet, inferred_noisy, dropped], M_min=2, seed=8)

    assert stacked.noisy is True
    assert stacked.noise_status_source == ("recorded", "inferred")
    assert stacked.vp_list == [recorded_quiet, inferred_noisy]


@pytest.mark.parametrize("override", [False, True])
def test_noise_override_needs_no_proxy_and_records_provenance(override):
    runs = [_run(seed=1), _run(seed=2)]
    for vp in runs:
        vp.stats.pop("elbo_sd")
    stacked = SVBMC(runs, M_min=2, noisy=override, seed=9)
    assert stacked.noisy is override
    assert stacked.noise_status_source == ("override", "override")


@pytest.mark.parametrize(
    "level, proxy, expected_noisy",
    [(0, 10.0, False), (1, 0.0, True), (2, 0.0, True)],
)
def test_recorded_noise_metadata_takes_precedence_over_proxy(
    level, proxy, expected_noisy
):
    vp = _run(seed=1)
    vp.stats["uncertainty_handling_level"] = level
    vp.stats["elbo_sd"] = proxy
    stacked = SVBMC([vp], M_min=1, seed=9)
    assert stacked.noisy is expected_noisy
    assert stacked.noise_status_source == ("recorded",)


@pytest.mark.parametrize(
    "fixture_name, expected_noisy",
    [("upstream_GMM_00", False), ("upstream_GMM_noisy_00", True)],
)
def test_legacy_fixture_noise_classification_uses_elbo_sd(
    fixture_name, expected_noisy
):
    vp, _ = load_vp(fixture_name, rng=0)
    vp.stats.pop("uncertainty_handling_level", None)
    stacked = SVBMC([vp], M_min=1, seed=10)
    assert stacked.noisy is expected_noisy
    assert stacked.noise_status_source == ("inferred",)


@pytest.mark.parametrize("proxy", [None, np.nan, -0.01])
def test_missing_or_invalid_legacy_noise_proxy_requires_override(proxy):
    vp = _run(seed=1)
    if proxy is None:
        vp.stats.pop("elbo_sd")
    else:
        vp.stats["elbo_sd"] = proxy
    with pytest.raises(ValueError, match="finite, nonnegative `elbo_sd`"):
        SVBMC([vp], M_min=1, seed=11)


def test_constructor_caches_corrections_without_rng_draws():
    runs = [_run(weights=(0.3, 0.7), seed=1), _run(seed=2)]
    input_states = [copy.deepcopy(vp.rng.bit_generator.state) for vp in runs]
    stacked = SVBMC(runs, M_min=2, noisy=False, seed=123)

    assert (
        stacked.rng.bit_generator.state
        == np.random.default_rng(123).bit_generator.state
    )
    for vp, state in zip(runs, input_states):
        assert vp.rng.bit_generator.state == state

    cached = stacked._jacobian_corrections.copy()
    _, first = stacked.stacked_entropy(
        torch.as_tensor(stacked.w.ravel()), n_samples=2
    )
    _, second = stacked.stacked_entropy(
        torch.as_tensor(stacked.w.ravel()), n_samples=2
    )
    np.testing.assert_array_equal(first, cached)
    np.testing.assert_array_equal(second, cached)
    np.testing.assert_array_equal(stacked._jacobian_corrections, cached)


@pytest.mark.parametrize("bad_count", [1, True, 2.5])
def test_invalid_final_sample_count_fails_before_draws_or_optimization(
    monkeypatch, bad_count
):
    stacked = SVBMC([_run(seed=1)], M_min=1, noisy=False, seed=12)
    state = copy.deepcopy(stacked.rng.bit_generator.state)
    monkeypatch.setattr(
        stacked,
        "maximize_ELBO",
        lambda **kwargs: pytest.fail("optimization started"),
    )
    with pytest.raises(ValueError, match="integer >= 2"):
        stacked.optimize(n_samples_final=bad_count)
    assert stacked.rng.bit_generator.state == state


@pytest.mark.parametrize(
    "kwargs", [{"version": "invalid"}, {"max_steps": 0}, {"n_samples": 0}]
)
def test_invalid_optimization_does_not_consume_tip_or_rng(kwargs, capsys):
    stacked = SVBMC([_run()], M_min=1, noisy=False, seed=12)
    stacked.logger.setLevel(logging.INFO)
    state = copy.deepcopy(stacked.rng.bit_generator.state)
    with pytest.raises(ValueError):
        stacked.optimize(**kwargs)
    assert not _runtime_tips._SEEN
    assert _runtime_tips._ELIGIBLE_STARTS == 0
    assert stacked.rng.bit_generator.state == state
    assert "Tip:" not in capsys.readouterr().out


def _recording_emitter(output):
    def emit(message, *, display):
        output.append((message, display))
        return bool(display)

    return emit


def test_tip_cadence_noisy_priority_and_once_only():
    _runtime_tips._ORDER = [TIPS[1], TIPS[2], TIPS[0]]
    output = []
    results = [
        _runtime_tips.consider_runtime_tip(
            enabled=True,
            display=True,
            noisy=True,
            emitter=_recording_emitter(output),
        )
        for _ in range(7)
    ]
    emitted = [(index, tip) for index, tip in enumerate(results, 1) if tip]

    assert [index for index, _ in emitted] == [1, 4, 7]
    assert emitted[0][1].id == "noisy_elbo"
    assert len({tip.id for _, tip in emitted}) == 3
    assert len(output) == 3
    assert all(message.startswith("Tip: ") for message, _ in output)
    assert (
        _runtime_tips.consider_runtime_tip(
            enabled=True,
            display=True,
            noisy=True,
            emitter=_recording_emitter(output),
        )
        is None
    )


def test_disabled_or_non_info_tips_do_not_consume_cadence():
    output = []
    emit = _recording_emitter(output)
    assert (
        _runtime_tips.consider_runtime_tip(
            enabled=False, display=True, noisy=True, emitter=emit
        )
        is None
    )
    assert (
        _runtime_tips.consider_runtime_tip(
            enabled=True, display=False, noisy=True, emitter=emit
        )
        is None
    )
    assert _runtime_tips._ELIGIBLE_STARTS == 0
    first = _runtime_tips.consider_runtime_tip(
        enabled=True, display=True, noisy=True, emitter=emit
    )
    assert first is not None and first.id == "noisy_elbo"


def test_noisy_only_tip_is_ineligible_for_noiseless_stack():
    _runtime_tips._ORDER = [TIPS[0], TIPS[1], TIPS[2]]
    emitted = _runtime_tips.consider_runtime_tip(
        enabled=True,
        display=True,
        noisy=False,
        emitter=lambda *args, **kwargs: True,
    )
    assert emitted is not None
    assert emitted.noisy_only is False
    assert "noisy_elbo" not in _runtime_tips._SEEN


@pytest.mark.parametrize(
    "show_tips, level, expected",
    [
        (True, logging.INFO, (True, True)),
        (True, logging.WARNING, (True, False)),
        (False, logging.INFO, (False, True)),
    ],
)
def test_optimize_maps_show_tips_and_info_logging_to_scheduler(
    monkeypatch, show_tips, level, expected
):
    module = importlib.import_module("pyvbmc.svbmc.svbmc")
    calls = []
    monkeypatch.setattr(
        module, "consider_runtime_tip", lambda **kwargs: calls.append(kwargs)
    )
    stacked = SVBMC(
        [_run(seed=1)],
        M_min=1,
        noisy=False,
        show_tips=show_tips,
        seed=13,
    )
    stacked.logger.setLevel(level)
    _fixed_final_evaluation(monkeypatch, stacked, selected=[1.0])

    stacked.optimize(n_samples=2, max_steps=1, n_samples_final=2)

    assert len(calls) == 1
    assert (calls[0]["enabled"], calls[0]["display"]) == expected


def test_tip_scheduler_does_not_touch_numerical_or_module_rngs():
    inference_rng = np.random.default_rng(91)
    inference_state = copy.deepcopy(inference_rng.bit_generator.state)
    numpy_state = np.random.get_state()
    stdlib_state = random.getstate()

    for _ in range(7):
        _runtime_tips.consider_runtime_tip(
            enabled=True,
            display=True,
            noisy=True,
            emitter=lambda *args, **kwargs: True,
        )

    assert inference_rng.bit_generator.state == inference_state
    after_numpy = np.random.get_state()
    assert numpy_state[0] == after_numpy[0]
    np.testing.assert_array_equal(numpy_state[1], after_numpy[1])
    assert numpy_state[2:] == after_numpy[2:]
    assert random.getstate() == stdlib_state
