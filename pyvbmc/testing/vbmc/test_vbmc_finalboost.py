import copy
import logging

import numpy as np
import pytest

from pyvbmc import VBMC
from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.vbmc.variational_optimization import _neg_elcbo


def create_vbmc(
    D: int,
    x0: float,
    lower_bounds: float,
    upper_bounds: float,
    plausible_lower_bounds: float,
    plausible_upper_bounds: float,
    options: dict = None,
):
    fun = lambda x: np.sum(x + 2)
    lb = np.ones((1, D)) * lower_bounds
    ub = np.ones((1, D)) * upper_bounds
    x0_array = np.ones((2, D)) * x0
    plb = np.ones((1, D)) * plausible_lower_bounds
    pub = np.ones((1, D)) * plausible_upper_bounds
    return VBMC(fun, x0_array, lb, ub, plb, pub, options)


_DEFAULT = object()


def _boost_options(tolerance=_DEFAULT, **updates):
    options = {
        "min_final_components": 5,
        "ns_ent": 1300,
        "ns_ent_fast": 0,
        "ns_ent_fine": 204800,
        "ns_ent_boost": 1300,
        "ns_ent_fast_boost": 20,
        "ns_ent_fine_boost": 204800,
        "ns_elbo": 2500,
    }
    if tolerance is not _DEFAULT:
        options["tol_elcbo_boost"] = tolerance
    options.update(updates)
    return options


def _set_pre_stats(vbmc, elbo=-10.0, elbo_sd=0.1, stable=True):
    vbmc.vp.stats = {
        "elbo": elbo,
        "elbo_sd": elbo_sd,
        "stable": stable,
        "source_only": np.array([1.0, 2.0]),
    }
    vbmc.vp.mu.fill(1.0)


def _mock_candidate(
    mocker,
    vbmc,
    candidate_elbo,
    candidate_elbo_sd,
    candidate_k=7,
):
    candidate = VariationalPosterior(vbmc.D, candidate_k, rng=123)
    candidate.mu.fill(7.0)
    candidate.stats = {
        "elbo": candidate_elbo,
        "elbo_sd": candidate_elbo_sd,
        "stable": False,
    }
    captured = {}

    def fake_optimize(options, optim_state, vp, gp, *args):
        captured.update(
            options=options,
            optim_state=optim_state,
            vp=vp,
            gp=gp,
            args=args,
        )
        candidate.parameter_transformer = vp.parameter_transformer
        candidate.rng = vp.rng
        return candidate, None, None

    optimize = mocker.patch(
        "pyvbmc.vbmc.vbmc.optimize_vp", side_effect=fake_optimize
    )
    return candidate, captured, optimize


@pytest.mark.parametrize(
    "option_updates",
    [
        {
            "ns_ent": lambda K: 100 * K ** (2 / 3),
            "ns_ent_fast": lambda K: 0,
            "ns_ent_fine": lambda K: 2**12 * K,
            "ns_ent_boost": lambda K: 100 * K ** (2 / 3) - 10,
            "ns_ent_fast_boost": lambda K: 0,
            "ns_ent_fine_boost": lambda K: 2**12 * K,
            "ns_elbo": lambda K: 50 * K,
        },
        {},
        {
            "ns_ent_boost": [],
            "ns_ent_fast_boost": [],
            "ns_ent_fine_boost": [],
        },
    ],
)
def test_final_boost_entropy_options(mocker, option_updates):
    options = _boost_options(**option_updates)
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    _set_pre_stats(vbmc)
    _, captured, optimize = _mock_candidate(mocker, vbmc, -9.9, 0.1)

    vp, elbo, elbo_sd, changed_flag = vbmc.final_boost(vbmc.vp, object())

    assert changed_flag
    assert (elbo, elbo_sd) == (vp.stats["elbo"], vp.stats["elbo_sd"])
    assert optimize.call_count == 1
    assert vbmc.options["tol_elcbo_boost"] == 0.1
    assert captured["options"]["tol_weight"] == 0
    assert captured["options"]["weight_penalty"] == 0


def test_final_boost_no_boost_does_not_require_stable(mocker):
    options = _boost_options(
        0.1,
        min_final_components=1,
        ns_ent_fast_boost=[],
        ns_ent_boost=[],
        ns_ent_fine_boost=[],
    )
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    vbmc.vp.stats = {"elbo": -3.0, "elbo_sd": 0.1}
    optimize = mocker.patch("pyvbmc.vbmc.vbmc.optimize_vp")

    vp, elbo, elbo_sd, changed_flag = vbmc.final_boost(vbmc.vp, object())

    assert not changed_flag
    assert (elbo, elbo_sd) == (-3.0, 0.1)
    assert vp is not vbmc.vp
    optimize.assert_not_called()


@pytest.mark.parametrize(
    "pre,candidate,tolerance,accepted",
    [
        ((-10.0, 0.2), (-9.8, 0.1), 0.1, True),
        ((-10.0, 0.1), (-10.0, 0.1), 0.1, True),
        # The beta=0 endpoint is worst.
        ((-10.0, 0.2), (-10.11, 0.1), 0.1, False),
        # The beta=5 endpoint is worst.
        ((-10.0, 0.1), (-9.8, 0.17), 0.1, False),
        # Exactly representable strict boundaries at beta=0 and beta=5.
        ((-10.0, 0.125), (-10.125, 0.125), 0.125, False),
        ((0.0, 0.125), (0.5, 0.25), 0.125, False),
        # Historical destructive boost is rejected at both test tolerances.
        ((-10.340, 0.044), (-9.031, 0.494), 0.1, False),
        ((-10.340, 0.044), (-9.031, 0.494), 0.2, False),
    ],
)
def test_final_boost_guard_selects_posterior(
    mocker, caplog, pre, candidate, tolerance, accepted
):
    options = _boost_options(tolerance, weight_penalty=0.37, tol_weight=0.023)
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    _set_pre_stats(vbmc, *pre, stable=np.array([True]))
    source = vbmc.vp
    source_mu = source.mu.copy()
    source_stats = copy.deepcopy(source.stats)
    gp = object()
    candidate_vp, captured, optimize = _mock_candidate(
        mocker, vbmc, *candidate
    )
    rng_state = copy.deepcopy(vbmc.rng.bit_generator.state)

    with caplog.at_level(logging.WARNING, logger="VBMC"):
        vp, elbo, elbo_sd, changed_flag = vbmc.final_boost(source, gp)

    assert changed_flag is accepted
    expected_stats = candidate if accepted else pre
    assert (elbo, elbo_sd) == expected_stats
    assert (vp.stats["elbo"], vp.stats["elbo_sd"]) == expected_stats
    assert vp.K == (candidate_vp.K if accepted else source.K)
    assert np.all(vp.mu == (7.0 if accepted else 1.0))
    assert np.array_equal(vp.stats["stable"], source_stats["stable"])
    assert ("Final boost rejected" in caplog.text) is (not accepted)

    assert optimize.call_count == 1
    assert captured["gp"] is gp
    assert captured["optim_state"] is vbmc.optim_state
    assert captured["vp"] is not source
    assert captured["vp"].rng is source.rng
    assert (
        captured["vp"].parameter_transformer
        is not source.parameter_transformer
    )
    assert vp.rng is source.rng
    assert vbmc.rng.bit_generator.state == rng_state

    assert captured["options"]["tol_weight"] == 0
    assert captured["options"]["weight_penalty"] == 0
    assert vbmc.options["tol_weight"] == 0.023
    assert vbmc.options["weight_penalty"] == 0.37
    assert np.array_equal(source.mu, source_mu)
    assert source.stats.keys() == source_stats.keys()
    for key in source_stats:
        assert np.array_equal(source.stats[key], source_stats[key])


def test_final_boost_none_retains_legacy_objective(mocker):
    options = _boost_options(None, weight_penalty=0.37, tol_weight=0.023)
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    _set_pre_stats(vbmc)
    _, captured, _ = _mock_candidate(mocker, vbmc, np.nan, -1.0)

    vp, elbo, elbo_sd, changed_flag = vbmc.final_boost(vbmc.vp, object())

    assert changed_flag
    assert np.isnan(elbo)
    assert elbo_sd == -1.0
    assert vp.stats["elbo_sd"] == -1.0
    assert captured["options"]["tol_weight"] == 0
    assert captured["options"]["weight_penalty"] == 0.37


@pytest.mark.parametrize(
    "candidate",
    [(np.nan, 0.1), (-9.0, np.inf), (-9.0, -0.1)],
)
def test_final_boost_rejects_invalid_candidate(mocker, caplog, candidate):
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, _boost_options(0.1))
    _set_pre_stats(vbmc, -10.0, 0.1)
    _mock_candidate(mocker, vbmc, *candidate)

    with caplog.at_level(logging.WARNING, logger="VBMC"):
        vp, elbo, elbo_sd, changed_flag = vbmc.final_boost(vbmc.vp, object())

    assert not changed_flag
    assert (elbo, elbo_sd, vp.K) == (-10.0, 0.1, vbmc.vp.K)
    assert "Final boost rejected" in caplog.text


def test_final_boost_accepts_valid_candidate_if_pre_score_invalid(mocker):
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, _boost_options(0.1))
    _set_pre_stats(vbmc, np.nan, 0.1)
    candidate, _, _ = _mock_candidate(mocker, vbmc, -9.0, 0.2)

    vp, elbo, elbo_sd, changed_flag = vbmc.final_boost(vbmc.vp, object())

    assert changed_flag
    assert vp is candidate
    assert (elbo, elbo_sd) == (-9.0, 0.2)


def test_final_boost_raises_if_neither_score_is_valid(mocker):
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, _boost_options(0.1))
    _set_pre_stats(vbmc, np.nan, 0.1)
    _mock_candidate(mocker, vbmc, -9.0, -0.2)

    with pytest.raises(RuntimeError, match="no posterior with a finite ELBO"):
        vbmc.final_boost(vbmc.vp, object())


def test_final_boost_endpoint_rule_matches_dense_continuum():
    cases = [
        (-10.0, 0.1, -9.95, 0.11),
        (-10.0, 0.2, -10.08, 0.1),
        (-10.0, 0.1, -9.8, 0.17),
        (-10.340, 0.044, -9.031, 0.494),
    ]
    betas = np.linspace(0.0, 5.0, 10001)

    for pre_elbo, pre_sd, candidate_elbo, candidate_sd in cases:
        delta_elbo = candidate_elbo - pre_elbo
        delta_sd = candidate_sd - pre_sd
        for tolerance in (0.1, 0.2):
            dense_accept = np.all(delta_elbo - betas * delta_sd > -tolerance)
            endpoint_accept = VBMC._accept_final_boost_candidate(
                pre_elbo,
                pre_sd,
                candidate_elbo,
                candidate_sd,
                tolerance,
            )
            assert endpoint_accept is bool(dense_accept)


def test_final_boost_selected_tolerance_boundary_nextafter():
    tolerance = 0.1
    boundary = -tolerance
    below = np.nextafter(boundary, -np.inf)
    above = np.nextafter(boundary, np.inf)

    assert not VBMC._accept_final_boost_candidate(
        0.0, 0.0, below, 0.0, tolerance
    )
    assert not VBMC._accept_final_boost_candidate(
        0.0, 0.0, boundary, 0.0, tolerance
    )
    assert VBMC._accept_final_boost_candidate(0.0, 0.0, above, 0.0, tolerance)


@pytest.mark.parametrize(
    "tolerance",
    [-0.1, np.nan, np.inf, True, np.bool_(False), "0.1", [0.1], 0.1j],
)
def test_final_boost_tolerance_validation(tolerance):
    with pytest.raises(ValueError, match="tol_elcbo_boost"):
        create_vbmc(
            3,
            3,
            1,
            5,
            2,
            4,
            {"tol_elcbo_boost": tolerance},
        )


def test_final_boost_old_options_without_tolerance_use_legacy(mocker):
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, _boost_options(None))
    del vbmc.options["tol_elcbo_boost"]
    _set_pre_stats(vbmc)
    _, captured, _ = _mock_candidate(mocker, vbmc, -10.5, 0.5)

    _, _, _, changed_flag = vbmc.final_boost(vbmc.vp, object())

    assert changed_flag
    assert (
        captured["options"]["weight_penalty"] == vbmc.options["weight_penalty"]
    )


def test_guarded_boost_weight_regularization_is_zero(mocker):
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, _boost_options(0.1))
    _set_pre_stats(vbmc)
    _, captured, _ = _mock_candidate(mocker, vbmc, -9.9, 0.1)
    vbmc.final_boost(vbmc.vp, object())

    options = captured["options"]
    vp = captured["vp"]
    vp.optimize_weights = True
    vp.w = np.full((1, vp.K), 0.999 / (vp.K - 1))
    vp.w[0, 0] = 0.001
    vp.eta = np.log(vp.w)
    theta = vp.get_parameters()
    X = np.vstack((-np.ones(vp.D), np.ones(vp.D)))
    theta_bnd = vp.get_bounds(X, options)

    zero_gradient = np.zeros(theta.shape)
    mocker.patch(
        "pyvbmc.vbmc.variational_optimization._gp_log_joint",
        return_value=(0.0, zero_gradient, None, None, None),
    )
    mocker.patch(
        "pyvbmc.vbmc.variational_optimization.entlb_vbmc",
        return_value=(0.0, zero_gradient),
    )

    objective, gradient, *_ = _neg_elcbo(
        theta.copy(),
        object(),
        vp,
        beta=0.0,
        Ns=0,
        compute_grad=True,
        separate_K=False,
        theta_bnd=theta_bnd,
    )

    assert theta_bnd["weight_penalty"] == 0
    assert objective == 0
    assert np.array_equal(gradient, zero_gradient)
