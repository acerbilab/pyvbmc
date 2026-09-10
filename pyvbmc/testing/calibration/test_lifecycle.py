import copy
import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import gpyreg as gpr
import numpy as np
import pytest

from pyvbmc import VBMC
from pyvbmc.calibration.profile import CalibrationProfile
from pyvbmc.entropy.entmc_vbmc import entmc_vbmc
from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.vbmc.variational_optimization import _candidate_vp
from pyvbmc.whitening import warp_gp_and_vp


def _profile(value, *, source="explicit"):
    return CalibrationProfile(
        pdf_chunk_elements=value,
        entropy_grad_chunk_elements=value + 1,
        entropy_value_chunk_elements=value + 2,
        source=source,
        provenance={"label": source},
    )


def _make_vbmc(calibration="off"):
    return VBMC(
        lambda x: -float(np.sum(np.asarray(x) ** 2)),
        np.zeros((1, 2)),
        np.full((1, 2), -5.0),
        np.full((1, 2), 5.0),
        np.full((1, 2), -2.0),
        np.full((1, 2), 2.0),
        options={
            "display": "off",
            "performance_calibration": calibration,
        },
        seed=3,
    )


def test_standalone_defaults_and_cached_resolution_once(monkeypatch):
    standalone = VariationalPosterior(2, calibration=None, rng=1)
    assert standalone.calibration_profile.uses_historical_defaults

    selected = _profile(97, source="cache")
    calls = []
    cache = importlib.import_module("pyvbmc.calibration._cache")
    monkeypatch.setattr(
        cache,
        "resolve_cached_profile",
        lambda: calls.append("resolve") or selected,
    )
    monkeypatch.setattr(
        cache, "suggest_calibration_once", lambda *a, **k: False
    )
    vp = VariationalPosterior(2, calibration="cached", rng=2)

    assert vp.calibration_profile is None
    vp.pdf(np.zeros((1, 2)), orig_flag=False)
    vp.pdf(np.ones((1, 2)), orig_flag=False)

    assert calls == ["resolve"]
    assert vp.calibration_profile is selected


def test_public_kernels_select_profile_budgets(monkeypatch):
    profile = _profile(101)
    vp = VariationalPosterior(2, calibration=profile, rng=1)
    pdf_budgets = []

    def fake_pdf(*args, chunk_elements, **kwargs):
        pdf_budgets.append(chunk_elements)
        return np.ones((1, 1))

    monkeypatch.setattr(vp, "_pdf", fake_pdf)
    vp.pdf(np.zeros((1, 2)), orig_flag=False)
    assert pdf_budgets == [profile.pdf_chunk_elements]

    entropy_module = importlib.import_module("pyvbmc.entropy.entmc_vbmc")
    entropy_budgets = []

    def fake_entropy(*args, budget, **kwargs):
        entropy_budgets.append(budget)
        return 0.0, np.empty(0)

    monkeypatch.setattr(entropy_module, "_entmc_vbmc", fake_entropy)
    entmc_vbmc(vp, 2, grad_flags=(False,) * 4)
    entmc_vbmc(vp, 2, grad_flags=(False, True, False, False))
    assert entropy_budgets == [
        profile.entropy_value_chunk_elements,
        profile.entropy_grad_chunk_elements,
    ]


def test_profile_survives_deepcopy_and_candidate_shell():
    profile = _profile(103)
    vp = VariationalPosterior(2, calibration=profile, rng=1)

    assert copy.deepcopy(vp).calibration_profile is profile
    assert _candidate_vp(vp).calibration_profile is profile


def test_profile_survives_whitening_and_undo_snapshot():
    profile = _profile(104)
    vbmc = _make_vbmc(profile)
    snapshot = copy.deepcopy(vbmc.vp)

    covariance = gpr.covariance_functions.SquaredExponential()
    noise = gpr.noise_functions.GaussianNoise(constant_add=True)
    mean = gpr.mean_functions.ConstantMean()
    hyperparameter_count = (
        covariance.hyperparameter_count(vbmc.D)
        + noise.hyperparameter_count()
        + mean.hyperparameter_count(vbmc.D)
    )
    gp = SimpleNamespace(
        X=np.zeros((2, vbmc.D)),
        covariance=covariance,
        noise=noise,
        mean=mean,
        posteriors=[SimpleNamespace(hyp=np.zeros(hyperparameter_count))],
    )
    transformer = copy.deepcopy(vbmc.parameter_transformer)

    warped, _ = warp_gp_and_vp(transformer, gp, vbmc.vp, vbmc)
    assert warped.calibration_profile is profile

    # The optimization loop's undo branch restores the pre-warp deepcopy.
    vbmc.vp = warped
    vbmc.vp = snapshot
    assert vbmc.vp.calibration_profile is profile


@pytest.mark.parametrize(
    ("mode", "candidate_elbo", "expected_changed"),
    [
        ("accepted", -9.9, True),
        ("rejected", -20.0, False),
        ("skipped", 0, False),
    ],
)
def test_profile_survives_final_boost_paths(
    monkeypatch, mode, candidate_elbo, expected_changed
):
    profile = _profile(105)
    vbmc = _make_vbmc(profile)
    vbmc.vp.stats = {"elbo": -10.0, "elbo_sd": 0.1, "stable": True}
    vbmc.options.__setitem__("tol_elcbo_boost", 0.1, force=True)

    if mode == "skipped":
        vbmc.options.__setitem__("min_final_components", 1, force=True)
        for name in (
            "ns_ent_boost",
            "ns_ent_fast_boost",
            "ns_ent_fine_boost",
        ):
            vbmc.options.__setitem__(name, [], force=True)
    else:
        vbmc.options.__setitem__("min_final_components", 5, force=True)

    def fake_optimize(options, optim_state, vp, gp, *args):
        assert vp.calibration_profile is profile
        candidate = copy.deepcopy(vp)
        candidate.stats = {
            "elbo": candidate_elbo,
            "elbo_sd": 0.1,
            "stable": False,
        }
        return candidate, None, None

    optimizer = Mock(side_effect=fake_optimize)
    module = importlib.import_module("pyvbmc.vbmc.vbmc")
    monkeypatch.setattr(module, "optimize_vp", optimizer)

    returned, _, _, changed = vbmc.final_boost(vbmc.vp, object())

    assert changed is expected_changed
    assert returned.calibration_profile is profile
    if mode == "skipped":
        optimizer.assert_not_called()
    else:
        optimizer.assert_called_once()


def test_vp_load_resolved_override_matrix(tmp_path):
    saved = _profile(107, source="saved")
    vp = VariationalPosterior(2, calibration=saved, rng=1)
    path = tmp_path / "resolved"
    vp.save(path)

    assert VariationalPosterior.load(path).calibration_profile == saved
    matching = _profile(107, source="override")
    loaded = VariationalPosterior.load(path, calibration=matching)
    assert loaded.calibration_profile.source == "saved"

    with pytest.raises(ValueError, match="already resolved"):
        VariationalPosterior.load(path, calibration="cached")
    with pytest.raises(ValueError, match="nondefault"):
        VariationalPosterior.load(path, calibration="off")
    with pytest.raises(ValueError, match="conflict"):
        VariationalPosterior.load(path, calibration=_profile(109))

    default_path = tmp_path / "resolved-default"
    VariationalPosterior(2, calibration="off", rng=1).save(default_path)
    assert VariationalPosterior.load(
        default_path, calibration="off"
    ).calibration_profile.uses_historical_defaults


def test_vp_load_unresolved_override_matrix_without_cache_io(
    tmp_path, monkeypatch
):
    cache = importlib.import_module("pyvbmc.calibration._cache")
    monkeypatch.setattr(
        cache,
        "resolve_cached_profile",
        lambda: pytest.fail("save/load must not read calibration cache"),
    )
    path = tmp_path / "pending"
    VariationalPosterior(2, calibration="cached", rng=1).save(path)

    assert VariationalPosterior.load(path).calibration_profile is None
    pending = VariationalPosterior.load(path, calibration="cached")
    assert pending.calibration_profile is None
    assert VariationalPosterior.load(
        path, calibration="off"
    ).calibration_profile.uses_historical_defaults
    explicit = _profile(113)
    loaded = VariationalPosterior.load(path, calibration=explicit)
    assert loaded.calibration_profile is explicit


def test_vp_load_migrates_legacy_state(tmp_path):
    vp = VariationalPosterior(2, calibration="cached", rng=1)
    del vp._calibration_profile
    del vp._calibration_request
    path = tmp_path / "legacy"
    vp.save(path)

    loaded = VariationalPosterior.load(path)
    assert loaded.calibration_profile.source == "legacy"
    assert loaded.calibration_profile.uses_historical_defaults


def test_vbmc_constructor_and_load_profile_rules(tmp_path):
    profile = _profile(127, source="saved")
    vbmc = _make_vbmc(profile)
    assert vbmc.vp.calibration_profile is profile
    assert vbmc.vp._calibration_display is False
    path = tmp_path / "vbmc"
    vbmc.save(path)

    matching = _profile(127, source="override")
    loaded = VBMC.load(path, new_options={"performance_calibration": matching})
    assert loaded.vp.calibration_profile.source == "saved"
    with pytest.raises(ValueError, match="already resolved"):
        VBMC.load(path, new_options={"performance_calibration": "cached"})

    with pytest.raises(ValueError, match="performance_calibration"):
        _make_vbmc("invalid")


def test_vbmc_load_selects_recorded_iteration_profile(tmp_path):
    vbmc = _make_vbmc(_profile(131, source="live"))
    first = VariationalPosterior(2, calibration=_profile(137, source="first"))
    last = VariationalPosterior(2, calibration=_profile(139, source="last"))
    vbmc.iteration_history.record("vp", first, 0)
    vbmc.iteration_history.record("vp", last, 1)
    vbmc.iteration = 1
    path = tmp_path / "iterations"
    vbmc.save(path)

    loaded = VBMC.load(path, iteration=0)
    assert loaded.vp.calibration_profile.source == "first"
    assert loaded.vp.calibration_profile.pdf_chunk_elements == 137


def test_completed_continuation_restores_before_resolution(monkeypatch):
    vbmc = _make_vbmc(_profile(149, source="live"))
    historical = VariationalPosterior(
        2, calibration=_profile(151, source="history"), rng=vbmc.rng
    )
    vbmc.iteration_history.record("vp", historical, 0)
    vbmc.iteration_history.record("optim_state", vbmc.optim_state, 0)
    vbmc.is_finished = True

    def stop_after_check(vp, *, display=None):
        assert vp.calibration_profile.source == "history"
        raise RuntimeError("resolved after history restore")

    monkeypatch.setattr(
        VariationalPosterior, "_resolve_calibration", stop_after_check
    )
    with pytest.raises(RuntimeError, match="after history restore"):
        vbmc.optimize()


def test_legacy_vbmc_load_pins_historical_defaults(tmp_path):
    vbmc = _make_vbmc("cached")
    del vbmc.options["performance_calibration"]
    del vbmc.vp._calibration_profile
    del vbmc.vp._calibration_request
    path = tmp_path / "legacy-vbmc"
    vbmc.save(path)

    loaded = VBMC.load(path)
    assert loaded.options["performance_calibration"] == "off"
    assert loaded.vp.calibration_profile.source == "legacy"
    assert loaded.vp.calibration_profile.uses_historical_defaults
