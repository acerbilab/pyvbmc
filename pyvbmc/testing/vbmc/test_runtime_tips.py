"""Focused tests for process-local optional runtime tips."""

import copy
import importlib
import random

import numpy as np
import pytest

from pyvbmc import VBMC
from pyvbmc._user_hints import emit_user_hint
from pyvbmc.calibration import _cache
from pyvbmc.calibration.profile import default_profile
from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.vbmc import _runtime_tips
from pyvbmc.vbmc._tip_catalog import TIPS, Tip


@pytest.fixture(autouse=True)
def isolated_tip_state():
    _runtime_tips._reset_runtime_tip_state(rng=random.Random(7))
    _cache._clear_process_state()
    yield
    _runtime_tips._reset_runtime_tip_state(rng=random.Random(7))
    _cache._clear_process_state()


def _tip(identifier, frequency="normal", urls=()):
    return Tip(identifier, f"message {identifier}", frequency, urls)


def _recording_emitter(output):
    def emit(message, *, display, urls=()):
        output.append((message, display, urls))
        return bool(display and display != "off")

    return emit


def _consider_many(count, *, catalog, output, **kwargs):
    emitted = []
    for _ in range(count):
        tip = _runtime_tips.consider_runtime_tip(
            display=kwargs.get("display", "iter"),
            enabled=kwargs.get("enabled", True),
            calibration_reminder_emitted=kwargs.get(
                "calibration_reminder_emitted", False
            ),
            catalog=catalog,
            emitter=_recording_emitter(output),
        )
        if tip is not None:
            emitted.append(tip)
    return emitted


def _make_vbmc(*, display="iter", calibration="off", show_tips=True):
    return VBMC(
        lambda x: -float(np.sum(np.asarray(x) ** 2)),
        np.zeros((1, 2)),
        np.full((1, 2), -5.0),
        np.full((1, 2), 5.0),
        np.full((1, 2), -2.0),
        np.full((1, 2), 2.0),
        options={
            "display": display,
            "performance_calibration": calibration,
            "show_tips": show_tips,
        },
        seed=3,
    )


def _stop_after_startup(monkeypatch, vbmc):
    def stop():
        raise RuntimeError("startup complete")

    monkeypatch.setattr(vbmc, "_log_column_headers", stop)
    with pytest.raises(RuntimeError, match="startup complete"):
        vbmc.optimize()


def test_emitter_display_modes_and_urls(capsys):
    assert not emit_user_hint("hidden", display=False, urls=("url",))
    assert not emit_user_hint("hidden", display="off", urls=("url",))
    assert capsys.readouterr().out == ""

    assert emit_user_hint("message", display="iter")
    assert capsys.readouterr().out == "message\n"
    assert emit_user_hint(
        "message", display="full", urls=("https://one/a?x=1", "https://two/b")
    )
    assert capsys.readouterr().out == (
        "message\nhttps://one/a?x=1\nhttps://two/b\n"
    )


def test_first_fourth_seventh_starts_and_exhaustion_are_reproducible():
    catalog = tuple(_tip(str(i)) for i in range(3))
    output = []
    starts = [
        _runtime_tips.consider_runtime_tip(
            display="iter",
            enabled=True,
            calibration_reminder_emitted=False,
            catalog=catalog,
            emitter=_recording_emitter(output),
        )
        for _ in range(7)
    ]
    emitted = [tip for tip in starts if tip is not None]

    expected = list(catalog)
    random.Random(7).shuffle(expected)
    assert [index for index, tip in enumerate(starts, 1) if tip] == [1, 4, 7]
    assert emitted == expected
    assert [item[0] for item in output] == [
        f"Tip: {tip.text}" for tip in expected
    ]
    assert len({tip.id for tip in emitted}) == len(emitted)
    assert not _consider_many(4, catalog=catalog, output=output)

    _runtime_tips._reset_runtime_tip_state(rng=random.Random(7))
    repeated = _consider_many(7, catalog=catalog, output=[])
    assert repeated == expected


def test_selection_preserves_inference_and_module_rng_states():
    inference_rng = np.random.default_rng(19)
    inference_state = copy.deepcopy(inference_rng.bit_generator.state)
    numpy_state = np.random.get_state()
    stdlib_state = random.getstate()

    _consider_many(4, catalog=(_tip("a"), _tip("b")), output=[])

    assert inference_rng.bit_generator.state == inference_state
    after_numpy = np.random.get_state()
    assert numpy_state[0] == after_numpy[0]
    assert np.array_equal(numpy_state[1], after_numpy[1])
    assert numpy_state[2:] == after_numpy[2:]
    assert random.getstate() == stdlib_state


@pytest.mark.parametrize(
    "catalog",
    [
        (),
        (_tip("only"),),
        (_tip("a"), _tip("b")),
        (_tip("a", "low_frequency"), _tip("b", "low_frequency")),
    ],
)
def test_small_catalogs_make_progress_without_retries(catalog):
    emitted = _consider_many(
        max(1, len(catalog) * 3 + 1), catalog=catalog, output=[]
    )
    assert len(emitted) == len(catalog)
    assert {tip.id for tip in emitted} == {tip.id for tip in catalog}


def test_low_frequency_can_appear_first_and_is_spaced_when_possible():
    class FixedOrderRng:
        @staticmethod
        def shuffle(items):
            return None

    catalog = (
        _tip("low-1", "low_frequency"),
        _tip("low-2", "low_frequency"),
        _tip("normal"),
    )
    _runtime_tips._reset_runtime_tip_state(rng=FixedOrderRng())
    emitted = _consider_many(7, catalog=catalog, output=[])

    assert [tip.id for tip in emitted] == ["low-1", "normal", "low-2"]


def test_skips_do_not_consume_cadence_catalog_or_category():
    class FixedOrderRng:
        @staticmethod
        def shuffle(items):
            return None

    catalog = (_tip("low", "low_frequency"), _tip("normal"))
    _runtime_tips._reset_runtime_tip_state(rng=FixedOrderRng())
    output = []

    for kwargs in (
        {"display": "off", "enabled": True},
        {"display": "iter", "enabled": False},
        {
            "display": "iter",
            "enabled": True,
            "calibration_reminder_emitted": True,
        },
    ):
        assert not _consider_many(1, catalog=catalog, output=output, **kwargs)

    first = _consider_many(1, catalog=catalog, output=output)
    assert [tip.id for tip in first] == ["low"]
    # Cadence-only skips do not select an entry or change the last category.
    assert not _consider_many(2, catalog=catalog, output=output)
    second = _consider_many(1, catalog=catalog, output=output)
    assert [tip.id for tip in second] == ["normal"]


def test_failed_emission_does_not_consume_tip():
    catalog = (_tip("a"),)
    assert (
        _runtime_tips.consider_runtime_tip(
            display="iter",
            enabled=True,
            calibration_reminder_emitted=False,
            catalog=catalog,
            emitter=lambda *args, **kwargs: False,
        )
        is None
    )
    emitted = _consider_many(3, catalog=catalog, output=[])
    assert [tip.id for tip in emitted] == ["a"]


@pytest.mark.parametrize(
    "catalog, error",
    [
        ((_tip("same"), _tip("same")), "unique"),
        ((Tip("", "text", "normal"),), "unique"),
        ((Tip("id", "", "normal"),), "text"),
        ((Tip("id", "text", "frequent"),), "frequency"),
    ],
)
def test_catalog_validation_has_no_fixed_size_or_categories(catalog, error):
    with pytest.raises(ValueError, match=error):
        _consider_many(1, catalog=catalog, output=[])


def test_catalog_can_add_replace_and_remove_either_category():
    variants = (
        (_tip("normal"),),
        (_tip("low", "low_frequency"),),
        (_tip("replacement"), _tip("new-low", "low_frequency")),
    )
    for catalog in variants:
        _runtime_tips._reset_runtime_tip_state(rng=random.Random(1))
        emitted = _consider_many(len(catalog) * 3, catalog=catalog, output=[])
        assert {tip.id for tip in emitted} == {tip.id for tip in catalog}


def test_catalog_records_are_well_formed_without_pinning_content():
    _runtime_tips._validate_catalog(TIPS)
    assert len({tip.id for tip in TIPS}) == len(TIPS)
    assert all(tip.id and tip.text for tip in TIPS)


def test_startup_considers_once_and_marks_quiet_start_handled(monkeypatch):
    module = importlib.import_module("pyvbmc.vbmc.vbmc")
    calls = []
    monkeypatch.setattr(
        module,
        "consider_runtime_tip",
        lambda **kwargs: calls.append(kwargs),
    )
    vbmc = _make_vbmc(display="off")

    _stop_after_startup(monkeypatch, vbmc)
    _stop_after_startup(monkeypatch, vbmc)

    assert len(calls) == 1
    assert calls[0]["display"] == "off"
    assert vbmc._runtime_tip_handled is True


def _mock_cache_miss(monkeypatch):
    profile = default_profile(
        source="default", status="cache_miss", provenance={"reason": "missing"}
    )
    monkeypatch.setattr(_cache, "resolve_cached_profile", lambda: profile)
    monkeypatch.setattr(
        _cache, "machine_identity", lambda: {"runtime_tip_test": True}
    )
    return profile


def test_real_startup_calibration_reminder_excludes_tip_and_target(
    monkeypatch, capsys
):
    _mock_cache_miss(monkeypatch)
    calls = []

    def target(x):
        calls.append(x)
        return -1.0

    vbmc = VBMC(
        target,
        np.zeros((1, 2)),
        np.full((1, 2), -5.0),
        np.full((1, 2), 5.0),
        np.full((1, 2), -2.0),
        np.full((1, 2), 2.0),
        options={"performance_calibration": "cached"},
        seed=3,
    )
    _stop_after_startup(monkeypatch, vbmc)
    output = capsys.readouterr().out

    assert output.count("PyVBMC is using standard performance settings.") == 1
    assert "Tip:" not in output
    assert vbmc.vp._calibration_hint_emitted is True
    assert calls == []


def test_early_vp_real_reminder_occupies_startup_slot(monkeypatch, capsys):
    _mock_cache_miss(monkeypatch)
    vbmc = _make_vbmc(calibration="cached")

    vbmc.vp.pdf(np.zeros((1, 2)), orig_flag=False)
    assert vbmc.vp._calibration_hint_emitted is True
    assert (
        "PyVBMC is using standard performance settings."
        in capsys.readouterr().out
    )
    _stop_after_startup(monkeypatch, vbmc)

    assert "Tip:" not in capsys.readouterr().out


def test_silent_cache_miss_after_other_vp_leaves_tip_slot(monkeypatch, capsys):
    _mock_cache_miss(monkeypatch)
    unrelated = VariationalPosterior(2, calibration="cached", rng=1)
    unrelated.pdf(np.zeros((1, 2)), orig_flag=False)
    assert unrelated._calibration_hint_emitted is True
    capsys.readouterr()

    vbmc = _make_vbmc(calibration="cached")
    _stop_after_startup(monkeypatch, vbmc)
    output = capsys.readouterr().out

    assert vbmc.vp._calibration_hint_emitted is False
    assert "PyVBMC is using standard performance settings." not in output
    assert "Tip:" in output


@pytest.mark.parametrize(
    "calibration", ["off", default_profile(source="cache")]
)
def test_nonreminding_profiles_leave_tip_slot_available(
    monkeypatch, calibration
):
    module = importlib.import_module("pyvbmc.vbmc.vbmc")
    calls = []
    monkeypatch.setattr(
        module,
        "consider_runtime_tip",
        lambda **kwargs: calls.append(kwargs),
    )
    vbmc = _make_vbmc(calibration=calibration)

    _stop_after_startup(monkeypatch, vbmc)

    assert calls[0]["calibration_reminder_emitted"] is False


def test_load_migrates_option_and_first_start_flag_without_output(
    tmp_path, capsys
):
    pre_run = _make_vbmc(display="off")
    pre_run_path = tmp_path / "pre-run"
    pre_run.save(pre_run_path)
    capsys.readouterr()
    loaded_pre_run = VBMC.load(pre_run_path, new_options={"show_tips": False})
    assert capsys.readouterr().out == ""
    assert loaded_pre_run._runtime_tip_handled is False
    assert loaded_pre_run.options["show_tips"] is False

    legacy_pre_run = _make_vbmc(display="off")
    del legacy_pre_run.options["show_tips"]
    del legacy_pre_run._runtime_tip_handled
    legacy_pre_run_path = tmp_path / "legacy-pre-run"
    legacy_pre_run.save(legacy_pre_run_path)
    loaded_legacy_pre_run = VBMC.load(legacy_pre_run_path)
    assert loaded_legacy_pre_run._runtime_tip_handled is False
    assert loaded_legacy_pre_run.options["show_tips"] is True

    legacy = _make_vbmc(display="off")
    del legacy.options["show_tips"]
    del legacy._runtime_tip_handled
    legacy.iteration = 0
    legacy_path = tmp_path / "legacy"
    legacy.save(legacy_path)
    loaded_legacy = VBMC.load(legacy_path, new_options={"show_tips": False})
    assert loaded_legacy._runtime_tip_handled is True
    assert loaded_legacy.options["show_tips"] is False

    with pytest.raises(ValueError, match="show_tips.*boolean"):
        VBMC.load(legacy_path, new_options={"show_tips": "yes"})


def test_constructor_rejects_nonboolean_show_tips():
    with pytest.raises(ValueError, match="show_tips.*boolean"):
        _make_vbmc(show_tips="yes")


def test_tip_uses_stdout_and_is_excluded_from_log_file(
    monkeypatch, capsys, tmp_path
):
    log_path = tmp_path / "vbmc.log"
    vbmc = VBMC(
        lambda x: -float(np.sum(np.asarray(x) ** 2)),
        np.zeros((1, 2)),
        np.full((1, 2), -5.0),
        np.full((1, 2), 5.0),
        np.full((1, 2), -2.0),
        np.full((1, 2), 2.0),
        options={
            "performance_calibration": "off",
            "log_file_name": str(log_path),
        },
        seed=3,
    )

    _stop_after_startup(monkeypatch, vbmc)

    assert "Tip:" in capsys.readouterr().out
    assert "Tip:" not in log_path.read_text(encoding="utf-8")


def test_current_started_state_survives_save_for_finished_and_unfinished(
    tmp_path,
):
    for is_finished in (False, True):
        vbmc = _make_vbmc(display="off")
        vbmc._runtime_tip_handled = True
        vbmc.is_finished = is_finished
        path = tmp_path / str(is_finished)
        vbmc.save(path)
        assert VBMC.load(path)._runtime_tip_handled is True


def test_vp_marker_migration_preserves_true_value():
    vbmc = _make_vbmc(display="off")
    vbmc.vp._calibration_hint_emitted = True
    vbmc.vp._ensure_calibration_state()
    assert vbmc.vp._calibration_hint_emitted is True

    del vbmc.vp._calibration_hint_emitted
    vbmc.vp._ensure_calibration_state()
    assert vbmc.vp._calibration_hint_emitted is False


def test_legacy_pending_vp_resolves_without_marker(monkeypatch):
    cache = importlib.import_module("pyvbmc.calibration._cache")
    profile = default_profile(
        source="default", status="cache_miss", provenance={"reason": "missing"}
    )
    monkeypatch.setattr(cache, "resolve_cached_profile", lambda: profile)
    monkeypatch.setattr(
        cache, "suggest_calibration_once", lambda *a, **k: True
    )
    vbmc = _make_vbmc(display="off", calibration="cached")
    del vbmc.vp._calibration_hint_emitted

    assert vbmc.vp._resolve_calibration() is profile
    assert vbmc.vp._calibration_hint_emitted is True
