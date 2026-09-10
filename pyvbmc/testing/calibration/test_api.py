from contextlib import contextmanager
from pathlib import Path

import pytest

import pyvbmc
from pyvbmc.calibration import _api, _cache
from pyvbmc.calibration.profile import CalibrationProfile
from pyvbmc.testing.calibration.test_cache import make_test_record


def identity():
    return {
        "cpu_model": "Test CPU",
        "architecture": "test64",
        "host_hash": "b" * 64,
        "operating_system": {"system": "TestOS", "release": "1"},
        "python": "3.10",
        "numpy": "2.0.0",
        "scipy": "1.15.0",
        "backends": [
            {
                "user_api": "blas",
                "internal_api": "openblas",
                "prefix": "libopenblas",
                "version": "1.0",
                "num_threads": 1,
                "threading_layer": "pthreads",
                "architecture": "Test CPU",
            }
        ],
        "thread_environment": {
            name: None for name in _cache._THREAD_ENVIRONMENT
        },
        "device": "cpu",
        "dtype": "float64",
    }


def result(status="complete"):
    settings = {
        "pdf_chunk_elements": 2**14,
        "entropy_grad_chunk_elements": 2**15,
        "entropy_value_chunk_elements": 2**16,
    }
    summary = {
        name: {
            "selected": value,
            "reason": "held-out gates passed",
            "heldout_speedup": 1.25,
        }
        for name, value in settings.items()
    }
    return {
        "settings": settings,
        "report": {
            "reason": "synthetic outcome",
            "summary": summary,
        },
        "status": status,
    }


def complete_result():
    value = result()
    record = make_test_record(identity(), **value["settings"])
    value["report"] = record["report"]
    return value


def guard(*, acquired=True, persistent=False, reason="test cache disabled"):
    data = identity()
    return _cache.CampaignGuard(
        acquired=acquired,
        persistent=persistent,
        reason=reason,
        identity=data,
        fingerprint=_cache.fingerprint(data),
        path=_cache.cache_path(_cache.fingerprint(data)),
    )


def install_guard(monkeypatch, value):
    @contextmanager
    def fake_guard():
        yield value

    monkeypatch.setattr(_api, "campaign_guard", fake_guard)


def test_public_api_is_exported_and_keyword_only():
    assert pyvbmc.CalibrationProfile is CalibrationProfile
    with pytest.raises(TypeError):
        pyvbmc.calibrate(False)
    with pytest.raises(TypeError, match="verbose"):
        pyvbmc.calibrate(verbose=1)


def test_every_quiet_call_runs_fresh_campaign(monkeypatch, capsys):
    install_guard(monkeypatch, guard())
    calls = []

    def fake_campaign(*, progress, deadline):
        calls.append((progress, deadline))
        return result()

    monkeypatch.setattr(_api, "_run_campaign", fake_campaign)
    monkeypatch.setattr(_api, "register_success", lambda profile, report: None)

    first = pyvbmc.calibrate(verbose=False)
    second = pyvbmc.calibrate(verbose=False)

    assert len(calls) == 2
    assert all(progress is None for progress, _ in calls)
    assert first.settings == second.settings
    assert capsys.readouterr().out == ""


def test_watchdog_is_five_minutes_from_api_entry(monkeypatch):
    install_guard(monkeypatch, guard())
    clock = iter([100.0, 135.0])
    monkeypatch.setattr(_api.time, "monotonic", lambda: next(clock))
    seen = {}

    def fake_campaign(*, progress, deadline):
        seen["deadline"] = deadline
        return result()

    monkeypatch.setattr(_api, "_run_campaign", fake_campaign)
    monkeypatch.setattr(_api, "register_success", lambda profile, report: None)

    pyvbmc.calibrate(verbose=False)

    assert seen["deadline"] == 400.0


def test_busy_returns_fallback_without_campaign(monkeypatch, capsys):
    install_guard(
        monkeypatch,
        guard(acquired=False, reason="another calibration is active"),
    )
    previous = CalibrationProfile(
        pdf_chunk_elements=2**18,
        source="cache",
        fingerprint="c" * 64,
        cache_path="C:/cache/previous.json",
    )
    monkeypatch.setattr(
        _api, "_resolve_cached_profile", lambda: (previous, None)
    )
    monkeypatch.setattr(
        _api,
        "_run_campaign",
        lambda **kwargs: pytest.fail("busy call ran a campaign"),
    )

    profile = pyvbmc.calibrate()

    assert profile.pdf_chunk_elements == 2**18
    assert profile.status == "busy"
    assert profile.provenance["fallback_source"] == "cache"
    output = capsys.readouterr().out
    assert "Calibration did not start:" in output
    assert "Using the previous saved settings." in output
    previous_location = _api._display_path(previous.cache_path)
    assert f"Previous results: {previous_location}" in output
    assert "262,144" not in output


@pytest.mark.parametrize("status", ["incomplete", "invalid"])
def test_failed_rerun_preserves_previous_valid_settings(
    monkeypatch, capsys, status
):
    install_guard(monkeypatch, guard())
    previous = CalibrationProfile(
        pdf_chunk_elements=2**18,
        entropy_grad_chunk_elements=2**17,
        source="cache",
        fingerprint="d" * 64,
        cache_path="C:/cache/previous.json",
    )
    monkeypatch.setattr(
        _api, "_resolve_cached_profile", lambda: (previous, None)
    )
    monkeypatch.setattr(_api, "_run_campaign", lambda **kwargs: result(status))

    profile = pyvbmc.calibrate()

    assert profile.settings == previous.settings
    assert profile.status == status
    assert profile.source == "cache"
    output = capsys.readouterr().out
    assert "Calibration could not complete after" in output
    assert f"Calibration {status}" not in output
    assert "synthetic outcome" in output
    assert "Using the previous saved settings." in output
    previous_location = _api._display_path(previous.cache_path)
    assert f"Previous results: {previous_location}" in output
    assert "262,144" not in output


def test_successful_persistent_campaign_returns_cache_path(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("PYVBMC_CACHE_DIR", str(tmp_path))
    install_guard(monkeypatch, guard(persistent=True, reason=None))
    monkeypatch.setattr(
        _api, "_run_campaign", lambda **kwargs: complete_result()
    )

    profile = pyvbmc.calibrate()

    assert profile.status == "complete"
    assert profile.source == "calibrated"
    assert Path(profile.cache_path).is_file()
    assert _cache._load_report(profile) == complete_result()["report"]
    output = capsys.readouterr().out
    assert f"Results saved to: {profile.cache_path}" in output
    assert "Future runs will use these settings automatically." in output
    assert "To recalibrate, run pyvbmc.calibrate() again." in output
    assert "Selected element budgets" not in output


def test_persistence_failure_keeps_success_in_memory(monkeypatch, capsys):
    guard_value = guard(persistent=True, reason=None)
    install_guard(monkeypatch, guard_value)
    monkeypatch.setattr(
        _api, "_run_campaign", lambda **kwargs: complete_result()
    )
    monkeypatch.setattr(
        _api,
        "write_record",
        lambda record: (_ for _ in ()).throw(PermissionError("read only")),
    )
    registered = []
    monkeypatch.setattr(
        _api,
        "register_success",
        lambda profile, report: registered.append(profile),
    )

    profile = pyvbmc.calibrate()

    assert profile.status == "complete"
    assert profile.source == "memory"
    assert profile.cache_path is None
    assert "write failed" in profile.provenance["persistence"]
    assert registered == [profile]
    output = capsys.readouterr().out
    assert "Results could not be saved. Attempted location:" in output
    assert str(Path(guard_value.path).absolute()) in output
    assert "cache write failed" in output
    assert "These settings are available in this process." in output
    assert "Future runs will use these settings automatically." not in output


def test_verbose_feedback_reports_progress_values_and_location(
    monkeypatch, capsys
):
    guard_value = guard()
    install_guard(monkeypatch, guard_value)

    def fake_campaign(*, progress, deadline):
        progress(
            {
                "stage": "setup",
                "completed": 1,
                "total": 1,
                "message": "synthetic inputs ready",
            }
        )
        return result()

    monkeypatch.setattr(_api, "_run_campaign", fake_campaign)
    monkeypatch.setattr(_api, "register_success", lambda profile, report: None)

    pyvbmc.calibrate()

    output = capsys.readouterr().out
    assert "Calibrating PyVBMC performance for this machine." in output
    assert "keep other demanding tasks paused" in output
    assert "Calibration usually takes tens of seconds." in output
    assert "synthetic inputs ready" in output
    assert "Faster settings were selected for this machine." in output
    assert "Results could not be saved. Attempted location:" in output
    assert str(Path(guard_value.path).absolute()) in output
    assert "16,384" not in output
    assert "1.25x" not in output


def test_verbose_default_completion_uses_plain_summary(monkeypatch, capsys):
    install_guard(monkeypatch, guard())
    value = result()
    value["settings"] = {name: 2**16 for name in value["settings"]}
    monkeypatch.setattr(_api, "_run_campaign", lambda **kwargs: value)
    monkeypatch.setattr(_api, "register_success", lambda profile, report: None)

    pyvbmc.calibrate()

    output = capsys.readouterr().out
    assert "The standard settings performed well" in output
    assert "no reliable improvement was found." in output
    assert "Faster settings were selected" not in output


def test_keyboard_interrupt_propagates(monkeypatch):
    install_guard(monkeypatch, guard())

    def interrupt(**kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(_api, "_run_campaign", interrupt)
    with pytest.raises(KeyboardInterrupt):
        pyvbmc.calibrate(verbose=False)
