"""Implementation of the explicit public calibration API."""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Mapping

from ._cache import (
    KERNEL_REVISION,
    WORKLOAD_REVISION,
    _package_version,
    _resolve_cached_profile,
    _validate_settings,
    campaign_guard,
    identity_reuse_reason,
    make_record,
    register_success,
    write_record,
)
from .profile import CalibrationProfile

WATCHDOG_SECONDS = 300.0


def _run_campaign(
    *, progress: Callable[[Mapping[str, Any]], None] | None, deadline: float
) -> Mapping[str, Any]:
    # Keep campaign imports off the import and cache-lookup paths.
    from ._campaign import run_campaign

    return run_campaign(progress=progress, deadline=deadline)


def _progress_printer(event: Mapping[str, Any]) -> None:
    message = event.get("message")
    if message:
        print(f"PyVBMC calibration: {message}")
        return
    stage = event.get("stage", "working")
    completed = event.get("completed")
    total = event.get("total")
    if completed is not None and total is not None:
        print(f"PyVBMC calibration: {stage} ({completed}/{total})")
    else:
        print(f"PyVBMC calibration: {stage}")


def _fallback_profile(
    *, status: str, reason: str, current_fingerprint: str
) -> CalibrationProfile:
    previous, _ = _resolve_cached_profile()
    provenance = dict(previous.provenance)
    provenance.update(
        {
            "outcome": status,
            "reason": reason,
            "fallback_source": previous.source,
        }
    )
    return CalibrationProfile(
        **previous.settings,
        source=previous.source,
        status=status,
        fingerprint=previous.fingerprint or current_fingerprint,
        cache_path=previous.cache_path,
        provenance=provenance,
    )


def _print_report_summary(
    profile: CalibrationProfile,
    *,
    elapsed: float,
    persistence: str,
    report_saved: bool,
    cache_location: str,
) -> None:
    if profile.uses_historical_defaults:
        print(
            f"Calibration finished in {elapsed:.0f} seconds. The standard "
            "settings performed well; no reliable improvement was found."
        )
    else:
        print(
            f"Calibration finished in {elapsed:.0f} seconds. Faster settings "
            "were selected for this machine."
        )
    if report_saved and profile.cache_path:
        print(f"Results saved to: {_display_path(profile.cache_path)}")
        print("Future runs will use these settings automatically.")
    else:
        print(
            "Results could not be saved. Attempted location: "
            f"{_display_path(cache_location)}"
        )
        print(f"Reason: {persistence}.")
        print("These settings are available in this process.")
    print("To recalibrate, run pyvbmc.calibrate() again.")


def _display_path(path: str) -> str:
    """Return an absolute path for user-facing status messages."""
    return os.path.abspath(os.path.expanduser(path))


def _print_fallback_summary(
    profile: CalibrationProfile,
    *,
    outcome: str,
    reason: str,
    elapsed: float,
) -> None:
    if outcome == "is busy":
        print(f"Calibration did not start: {reason}.")
    else:
        print(
            "Calibration could not complete after "
            f"{elapsed:.0f} seconds: {reason}."
        )
    if profile.source == "cache":
        print("Using the previous saved settings.")
    elif profile.source == "memory":
        print("Using the previous in-process settings.")
    else:
        print("Using the standard settings.")
    if profile.cache_path:
        print(f"Previous results: {_display_path(profile.cache_path)}")


def calibrate(*, verbose: bool = True) -> CalibrationProfile:
    """Measure and return performance settings for this machine.

    Each call requests a fresh campaign. Normal constructors and numerical
    methods only read compatible cached settings and never invoke this work.

    Parameters
    ----------
    verbose : bool, optional
        Print progress and the resulting settings. The default is ``True``.

    Returns
    -------
    CalibrationProfile
        Immutable settings together with compact outcome and cache metadata.
    """
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a bool")

    api_entry = time.monotonic()
    deadline = api_entry + WATCHDOG_SECONDS
    if verbose:
        print(
            "Calibrating PyVBMC performance for this machine. For useful "
            "measurements, keep other demanding tasks paused. Progress will "
            "appear below."
        )
        print("Calibration usually takes tens of seconds.")

    with campaign_guard() as guard:
        if not guard.acquired:
            elapsed = time.monotonic() - api_entry
            profile = _fallback_profile(
                status="busy",
                reason=guard.reason or "another calibration is active",
                current_fingerprint=guard.fingerprint,
            )
            if verbose:
                _print_fallback_summary(
                    profile,
                    outcome="is busy",
                    reason=guard.reason or "another calibration is active",
                    elapsed=elapsed,
                )
            return profile

        if verbose and guard.reason:
            print("PyVBMC calibration will run in memory; " f"{guard.reason}.")

        result = _run_campaign(
            progress=_progress_printer if verbose else None,
            deadline=deadline,
        )
        if not isinstance(result, Mapping):
            raise TypeError("calibration campaign returned an invalid result")
        required = {"settings", "report", "status"}
        if set(result) != required:
            raise ValueError("calibration campaign result has invalid fields")
        status = result["status"]
        report = result["report"]
        if not isinstance(status, str) or not isinstance(report, Mapping):
            raise ValueError("calibration campaign result is invalid")
        if status not in {"complete", "incomplete", "invalid"}:
            raise ValueError("calibration campaign returned an unknown status")
        elapsed = time.monotonic() - api_entry

        if status != "complete":
            reason = str(report.get("reason", f"campaign status: {status}"))
            profile = _fallback_profile(
                status=status,
                reason=reason,
                current_fingerprint=guard.fingerprint,
            )
            if verbose:
                _print_fallback_summary(
                    profile,
                    outcome=status,
                    reason=reason,
                    elapsed=elapsed,
                )
            return profile

        settings = result["settings"]
        if not isinstance(settings, Mapping):
            raise ValueError("calibration campaign settings are invalid")
        settings = _validate_settings(dict(settings))

        reuse_reason = identity_reuse_reason(guard.identity)
        persistence_reason = guard.reason or reuse_reason
        persistence = "not written"
        cache_path = None
        provenance = {
            "pyvbmc_version": _package_version(),
            "kernel_revision": KERNEL_REVISION,
            "workload_revision": WORKLOAD_REVISION,
            "elapsed_seconds": float(elapsed),
        }

        if guard.persistent and reuse_reason is None:
            record = make_record(
                identity=guard.identity,
                settings=settings,
                report=report,
                elapsed_seconds=elapsed,
            )
            try:
                written_path = write_record(record)
            except OSError as error:
                persistence_reason = f"cache write failed ({error})"
            else:
                cache_path = str(written_path)
                provenance = dict(record["provenance"])
                provenance.update(
                    {
                        "kernel_revision": KERNEL_REVISION,
                        "workload_revision": WORKLOAD_REVISION,
                    }
                )
                persistence = "saved"

        if persistence != "saved":
            provenance["persistence"] = persistence_reason or "unavailable"

        profile = CalibrationProfile(
            **settings,
            source="calibrated" if persistence == "saved" else "memory",
            status="complete",
            fingerprint=guard.fingerprint,
            cache_path=cache_path,
            provenance=provenance,
        )
        register_success(profile, report)
        if verbose:
            detail = persistence
            if persistence_reason and persistence != "saved":
                detail = f"{persistence}: {persistence_reason}"
            _print_report_summary(
                profile,
                elapsed=elapsed,
                persistence=detail,
                report_saved=persistence == "saved",
                cache_location=str(guard.path),
            )
        return profile
