"""Machine-local performance calibration for PyVBMC."""

from .profile import CalibrationProfile


def calibrate(*, verbose=True):
    """Measure and return performance settings for this machine.

    Each call requests a fresh calibration campaign. If another campaign is
    active, or the measurements are incomplete or invalid, the call returns
    the best compatible saved or in-process profile, or the standard profile
    when none is available. The returned profile's ``status`` records that
    outcome and its ``source`` records where the settings came from. A
    failed campaign does not replace a valid saved record. Importing PyVBMC
    and using a compatible cached profile do not run the campaign.

    Parameters
    ----------
    verbose : bool, optional
        Print the expected duration, campaign progress, whether faster
        settings were selected, and where the results were saved. The
        default is ``True``.

    Returns
    -------
    CalibrationProfile
        Immutable settings together with compact outcome and cache metadata.
        A completed campaign has status ``"complete"``. A fallback has
        status ``"busy"``, ``"incomplete"`` or ``"invalid"`` and uses the
        best compatible saved, in-process or standard settings. The held-out
        measurements behind completed, persisted settings are in the JSON
        report named by ``cache_path``.

    Raises
    ------
    TypeError
        If `verbose` is not a bool.

    Notes
    -----
    The selected budgets are attributes of the returned profile. A campaign
    takes tens of seconds, an estimate rather than a deadline. A five-minute
    watchdog stops further work only after a numerical call returns.
    """
    from ._api import calibrate as _calibrate

    return _calibrate(verbose=verbose)


__all__ = ["CalibrationProfile", "calibrate"]
