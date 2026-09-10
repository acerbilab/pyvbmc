"""Machine-local performance calibration for PyVBMC."""

from .profile import CalibrationProfile


def calibrate(*, verbose=True):
    """Measure and return performance settings for this machine.

    Every call runs a fresh calibration campaign. Importing PyVBMC and using
    a compatible cached profile do not run the campaign.

    Parameters
    ----------
    verbose : bool, optional
        Print the runtime estimate, campaign progress, selected settings,
        held-out results, and persistence status. The default is ``True``.

    Returns
    -------
    CalibrationProfile
        Immutable settings and compact outcome metadata. Detailed timings
        are stored separately in the JSON file named by ``cache_path`` when
        persistence succeeds.

    Notes
    -----
    The displayed 30-second duration is an estimate rather than a deadline.
    A five-minute watchdog stops further work only after a numerical call
    returns. If another calibration is active or the campaign is incomplete,
    this function returns a compatible prior profile or historical defaults
    with the outcome recorded in its ``status`` and ``provenance``.
    """
    from ._api import calibrate as _calibrate

    return _calibrate(verbose=verbose)


__all__ = ["CalibrationProfile", "calibrate"]
