"""Checks on the string summary of a ``VBMC`` instance."""

import copy
from pathlib import Path

import numpy as np

from pyvbmc import VBMC
from pyvbmc.formatting import summarize

base_path = Path(__file__).parent


def _summary_field(vbmc, prefix):
    """The text the summary prints for the field with the given prefix."""
    for line in str(vbmc).splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped[len(prefix) :].rstrip(",")
    raise AssertionError(f"No field '{prefix}' in the summary:\n{vbmc}")


def _unbounded_vbmc():
    """An instance whose starting point is off the centre of its box."""
    x0 = np.array([[3.5, 0.0]])
    return (
        VBMC(
            lambda x: -0.5 * np.sum(x**2),
            x0,
            np.array([[-np.inf, -np.inf]]),
            np.array([[np.inf, np.inf]]),
            np.array([[2.0, -3.0]]),
            np.array([[4.0, 5.0]]),
            options={"display": "off"},
            seed=1,
        ),
        x0,
    )


def test_summary_prints_the_starting_point_in_original_coordinates():
    """The summary reports the starting point as the caller gave it."""
    vbmc, x0 = _unbounded_vbmc()

    assert _summary_field(vbmc, "x0: ") == summarize(x0)


def test_summary_keeps_the_starting_point_across_a_warp():
    """A warp of the inference space does not move the starting point.

    Warping replaces the map between the user's coordinates and the
    inference space, and the coordinates of the starting point the caller
    gave are unaffected by it.
    """
    vbmc, x0 = _unbounded_vbmc()
    before = _summary_field(vbmc, "x0: ")

    warped = copy.deepcopy(vbmc.parameter_transformer)
    warped.mu = np.zeros(vbmc.D)
    warped.delta = np.ones(vbmc.D)
    vbmc.parameter_transformer = warped
    vbmc.vp.parameter_transformer = warped
    vbmc.function_logger.parameter_transformer = warped
    # The warp is a real change of map: it moves the starting point of the
    # inference space somewhere else in the user's coordinates.
    assert not np.allclose(warped.inverse(vbmc.x0), x0)

    assert _summary_field(vbmc, "x0: ") == before == summarize(x0)


def test_summary_of_a_run_loaded_from_a_file():
    """A stored run prints, and reports its starting point to the user."""
    loaded = VBMC.load(base_path.joinpath("test_vbmc_save_static.pkl"))

    assert _summary_field(loaded, "x0: ") == summarize(
        loaded.parameter_transformer.inverse(loaded.x0)
    )


def test_summary_reports_the_prior_the_caller_gave():
    """The summary reports the prior density and sampler in use."""

    def log_prior(x):
        return -0.5 * np.sum(x**2)

    def sample_prior(n):
        return np.zeros((n, 2))

    vbmc = VBMC(
        lambda x: -0.5 * np.sum(x**2),
        np.zeros((1, 2)),
        np.array([[-np.inf, -np.inf]]),
        np.array([[np.inf, np.inf]]),
        np.array([[-1.0, -1.0]]),
        np.array([[1.0, 1.0]]),
        log_prior=log_prior,
        sample_prior=sample_prior,
        options={"display": "off"},
        seed=1,
    )

    assert _summary_field(vbmc, "log-prior = ") == str(log_prior)
    assert _summary_field(vbmc, "prior sampler = ") == str(sample_prior)


def test_summary_reports_the_gaussian_process_of_the_run():
    """The summary reports the Gaussian process the instance holds."""
    vbmc, _ = _unbounded_vbmc()
    assert vbmc.gp is None
    assert _summary_field(vbmc, "Gaussian process = ") == "None"

    loaded = VBMC.load(base_path.joinpath("test_vbmc_save_static.pkl"))
    assert loaded.gp is not None
    assert _summary_field(loaded, "Gaussian process = ") != "None"
    assert str(loaded.gp).splitlines()[0] in str(loaded)
