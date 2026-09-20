"""Checks on the variational posterior a ``VBMC`` instance starts from."""

import numpy as np

from pyvbmc import VBMC


def _vbmc(x0, lb, ub, plb, pub, **options):
    return VBMC(
        lambda x: -0.5 * np.sum(x**2),
        np.array(x0),
        np.array(lb),
        np.array(ub),
        np.array(plb),
        np.array(pub),
        options={"display": "off", **options},
        seed=1,
    )


def _initial_means(vbmc):
    """The initial component means, in the coordinates of the caller."""
    return vbmc.parameter_transformer.inverse(vbmc.vp.mu.T)


def test_initial_means_sit_on_the_starting_point_in_a_bounded_box():
    """The components start where the caller's starting point is.

    The component means of a variational posterior are coordinates of the
    transformed space in which the algorithm runs, so mapping the initial
    means back must return the starting point the caller gave.
    """
    x0 = [[5.0]]
    vbmc = _vbmc(x0, [[0.0]], [[10.0]], [[4.0]], [[6.0]])

    assert np.allclose(_initial_means(vbmc), x0, atol=1e-4)


def test_initial_means_follow_a_starting_point_off_the_centre():
    """A starting point away from the centre of the box is kept."""
    x0 = [[5.5]]
    vbmc = _vbmc(x0, [[0.0]], [[10.0]], [[4.0]], [[6.0]])

    assert np.allclose(_initial_means(vbmc), x0, atol=1e-4)


def test_initial_means_follow_the_starting_point_under_a_logit_transform():
    """The map of the bounded transform in use is the one applied."""
    x0 = [[5.0]]
    vbmc = _vbmc(
        x0, [[0.0]], [[10.0]], [[4.0]], [[6.0]], bounded_transform="logit"
    )

    assert np.allclose(_initial_means(vbmc), x0, atol=1e-4)


def test_initial_means_follow_the_starting_point_in_an_unbounded_space():
    """An unbounded space is shifted and scaled, and the means follow."""
    x0 = [[3.5, 0.0]]
    vbmc = _vbmc(
        x0,
        [[-np.inf, -np.inf]],
        [[np.inf, np.inf]],
        [[2.0, -3.0]],
        [[4.0, 5.0]],
    )

    assert np.allclose(_initial_means(vbmc), x0, atol=1e-4)
