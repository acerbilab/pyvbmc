"""The rounding of a half, as MATLAB's ``round`` rounds it."""

import numpy as np
import pytest

from pyvbmc.stats._rounding import round_half_away_from_zero

# The largest double below a half, and its negative: `floor(abs(x) + 0.5)`
# sends them to one, the sum rounding up to exactly 1.0.
JUST_BELOW_A_HALF = np.nextafter(0.5, 0.0)

# The largest odd integer a double represents exactly. Adding a half to it
# is inexact and gives the next even integer.
LARGE_ODD_INTEGER = 2.0**53 - 1


@pytest.mark.parametrize(
    "value, rounded",
    [
        (-2.5, -3),
        (-1.5, -2),
        (-0.5, -1),
        (0.5, 1),
        (1.5, 2),
        (2.5, 3),
    ],
)
def test_a_half_goes_away_from_zero(value, rounded):
    """MATLAB's ``round`` sends a half away from zero, where Python's
    built-in ``round`` and ``numpy.round`` send it to the nearer even
    integer."""
    assert round_half_away_from_zero(value) == rounded


@pytest.mark.parametrize("value", [JUST_BELOW_A_HALF, -JUST_BELOW_A_HALF])
def test_the_largest_value_below_a_half_goes_to_zero(value):
    assert round_half_away_from_zero(value) == 0


def test_a_large_integer_is_returned_as_it_is():
    assert round_half_away_from_zero(LARGE_ODD_INTEGER) == LARGE_ODD_INTEGER
    assert round_half_away_from_zero(-LARGE_ODD_INTEGER) == -LARGE_ODD_INTEGER


def test_a_scalar_gives_a_count():
    """Every caller inside the package uses the value as a count: as a
    number of samples, or to slice an array."""
    rounded = round_half_away_from_zero(np.float64(2.5))
    assert isinstance(rounded, int)
    assert rounded == 3


def test_an_array_is_rounded_elementwise():
    values = np.array(
        [[-2.5, -1.5, -0.5, 0.5], [1.5, 2.5, JUST_BELOW_A_HALF, 3.4]]
    )
    expected = np.array([[-3.0, -2.0, -1.0, 1.0], [2.0, 3.0, 0.0, 3.0]])

    rounded = round_half_away_from_zero(values)

    assert isinstance(rounded, np.ndarray)
    assert rounded.shape == values.shape
    assert np.array_equal(rounded, expected)
    # The argument is left alone.
    assert values[0, 0] == -2.5


def test_a_list_is_taken_as_an_array():
    assert np.array_equal(
        round_half_away_from_zero([0.5, -0.5, 2.5]),
        np.array([1.0, -1.0, 3.0]),
    )
