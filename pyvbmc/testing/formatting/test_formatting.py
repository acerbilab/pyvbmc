"""Checks on the string summaries built by ``pyvbmc.formatting``."""

import numpy as np
import pytest

from pyvbmc import VariationalPosterior
from pyvbmc.formatting import format_dict, get_repr, summarize


@pytest.mark.parametrize(
    "value, expected",
    [
        (np.float64(-1.5), "-1.5"),
        (np.float32(0.5), "0.5"),
        (np.int64(5), "5"),
        (np.True_, "True"),
        (np.str_("a"), "'a'"),
    ],
)
def test_numpy_scalars_print_as_python_values(value, expected):
    """NumPy 2 writes a NumPy scalar with its type, ``np.float64(-1.5)``; the
    summaries write the value it holds, as for the Python number."""
    assert get_repr(value) == expected
    assert get_repr(value, expand=True) == expected
    assert summarize(value) == expected


def test_format_dict_prints_numpy_scalars_as_values():
    string = format_dict(
        {
            "elbo": np.float64(-1.5),
            "stable": np.True_,
            "best_iter": np.int64(5),
            "I_sk": np.zeros((8, 50)),
        }
    )
    assert "'elbo': -1.5," in string
    assert "'stable': True," in string
    assert "'best_iter': 5," in string
    assert "'I_sk': (8, 50) ndarray," in string
    assert "np." not in string


def test_variational_posterior_summary_prints_its_stats_as_values():
    vp = VariationalPosterior(2, 2)
    vp.stats = {"elbo": np.float64(-1.5), "elbo_sd": np.float64(0.25)}
    assert "'elbo': -1.5," in str(vp)
    assert "'elbo_sd': 0.25," in str(vp)
    assert "np.float64" not in str(vp)
