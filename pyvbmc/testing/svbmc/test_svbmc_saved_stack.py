"""Loading the stack file under ``fixtures/saved/``.

The file is a stack of the ``bounded_D2`` posteriors written by
``SVBMC.save`` (``dev/scripts/make_svbmc_fixtures.py saved-stack``) under
the Python version, platform and library versions that its JSON sidecar
records, so every CI cell that runs another one loads a file written by
another interpreter. Loading needs no torch, and neither does this module:
it runs in every cell, those without torch included.
"""

import json

import numpy as np
import pytest

from pyvbmc.svbmc import SVBMC
from pyvbmc.testing.svbmc._fixtures import (
    SAVED_DIR,
    SAVED_GROUP,
    assert_roundtrip,
)

PATH = SAVED_DIR / f"{SAVED_GROUP}.pkl"


@pytest.fixture(scope="module")
def reference():
    return json.loads(PATH.with_suffix(".json").read_text(encoding="utf-8"))


@pytest.fixture
def stacked():
    return SVBMC.load(PATH)


def test_the_saved_stack_holds_its_state(stacked, reference):
    assert (stacked.D, stacked.M, stacked.K) == (
        reference["D"],
        reference["M"],
        reference["K"],
    )
    assert stacked.w.dtype == np.float64
    np.testing.assert_array_equal(stacked.w, np.array([reference["w"]]))
    assert stacked.elbo == reference["elbo"]
    assert stacked.elbo_sd == reference["elbo_sd"]
    assert stacked.entropy == reference["entropy"]
    details = dict(stacked.elbo_details)
    details["noise_status_source"] = list(details["noise_status_source"])
    assert details == reference["elbo_details"]
    # The retained posteriors are the fixture posteriors of the group.
    assert len(stacked.vp_list) == len(reference["posteriors"])
    for vp, name in zip(stacked.vp_list, reference["posteriors"]):
        assert_roundtrip(vp, name)


def test_the_saved_stack_draws_as_when_it_was_saved(stacked, reference):
    """The draws go through the bounded transforms, which the transformer
    rebuilds on loading; the tolerance leaves room for the rounding of
    another platform's special functions."""
    stacked.rng = np.random.default_rng(reference["draw_seed"])
    X = stacked.sample(len(reference["draws"]))
    np.testing.assert_allclose(X, reference["draws"], rtol=1e-12, atol=1e-12)


def test_the_saved_stack_saves_again(stacked, tmp_path):
    path = tmp_path / "again.pkl"
    stacked.save(path)
    assert b"_create_function" not in path.read_bytes()
    again = SVBMC.load(path)
    np.testing.assert_array_equal(again.w, stacked.w)
    assert again.elbo == stacked.elbo
