"""Regression references of the stacked ELBO, group by group and mode.

Every cell replays the recipe stored in ``references.json``:
``SVBMC(load_group(group, rng=0)[0], seed=0).optimize(n_samples=20,
n_samples_final=100, lr=0.1, max_steps=3, version=mode)``, every argument
taken from the sidecar. A failure means the numerics of the stacking moved (the
weights, the ELBO or the Monte Carlo entropy), or that its random stream
did; ``FIXTURES.md`` records how the references were generated.
"""

import logging

import numpy as np
import pytest

pytest.importorskip("torch")

from pyvbmc.svbmc import SVBMC  # noqa: E402
from pyvbmc.testing.svbmc._fixtures import (  # noqa: E402
    load_group,
    load_references,
)

REFERENCES = load_references()
STEPS = int(REFERENCES["meta"]["max_steps"])
SEED = int(REFERENCES["meta"]["seed"])
N_SAMPLES = int(REFERENCES["meta"]["n_samples"])
N_SAMPLES_FINAL = int(REFERENCES["meta"]["n_samples_final"])
LR = float(REFERENCES["meta"]["lr"])
CELLS = [
    (group, mode)
    for group in sorted(REFERENCES["groups"])
    for mode in sorted(REFERENCES["groups"][group])
]
TOLERANCE = dict(rtol=1e-8, atol=1e-10)


@pytest.fixture(scope="module", autouse=True)
def quiet_logger():
    logger = logging.getLogger("SVBMC")
    previous = logger.level
    logger.setLevel(logging.WARNING)
    yield
    logger.setLevel(previous)


def test_every_group_has_references():
    assert len(CELLS) == 18
    assert {mode for _, mode in CELLS} == {
        "all-weights",
        "posterior-only",
        "ns",
    }


@pytest.mark.parametrize("group,mode", CELLS)
def test_matches_reference(group, mode):
    reference = REFERENCES["groups"][group][mode]

    vps = load_group(group, rng=0)[0]
    stacked = SVBMC(vps, seed=SEED)
    stacked.optimize(
        n_samples=N_SAMPLES,
        n_samples_final=N_SAMPLES_FINAL,
        lr=LR,
        max_steps=STEPS,
        version=mode,
    )

    assert stacked.M == reference["M"]
    assert list(stacked.K) == list(reference["K"])
    assert stacked.w.shape == reference["w"].shape

    np.testing.assert_allclose(
        stacked.w, reference["w"], err_msg="weights", **TOLERANCE
    )
    np.testing.assert_allclose(stacked.elbo, reference["elbo"], **TOLERANCE)
    np.testing.assert_allclose(
        stacked.elbo_sd, reference["elbo_sd"], **TOLERANCE
    )
    assert set(stacked.elbo_details) == set(reference["elbo_details"])
    for key, expected in reference["elbo_details"].items():
        actual = stacked.elbo_details[key]
        if key == "noise_status_source":
            assert list(actual) == expected
            continue
        if isinstance(expected, (str, bool)):
            assert actual == expected
            continue
        np.testing.assert_allclose(actual, expected, err_msg=key, **TOLERANCE)
    np.testing.assert_allclose(
        stacked.entropy, reference["entropy"], err_msg="entropy", **TOLERANCE
    )
