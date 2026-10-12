"""Single live VBMC loop smoke for the PyMC adapter."""

import numpy as np
import pytest

pytest.importorskip("pymc")
pytest.importorskip("arviz_base")

from pyvbmc import VBMC
from pyvbmc.pymc import PyMCTarget
from pyvbmc.testing.pymc.models import scalar_model


def test_short_seeded_adapter_optimization():
    spec = scalar_model(np.random.default_rng(901))
    target = PyMCTarget(spec["model"], seed=902)
    vbmc = VBMC(
        target,
        options={
            "display": "off",
            "max_iter": 3,
            "max_fun_evals": 40,
        },
        seed=903,
    )
    vp, results = vbmc.optimize()
    assert np.isfinite(results["elbo"])
    data = target.to_arviz(vp, 8)
    assert set(data["posterior"].data_vars) == {"mu"}
    assert data["posterior"].mu.shape == (1, 8)
