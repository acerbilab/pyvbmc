"""Old saved runs activate acquisition regularization after loading."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyvbmc import VBMC
from pyvbmc.acquisition_functions import AbstractAcqFcn


class _UnitLogAcquisition(AbstractAcqFcn):
    def _compute_acquisition_function(self, Xs, *args):
        return np.ones(Xs.shape[0])


@pytest.mark.parametrize("iteration", [None, 0])
def test_loaded_legacy_state_activates_regularization(iteration):
    path = Path(__file__).parents[1] / "vbmc" / "test_vbmc_save_static.pkl"
    vbmc = VBMC.load(path, iteration=iteration)
    state = vbmc.optim_state
    assert "variance_regularized_acq_fcn" not in state
    assert state["variance_regularized_acqfcn"]
    threshold = state["tol_gp_var"]
    assert threshold > 0
    gp = SimpleNamespace(
        predict=lambda **kwargs: (
            np.zeros((1, 1)),
            np.full((1, 1), threshold / 2),
        )
    )
    acquisition = _UnitLogAcquisition()
    acquisition.acq_info["log_flag"] = True
    actual = acquisition(
        np.zeros((1, vbmc.D)), gp, vbmc.vp, vbmc.function_logger, state
    )
    # Unit raw acquisition plus threshold / variance - 1 = 1.
    np.testing.assert_array_equal(actual, np.array([2.0]))
