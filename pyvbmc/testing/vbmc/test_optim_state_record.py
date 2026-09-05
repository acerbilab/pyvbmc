"""What the iteration history keeps of ``optim_state``.

The importance samples of the noisy acquisitions
(``optim_state["active_importance_sampling"]``) are drawn afresh in every
active-sampling step, so the history's copy of ``optim_state`` leaves them
out unless the option ``record_full_history_details`` asks to keep them.
"""

import numpy as np

from pyvbmc import VBMC

D = 3


def make_vbmc(**options):
    return VBMC(
        lambda x: np.sum(x + 2),
        np.ones((2, D)) * 3,
        np.ones((1, D)) * 1,
        np.ones((1, D)) * 5,
        np.ones((1, D)) * 2,
        np.ones((1, D)) * 4,
        options=options,
    )


def importance_samples():
    return {
        "X": np.arange(6.0).reshape(2, D),
        "ln_weights": np.zeros((1, 2)),
        "f_s2": np.ones((2, 1)),
        "K_Xa_X": np.ones((1, 2, 4)),
        "C_tmp": np.ones((1, 4, 2)),
    }


def test_importance_samples_are_left_out_by_default():
    vbmc = make_vbmc()
    assert vbmc.options["record_full_history_details"] is False
    vbmc.optim_state["active_importance_sampling"] = importance_samples()

    record = vbmc._optim_state_record()

    assert record is not vbmc.optim_state
    assert record["active_importance_sampling"] is None
    # Every other entry is the live one; the live dict is untouched.
    for key, value in vbmc.optim_state.items():
        if key != "active_importance_sampling":
            assert record[key] is value
    assert set(record) == set(vbmc.optim_state)
    assert vbmc.optim_state["active_importance_sampling"] is not None
    assert vbmc.optim_state["active_importance_sampling"]["X"].shape == (2, D)


def test_importance_samples_are_kept_when_asked():
    vbmc = make_vbmc(record_full_history_details=True)
    vbmc.optim_state["active_importance_sampling"] = importance_samples()

    record = vbmc._optim_state_record()

    assert record is vbmc.optim_state


def test_record_is_the_live_state_without_importance_samples():
    """Noiseless runs never set the key; the record is the live dict."""
    vbmc = make_vbmc()
    assert vbmc.optim_state.get("active_importance_sampling") is None

    assert vbmc._optim_state_record() is vbmc.optim_state


def test_recorded_history_copy_has_no_importance_samples():
    """Through ``IterationHistory.record`` the copy stored is a deep copy of
    the record, so it carries ``None`` while the live state keeps the arrays.
    """
    vbmc = make_vbmc()
    vbmc.optim_state["active_importance_sampling"] = importance_samples()

    vbmc.iteration_history.record("optim_state", vbmc._optim_state_record(), 0)

    stored = vbmc.iteration_history["optim_state"][0]
    assert stored["active_importance_sampling"] is None
    assert vbmc.optim_state["active_importance_sampling"]["X"].shape == (2, D)
    assert stored is not vbmc.optim_state
