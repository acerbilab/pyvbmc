import importlib

import pytest

from pyvbmc.vbmc import IterationHistory

# The package exports the function under the name of its module, so the
# module is taken from the import system.
animation_module = importlib.import_module("pyvbmc.vbmc.create_vbmc_animation")


def _history_of_three_iterations():
    history = IterationHistory(["iter", "logging_action"])
    actions = [["start warm-up"], [], ["end warm-up", "trim data"]]
    for iteration, logged in enumerate(actions):
        history.record_iteration(
            {"iter": iteration, "logging_action": logged}, iteration
        )
    return history


def test_full_titles_give_the_actions_of_the_iteration():
    """``"full"`` names the logged actions of an iteration, joined as the
    iteration log joins them, and the iteration alone where there are
    none."""
    history = _history_of_three_iterations()
    title = animation_module._frame_title
    assert title(history, 0, "full") == "PyVBMC iteration 0 (start warm-up)"
    assert title(history, 1, "full") == "PyVBMC iteration 1"
    assert (
        title(history, 2, "full")
        == "PyVBMC iteration 2 (end warm-up, trim data)"
    )


def test_the_other_titles():
    """``"iteration"`` gives the iteration alone and ``"none"`` nothing;
    the frame after the last iteration, the final posterior, is titled with
    the number of iterations whatever the choice."""
    history = _history_of_three_iterations()
    title = animation_module._frame_title
    assert title(history, 0, "iteration") == "PyVBMC iteration 0"
    assert title(history, 2, "iteration") == "PyVBMC iteration 2"
    assert title(history, 0, "none") is None
    for suptitle in ("full", "iteration", "none"):
        assert title(history, 3, suptitle) == "PyVBMC final (3 iterations)"


def test_an_unsupported_suptitle_is_refused_before_any_plot(tmp_path):
    with pytest.raises(ValueError, match="Unsupported suptitle"):
        animation_module.create_vbmc_animation(
            None, tmp_path / "animation.gif", suptitle="title"
        )
