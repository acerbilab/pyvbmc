"""Checks on the fields of the results dictionary ``VBMC.optimize`` returns."""

import numpy as np

from pyvbmc import VBMC
from pyvbmc.variational_posterior import VariationalPosterior


def results_of_a_recorded_run(
    lower_bounds, upper_bounds, n_iterations=1, idx_best=0
):
    """Build the results of a run with ``n_iterations`` recorded iterations.

    The iterations are filled in by hand: the fields checked here are
    written from the recorded history and from the bounds, not from the
    numbers a run computes.
    """
    D = np.size(lower_bounds)
    lb = np.reshape(np.asarray(lower_bounds, dtype=float), (1, D))
    ub = np.reshape(np.asarray(upper_bounds, dtype=float), (1, D))
    finite = np.isfinite(lb) & np.isfinite(ub)
    plb = np.where(finite, lb + 1.0, -1.0)
    pub = np.where(finite, ub - 1.0, 1.0)
    vbmc = VBMC(
        lambda x: -0.5 * np.sum(x**2),
        0.5 * (plb + pub),
        lb,
        ub,
        plb,
        pub,
        options={"display": "off"},
    )
    vbmc.vp.stats = {"elbo": -1.0, "elbo_sd": 0.1, "stable": True}
    for iteration in range(n_iterations):
        vbmc.iteration_history.record_iteration(
            {
                "iter": iteration,
                "n_eff": 10,
                "r_index": 0.5,
                "stable": True,
            },
            iteration,
        )
    vbmc.optim_state["iter"] = n_iterations - 1
    return vbmc._create_result_dict(idx_best, "done", True)


def test_iterations_counts_them_and_best_iter_indexes_them():
    """The results report how many iterations the run performed, and the
    0-based index into the iteration history of the one whose variational
    posterior is returned, as the docstrings of the two fields say."""
    unbounded = ([-np.inf, -np.inf], [np.inf, np.inf])
    results = results_of_a_recorded_run(*unbounded, n_iterations=7, idx_best=5)
    assert results["iterations"] == 7
    assert results["best_iter"] == 5

    single = results_of_a_recorded_run(*unbounded, n_iterations=1, idx_best=0)
    assert single["iterations"] == 1
    assert single["best_iter"] == 0


def test_problem_type_reports_the_bounds_the_run_was_given():
    """The reported problem type says whether the problem is bound
    constrained. The transform of a bounded variable sends its bounds to
    minus and plus infinity, so the type is read from the bounds in the
    original coordinates the user gave."""
    assert (
        results_of_a_recorded_run([-5.0, -5.0], [5.0, 5.0])["problem_type"]
        == "bounded"
    )
    assert (
        results_of_a_recorded_run([-5.0, -np.inf], [5.0, np.inf])[
            "problem_type"
        ]
        == "bounded"
    )
    assert (
        results_of_a_recorded_run([-np.inf, -np.inf], [np.inf, np.inf])[
            "problem_type"
        ]
        == "unconstrained"
    )


def test_the_final_plot_is_titled_with_the_number_of_iterations(mocker):
    """The title of the final plot counts the iterations as
    ``results["iterations"]`` does."""
    plot = mocker.patch.object(VariationalPosterior, "plot")
    mocker.patch("matplotlib.pyplot.show")
    D = 2
    vbmc = VBMC(
        lambda x: -0.5 * np.sum(x**2),
        np.zeros((1, D)),
        np.full((1, D), -np.inf),
        np.full((1, D), np.inf),
        np.full((1, D), -1.0),
        np.full((1, D), 1.0),
        options={
            "max_iter": 1,
            "min_iter": 1,
            "do_final_boost": False,
            "plot": True,
            "display": "off",
        },
        seed=1,
    )
    _, results = vbmc.optimize()

    assert results["iterations"] == 1
    assert plot.call_args.kwargs["title"] == "VBMC final (1 iterations)"
