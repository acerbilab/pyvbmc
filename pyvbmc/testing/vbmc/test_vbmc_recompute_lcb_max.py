"""Checks on the recomputed running maximum of the lower confidence bound.

Warm-up ends on the history of the largest lower confidence bound of the
log joint reached so far. Each iteration records that maximum as its own
Gaussian process predicted it; the recomputation replaces the whole
sequence with what the current Gaussian process says about the same
points, which is what the two warm-up criteria then read.
"""

import gpyreg as gpr
import numpy as np

from pyvbmc import VBMC

D = 1


def build_state(n_points=6, dropped=(2,), counts=(3, 5, 6), options=None):
    """A VBMC instance with a logged training set and a fitted process.

    ``dropped`` names the logged points that a warm-up trim removed from
    the training set, and ``counts`` the number of logged points at the
    end of each recorded iteration. The logged values rise with the point
    index, so the running maximum rises along the training set.
    """
    settings = {"display": "off"}
    settings.update(options or {})
    vbmc = VBMC(
        lambda x: -0.5 * float(np.sum(np.asarray(x) ** 2)),
        np.zeros((1, D)),
        np.full((1, D), -np.inf),
        np.full((1, D), np.inf),
        np.full((1, D), -3.0),
        np.full((1, D), 3.0),
        options=settings,
        seed=20260920,
    )
    xs = np.linspace(-2.5, 2.5, n_points)
    for i, x in enumerate(xs):
        # The dropped points carry the largest values, so a running
        # maximum that kept them would be visibly larger.
        y = 50.0 if i in dropped else float(i)
        vbmc.function_logger.add(np.array([x]), y)
    for i in dropped:
        vbmc.function_logger.X_flag[i] = False

    in_training_set = vbmc.function_logger.X_flag
    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    hyp = np.array([[0.0, 1.0, -5.0, 0.0, 0.0, 0.0]])
    gp.update(
        X_new=vbmc.function_logger.X[in_training_set, :],
        y_new=vbmc.function_logger.y[in_training_set],
        hyp=hyp,
    )
    vbmc.gp = gp
    for iteration, count in enumerate(counts):
        vbmc.iteration_history.record("N", count, iteration)
    return vbmc


def lower_confidence_bounds(vbmc):
    """The bound at each logged point still in the training set, one
    prediction at a time."""
    bounds = {}
    weight = vbmc.options.get("elcbo_impro_weight")
    for row in range(vbmc.function_logger.Xn + 1):
        if not vbmc.function_logger.X_flag[row]:
            continue
        f_mu, f_s2 = vbmc.gp.predict(
            vbmc.function_logger.X[row : row + 1, :], add_noise=False
        )
        bounds[row] = np.ravel(f_mu)[0] - weight * np.sqrt(np.ravel(f_s2)[0])
    return bounds


def test_recomputed_maximum_is_the_running_maximum_of_the_current_process():
    """Each entry is the largest lower confidence bound, under the current
    Gaussian process, over the points logged up to the end of that
    iteration; points no longer in the training set are passed over."""
    counts = (3, 5, 6)
    vbmc = build_state(counts=counts, dropped=(2,))
    bounds = lower_confidence_bounds(vbmc)

    recomputed = vbmc._recompute_lcb_max()

    assert recomputed.shape == (len(counts),)
    for iteration, count in enumerate(counts):
        expected = max(bound for row, bound in bounds.items() if row < count)
        assert np.isclose(recomputed[iteration], expected)
    # A running maximum never decreases.
    assert np.all(np.diff(recomputed) >= 0)
    # The dropped point held by far the largest logged value: it is not in
    # the recomputed maxima.
    assert np.all(recomputed < 40.0)


def test_recomputed_maximum_reaches_the_warmup_check():
    """The warm-up criteria read the recomputed sequence when there is one
    and the maxima the iterations recorded otherwise (MATLAB VBMC,
    ``private/vbmc_warmup.m:46-50``). Here the recorded maxima are flat,
    which makes the criterion of no long-term improvement fire, while the
    recomputed ones rise to the last iteration, which holds it back."""
    counts = (3, 5, 8)
    vbmc = build_state(n_points=8, counts=counts, dropped=())
    for iteration in range(len(counts)):
        vbmc.iteration_history.record("lcb_max", -20.0, iteration)
        vbmc.iteration_history.record(
            "func_count", 10 * (iteration + 1), iteration
        )
    vbmc.optim_state["iter"] = len(counts) - 1
    vbmc.optim_state["N"] = counts[-1]
    vbmc.optim_state["data_trim_list"] = []
    vbmc.function_logger.func_count = 100
    # 100 - 10 exceeds the threshold, 100 - 30 does not.
    vbmc.options.__setitem__("warmup_no_impro_threshold", 80, force=True)

    recomputed = vbmc._recompute_lcb_max()
    assert recomputed[2] - recomputed[1] > vbmc.options.get("tol_improvement")
    assert recomputed[1] - recomputed[0] > vbmc.options.get("tol_improvement")

    vbmc.optim_state["lcb_max_vec"] = np.array([])
    assert vbmc._check_warmup_end_conditions()

    vbmc.optim_state["lcb_max_vec"] = recomputed
    assert not vbmc._check_warmup_end_conditions()


def _warmup_check_state(lcb_max_vec, func_count):
    """An instance at its third recorded iteration, where only the vector
    of maxima decides the criterion of no long-term improvement."""
    counts = (3, 5, 8)
    vbmc = build_state(n_points=8, counts=counts, dropped=())
    for iteration in range(len(counts)):
        vbmc.iteration_history.record("lcb_max", -20.0, iteration)
        vbmc.iteration_history.record(
            "func_count", 10 * (iteration + 1), iteration
        )
    vbmc.optim_state["iter"] = len(counts) - 1
    vbmc.optim_state["N"] = counts[-1]
    vbmc.optim_state["data_trim_list"] = []
    vbmc.function_logger.func_count = func_count
    vbmc.options.__setitem__("warmup_no_impro_threshold", 80, force=True)
    vbmc.optim_state["lcb_max_vec"] = np.asarray(lcb_max_vec, dtype=float)
    return vbmc


def test_an_iteration_with_no_point_left_has_no_recomputed_maximum():
    """A warm-up trim can drop every point an early iteration logged; the
    running maximum has nothing to read there, as MATLAB's ``movmax`` over
    NaN entries has not."""
    vbmc = build_state(n_points=8, counts=(3, 5, 8), dropped=(0, 1, 2))

    recomputed = vbmc._recompute_lcb_max()

    assert np.isnan(recomputed[0])
    assert np.all(np.isfinite(recomputed[1:]))


def test_warmup_check_passes_over_an_iteration_without_a_maximum():
    """The warm-up criteria take their maxima over the entries that are
    there, as MATLAB's ``max`` does (``private/vbmc_warmup.m:64``,
    ``:75-76``): the iteration without one neither raises nor counts as
    the place of the last improvement."""
    tol = 0.01  # the default tol_improvement

    # Flat after the missing entry: the last improvement is at the second
    # iteration, 105 - 20 evaluations ago, beyond the threshold of 80.
    flat = _warmup_check_state([np.nan, -5.0, -5.0 + 0.5 * tol], 105)
    assert flat.options.get("tol_improvement") == tol
    assert flat._check_warmup_end_conditions()

    # Rising to the last iteration: 105 - 30 evaluations ago, within it.
    rising = _warmup_check_state([np.nan, -5.0, -4.0], 105)
    assert not rising._check_warmup_end_conditions()
