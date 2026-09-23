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


def build_state(
    n_points=6, dropped=(2,), counts=(3, 5, 6), options=None, hyp=None
):
    """A VBMC instance with a logged training set and a fitted process.

    ``dropped`` names the logged points that a warm-up trim removed from
    the training set, and ``counts`` the number of logged points at the
    end of each recorded iteration. The logged values rise with the point
    index, so the running maximum rises along the training set. ``hyp``
    holds the hyperparameters of the process, ``[log_ell, log_sf, log_sn,
    m0, xm, log_omega]``, a small noise by default.
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
    if hyp is None:
        hyp = np.array([0.0, 1.0, -5.0, 0.0, 0.0, 0.0])
    gp.update(
        X_new=vbmc.function_logger.X[in_training_set, :],
        y_new=vbmc.function_logger.y[in_training_set],
        hyp=np.atleast_2d(hyp),
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


def gp_posterior_at_training_inputs(X, y, s2, hyp):
    """The latent posterior mean and variance of a one-dimensional GP at its
    own training inputs, written out from the formulas of GP regression.

    The kernel is the squared exponential, the mean the negative quadratic
    and the noise Gaussian with variance ``exp(2 * log_sn) + s2`` at each
    input, where ``s2`` is the noise variance given per input (zero for
    none). ``hyp`` is ``[log_ell, log_sf, log_sn, m0, xm, log_omega]``.
    """
    log_ell, log_sf, log_sn, m0, xm, log_omega = hyp
    x = np.ravel(X)
    distance = (x[:, None] - x[None, :]) / np.exp(log_ell)
    K = np.exp(2 * log_sf) * np.exp(-0.5 * distance**2)
    mean = m0 - 0.5 * ((x - xm) / np.exp(log_omega)) ** 2
    noise = np.exp(2 * log_sn) + np.broadcast_to(np.ravel(s2), x.shape)
    A = K + np.diag(noise)
    f_mu = mean + K @ np.linalg.solve(A, np.ravel(y) - mean)
    f_s2 = np.diag(K) - np.einsum("ij,ji->i", K, np.linalg.solve(A, K))
    return f_mu, f_s2


def running_maxima(bounds, vbmc, counts):
    """The largest of ``bounds``, given at the training inputs in the order
    of the logged points, over the points logged up to each count."""
    in_training_set = vbmc.function_logger.X_flag[
        : vbmc.function_logger.Xn + 1
    ]
    at_row = np.full(in_training_set.size, -np.inf)
    at_row[in_training_set] = bounds
    return np.array([np.max(at_row[:count]) for count in counts])


def build_noisy_state(noise_sd, dropped, counts, hyp):
    """A VBMC instance on a target that returns the standard deviation of
    its noise, with one point logged for each entry of ``noise_sd``, which
    is that point's standard deviation, and a GP conditioned on their
    variances, as a noisy run conditions its GP."""
    vbmc = VBMC(
        lambda x: (-0.5 * float(np.sum(np.asarray(x) ** 2)), 1.0),
        np.zeros((1, D)),
        np.full((1, D), -np.inf),
        np.full((1, D), np.inf),
        np.full((1, D), -3.0),
        np.full((1, D), 3.0),
        options={"display": "off", "specify_target_noise": True},
        seed=20260923,
    )
    xs = np.linspace(-2.5, 2.5, len(noise_sd))
    for i, (x, sd) in enumerate(zip(xs, noise_sd)):
        vbmc.function_logger.add(np.array([x]), float(i), sd)
    for i in dropped:
        vbmc.function_logger.X_flag[i] = False

    logger = vbmc.function_logger
    assert logger.noise_flag
    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(
            constant_add=True, user_provided_add=True
        ),
    )
    gp.update(
        X_new=logger.X[logger.X_flag, :],
        y_new=logger.y[logger.X_flag],
        s2_new=logger.S[logger.X_flag] ** 2,
        hyp=np.atleast_2d(hyp),
    )
    vbmc.gp = gp
    for iteration, count in enumerate(counts):
        vbmc.iteration_history.record("N", count, iteration)
    return vbmc


def test_a_noisy_run_recomputes_with_the_noise_of_each_point():
    """On a target that returns the noise of each value, the GP of the run
    is conditioned on those per-point variances, and the recomputed maxima
    are the bounds of that GP, which differ from those of a GP that ignores
    them. The variances reach the bound through that conditioning alone:
    MATLAB passes ``s2 = S.^2`` to ``gplite_pred``
    (``private/recompute_lcbmax.m:9-16``), which reads it only for the
    predictive variance with the noise added (``gplite/gplite_pred.m:63``),
    and so does gpyreg's ``GP.predict``."""
    noise_sd = np.array([0.3, 1.5, 0.6, 2.0, 0.9, 0.2, 1.2])
    dropped = (1,)
    counts = (3, 5, 7)
    hyp = np.array([0.0, 1.0, -5.0, 0.0, 0.0, 0.0])
    vbmc = build_noisy_state(noise_sd, dropped, counts, hyp)

    logger = vbmc.function_logger
    X = logger.X[logger.X_flag, :]
    y = logger.y[logger.X_flag]
    weight = vbmc.options.get("elcbo_impro_weight")

    def maxima(s2):
        f_mu, f_s2 = gp_posterior_at_training_inputs(X, y, s2, hyp)
        return running_maxima(f_mu - weight * np.sqrt(f_s2), vbmc, counts)

    with_noise = maxima(noise_sd[logger.X_flag[: len(noise_sd)]] ** 2)
    without_noise = maxima(0.0)

    recomputed = vbmc._recompute_lcb_max()

    np.testing.assert_allclose(recomputed, with_noise, rtol=1e-9, atol=1e-9)
    assert np.all(np.abs(with_noise - without_noise) > 1.0)


def test_the_recomputed_bound_takes_the_latent_variance():
    """The bound is ``fmu - ELCBOImproWeight * sqrt(fs2)`` with the latent
    mean and variance, the third and fourth outputs of ``gplite_pred``
    (``private/recompute_lcbmax.m:16-18``), not the predictive variance
    with the observation noise added. With a noise of unit standard
    deviation the two bounds are far apart."""
    counts = (3, 5, 6)
    hyp = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])  # log_sn = 0
    vbmc = build_state(counts=counts, dropped=(2,), hyp=hyp)
    logger = vbmc.function_logger
    X = logger.X[logger.X_flag, :]
    y = logger.y[logger.X_flag]
    weight = vbmc.options.get("elcbo_impro_weight")

    f_mu, f_s2 = gp_posterior_at_training_inputs(X, y, 0.0, hyp)
    latent = running_maxima(f_mu - weight * np.sqrt(f_s2), vbmc, counts)
    noise_added = running_maxima(
        f_mu - weight * np.sqrt(f_s2 + np.exp(2 * hyp[2])), vbmc, counts
    )

    recomputed = vbmc._recompute_lcb_max()

    np.testing.assert_allclose(recomputed, latent, rtol=1e-9, atol=1e-9)
    assert np.all(latent - noise_added > 1.0)


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
