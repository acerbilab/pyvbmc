import copy
import os

import gpyreg
import numpy as np
import pytest
import scipy.stats as st

from pyvbmc import VBMC
from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.whitening import unscent_warp, warp_gp_and_vp, warp_input
from pyvbmc.whitening.whitening import _drop_low_correlations

D = 2


def test_rotoscaling_rotation_2d():
    angle = np.random.uniform(0, 0.9 * np.pi)
    R = np.array(
        [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
    )
    rands = np.random.normal(size=(50, 2))
    rands[:, 0] = 10 * rands[:, 0]
    mus = rands @ R
    vp = VariationalPosterior(D, 50, mus)
    vbmc = VBMC(
        lambda x: np.sum(x),
        np.ones((1, D)),
        np.full((1, D), -np.inf),
        np.full((1, D), np.inf),
        np.ones((1, D)) * -10,
        np.ones((1, D)) * 10,
    )
    vbmc.vp = vp
    parameter_transformer, __, __, __ = warp_input(
        vp, vbmc.optim_state, vbmc.function_logger, vbmc.options
    )
    U = parameter_transformer.R_mat

    # U should undo rotation of R, up to a sign:
    assert np.all(np.isclose(np.abs(U @ R), np.eye(D), atol=0.05))


def test_unscent_warp():
    angle = np.pi / 6.6
    R1 = np.array(
        [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
    )
    angle = np.pi / 3
    R2 = np.array(
        [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
    )
    parameter_transformer_1 = ParameterTransformer(
        D, rotation_matrix=R1, scale=np.array([0.9, 0.7])
    )
    parameter_transformer_2 = ParameterTransformer(
        D, rotation_matrix=R2, scale=np.array([1.2, 0.5])
    )

    def warpfun(x):
        # Copy probably unneccesary:
        x = np.copy(x)
        return parameter_transformer_2(parameter_transformer_1.inverse(x))

    sigma = np.array([3.0, 0.7])
    mu = np.array([[1.2, -3.3], [-1.7, 0.5], [-3.4, -np.pi], [5.01, 9.8]])
    [muw, sigmaw, muu] = unscent_warp(warpfun, mu, sigma)
    matlab_result_muw = np.array(
        [
            [1.79786175315009, -2.71880715597597],
            [-1.23028515945097, -1.06548342843230],
            [-1.15442046351576, -7.00874808879672],
            [0.0703468098253310, 16.4174973622584],
        ]
    )
    matlab_result_sigmaw = np.array(
        [
            [1.90565079837152, 3.03363336606057],
            [1.90565079837152, 3.03363336606057],
            [1.90565079837152, 3.03363336606057],
            [1.90565079837152, 3.03363336606057],
        ]
    )

    muu_1 = [
        1.7979,
        -1.2303,
        -1.1544,
        0.0703,
        4.4747,
        1.4466,
        1.5224,
        2.7472,
        -0.879,
        -3.9071,
        -3.8313,
        -2.6065,
        1.4857,
        -1.5425,
        -1.4666,
        -0.2419,
        2.1101,
        -0.9181,
        -0.8422,
        0.3826,
    ]
    muu_2 = [
        -2.7188,
        -1.0655,
        -7.0087,
        16.4175,
        1.4099,
        3.0633,
        -2.8800,
        20.5462,
        -6.8475,
        -5.1942,
        -11.1375,
        12.2888,
        -1.5529,
        0.1004,
        -5.8428,
        17.5834,
        -3.8847,
        -2.2314,
        -8.1747,
        15.2516,
    ]
    matlab_result_muu = np.array(list(zip(muu_1, muu_2))).reshape([5, 4, 2])

    assert np.all(np.isclose(muw, matlab_result_muw, atol=0.0001))
    assert np.all(np.isclose(muu, matlab_result_muu, atol=0.0001))
    assert np.all(np.isclose(sigmaw, matlab_result_sigmaw, atol=0.0001))


def test_unscent_warp_does_not_truncate_an_integer_mean():
    """The sigma points are taken in floating point whatever the dtype of
    the given mean, so an array of integers does not truncate them. Under
    the identity warp the transform returns the mean and the scales it was
    given."""
    x = np.array([[1, 2], [3, 4]])
    sigma = np.array([0.25, 0.25])

    x_warped_mean, x_warped_sigma, __ = unscent_warp(lambda u: u, x, sigma)

    assert np.allclose(x_warped_mean, x)
    assert np.allclose(x_warped_sigma, np.tile(sigma, (2, 1)))


def test_unscent_warp_broadcasts_one_mean_over_several_scales():
    """A single row of `x` against several rows of `sigma` gives one
    result per row of `sigma`, as several rows of `x` against a single row
    of `sigma` give one result per row of `x`."""
    D = 2
    x = np.array([[1.5, -2.0]])
    sigma = np.array([[0.5, 0.25], [1.0, 2.0], [0.125, 3.0]])

    x_warped_mean, x_warped_sigma, x_warped = unscent_warp(
        lambda u: u, x, sigma
    )

    assert x_warped_mean.shape == (3, D)
    assert x_warped_sigma.shape == (3, D)
    assert x_warped.shape == (2 * D + 1, 3, D)
    assert np.allclose(x_warped_mean, np.tile(x, (3, 1)))
    assert np.allclose(x_warped_sigma, sigma)


def test_parameter_transformer_log_abs_det():
    D = 3
    x = np.array([1.0, -3.0, 8.5])
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
    )
    u = parameter_transformer(x)
    # MATLAB result:
    assert np.allclose(
        u, np.array([0.200670695462151, -0.619039208406224, 2.51230562397612])
    )
    log_abs_det = parameter_transformer.log_abs_det_jacobian(u)
    assert np.isclose(log_abs_det, 3.44201837618191)

    # Now with rotation and scale:
    angle = np.pi / 6.6
    R1 = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ]
    )
    angle = np.pi / 3
    R2 = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    R = R1 @ R2
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
        rotation_matrix=R,
        scale=np.array([0.9, 0.7, 2.3]),
    )
    u = parameter_transformer(x)
    assert np.allclose(
        u, np.array([0.689778028799817, 0.181006596372446, 1.09421151292434])
    )
    log_abs_det = parameter_transformer.log_abs_det_jacobian(u)
    assert np.isclose(log_abs_det, 3.81289203952045)


def test_warp_input():
    D = 2
    angle = 1.309355600770139
    R = np.array(
        [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
    )
    filepath = os.path.join(
        os.path.dirname(__file__), "test_warp_input_rands.txt"
    )
    rands = np.loadtxt(filepath, delimiter=",")
    rands[:, 0] = 10 * rands[:, 0]
    mus = rands @ R
    vp = VariationalPosterior(D, 50, mus)
    vbmc = VBMC(
        lambda x: np.sum(x),
        mus,
        np.full((1, D), -np.inf),
        np.full((1, D), np.inf),
        np.ones((1, D)) * -10,
        np.ones((1, D)) * 10,
    )
    # vbmc.vp = vp
    (
        parameter_transformer_warp,
        vbmc.optim_state,
        vbmc.function_logger,
        warp_action,
    ) = warp_input(vp, vbmc.optim_state, vbmc.function_logger, vbmc.options)

    assert np.all(parameter_transformer_warp.lb_orig == [-np.inf, -np.inf])
    assert np.all(parameter_transformer_warp.ub_orig == [np.inf, np.inf])
    assert np.all(parameter_transformer_warp.type == [0, 0])
    assert np.all(parameter_transformer_warp.mu == [0.0, 0.0])
    assert np.all(parameter_transformer_warp.delta == [1.0, 1.0])
    assert np.all(
        np.isclose(
            parameter_transformer_warp.R_mat,
            np.array(
                [
                    [0.278003671780883, -0.960580011491155],
                    [0.960580011491155, 0.278003671780883],
                ]
            ),
        )
    )
    assert np.all(
        np.isclose(
            parameter_transformer_warp.scale,
            np.array([11.0521101052146, 1.00626951493545]),
        )
    )


def _cov_reg_state():
    """A posterior with correlated coordinates and a ``VBMC`` instance
    whose state records 42 training points, for the regularization of the
    covariance of the warp."""
    D = 2
    angle = 1.309355600770139
    R = np.array(
        [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
    )
    filepath = os.path.join(
        os.path.dirname(__file__), "test_warp_input_rands.txt"
    )
    rands = np.loadtxt(filepath, delimiter=",")
    rands[:, 0] = 10 * rands[:, 0]
    mus = rands @ R
    vp = VariationalPosterior(D, 50, mus)
    vbmc = VBMC(
        lambda x: np.sum(x),
        mus,
        np.full((1, D), -np.inf),
        np.full((1, D), np.inf),
        np.ones((1, D)) * -10,
        np.ones((1, D)) * 10,
    )
    vbmc.optim_state["N"] = 42
    return vp, vbmc


def test_warp_input_cov_reg():
    """The covariance regularization is a number, of any scalar type, or
    a function of the number of training points, which may return its
    number in an array of one element."""
    vp, vbmc = _cov_reg_state()
    seen = []

    def cov_reg_of_N(N):
        seen.append(N)
        return 0.75

    transforms = []
    for value in (
        0.75,
        np.float64(0.75),
        np.array(0.75),
        cov_reg_of_N,
        lambda N: np.array([0.75]),
    ):
        vbmc.options.__setitem__("warp_cov_reg", value, force=True)
        parameter_transformer_warp, _, _, _ = warp_input(
            vp, vbmc.optim_state, vbmc.function_logger, vbmc.options
        )
        transforms.append(
            (
                parameter_transformer_warp.R_mat,
                parameter_transformer_warp.scale,
            )
        )

    # The callable was given the number of training points, and all the
    # forms of the same amount give the same transform.
    assert seen == [42]
    for R_mat, scale in transforms[1:]:
        assert np.array_equal(R_mat, transforms[0][0])
        assert np.array_equal(scale, transforms[0][1])

    # The amount reaches the transform: no regularization gives another.
    vbmc.options.__setitem__("warp_cov_reg", 0.0, force=True)
    unregularized, _, _, _ = warp_input(
        vp, vbmc.optim_state, vbmc.function_logger, vbmc.options
    )
    assert not np.allclose(unregularized.R_mat, transforms[0][0])


@pytest.mark.parametrize(
    "returned",
    [None, "0.75", 0.75 + 0.1j, True, np.nan, np.inf, np.array([0.5, 0.5])],
    ids=["None", "str", "complex", "bool", "nan", "inf", "two-elements"],
)
def test_warp_cov_reg_function_must_return_a_finite_number(returned):
    """The value that a function of the number of training points returns
    is checked where it is used, at the warp: anything but a finite real
    number (in an array of one element or not) is refused with an error
    that names the option, the number of training points it was given and
    what it returned."""
    vp, vbmc = _cov_reg_state()
    vbmc.options.__setitem__("warp_cov_reg", lambda N: returned, force=True)
    with pytest.raises(ValueError) as execinfo:
        warp_input(vp, vbmc.optim_state, vbmc.function_logger, vbmc.options)
    message = execinfo.value.args[0]
    assert "warp_cov_reg" in message
    assert "42" in message
    assert repr(returned) in message


@pytest.mark.parametrize(
    "value", [None, True, np.nan, [0.75]], ids=["None", "bool", "nan", "list"]
)
def test_warp_cov_reg_written_into_built_options_is_checked_at_use(value):
    """Construction refuses such values; one written into the options of a
    built instance is refused at the warp, with an error that names the
    option."""
    vp, vbmc = _cov_reg_state()
    vbmc.options.__setitem__("warp_cov_reg", value, force=True)
    with pytest.raises(ValueError) as execinfo:
        warp_input(vp, vbmc.optim_state, vbmc.function_logger, vbmc.options)
    assert "warp_cov_reg" in execinfo.value.args[0]


def _unbounded_vbmc(D):
    """A `VBMC` instance on an unbounded problem of dimension ``D``."""
    return VBMC(
        lambda x: np.sum(x),
        np.zeros((1, D)),
        np.full((1, D), -np.inf),
        np.full((1, D), np.inf),
        np.ones((1, D)) * -10,
        np.ones((1, D)) * 10,
    )


def _posterior_with_covariance(cov):
    """A posterior whose covariance is ``cov``.

    The covariance is the only quantity of the posterior the whitening
    transformation is computed from, so it is supplied directly. The
    posterior's own transformation is the identity, so ``cov`` is its
    covariance in the original space as well.
    """
    D = cov.shape[0]
    vp = VariationalPosterior(D, 2, np.zeros((2, D)))

    def moments(orig_flag=True, cov_flag=False):
        return np.zeros((1, D)), np.copy(cov)

    vp.moments = moments
    return vp


def _whitening_map(parameter_transformer, D):
    """The matrix of the linear map into the whitened inference space."""
    return parameter_transformer(np.eye(D)) - parameter_transformer(
        np.zeros((1, D))
    )


def test_warp_input_keeps_a_covariance_the_threshold_makes_indefinite():
    """Dropping the low-correlation entries of a covariance matrix can
    leave one that is not positive definite, and the whitening
    transformation computed from such a matrix does not whiten. The
    covariance before the threshold is used instead, so that the posterior
    has unit variance along every coordinate of the new inference space."""
    D = 3
    cov = np.array([[1.0, 0.72, 0.04], [0.72, 1.0, 0.72], [0.04, 0.72, 1.0]])
    vbmc = _unbounded_vbmc(D)
    thresh = vbmc.options["warp_roto_corr_thresh"]
    thresholded = np.copy(cov)
    thresholded[np.abs(cov) <= thresh] = 0
    assert np.min(np.linalg.eigvalsh(thresholded)) < 0

    parameter_transformer_warp, __, __, __ = warp_input(
        _posterior_with_covariance(cov),
        vbmc.optim_state,
        vbmc.function_logger,
        vbmc.options,
    )

    linear_map = _whitening_map(parameter_transformer_warp, D)
    attained = linear_map.T @ cov @ linear_map
    assert np.allclose(np.diag(attained), np.ones(D))


def test_warp_input_whitens_the_thresholded_covariance_when_it_is_definite():
    """Where the thresholded covariance is positive definite, it is the
    matrix the whitening transformation is computed from."""
    D = 3
    sd = np.array([np.sqrt(2.0), 1.0, np.sqrt(0.5)])
    corr = np.array([[1.0, 0.3, 0.02], [0.3, 1.0, 0.3], [0.02, 0.3, 1.0]])
    cov = np.outer(sd, sd) * corr
    vbmc = _unbounded_vbmc(D)
    thresh = vbmc.options["warp_roto_corr_thresh"]
    thresholded = np.copy(cov)
    thresholded[np.abs(corr) <= thresh] = 0
    assert np.min(np.linalg.eigvalsh(thresholded)) > 0

    parameter_transformer_warp, __, __, __ = warp_input(
        _posterior_with_covariance(cov),
        vbmc.optim_state,
        vbmc.function_logger,
        vbmc.options,
    )

    rotation, singular_values, __ = np.linalg.svd(thresholded)
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] = -rotation[:, 0]
    assert np.array_equal(parameter_transformer_warp.R_mat, rotation)
    assert np.array_equal(
        parameter_transformer_warp.scale,
        np.sqrt(singular_values + np.finfo(np.float64).eps),
    )
    # The entries dropped by the threshold make a difference: the
    # untouched covariance gives another transformation.
    untouched, __, __ = np.linalg.svd(cov)
    assert not np.allclose(parameter_transformer_warp.R_mat, untouched)


def test_drop_low_correlations_keeps_only_strong_correlations():
    """An entry of the covariance survives the threshold only where the
    absolute value of its correlation exceeds it."""
    sd = np.array([np.sqrt(2.0), 1.0, np.sqrt(0.5)])
    corr = np.array([[1.0, 0.3, 0.02], [0.3, 1.0, 0.3], [0.02, 0.3, 1.0]])
    cov = np.outer(sd, sd) * corr

    dropped = _drop_low_correlations(cov, 0.05)

    expected = np.copy(cov)
    expected[0, 2] = 0
    expected[2, 0] = 0
    assert np.array_equal(dropped, expected)
    # The threshold is exclusive: an entry exactly at it goes too.
    unit = np.array([[1.0, 0.2], [0.2, 1.0]])
    assert np.array_equal(_drop_low_correlations(unit, 0.2), np.eye(2))


def test_drop_low_correlations_drops_an_undefined_correlation():
    """A correlation that is not a number does not exceed the threshold,
    so its entry is dropped with the weakly correlated ones."""
    # A negative variance leaves the correlations of that coordinate
    # undefined.
    cov = np.array([[1.0, 0.3, 0.02], [0.3, -1.0, 0.3], [0.02, 0.3, 1.0]])
    with np.errstate(invalid="ignore"):
        dropped = _drop_low_correlations(cov, 0.05)
    assert np.array_equal(dropped, np.diag([1.0, -1.0, 1.0]))


def test_warp_input_search_cache():
    """A populated search cache is warped into the new space."""
    D = 2
    angle = 1.309355600770139
    R = np.array(
        [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
    )
    filepath = os.path.join(
        os.path.dirname(__file__), "test_warp_input_rands.txt"
    )
    rands = np.loadtxt(filepath, delimiter=",")
    rands[:, 0] = 10 * rands[:, 0]
    mus = rands @ R
    vp = VariationalPosterior(D, 50, mus)
    vbmc = VBMC(
        lambda x: np.sum(x),
        mus,
        np.full((1, D), -np.inf),
        np.full((1, D), np.inf),
        np.ones((1, D)) * -10,
        np.ones((1, D)) * 10,
    )
    search_cache = np.linspace(-1.0, 1.0, 4 * D).reshape(4, D)
    vbmc.optim_state["search_cache"] = np.copy(search_cache)

    (
        parameter_transformer_warp,
        optim_state,
        _,
        _,
    ) = warp_input(vp, vbmc.optim_state, vbmc.function_logger, vbmc.options)

    expected = parameter_transformer_warp(
        vbmc.function_logger.parameter_transformer.inverse(search_cache)
    )
    assert optim_state["search_cache"].shape == (4, D)
    assert np.allclose(optim_state["search_cache"], expected)


@pytest.mark.parametrize("bound", [np.inf, 40.0])
def test_warp_input_rewrites_every_filled_row_of_the_logger(bound):
    """The warp re-expresses the stored points in the new inference space.

    Every filled row of the function logger is rewritten, whether or not it
    is active, so that ``X`` remains the new transform of ``X_orig`` and
    ``y`` the stored original-space value plus the new log-Jacobian. On a
    bounded problem the log-Jacobian differs from point to point, so each
    row has to get its own.
    """
    D = 2
    angle = 1.309355600770139
    R = np.array(
        [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
    )
    filepath = os.path.join(
        os.path.dirname(__file__), "test_warp_input_rands.txt"
    )
    rands = np.loadtxt(filepath, delimiter=",")
    rands[:, 0] = 10 * rands[:, 0]
    mus = rands @ R
    vbmc = VBMC(
        lambda x: np.sum(x),
        mus,
        np.full((1, D), -bound),
        np.full((1, D), bound),
        np.ones((1, D)) * -10,
        np.ones((1, D)) * 10,
    )
    vp = VariationalPosterior(
        D, 50, mus, parameter_transformer=vbmc.parameter_transformer
    )
    function_logger = vbmc.function_logger
    points = np.array(
        [[0.3, -0.7], [1.1, 0.2], [-0.5, 0.9], [2.0, -1.4]], dtype=float
    )
    for i, point in enumerate(points):
        function_logger.add(point, -0.5 * (i + 1))
    # A row deactivated, as the trim at the end of warm-up deactivates one.
    function_logger.X_flag[1] = False

    __, __, warped_logger, __ = warp_input(
        vp, vbmc.optim_state, function_logger, vbmc.options
    )
    warped_transformer = warped_logger.parameter_transformer

    assert warped_logger.Xn == len(points) - 1
    # The warp does not revive or retire a row.
    assert np.array_equal(
        warped_logger.X_flag[: warped_logger.Xn + 1],
        [True, False, True, True],
    )

    filled = slice(0, warped_logger.Xn + 1)
    X_orig = warped_logger.X_orig[filled]
    expected_X = warped_transformer(X_orig)
    log_jacobian = warped_transformer.log_abs_det_jacobian(expected_X)
    expected_y = warped_logger.y_orig[filled, 0] + log_jacobian
    if np.isfinite(bound):
        assert np.unique(log_jacobian).size == len(points)

    assert np.allclose(warped_logger.X[filled], expected_X)
    assert np.allclose(warped_logger.y[filled, 0], expected_y)
    # The rewrite is not vacuous: the warp moved every stored point.
    assert not np.any(np.isclose(warped_logger.X[filled], points))


def _same_posterior_in_a_rescaled_space(vp, scale):
    """Return the same distribution, expressed in another inference space.

    The space differs from the one ``vp`` is given in by a per-coordinate
    rescaling, so the component means and the scale vector divide by it and
    every other parameter is unchanged.
    """
    other = copy.deepcopy(vp)
    other.parameter_transformer = copy.deepcopy(vp.parameter_transformer)
    other.parameter_transformer.scale = np.asarray(scale, dtype=float)
    other.mu = vp.mu / np.reshape(scale, (-1, 1))
    other.lambd = vp.lambd / np.reshape(scale, (-1, 1))
    return other


def test_warp_input_inverts_the_search_state_with_the_current_transform():
    """The search bounds and the cached search points are points of the
    inference space the run is in, so they are inverted with that space's
    transformation. A posterior recorded before an earlier warp carries
    another one, and handing it in gives the same search state as handing
    in a posterior of the current space."""
    D = 2
    angle = 1.309355600770139
    R = np.array(
        [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
    )
    filepath = os.path.join(
        os.path.dirname(__file__), "test_warp_input_rands.txt"
    )
    rands = np.loadtxt(filepath, delimiter=",")
    rands[:, 0] = 10 * rands[:, 0]
    mus = rands @ R
    vbmc = VBMC(
        lambda x: np.sum(x),
        mus,
        np.full((1, D), -np.inf),
        np.full((1, D), np.inf),
        np.ones((1, D)) * -10,
        np.ones((1, D)) * 10,
    )
    search_cache = np.linspace(-1.0, 1.0, 4 * D).reshape(4, D)
    vbmc.optim_state["search_cache"] = np.copy(search_cache)

    current = VariationalPosterior(
        D, 50, mus, parameter_transformer=vbmc.parameter_transformer
    )
    recorded_elsewhere = _same_posterior_in_a_rescaled_space(
        current, [2.0, 5.0]
    )
    assert not np.allclose(
        recorded_elsewhere.parameter_transformer.scale, np.ones(D)
    )
    # Both calls start from the same draws.
    current.rng = 20260920
    recorded_elsewhere.rng = 20260920

    from_current = warp_input(
        current, vbmc.optim_state, vbmc.function_logger, vbmc.options
    )
    from_elsewhere = warp_input(
        recorded_elsewhere,
        vbmc.optim_state,
        vbmc.function_logger,
        vbmc.options,
    )

    # The two posteriors describe the same distribution, so they give the
    # same whitening transformation; only the space the search state is
    # inverted from could differ.
    assert np.allclose(from_current[0].R_mat, from_elsewhere[0].R_mat)
    assert np.allclose(from_current[0].scale, from_elsewhere[0].scale)

    for key in ("lb_search", "ub_search", "search_cache"):
        assert np.allclose(from_current[1][key], from_elsewhere[1][key])


def test_warp_gp_and_vp():
    D = 2
    angle = 1.309355600770139
    R = np.array(
        [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
    )
    filepath = os.path.join(
        os.path.dirname(__file__), "test_warp_input_rands.txt"
    )
    rands = np.loadtxt(filepath, delimiter=",")
    rands[:, 0] = 10 * rands[:, 0]
    mus = rands @ R
    vp = VariationalPosterior(D, 50, mus)
    vbmc = VBMC(
        lambda x: st.multivariate_normal.logpdf(x),
        mus,
        np.full((1, D), -np.inf),
        np.full((1, D), np.inf),
        np.ones((1, D)) * -10,
        np.ones((1, D)) * 10,
    )
    (
        parameter_transformer_warp,
        vbmc.optim_state,
        vbmc.function_logger,
        warp_action,
    ) = warp_input(vp, vbmc.optim_state, vbmc.function_logger, vbmc.options)

    filepath = os.path.join(
        os.path.dirname(__file__), "test_warp_gp_and_vp_gp_X.txt"
    )
    gp_X = np.loadtxt(filepath, delimiter=",")
    filepath = os.path.join(
        os.path.dirname(__file__), "test_warp_gp_and_vp_gp_y.txt"
    )
    gp_y = np.loadtxt(filepath, delimiter=",")
    filepath = os.path.join(
        os.path.dirname(__file__), f"test_warp_gp_and_vp_gp_hyps.txt"
    )
    gp_posterior_hyps = np.atleast_2d(np.loadtxt(filepath, delimiter=","))

    vp.gp = gpyreg.GP(
        D,
        gpyreg.covariance_functions.SquaredExponential(),
        gpyreg.mean_functions.NegativeQuadratic(),
        gpyreg.noise_functions.GaussianNoise(constant_add=True),
    )
    # __ = vp.gp.get_hyperparameters()
    # vp.gp.X = gp_X
    # vp.gp.y = gp_y
    # for i in range(gp_posterior_hyps.shape[1]):
    #     vp.gp.posteriors[i].hyp = gp_posterior_hyps[:, i]
    vp.gp.update(X_new=gp_X, y_new=gp_y, hyp=gp_posterior_hyps)
    vp_new, hyps_new = warp_gp_and_vp(
        parameter_transformer_warp, vp.gp, vp, vbmc
    )

    filepath = os.path.join(
        os.path.dirname(__file__), f"test_warp_gp_and_vp_gp_hyps_new.txt"
    )
    hyps_new_MATLAB = np.atleast_2d(np.loadtxt(filepath, delimiter=","))
    assert np.allclose(hyps_new, hyps_new_MATLAB)

    filepath = os.path.join(
        os.path.dirname(__file__), f"test_warp_gp_and_vp_vp_mu.txt"
    )
    vp_mu_MATLAB = np.atleast_2d(np.loadtxt(filepath, delimiter=","))
    filepath = os.path.join(
        os.path.dirname(__file__), f"test_warp_gp_and_vp_vp_w.txt"
    )
    vp_w_MATLAB = np.atleast_2d(np.loadtxt(filepath, delimiter=","))
    filepath = os.path.join(
        os.path.dirname(__file__), f"test_warp_gp_and_vp_vp_K.txt"
    )
    vp_K_MATLAB = np.atleast_2d(np.loadtxt(filepath, delimiter=","))
    filepath = os.path.join(
        os.path.dirname(__file__), f"test_warp_gp_and_vp_vp_sigma.txt"
    )
    vp_sigma_MATLAB = np.atleast_2d(np.loadtxt(filepath, delimiter=","))
    filepath = os.path.join(
        os.path.dirname(__file__), f"test_warp_gp_and_vp_vp_lambda.txt"
    )
    vp_lambda_MATLAB = np.atleast_2d(np.loadtxt(filepath, delimiter=",")).T

    assert np.allclose(vp_new.mu, vp_mu_MATLAB, atol=1e-5)
    assert np.allclose(vp_new.w, vp_w_MATLAB)
    assert np.allclose(vp_new.K, vp_K_MATLAB)
    assert np.allclose(vp_new.lambd, vp_lambda_MATLAB)
    assert np.allclose(vp_new.sigma, vp_sigma_MATLAB)

    assert np.all(parameter_transformer_warp.lb_orig == [-np.inf, -np.inf])
    assert np.all(parameter_transformer_warp.ub_orig == [np.inf, np.inf])
    assert np.all(parameter_transformer_warp.type == [0, 0])
    assert np.all(parameter_transformer_warp.mu == [0.0, 0.0])
    assert np.all(parameter_transformer_warp.delta == [1.0, 1.0])
    assert np.all(
        np.isclose(
            parameter_transformer_warp.R_mat,
            np.array(
                [
                    [0.278003671780883, -0.960580011491155],
                    [0.960580011491155, 0.278003671780883],
                ]
            ),
        )
    )
    assert np.all(
        np.isclose(
            parameter_transformer_warp.scale,
            np.array([11.0521101052146, 1.00626951493545]),
        )
    )
