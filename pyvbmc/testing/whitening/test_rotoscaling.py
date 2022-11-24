import os

import gpyreg
import numpy as np
import pytest
import scipy.stats as st

from pyvbmc import VBMC
from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.whitening import unscent_warp, warp_gp_and_vp, warp_input

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
