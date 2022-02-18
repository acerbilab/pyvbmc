import numpy as np
import pytest

from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.vbmc import VBMC
from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.whitening.whitening import unscent_warp, warp_input_vbmc

D = 2

def test_rotoscaling_rotation_2d():
    angle = np.random.uniform(0, 0.9*np.pi)
    R = np.array([[np.cos(angle), np.sin(angle)],
                 [-np.sin(angle), np.cos(angle)]]
                )
    rands = np.random.normal(size=(50, 2))
    rands[:,0] = 10*rands[:,0]
    mus   = rands@R
    vp = VariationalPosterior(D, 50, mus)
    vbmc = VBMC(lambda x: np.sum(x),
                np.ones((1, D)),
                np.full((1, D), -np.inf),
                np.full((1, D), np.inf),
                np.ones((1, D)) * -10,
                np.ones((1, D)) * 10)
    vbmc.vp = vp
    parameter_transformer, __, __, __ = warp_input_vbmc(vp, vbmc.optim_state, vbmc.function_logger, vbmc.options)
    U = parameter_transformer.R_mat

    # U should undo rotation of R, up to a sign:
    assert np.all(np.isclose(np.abs(U@R), np.eye(D), atol=0.05))

def test_unscent_warp():
    angle = np.pi/6.6
    R1 = np.array([[np.cos(angle), np.sin(angle)],
                  [-np.sin(angle), np.cos(angle)]]
                  )
    angle = np.pi/3
    R2 = np.array([[np.cos(angle), np.sin(angle)],
                  [-np.sin(angle), np.cos(angle)]]
                  )
    parameter_transformer_1 = ParameterTransformer(D,
                                                   rotation_matrix = R1,
                                                   scale = np.array([0.9, 0.7]))
    parameter_transformer_2 = ParameterTransformer(D,
                                                   rotation_matrix = R2,
                                                   scale = np.array([1.2,0.5]))
    def warpfun(x):
        # Copy probably unneccesary:
        x = np.copy(x)
        return parameter_transformer_2(
                                parameter_transformer_1.inverse(x), no_infs=True
                                      )
    sigma = np.array([3.0, 0.7])
    mu = np.array([[1.2, -3.3],
                   [-1.7, 0.5],
                   [-3.4, -np.pi],
                   [5.01, 9.8]
                   ])
    [muw, sigmaw, muu] = unscent_warp(warpfun, mu, sigma)
    matlab_result_muw = np.array([
        [1.79786175315009,	-2.71880715597597],
        [-1.23028515945097,	-1.06548342843230],
        [-1.15442046351576,	-7.00874808879672],
        [0.0703468098253310,	16.4174973622584]
    ])
    matlab_result_sigmaw = np.array([
        [1.90565079837152,	3.03363336606057],
        [1.90565079837152,	3.03363336606057],
        [1.90565079837152,	3.03363336606057],
        [1.90565079837152,	3.03363336606057]
         ])

    muu_1 = [
            1.7979,   -1.2303,   -1.1544,    0.0703,
            4.4747,    1.4466,    1.5224,    2.7472,
            -0.879,   -3.9071,   -3.8313,   -2.6065,
            1.4857,   -1.5425,   -1.4666,   -0.2419,
            2.1101,   -0.9181,   -0.8422,    0.3826
        ]
    muu_2 = [
            -2.7188,   -1.0655,   -7.0087,   16.4175,
             1.4099,    3.0633,   -2.8800,   20.5462,
            -6.8475,   -5.1942,  -11.1375,   12.2888,
            -1.5529,    0.1004,   -5.8428,   17.5834,
            -3.8847,   -2.2314,   -8.1747,   15.2516
        ]
    matlab_result_muu = np.array(list(zip(muu_1, muu_2))).reshape([5,4,2])

    assert np.all(np.isclose(muw, matlab_result_muw, atol=0.0001))
    assert np.all(np.isclose(muu, matlab_result_muu, atol=0.0001))
    assert np.all(np.isclose(sigmaw, matlab_result_sigmaw, atol=0.0001))

def test_direct_transform_type3_within():
    parameter_transformer = ParameterTransformer(
        D=D,
        lower_bounds=np.ones((1, D)) * -10,
        upper_bounds=np.ones((1, D)) * 10,
    )
    X = np.ones((10, D)) * 3
    Y = parameter_transformer(X)
    Y2 = np.ones((10, D)) * 0.619
    assert np.all(np.isclose(Y, Y2, atol=1e-04))


def test_transform_direct_inverse():
    parameter_transformer = ParameterTransformer(
        D=D,
        lower_bounds=np.ones((1, D)) * -10,
        upper_bounds=np.ones((1, D)) * 10,
    )
    X = np.ones((10, D)) * 0.05
    U = parameter_transformer(X)
    X2 = parameter_transformer.inverse(U)
    assert np.all(np.isclose(X, X2, rtol=1e-12, atol=1e-14))


def test_transform_inverse_direct():
    parameter_transformer = ParameterTransformer(
        D=D,
        lower_bounds=np.ones((1, D)) * -10,
        upper_bounds=np.ones((1, D)) * 10,
    )
    U = np.ones((10, D)) * 0.2
    X = parameter_transformer.inverse(U)
    U2 = parameter_transformer(X)
    assert np.all(np.isclose(U, U2, rtol=1e-12, atol=1e-14))
