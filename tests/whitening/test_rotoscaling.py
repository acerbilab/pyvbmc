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
                                parameter_transformer_1.inverse(x)
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


def test_parameter_transformer_log_det_abs():
    D = 3
    x = np.array([1.0, -3.0, 8.5])
    parameter_transformer = ParameterTransformer(
                                                D=D,
                                                lower_bounds=np.ones((1, D)) * -10,
                                                upper_bounds=np.ones((1, D)) * 10,
    )
    u = parameter_transformer(x)
    # MATLAB result:
    assert np.allclose(u, np.array([0.200670695462151, -0.619039208406224, 2.51230562397612]))
    log_abs_det = parameter_transformer.log_abs_det_jacobian(u)
    assert np.isclose(log_abs_det, 3.44201837618191)

    # Now with rotation and scale:
    angle = np.pi/6.6
    R1 = np.array([
                  [1.0, 0.0, 0.0],
                  [0.0, np.cos(angle), -np.sin(angle)],
                  [0.0, np.sin(angle), np.cos(angle)]
                 ])
    angle = np.pi/3
    R2 = np.array([
                   [np.cos(angle), -np.sin(angle), 0.0],
                   [np.sin(angle), np.cos(angle), 0.0],
                   [0.0, 0.0, 1.0]
                 ])
    R = R1@R2
    parameter_transformer = ParameterTransformer(
                                                D=D,
                                                lower_bounds=np.ones((1, D)) * -10,
                                                upper_bounds=np.ones((1, D)) * 10,
                                                rotation_matrix = R,
                                                scale = np.array([0.9, 0.7, 2.3])
    )
    u = parameter_transformer(x)
    print(u)
    assert np.allclose(u, np.array([0.689778028799817, 0.181006596372446, 1.09421151292434]))
    log_abs_det = parameter_transformer.log_abs_det_jacobian(u)
    print(log_abs_det)
    print(parameter_transformer.delta)
    print(parameter_transformer.mu)
    print(parameter_transformer.scale)
    print(parameter_transformer.type)
    assert np.isclose(log_abs_det, 3.81289203952045)


@pytest.mark.skip(reason="Test not finished.")
def test_warp_input_vbmc():
    angle = 1.309355600770139
    R = np.array([[np.cos(angle), np.sin(angle)],
                 [-np.sin(angle), np.cos(angle)]]
                )
    rands = np.array([
        [0.863077542531699, -0.289939258766329],
        [0.101677144891506, 1.69671085524996],
        [-1.17160077378450, 1.29031232184773],
        [-0.526269140864695, -0.428892744565227],
        [0.0705080370696098, 0.146809594851066],
        [0.135530782339547, -0.828601492654749],
        [0.466400424678227, -0.348101222370484],
        [0.982295558613529, -0.530377396742441],
        [-0.869041937718951, 0.213931172533040],
        [1.69021741525964, 0.342465817734047],
        [-0.999222742785420, 0.907289056500305],
        [-1.01540654957279, 0.0353655540333276],
        [2.27105278214723, -2.19296246215682],
        [-0.424485568767445, -0.448203991593693],
        [-0.387472065949968, 1.56389797220726],
        [1.86824800611932, 0.716488449201314],
        [0.818844814102036, -1.18763931148113],
        [0.659882563140113, -0.689711355484945],
        [0.164481921324160, 0.0973280257720499],
        [0.238865495645372, -0.499735706323105],
        [-0.211254607409097, -2.13406450434705],
        [-0.541986424337808, -0.546479319514447],
        [1.39098281872649, 0.921037471295941],
        [-0.393516065199629, 2.04861663786019],
        [-0.643206933657155, -0.145338009545600],
        [1.58849304511189, -1.39562083957419],
        [0.751025462317757, -0.701659075823514],
        [0.911078511724154, -0.358878957781475],
        [-0.593717984063518, -0.146240078627267],
        [-0.420334072643720, 1.43205266879963],
        [-0.215926641120469, 0.714435897821756],
        [1.05554244043582, 1.21619028188551],
        [0.912810629960210, 0.425178531426115],
        [-0.365417499940999, -0.346584887549622],
        [1.37687121914998, -1.38216937459364],
        [-1.06048073942646, 0.786685955156052],
        [-0.698311279064511, 2.22617972777297],
        [-0.508466707129491, -0.506954526924362],
        [-0.549555921013203, 0.208964515691570],
        [0.708676552007701,  -1.49272448916073],
        [-1.13511844745561,  -0.282654228212148],
        [-2.04179725039980,  -0.769244891917382],
        [0.505696022291427,  -0.571390826423346],
        [0.357174924064903,  -1.29273752721148],
        [0.370761962746280,  -0.579564581236171],
        [-0.499694205126650, -0.369808886585155],
        [0.726746148964249,  0.529845152234478],
        [-0.758195880280548, 0.399606668462689],
        [1.42005120331928,   -0.184391418591077],
        [1.40306400326931,   0.122953370152712]
    ])
    rands[:,0] = 10*rands[:,0]
    mus   = rands@R
    vp = VariationalPosterior(D, 50, mus)
    vbmc = VBMC(lambda x: np.sum(x),
                mus,
                np.full((1, D), -np.inf),
                np.full((1, D), np.inf),
                np.ones((1, D)) * -10,
                np.ones((1, D)) * 10)
    # vbmc.vp = vp
    parameter_transformer_warp, vbmc.optim_state, vbmc.function_logger, warp_action = warp_input_vbmc(vp, vbmc.optim_state, vbmc.function_logger, vbmc.options)

    assert np.all(parameter_transformer_warp.lb_orig == [-np.inf, -np.inf])
    assert np.all(parameter_transformer_warp.ub_orig == [np.inf, np.inf])
    assert np.all(parameter_transformer_warp.type == [0, 0])
    assert np.all(parameter_transformer_warp.mu == [0.0, 0.0])
    assert np.all(parameter_transformer_warp.delta == [1.0, 1.0])
    print(parameter_transformer_warp.R_mat)
    assert np.all(np.isclose(parameter_transformer_warp.R_mat, np.array([
        [0.493952340977679, -0.869488979138132],
        [0.869488979138132, 0.493952340977679]
    ])))
    assert np.all(np.isclose(parameter_transformer_warp.scale, np.array([
        3.93465312348704, 0.0268047056002401
    ])))
