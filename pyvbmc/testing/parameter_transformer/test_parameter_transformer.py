import inspect
import re

import numpy as np
import pytest
import scipy.stats as sps

from pyvbmc import parameter_transformer
from pyvbmc.parameter_transformer import ParameterTransformer

bounded_transform_types = [3, 12, 13]  # Update as needed with new types.
D = 3


def test_init_no_lower_bounds():
    parameter_transformer = ParameterTransformer(D=D)
    assert np.all(np.isinf(parameter_transformer.lb_orig))


def test_init_lower_bounds():
    parameter_transformer = ParameterTransformer(
        D=D, lb_orig=np.ones((1, D)), ub_orig=np.ones((1, D)) * 2
    )
    assert np.all(parameter_transformer.lb_orig == np.ones(D))


def test_init_no_upper_bounds():
    parameter_transformer = ParameterTransformer(D=D)
    assert np.all(np.isinf(parameter_transformer.ub_orig))


def test_init_upper_bounds():
    parameter_transformer = ParameterTransformer(
        D=D, lb_orig=np.zeros((1, D)), ub_orig=np.ones((1, D))
    )
    assert np.all(parameter_transformer.ub_orig == np.ones(D))


def test_init_type_3():
    # logit (default)
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)),
        ub_orig=np.ones((1, D)) * 2,
    )
    assert np.all(parameter_transformer.type == np.ones(D) * 3)

    # logit (keyword)
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)),
        ub_orig=np.ones((1, D)) * 2,
        transform_type="logit",
    )
    assert np.all(parameter_transformer.type == np.ones(D) * 3)

    # logit (number)
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)),
        ub_orig=np.ones((1, D)) * 2,
        transform_type=3,
    )
    assert np.all(parameter_transformer.type == np.ones(D) * 3)

    # probit
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)),
        ub_orig=np.ones((1, D)) * 2,
        transform_type="probit",
    )
    assert np.all(parameter_transformer.type == np.ones(D) * 12)

    # probit (alternate name)
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)),
        ub_orig=np.ones((1, D)) * 2,
        transform_type="norminv",
    )

    # probit (number)
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)),
        ub_orig=np.ones((1, D)) * 2,
        transform_type=12,
    )
    assert np.all(parameter_transformer.type == np.ones(D) * 12)

    # Student's T
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)),
        ub_orig=np.ones((1, D)) * 2,
        transform_type="student4",
    )
    assert np.all(parameter_transformer.type == np.ones(D) * 13)

    # Student's T (number)
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)),
        ub_orig=np.ones((1, D)) * 2,
        transform_type=13,
    )
    assert np.all(parameter_transformer.type == np.ones(D) * 13)

    # Bad transform (keyword)
    with pytest.raises(Exception) as e_info:
        parameter_transformer = ParameterTransformer(
            D=D,
            lb_orig=np.ones((1, D)),
            ub_orig=np.ones((1, D)) * 2,
            transform_type="this_is_not_a_transform_type",
        )
    assert "Unrecognized bounded transform" in e_info.value.args[0]

    # Bad transform (number)
    with pytest.raises(Exception) as e_info:
        parameter_transformer = ParameterTransformer(
            D=D,
            lb_orig=np.ones((1, D)),
            ub_orig=np.ones((1, D)) * 2,
            transform_type=666,
        )
    assert "Unrecognized bounded transform" in e_info.value.args[0]


def test_init_mixed_bounds():
    parameter_transformer = ParameterTransformer(
        D=4,
        lb_orig=np.array([[0.0, -np.inf, -10.0, -np.inf]]),
        ub_orig=np.array([[10.0, np.inf, 0.0, np.inf]]),
    )
    assert np.all(
        parameter_transformer.type == np.array([3, 0, 3, 0], dtype=int)
    )

    parameter_transformer = ParameterTransformer(
        D=4,
        lb_orig=np.array([[0.0, -np.inf, -10.0, -np.inf]]),
        ub_orig=np.array([[10.0, np.inf, 0.0, np.inf]]),
        transform_type="probit",
    )
    assert np.all(
        parameter_transformer.type == np.array([12, 0, 12, 0], dtype=int)
    )

    parameter_transformer = ParameterTransformer(
        D=4,
        lb_orig=np.array([[0.0, -np.inf, -10.0, -np.inf]]),
        ub_orig=np.array([[10.0, np.inf, 0.0, np.inf]]),
        transform_type="student4",
    )
    assert np.all(
        parameter_transformer.type == np.array([13, 0, 13, 0], dtype=int)
    )

    for t in bounded_transform_types:
        parameter_transformer = ParameterTransformer(
            D=4,
            lb_orig=np.array([[0.0, -np.inf, -10.0, -np.inf]]),
            ub_orig=np.array([[10.0, np.inf, 0.0, np.inf]]),
            transform_type=t,
        )
        assert np.all(
            parameter_transformer.type == np.array([t, 0, t, 0], dtype=int)
        )


def test_init_bounds_check():
    with pytest.raises(ValueError):
        ParameterTransformer(
            D=D,
            lb_orig=np.ones((1, D)) * 3,
            ub_orig=np.ones((1, D)) * 2,
        )
    with pytest.raises(ValueError):
        ParameterTransformer(
            D=D,
            lb_orig=np.ones((1, D)) * 0,
            ub_orig=np.ones((1, D)) * 10,
            plb_orig=np.ones((1, D)) * -1,
        )
    with pytest.raises(ValueError):
        ParameterTransformer(
            D=D,
            lb_orig=np.ones((1, D)) * 0,
            ub_orig=np.ones((1, D)) * 10,
            pub_orig=np.ones((1, D)) * 11,
        )
    with pytest.raises(ValueError):
        ParameterTransformer(
            D=D,
            lb_orig=np.ones((1, D)) * 0,
            ub_orig=np.ones((1, D)) * 10,
            plb_orig=np.ones((1, D)) * 100,
            pub_orig=np.ones((1, D)) * -20,
        )


def test_init_rejects_half_bounded_variables():
    """A variable with one finite and one infinite bound would need a log
    transform, which this class does not provide, so it is refused rather
    than carried through the identity."""
    for lb_orig, ub_orig in (
        (np.array([[0.0]]), np.array([[np.inf]])),
        (np.array([[-np.inf]]), np.array([[3.0]])),
    ):
        with pytest.raises(ValueError) as e_info:
            ParameterTransformer(D=1, lb_orig=lb_orig, ub_orig=ub_orig)
        assert "one side only" in e_info.value.args[0]

    # The offending dimensions are named.
    with pytest.raises(ValueError) as e_info:
        ParameterTransformer(
            D=3,
            lb_orig=np.array([[-1.0, -np.inf, 0.0]]),
            ub_orig=np.array([[1.0, 2.0, np.inf]]),
        )
    assert "[1, 2]" in e_info.value.args[0]


def test_init_rotation_matrix_validation():
    reflected = np.diag([-1.0, 1.0, 1.0])
    transformer = ParameterTransformer(D=D, rotation_matrix=reflected)
    assert np.array_equal(transformer.R_mat, reflected)

    rng = np.random.default_rng(1234)
    near_orthogonal, __, __ = np.linalg.svd(rng.standard_normal((D, D)))
    transformer = ParameterTransformer(D=D, rotation_matrix=near_orthogonal)
    assert np.array_equal(transformer.R_mat, near_orthogonal)

    invalid_rotations = [
        np.eye(D - 1),
        np.full((D, D), np.nan),
        np.eye(D, dtype=complex),
        np.diag([1.0, 1.0, 0.0]),
        np.diag([1.0, 1.0, 2.0]),
    ]
    for rotation_matrix in invalid_rotations:
        with pytest.raises(ValueError):
            ParameterTransformer(D=D, rotation_matrix=rotation_matrix)


def test_class_docstring_documents_the_constructor_arguments():
    """The class docstring is the published documentation of the
    constructor, so the names it documents are the ones a caller passes."""
    documented = set(
        re.findall(r"^    (\w+) : ", ParameterTransformer.__doc__, re.M)
    )
    signature = inspect.signature(ParameterTransformer.__init__).parameters
    assert documented == set(signature) - {"self"}


def test_init_copies_the_arrays_it_is_given():
    """The transform is fixed at construction: changing an array the
    caller passed does not move it afterwards."""
    lb_orig = np.full((1, D), -2.0)
    ub_orig = np.full((1, D), 2.0)
    scale = np.full(D, 2.0)
    rotation_matrix = np.eye(D)
    transformer = ParameterTransformer(
        D,
        lb_orig=lb_orig,
        ub_orig=ub_orig,
        scale=scale,
        rotation_matrix=rotation_matrix,
    )
    x = np.full((1, D), 0.5)
    transformed = transformer(x)

    lb_orig[0, 0] = -100.0
    ub_orig[0, 0] = 100.0
    scale[0] = 10.0
    rotation_matrix[0, 0] = -1.0

    assert np.array_equal(transformer(x), transformed)
    assert transformer.lb_orig[0, 0] == -2.0
    assert transformer.ub_orig[0, 0] == 2.0
    assert transformer.scale[0] == 2.0
    assert transformer.R_mat[0, 0] == 1.0


def test_init_scale_validation():
    """The transform divides by ``scale`` and its log-Jacobian adds
    ``log(scale)``, so a scale must have one finite positive entry per
    dimension."""
    scale = np.array([0.5, 1.0, 2.0])
    transformer = ParameterTransformer(D=D, scale=scale)
    assert np.array_equal(transformer.scale, scale)

    invalid_scales = [
        np.ones(D - 1),
        np.ones((1, D)),
        np.array([1.0, 1.0, np.inf]),
        np.array([1.0, 1.0, np.nan]),
        np.array([1.0, 1.0, 0.0]),
        np.array([1.0, 1.0, -2.0]),
        np.ones(D, dtype=complex),
    ]
    for invalid in invalid_scales:
        with pytest.raises(ValueError) as e_info:
            ParameterTransformer(D=D, scale=invalid)
        assert "`scale`" in e_info.value.args[0]


def test_equality_handles_optional_arrays_and_shapes():
    first = ParameterTransformer(D=D)
    second = ParameterTransformer(D=D)
    assert first == second
    assert not (first == object())

    first.scale = np.ones(D)
    assert first != second
    second.scale = np.ones((1, D))
    assert first != second
    second.scale = np.ones(D)
    assert first == second
    second.scale[-1] = 2.0
    assert first != second
    second.scale = np.ones(D)

    first.R_mat = np.eye(D)
    assert first != second
    second.R_mat = np.eye(D)
    assert first == second
    second.R_mat[[0, 1]] = second.R_mat[[1, 0]]
    assert first != second


def test_init_mu_inf_bounds():
    parameter_transformer = ParameterTransformer(D=D)
    assert np.all(parameter_transformer.mu == np.zeros(D))


def test_init_delta_inf_bounds():
    parameter_transformer = ParameterTransformer(D=D)
    assert np.all(parameter_transformer.delta == np.ones(D))


def test_init_type3_mu_all_params():
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
        plb_orig=np.ones((1, D)) * 2,
        pub_orig=np.ones((1, D)) * 4,
    )
    parameter_transformer2 = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
    )
    plb = parameter_transformer2(np.ones((1, D)) * 2)
    pub = parameter_transformer2(np.ones((1, D)) * 4)
    mu2 = (plb + pub) * 0.5
    assert np.all(
        np.isclose(parameter_transformer.mu, mu2, rtol=1e-12, atol=1e-14)
    )


def test_init_type3_delta_all_params():
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
        plb_orig=np.ones((1, D)) * 2,
        pub_orig=np.ones((1, D)) * 4,
    )
    parameter_transformer2 = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
    )
    plb = parameter_transformer2(np.ones((1, D)) * 2)
    pub = parameter_transformer2(np.ones((1, D)) * 4)
    delta2 = pub - plb
    assert np.all(
        np.isclose(parameter_transformer.delta, delta2, rtol=1e-12, atol=1e-14)
    )


def test_centering_takes_the_plausible_box_to_the_unit_interval():
    """The centering maps the plausible box to [-0.5, 0.5] in every
    coordinate."""
    lb_orig = np.array([[-5.0, -3.0, -1.0]])
    ub_orig = np.array([[5.0, 7.0, 4.0]])
    plb_orig = np.array([[-1.0, -2.0, 0.0]])
    pub_orig = np.array([[1.0, 5.0, 2.0]])
    transformer = ParameterTransformer(D, lb_orig, ub_orig, plb_orig, pub_orig)

    assert np.allclose(transformer(plb_orig), -0.5)
    assert np.allclose(transformer(pub_orig), 0.5)


def test_centering_is_derived_before_the_rotation_and_the_rescaling():
    """A rotation and a rescaling act on the centered coordinates, so the
    transform built with them is the rotated and rescaled image of the one
    built without them."""
    lb_orig = np.array([[-5.0, -3.0, -1.0]])
    ub_orig = np.array([[5.0, 7.0, 4.0]])
    plb_orig = np.array([[-1.0, -2.0, 0.0]])
    pub_orig = np.array([[1.0, 5.0, 2.0]])
    angle = np.pi / 5
    rotation = np.array(
        [
            [np.cos(angle), np.sin(angle), 0.0],
            [-np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    scale = np.array([2.0, 0.5, 3.0])

    centered = ParameterTransformer(D, lb_orig, ub_orig, plb_orig, pub_orig)
    warped = ParameterTransformer(
        D,
        lb_orig,
        ub_orig,
        plb_orig,
        pub_orig,
        scale=scale,
        rotation_matrix=rotation,
    )

    x = np.array([[0.3, 1.0, 1.5], [-2.0, 4.0, 3.0], [4.5, -2.5, -0.5]])
    assert np.allclose(warped(x), (centered(x) @ rotation) / scale)


def test_direct_transform_type3_within():
    # logit
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
    )
    X = np.ones((10, D)) * 3
    Y = parameter_transformer(X)
    Y2 = np.ones((10, D)) * 0.619
    assert np.all(np.isclose(Y, Y2, atol=1e-04))

    # probit
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
        transform_type="probit",
    )
    X = np.ones((10, D)) * 3
    Y = parameter_transformer(X)
    Y2 = np.ones((10, D)) * sps.norm.ppf(13 / 20)
    assert np.all(np.isclose(Y, Y2))

    # student4
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
        transform_type="student4",
    )
    X = np.ones((10, D)) * 3
    Y = parameter_transformer(X)
    Y2 = np.ones((10, D)) * sps.t.ppf(13 / 20, df=4)
    assert np.all(np.isclose(Y, Y2))


def test_direct_transform_type3_within_negative():
    # logit
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
    )
    X = np.ones((10, D)) * -4
    Y = parameter_transformer(X)
    Y2 = np.ones((10, D)) * -0.8473
    assert np.all(np.isclose(Y, Y2))

    # probit
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
        transform_type="probit",
    )
    X = np.ones((10, D)) * -4
    Y = parameter_transformer(X)
    Y2 = np.ones((10, D)) * sps.norm.ppf(6 / 20)
    assert np.all(np.isclose(Y, Y2))

    # student4
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
        transform_type="student4",
    )
    X = np.ones((10, D)) * -4
    Y = parameter_transformer(X)
    Y2 = np.ones((10, D)) * sps.t.ppf(6 / 20, df=4)
    assert np.all(np.isclose(Y, Y2))


def test_direct_transform_type0():
    parameter_transformer = ParameterTransformer(D=D)
    X = np.ones((10, D)) * 3
    Y = parameter_transformer(X)
    assert np.all(Y == X)


def test_direct_transform_type0_negative():
    parameter_transformer = ParameterTransformer(D=D)
    X = np.ones((10, D)) * -4
    Y = parameter_transformer(X)
    assert np.all(Y == X)


def test_inverse_type3_within():
    # logit
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
    )
    Y = np.ones((10, D)) * 3
    X = parameter_transformer.inverse(Y)
    X2 = np.ones((10, D)) * 9.0515
    assert np.all(np.isclose(X, X2))

    # probit
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
        transform_type="probit",
    )
    Y = np.ones((10, D)) * 3
    X = parameter_transformer.inverse(Y)
    X2 = np.ones((10, D)) * sps.norm.cdf(3) * 20 - 10
    assert np.all(np.isclose(X, X2))

    # student4
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
        transform_type="student4",
    )
    Y = np.ones((10, D)) * 3
    X = parameter_transformer.inverse(Y)
    X2 = np.ones((10, D)) * sps.t.cdf(3, df=4) * 20 - 10
    assert np.all(np.isclose(X, X2))


def test_inverse_type3_within_negative():
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
    )
    Y = np.ones((10, D)) * -4
    X = parameter_transformer.inverse(Y)
    X2 = np.ones((10, D)) * -9.6403
    assert np.all(np.isclose(X, X2))

    # probit
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
        transform_type="probit",
    )
    Y = np.ones((10, D)) * -4
    X = parameter_transformer.inverse(Y)
    X2 = np.ones((10, D)) * sps.norm.cdf(-4) * 20 - 10
    assert np.all(np.isclose(X, X2))

    # student4
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
        transform_type="student4",
    )
    Y = np.ones((10, D)) * -4
    X = parameter_transformer.inverse(Y)
    X2 = np.ones((10, D)) * sps.t.cdf(-4, df=4) * 20 - 10
    assert np.all(np.isclose(X, X2))


def test_inverse_type0():
    parameter_transformer = ParameterTransformer(D=D)
    Y = np.ones((10, D)) * 3
    X = parameter_transformer.inverse(Y)
    assert np.all(Y == X)


def test_inverse_type0_negative():
    parameter_transformer = ParameterTransformer(D=D)
    Y = np.ones((10, D)) * -4
    X = parameter_transformer.inverse(Y)
    assert np.all(Y == X)


def test_inverse_type3_min_space():
    # logit
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
    )
    Y = np.ones((10, D)) * -500
    X = parameter_transformer.inverse(Y)
    assert np.allclose(X, np.ones((1, D)) * -10)

    for t in [12, 13]:  # probit, student4
        parameter_transformer = ParameterTransformer(
            D=D,
            lb_orig=np.ones((1, D)) * -10,
            ub_orig=np.ones((1, D)) * 10,
            transform_type=t,
        )
        Y = np.ones((10, D)) * -500
        X = parameter_transformer.inverse(Y)
        assert np.allclose(X, np.ones((1, D)) * -10)


def test_inverse_type3_max_space():
    # logit
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
    )
    Y = np.ones((10, D)) * 3000
    X = parameter_transformer.inverse(Y)
    assert np.allclose(X, np.ones((10, D)) * 10)

    # The inverse keeps its result strictly inside the bounds: a point whose
    # image rounds onto the upper bound comes back as the largest number
    # below it. The tail of the Student's t is polynomial, so its points
    # round onto the bound much farther out than those of the probit.
    for t in [12, 13]:  # probit, student4
        parameter_transformer = ParameterTransformer(
            D=D,
            lb_orig=np.ones((1, D)) * -10,
            ub_orig=np.ones((1, D)) * 10,
            transform_type=t,
        )
        Y = np.ones((10, D)) * 1e6
        X = parameter_transformer.inverse(Y)
        assert np.all(X == np.nextafter(10, -np.inf))


def test_transform_direct_inverse():
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
    )
    X = np.ones((10, D)) * 0.05
    U = parameter_transformer(X)
    X2 = parameter_transformer.inverse(U)
    assert np.all(np.isclose(X, X2, rtol=1e-12, atol=1e-14))

    for t in [12, 13]:  # probit, student4
        parameter_transformer = ParameterTransformer(
            D=D,
            lb_orig=np.ones((1, D)) * -10,
            ub_orig=np.ones((1, D)) * 10,
            transform_type=t,
        )
        X = np.ones((10, D)) * 0.05
        U = parameter_transformer(X)
        X2 = parameter_transformer.inverse(U)
        assert np.all(np.isclose(X, X2, rtol=1e-11, atol=1e-14))


def test_transform_inverse_direct():
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
    )
    U = np.ones((10, D)) * 0.2
    X = parameter_transformer.inverse(U)
    U2 = parameter_transformer(X)
    assert np.all(np.isclose(U, U2, rtol=1e-12, atol=1e-14))

    for t in [12, 13]:  # probit, student4
        parameter_transformer = ParameterTransformer(
            D=D,
            lb_orig=np.ones((1, D)) * -10,
            ub_orig=np.ones((1, D)) * 10,
            transform_type=t,
        )
        U = np.ones((10, D)) * 0.2
        X = parameter_transformer.inverse(U)
        U2 = parameter_transformer(X)
        assert np.all(np.isclose(U, U2, rtol=1e-12, atol=1e-14))


def test_transform_direct_inverse_largeN():
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
    )
    X = np.ones((10 ^ 6, D)) * 0.4
    U = parameter_transformer(X)
    X2 = parameter_transformer.inverse(U)
    assert np.all(np.isclose(X, X2, rtol=1e-12, atol=1e-14))

    for t in [12, 13]:  # probit, student4
        parameter_transformer = ParameterTransformer(
            D=D,
            lb_orig=np.ones((1, D)) * -10,
            ub_orig=np.ones((1, D)) * 10,
            transform_type=t,
        )
        X = np.ones((10 ^ 6, D)) * 0.4
        U = parameter_transformer(X)
        X2 = parameter_transformer.inverse(U)
        assert np.all(np.isclose(X, X2, rtol=1e-11, atol=1e-14))


def test_transform_inverse_direct_largeN():
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
    )
    U = np.ones((10 ^ 6, D)) * 0.11
    X = parameter_transformer.inverse(U)
    U2 = parameter_transformer(X)
    assert np.all(np.isclose(U, U2, rtol=1e-12, atol=1e-14))

    for t in [12, 13]:  # probit, student4
        parameter_transformer = ParameterTransformer(
            D=D,
            lb_orig=np.ones((1, D)) * -10,
            ub_orig=np.ones((1, D)) * 10,
            transform_type=t,
        )
        U = np.ones((10 ^ 6, D)) * 0.11
        X = parameter_transformer.inverse(U)
        U2 = parameter_transformer(X)
        assert np.all(np.isclose(U, U2, rtol=1e-12, atol=1e-14))


def test_log_abs_det_jacobian_type3_within():
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
    )
    U = np.ones((10, D)) * 3
    log_j = parameter_transformer.log_abs_det_jacobian(U)
    log_j2 = np.ones((10)) * -0.3043
    assert np.all(np.isclose(log_j, log_j2, atol=1e-04))


def test_log_abs_det_jacobian_type3_within_negative():
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.ones((1, D)) * -10,
        ub_orig=np.ones((1, D)) * 10,
    )
    U = np.ones((10, D)) * -4
    log_j = parameter_transformer.log_abs_det_jacobian(U)
    log_j2 = np.ones((10)) * -3.1217
    assert np.all(np.isclose(log_j, log_j2))


def test_log_abs_det_jacobian_logit_extreme_tails():
    parameter_transformer = ParameterTransformer(
        D=1, lb_orig=np.zeros((1, 1)), ub_orig=np.ones((1, 1))
    )
    ordinary = np.array([[-100.0], [-1.0], [0.0], [1.0], [100.0]])
    legacy = -ordinary[:, 0] + 2 * (-np.log1p(np.exp(-ordinary[:, 0])))
    assert np.array_equal(
        parameter_transformer.log_abs_det_jacobian(ordinary), legacy
    )

    tails = np.array([[-1000.0], [-710.0], [710.0], [1000.0]])
    log_j = parameter_transformer.log_abs_det_jacobian(tails)
    expected = -np.abs(tails[:, 0]) - 2 * np.log1p(
        np.exp(-np.abs(tails[:, 0]))
    )
    assert np.all(np.isfinite(log_j))
    assert np.array_equal(log_j, expected)


def test_log_abs_det_jacobian_type0():
    parameter_transformer = ParameterTransformer(D=D)
    U = np.ones((10, D)) * 5
    log_j = parameter_transformer.log_abs_det_jacobian(U)
    log_j2 = np.ones((10)) * 0
    assert np.all(np.isclose(log_j, log_j2, atol=1e-04))


def test_log_abs_det_jacobian_type0_negative():
    parameter_transformer = ParameterTransformer(D=D)
    U = np.ones((10, D)) * -6
    log_j = parameter_transformer.log_abs_det_jacobian(U)
    log_j2 = np.ones((10)) * 0
    assert np.all(np.isclose(log_j, log_j2))


def test_1D_input_call():
    parameter_transformer = ParameterTransformer(D=D)
    X = np.ones(D)
    Y = parameter_transformer(X)
    assert X.shape == Y.shape
    Y2 = parameter_transformer(x=X)
    assert X.shape == Y2.shape

    for t in [12, 13]:  # probit, student4
        parameter_transformer = ParameterTransformer(D=D, transform_type=t)
        X = np.ones(D)
        Y = parameter_transformer(X)
        assert X.shape == Y.shape
        Y2 = parameter_transformer(x=X)
        assert X.shape == Y2.shape


def test_1D_input_inverse():
    parameter_transformer = ParameterTransformer(D=D)
    Y = np.ones((D))
    X = parameter_transformer.inverse(Y)
    assert X.shape == Y.shape
    X2 = parameter_transformer.inverse(u=Y)
    assert X2.shape == Y.shape

    for t in [12, 13]:  # probit, student4
        parameter_transformer = ParameterTransformer(D=D, transform_type=t)
        Y = np.ones((D))
        X = parameter_transformer.inverse(Y)
        assert X.shape == Y.shape
        X2 = parameter_transformer.inverse(u=Y)
        assert X2.shape == Y.shape


def test_1D_input_log_abs_det_jacobian():
    parameter_transformer = ParameterTransformer(D=D)
    U = np.ones((D))
    log_j = parameter_transformer.log_abs_det_jacobian(U)
    assert np.ndim(log_j) == 0
    log_j2 = parameter_transformer.log_abs_det_jacobian(u=U)
    assert np.ndim(log_j2) == 0

    for t in [12, 13]:  # probit, student4
        parameter_transformer = ParameterTransformer(D=D, transform_type=t)
        U = np.ones((D))
        log_j = parameter_transformer.log_abs_det_jacobian(U)
        assert np.ndim(log_j) == 0
        log_j2 = parameter_transformer.log_abs_det_jacobian(u=U)
        assert np.ndim(log_j2) == 0


def test_bounded_log_abs_det_jacobian_numerically():
    D = np.random.randint(1, 13)
    LB = -2 * np.ones((1, D))
    UB = 2 * np.ones((1, D))
    for t in bounded_transform_types:  # logit, probit, student4
        parameter_transformer = ParameterTransformer(
            D=D, lb_orig=LB, ub_orig=UB, transform_type=t
        )

        dx = 1e-6
        # Pick a random point where the Jacobian is not 1 but not too
        # close to boundaries
        x1 = np.random.choice([-1, 1]) * np.random.uniform(
            0.5 * np.ones((1, D)), 1.5 * np.ones((1, D))
        )
        x2 = x1 + dx * np.ones((1, D))
        x1_t = parameter_transformer(x1)
        x2_t = parameter_transformer(x2)

        vol1 = np.prod(x2 - x1)
        vol2 = np.prod(x2_t - x1_t)

        assert np.isclose(
            vol1 / vol2,
            np.exp(parameter_transformer.log_abs_det_jacobian(x1_t)),
        )


def test_transform_bounded_and_unbounded():
    D = 2
    n = 10
    x1 = np.random.uniform(0, 10, size=(n, 1))
    x2 = np.random.normal(size=(n, 1))
    x = np.hstack([x1, x2])

    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.array([[0.0, -np.inf]]),
        ub_orig=np.array([[10.0, np.inf]]),
    )
    assert np.allclose(
        x, parameter_transformer.inverse(parameter_transformer(x))
    )
    assert np.allclose(
        x,
        parameter_transformer(parameter_transformer.inverse(x)),
    )

    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.array([[0.0, -np.inf]]),
        ub_orig=np.array([[10.0, np.inf]]),
        plb_orig=np.array([[4.5, -np.inf]]),
        pub_orig=np.array([[5.5, np.inf]]),
        transform_type="probit",
    )
    assert np.allclose(
        x, parameter_transformer.inverse(parameter_transformer(x))
    )
    assert np.allclose(
        x,
        parameter_transformer(parameter_transformer.inverse(x)),
    )

    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.array([[0.0, -np.inf]]),
        ub_orig=np.array([[10.0, np.inf]]),
        transform_type="student4",
    )
    assert np.allclose(
        x, parameter_transformer.inverse(parameter_transformer(x))
    )
    assert np.allclose(
        x,
        parameter_transformer(parameter_transformer.inverse(x)),
    )


def test_abs_det_jacobian_bounded_and_unbounded():
    D = 2
    n = 10
    x1 = np.random.uniform(0, 10, size=(n, 1))
    x2 = np.random.normal(size=(n, 1))
    x = np.hstack([x1, x2])

    bounded_transformer = ParameterTransformer(
        D=1,
        lb_orig=np.array([[0.0]]),
        ub_orig=np.array([[10.0]]),
    )
    unbounded_transformer = ParameterTransformer(
        D=1,
        lb_orig=np.array([[-np.inf]]),
        ub_orig=np.array([[np.inf]]),
        plb_orig=np.array([[-0.5]]),
        pub_orig=np.array([[0.5]]),
    )
    parameter_transformer = ParameterTransformer(
        D=D,
        lb_orig=np.array([[0.0, -np.inf]]),
        ub_orig=np.array([[10.0, np.inf]]),
        plb_orig=np.array([[0.0, -0.5]]),
        pub_orig=np.array([[10.0, 0.5]]),
    )

    j1 = bounded_transformer.log_abs_det_jacobian(x1)
    j2 = unbounded_transformer.log_abs_det_jacobian(x2)
    j = parameter_transformer.log_abs_det_jacobian(x)

    assert np.allclose(j2, 0.0)
    assert np.allclose(j, j1)


def test_boundary_edge_cases():
    D = 4
    for t in bounded_transform_types:
        lb = np.full((1, D), 1000.0)
        ub = np.full((1, D), 1001.0)
        parameter_transformer = ParameterTransformer(
            D=D, lb_orig=lb, ub_orig=ub, transform_type=t
        )
        close_to_lb = np.nextafter(lb, np.inf)
        close_to_ub = np.nextafter(ub, -np.inf)
        close_to_lb_transformed = parameter_transformer(close_to_lb)
        close_to_ub_transformed = parameter_transformer(close_to_ub)
        assert np.all(np.isfinite(close_to_lb_transformed))
        assert np.all(np.isfinite(close_to_ub_transformed))
        assert np.all(
            parameter_transformer.inverse(close_to_lb_transformed)
            == close_to_lb
        )
        assert np.all(
            parameter_transformer.inverse(close_to_ub_transformed)
            == close_to_ub
        )
        big_num = np.sqrt(np.finfo(np.float64).max)
        assert np.all(
            parameter_transformer.inverse(np.full((1, D), -big_num))
            == close_to_lb
        )
        assert np.all(
            parameter_transformer.inverse(np.full((1, D), big_num))
            == close_to_ub
        )

        lb = np.full((1, D), -1000.0)
        ub = np.full((1, D), 0.0)
        parameter_transformer = ParameterTransformer(
            D=D, lb_orig=lb, ub_orig=ub, transform_type=t
        )
        close_to_lb = np.nextafter(lb, np.inf)
        close_to_ub = np.nextafter(ub, -np.inf)
        close_to_lb_transformed = parameter_transformer(close_to_lb)
        close_to_ub_transformed = parameter_transformer(close_to_ub)
        assert np.all(np.isfinite(close_to_lb_transformed))
        assert np.all(np.isfinite(close_to_ub_transformed))
        assert np.allclose(
            parameter_transformer.inverse(close_to_lb_transformed), close_to_lb
        )
        assert np.allclose(
            parameter_transformer.inverse(close_to_ub_transformed), close_to_ub
        )
        big_num = np.sqrt(np.finfo(np.float64).max)
        assert np.all(
            parameter_transformer.inverse(np.full((1, D), -big_num))
            == close_to_lb
        )
        assert np.all(
            parameter_transformer.inverse(np.full((1, D), big_num))
            == close_to_ub
        )


def test_lb_ub_map_to_inf():
    D = 4
    lb_orig = -np.random.normal(scale=100, size=(1, D))
    lb_orig[0, 0], lb_orig[0, 2] = -np.inf, -np.inf  # Mixed bound types
    ub_orig = lb_orig + np.random.lognormal(sigma=2, size=(1, D))
    ub_orig[0, 0], ub_orig[0, 2] = np.inf, np.inf  # Mixed bound types

    for t in bounded_transform_types:
        parameter_transformer = ParameterTransformer(
            D=D,
            lb_orig=lb_orig,
            ub_orig=ub_orig,
            transform_type=t,
        )
        # Hard bounds should map to +- infinity for all variables,
        # both bounded + unbounded, and all types of bounded transforms:
        assert np.all(parameter_transformer(lb_orig) == -np.inf)
        assert np.all(parameter_transformer(ub_orig) == np.inf)


def test__str__and__repr__():
    D = 4
    bounded_names = {
        0: "unbounded",
        3: "logit",
        12: "probit",
        13: "student4",
    }
    for t in bounded_transform_types:
        lb = np.array([[-1.0, -np.inf, -1.0, -np.inf]])
        ub = np.array([[1.0, np.inf, 1.0, np.inf]])
        parameter_transformer = ParameterTransformer(
            D=D, lb_orig=lb, ub_orig=ub, transform_type=t
        )
        string = parameter_transformer.__str__()
        assert bounded_names[t] in string
        parameter_transformer.__repr__()
