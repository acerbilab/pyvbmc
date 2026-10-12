"""Finite-difference checks of ``ParameterTransformer.log_abs_det_jacobian``.

Convention. ``log_abs_det_jacobian(u)`` returns the log absolute determinant
of the Jacobian of ``inverse``, the map from transformed (unconstrained)
coordinates ``u`` back to the original constrained space, evaluated at the
transformed points ``u``: one value per row of ``u``. It is the term that
turns a density in the original space into a density in transformed space, so
its sign is the sign of the expansion of ``inverse``: a transformer whose
``delta`` stretches the original space by a factor of two per coordinate
returns ``+D log 2``, not ``-D log 2``.

The transform is a per-coordinate map (centering by ``mu`` and ``delta``, plus
a bounded transform wherever both bounds are finite) followed by a rotation
``R_mat`` and a rescaling ``scale``. The closed form adds up per-coordinate
log-derivatives together with ``log(scale)``, and takes ``|det R_mat| == 1``
for granted. The checks here therefore build the whole ``D x D`` Jacobian of
``inverse`` by central differences and take its ``slogdet``, which sees the
rotation and any coupling between coordinates that a per-coordinate check
cannot.

The points are drawn well inside the transformed space. A bounded transform
saturates in float64 long before its closed-form log-derivative under-runs,
and ``inverse`` nudges saturated points away from the hard bounds, so out
there the map is flat while the closed form is not: the two agree only where
the transform is a bijection in floating point.
"""

import numpy as np
import pytest

from pyvbmc.parameter_transformer import ParameterTransformer

# Central differences of a map whose values are of order one keep about
# eight digits, and the log-determinant adds up D such entries. The absolute
# tolerance is what compares the cases whose log-determinant is exactly zero,
# where a relative tolerance says nothing.
FD_STEP = 1e-6
RTOL = 1e-6
ATOL = 1e-8

SEED = 20260906

BOUNDED_TYPES = {"logit": 3, "probit": 12, "student4": 13}

# Bounded coordinates use these asymmetric hard bounds.
LB, UB = -1.0, 3.0

CASES = [
    "unbounded",
    "unbounded_centered",
    "bounded_logit",
    "bounded_logit_centered",
    "bounded_probit",
    "bounded_probit_centered",
    "bounded_student4",
    "bounded_student4_centered",
    "mixed_logit",
    "mixed_probit",
    "mixed_student4",
]

# "whitening" installs a rotation and a rescaling and resets the centering,
# which is what a whitening warp does; "warp_over_centering" keeps the
# centering the plausible bounds produced, a combination the transform also
# allows.
WARPS = ["none", "whitening", "warp_over_centering"]

DIMENSIONS = [1, 2, 5]

# Transformed points are kept in a range where no bounded transform is close
# to saturating, so that the finite differences stay well conditioned.
POINT_RANGE = 1.2
N_POINTS = 5


def _seeded_rng(case, D, warp):
    """A generator that depends only on the parametrization."""
    return np.random.default_rng(
        [SEED, CASES.index(case), D, WARPS.index(warp)]
    )


def _build_transformer(case, D, rng):
    """Build the transformer of a named case, or None if D is too small."""
    if case == "unbounded":
        return ParameterTransformer(D=D)

    if case == "unbounded_centered":
        # Finite plausible bounds on an unbounded variable are the only
        # source of mu != 0 and delta != 1 there.
        return ParameterTransformer(
            D=D,
            plb_orig=-rng.uniform(1.0, 4.0, size=(1, D)),
            pub_orig=rng.uniform(1.0, 6.0, size=(1, D)),
        )

    kind, name = case.split("_")[0], case.split("_")[1]
    transform_type = BOUNDED_TYPES[name]

    if kind == "bounded":
        lb = np.full((1, D), LB)
        ub = np.full((1, D), UB)
        if case.endswith("centered"):
            plb = LB + rng.uniform(0.2, 0.8, size=(1, D))
            pub = UB - rng.uniform(0.8, 1.4, size=(1, D))
        else:
            plb, pub = lb, ub
        return ParameterTransformer(
            D=D,
            lb_orig=lb,
            ub_orig=ub,
            plb_orig=plb,
            pub_orig=pub,
            transform_type=transform_type,
        )

    if D < 2:
        return None
    # Even coordinates bounded, odd coordinates unbounded, all of them with
    # plausible bounds inside the hard bounds.
    bounded = np.arange(D) % 2 == 0
    lb = np.where(bounded, LB, -np.inf).reshape((1, D))
    ub = np.where(bounded, UB, np.inf).reshape((1, D))
    plb = np.where(
        bounded,
        LB + rng.uniform(0.2, 0.8, size=D),
        -rng.uniform(1.0, 4.0, size=D),
    ).reshape((1, D))
    pub = np.where(
        bounded,
        UB - rng.uniform(0.8, 1.4, size=D),
        rng.uniform(1.0, 6.0, size=D),
    ).reshape((1, D))
    return ParameterTransformer(
        D=D,
        lb_orig=lb,
        ub_orig=ub,
        plb_orig=plb,
        pub_orig=pub,
        transform_type=transform_type,
    )


def _random_rotation(rng, D):
    """A random rotation matrix: orthogonal with determinant +1."""
    Q, R = np.linalg.qr(rng.standard_normal((D, D)))
    Q = Q * np.sign(np.diag(R))  # Pin down the sign freedom of the QR
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def _apply_warp(parameter_transformer, warp, D, rng):
    """Install a rotoscale warp on a transformer, in place."""
    if warp == "none":
        return
    parameter_transformer.R_mat = _random_rotation(rng, D)
    parameter_transformer.scale = np.exp(rng.uniform(-0.3, 0.3, size=D))
    if warp == "whitening":
        parameter_transformer.mu = np.zeros(D)
        parameter_transformer.delta = np.ones(D)


def _points(rng, D, n=N_POINTS):
    return rng.uniform(-POINT_RANGE, POINT_RANGE, size=(n, D))


def _fd_jacobian(f, u, h=FD_STEP):
    """The full Jacobian of `f` at the point `u` by central differences.

    Entry ``[i, j]`` is the derivative of the i-th output with respect to the
    j-th input. `f` maps an (N, D) array of points to an (N, D) array.
    """
    D = u.size
    points = np.tile(u, (2 * D, 1))
    for j in range(D):
        points[2 * j, j] += h
        points[2 * j + 1, j] -= h
    values = f(points)
    jacobian = np.empty((D, D))
    for j in range(D):
        jacobian[:, j] = (values[2 * j] - values[2 * j + 1]) / (2 * h)
    return jacobian


@pytest.mark.parametrize("warp", WARPS)
@pytest.mark.parametrize("D", DIMENSIONS)
@pytest.mark.parametrize("case", CASES)
def test_log_abs_det_jacobian_fd(case, D, warp):
    rng = _seeded_rng(case, D, warp)
    parameter_transformer = _build_transformer(case, D, rng)
    if parameter_transformer is None:
        pytest.skip("mixed bounded and unbounded coordinates need D > 1")
    _apply_warp(parameter_transformer, warp, D, rng)

    U = _points(rng, D)
    log_j = parameter_transformer.log_abs_det_jacobian(U)
    assert log_j.shape == (U.shape[0],)

    for i, u in enumerate(U):
        jacobian = _fd_jacobian(parameter_transformer.inverse, u)
        sign, log_det = np.linalg.slogdet(jacobian)
        assert sign != 0
        assert np.isclose(log_det, log_j[i], rtol=RTOL, atol=ATOL), (
            f"row {i}: finite differences give {log_det}, "
            f"log_abs_det_jacobian gives {log_j[i]}"
        )
        # One value per row, each depending on that row alone.
        assert np.isclose(
            parameter_transformer.log_abs_det_jacobian(u[np.newaxis, :])[0],
            log_j[i],
            rtol=1e-12,
            atol=1e-14,
        )


@pytest.mark.parametrize("warp", WARPS)
@pytest.mark.parametrize("D", DIMENSIONS)
@pytest.mark.parametrize("case", CASES)
def test_round_trips(case, D, warp):
    rng = _seeded_rng(case, D, warp)
    parameter_transformer = _build_transformer(case, D, rng)
    if parameter_transformer is None:
        pytest.skip("mixed bounded and unbounded coordinates need D > 1")
    _apply_warp(parameter_transformer, warp, D, rng)

    U = _points(rng, D)
    X = parameter_transformer.inverse(U)
    assert np.allclose(parameter_transformer(X), U, rtol=1e-10, atol=1e-10)
    assert np.allclose(
        parameter_transformer.inverse(parameter_transformer(X)),
        X,
        rtol=1e-10,
        atol=1e-10,
    )


def test_sign_is_that_of_the_inverse_map():
    """A transform that halves the space returns a positive log-Jacobian.

    Plausible bounds at -1 and 1 on unbounded variables give delta = 2, so
    ``inverse`` stretches every coordinate by two and the log absolute
    determinant is +D log 2. The forward map contracts by the same factor,
    and its log absolute determinant is the negative of that.
    """
    D = 3
    parameter_transformer = ParameterTransformer(
        D=D,
        plb_orig=np.full((1, D), -1.0),
        pub_orig=np.full((1, D), 1.0),
    )
    assert np.allclose(parameter_transformer.delta, 2.0)

    rng = np.random.default_rng(SEED)
    U = _points(rng, D)
    log_j = parameter_transformer.log_abs_det_jacobian(U)
    assert np.allclose(log_j, D * np.log(2.0))

    for u in U:
        __, log_det_forward = np.linalg.slogdet(
            _fd_jacobian(parameter_transformer, u)
        )
        assert np.isclose(log_det_forward, -D * np.log(2.0), rtol=RTOL)


@pytest.mark.parametrize("case", CASES)
def test_shape_and_1D_input(case):
    D = 3
    rng = _seeded_rng(case, D, "none")
    parameter_transformer = _build_transformer(case, D, rng)
    _apply_warp(parameter_transformer, "whitening", D, rng)

    U = _points(rng, D, n=7)
    log_j = parameter_transformer.log_abs_det_jacobian(U)
    assert log_j.shape == (7,)

    for i, u in enumerate(U):
        # A single row goes through a matrix-vector product where the batch
        # goes through a matrix-matrix one, so the two agree to rounding
        # rather than bit for bit.
        scalar = parameter_transformer.log_abs_det_jacobian(u)
        assert np.ndim(scalar) == 0
        assert np.isclose(scalar, log_j[i], rtol=1e-12, atol=1e-14)
        by_keyword = parameter_transformer.log_abs_det_jacobian(u=u)
        assert np.ndim(by_keyword) == 0
        assert by_keyword == scalar
