"""Optional torch export: density, derivatives, snapshot and support contracts."""

import copy
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ndtri
from scipy.stats import t

from pyvbmc import VariationalPosterior
from pyvbmc.parameter_transformer import ParameterTransformer
from pyvbmc.testing._dtype import assert_float64

# Missing-dependency behavior is covered in the non-optional export tests.
torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def preserve_torch_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(591)
        yield


def _vp(D=3, K=3, kind="probit", warp=True, bounded=True):
    lower = np.full((1, D), -2.0)
    upper = np.full((1, D), 3.0)
    if D > 1 or not bounded:
        lower[0, -1], upper[0, -1] = -np.inf, np.inf
    pt = ParameterTransformer(D, lower, upper, transform_type=kind)
    pt.mu = np.linspace(-0.2, 0.3, D)
    pt.delta = np.linspace(0.7, 1.2, D)
    if warp:
        pt.scale = np.linspace(0.8, 1.4, D)
        pt.R_mat = np.linalg.qr(
            np.random.default_rng(784).normal(size=(D, D))
        )[0]
    vp = VariationalPosterior(D, K, parameter_transformer=pt, rng=851)
    vp.mu = np.linspace(-0.5, 0.7, D * K).reshape(D, K)
    vp.sigma = np.linspace(0.65, 1.1, K).reshape(1, K)
    vp.lambd = np.linspace(0.8, 1.2, D).reshape(D, 1)
    vp.w = np.arange(1.0, K + 1).reshape(1, K)
    vp.w /= vp.w.sum()
    return vp


@pytest.mark.parametrize("kind", ["logit", "probit", "student4"])
@pytest.mark.parametrize("D,K", [(1, 1), (3, 3)])
@pytest.mark.parametrize("orig_flag", [False, True])
def test_density_and_gradient(kind, D, K, orig_flag):
    vp = _vp(D, K, kind)
    u = np.linspace(-0.6, 0.8, 4 * D).reshape(4, D)
    x = vp.parameter_transformer.inverse(u) if orig_flag else u
    dist = vp.to_torch(orig_flag=orig_flag)
    tensor = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    actual = dist.log_prob(tensor)
    expected = vp.log_pdf(x, orig_flag=orig_flag).ravel()
    np.testing.assert_allclose(actual.detach().numpy(), expected, atol=2e-11)
    gradient = torch.autograd.grad(actual.sum(), tensor)[0].numpy()
    if orig_flag:
        expected_gradient = np.empty_like(x)
        for d in range(D):
            step = np.zeros_like(x)
            step[:, d] = 2e-5
            expected_gradient[:, d] = (
                vp.log_pdf(x + step).ravel() - vp.log_pdf(x - step).ravel()
            ) / 4e-5
    else:
        _, expected_gradient = vp.log_pdf(x, orig_flag=False, grad_flag=True)
    np.testing.assert_allclose(
        gradient, expected_gradient, rtol=2e-5, atol=2e-6
    )
    assert torch.autograd.gradcheck(dist.log_prob, (tensor,))


@pytest.mark.parametrize("kind", ["logit", "probit", "student4"])
def test_centers_and_tails(kind):
    vp = _vp(1, 1, kind, warp=False)
    pt = vp.parameter_transformer
    pt.lb_orig[:] = 0
    pt.ub_orig[:] = 1
    pt.mu[:] = 0
    pt.delta[:] = 1
    # Include the exact center, both sides of the student4 branch switch,
    # and tails where erfinv(2*p-1) would lose lower-tail information.
    p = np.array([1e-20, 0.099, 0.101, 0.49999, 0.5, 0.50001, 0.9, 1 - 1e-12])
    x = torch.tensor(p[:, None], dtype=torch.float64, requires_grad=True)
    dist = vp.to_torch()
    z = dist.transforms[0].inv(x)
    if kind == "probit":
        expected = ndtri(p)
        jac = np.exp(-0.5 * expected**2) / np.sqrt(2 * np.pi)
    elif kind == "student4":
        expected = t.ppf(p, 4)
        jac = t.pdf(expected, 4)
    else:
        expected = np.log(p) - np.log1p(-p)
        jac = p * (1 - p)
    np.testing.assert_allclose(
        z.detach().numpy().ravel(), expected, rtol=2e-10, atol=2e-11
    )
    derivative = torch.autograd.grad(z.sum(), x, retain_graph=True)[0]
    np.testing.assert_allclose(derivative.numpy().ravel(), 1 / jac, rtol=2e-9)
    gradient = torch.autograd.grad(dist.log_prob(x).sum(), x)[0]
    assert torch.isfinite(gradient).all()
    # The forward bijector also has the correct derivative at zero.
    zero = torch.zeros((1, 1), dtype=torch.float64, requires_grad=True)
    forward = dist.transforms[0](zero)
    derivative = torch.autograd.grad(forward.sum(), zero)[0]
    np.testing.assert_allclose(derivative.numpy().item(), jac[4], rtol=1e-14)


@pytest.mark.parametrize("D,K", [(1, 1), (3, 3)])
@pytest.mark.parametrize("orig_flag", [False, True])
@pytest.mark.parametrize("shape", [(), (7,), (2, 3)])
def test_sample_shapes_and_log_prob(D, K, orig_flag, shape):
    dist = _vp(D, K).to_torch(orig_flag=orig_flag)
    draws = dist.sample(torch.Size(shape))
    assert dist.batch_shape == torch.Size()
    assert dist.event_shape == torch.Size((D,))
    assert draws.shape == torch.Size(shape + (D,))
    assert dist.log_prob(draws).shape == torch.Size(shape)
    assert not dist.has_rsample
    expanded = dist.expand((2,)).sample((3,))
    assert expanded.shape == (3, 2, D)


def test_snapshot_rng_and_dtype_canary():
    vp = _vp()
    before = copy.deepcopy(vp)
    rng_before = copy.deepcopy(vp.rng.bit_generator.state)
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()
    dist = vp.to_torch()
    assert vp.rng.bit_generator.state == rng_before
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert torch.equal(torch_before, torch.random.get_rng_state())
    point = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float64)
    reference = dist.log_prob(point).clone()
    for obj, names in (
        (vp, ("w", "mu", "sigma", "lambd")),
        (
            vp.parameter_transformer,
            ("mu", "delta", "scale", "R_mat", "lb_orig", "ub_orig"),
        ),
    ):
        for name in names:
            getattr(obj, name)[:] = 42
    assert torch.equal(reference, dist.log_prob(point))
    # A float32 export must never install float32 arrays/tensors on the VP.
    before.to_torch(dtype=torch.float32)
    assert_float64(before, min_leaves=10)
    torch.manual_seed(111)
    first = dist.sample((10,))
    torch.manual_seed(111)
    assert torch.equal(first, dist.sample((10,)))
    assert vp.rng.bit_generator.state == rng_before
    # Mutating the returned tensors cannot mutate NumPy either.
    export = before.to_torch(orig_flag=False)
    mu = before.mu.copy()
    export.component_distribution.base_dist.loc.add_(3)
    np.testing.assert_array_equal(before.mu, mu)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_explicit_dtype_device(dtype):
    vp = _vp()
    dist = vp.to_torch(dtype=dtype, device="cpu")
    draws = dist.sample((4,))
    assert draws.dtype == dtype
    assert draws.device.type == "cpu"
    assert dist.log_prob(draws).dtype == dtype
    assert_float64(vp, min_leaves=10)


def test_explicit_defaults_ignore_torch_globals():
    old_dtype = torch.get_default_dtype()
    old_device = torch.get_default_device()
    try:
        torch.set_default_dtype(torch.float32)
        torch.set_default_device("meta")
        dist = _vp().to_torch()
        assert (
            dist.base_dist.component_distribution.base_dist.loc.dtype
            == torch.float64
        )
        assert (
            dist.base_dist.component_distribution.base_dist.loc.device.type
            == "cpu"
        )
        assert torch.get_default_dtype() == torch.float32
        assert torch.get_default_device().type == "meta"
    finally:
        torch.set_default_device(old_device)
        torch.set_default_dtype(old_dtype)


@pytest.mark.parametrize(
    "dtype",
    [torch.int64, torch.complex128, torch.float16, torch.bfloat16, "float64"],
)
def test_invalid_dtype(dtype):
    with pytest.raises(ValueError, match="dtype"):
        _vp().to_torch(dtype=dtype)


@pytest.mark.parametrize("kind", ["logit", "probit", "student4"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_strict_bounds_and_sample_only_correction(kind, dtype):
    vp = _vp(1, 1, kind, warp=False)
    vp.mu[:] = 1e10
    dist = vp.to_torch(dtype=dtype)
    lb, ub = dist.transforms[0].lower, dist.transforms[0].upper
    for invalid in (
        lb,
        ub,
        lb - 1,
        ub + 1,
        lb * float("nan"),
        lb * float("inf"),
    ):
        assert not dist.support.check(invalid)
        with pytest.raises(ValueError, match="support"):
            dist.log_prob(invalid)
    draws = dist.sample((5,))
    assert dist.support.check(draws).all()
    assert torch.equal(draws[0], torch.nextafter(ub, lb))
    # Saturation is untouched in the analytic transform.
    raw = dist.transforms[0](torch.full((1,), 1e10, dtype=dtype))
    assert torch.equal(raw, ub)


@pytest.mark.parametrize(
    "attribute,value,message",
    [
        ("scale", np.array([1.0, 0.0, 1.0]), "positive"),
        ("scale", np.array([1.0, np.inf, 1.0]), "positive"),
        ("delta", np.array([1.0, -1.0, 1.0]), "positive"),
        ("R_mat", np.diag([1.0, 2.0, 1.0]), "orthogonal"),
    ],
)
def test_invalid_transform_state(attribute, value, message):
    vp = _vp()
    setattr(vp.parameter_transformer, attribute, value)
    with pytest.raises(ValueError, match=message):
        vp.to_torch()
    # Internal coordinates do not need a parameter transform.
    vp.to_torch(orig_flag=False)


def test_unrepresentable_bounds():
    vp = _vp(1, 1)
    vp.parameter_transformer.lb_orig[:] = 1
    vp.parameter_transformer.ub_orig[:] = np.nextafter(1.0, np.inf)
    with pytest.raises(ValueError, match="representable interior"):
        vp.to_torch()


def test_zero_weight_is_exactly_absent():
    vp = _vp(1, 2, bounded=False)
    vp.w[:] = [[1, 0]]
    vp.mu[:] = [[0, 30]]
    vp.sigma[:] = 1
    vp.lambd[:] = 1
    dist = vp.to_torch(orig_flag=False)
    point = torch.tensor([[30.0]], dtype=torch.float64)
    expected = torch.distributions.Normal(0.0, 1.0).log_prob(point).squeeze(-1)
    torch.testing.assert_close(dist.log_prob(point), expected)
    assert torch.isneginf(dist.mixture_distribution.logits[1])


def test_old_saved_posterior():
    vp = VariationalPosterior.load(
        Path(__file__).with_name("test_vp_save_static.pkl")
    )
    dist = vp.to_torch(orig_flag=False)
    points = torch.tensor(vp.mu.T.copy(), dtype=torch.float64)
    np.testing.assert_allclose(
        dist.log_prob(points).numpy(),
        vp.log_pdf(points.numpy(), orig_flag=False).ravel(),
        atol=1e-10,
    )
    assert_float64(vp, min_leaves=5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_export():
    dist = _vp().to_torch(device="cuda")
    x = dist.sample((3,)).requires_grad_()
    assert x.device.type == "cuda"
    assert torch.isfinite(
        torch.autograd.grad(dist.log_prob(x).sum(), x)[0]
    ).all()


def test_unbounded_original_whitening():
    vp = _vp()
    pt = vp.parameter_transformer
    pt.lb_orig[:] = -np.inf
    pt.ub_orig[:] = np.inf
    pt.type[:] = 0
    u = np.array([[-0.4, 0.2, 0.7], [0.3, -0.5, 0.1]])
    x = torch.tensor(pt.inverse(u), dtype=torch.float64, requires_grad=True)
    dist = vp.to_torch()
    np.testing.assert_allclose(
        dist.log_prob(x).detach().numpy(),
        vp.log_pdf(x.detach().numpy()).ravel(),
        atol=1e-12,
    )
    assert torch.autograd.gradcheck(dist.log_prob, (x,))
    assert not dist.support.check(torch.full((3,), float("inf")))


def test_mixed_bounded_transform_types():
    vp = _vp(D=4)
    pt = vp.parameter_transformer
    pt.type[:] = [3, 12, 13, 0]
    pt.bounded_types = [3, 12, 13]
    pt._set_bounded_transforms()
    u = np.array([[-0.2, 0.1, 0.5, -0.4]])
    x = torch.tensor(pt.inverse(u), dtype=torch.float64, requires_grad=True)
    dist = vp.to_torch()
    np.testing.assert_allclose(
        dist.log_prob(x).detach().numpy(),
        vp.log_pdf(x.detach().numpy()).ravel(),
        atol=1e-12,
    )
    assert torch.autograd.gradcheck(dist.log_prob, (x,))


@pytest.mark.parametrize(
    "name,value,message",
    [
        ("mu", np.full((3, 3), np.inf), "finite"),
        ("sigma", np.full((1, 3), np.inf), "finite"),
        ("lambd", np.full((3, 1), np.inf), "finite"),
        ("sigma", np.full((1, 3), -1.0), "positive"),
        ("lambd", np.zeros((3, 1)), "positive"),
        ("w", np.zeros((1, 3)), "nonzero"),
        ("w", np.array([[1.0, -1.0, 1.0]]), "nonnegative"),
        ("mu", np.ones((9,)), "shape"),
        ("w", np.ones((3, 1)), "shape"),
        ("sigma", np.ones((3, 1)), "shape"),
        ("lambd", np.ones((1, 3)), "shape"),
    ],
)
def test_invalid_mixture_state(name, value, message):
    vp = _vp()
    setattr(vp, name, value)
    with pytest.raises(ValueError, match=message):
        vp.to_torch(orig_flag=False)


def test_export_dtype_scale_overflow():
    vp = _vp()
    vp.sigma[:] = 1e30
    vp.lambd[:] = 1e30
    with pytest.raises(ValueError, match="Component scales"):
        vp.to_torch(dtype=torch.float32)


@pytest.mark.parametrize("kind", ["logit", "probit", "student4"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("upper_tail", [False, True])
def test_subnormal_interior_endpoints_have_finite_density(
    kind, dtype, upper_tail
):
    lower, upper = (-5.0, 0.0) if upper_tail else (0.0, 5.0)
    pt = ParameterTransformer(
        1, np.array([[lower]]), np.array([[upper]]), transform_type=kind
    )
    vp = VariationalPosterior(1, K=1, parameter_transformer=pt, rng=4)
    vp.mu[:] = 1e10 if upper_tail else -1e10
    dist = vp.to_torch(dtype=dtype)
    boundary = torch.tensor(0.0, dtype=dtype)
    direction = torch.tensor(-1.0 if upper_tail else 1.0, dtype=dtype)
    point = torch.nextafter(boundary, direction).reshape(1, 1)
    assert dist.support.check(point).all()
    assert torch.isfinite(dist.log_prob(point)).all()
    samples = dist.sample((4,))
    assert dist.support.check(samples).all()
    assert torch.isfinite(dist.log_prob(samples)).all()
