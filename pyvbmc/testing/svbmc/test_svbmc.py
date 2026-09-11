"""Behavior of :class:`~pyvbmc.svbmc.SVBMC` on the fixture posteriors.

The optimizations here are deliberately short (a few steps, few entropy
draws): what is checked is the contract — shapes, dtypes, where the
randomness comes from, that the input posteriors are left alone, and that
the stacked posterior covers its targets — not convergence. The
per-mode numbers are pinned by ``test_svbmc_references.py``.
"""

import copy
import logging

import matplotlib
import numpy as np
import pytest

pytest.importorskip("torch")

import torch  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from pyvbmc.svbmc import SVBMC  # noqa: E402
from pyvbmc.svbmc.svbmc import _balanced_counts  # noqa: E402
from pyvbmc.testing._dtype import assert_float64  # noqa: E402
from pyvbmc.testing.svbmc._fixtures import load_group  # noqa: E402

VERSIONS = ["all-weights", "posterior-only", "ns"]


@pytest.fixture(scope="module", autouse=True)
def quiet_logger():
    """Progress messages are checked in ``test_svbmc_filters.py``."""
    logger = logging.getLogger("SVBMC")
    previous = logger.level
    logger.setLevel(logging.WARNING)
    yield
    logger.setLevel(previous)


@pytest.fixture(scope="module", autouse=True)
def agg_backend():
    """``plot`` shows the figure on an interactive backend; avoid that."""
    from matplotlib import pyplot as plt

    previous = matplotlib.get_backend()
    plt.switch_backend("Agg")
    yield
    plt.switch_backend(previous)


@pytest.fixture(scope="module")
def gmm_vps():
    return load_group("upstream_GMM", rng=0)[0]


@pytest.fixture(scope="module")
def d1_vps():
    return load_group("normal_D1", rng=0)[0]


@pytest.fixture(scope="module")
def d2_vps():
    return load_group("bounded_D2", rng=0)[0]


@pytest.fixture(scope="module")
def d3_vps():
    return load_group("corr_D3", rng=0)[0]


def _stack(vp_list, seed=0, **kwargs):
    """A stacked posterior optimized with cheap settings."""
    stacked = SVBMC(vp_list, seed=seed)
    stacked.optimize(
        n_samples=kwargs.pop("n_samples", 5), max_steps=3, **kwargs
    )
    return stacked


@pytest.fixture(scope="module")
def stacked_d1(d1_vps):
    return _stack(d1_vps)


@pytest.fixture(scope="module")
def stacked_d2(d2_vps):
    return _stack(d2_vps)


@pytest.fixture(scope="module")
def stacked_d3(d3_vps):
    return _stack(d3_vps)


# --------------------------------------------------------------------- #
# (a) what optimize() leaves behind                                     #
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("version", VERSIONS)
def test_optimize_sets_weights_elbo_and_entropy(gmm_vps, version):
    stacked = SVBMC(gmm_vps, seed=0)
    assert stacked.M == 10
    assert stacked.K == [50] * 10
    assert stacked.elbo is None and stacked.entropy is None

    assert stacked.optimize(n_samples=5, max_steps=3, version=version) is None

    assert stacked.w.shape == (1, 500)
    assert stacked.w.dtype == np.float64
    assert np.all(stacked.w >= 0)
    assert stacked.w.sum() == pytest.approx(1.0)
    assert set(stacked.elbo) == {
        "estimated",
        "debiased_I_median",
        "debiased_E_median",
    }
    assert all(isinstance(v, float) for v in stacked.elbo.values())
    assert all(np.isfinite(v) for v in stacked.elbo.values())
    assert stacked.elbo["debiased_I_median"] <= stacked.elbo["estimated"]
    assert stacked.elbo["debiased_E_median"] <= stacked.elbo["estimated"]
    assert isinstance(stacked.entropy, float)
    assert np.isfinite(stacked.entropy)


def test_unknown_version_raises(d1_vps):
    stacked = SVBMC(d1_vps, seed=0)
    with pytest.raises(ValueError, match="Unknown S-VBMC version"):
        stacked.optimize(n_samples=2, max_steps=1, version="nope")
    with pytest.raises(ValueError, match="Unknown S-VBMC version"):
        stacked.maximize_ELBO(n_samples=2, max_steps=1, version="nope")


def test_zero_steps_raises(d1_vps):
    stacked = SVBMC(d1_vps, seed=0)
    with pytest.raises(ValueError, match="max_steps"):
        stacked.optimize(n_samples=2, max_steps=0)


# --------------------------------------------------------------------- #
# (b) randomness comes from the object's generator                      #
# --------------------------------------------------------------------- #
def test_same_seed_reproduces_everything(d1_vps):
    first = _stack(d1_vps, seed=7)
    second = _stack(d1_vps, seed=7)
    np.testing.assert_array_equal(first.w, second.w)
    assert first.elbo == second.elbo
    assert first.entropy == second.entropy
    np.testing.assert_array_equal(first.sample(64), second.sample(64))


def test_different_seeds_give_different_draws(d1_vps):
    first = _stack(d1_vps, seed=7)
    second = _stack(d1_vps, seed=8)
    assert not np.array_equal(first.sample(64), second.sample(64))


def test_unseeded_follows_the_global_state(d1_vps):
    state = np.random.get_state()
    try:
        np.random.seed(12345)
        first = _stack(d1_vps, seed=None)
        first_draws = first.sample(32)
        np.random.seed(12345)
        second = _stack(d1_vps, seed=None)
        second_draws = second.sample(32)
    finally:
        np.random.set_state(state)
    np.testing.assert_array_equal(first.w, second.w)
    np.testing.assert_array_equal(first_draws, second_draws)


# --------------------------------------------------------------------- #
# (c) the input posteriors are read, never written                      #
# --------------------------------------------------------------------- #
def test_input_posteriors_are_untouched(d2_vps):
    states = [copy.deepcopy(vp.rng.bit_generator.state) for vp in d2_vps]
    weights = [vp.w.copy() for vp in d2_vps]

    stacked = _stack(d2_vps, seed=1)
    stacked.sample(128)
    stacked.sample(128, balance_flag=True)

    for vp, state, w in zip(d2_vps, states, weights):
        assert vp.rng.bit_generator.state == state
        assert vp.w.shape == (1, 50)
        np.testing.assert_array_equal(vp.w, w)


# --------------------------------------------------------------------- #
# (d) sample(): exact row counts                                        #
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [0, 1, 7, 64, 1001])
@pytest.mark.parametrize("balance_flag", [False, True])
def test_sample_returns_exactly_n_rows(stacked_d2, n, balance_flag):
    X = stacked_d2.sample(n, balance_flag=balance_flag)
    assert X.shape == (n, 2)
    assert X.dtype == np.float64
    assert np.all(np.isfinite(X))


@pytest.mark.parametrize("n", [0, 1, 3, 7, 64, 1001])
@pytest.mark.parametrize(
    "omega",
    [
        np.array([1.0]),
        np.array([0.5, 0.5]),
        np.array([0.5, 0.0, 0.5]),
        np.array([0.2, 0.3, 0.5]),
        np.array([0.7, 0.1, 0.1, 0.1]),
        np.full(10, 0.1),
    ],
)
def test_balanced_counts_are_exact_and_proportional(n, omega):
    rng = np.random.default_rng(0)
    counts = _balanced_counts(n, omega, rng)
    assert counts.sum() == n
    assert np.all(counts >= 0)
    assert np.all(np.abs(counts - n * omega) <= 1.0 + 1e-12)


# --------------------------------------------------------------------- #
# (e) the stacked posterior covers its target                           #
# --------------------------------------------------------------------- #
def test_bounded_samples_stay_inside_the_box(stacked_d2):
    X = stacked_d2.sample(4000)
    assert np.all(X > -3.0) and np.all(X < 3.0)
    # The target is N((0.5, -0.5), diag(0.8^2, 0.6^2)) on the box.
    np.testing.assert_allclose(X.mean(axis=0), [0.5, -0.5], atol=0.25)


def test_correlated_samples_recover_scales_and_sign(stacked_d3):
    X = stacked_d3.sample(4000)
    assert X.shape == (4000, 3)
    expected_sd = np.array([1.0, 2.0, 0.5])
    np.testing.assert_allclose(X.std(axis=0), expected_sd, rtol=0.25)
    assert np.corrcoef(X[:, 0], X[:, 1])[0, 1] > 0.0


def test_one_dimensional_samples_recover_the_mean(stacked_d1):
    X = stacked_d1.sample(4000)
    assert X.shape == (4000, 1)
    assert abs(X.mean() - 0.5) < 0.5


# --------------------------------------------------------------------- #
# (f) D = 1                                                             #
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("version", VERSIONS)
def test_one_dimensional_stacking_runs(d1_vps, version):
    stacked = SVBMC(d1_vps, seed=0)
    stacked.optimize(n_samples=5, max_steps=3, version=version)
    assert stacked.D == 1
    assert stacked.w.shape == (1, 150)
    assert stacked.sample(10).shape == (10, 1)


def test_single_entropy_draw_per_component(d1_vps):
    stacked = SVBMC(d1_vps, seed=0)
    stacked.optimize(n_samples=1, max_steps=2)
    assert np.isfinite(stacked.entropy)
    assert stacked.w.sum() == pytest.approx(1.0)


# --------------------------------------------------------------------- #
# (g) heterogeneous parameter transforms                                #
# --------------------------------------------------------------------- #
def test_bounded_runs_have_different_transforms(d2_vps, stacked_d2):
    transforms = [
        (
            np.ravel(vp.parameter_transformer.mu).copy(),
            np.ravel(vp.parameter_transformer.delta).copy(),
        )
        for vp in d2_vps
    ]
    distinct = {(tuple(mu), tuple(delta)) for mu, delta in transforms}
    assert len(distinct) == len(d2_vps)
    assert stacked_d2.M == 3
    assert np.isfinite(stacked_d2.elbo["estimated"])


def test_warped_and_unwarped_runs_stack(d3_vps, stacked_d3):
    rotations = [vp.parameter_transformer.R_mat for vp in d3_vps]
    assert rotations[0] is None
    assert all(R is not None for R in rotations[1:])
    assert not np.array_equal(rotations[1], rotations[2])
    assert stacked_d3.M == 3
    assert stacked_d3.w.shape == (1, 150)
    assert np.isfinite(stacked_d3.elbo["estimated"])


# --------------------------------------------------------------------- #
# (h) float64 policy                                                    #
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("version", VERSIONS)
def test_maximize_ELBO_returns_float64_tensors(d1_vps, version):
    stacked = SVBMC(d1_vps, seed=0)
    w, elbo, entropy = stacked.maximize_ELBO(
        n_samples=2, max_steps=2, version=version
    )
    assert w.dtype is torch.float64
    assert w.ndim == 1 and w.shape[0] == 150
    assert elbo.dtype is torch.float64 and elbo.ndim == 0
    assert entropy.dtype is torch.float64 and entropy.ndim == 0


def test_stacked_ELBO_accepts_arrays_lists_and_float32(d1_vps):
    stacked = SVBMC(d1_vps, seed=0)
    default_dtype = torch.get_default_dtype()
    K_total = int(np.sum(stacked.K))
    inputs = [
        np.full(K_total, 1.0 / K_total),
        [1.0] * K_total,
        torch.ones(K_total, dtype=torch.float32),
        torch.ones(K_total, dtype=torch.float64),
    ]
    for w in inputs:
        elbo, entropy = stacked.stacked_ELBO(w, n_samples=2)
        assert elbo.dtype is torch.float64
        assert entropy.dtype is torch.float64
        assert torch.isfinite(elbo) and torch.isfinite(entropy)
    assert torch.get_default_dtype() is default_dtype


def test_array_input_is_normalized(d1_vps):
    """A NumPy or list input is a weight vector, normalized on the way in.

    Two objects seeded alike draw the same entropy samples, so the ELBO of
    ``np.ones(K)`` must equal the ELBO of the normalized tensor.
    """
    K_total = 150
    from_array = SVBMC(d1_vps, seed=0).stacked_ELBO(
        np.ones(K_total), n_samples=2
    )[0]
    from_tensor = SVBMC(d1_vps, seed=0).stacked_ELBO(
        torch.full((K_total,), 1.0 / K_total, dtype=torch.float64),
        n_samples=2,
    )[0]
    assert float(from_array) == pytest.approx(float(from_tensor), rel=1e-12)
    # An unnormalized tensor is a weight vector too.
    from_raw_tensor = SVBMC(d1_vps, seed=0).stacked_ELBO(
        torch.ones(K_total, dtype=torch.float64), n_samples=2
    )[0]
    assert float(from_raw_tensor) == pytest.approx(
        float(from_tensor), rel=1e-12
    )


def test_stacked_ELBO_rejects_other_types(d1_vps):
    stacked = SVBMC(d1_vps, seed=0)
    with pytest.raises(TypeError, match="should be a torch.Tensor"):
        stacked.stacked_ELBO(object())


# --------------------------------------------------------------------- #
# (i) the dtype canary                                                  #
# --------------------------------------------------------------------- #
def test_stacked_state_is_float64(stacked_d3):
    assert_float64(stacked_d3.w, "svbmc.w")
    assert_float64(stacked_d3, "svbmc", min_leaves=10)
    assert np.asarray(stacked_d3.sample(16)).dtype == np.float64


# --------------------------------------------------------------------- #
# (j) plotting                                                          #
# --------------------------------------------------------------------- #
def test_plot_returns_a_figure(stacked_d2):
    from matplotlib import pyplot as plt

    fig = stacked_d2.plot(n_samples=200)
    try:
        assert isinstance(fig, Figure)
        assert len(fig.axes) > 0
    finally:
        plt.close(fig)


# --------------------------------------------------------------------- #
# (k) the ELBO is differentiable in the weights                         #
# --------------------------------------------------------------------- #
def test_stacked_ELBO_is_differentiable(d1_vps):
    stacked = SVBMC(d1_vps, seed=0)
    K_total = int(np.sum(stacked.K))
    w = torch.full(
        (K_total,), 1.0 / K_total, dtype=torch.float64, requires_grad=True
    )
    elbo, _ = stacked.stacked_ELBO(w, n_samples=3)
    elbo.backward()
    assert w.grad is not None
    assert w.grad.shape == w.shape
    assert w.grad.dtype is torch.float64
    assert torch.isfinite(w.grad).all()
    assert torch.any(w.grad != 0.0)
