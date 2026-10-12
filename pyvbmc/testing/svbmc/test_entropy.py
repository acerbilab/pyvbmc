"""The vectorized entropy preparation against its literal per-component form.

``SVBMC._stacked_entropy`` maps the draws of a run's components to the
original space with one inverse transform, evaluates each run's forward
transform and Jacobian once for all draws, forms the log densities of a
run's components in one broadcast and reduces per component with array
operations. The references below are literal per-component transcriptions
of the same estimator (``scipy.stats.norm.logpdf`` and a masked loop in
torch); the two must agree on the density matrix, the entropy, its weight
gradient, the stratified variance and the state of the generator.

When no gradient is needed, the density matrix is formed and reduced a
chunk of rows at a time instead of whole. At every chunk size, from one row
to more rows than the matrix has, that path must equal the whole matrix
exactly: the same calls to the generator, the same rows, the same mixture
log densities, entropy and variance.

The transforms of ``torch.func`` give the weights a batch dimension or a
tangent and no ``requires_grad``, so they take the chunked path. At the
same draws, ``vmap`` over a batch of weight vectors must equal one call per
vector, and the forward-mode Jacobian (``jacfwd``) must equal the
reverse-mode gradient, up to rounding in both cases.
"""

import importlib
import logging

import numpy as np
import pytest
import scipy.stats

from pyvbmc.svbmc import _entropy
from pyvbmc.svbmc._entropy import component_log_densities, log_density_chunks
from pyvbmc.testing.svbmc._fixtures import load_group

GROUPS = ["normal_D1", "bounded_D2", "corr_D3", "upstream_GMM"]
# One draw per component is the degenerate layout and, for warped runs, the
# case where the batched inverse transform rounds least like the reference.
DRAWS = [1, 3]
# One draw, the fewest with a variance, and the final evaluation's default.
CHUNK_DRAWS = [1, 2, 100]
TOLERANCE = dict(rtol=1e-12, atol=1e-12)


def _posteriors(group):
    vps = load_group(group, rng=0)[0]
    if group == "upstream_GMM":
        # Three of the ten runs, fifty components each; ``_01`` is warped.
        vps = vps[:3]
    return vps


def _chunk_sizes(S):
    """One row, a few, and around all ``S``; ``None`` is the default."""
    return [1, 7, S - 1, S, S + 1, None]


class _RecordingGenerator:
    """A NumPy generator that records every call made to it."""

    def __init__(self, seed):
        self._rng = np.random.default_rng(seed)
        self.calls = []

    def __getattr__(self, name):
        method = getattr(self._rng, name)

        def recorded(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return method(*args, **kwargs)

        return recorded


def _reference_log_densities(vp_list, n_samples, rng):
    """One component at a time, as the estimator was first written."""
    subcomps = []
    for vp in vp_list:
        sigma = vp.lambd * vp.sigma
        for k in range(vp.mu.shape[1]):
            subcomps.append(
                (vp.parameter_transformer, vp.mu[:, k], sigma[:, k])
            )
    D = vp_list[0].D
    K_total = len(subcomps)
    S = K_total * n_samples

    X_orig = np.zeros((S, D))
    for mk, (transform, mu, sigma) in enumerate(subcomps):
        z = rng.standard_normal((n_samples, D))
        rows = slice(mk * n_samples, (mk + 1) * n_samples)
        X_orig[rows, :] = transform.inverse(z * sigma + mu)

    logq = np.zeros((S, K_total))
    for mk, (transform, mu, sigma) in enumerate(subcomps):
        U = transform(X_orig)
        jac = transform.log_abs_det_jacobian(U)
        logq[:, mk] = (
            np.sum(scipy.stats.norm.logpdf(U, mu, sigma), axis=1) - jac
        )
    return logq


def _reference_entropy(logq, w, n_samples, torch):
    """The masked per-component reduction the estimator first used."""
    K_total = w.numel()
    w = w.to(torch.float64).reshape(1, K_total)
    w = w / w.sum()
    log_w = torch.log(w + 1e-40)
    logq_orig = torch.logsumexp(
        torch.as_tensor(logq, dtype=torch.float64) + log_w, dim=1
    )
    comp_index = np.repeat(np.arange(K_total), n_samples)
    sum_logq = torch.zeros(K_total, dtype=torch.float64)
    count_logq = torch.zeros(K_total, dtype=torch.float64)
    var_logq = torch.zeros(K_total, dtype=torch.float64)
    for mk in range(K_total):
        mask = comp_index == mk
        sum_logq[mk] = logq_orig[mask].sum()
        count_logq[mk] = mask.sum()
        if n_samples > 1:
            var_logq[mk] = logq_orig[mask].var(unbiased=True)
    E_mk_logq = sum_logq / (count_logq + 1e-40)
    H = -w @ E_mk_logq
    varH = None
    if n_samples > 1:
        varH = float((w.square() @ (var_logq / n_samples)).item())
    return H[0], varH


@pytest.mark.parametrize("n_samples", DRAWS)
@pytest.mark.parametrize("group", GROUPS)
def test_log_densities_match_the_literal_reference(group, n_samples):
    vps = _posteriors(group)
    K_total = sum(vp.K for vp in vps)
    rng = np.random.default_rng(11)
    rng_reference = np.random.default_rng(11)

    logq = component_log_densities(vps, n_samples, rng)
    expected = _reference_log_densities(vps, n_samples, rng_reference)

    assert logq.shape == (K_total * n_samples, K_total)
    assert logq.dtype == np.float64
    assert np.all(np.isfinite(logq))
    np.testing.assert_allclose(logq, expected, **TOLERANCE)
    assert rng.bit_generator.state == rng_reference.bit_generator.state


@pytest.mark.parametrize("group", ["bounded_D2", "corr_D3"])
def test_chunking_does_not_change_the_values(group):
    vps = _posteriors(group)
    whole = component_log_densities(vps, 3, np.random.default_rng(3))
    # A one-byte budget processes the rows one at a time.
    chunked = component_log_densities(
        vps, 3, np.random.default_rng(3), workspace_bytes=1
    )
    np.testing.assert_array_equal(chunked, whole)


def test_at_least_one_draw_is_required():
    vps = _posteriors("normal_D1")
    with pytest.raises(ValueError, match="n_samples"):
        component_log_densities(vps, 0, np.random.default_rng(0))


@pytest.mark.parametrize("n_samples", DRAWS)
@pytest.mark.parametrize("group", GROUPS)
def test_stacked_entropy_matches_the_literal_reference(group, n_samples):
    torch = pytest.importorskip("torch")
    from pyvbmc.svbmc import SVBMC

    logging.getLogger("SVBMC").setLevel(logging.WARNING)
    vps = _posteriors(group)
    stacked = SVBMC(vps, seed=5)
    # Construction consumes no draws, so a twin starts from the same state.
    twin = SVBMC(vps, seed=5)
    K_total = int(np.sum(stacked.K))
    # Unnormalized, uneven weights exercise the normalization too.
    w_np = np.linspace(0.5, 2.0, K_total)

    w = torch.tensor(w_np, dtype=torch.float64, requires_grad=True)
    if n_samples == 1:
        # A single draw has no sample variance; the estimate itself is fine.
        H, corrections, varH = stacked._stacked_entropy(w, n_samples)
    else:
        H, corrections, varH = stacked._stacked_entropy(
            w, n_samples, compute_variance=True
        )
    H.backward()

    # The reference works on the runs the object retained.
    logq = _reference_log_densities(stacked.vp_list, n_samples, twin.rng)
    w_ref = torch.tensor(w_np, dtype=torch.float64, requires_grad=True)
    H_ref, varH_ref = _reference_entropy(logq, w_ref, n_samples, torch)
    H_ref.backward()

    assert H.dtype == torch.float64 and H.shape == ()
    np.testing.assert_allclose(H.item(), H_ref.item(), **TOLERANCE)
    if n_samples > 1:
        np.testing.assert_allclose(varH, varH_ref, **TOLERANCE)
    else:
        assert varH is None
    np.testing.assert_allclose(w.grad.numpy(), w_ref.grad.numpy(), **TOLERANCE)
    np.testing.assert_array_equal(corrections, stacked._jacobian_corrections)
    assert corrections is not stacked._jacobian_corrections
    assert stacked.rng.bit_generator.state == twin.rng.bit_generator.state


@pytest.mark.parametrize("n_samples", CHUNK_DRAWS)
@pytest.mark.parametrize("group", GROUPS)
def test_chunks_equal_the_whole_matrix(group, n_samples):
    vps = _posteriors(group)
    rng_whole = np.random.default_rng(13)
    whole = component_log_densities(vps, n_samples, rng_whole)
    S = whole.shape[0]

    for chunk_rows in _chunk_sizes(S):
        rng = np.random.default_rng(13)
        chunks = log_density_chunks(vps, n_samples, rng, chunk_rows=chunk_rows)
        # Every draw is made by the call, before the first chunk.
        assert rng.bit_generator.state == rng_whole.bit_generator.state
        covered = 0
        for r0, r1, logq_rows in chunks:
            assert r0 == covered
            if chunk_rows is not None:
                assert r1 - r0 == min(chunk_rows, S - r0)
            np.testing.assert_array_equal(logq_rows, whole[r0:r1])
            covered = r1
        assert covered == S
        assert rng.bit_generator.state == rng_whole.bit_generator.state


def test_chunks_make_the_calls_of_the_whole_matrix():
    vps = _posteriors("corr_D3")
    K_total = sum(vp.K for vp in vps)
    whole = _RecordingGenerator(2)
    component_log_densities(vps, 2, whole)
    chunked = _RecordingGenerator(2)
    for _ in log_density_chunks(vps, 2, chunked, chunk_rows=5):
        pass

    draw = ("standard_normal", ((2, vps[0].D),), {})
    assert whole.calls == [draw] * K_total
    assert chunked.calls == whole.calls


def test_at_least_one_row_per_chunk_is_required():
    vps = _posteriors("normal_D1")
    rng = np.random.default_rng(0)
    state = rng.bit_generator.state
    with pytest.raises(ValueError, match="chunk_rows"):
        log_density_chunks(vps, 2, rng, chunk_rows=0)
    assert rng.bit_generator.state == state


@pytest.mark.parametrize("rows", [0, 1, 40])
def test_default_chunks_are_the_largest_within_the_bound(rows, monkeypatch):
    vps = _posteriors("bounded_D2")
    K_total = sum(vp.K for vp in vps)
    row_bytes = _entropy._CHUNK_COPIES * 8 * K_total
    # A bound that holds ``rows`` rows and part of another.
    monkeypatch.setattr(_entropy, "_CHUNK_BYTES", rows * row_bytes + 8)
    chunks = log_density_chunks(vps, 3, np.random.default_rng(0))
    sizes = [r1 - r0 for r0, r1, _ in chunks]

    assert sum(sizes) == K_total * 3
    # At least one row, whatever the bound.
    assert sizes[0] == max(rows, 1)
    assert all(size <= sizes[0] for size in sizes)


@pytest.mark.parametrize("n_samples", CHUNK_DRAWS)
@pytest.mark.parametrize("group", GROUPS)
def test_chunked_entropy_equals_the_whole_matrix(
    group, n_samples, monkeypatch
):
    torch = pytest.importorskip("torch")
    from pyvbmc.svbmc import SVBMC

    logging.getLogger("SVBMC").setLevel(logging.WARNING)
    vps = _posteriors(group)
    compute_variance = n_samples > 1
    # Record the mixture log density at every draw, chunk by chunk.
    mixture_log_densities = []
    logsumexp = torch.logsumexp

    def recording(*args, **kwargs):
        out = logsumexp(*args, **kwargs)
        mixture_log_densities.append(out.detach().clone())
        return out

    monkeypatch.setattr(torch, "logsumexp", recording)

    # A weight gradient reduces the whole matrix.
    whole = SVBMC(vps, seed=5)
    K_total = int(np.sum(whole.K))
    w_np = np.linspace(0.5, 2.0, K_total)
    w = torch.tensor(w_np, dtype=torch.float64, requires_grad=True)
    mixture_log_densities.clear()
    H_whole, _, varH_whole = whole._stacked_entropy(
        w, n_samples, compute_variance=compute_variance
    )
    (logq_whole,) = mixture_log_densities
    S = logq_whole.numel()

    for chunk_rows in _chunk_sizes(S):
        stacked = SVBMC(vps, seed=5)
        mixture_log_densities.clear()
        with torch.no_grad():
            H, _, varH = stacked._stacked_entropy(
                torch.as_tensor(w_np, dtype=torch.float64),
                n_samples,
                compute_variance=compute_variance,
                chunk_rows=chunk_rows,
            )
        if chunk_rows is not None:
            assert len(mixture_log_densities) == -(-S // chunk_rows)
        assert torch.equal(torch.cat(mixture_log_densities), logq_whole)
        assert H.item() == H_whole.item()
        assert varH == varH_whole
        assert stacked.rng.bit_generator.state == whole.rng.bit_generator.state


def test_only_a_weight_gradient_reduces_the_whole_matrix(monkeypatch):
    torch = pytest.importorskip("torch")
    from pyvbmc.svbmc import SVBMC

    svbmc_module = importlib.import_module("pyvbmc.svbmc.svbmc")
    paths = []

    def spy(name, function):
        def spied(*args, **kwargs):
            paths.append(name)
            return function(*args, **kwargs)

        return spied

    for name in ("component_log_densities", "log_density_chunks"):
        function = getattr(svbmc_module, name)
        monkeypatch.setattr(svbmc_module, name, spy(name, function))

    logging.getLogger("SVBMC").setLevel(logging.WARNING)
    stacked = SVBMC(_posteriors("bounded_D2"), seed=0)
    w = torch.ones(int(np.sum(stacked.K)), dtype=torch.float64)
    stacked._stacked_entropy(w.requires_grad_(True), 2)
    with torch.no_grad():
        stacked._stacked_entropy(w, 2)
    stacked._stacked_entropy(w.detach(), 2)

    assert paths == [
        "component_log_densities",
        "log_density_chunks",
        "log_density_chunks",
    ]


# The default chunk, which holds every row of these stacks, and chunks of
# seven rows.
FUNC_CHUNK_ROWS = [None, 7]
FUNC_GROUPS = ["bounded_D2", "upstream_GMM"]


def _entropy_at_fixed_draws(stacked, n_samples):
    """``stacked_entropy`` with the generator reset before every call."""
    state = stacked.rng.bit_generator.state

    def entropy(w):
        stacked.rng.bit_generator.state = state
        return stacked.stacked_entropy(w, n_samples)[0]

    return entropy


def _set_default_chunk_rows(monkeypatch, K_total, chunk_rows):
    """Make the default chunk hold ``chunk_rows`` rows, unless ``None``."""
    if chunk_rows is not None:
        row_bytes = _entropy._CHUNK_COPIES * 8 * K_total
        monkeypatch.setattr(_entropy, "_CHUNK_BYTES", chunk_rows * row_bytes)


@pytest.mark.parametrize("chunk_rows", FUNC_CHUNK_ROWS)
@pytest.mark.parametrize("group", FUNC_GROUPS)
def test_vmap_over_the_weights_equals_separate_calls(
    group, chunk_rows, monkeypatch
):
    torch = pytest.importorskip("torch")
    from pyvbmc.svbmc import SVBMC

    logging.getLogger("SVBMC").setLevel(logging.WARNING)
    stacked = SVBMC(_posteriors(group), seed=5)
    K_total = int(np.sum(stacked.K))
    _set_default_chunk_rows(monkeypatch, K_total, chunk_rows)
    entropy = _entropy_at_fixed_draws(stacked, 3)
    W = torch.as_tensor(
        np.random.default_rng(1).uniform(0.5, 2.0, (4, K_total))
    )

    H_batch = torch.func.vmap(entropy)(W)
    state_batch = stacked.rng.bit_generator.state

    assert H_batch.shape == (4,) and H_batch.dtype == torch.float64
    for b, w in enumerate(W):
        H = entropy(w)
        assert stacked.rng.bit_generator.state == state_batch
        # A weight gradient takes the whole matrix, with the same values.
        H_whole = entropy(w.clone().requires_grad_(True))
        assert H.item() == H_whole.item()
        # The batched reductions round differently from single ones.
        np.testing.assert_allclose(H_batch[b].item(), H.item(), **TOLERANCE)


@pytest.mark.parametrize("chunk_rows", FUNC_CHUNK_ROWS)
@pytest.mark.parametrize("group", FUNC_GROUPS)
def test_forward_mode_jacobian_equals_the_gradient(
    group, chunk_rows, monkeypatch
):
    torch = pytest.importorskip("torch")
    from pyvbmc.svbmc import SVBMC

    logging.getLogger("SVBMC").setLevel(logging.WARNING)
    stacked = SVBMC(_posteriors(group), seed=5)
    K_total = int(np.sum(stacked.K))
    _set_default_chunk_rows(monkeypatch, K_total, chunk_rows)
    entropy = _entropy_at_fixed_draws(stacked, 3)
    w_np = np.linspace(0.5, 2.0, K_total)

    jacobian = torch.func.jacfwd(entropy)(torch.as_tensor(w_np))
    w = torch.tensor(w_np, dtype=torch.float64, requires_grad=True)
    entropy(w).backward()

    assert jacobian.shape == (K_total,)
    np.testing.assert_allclose(jacobian.numpy(), w.grad.numpy(), **TOLERANCE)
