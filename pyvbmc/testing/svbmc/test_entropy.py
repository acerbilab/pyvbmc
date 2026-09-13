"""The vectorized entropy preparation against its literal per-component form.

``SVBMC._stacked_entropy`` maps the draws of a run's components to the
original space with one inverse transform, evaluates each run's forward
transform and Jacobian once for all draws, forms the log densities of a
run's components in one broadcast and reduces per component with array
operations. The references below are literal per-component transcriptions
of the same estimator (``scipy.stats.norm.logpdf`` and a masked loop in
torch); the two must agree on the density matrix, the entropy, its weight
gradient, the stratified variance and the state of the generator.
"""

import logging

import numpy as np
import pytest
import scipy.stats

from pyvbmc.svbmc._entropy import component_log_densities
from pyvbmc.testing.svbmc._fixtures import load_group

GROUPS = ["normal_D1", "bounded_D2", "corr_D3", "upstream_GMM"]
# One draw per component is the degenerate layout and, for warped runs, the
# case where the batched inverse transform rounds least like the reference.
DRAWS = [1, 3]
TOLERANCE = dict(rtol=1e-12, atol=1e-12)


def _posteriors(group):
    vps = load_group(group, rng=0)[0]
    if group == "upstream_GMM":
        # Three of the ten runs, fifty components each; ``_01`` is warped.
        vps = vps[:3]
    return vps


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
