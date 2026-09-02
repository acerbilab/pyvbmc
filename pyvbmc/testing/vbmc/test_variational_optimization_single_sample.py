"""Regression tests for the single-hyperparameter-sample GP path.

Once GP hyperparameter sampling stops (``N >= stable_gp_sampling``, by default
``200 + 10 D``), ``gp.posteriors`` holds one sample and ``_gp_log_joint`` runs
with ``Ns == 1``. It dropped the sample axis from ``G`` and ``dG`` but not from
``varG``, so ``_neg_elcbo`` returned a length-1 array for ``varF`` and
``_eval_full_elcbo`` raised ``ValueError: setting an array element with a
sequence`` when storing it (NumPy 2; NumPy 1 squeezed silently). Found on
2026-09-02 by the budget-exhausting run of the benchmark suite
(``dev/plans/benchmark-suite-and-golden-traces.md``): every run that reached
the optimize-only GP regime crashed.

The GP fixture is the one shared with the finite-difference tests, reduced to
its first hyperparameter sample.
"""

import numpy as np

from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.vbmc.variational_optimization import (
    _eval_full_elcbo,
    _gp_log_joint,
    _initialize_full_elcbo,
    _neg_elcbo,
)

from .test_variational_optimization import setup_options
from .test_variational_optimization_grad_fd import (
    D,
    K,
    _fixture_gp,
    _load,
    _raw_theta0,
    _vp_from_raw_theta,
)


def _single_sample_gp():
    gp, _ = _fixture_gp()
    gp.update(hyp=_load("hyp.txt")[0:1])
    assert len(gp.posteriors) == 1
    return gp


def test_gp_log_joint_single_sample_variance_is_scalar():
    # Variance without gradients, as ``_eval_full_elcbo`` requests it (the
    # variance-gradient path is an unfinished stub that raises).
    gp = _single_sample_gp()
    vp = _vp_from_raw_theta(_raw_theta0(seed=3))
    G, dG, varG, dvarG, var_ss = _gp_log_joint(vp, gp, False, compute_var=True)
    assert np.ndim(G) == 0 and np.isfinite(G)
    assert dG is None and dvarG is None
    assert np.ndim(varG) == 0 and varG > 0
    assert var_ss == 0
    # and the gradient path still drops the sample axis
    _, dG, *_ = _gp_log_joint(vp, gp, True)
    assert dG.shape == (D * K + K + D + K,)


def test_neg_elcbo_single_sample_var_f_is_scalar():
    gp = _single_sample_gp()
    theta = _raw_theta0(seed=3)
    F, dF, G, H, varF = _neg_elcbo(
        theta.copy(),
        gp,
        VariationalPosterior(D, K, rng=1),
        0.0,
        0,
        False,
        True,
    )
    assert np.ndim(F) == 0 and np.isfinite(F)
    assert np.ndim(varF) == 0 and varF > 0
    assert dF is None


def test_eval_full_elcbo_single_sample_stores_var_f():
    """The exact failure site: storing ``varF`` into the stats arrays."""
    gp = _single_sample_gp()
    theta = _raw_theta0(seed=3)
    vp = VariationalPosterior(D, K, rng=1)
    options = setup_options(D)
    elbo_stats = _initialize_full_elcbo(2, theta.size, K, 1)
    elbo_stats = _eval_full_elcbo(0, theta, vp, gp, elbo_stats, 3.0, options)
    assert np.isfinite(elbo_stats["varF"][0]) and elbo_stats["varF"][0] > 0
    assert np.isfinite(elbo_stats["nelcbo"][0])
    assert elbo_stats["nelcbo"][0] > elbo_stats["nelbo"][0]
