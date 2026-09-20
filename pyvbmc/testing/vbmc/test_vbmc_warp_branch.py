"""Checks on the input-warping branch of ``VBMC.optimize``.

The branch warps the inference space, refits the Gaussian process and the
variational posterior there, and keeps the warp only if the ELBO improves.
It is reached through the loop, so the checks below drive one short seeded
run whose options bring a warp forward: no warm-up, variational components
fixed to the training inputs, and a cheap sieve.
"""

import math

import numpy as np
import pytest

import pyvbmc.vbmc.vbmc as vbmc_module
from pyvbmc import VBMC

D = 2


@pytest.fixture(scope="module")
def warped_run():
    """One short run that warps, with every ``optimize_vp`` call recorded."""
    options = {
        "max_iter": 4,
        "min_iter": 1,
        "max_fun_evals": 100,
        "warmup": False,
        "variable_means": False,
        "ns_elbo": lambda K: 2 * K,
        # Pruning keeps the posterior's component count below the size of
        # the training set the refit fixes its components to.
        "tol_weight": 0.2,
        # Components fixed to the training inputs cannot be boosted to a
        # larger count afterwards.
        "do_final_boost": False,
        "display": "off",
        "plot": False,
        "print_iteration_header": False,
    }
    vbmc = VBMC(
        lambda x: -0.5 * np.sum(x**2),
        np.zeros((1, D)),
        np.full((1, D), -np.inf),
        np.full((1, D), np.inf),
        np.full((1, D), -1.0),
        np.full((1, D), 1.0),
        options=options,
        seed=20260920,
    )

    calls = []
    unwired = vbmc_module.optimize_vp

    def record_call(options, optim_state, vp, gp, n_fast, n_slow, K):
        calls.append(
            {
                # The warp leaves active sampling to be skipped, so the
                # refit it drives is the call that sees the flag set.
                "after_warp": bool(optim_state.get("skip_active_sampling")),
                "vp_K": vp.K,
                "K": K,
                "n_fast_opts": n_fast,
                "mu_aliases_gp_X": np.shares_memory(vp.mu, gp.X),
            }
        )
        return unwired(options, optim_state, vp, gp, n_fast, n_slow, K)

    vbmc_module.optimize_vp = record_call
    try:
        vbmc.optimize()
    finally:
        vbmc_module.optimize_vp = unwired
    return vbmc, calls


def test_the_run_warps(warped_run):
    """The options above do bring a warp about, so the checks that read the
    warp branch's calls are not vacuous."""
    vbmc, calls = warped_run
    actions = [
        action
        for iteration in vbmc.iteration_history["logging_action"]
        for action in iteration
    ]
    assert any("rotoscale" in action for action in actions)
    assert any(call["after_warp"] for call in calls)


def test_warp_refit_sieve_is_sized_at_the_new_component_count(warped_run):
    """The refit that follows a warp sieves as many candidate posteriors as
    the number of components it optimizes, not as many as the incoming
    posterior has (MATLAB VBMC, ``vbmc.m:584``:
    ``Nfastopts = ceil(evaloption_vbmc(options.NSelbo,Knew))``)."""
    vbmc, calls = warped_run
    refits = [call for call in calls if call["after_warp"]]
    assert refits

    for call in refits:
        # With the components fixed to the training inputs, the count the
        # refit optimizes is the size of the training set and differs from
        # the count of the posterior that comes in.
        assert call["K"] != call["vp_K"]
        assert call["n_fast_opts"] == math.ceil(
            vbmc.options.eval("ns_elbo", {"K": call["K"]})
        )


def test_warp_refit_gets_its_own_copy_of_the_training_inputs(warped_run):
    """Fixing the components of the posterior to the training inputs gives
    it a copy of them, as the same step of an ordinary iteration does, so
    that writing to the component means cannot reach the Gaussian process's
    training set (MATLAB VBMC, ``vbmc.m:576``: ``vp.mu = gp.X'``, an
    assignment by value)."""
    __, calls = warped_run
    refits = [call for call in calls if call["after_warp"]]
    assert refits
    assert not any(call["mu_aliases_gp_X"] for call in refits)
    # The same step of an ordinary iteration is the comparison.
    assert not any(call["mu_aliases_gp_X"] for call in calls)
