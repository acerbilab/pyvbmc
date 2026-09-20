"""Checks on the state that ``VBMC.optimize`` keeps and records."""

import numpy as np

from pyvbmc import VBMC


def run_short(options: dict, seed: int = 20260920, D: int = 2):
    """Run a seeded spherical-Gaussian problem for a couple of iterations."""
    settings = {
        "display": "off",
        "plot": False,
        "print_iteration_header": False,
    }
    settings.update(options)
    vbmc = VBMC(
        lambda x: -0.5 * np.sum(x**2),
        np.zeros((1, D)),
        np.full((1, D), -np.inf),
        np.full((1, D), np.inf),
        np.full((1, D), -1.0),
        np.full((1, D), 1.0),
        options=settings,
        seed=seed,
    )
    vbmc.optimize()
    return vbmc


def test_running_moments_are_an_exponentially_weighted_average():
    """After the first iteration the recorded moments of the variational
    posterior are a running average, not the moments of that iteration
    alone (MATLAB VBMC, ``vbmc.m:781-792``: the instantaneous moments are
    stored only while ``RunMean`` or ``RunCov`` is empty, and afterwards
    ``RunMean = wRun*RunMean + (1-wRun)*mubar`` with
    ``wRun = MomentsRunWeight^(N - LastRunAvg)``)."""
    vbmc = run_short({"max_iter": 2, "min_iter": 2, "max_fun_evals": 60})
    history = vbmc.iteration_history
    assert len(history["optim_state"]) >= 2

    first, second = history["optim_state"][0], history["optim_state"][1]
    weight = vbmc.options.get("moments_run_weight")

    # The first iteration has nothing to average with.
    mubar_first, sigma_first = history["vp"][0].moments(
        orig_flag=False, cov_flag=True
    )
    assert np.allclose(first["run_mean"], mubar_first.reshape(1, -1))
    assert np.allclose(first["run_cov"], sigma_first)
    assert first["last_run_avg"] == first["N"]

    # The second one averages the new moments into the stored ones, with a
    # weight that decays with the number of points added since.
    n_new = second["N"] - first["last_run_avg"]
    assert n_new > 0
    w_run = weight**n_new
    mubar, sigma = history["vp"][1].moments(orig_flag=False, cov_flag=True)
    assert np.allclose(
        second["run_mean"],
        w_run * first["run_mean"] + (1 - w_run) * mubar.reshape(1, -1),
    )
    assert np.allclose(
        second["run_cov"], w_run * first["run_cov"] + (1 - w_run) * sigma
    )
    assert second["last_run_avg"] == second["N"]
    # The average differs from the moments of the second iteration alone,
    # so the check above is not satisfied by storing those.
    assert not np.allclose(second["run_mean"], mubar.reshape(1, -1))
