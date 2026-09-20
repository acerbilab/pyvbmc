"""Checks on the state that ``VBMC.optimize`` keeps and records."""

import logging

import numpy as np

from pyvbmc import VBMC


def build_short(options: dict, seed: int = 20260920, D: int = 2):
    """A seeded spherical-Gaussian problem, ready to run."""
    settings = {
        "display": "off",
        "plot": False,
        "print_iteration_header": False,
    }
    settings.update(options)
    return VBMC(
        lambda x: -0.5 * np.sum(x**2),
        np.zeros((1, D)),
        np.full((1, D), -np.inf),
        np.full((1, D), np.inf),
        np.full((1, D), -1.0),
        np.full((1, D), 1.0),
        options=settings,
        seed=seed,
    )


def run_short(options: dict, seed: int = 20260920, D: int = 2):
    """Run a seeded spherical-Gaussian problem for a couple of iterations."""
    vbmc = build_short(options, seed=seed, D=D)
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


def test_a_closing_line_reports_a_posterior_from_an_earlier_iteration(caplog):
    """A run closes its display with a line for the posterior it returns
    whenever that posterior is not the one the last iteration ended on: it
    was selected from an earlier iteration, or the final boost changed it
    (MATLAB VBMC, ``vbmc.m:889`` and ``:908``:
    ``new_final_vp_flag = idx_best ~= iter``, set as well when the boost
    reports a change)."""
    vbmc = build_short(
        {
            "max_iter": 2,
            "min_iter": 2,
            "max_fun_evals": 60,
            "do_final_boost": False,
            "display": "iter",
        }
    )
    unwired = VBMC.determine_best_vp

    def select_the_first_iteration(self, **kwargs):
        kwargs["max_idx"] = 0
        return unwired(self, **kwargs)

    vbmc.determine_best_vp = select_the_first_iteration.__get__(vbmc, VBMC)

    with caplog.at_level(logging.INFO, logger="VBMC"):
        vbmc.optimize()

    assert vbmc.iteration > 0
    closing = [
        record.getMessage()
        for record in caplog.records
        if "finalize" in record.getMessage()
    ]
    assert len(closing) == 1


def test_the_closing_line_leaves_the_random_stream_of_an_unboosted_run():
    """The closing line is display only. For a run that ends without a
    boost, reporting a posterior selected from an earlier iteration leaves
    the run's generator where the loop left it, so that the run can be
    continued as an uninterrupted one would proceed."""
    options = {
        "max_iter": 2,
        "min_iter": 2,
        "max_fun_evals": 60,
        "do_final_boost": False,
    }
    unwired = VBMC.determine_best_vp

    def select_the_first_iteration(self, **kwargs):
        kwargs["max_idx"] = 0
        return unwired(self, **kwargs)

    def report_the_last_iteration(self, **kwargs):
        vp, elbo, elbo_sd, __ = unwired(self, **kwargs)
        return vp, elbo, elbo_sd, self.iteration

    states = []
    for selection in (select_the_first_iteration, report_the_last_iteration):
        vbmc = build_short(options)
        vbmc.determine_best_vp = selection.__get__(vbmc, VBMC)
        vbmc.optimize()
        assert vbmc.iteration > 0
        states.append(vbmc.rng.bit_generator.state)

    assert states[0] == states[1]


def test_the_closing_line_leaves_the_random_stream_of_a_boosted_run():
    """When the final boost changes the posterior, the closing line reports
    the divergence of the boosted posterior, estimated from samples. They
    do not come out of the run's generator, which stays where the boost
    left it."""
    vbmc = build_short({"max_iter": 2, "min_iter": 2, "max_fun_evals": 60})
    unwired = VBMC.final_boost
    after_the_boost = {}

    def boost_and_note_the_stream(self, vp, gp):
        result = unwired(self, vp, gp)
        after_the_boost["changed"] = result[3]
        after_the_boost["state"] = self.rng.bit_generator.state
        return result

    vbmc.final_boost = boost_and_note_the_stream.__get__(vbmc, VBMC)
    vbmc.optimize()

    assert after_the_boost["changed"]
    assert vbmc.rng.bit_generator.state == after_the_boost["state"]
