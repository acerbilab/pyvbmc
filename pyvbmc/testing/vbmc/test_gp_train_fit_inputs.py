"""Checks on what ``train_gp`` hands to the hyperparameter fit.

The GP hyperparameters are fitted by ``gpyreg.GP.fit``, which
:py:func:`pyvbmc.vbmc.gaussian_process_train.train_gp` feeds with starting
points, bounds, priors and sampler widths. The tests here pin those inputs
against the algorithm they come from (MATLAB VBMC, ``misc/gptrain_vbmc.m``
and its local ``vbmc_gphyp``).
"""

import copy

import numpy as np

import pyvbmc.vbmc.vbmc as vbmc_module
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


def test_ending_warmup_clears_the_covariance_the_fit_reads(monkeypatch):
    """The end of warm-up discards the running covariance of the GP
    hyperparameters, so that the widths of the hyperparameter sampler in
    the main algorithm are built from the fits of the main algorithm alone
    (MATLAB VBMC, ``vbmc.m:831``: ``hypstruct.runcov = []``). The
    covariance the fit reads is ``hyp_dict["run_cov"]``
    (``misc/get_GPTrainOptions.m:59``, ``gaussian_process_train.py``), so a
    reset under any other name leaves the warm-up covariance in place."""
    vbmc = build_short(
        {
            "max_iter": 4,
            "min_iter": 4,
            "max_fun_evals": 60,
            "weighted_hyp_cov": False,
            # Reach the end of warm-up at the first opportunity, and take
            # it for a real one rather than a false alarm.
            "stop_warmup_reliability": np.inf,
        }
    )
    monkeypatch.setattr(
        VBMC, "_check_warmup_end_conditions", lambda self: True
    )

    seen = []
    unwired = vbmc_module.train_gp

    def note_the_covariance(hyp_dict, optim_state, *args, **kwargs):
        seen.append(
            {
                "warmup": optim_state["warmup"],
                "run_cov": copy.deepcopy(hyp_dict.get("run_cov")),
            }
        )
        return unwired(hyp_dict, optim_state, *args, **kwargs)

    monkeypatch.setattr(vbmc_module, "train_gp", note_the_covariance)
    vbmc.optimize()

    assert not vbmc.optim_state["warmup"]
    # A warm-up fit builds a covariance, and the first fit of the main
    # algorithm does not inherit it.
    during_warmup = [call for call in seen if call["warmup"]]
    after_warmup = [call for call in seen if not call["warmup"]]
    assert len(during_warmup) >= 2 and len(after_warmup) >= 1
    assert during_warmup[-1]["run_cov"] is not None
    assert after_warmup[0]["run_cov"] is None
    # The summary statistics carry no key beyond the ones the fit manages.
    assert set(vbmc.hyp_dict) <= {"hyp", "warp", "logp", "full", "run_cov"}
