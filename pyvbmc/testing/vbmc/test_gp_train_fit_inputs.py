"""Checks on what ``train_gp`` hands to the hyperparameter fit.

The GP hyperparameters are fitted by ``gpyreg.GP.fit``, which
:py:func:`pyvbmc.vbmc.gaussian_process_train.train_gp` feeds with starting
points, bounds, priors and sampler widths. The tests here pin those inputs
against the algorithm they come from (MATLAB VBMC, ``misc/gptrain_vbmc.m``
and its local ``vbmc_gphyp``).
"""

import copy

import gpyreg as gpr
import numpy as np

import pyvbmc.vbmc.vbmc as vbmc_module
from pyvbmc import VBMC
from pyvbmc.vbmc.gaussian_process_train import _gp_hyp, train_gp


def build_trained_state(
    options: dict = None,
    D: int = 2,
    sample_count: int = 12,
    seed: int = 20260920,
):
    """A VBMC instance whose function logger holds a small training set.

    The points are written into the logger directly, as
    ``test_gaussian_process_train.test_gp_hyp`` does, so that no target
    evaluation and no parameter transform stand between the test and the
    training set ``train_gp`` will read.
    """
    settings = {
        "display": "off",
        "plot": False,
        "print_iteration_header": False,
    }
    settings.update(options or {})
    vbmc = VBMC(
        lambda x: -0.5 * np.sum(x**2),
        np.zeros((1, D)),
        None,
        None,
        np.full((1, D), -1.0),
        np.full((1, D), 1.0),
        options=settings,
        seed=seed,
    )
    logger = vbmc.function_logger
    rng = np.random.default_rng(seed)
    window = vbmc.optim_state["pub_tran"] - vbmc.optim_state["plb_tran"]
    X = vbmc.optim_state["plb_tran"] + window * rng.random((sample_count, D))
    y = -0.5 * np.sum(X**2, axis=1)
    for i in range(sample_count):
        logger.X_flag[i] = True
        logger.X[i] = X[i]
        logger.y[i] = y[i]
        logger.fun_eval_time[i] = 1e-5
        if logger.noise_flag:
            # A point observed once at the level's default accuracy.
            logger.S[i] = 1.0
            logger.n_evals[i] = 1
    vbmc.optim_state["N"] = sample_count
    vbmc.optim_state["n_eff"] = sample_count
    return vbmc


def training_data(vbmc):
    """The training inputs and targets the function logger holds."""
    logger = vbmc.function_logger
    return logger.X[logger.X_flag, :], logger.y[logger.X_flag]


def default_gp(vbmc):
    """The GP model ``train_gp`` builds for a noiseless target."""
    return gpr.GP(
        D=vbmc.D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )


def hyperparameter_bounds(vbmc, gp):
    """Install the bounds and priors of ``vbmc`` on ``gp``.

    Returns the GP, the bounds it carries and the ones ``gpyreg.GP.fit``
    will work with, which fills every entry left unset.
    """
    X, y = training_data(vbmc)
    gp, _, _ = _gp_hyp(
        vbmc.optim_state,
        vbmc.options,
        vbmc.optim_state["plb_tran"],
        vbmc.optim_state["pub_tran"],
        gp,
        X,
        y,
    )
    gp.X, gp.y = X, y
    return (
        gp,
        gp.get_bounds(),
        gp.get_recommended_bounds(gp.lower_bounds, gp.upper_bounds),
    )


def fit_the_gp(vbmc, hyp_dict, seed: int = 1):
    """Train the GP of ``vbmc`` on what its function logger holds."""
    return train_gp(
        hyp_dict,
        vbmc.optim_state,
        vbmc.function_logger,
        vbmc.iteration_history,
        vbmc.options,
        vbmc.optim_state["plb_tran"],
        vbmc.optim_state["pub_tran"],
        rng=np.random.default_rng(seed),
    )


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


def test_a_fit_without_sampling_holds_the_optimized_hyperparameters():
    """A fit that draws no samples leaves the optimized hyperparameters in
    ``hyp_dict["full"]`` and no running covariance. MATLAB VBMC assigns
    ``hypstruct.full = gpoutput.hyp_prethin`` after every fit
    (``misc/gptrain_vbmc.m:65``), and with no samples ``gplite_train.m:465``
    makes that the single optimized vector, so the test on the number of
    samples at ``:83`` fails and ``hypstruct.runcov = []`` follows."""
    vbmc = build_trained_state()
    hyp_dict = {}

    gp, gp_s_N, _, hyp_dict = fit_the_gp(vbmc, hyp_dict)
    assert gp_s_N > 1
    assert hyp_dict["full"].shape[0] == gp_s_N * vbmc.options["gp_sample_thin"]
    assert hyp_dict["run_cov"] is not None

    # Stop sampling, as a run does once it has enough training points.
    vbmc.optim_state["stop_sampling"] = vbmc.optim_state["N"]
    gp, gp_s_N, _, hyp_dict = fit_the_gp(vbmc, hyp_dict)

    assert gp_s_N == 0
    fitted = gp.get_hyperparameters(as_array=True)
    assert hyp_dict["full"].shape == (1, fitted.shape[1])
    np.testing.assert_array_equal(hyp_dict["full"], fitted)
    # The covariance of a single vector is not defined, and the chain of
    # the earlier fit is not folded into it once more.
    assert hyp_dict["run_cov"] is None
    assert hyp_dict["logp"] is None


def test_output_dependent_noise_is_bounded_by_the_training_values():
    """The threshold of the rectified output-dependent noise is bounded by
    the training targets: above by ``max(y) - 10*D`` and below by
    ``min(min(y), max(y) - 20*D)`` (MATLAB VBMC,
    ``misc/gptrain_vbmc.m:235-236``). No route through ``VBMC`` switches
    this noise feature on, so the bounds are built here by hand."""
    vbmc = build_trained_state()
    X, y = training_data(vbmc)
    D = X.shape[1]
    vbmc.optim_state["gp_noise_fun"] = [1, 0, 1]
    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(
            constant_add=True, rectified_linear_output_dependent_add=True
        ),
    )

    _, bounds, _ = hyperparameter_bounds(vbmc, gp)

    lower, upper = bounds["noise_rectified_log_multiplier"]
    assert lower[0] == min(np.min(y), np.max(y) - 20 * D)
    assert upper[0] == np.max(y) - 10 * D


def test_the_longest_length_scale_the_option_allows_reaches_the_fit():
    """``upper_gp_length_factor`` caps the GP input length scales at
    ``log(factor * (PUB - PLB))`` (MATLAB VBMC,
    ``misc/gptrain_vbmc.m:177``, which assigns ``UB_gp(1:D)`` and leaves
    the lower entries to their own statement). At the default 0 no cap is
    asked for and the entry is left for gpyreg to fill from the training
    set."""
    capped = build_trained_state({"upper_gp_length_factor": 3})
    _, bounds, filled = hyperparameter_bounds(capped, default_gp(capped))
    asked = np.log(
        3 * (capped.optim_state["pub_tran"] - capped.optim_state["plb_tran"])
    ).ravel()
    np.testing.assert_array_equal(
        bounds["covariance_log_lengthscale"][1], asked
    )
    # gpyreg fills the entries left unset and keeps the ones installed, so
    # the cap is what the fit works with.
    np.testing.assert_array_equal(
        filled["covariance_log_lengthscale"][1], asked
    )

    uncapped = build_trained_state()
    _, default_bounds, default_filled = hyperparameter_bounds(
        uncapped, default_gp(uncapped)
    )
    # The option touches neither the lower bounds nor any other entry.
    np.testing.assert_array_equal(
        bounds["covariance_log_lengthscale"][0],
        default_bounds["covariance_log_lengthscale"][0],
    )
    assert np.all(np.isnan(default_bounds["covariance_log_lengthscale"][1]))
    # Without the option the fit works with gpyreg's recommendation, which
    # is not the cap the option asks for.
    assert np.all(default_filled["covariance_log_lengthscale"][1] != asked)
