import gpyreg as gpr
import numpy as np
import pytest
from scipy.stats import norm

import pyvbmc.vbmc.gaussian_process_train as gp_train_module
from pyvbmc import VBMC
from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.vbmc.gaussian_process_train import (
    _cov_identifier_to_covariance_function,
    _estimate_noise,
    _get_gp_training_options,
    _get_hyp_cov,
    _get_training_data,
    _meanfun_name_to_mean_function,
    train_gp,
)


def test_estimate_noise():
    # Back-up random number generator so as not to affect other tests
    # that might want to use different random numbers each run.
    state = np.random.get_state()
    np.random.seed(1234)

    N = 31
    D = 1
    X = -5 + np.random.rand(N, 1) * 10
    s2 = 0.05 * np.exp(0.5 * X)
    y = np.sin(X) + np.sqrt(s2) * norm.ppf(np.random.random_sample(X.shape))
    y[y < 0] = -np.abs(3 * y[y < 0]) ** 2

    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.Matern(degree=3),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(
            constant_add=True, user_provided_add=True
        ),
    )

    hyp = np.array([[-2.5, 1.7, -7.5, 0.3, 2.6, 1.2]])
    gp.update(X_new=X, y_new=y, s2_new=s2, hyp=hyp)

    noise_estimate = _estimate_noise(gp)

    np.random.set_state(state)

    # Value taken from MATLAB which only applies for this exact setup.
    # Change any part of the test and it will not apply.
    assert np.isclose(noise_estimate, 0.106582207806606)


def test_get_training_data_no_noise():
    D = 3
    f = lambda x: np.sum(x + 2, axis=1)
    x0 = np.ones((2, D)) * 3
    plb = np.ones((1, D)) * -1
    pub = np.ones((1, D)) * 1

    vbmc = VBMC(f, x0, None, None, plb, pub)

    # Make sure we get nothing out before data has not been added.
    X_train, y_train, s2_train, t_train = _get_training_data(
        vbmc.function_logger
    )

    assert X_train.shape == (0, 3)
    assert y_train.shape == (0, 1)
    assert s2_train is None
    assert t_train.shape == (0, 1)

    # Create dummy data.
    sample_count = 10
    window = vbmc.optim_state["pub_tran"] - vbmc.optim_state["plb_tran"]
    rnd_tmp = np.random.rand(sample_count, window.shape[1])
    Xs = window * rnd_tmp + vbmc.optim_state["plb_tran"]
    ys = f(Xs)

    # Add dummy training data explicitly since function_logger
    # has a parameter transformer which makes everything hard.
    for sample_idx in range(sample_count):
        vbmc.function_logger.X_flag[sample_idx] = True
        vbmc.function_logger.X[sample_idx] = Xs[sample_idx]
        vbmc.function_logger.y[sample_idx] = ys[sample_idx]
        vbmc.function_logger.fun_eval_time[sample_idx] = 1e-5

    # Then make sure we get that data back.
    X_train, y_train, s2_train, t_train = _get_training_data(
        vbmc.function_logger
    )

    assert np.all(X_train == Xs)
    assert np.all(y_train.flatten() == ys)
    assert s2_train is None
    assert np.all(t_train == 1e-5)


def test_get_training_data_noise():
    D = 3
    f = lambda x: np.sum(x + 2, axis=1)
    x0 = np.ones((2, D)) * 3
    plb = np.ones((1, D)) * -1
    pub = np.ones((1, D)) * 1
    options = {"specify_target_noise": True}

    vbmc = VBMC(f, x0, None, None, plb, pub, options)

    # Make sure we get nothing out before data has not been added.
    X_train, y_train, s2_train, t_train = _get_training_data(
        vbmc.function_logger
    )

    assert X_train.shape == (0, 3)
    assert y_train.shape == (0, 1)
    assert s2_train.shape == (0, 1)
    assert t_train.shape == (0, 1)

    # Create dummy data.
    sample_count = 10
    window = vbmc.optim_state["pub_tran"] - vbmc.optim_state["plb_tran"]
    rnd_tmp = np.random.rand(sample_count, window.shape[1])
    Xs = window * rnd_tmp + vbmc.optim_state["plb_tran"]
    ys = f(Xs)

    # Add dummy training data explicitly since function_logger
    # has a parameter transformer which makes everything hard.
    for sample_idx in range(sample_count):
        vbmc.function_logger.X_flag[sample_idx] = True
        vbmc.function_logger.X[sample_idx] = Xs[sample_idx]
        vbmc.function_logger.y[sample_idx] = ys[sample_idx]
        vbmc.function_logger.S[sample_idx] = 1
        vbmc.function_logger.fun_eval_time[sample_idx] = 1e-5

    # Then make sure we get that data back.
    X_train, y_train, s2_train, t_train = _get_training_data(
        vbmc.function_logger
    )

    assert np.all(X_train == Xs)
    assert np.all(y_train.flatten() == ys)
    assert np.all(s2_train == 1)
    assert np.all(t_train == 1e-5)


def test_meanfun_name_to_mean_function():
    m1 = _meanfun_name_to_mean_function("zero")
    m2 = _meanfun_name_to_mean_function("const")
    m3 = _meanfun_name_to_mean_function("negquad")

    assert isinstance(m1, gpr.mean_functions.ZeroMean)
    assert isinstance(m2, gpr.mean_functions.ConstantMean)
    assert isinstance(m3, gpr.mean_functions.NegativeQuadratic)

    with pytest.raises(ValueError):
        m4 = _meanfun_name_to_mean_function("linear")
    with pytest.raises(ValueError):
        m5 = _meanfun_name_to_mean_function("quad")
    with pytest.raises(ValueError):
        m6 = _meanfun_name_to_mean_function("posquad")
    with pytest.raises(ValueError):
        m7 = _meanfun_name_to_mean_function("se")
    with pytest.raises(ValueError):
        m8 = _meanfun_name_to_mean_function("negse")
    with pytest.raises(ValueError):
        m9 = _meanfun_name_to_mean_function("linear")


def test_cov_identifier_to_covariance_function():
    c1 = _cov_identifier_to_covariance_function(1)
    c2 = _cov_identifier_to_covariance_function(3)
    c3 = _cov_identifier_to_covariance_function([3, 1])
    c4 = _cov_identifier_to_covariance_function([3, 3])
    c5 = _cov_identifier_to_covariance_function([3, 5])

    assert isinstance(c1, gpr.covariance_functions.SquaredExponential)
    assert isinstance(c2, gpr.covariance_functions.Matern)
    assert isinstance(c3, gpr.covariance_functions.Matern)
    assert isinstance(c4, gpr.covariance_functions.Matern)
    assert isinstance(c5, gpr.covariance_functions.Matern)

    assert c2.degree == 5
    assert c3.degree == 1
    assert c4.degree == 3
    assert c5.degree == 5

    with pytest.raises(ValueError):
        c6 = _cov_identifier_to_covariance_function(0)
    with pytest.raises(ValueError):
        c7 = _cov_identifier_to_covariance_function(2)


def _weighted_hyp_cov_inputs():
    """Asymmetric ragged history with one incompatible middle block."""
    history = {
        "gp_hyp_full": [
            np.array([[0.0, 0.0]]),
            np.array([[1.0, 3.0], [4.0, 2.0], [2.0, 5.0]]),
            np.array([[7.0, 8.0, 9.0]]),
            np.array([[10.0, 1.0], [12.0, 4.0]]),
        ],
        # For iter=4 the decay reads entries 3, 2, 1 in that order.
        # With tol_skl * fun_evals_per_iter = 1, these give decay
        # multipliers 2, 1 and 1.5.
        "sKL": np.array([1.0, np.exp(1.5), np.exp(0.2), np.exp(2), 1.0]),
        "r_index": np.ones(4),
    }
    options = {
        "weighted_hyp_cov": True,
        "hyp_run_weight": 0.5,
        "fun_evals_per_iter": 2,
        "tol_skl": 0.5,
        "tol_cov_weight": 0.0,
    }
    return {"iter": 4}, history, options, {"hyp": np.zeros(2)}


def test_get_hyp_cov():
    optim_state, history, options, hyp_dict = _weighted_hyp_cov_inputs()
    expected = np.array(
        [
            [4.580701754385967, 2.717885869909433],
            [2.717885869909433, 4.36639432516309],
        ]
    )

    result = _get_hyp_cov(optim_state, history, options, hyp_dict)

    np.testing.assert_allclose(result, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(result, result.T, rtol=0, atol=1e-15)
    assert result.shape == (2, 2)
    sampled_hyp_dict = {"hyp": np.zeros((3, 2))}
    np.testing.assert_allclose(
        _get_hyp_cov(optim_state, history, options, sampled_hyp_dict),
        expected,
        rtol=1e-14,
        atol=1e-14,
    )

    # The current GP model count is authoritative when the summary holds a
    # stale hyperparameter vector from a differently shaped model.
    stale_hyp_dict = {"hyp": np.zeros(3)}
    result_from_model_count = _get_hyp_cov(
        optim_state, history, options, stale_hyp_dict, hyp_n=2
    )
    np.testing.assert_allclose(
        result_from_model_count, expected, rtol=1e-14, atol=1e-14
    )
    assert _get_hyp_cov(optim_state, history, options, stale_hyp_dict) is None


def test_get_hyp_cov_cutoff_and_degenerate_history():
    optim_state, history, options, hyp_dict = _weighted_hyp_cov_inputs()
    options["tol_cov_weight"] = 0.02
    expected_newest = np.array([[2.0, 3.0], [3.0, 4.5]])
    result = _get_hyp_cov(optim_state, history, options, hyp_dict)
    np.testing.assert_allclose(result, expected_newest, rtol=0, atol=0)

    one_sample_history = {
        "gp_hyp_full": [np.array([[1.0, 2.0]])],
        "sKL": np.ones(2),
    }
    assert (
        _get_hyp_cov(
            {"iter": 1}, one_sample_history, options, hyp_dict, hyp_n=2
        )
        is None
    )
    assert (
        _get_hyp_cov(
            {"iter": 1}, one_sample_history, options, hyp_dict, hyp_n=3
        )
        is None
    )

    newer_incompatible = {
        "gp_hyp_full": [
            np.array([[1.0, 2.0], [3.0, 5.0]]),
            np.array([[7.0, 8.0, 9.0]]),
        ],
        "sKL": np.ones(3),
    }
    compatible_result = _get_hyp_cov(
        {"iter": 2}, newer_incompatible, options, hyp_dict, hyp_n=2
    )
    np.testing.assert_allclose(
        compatible_result, np.array([[2.0, 3.0], [3.0, 4.5]])
    )

    empty_history = {"gp_hyp_full": [], "sKL": np.ones(2)}
    assert (
        _get_hyp_cov({"iter": 1}, empty_history, options, hyp_dict, hyp_n=2)
        is None
    )
    malformed_hyp_dict = {"hyp": np.zeros((2, 2, 1))}
    assert (
        _get_hyp_cov(
            {"iter": 1}, one_sample_history, options, malformed_hyp_dict
        )
        is None
    )

    options["weighted_hyp_cov"] = False
    hyp_dict["run_cov"] = 42
    assert _get_hyp_cov({"iter": 1}, history, options, hyp_dict) == 42
    assert _get_hyp_cov({"iter": 0}, history, options, hyp_dict) is None


def test_weighted_hyp_cov_delivers_sampler_widths():
    D = 2
    f = lambda x: np.sum(x + 2)
    vbmc = VBMC(
        f,
        np.full((2, D), 3.0),
        np.full((1, D), 1.0),
        np.full((1, D), 5.0),
        np.full((1, D), 2.0),
        np.full((1, D), 4.0),
    )
    optim_state, history, settings, hyp_dict = _weighted_hyp_cov_inputs()
    vbmc.optim_state["iter"] = optim_state["iter"]
    vbmc.optim_state["n_eff"] = 10
    for key, value in settings.items():
        vbmc.options.__setitem__(key, value, force=True)

    gp_train = _get_gp_training_options(
        vbmc.optim_state,
        history,
        vbmc.options,
        hyp_dict,
        gp_s_N=3,
        hyp_n=2,
    )

    expected_cov = np.array(
        [
            [4.580701754385967, 2.717885869909433],
            [2.717885869909433, 4.36639432516309],
        ]
    )
    expected_widths = np.sqrt(np.diag(expected_cov)) * 5
    np.testing.assert_allclose(
        gp_train["widths"], expected_widths, rtol=1e-14, atol=1e-14
    )


def test_train_gp_passes_current_hyp_count(monkeypatch):
    D = 2
    vbmc = VBMC(
        lambda x: np.sum(x, axis=1),
        np.zeros((1, D)),
        None,
        None,
        -np.ones((1, D)),
        np.ones((1, D)),
    )
    x_train = np.array([[-1.0, -0.5], [0.0, 0.5], [1.0, 0.25]])
    y_train = np.sum(x_train, axis=1, keepdims=True)
    monkeypatch.setattr(
        gp_train_module,
        "_get_training_data",
        lambda logger: (x_train, y_train, None, np.zeros((3, 1))),
    )

    captured = {}

    def fake_training_options(
        optim_state,
        iteration_history,
        options,
        hyp_dict,
        gp_s_N,
        hyp_n=None,
    ):
        captured["hyp_n"] = hyp_n
        return {"widths": None, "init_N": 0, "sampler": "slicesample"}

    def fake_fit(gp, x, y, s2, hyp0=None, options=None, rng=None):
        return np.zeros(np.size(gp.hyper_priors["mu"])), None, None

    monkeypatch.setattr(
        gp_train_module, "_get_gp_training_options", fake_training_options
    )
    monkeypatch.setattr(gpr.GP, "fit", fake_fit)
    monkeypatch.setattr(gp_train_module, "_estimate_noise", lambda gp: 0.0)
    vbmc.optim_state["N"] = 3
    vbmc.optim_state["n_eff"] = 3
    stale_hyp_dict = {"hyp": np.zeros(99)}

    gp, _, _, _ = train_gp(
        stale_hyp_dict,
        vbmc.optim_state,
        vbmc.function_logger,
        vbmc.iteration_history,
        vbmc.options,
        vbmc.plausible_lower_bounds,
        vbmc.plausible_upper_bounds,
        rng=np.random.default_rng(1),
    )

    assert captured["hyp_n"] == np.size(gp.hyper_priors["mu"])
    assert captured["hyp_n"] != 99


def test_get_gp_training_options_samplers():
    D = 3
    lb = np.ones((1, D)) * 1
    ub = np.ones((1, D)) * 5
    x0 = np.ones((2, D)) * 3
    plb = np.ones((1, D)) * 2
    pub = np.ones((1, D)) * 4
    f = lambda x: np.sum(x + 2)
    options = {"weighted_hyp_cov": False}
    vbmc = VBMC(f, x0, lb, ub, plb, pub, options)

    hyp_dict = {"run_cov": np.eye(3)}
    hyp_dict_none = {"run_cov": None}
    vbmc.optim_state["n_eff"] = 10
    vbmc.optim_state["iter"] = 1
    vbmc.iteration_history.record("r_index", 5, 0)

    res1 = _get_gp_training_options(
        vbmc.optim_state, vbmc.iteration_history, vbmc.options, hyp_dict, 8
    )
    assert res1["sampler"] == "slicesample"

    vbmc.options.__setitem__("gp_hyp_sampler", "npv", force=True)
    res2 = _get_gp_training_options(
        vbmc.optim_state, vbmc.iteration_history, vbmc.options, hyp_dict, 8
    )
    assert res2["sampler"] == "npv"

    vbmc.options.__setitem__("gp_hyp_sampler", "mala", force=True)
    vbmc.optim_state["gp_mala_step_size"] = 10
    res3 = _get_gp_training_options(
        vbmc.optim_state, vbmc.iteration_history, vbmc.options, hyp_dict, 8
    )
    assert res3["sampler"] == "mala"
    assert res3["step_size"] == 10

    vbmc.options.__setitem__("gp_hyp_sampler", "slicelite", force=True)
    res4 = _get_gp_training_options(
        vbmc.optim_state, vbmc.iteration_history, vbmc.options, hyp_dict, 8
    )
    assert res4["sampler"] == "slicelite"

    vbmc.options.__setitem__("gp_hyp_sampler", "splitsample", force=True)
    res5 = _get_gp_training_options(
        vbmc.optim_state, vbmc.iteration_history, vbmc.options, hyp_dict, 8
    )
    assert res5["sampler"] == "splitsample"

    vbmc.options.__setitem__("gp_hyp_sampler", "covsample", force=True)
    res6 = _get_gp_training_options(
        vbmc.optim_state, vbmc.iteration_history, vbmc.options, hyp_dict, 8
    )
    assert res6["sampler"] == "covsample"

    # Test too large r_index for covsample
    vbmc.iteration_history.record("r_index", 50, 0)
    res7 = _get_gp_training_options(
        vbmc.optim_state, vbmc.iteration_history, vbmc.options, hyp_dict, 8
    )
    assert res7["sampler"] == "slicesample"

    res8 = _get_gp_training_options(
        vbmc.optim_state,
        vbmc.iteration_history,
        vbmc.options,
        hyp_dict_none,
        8,
    )
    assert res8["sampler"] == "covsample"

    # Test too small n_eff laplace sampler
    vbmc.options.__setitem__("gp_hyp_sampler", "laplace", force=True)
    res9 = _get_gp_training_options(
        vbmc.optim_state, vbmc.iteration_history, vbmc.options, hyp_dict, 8
    )
    assert res9["sampler"] == "slicesample"

    # Test enough n_eff laplace sampler
    vbmc.optim_state["n_eff"] = 50
    vbmc.options.__setitem__("gp_hyp_sampler", "laplace", force=True)
    res10 = _get_gp_training_options(
        vbmc.optim_state, vbmc.iteration_history, vbmc.options, hyp_dict, 8
    )
    assert res10["sampler"] == "laplace"

    # Test sampler that does not exist.
    vbmc.options.__setitem__("gp_hyp_sampler", "does_not_exist", force=True)
    with pytest.raises(ValueError):
        res11 = _get_gp_training_options(
            vbmc.optim_state, vbmc.iteration_history, vbmc.options, hyp_dict, 8
        )


def test_get_gp_training_options_opts_N():
    D = 3
    lb = np.ones((1, D)) * 1
    ub = np.ones((1, D)) * 5
    x0 = np.ones((2, D)) * 3
    plb = np.ones((1, D)) * 2
    pub = np.ones((1, D)) * 4
    f = lambda x: np.sum(x + 2)
    vbmc = VBMC(f, x0, lb, ub, plb, pub)

    vbmc.optim_state["n_eff"] = 10
    vbmc.optim_state["iter"] = 2
    vbmc.iteration_history.record("r_index", 5, 1)
    vbmc.options.__setitem__("weighted_hyp_cov", False, force=True)
    hyp_dict = {"run_cov": np.eye(3)}
    hyp_dict_none = {"run_cov": None}
    vbmc.options.__setitem__("gp_retrain_threshold", 10, force=True)

    res1 = _get_gp_training_options(
        vbmc.optim_state, vbmc.iteration_history, vbmc.options, hyp_dict, 0
    )
    assert res1["opts_N"] == 2

    vbmc.optim_state["recompute_var_post"] = False
    vbmc.options.__setitem__("gp_hyp_sampler", "slicelite", force=True)
    res2 = _get_gp_training_options(
        vbmc.optim_state, vbmc.iteration_history, vbmc.options, hyp_dict, 0
    )
    assert res2["opts_N"] == 1

    res3 = _get_gp_training_options(
        vbmc.optim_state, vbmc.iteration_history, vbmc.options, hyp_dict, 8
    )
    assert res3["opts_N"] == 0

    vbmc.options.__setitem__("gp_retrain_threshold", 1, force=True)
    res4 = _get_gp_training_options(
        vbmc.optim_state, vbmc.iteration_history, vbmc.options, hyp_dict, 0
    )
    assert res4["opts_N"] == 2


def test_gp_hyp():
    D = 3
    f = lambda x: np.sum(x + 2, axis=1)
    x0 = np.ones((2, D)) * 3
    plb = np.ones((1, D)) * -1
    pub = np.ones((1, D)) * 1

    options = {"specify_target_noise": True}
    vbmc = VBMC(f, x0, None, None, plb, pub, options)

    # Create dummy data.
    sample_count = 10
    window = vbmc.optim_state["pub_tran"] - vbmc.optim_state["plb_tran"]
    rnd_tmp = np.random.rand(sample_count, window.shape[1])
    Xs = window * rnd_tmp + vbmc.optim_state["plb_tran"]
    ys = f(Xs)

    # Add dummy training data explicitly since function_logger
    # has a parameter transformer which makes everything hard.
    for sample_idx in range(sample_count):
        vbmc.function_logger.X_flag[sample_idx] = True
        vbmc.function_logger.X[sample_idx] = Xs[sample_idx]
        vbmc.function_logger.y[sample_idx] = ys[sample_idx]
        vbmc.function_logger.S[sample_idx] = 1
        vbmc.function_logger.fun_eval_time[sample_idx] = 1e-5

    vbmc.optim_state["N"] = 10
    vbmc.optim_state["n_eff"] = np.sum(
        vbmc.function_logger.n_evals[vbmc.function_logger.X_flag]
    )
    assert not np.isnan(vbmc.optim_state["n_eff"])

    gp, Ns_gp, _, _ = train_gp(
        {},
        vbmc.optim_state,
        vbmc.function_logger,
        vbmc.iteration_history,
        vbmc.options,
        vbmc.plausible_lower_bounds,
        vbmc.plausible_upper_bounds,
    )
    priors = gp.get_priors()
    assert priors["noise_log_scale"][1][0] == np.log(
        vbmc.options["tol_gp_noise"]
    )
    assert priors["noise_log_scale"][1][1] == 0.5
