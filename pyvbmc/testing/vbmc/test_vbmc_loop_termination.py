import numpy as np

from pyvbmc import VBMC

fun = lambda x: np.sum(x + 2)


def create_vbmc(
    D: int,
    x0: float,
    lower_bounds: float,
    upper_bounds: float,
    plausible_lower_bounds: float,
    plausible_upper_bounds: float,
    options: dict = None,
):
    lb = np.ones((1, D)) * lower_bounds
    ub = np.ones((1, D)) * upper_bounds
    x0_array = np.ones((2, D)) * x0
    plb = np.ones((1, D)) * plausible_lower_bounds
    pub = np.ones((1, D)) * plausible_upper_bounds
    return VBMC(fun, x0_array, lb, ub, plb, pub, options)


def test_vbmc_check_termination_conditions_max_fun_evals(mocker):
    options = {
        "max_fun_evals": 10,
        "min_fun_evals": 5,
        "min_iter": 5,
        "max_iter": 100,
        "tol_stable_count": 60,
        "fun_evals_per_iter": 5,
    }
    mocker.patch(
        "pyvbmc.vbmc.VBMC._compute_reliability_index",
        return_value=(np.Inf, np.NaN),
    )
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    vbmc.function_logger.func_count = 10
    vbmc.optim_state["iter"] = 10
    terminated, _, _ = vbmc._check_termination_conditions()
    assert terminated
    vbmc.function_logger.func_count = 9
    terminated, _, _ = vbmc._check_termination_conditions()
    assert not terminated


def test_vbmc_check_termination_conditions_max_iter(mocker):
    mocker.patch(
        "pyvbmc.vbmc.VBMC._compute_reliability_index",
        return_value=(np.Inf, np.NaN),
    )
    options = {
        "max_fun_evals": 10,
        "min_fun_evals": 5,
        "min_iter": 5,
        "max_iter": 100,
    }
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    vbmc.function_logger.func_count = 9
    vbmc.optim_state["entropy_switch"] = True
    vbmc.optim_state["iter"] = 99
    terminated, _, _ = vbmc._check_termination_conditions()
    assert terminated
    vbmc.optim_state["iter"] = 98
    terminated, _, _ = vbmc._check_termination_conditions()
    assert not terminated


def test_vbmc_check_termination_conditions_prevent_early_termination(mocker):
    options = {
        "max_fun_evals": 10,
        "min_fun_evals": 5,
        "min_iter": 101,
        "max_iter": 100,
    }
    mocker.patch(
        "pyvbmc.vbmc.VBMC._compute_reliability_index",
        return_value=(np.Inf, np.NaN),
    )
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    vbmc.function_logger.func_count = 9
    vbmc.optim_state["iter"] = 100
    vbmc.optim_state["entropy_switch"] = True
    terminated, _, _ = vbmc._check_termination_conditions()
    assert not terminated

    options = {
        "max_fun_evals": 10,
        "min_fun_evals": 11,
        "min_iter": 5,
        "max_iter": 100,
    }
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    vbmc.function_logger.func_count = 9
    vbmc.optim_state["iter"] = 100
    vbmc.optim_state["entropy_switch"] = True
    terminated, _, _ = vbmc._check_termination_conditions()
    assert not terminated


def test_vbmc_check_termination_conditions_stability(mocker):
    options = {
        "max_fun_evals": 10,
        "min_fun_evals": 5,
        "min_iter": 5,
        "max_iter": 100,
        "tol_improvement": 0.01,
        "tol_stable_count": 60,
        "fun_evals_per_iter": 5,
        "tol_stable_excpt_frac": 0.2,
    }
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    vbmc.function_logger.func_count = 9
    vbmc.optim_state["entropy_switch"] = False
    vbmc.optim_state["iter"] = 98
    vbmc.iteration_history["r_index"] = np.ones(100) * 0.5

    mocker.patch(
        "pyvbmc.vbmc.VBMC._compute_reliability_index",
        return_value=(0.5, 0.005),
    )
    terminated, _, _ = vbmc._check_termination_conditions()
    assert terminated
    mocker.patch(
        "pyvbmc.vbmc.VBMC._compute_reliability_index",
        return_value=(1, 0.005),
    )
    terminated, _, _ = vbmc._check_termination_conditions()
    assert not terminated

    mocker.patch(
        "pyvbmc.vbmc.VBMC._compute_reliability_index",
        return_value=(0.5, 0.1),
    )
    terminated, _, _ = vbmc._check_termination_conditions()
    assert not terminated

    vbmc.optim_state["iter"] = 9
    mocker.patch(
        "pyvbmc.vbmc.VBMC._compute_reliability_index",
        return_value=(1, 0.005),
    )
    terminated, _, _ = vbmc._check_termination_conditions()
    assert not terminated


def test_vbmc_is_finished_stability_entropy_switch(mocker):
    options = {
        "max_fun_evals": 10,
        "min_fun_evals": 5,
        "min_iter": 5,
        "max_iter": 100,
        "tol_improvement": 0.01,
        "tol_stable_entropy_iters": 6,
        "tol_stable_excpt_frac": 0.2,
    }
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    vbmc.function_logger.func_count = 9
    vbmc.optim_state["entropy_switch"] = True
    vbmc.optim_state["iter"] = 98
    vbmc.iteration_history["r_index"] = np.ones(100) * 0.5
    mocker.patch(
        "pyvbmc.vbmc.VBMC._compute_reliability_index",
        return_value=(0.5, 0.005),
    )
    terminated, _, _ = vbmc._check_termination_conditions()
    assert not terminated


def test_vbmc_compute_reliability_index_less_than_2_iter():
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4)
    vbmc.optim_state["iter"] = 1
    r_index, ELCBO_improvement = vbmc._compute_reliability_index(6)
    assert r_index == np.Inf
    assert np.isnan(ELCBO_improvement)


def test_vbmc_compute_reliability_index():
    options = {"elcbo_impro_weight": 0, "tol_skl": 0.03}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    vbmc.optim_state["iter"] = 49
    vbmc.iteration_history.check_keys = False
    vbmc.iteration_history["elbo"] = np.arange(50) * 10
    vbmc.iteration_history["elbo_sd"] = np.ones(50)
    vbmc.iteration_history["sKL"] = np.ones(50)
    vbmc.iteration_history["func_count"] = np.arange(50) * 10
    r_index, ELCBO_improvement = vbmc._compute_reliability_index(6)
    assert r_index == np.mean([10, 1, 1 / 0.03])
    assert np.isclose(ELCBO_improvement, 1)


def test_is_gp_sampling_finished():
    options = {"tol_gp_var_mcmc": 1e-4}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    vbmc.optim_state["N"] = 300
    vbmc.optim_state["iter"] = 9
    vbmc.optim_state["warmup"] = False
    vbmc.iteration_history = dict()
    vbmc.iteration_history["N"] = np.ones(10)

    # all variances low
    vbmc.iteration_history["gp_sample_var"] = np.ones(10) * 1e-5
    vbmc.optim_state["stop_gp_sampling"] = 0
    vbmc._is_gp_sampling_finished()
    assert vbmc._is_gp_sampling_finished()

    # all variances high
    vbmc.iteration_history["gp_sample_var"] = np.ones(10)
    vbmc.optim_state["stop_gp_sampling"] = 0
    vbmc._is_gp_sampling_finished()
    assert not vbmc._is_gp_sampling_finished()

    # last variance high
    vbmc.iteration_history["gp_sample_var"] = np.ones(10) * 1e-10
    vbmc.iteration_history["gp_sample_var"][-1] = 1e-2
    vbmc.optim_state["stop_gp_sampling"] = 0
    vbmc._is_gp_sampling_finished()
    assert not vbmc._is_gp_sampling_finished()


def test_check_warmup_end_conditions_false():
    """
    no_recent_trim_flag is False
    """
    options = {"fun_evals_per_iter": 5}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    vbmc.optim_state["iter"] = 100
    vbmc.function_logger.func_count = 5
    vbmc.optim_state["N"] = 100
    vbmc.optim_state["data_trim_list"] = np.ones(101) * 91
    vbmc.iteration_history["lcb_max"] = np.ones(101)
    vbmc.iteration_history["elbo"] = np.ones(101)
    vbmc.iteration_history["elbo_sd"] = np.ones(101) * 1e-4
    vbmc.iteration_history["func_count"] = np.ones(101)
    assert not vbmc._check_warmup_end_conditions()


def test_check_warmup_end_conditions_bo_warmup():
    """
    stable_count_flag and no_recent_improvement_flag and no_recent_trim_flag
    are True
    """
    options = {"warmup_check_max": False}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    vbmc.optim_state["iter"] = 100
    vbmc.function_logger.func_count = 5
    vbmc.optim_state["N"] = 100
    vbmc.optim_state["data_trim_list"] = np.ones(101)
    vbmc.iteration_history["lcb_max"] = np.ones(101)
    vbmc.iteration_history["elbo"] = np.ones(101)
    vbmc.iteration_history["elbo_sd"] = np.ones(101) * 1e-4
    vbmc.iteration_history["func_count"] = np.ones(101)
    assert vbmc._check_warmup_end_conditions()


def test_check_warmup_end_conditions_first_conditions_true():
    """
    stable_count_flag and no_recent_improvement_flag and no_recent_trim_flag
    are True
    """
    options = {"fun_evals_per_iter": 5}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    vbmc.optim_state["iter"] = 100
    vbmc.function_logger.func_count = 5
    vbmc.optim_state["N"] = 11
    vbmc.optim_state["data_trim_list"] = []
    vbmc.iteration_history["lcb_max"] = np.ones(101)
    vbmc.iteration_history["elbo"] = np.ones(101)
    vbmc.iteration_history["elbo_sd"] = np.ones(101) * 1e-4
    vbmc.iteration_history["func_count"] = np.ones(101)
    assert vbmc._check_warmup_end_conditions()


def test_check_warmup_end_conditions_alternative_conditions_true():
    """
    no_longterm_improvement_flag and no_recent_trim_flag are True
    """
    options = {
        "fun_evals_per_iter": 5,
        "stop_warmup_thresh": 0,
        "warmup_no_impro_threshold": 0,
    }
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    vbmc.optim_state["iter"] = 100
    vbmc.function_logger.func_count = 5
    vbmc.optim_state["N"] = 100
    vbmc.optim_state["data_trim_list"] = np.ones(101)
    vbmc.iteration_history["lcb_max"] = np.ones(101)
    vbmc.iteration_history["elbo"] = np.ones(101)
    vbmc.iteration_history["elbo_sd"] = np.ones(101) * 1e-4
    vbmc.iteration_history["func_count"] = np.ones(101)
    assert vbmc._check_warmup_end_conditions()


def test_setup_vbmc_after_warmup_no_false_alarm_still_keep_points():
    """
    Test the behaviour when the warmup has ended. Some points will be kept
    despite being above the threshold to keep D+1 points in total.
    """
    options = {
        "fun_evals_per_iter": 5,
        "stop_warmup_thresh": 0,
        "warmup_no_impro_threshold": 0,
        "skip_active_sampling_after_warmup": False,
    }
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    vbmc.optim_state["iter"] = 100
    vbmc.function_logger.func_count = 5
    vbmc.optim_state["N"] = 100
    vbmc.optim_state["data_trim_list"] = np.ones(101)
    vbmc.iteration_history["lcb_max"] = np.ones(101)
    vbmc.iteration_history["elbo"] = np.ones(101)
    vbmc.iteration_history["elbo_sd"] = np.ones(101) * 1e-4
    vbmc.iteration_history["func_count"] = np.ones(101)
    vbmc.iteration_history["r_index"] = np.ones(101) * 1e-4
    assert vbmc._check_warmup_end_conditions()
    for i in range(3):
        vbmc.function_logger.add(np.ones((3)) * i, 3000 * i)
    vbmc._setup_vbmc_after_warmup()
    for i in range(3):
        assert vbmc.function_logger.X_flag[i]

    assert not vbmc.optim_state.get("warmup")
    assert vbmc.optim_state.get("last_warmup") == 100
    assert vbmc.optim_state.get("recompute_var_post")
    assert not vbmc.optim_state.get("skip_active_sampling")
    assert vbmc.optim_state.get("data_trim_list")[-1] == 1


def test_setup_vbmc_after_warmup_false_alarm():
    """
    Test the behaviour when it has been detected as a false alarm.
    """
    options = {
        "fun_evals_per_iter": 5,
        "stop_warmup_thresh": 0,
        "warmup_no_impro_threshold": 0,
        "skip_active_sampling_after_warmup": False,
        "warmup_keep_threshold_false_alarm": 400,
        "stop_warmup_reliability": 1,
    }
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    vbmc.optim_state["iter"] = 100
    vbmc.function_logger.func_count = 5
    vbmc.optim_state["N"] = 100
    vbmc.optim_state["data_trim_list"] = np.ones(0)
    vbmc.iteration_history["lcb_max"] = np.ones(101)
    vbmc.iteration_history["elbo"] = np.ones(101)
    vbmc.iteration_history["elbo_sd"] = np.ones(101) * 1e-4
    vbmc.iteration_history["func_count"] = np.ones(101)
    vbmc.iteration_history["r_index"] = np.ones(101) * 2
    assert vbmc._check_warmup_end_conditions()
    for i in range(6):
        vbmc.function_logger.add(np.ones((3)) * i, 3000 * i)
    vbmc._setup_vbmc_after_warmup()
    assert vbmc.function_logger.X_flag[0] == False
    assert vbmc.function_logger.X_flag[1] == False
    for i in range(2, 6):
        assert vbmc.function_logger.X_flag[i]

    assert vbmc.optim_state.get("recompute_var_post")
    assert not vbmc.optim_state.get("skip_active_sampling")
    assert vbmc.optim_state.get("data_trim_list")[-1] == 100


def test_setup_vbmc_after_warmup_false_alarm_no_warmup_keep_threshold_false_alarm():
    """
    Same as the test_setup_vbmc_after_warmup_false_alarm test but with one
    option being None.
    """
    options = {
        "fun_evals_per_iter": 5,
        "stop_warmup_thresh": 0,
        "warmup_no_impro_threshold": 0,
        "skip_active_sampling_after_warmup": False,
        "warmup_keep_threshold": 400,
        "warmup_keep_threshold_false_alarm": None,
        "stop_warmup_reliability": 1,
    }
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, options)
    vbmc.optim_state["iter"] = 100
    vbmc.function_logger.func_count = 5
    vbmc.optim_state["N"] = 100
    vbmc.optim_state["data_trim_list"] = np.ones(0)
    vbmc.iteration_history["lcb_max"] = np.ones(101)
    vbmc.iteration_history["elbo"] = np.ones(101)
    vbmc.iteration_history["elbo_sd"] = np.ones(101) * 1e-4
    vbmc.iteration_history["func_count"] = np.ones(101)
    vbmc.iteration_history["r_index"] = np.ones(101) * 2
    assert vbmc._check_warmup_end_conditions()
    for i in range(6):
        vbmc.function_logger.add(np.ones((3)) * i, 3000 * i)
    vbmc._setup_vbmc_after_warmup()
    assert vbmc.function_logger.X_flag[0] == False
    assert vbmc.function_logger.X_flag[1] == False
    for i in range(2, 6):
        assert vbmc.function_logger.X_flag[i]

    assert vbmc.optim_state.get("recompute_var_post")
    assert not vbmc.optim_state.get("skip_active_sampling")
    assert vbmc.optim_state.get("data_trim_list")[-1] == 100
