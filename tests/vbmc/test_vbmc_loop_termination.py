import numpy as np

from pyvbmc.vbmc import VBMC

fun = lambda x: np.sum(x + 2)


def create_vbmc(
    D: int,
    x0: float,
    lower_bounds: float,
    upper_bounds: float,
    plausible_lower_bounds: float,
    plausible_upper_bounds: float,
    user_options: dict = None,
):
    lb = np.ones((1, D)) * lower_bounds
    ub = np.ones((1, D)) * upper_bounds
    x0_array = np.ones((2, D)) * x0
    plb = np.ones((1, D)) * plausible_lower_bounds
    pub = np.ones((1, D)) * plausible_upper_bounds
    return VBMC(fun, x0_array, lb, ub, plb, pub, user_options)


def test_vbmc_check_termination_conditions_maxfunevals(mocker):
    user_options = {
        "maxfunevals": 10,
        "minfunevals": 5,
        "miniter": 5,
        "maxiter": 100,
        "tolstablecount": 60,
        "funevalsperiter": 5,
    }
    mocker.patch(
        "pyvbmc.vbmc.VBMC._compute_reliability_index",
        return_value=(np.Inf, np.NaN),
    )
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    vbmc.function_logger.func_count = 10
    vbmc.optim_state["iter"] = 10
    assert vbmc._check_termination_conditions() == True
    vbmc.function_logger.func_count = 9
    assert vbmc._check_termination_conditions() == False


def test_vbmc_check_termination_conditions_maxiter(mocker):
    mocker.patch(
        "pyvbmc.vbmc.VBMC._compute_reliability_index",
        return_value=(np.Inf, np.NaN),
    )
    user_options = {
        "maxfunevals": 10,
        "minfunevals": 5,
        "miniter": 5,
        "maxiter": 100,
    }
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    vbmc.function_logger.func_count = 9
    vbmc.optim_state["entropy_switch"] = True
    vbmc.optim_state["iter"] = 100
    assert vbmc._check_termination_conditions() == True
    vbmc.optim_state["iter"] = 99
    assert vbmc._check_termination_conditions() == False


def test_vbmc_check_termination_conditions_prevent_early_termination(mocker):
    user_options = {
        "maxfunevals": 10,
        "minfunevals": 5,
        "miniter": 101,
        "maxiter": 100,
    }
    mocker.patch(
        "pyvbmc.vbmc.VBMC._compute_reliability_index",
        return_value=(np.Inf, np.NaN),
    )
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    vbmc.function_logger.func_count = 9
    vbmc.optim_state["iter"] = 100
    vbmc.optim_state["entropy_switch"] = True
    assert vbmc._check_termination_conditions() == False
    user_options = {
        "maxfunevals": 10,
        "minfunevals": 11,
        "miniter": 5,
        "maxiter": 100,
    }
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    vbmc.function_logger.func_count = 9
    vbmc.optim_state["iter"] = 100
    vbmc.optim_state["entropy_switch"] = True
    assert vbmc._check_termination_conditions() == False


def test_vbmc_check_termination_conditions_stability(mocker):
    user_options = {
        "maxfunevals": 10,
        "minfunevals": 5,
        "miniter": 5,
        "maxiter": 100,
        "tolimprovement": 0.01,
        "tolstablecount": 60,
        "funevalsperiter": 5,
        "tolstableexcptfrac": 0.2,
    }
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    vbmc.function_logger.func_count = 9
    vbmc.optim_state["entropy_switch"] = False
    vbmc.optim_state["iter"] = 99
    vbmc.iteration_history["rindex"] = np.ones(100) * 0.5

    mocker.patch(
        "pyvbmc.vbmc.VBMC._compute_reliability_index",
        return_value=(0.5, 0.005),
    )
    assert vbmc._check_termination_conditions() == True
    mocker.patch(
        "pyvbmc.vbmc.VBMC._compute_reliability_index",
        return_value=(1, 0.005),
    )
    assert vbmc._check_termination_conditions() == False
    mocker.patch(
        "pyvbmc.vbmc.VBMC._compute_reliability_index",
        return_value=(0.5, 0.1),
    )
    assert vbmc._check_termination_conditions() == False
    vbmc.optim_state["iter"] = 9
    mocker.patch(
        "pyvbmc.vbmc.VBMC._compute_reliability_index",
        return_value=(1, 0.005),
    )
    assert vbmc._check_termination_conditions() == False


def test_vbmc_is_finished_stability_entropyswitch(mocker):
    user_options = {
        "maxfunevals": 10,
        "minfunevals": 5,
        "miniter": 5,
        "maxiter": 100,
        "tolimprovement": 0.01,
        "tolstableentropyiters": 6,
        "tolstableexcptfrac": 0.2,
    }
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    vbmc.function_logger.func_count = 9
    vbmc.optim_state["entropy_switch"] = True
    vbmc.optim_state["iter"] = 99
    vbmc.iteration_history["rindex"] = np.ones(100) * 0.5
    mocker.patch(
        "pyvbmc.vbmc.VBMC._compute_reliability_index",
        return_value=(0.5, 0.005),
    )
    assert vbmc._check_termination_conditions() == False


def test_vbmc_compute_reliability_index_less_than_3_iter():
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4)
    vbmc.optim_state["iter"] = 2
    rindex, ELCBO_improvement = vbmc._compute_reliability_index(6)
    assert rindex == np.Inf
    assert np.isnan(ELCBO_improvement)


def test_vbmc_compute_reliability_index():
    user_options = {"elcboimproweight": 0, "tolskl": 0.03}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    vbmc.optim_state["iter"] = 49
    vbmc.iteration_history.check_keys = False
    vbmc.iteration_history["elbo"] = np.arange(50) * 10
    vbmc.iteration_history["elbo_sd"] = np.ones(50)
    vbmc.iteration_history["sKL"] = np.ones(50)
    vbmc.iteration_history["func_count"] = np.arange(50) * 10
    rindex, ELCBO_improvement = vbmc._compute_reliability_index(6)
    assert rindex == np.mean([10, 1, 1 / 0.03])
    assert np.isclose(ELCBO_improvement, 1)


def test_is_gp_sampling_finished():
    user_options = {"tolgpvarmcmc": 1e-4}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
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
    user_options = {"funevalsperiter": 5}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    vbmc.optim_state["iter"] = 100
    vbmc.function_logger.func_count = 5
    vbmc.optim_state["N"] = 100
    vbmc.optim_state["data_trim_list"] = np.ones(101) * 91
    vbmc.iteration_history["lcbmax"] = np.ones(101)
    vbmc.iteration_history["elbo"] = np.ones(101)
    vbmc.iteration_history["elbo_sd"] = np.ones(101) * 1e-4
    vbmc.iteration_history["func_count"] = np.ones(101)
    assert not vbmc._check_warmup_end_conditions()


def test_check_warmup_end_conditions_bo_warmup():
    """
    stable_count_flag and no_recent_improvement_flag and no_recent_trim_flag
    are True
    """
    user_options = {"warmupcheckmax": False}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    vbmc.optim_state["iter"] = 100
    vbmc.function_logger.func_count = 5
    vbmc.optim_state["N"] = 100
    vbmc.optim_state["data_trim_list"] = np.ones(101)
    vbmc.iteration_history["lcbmax"] = np.ones(101)
    vbmc.iteration_history["elbo"] = np.ones(101)
    vbmc.iteration_history["elbo_sd"] = np.ones(101) * 1e-4
    vbmc.iteration_history["func_count"] = np.ones(101)
    assert vbmc._check_warmup_end_conditions()


def test_check_warmup_end_conditions_first_conditions_true():
    """
    stable_count_flag and no_recent_improvement_flag and no_recent_trim_flag
    are True
    """
    user_options = {"funevalsperiter": 5}
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    vbmc.optim_state["iter"] = 100
    vbmc.function_logger.func_count = 5
    vbmc.optim_state["N"] = 11
    vbmc.optim_state["data_trim_list"] = []
    vbmc.iteration_history["lcbmax"] = np.ones(101)
    vbmc.iteration_history["elbo"] = np.ones(101)
    vbmc.iteration_history["elbo_sd"] = np.ones(101) * 1e-4
    vbmc.iteration_history["func_count"] = np.ones(101)
    assert vbmc._check_warmup_end_conditions()


def test_check_warmup_end_conditions_alternative_conditions_true():
    """
    no_longterm_improvement_flag and no_recent_trim_flag are True
    """
    user_options = {
        "funevalsperiter": 5,
        "stopwarmupthresh": 0,
        "warmupnoimprothreshold": 0,
    }
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    vbmc.optim_state["iter"] = 100
    vbmc.function_logger.func_count = 5
    vbmc.optim_state["N"] = 100
    vbmc.optim_state["data_trim_list"] = np.ones(101)
    vbmc.iteration_history["lcbmax"] = np.ones(101)
    vbmc.iteration_history["elbo"] = np.ones(101)
    vbmc.iteration_history["elbo_sd"] = np.ones(101) * 1e-4
    vbmc.iteration_history["func_count"] = np.ones(101)
    assert vbmc._check_warmup_end_conditions()


def test_setup_vbmc_after_warmup_no_false_alarm_still_keep_points():
    """
    Test the behaviour when the warmup has ended. Some points will be kept
    despite being above the threshold to keep D+1 points in total.
    """
    user_options = {
        "funevalsperiter": 5,
        "stopwarmupthresh": 0,
        "warmupnoimprothreshold": 0,
        "skipactivesamplingafterwarmup": False,
    }
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    vbmc.optim_state["iter"] = 100
    vbmc.function_logger.func_count = 5
    vbmc.optim_state["N"] = 100
    vbmc.optim_state["data_trim_list"] = np.ones(101)
    vbmc.iteration_history["lcbmax"] = np.ones(101)
    vbmc.iteration_history["elbo"] = np.ones(101)
    vbmc.iteration_history["elbo_sd"] = np.ones(101) * 1e-4
    vbmc.iteration_history["func_count"] = np.ones(101)
    vbmc.iteration_history["rindex"] = np.ones(101) * 1e-4
    assert vbmc._check_warmup_end_conditions()
    for i in range(3):
        vbmc.function_logger.add(np.ones((3)) * i, 3000 * i)
    vbmc._setup_vbmc_after_warmup()
    for i in range(3):
        assert vbmc.function_logger.X_flag[i]

    assert not vbmc.optim_state.get("warmup")
    assert vbmc.optim_state.get("lastwarmup") == 100
    assert vbmc.optim_state.get("recompute_var_post")
    assert not vbmc.optim_state.get("skipactivesampling")
    assert vbmc.optim_state.get("data_trim_list")[-1] == 100


def test_setup_vbmc_after_warmup_false_alarm():
    """
    Test the behaviour when it has been detected as a false alarm.
    """
    user_options = {
        "funevalsperiter": 5,
        "stopwarmupthresh": 0,
        "warmupnoimprothreshold": 0,
        "skipactivesamplingafterwarmup": False,
        "warmupkeepthresholdfalsealarm": 400,
        "stopwarmupreliability": 1,
    }
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    vbmc.optim_state["iter"] = 100
    vbmc.function_logger.func_count = 5
    vbmc.optim_state["N"] = 100
    vbmc.optim_state["data_trim_list"] = np.ones(0)
    vbmc.iteration_history["lcbmax"] = np.ones(101)
    vbmc.iteration_history["elbo"] = np.ones(101)
    vbmc.iteration_history["elbo_sd"] = np.ones(101) * 1e-4
    vbmc.iteration_history["func_count"] = np.ones(101)
    vbmc.iteration_history["rindex"] = np.ones(101) * 2
    assert vbmc._check_warmup_end_conditions()
    for i in range(6):
        vbmc.function_logger.add(np.ones((3)) * i, 3000 * i)
    vbmc._setup_vbmc_after_warmup()
    assert vbmc.function_logger.X_flag[0] == False
    for i in range(1, 6):
        assert vbmc.function_logger.X_flag[i]

    assert vbmc.optim_state.get("recompute_var_post")
    assert not vbmc.optim_state.get("skipactivesampling")
    assert vbmc.optim_state.get("data_trim_list")[-1] == 100


def test_setup_vbmc_after_warmup_false_alarm_no_warmupkeepthresholdfalsealarm():
    """
    Same as the test_setup_vbmc_after_warmup_false_alarm test but with one
    option being None.
    """
    user_options = {
        "funevalsperiter": 5,
        "stopwarmupthresh": 0,
        "warmupnoimprothreshold": 0,
        "skipactivesamplingafterwarmup": False,
        "warmupkeepthreshold": 400,
        "warmupkeepthresholdfalsealarm": None,
        "stopwarmupreliability": 1,
    }
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4, user_options)
    vbmc.optim_state["iter"] = 100
    vbmc.function_logger.func_count = 5
    vbmc.optim_state["N"] = 100
    vbmc.optim_state["data_trim_list"] = np.ones(0)
    vbmc.iteration_history["lcbmax"] = np.ones(101)
    vbmc.iteration_history["elbo"] = np.ones(101)
    vbmc.iteration_history["elbo_sd"] = np.ones(101) * 1e-4
    vbmc.iteration_history["func_count"] = np.ones(101)
    vbmc.iteration_history["rindex"] = np.ones(101) * 2
    assert vbmc._check_warmup_end_conditions()
    for i in range(6):
        vbmc.function_logger.add(np.ones((3)) * i, 3000 * i)
    vbmc._setup_vbmc_after_warmup()
    assert vbmc.function_logger.X_flag[0] == False
    for i in range(1, 6):
        assert vbmc.function_logger.X_flag[i]

    assert vbmc.optim_state.get("recompute_var_post")
    assert not vbmc.optim_state.get("skipactivesampling")
    assert vbmc.optim_state.get("data_trim_list")[-1] == 100
