import copy
from pathlib import Path

import dill
import numpy as np
import pytest
import scipy.stats as scs

from pyvbmc import VBMC
from pyvbmc.priors import SciPy
from pyvbmc.vbmc.gaussian_process_train import _get_gp_training_options

base_path = Path(__file__).parent


def test_vbmc_save_dynamic():
    """Test saving an arbitrary VBMC instance."""

    D = np.random.choice(range(1, 21))
    lb = np.random.uniform(size=(1, D))
    plb = lb + 0.1 + np.random.uniform(size=(1, D))
    pub = plb + 1.0 + np.random.uniform(size=(1, D))
    ub = pub + 0.1 + np.random.uniform(size=(1, D))
    x0 = np.random.uniform(plb, pub)
    x_random = np.random.uniform(lb, ub, size=(100, D))

    # Test saving lambda expression likelihood and "complicated" prior object
    def log_likelihood(x):
        return np.sum(x**2 + x + 1)

    prior = SciPy._generic(D)
    vbmc = VBMC(log_likelihood, x0, lb, ub, plb, pub, prior=prior)
    vbmc.save(base_path.joinpath("test_vbmc_save_dynamic.pkl"), overwrite=True)

    y_random = vbmc.log_joint(x_random)
    data = {
        "D": D,
        "lb": lb,
        "plb": plb,
        "pub": pub,
        "ub": ub,
        "x0": x0,
        "x_random": x_random,
        "y_random": y_random,
    }
    with open(
        base_path.joinpath("test_vbmc_save_dynamic_data.pkl"), "wb"
    ) as f:
        dill.dump(data, f)


def test_vbmc_load_dynamic():
    """Test loading the object saved above."""
    vbmc = VBMC.load(base_path.joinpath("test_vbmc_save_dynamic"))
    with open(
        base_path.joinpath("test_vbmc_save_dynamic_data.pkl"), "rb"
    ) as f:
        data = dill.load(f)

    assert vbmc.D == data["D"]
    assert np.all(vbmc.lower_bounds == data["lb"])
    assert np.all(vbmc.plausible_lower_bounds == data["plb"])
    assert np.all(vbmc.plausible_upper_bounds == data["pub"])
    assert np.all(vbmc.upper_bounds == data["ub"])
    assert np.allclose(vbmc.parameter_transformer.inverse(vbmc.x0), data["x0"])
    assert np.all(vbmc.log_joint(data["x_random"]) == data["y_random"])


def test_vbmc_load_static():
    """Test loading VBMC object which has already been optimized."""
    D = 4
    prior_mu = np.zeros(D)
    prior_var = 3 * np.ones(D)
    LB = np.full(D, -np.inf)  # Lower bounds
    PLB = np.full(D, prior_mu - np.sqrt(prior_var))  # Plausible lower bounds
    PUB = np.full(D, prior_mu + np.sqrt(prior_var))  # Plausible upper bounds
    UB = np.full(D, np.inf)  # Upper bounds

    random_state = np.random.get_state()

    vbmc = VBMC.load(base_path.joinpath("test_vbmc_save_static.pkl"))
    assert vbmc.D == D
    assert np.all(np.equal(vbmc.lower_bounds, LB))
    assert np.all(np.equal(vbmc.plausible_lower_bounds, PLB))
    assert np.all(np.equal(vbmc.plausible_upper_bounds, PUB))
    assert np.all(np.equal(vbmc.upper_bounds, UB))
    assert vbmc.options["max_iter"] == 300
    for i, val in enumerate(random_state):
        assert np.all(np.random.get_state()[i] == val)
    assert vbmc.options["max_fun_evals"] == 40
    assert vbmc.iteration == 6
    assert vbmc.options["vectorized_target"] is False
    assert vbmc.function_logger.vectorized_target is False

    vbmc = VBMC.load(
        base_path.joinpath("test_vbmc_save_static.pkl"),
        iteration=0,
    )
    assert vbmc.D == D
    assert np.all(np.equal(vbmc.lower_bounds, LB))
    assert np.all(np.equal(vbmc.plausible_lower_bounds, PLB))
    assert np.all(np.equal(vbmc.plausible_upper_bounds, PUB))
    assert np.all(np.equal(vbmc.upper_bounds, UB))
    assert vbmc.options["max_iter"] == 300
    for i, val in enumerate(random_state):
        assert np.all(np.random.get_state()[i] == val)
    assert vbmc.options["max_fun_evals"] == 40
    assert vbmc.iteration == 0
    # The instance is set up to continue, so its GP carries the posterior
    # factors the iteration history does not store, and is a copy of the
    # record rather than the record itself.
    assert vbmc.gp.posteriors[0].alpha is not None
    assert vbmc.gp is not vbmc.iteration_history["gp"][0]

    vbmc = VBMC.load(
        base_path.joinpath("test_vbmc_save_static.pkl"),
        new_options={"max_fun_evals": 42},
        iteration=0,
    )
    assert vbmc.D == D
    assert np.all(np.equal(vbmc.lower_bounds, LB))
    assert np.all(np.equal(vbmc.plausible_lower_bounds, PLB))
    assert np.all(np.equal(vbmc.plausible_upper_bounds, PUB))
    assert np.all(np.equal(vbmc.upper_bounds, UB))
    assert vbmc.options["max_iter"] == 300
    for i, val in enumerate(random_state):
        assert np.all(np.random.get_state()[i] == val)
    assert vbmc.options["max_fun_evals"] == 42
    assert vbmc.iteration == 0

    vbmc = VBMC.load(
        base_path.joinpath("test_vbmc_save_static.pkl"),
        new_options={"max_fun_evals": 42},
        iteration=0,
        set_random_state=True,
    )
    assert vbmc.D == D
    assert np.all(np.equal(vbmc.lower_bounds, LB))
    assert np.all(np.equal(vbmc.plausible_lower_bounds, PLB))
    assert np.all(np.equal(vbmc.plausible_upper_bounds, PUB))
    assert np.all(np.equal(vbmc.upper_bounds, UB))
    assert vbmc.options["max_iter"] == 300
    assert not np.all(np.random.get_state()[1] == random_state[1])
    assert vbmc.options["max_fun_evals"] == 42
    assert vbmc.iteration == 0


def _gp_fit_starting_points(vbmc, n_eff):
    """The starting points the GP hyperparameter fit would use at ``n_eff``.

    The count follows a schedule which runs from ``gp_train_n_init`` down to
    ``gp_train_n_init_final`` over the evaluation budget of the run, so it
    reports which budget the schedule is working from.
    """
    optim_state = dict(vbmc.optim_state)
    optim_state["iter"] = 0
    optim_state["n_eff"] = n_eff
    optim_state["recompute_var_post"] = True
    return _get_gp_training_options(
        optim_state, vbmc.iteration_history, vbmc.options, {}, 0
    )["init_N"]


def _fresh_vbmc(D, budget):
    """A fresh instance of dimension ``D`` with ``max_fun_evals = budget``."""
    return VBMC(
        lambda x: -0.5 * np.sum(x**2),
        np.zeros((1, D)),
        np.full((1, D), -np.inf),
        np.full((1, D), np.inf),
        np.full((1, D), -1.0),
        np.full((1, D), 1.0),
        options={"max_fun_evals": budget, "display": "off"},
        seed=1,
    )


class _ReachedFirstIteration(Exception):
    """Raised to leave ``optimize()`` once it is about to iterate."""


def test_load_gives_an_older_run_the_final_boost_it_was_made_with():
    """A run saved before ``tol_elcbo_boost`` existed keeps the unguarded
    final boost of its day, which the option states as ``None``."""
    with open(base_path.joinpath("test_vbmc_save_static.pkl"), "rb") as f:
        assert "tol_elcbo_boost" not in dill.load(f).options

    loaded = VBMC.load(base_path.joinpath("test_vbmc_save_static.pkl"))
    assert "tol_elcbo_boost" in loaded.options
    assert loaded.options["tol_elcbo_boost"] is None


def test_load_with_a_larger_budget_schedules_the_gp_fit_as_a_fresh_run():
    """The hyperparameter fit follows the budget the run was loaded with.

    A run continued with a larger ``max_fun_evals`` must schedule the
    number of starting points of its GP hyperparameter fit as a fresh run
    with that budget would at the same number of effective evaluations.
    """
    budget = 500
    loaded = VBMC.load(
        base_path.joinpath("test_vbmc_save_static.pkl"),
        new_options={"max_fun_evals": budget},
    )
    fresh = _fresh_vbmc(loaded.D, budget)
    assert loaded.options["fun_eval_start"] == fresh.options["fun_eval_start"]

    for n_eff in (40, 60, 100, 200, 350, 500):
        assert _gp_fit_starting_points(
            loaded, n_eff
        ) == _gp_fit_starting_points(fresh, n_eff)


def test_resumed_run_schedules_the_gp_fit_as_a_fresh_run(monkeypatch):
    """Continuing a finished run keeps the budget it was loaded with.

    ``optimize()`` restores the state of the last recorded iteration before
    it continues, and that record carries the budget of the finished run.
    The schedule of the hyperparameter fit must still follow the budget of
    this call.
    """
    budget = 500
    loaded = VBMC.load(
        base_path.joinpath("test_vbmc_save_static.pkl"),
        new_options={"max_fun_evals": budget},
    )
    assert loaded.is_finished
    fresh = _fresh_vbmc(loaded.D, budget)

    # Leave `optimize()` at the last step before its first iteration, so
    # that the check sees the state the continuation left behind without
    # paying for an iteration.
    def stop(self):
        raise _ReachedFirstIteration

    monkeypatch.setattr(VBMC, "_log_column_headers", stop)
    with pytest.raises(_ReachedFirstIteration):
        loaded.optimize()

    for n_eff in (40, 60, 100, 200, 350, 500):
        assert _gp_fit_starting_points(
            loaded, n_eff
        ) == _gp_fit_starting_points(fresh, n_eff)


@pytest.mark.parametrize(
    "new_options",
    [
        {"max_iter": 0},
        {"max_iter": 7.5},
        {"max_fun_evals": -5},
        {"max_fun_evals": 120.5},
    ],
)
def test_load_refuses_the_run_limits_that_construction_refuses(
    tmp_path, new_options
):
    """A limit given to ``load`` is checked as one given at construction.

    ``max_fun_evals`` and ``max_iter`` are the options a continued run is
    most often given, and both have to be positive integers on either
    route.
    """
    with pytest.raises(ValueError, match="positive integer"):
        VBMC(
            lambda x: -0.5 * np.sum(x**2),
            np.zeros((1, 2)),
            options={**new_options, "display": "off"},
        )

    path = tmp_path / "fresh"
    _fresh_vbmc(2, 100).save(path)
    with pytest.raises(ValueError, match="positive integer"):
        VBMC.load(path, new_options=new_options)


@pytest.mark.parametrize("min_iter", [-1, 2.5, np.inf])
def test_load_refuses_the_min_iter_that_construction_refuses(
    tmp_path, min_iter
):
    """``min_iter`` has to be a finite non-negative integer on either route."""
    D = 2
    new_options = {"min_iter": min_iter, "max_iter": 2}
    with pytest.raises(
        ValueError, match="min_iter needs to be a finite non-neg"
    ):
        VBMC(
            lambda x: -0.5 * np.sum(x**2),
            np.zeros((1, D)),
            np.full((1, D), -np.inf),
            np.full((1, D), np.inf),
            np.full((1, D), -1.0),
            np.full((1, D), 1.0),
            options={**new_options, "display": "off"},
        )

    path = tmp_path / "fresh"
    _fresh_vbmc(D, 100).save(path)
    with pytest.raises(
        ValueError, match="min_iter needs to be a finite non-neg"
    ):
        VBMC.load(path, new_options=new_options)


def test_min_iter_of_zero_and_the_default_are_accepted_on_either_route(
    tmp_path,
):
    """0, which sets no minimum, and the default ``min_iter = D`` pass at
    construction and through ``load``."""
    D = 2
    fresh = _fresh_vbmc(D, 100)
    assert fresh.options["min_iter"] == D
    built = VBMC(
        lambda x: -0.5 * np.sum(x**2),
        np.zeros((1, D)),
        np.full((1, D), -np.inf),
        np.full((1, D), np.inf),
        np.full((1, D), -1.0),
        np.full((1, D), 1.0),
        options={"min_iter": 0, "display": "off"},
    )
    assert built.options["min_iter"] == 0

    path = tmp_path / "fresh"
    fresh.save(path)
    assert VBMC.load(path).options["min_iter"] == D
    loaded = VBMC.load(path, new_options={"min_iter": 0})
    assert loaded.options["min_iter"] == 0


def test_load_raises_max_iter_to_min_iter_as_construction_does(tmp_path):
    """A ``max_iter`` below ``min_iter`` is raised to it on either route."""
    path = tmp_path / "fresh"
    _fresh_vbmc(2, 100).save(path)
    loaded = VBMC.load(path, new_options={"max_iter": 2, "min_iter": 7})
    assert loaded.options["max_iter"] == 7

    unchanged = VBMC.load(path, new_options={"max_iter": 9, "min_iter": 7})
    assert unchanged.options["max_iter"] == 9


def _vbmc_with_options(options, D=2):
    """A seeded run on a Gaussian target with the given options."""
    return VBMC(
        lambda x: -0.5 * np.sum(x**2),
        np.zeros((1, D)),
        np.full((1, D), -np.inf),
        np.full((1, D), np.inf),
        np.full((1, D), -1.0),
        np.full((1, D), 1.0),
        options={"display": "off", **options},
        seed=5,
    )


@pytest.mark.parametrize(
    "built, given",
    [(0, 80), (80, 0), (0, 0), (80, 40)],
    ids=["raised_from_zero", "lowered_to_zero", "zero_again", "positive"],
)
def test_load_starts_or_stops_gp_sampling_as_ns_gp_max_says(
    tmp_path, built, given
):
    """``ns_gp_max`` sets the number of GP hyperparameter samples at every
    fit, and construction also derives from it whether the fits sample at
    all: ``optim_state["stop_sampling"]`` is 0, sampling, for a positive
    value, and infinity, no sampling, for 0. A value given to ``load``
    leaves the run in the state construction gives for it."""
    path = tmp_path / "run"
    _vbmc_with_options({"ns_gp_max": built}).save(path)

    loaded = VBMC.load(path, new_options={"ns_gp_max": given})

    expected = _vbmc_with_options({"ns_gp_max": given}).optim_state
    assert loaded.optim_state["stop_sampling"] == expected["stop_sampling"]


@pytest.mark.parametrize("given", [0, 80])
def test_load_leaves_gp_sampling_stopped_in_the_stable_regime(tmp_path, given):
    """A run whose sampling has stopped in the stable regime holds the
    number of training points at the stop in ``stop_sampling``, and its
    fits take ``stable_gp_samples`` whatever ``ns_gp_max`` says, so a value
    given to ``load`` leaves that state alone."""
    vbmc = _vbmc_with_options({"ns_gp_max": 80})
    vbmc.optim_state["stop_sampling"] = 37
    path = tmp_path / "run"
    vbmc.save(path)

    loaded = VBMC.load(path, new_options={"ns_gp_max": given})

    assert loaded.optim_state["stop_sampling"] == 37


def test_a_run_given_a_positive_ns_gp_max_on_load_samples(tmp_path):
    """A run built with ``ns_gp_max=0`` fits the GP hyperparameters by
    optimization alone. Loaded with a positive value and continued, it
    samples them: its next fit draws several samples."""
    options = {
        "ns_gp_max": 0,
        "max_iter": 1,
        "min_iter": 0,
        "max_fun_evals": 60,
        "do_final_boost": False,
    }
    vbmc = _vbmc_with_options(options)
    vbmc.optimize()
    assert len(vbmc.gp.posteriors) == 1
    path = tmp_path / "run"
    vbmc.save(path)

    loaded = VBMC.load(path, new_options={"ns_gp_max": 80, "max_iter": 2})
    loaded.optimize()

    assert loaded.iteration == 1
    assert len(loaded.gp.posteriors) > 1


def test_load_shares_the_parameter_transformer_of_the_chosen_iteration():
    """One transformer is shared, and it is the chosen iteration's.

    A run keeps a single ``ParameterTransformer`` object on the instance,
    on the variational posterior and on the function logger, so that the
    three always agree on the map between the user's coordinates and the
    inference space. Restoring the state recorded at an iteration must
    restore that sharing with the map of that iteration.
    """
    path = base_path.joinpath("test_vbmc_save_static.pkl")
    for iteration in (None, 0, 3, 6):
        vbmc = VBMC.load(path, iteration=iteration)
        transformer = vbmc.parameter_transformer
        assert transformer is vbmc.vp.parameter_transformer
        assert transformer is vbmc.function_logger.parameter_transformer

        recorded = vbmc.iteration_history["vp"][vbmc.iteration]
        probe = np.linspace(-1.0, 1.0, vbmc.D).reshape((1, vbmc.D))
        assert np.array_equal(
            transformer(probe), recorded.parameter_transformer(probe)
        )


def test_load_prefers_the_iterations_map_over_the_saved_live_one(tmp_path):
    """The restored map comes from the iteration, not from the run's end.

    A run that warps its inference space leaves the instance holding a map
    later than the ones recorded at earlier iterations, so a state restored
    from such an iteration must carry the recorded map.
    """
    # A short run of its own, written and read by the same interpreter. The
    # stored run of `test_vbmc_save_static.pkl` will not do: it holds its
    # target function pickled by value, as bytecode of the Python version
    # that wrote the file, and pickling such a function again makes dill
    # disassemble it, which crashes Python 3.11 on bytecode of another
    # version.
    D = 2
    vbmc = VBMC(
        lambda x: -0.5 * np.sum(x**2),
        np.zeros((1, D)),
        np.full((1, D), -np.inf),
        np.full((1, D), np.inf),
        np.full((1, D), -1.0),
        np.full((1, D), 1.0),
        options={"max_iter": 2, "display": "off"},
        seed=1,
    )
    vbmc.optimize()
    assert vbmc.iteration == 1

    later_map = copy.deepcopy(vbmc.parameter_transformer)
    later_map.mu = np.full(D, 0.5)
    later_map.delta = np.full(D, 2.0)
    vbmc.parameter_transformer = later_map
    doctored = tmp_path / "warped"
    vbmc.save(doctored)

    loaded = VBMC.load(doctored, iteration=0)
    probe = np.linspace(-1.0, 1.0, D).reshape((1, D))
    recorded = loaded.iteration_history["vp"][0].parameter_transformer
    assert np.array_equal(loaded.parameter_transformer(probe), recorded(probe))
    assert not np.array_equal(
        loaded.parameter_transformer(probe), later_map(probe)
    )


def test_load_recovers_the_starting_point_of_a_file_saved_without_it(tmp_path):
    """``x0_orig`` of an older file is the starting point the caller gave.

    A file saved before ``x0_orig`` existed carries the starting point in
    the inference space of construction alone. When the run has warped
    that space since, the map of the restored iteration is another one, and
    the starting point has to come back through the map the run started
    with.
    """
    D = 2
    x0 = np.array([[3.5, 0.0]])
    vbmc = VBMC(
        lambda x: -0.5 * np.sum(x**2),
        x0,
        np.full((1, D), -np.inf),
        np.full((1, D), np.inf),
        np.array([[2.0, -3.0]]),
        np.array([[4.0, 5.0]]),
        options={"max_iter": 2, "display": "off"},
        seed=1,
    )
    vbmc.optimize()
    assert vbmc.iteration == 1

    # The state a warp at the second iteration leaves: its map on the
    # record of that iteration and on the live objects, the first record
    # untouched.
    warped = copy.deepcopy(vbmc.parameter_transformer)
    warped.mu = np.zeros(D)
    warped.delta = np.ones(D)
    assert not np.allclose(warped.inverse(vbmc.x0), x0)
    vbmc.iteration_history["vp"][1].parameter_transformer = warped
    vbmc.vp.parameter_transformer = warped
    vbmc.parameter_transformer = warped
    vbmc.function_logger.parameter_transformer = warped
    del vbmc.x0_orig
    older = tmp_path / "older"
    vbmc.save(older)

    for iteration in (None, 0, 1):
        loaded = VBMC.load(older, iteration=iteration)
        assert np.allclose(loaded.x0_orig, x0)


def test_vbmc_save_load_error_handling():
    vbmc = VBMC.load(base_path.joinpath("test_vbmc_save_static.pkl"))
    with pytest.raises(FileExistsError) as err:
        vbmc.save(base_path.joinpath("test_vbmc_save_static.pkl"))
    with pytest.raises(OSError) as err:
        vbmc.save("/this/path/does/not/exist.pkl")
    with pytest.raises(OSError) as err:
        vbmc = VBMC.load("/this/path/does/not/exist.pkl")
    with pytest.raises(ValueError) as err:
        vbmc = VBMC.load(
            base_path.joinpath("test_vbmc_save_static.pkl"), iteration=10
        )
    assert (
        "Specified iteration (10) should be >= 0 and <= last stored iteration (6)."
        in err.value.args[0]
    )
    with pytest.raises(ValueError) as err:
        vbmc = VBMC.load(
            base_path.joinpath("test_vbmc_save_static.pkl"), iteration=-1
        )
    assert (
        "Specified iteration (-1) should be >= 0 and <= last stored iteration (6)."
        in err.value.args[0]
    )


def _save_load_vbmc(target, tmp_path, *, vectorized):
    D = 2
    vbmc = VBMC(
        target,
        np.zeros((1, D)),
        np.full((1, D), -np.inf),
        np.full((1, D), np.inf),
        np.full((1, D), -1.0),
        np.full((1, D), 1.0),
        prior=SciPy._generic(D),
        options={"vectorized_target": vectorized},
    )
    path = tmp_path / ("vectorized" if vectorized else "scalar")
    vbmc.save(path)
    return path


def test_vbmc_save_load_vectorized_mode_and_override(tmp_path):
    def dual_target(x):
        if x.ndim == 1:
            return np.sum(x)
        return np.sum(x, axis=1)

    path = _save_load_vbmc(dual_target, tmp_path, vectorized=True)
    loaded = VBMC.load(path)
    assert loaded.options["vectorized_target"] is True
    assert loaded.function_logger.vectorized_target is True
    values, _, _ = loaded.function_logger.batch_call(np.zeros((2, loaded.D)))
    assert values.shape == (2,)

    scalar = VBMC.load(path, new_options={"vectorized_target": False})
    assert scalar.options["vectorized_target"] is False
    assert scalar.function_logger.vectorized_target is False
    value, _, _ = scalar.function_logger(np.zeros(scalar.D))
    assert np.isfinite(value)


def test_vbmc_load_rebuilds_vectorized_prior_wrapper(tmp_path):
    def dual_target(x):
        if x.ndim == 1:
            return np.sum(x)
        return np.sum(x, axis=1)

    path = _save_load_vbmc(dual_target, tmp_path, vectorized=False)
    loaded = VBMC.load(path, new_options={"vectorized_target": True})

    assert loaded.options["vectorized_target"] is True
    assert loaded.function_logger.vectorized_target is True
    values, _, _ = loaded.function_logger.batch_call(np.zeros((2, loaded.D)))
    assert values.shape == (2,)


def test_vbmc_load_defaults_missing_vectorized_flags_to_false(tmp_path):
    path = _save_load_vbmc(np.sum, tmp_path, vectorized=False)
    with open(path.with_suffix(".pkl"), "rb") as file:
        vbmc = dill.load(file)
    vbmc.options.__delitem__("vectorized_target", force=True)
    del vbmc.function_logger.vectorized_target
    vbmc.save(path, overwrite=True)

    loaded = VBMC.load(path)

    assert loaded.options["vectorized_target"] is False
    assert loaded.function_logger.vectorized_target is False


def test_vbmc_load_backfills_missing_N_history(tmp_path):
    path = _save_load_vbmc(np.sum, tmp_path, vectorized=False)
    with open(path.with_suffix(".pkl"), "rb") as file:
        vbmc = dill.load(file)
    del vbmc.iteration_history["N"]
    vbmc.iteration = 2
    for iteration in range(3):
        optim_state = copy.deepcopy(vbmc.optim_state)
        optim_state["iter"] = iteration
        function_logger = copy.deepcopy(vbmc.function_logger)
        if iteration == 0:
            optim_state["N"] = 5
            function_logger.Xn = 99
        elif iteration == 1:
            function_logger.Xn = 7
        else:
            function_logger.Xn = np.nan
        vbmc.iteration_history.record_iteration(
            {
                "optim_state": optim_state,
                "function_logger": function_logger,
            },
            iteration,
        )
    vbmc.save(path, overwrite=True)

    loaded = VBMC.load(path)

    assert "N" in loaded.iteration_history
    assert list(loaded.iteration_history["N"][:2]) == [5, 8]
    assert loaded.iteration_history["N"][2] is None


def test_vbmc_load_mode_override_requires_likelihood_provenance(tmp_path):
    path = _save_load_vbmc(np.sum, tmp_path, vectorized=False)
    with open(path.with_suffix(".pkl"), "rb") as file:
        vbmc = dill.load(file)
    vbmc.log_likelihood = None
    vbmc.save(path, overwrite=True)

    with pytest.raises(ValueError, match="original likelihood is unavailable"):
        VBMC.load(path, new_options={"vectorized_target": True})
