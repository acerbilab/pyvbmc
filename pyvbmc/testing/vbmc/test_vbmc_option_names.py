"""An option name that PyVBMC does not declare is refused, whichever way
it was supplied: in the ``options`` dictionary, in the file of
``options_path``, or in the ``new_options`` of :py:meth:`VBMC.load`. The
same holds for an option value that construction refuses."""

import logging

import numpy as np
import pytest

from pyvbmc import VBMC

D = 2


def log_joint(x):
    return -0.5 * np.sum(x**2)


def _vbmc_of_dimension(dimension, **kwargs):
    return VBMC(
        log_joint,
        np.zeros((1, dimension)),
        np.full((1, dimension), -10.0),
        np.full((1, dimension), 10.0),
        np.full((1, dimension), -1.0),
        np.full((1, dimension), 1.0),
        **kwargs,
    )


def _vbmc(**kwargs):
    return _vbmc_of_dimension(D, **kwargs)


def test_unknown_name_in_the_options_dictionary_raises():
    with pytest.raises(ValueError) as execinfo:
        _vbmc(options={"max_fun_eval": 123})
    assert "The option max_fun_eval does not exist." in execinfo.value.args[0]


def test_unknown_name_in_an_options_file_raises(tmp_path):
    path = tmp_path.joinpath("user_options.ini")
    path.write_text("[UserOptions]\n# a misspelt budget\nmax_fun_eval = 123\n")
    with pytest.raises(ValueError) as execinfo:
        _vbmc(options_path=path)
    assert "The option max_fun_eval does not exist." in execinfo.value.args[0]


def test_declared_name_in_an_options_file_is_accepted(tmp_path):
    path = tmp_path.joinpath("user_options.ini")
    path.write_text("[UserOptions]\n# the budget\nmax_fun_evals = 123\n")
    vbmc = _vbmc(options_path=path)
    assert vbmc.options["max_fun_evals"] == 123


def test_unknown_name_in_new_options_raises(tmp_path):
    saved = tmp_path.joinpath("run.pkl")
    _vbmc().save(saved)
    with pytest.raises(ValueError) as execinfo:
        VBMC.load(saved, new_options={"max_fun_eval": 999})
    assert "The option max_fun_eval does not exist." in execinfo.value.args[0]


def test_declared_name_in_new_options_is_accepted(tmp_path):
    saved = tmp_path.joinpath("run.pkl")
    _vbmc().save(saved)
    vbmc = VBMC.load(saved, new_options={"max_fun_evals": 999})
    assert vbmc.options["max_fun_evals"] == 999


@pytest.mark.parametrize(
    "new_options, error, message",
    [
        ({"gp_hyp_sampler": "covsample"}, NotImplementedError, "slicesample"),
        ({"noise_shaping": True}, NotImplementedError, "noise_shaping"),
        ({"acq_hedge": True}, NotImplementedError, "acq_hedge"),
        ({"search_acq_fcn": "AcqFcnLog()"}, ValueError, "search_acq_fcn"),
        (
            {"search_optimizer": "Nelder-Mead"},
            ValueError,
            "search_optimizer",
        ),
        ({"search_optimizer": "bounded"}, ValueError, "search_optimizer"),
        ({"search_cache_frac": 0.5}, ValueError, "search_cache_frac"),
        ({"cache_frac": 1.5}, ValueError, "cache_frac"),
        ({"warp_cov_reg": np.nan}, ValueError, "warp_cov_reg"),
        ({"hpd_frac": 0.1}, ValueError, "hpd_frac"),
    ],
)
def test_new_options_are_checked_as_at_construction(
    tmp_path, new_options, error, message
):
    """A value that construction refuses is refused by ``load`` as well,
    with the same message, where the run would otherwise fail at the first
    use of the option, after that iteration's evaluations of the target."""
    with pytest.raises(error) as at_construction:
        _vbmc(options=new_options)
    assert message in at_construction.value.args[0]

    saved = tmp_path.joinpath("run.pkl")
    _vbmc().save(saved)
    with pytest.raises(error) as at_load:
        VBMC.load(saved, new_options=new_options)
    assert at_load.value.args[0] == at_construction.value.args[0]


@pytest.mark.parametrize(
    "new_options",
    [
        {"uncertainty_handling": True},
        {"specify_target_noise": True},
        {"gp_mean_fun": "const"},
        {"integer_vars": [0]},
        {"warmup": False},
        {"k_warmup": 7},
        {"entropy_switch": True},
        {"active_search_bound": 4},
        {"tol_bound_x": 1e-3},
        {"cache_size": 1000},
        {"k_warmup": 7, "max_iter": 9},
    ],
)
def test_load_refuses_an_option_only_construction_reads(tmp_path, new_options):
    """An option that PyVBMC reads only while it builds a ``VBMC`` object
    cannot be changed by ``load``: the saved run carries the state that was
    built from it, and ``load`` restores that state, so a value given here
    would be stored and never consulted. The refusal says so, rather than
    let the run continue with an option and a state that disagree."""
    saved = tmp_path.joinpath("run.pkl")
    _vbmc().save(saved)
    with pytest.raises(ValueError) as execinfo:
        VBMC.load(saved, new_options=new_options)
    message = execinfo.value.args[0]
    assert "construct a new VBMC object" in message
    for name in new_options:
        if name == "max_iter":
            continue
        assert repr(name) in message
    assert "'max_iter'" not in message


def test_load_takes_an_option_a_continued_run_reads(tmp_path):
    """The budget of a continued run is what ``load`` is most often given,
    and it takes effect."""
    saved = tmp_path.joinpath("run.pkl")
    _vbmc().save(saved)
    loaded = VBMC.load(saved, new_options={"max_iter": 9})
    assert loaded.options["max_iter"] == 9


@pytest.mark.parametrize(
    "name, value",
    [("gp_int_mean_fun", 1), ("proposal_fcn", "@(x)my_proposal")],
)
def test_load_takes_an_option_that_has_no_effect(
    tmp_path, caplog, name, value
):
    """An option that no module reads is taken by ``load`` as it is by
    construction, with the same warning: refusing it as an option that only
    construction reads would advise a new ``VBMC`` object, where the value
    has no effect either."""
    saved = tmp_path.joinpath("run.pkl")
    _vbmc().save(saved)
    caplog.set_level(logging.WARNING)
    loaded = VBMC.load(saved, new_options={name: value})
    assert loaded.options[name] == value
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        name in message and "no effect" in message for message in messages
    )


def _warnings_naming(caplog, name):
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING and name in record.getMessage()
    ]


def test_load_warns_of_an_option_that_has_no_effect(tmp_path, caplog):
    """A value that ``load`` is given for an option without effect is named
    as having none, as construction names it, and the option joins those
    the user set. A repeated default is silent, and so is any value of an
    option whose default is a function, which cannot be told apart from
    the default by value."""
    saved = tmp_path.joinpath("run.pkl")
    _vbmc().save(saved)
    caplog.set_level(logging.WARNING)

    loaded = VBMC.load(saved, new_options={"double_gp": True})
    warned = _warnings_naming(caplog, "double_gp")
    assert len(warned) == 1 and "no effect" in warned[0]
    assert "double_gp" in loaded.options["useroptions"]

    caplog.clear()
    VBMC.load(saved, new_options={"double_gp": False})
    VBMC.load(
        saved, new_options={"annealed_gp_mean": lambda N, NMAX: 5 * N / NMAX}
    )
    assert _warnings_naming(caplog, "double_gp") == []
    assert _warnings_naming(caplog, "annealed_gp_mean") == []


def test_load_warns_only_of_the_options_it_is_given(tmp_path, caplog):
    """A value given at construction was named there; loading the run
    with other options does not name it again."""
    saved = tmp_path.joinpath("run.pkl")
    caplog.set_level(logging.WARNING)
    _vbmc(options={"double_gp": True}).save(saved)
    assert len(_warnings_naming(caplog, "double_gp")) == 1
    caplog.clear()
    VBMC.load(saved, new_options={"max_iter": 9})
    assert _warnings_naming(caplog, "double_gp") == []


def _noisy_log_joint(x):
    return -0.5 * np.sum(x**2), 1.0


def _vbmc_with_noise(options):
    """A run whose target returns its noise when the options say so."""
    noisy = options.get("specify_target_noise", False)
    return VBMC(
        _noisy_log_joint if noisy else log_joint,
        np.zeros((1, D)),
        np.full((1, D), -10.0),
        np.full((1, D), 10.0),
        np.full((1, D), -1.0),
        np.full((1, D), 1.0),
        options=options,
    )


@pytest.mark.parametrize("noise_size", [0.1, np.float64(0.1)])
def test_noise_size_with_specify_target_noise_warns(caplog, noise_size):
    """``misc/setupoptions_vbmc.m:139-140`` warns that ``NoiseSize`` is
    ignored when ``SpecifyTargetNoise`` is active: the target returns the
    noise of each evaluation, and the GP noise model does not read
    ``noise_size``."""
    caplog.set_level(logging.WARNING)
    _vbmc_with_noise({"specify_target_noise": True, "noise_size": noise_size})
    warned = _warnings_naming(caplog, "noise_size")
    assert len(warned) == 1
    assert "no effect" in warned[0] and "specify_target_noise" in warned[0]


@pytest.mark.parametrize(
    "options",
    [
        {"specify_target_noise": True},
        {"specify_target_noise": True, "noise_size": []},
        {"specify_target_noise": True, "noise_size": None},
        {"uncertainty_handling": True, "noise_size": 0.1},
        {"noise_size": 0.1},
    ],
    ids=["default", "empty", "none", "level_1", "level_0"],
)
def test_noise_size_is_silent_where_it_is_read_or_empty(caplog, options):
    """An empty ``noise_size`` states nothing, and without
    ``specify_target_noise`` the GP noise model reads the value."""
    caplog.set_level(logging.WARNING)
    _vbmc_with_noise(options)
    assert _warnings_naming(caplog, "noise_size") == []


def test_load_warns_of_noise_size_with_specify_target_noise(tmp_path, caplog):
    """A ``noise_size`` given to ``load`` for a run whose target returns its
    own noise estimates is named as having no effect, as construction names
    it; for a run that infers its noise it is read and is silent."""
    noisy = tmp_path.joinpath("noisy.pkl")
    _vbmc_with_noise({"specify_target_noise": True}).save(noisy)
    inferred = tmp_path.joinpath("inferred.pkl")
    _vbmc_with_noise({"uncertainty_handling": True}).save(inferred)
    caplog.set_level(logging.WARNING)

    loaded = VBMC.load(noisy, new_options={"noise_size": 0.1})
    assert loaded.options["noise_size"] == 0.1
    warned = _warnings_naming(caplog, "noise_size")
    assert len(warned) == 1 and "no effect" in warned[0]

    caplog.clear()
    VBMC.load(noisy, new_options={"max_iter": 9})
    VBMC.load(inferred, new_options={"noise_size": 0.1})
    assert _warnings_naming(caplog, "noise_size") == []


def test_load_refuses_a_gp_mean_fun_construction_refuses(tmp_path):
    """A value that construction refuses is refused by ``load`` with the
    same message, before the name of the option is weighed: the check of
    ``gp_mean_fun`` is run by both routes."""
    with pytest.raises(ValueError) as at_construction:
        _vbmc(options={"gp_mean_fun": "nonsense"})
    saved = tmp_path.joinpath("run.pkl")
    _vbmc().save(saved)
    with pytest.raises(ValueError) as at_load:
        VBMC.load(saved, new_options={"gp_mean_fun": "nonsense"})
    assert at_load.value.args[0] == at_construction.value.args[0]
    assert "vbmc:UnknownGPmean" in at_load.value.args[0]


def test_load_refuses_an_integer_vars_construction_refuses(tmp_path):
    """The form of ``integer_vars`` is checked by both routes as well."""
    with pytest.raises(ValueError) as at_construction:
        _vbmc(options={"integer_vars": np.array([1, 0])})
    saved = tmp_path.joinpath("run.pkl")
    _vbmc().save(saved)
    with pytest.raises(ValueError) as at_load:
        VBMC.load(saved, new_options={"integer_vars": np.array([1, 0])})
    assert at_load.value.args[0] == at_construction.value.args[0]
    assert "integer_vars" in at_load.value.args[0]


@pytest.mark.parametrize(
    "stored, mask",
    [
        (np.zeros(2), [False, False]),
        (np.array([1, 0]), [True, False]),
        (np.array([1.0, 0.0]), [True, False]),
        ([1], [True, True]),
    ],
    ids=["zeros", "int", "float", "list"],
)
def test_load_reads_the_integer_vars_mask_of_release_1_0_4(
    tmp_path, stored, mask
):
    """Release 1.0.4 read ``integer_vars`` through ``integer_vars != 0``,
    so a run it saved can store a form that is refused when it is given,
    such as an array of zeros, an array of one zero or one per variable,
    which reads both as a mask and as a list of indices, or an array of
    floats; or a form that is read otherwise, such as the list ``[1]``,
    which 1.0.4 took for every variable and which reads as the index of
    the second. ``load`` gives the option the mask the run was made with,
    which the run's state holds, so the file opens and the option and the
    state agree."""
    lower = np.where(mask, -5.5, -5.0)
    vbmc = VBMC(
        log_joint,
        np.array([[0.0, 0.0]]),
        lower[None, :],
        -lower[None, :],
        np.array([[-2.5, -2.0]]),
        np.array([[2.5, 2.0]]),
        options={"integer_vars": np.array(mask)},
    )
    vbmc.options.__setitem__("integer_vars", stored, force=True)
    saved = tmp_path.joinpath("run.pkl")
    vbmc.save(saved)

    loaded = VBMC.load(saved)

    assert np.array_equal(loaded.options["integer_vars"], mask)
    assert loaded.options["integer_vars"].dtype == bool
    assert np.array_equal(loaded.optim_state["integer_vars"], mask)


@pytest.mark.parametrize(
    "options",
    [
        {"acq_hedge": True},
        {"search_cache_frac": 0.5},
        {"search_cache_frac": "a quarter"},
        {"search_optimizer": "Nelder-Mead"},
        {"cache_frac": -0.1},
        {"warp_cov_reg": True},
    ],
)
def test_a_refused_value_says_how_a_saved_run_carrying_it_is_loaded(options):
    """A value that construction refuses is refused by ``load`` too, so
    the refusal says which argument of ``load`` replaces it."""
    with pytest.raises((ValueError, NotImplementedError)) as execinfo:
        _vbmc(options=options)
    message = execinfo.value.args[0]
    assert "VBMC.load(file, new_options=" in message


def test_a_saved_run_that_carries_noise_shaping_is_loaded_as_the_refusal_says(
    tmp_path,
):
    """Release 1.0.4 took ``noise_shaping = True``, which is now refused at
    construction and when a saved run that carries it is loaded. The
    refusal names the argument of ``load`` that replaces the value, and
    the run loads with it."""
    vbmc = _vbmc()
    vbmc.options.__setitem__("noise_shaping", True, force=True)
    saved = tmp_path.joinpath("run.pkl")
    vbmc.save(saved)
    remedy = "VBMC.load(file, new_options={'noise_shaping': False})"

    with pytest.raises(NotImplementedError) as at_construction:
        _vbmc(options={"noise_shaping": True})
    assert remedy in at_construction.value.args[0]
    with pytest.raises(NotImplementedError) as at_load:
        VBMC.load(saved)
    assert remedy in at_load.value.args[0]

    loaded = VBMC.load(saved, new_options={"noise_shaping": False})
    assert loaded.options["noise_shaping"] is False


@pytest.mark.parametrize("value", ["Nelder-Mead", "bounded", "fmincon", ""])
def test_a_search_optimizer_outside_the_two_values_is_refused(value):
    """``search_optimizer`` names one of the two local searches of the
    acquisition; every other value is refused, the internal ``"bounded"``
    of the one-dimensional search among them."""
    with pytest.raises(ValueError) as execinfo:
        _vbmc(options={"search_optimizer": value})
    message = execinfo.value.args[0]
    assert "search_optimizer" in message
    assert "'cmaes'" in message and "'none'" in message


@pytest.mark.parametrize("value", ["cmaes", "none"])
def test_the_two_search_optimizers_are_accepted(value):
    assert (
        _vbmc(options={"search_optimizer": value}).options["search_optimizer"]
        == value
    )


def _saved_carrying_nelder_mead(tmp_path, dimension, name):
    """A file whose stored options hold the value that release 1.0.4 wrote
    into every one-dimensional run."""
    vbmc = _vbmc_of_dimension(dimension)
    vbmc.options.__setitem__("search_optimizer", "Nelder-Mead", force=True)
    saved = tmp_path.joinpath(name)
    vbmc.save(saved)
    return saved


def test_a_one_dimensional_run_that_stored_nelder_mead_loads(tmp_path):
    """With one variable the option has no effect, the acquisition being
    searched by a bounded scalar method, so the stored value stands for the
    default and the file loads with it."""
    saved = _saved_carrying_nelder_mead(tmp_path, 1, "one.pkl")
    assert VBMC.load(saved).options["search_optimizer"] == "cmaes"


def test_a_wider_run_that_stored_nelder_mead_is_refused(tmp_path):
    """With more than one variable the value was the caller's own, and it
    named a search PyVBMC does not run."""
    saved = _saved_carrying_nelder_mead(tmp_path, 2, "two.pkl")
    with pytest.raises(ValueError) as execinfo:
        VBMC.load(saved)
    assert "new_options" in execinfo.value.args[0]


def test_a_stored_nelder_mead_that_new_options_replaces_loads(tmp_path):
    saved = _saved_carrying_nelder_mead(tmp_path, 2, "two.pkl")
    vbmc = VBMC.load(saved, new_options={"search_optimizer": "cmaes"})
    assert vbmc.options["search_optimizer"] == "cmaes"


SEARCH_FRACTIONS = (
    "search_cache_frac",
    "heavy_tail_search_frac",
    "mvn_search_frac",
    "hpd_search_frac",
    "box_search_frac",
)


def test_search_fractions_that_claim_more_than_the_whole_are_refused():
    """The five fractions divide the candidates of the acquisition search
    among their sources, and the share they leave is drawn from the
    variational posterior, so together they cannot claim more than the
    whole. With the shipped fractions the search set is complete from the
    second step of an iteration on, and a share too large used to raise
    there, one iteration into the run."""
    with pytest.raises(ValueError) as execinfo:
        _vbmc(options={"search_cache_frac": 0.5})
    message = execinfo.value.args[0]
    for name in SEARCH_FRACTIONS:
        assert name in message
    assert "1.25" in message


def test_a_search_fraction_outside_the_unit_interval_is_refused():
    with pytest.raises(ValueError) as execinfo:
        _vbmc(options={"mvn_search_frac": -0.25})
    message = execinfo.value.args[0]
    assert "mvn_search_frac = -0.25" in message
    for name in SEARCH_FRACTIONS:
        assert name in message


@pytest.mark.parametrize("value", [-0.1, 1.5, np.nan, "half"])
def test_a_starting_cache_share_outside_the_unit_interval_is_refused(value):
    """``cache_frac`` is the share of the whole search set that the starting
    cache gives. A negative one counts the rows to take from the end of the
    cache, and one above one asks for more candidates than the search has."""
    with pytest.raises(ValueError) as execinfo:
        _vbmc(options={"cache_frac": value})
    message = execinfo.value.args[0]
    assert "cache_frac" in message
    assert repr(value) in message


@pytest.mark.parametrize("value", [0, 0.5, 1])
def test_a_starting_cache_share_in_the_unit_interval_is_taken(value):
    """The ends of the interval are shares like any other, and the share
    stands beside the five fractions without entering their sum."""
    vbmc = _vbmc(options={"cache_frac": value, "search_cache_frac": 0.25})
    assert vbmc.options["cache_frac"] == value


def test_the_shipped_search_fractions_leave_a_quarter_unclaimed():
    """0.25 each for the heavy-tailed, multivariate-normal and box shares,
    none for the high-posterior-density share: a quarter of the search set
    is left, which a search cache can be given."""
    vbmc = _vbmc(options={"search_cache_frac": 0.25})
    assert vbmc.options["search_cache_frac"] == 0.25
    assert sum(vbmc.options[name] for name in SEARCH_FRACTIONS) == 1


@pytest.mark.parametrize(
    "value",
    [None, "0.5", 0.5 + 0.1j, True, np.True_, np.nan, np.inf, [0.5]],
    ids=[
        "None",
        "str",
        "complex",
        "bool",
        "numpy-bool",
        "nan",
        "inf",
        "list",
    ],
)
def test_a_warp_cov_reg_that_is_not_a_finite_number_is_refused(value):
    """``warp_cov_reg`` weighs the regularization of the covariance of the
    warp towards its diagonal: a finite real number or a function of the
    number of training points. Any other value is refused at construction,
    naming the option, where it would otherwise fail, or be read as a
    number it is not, at the first warp, after the warm-up."""
    with pytest.raises(ValueError) as execinfo:
        _vbmc(options={"warp_cov_reg": value})
    assert "warp_cov_reg" in execinfo.value.args[0]


@pytest.mark.parametrize(
    "value",
    [
        0,
        0.5,
        np.float64(0.5),
        np.float32(0.5),
        np.int64(1),
        np.array(0.5),
        -1.0,
        2,
        lambda N: 0.5,
    ],
    ids=[
        "int",
        "float",
        "float64",
        "float32",
        "int64",
        "0-d-array",
        "below-zero",
        "above-one",
        "function",
    ],
)
def test_a_warp_cov_reg_that_is_a_finite_number_or_a_function_is_taken(
    value,
):
    """A number outside ``[0, 1]`` is taken as well: the warp clamps the
    weight into the interval, as MATLAB VBMC does."""
    vbmc = _vbmc(options={"warp_cov_reg": value})
    assert vbmc.options["warp_cov_reg"] is value


@pytest.mark.parametrize(
    "value",
    [0, 0.04, 0.1, -0.5, 1.5, np.nan, True, "0.8", None],
    ids=[
        "zero",
        "empty-subset",
        "one-point",
        "negative",
        "above-one",
        "nan",
        "bool",
        "str",
        "None",
    ],
)
def test_an_hpd_frac_that_leaves_too_few_points_is_refused(value):
    """``hpd_frac`` is the fraction of the training inputs, those of
    highest density, from which the bounds of the GP hyperparameters are
    set. Of the initial design of ten points, 0.04 leaves none and 0.1
    leaves one, from which no bound can be set, and the first GP fit
    fails. Such a value, and one that is not a fraction, is refused at
    construction, naming the option."""
    with pytest.raises(ValueError) as execinfo:
        _vbmc(options={"hpd_frac": value})
    assert "hpd_frac" in execinfo.value.args[0]


@pytest.mark.parametrize("value", [0.15, 0.8, 1, np.float64(0.5)])
def test_an_hpd_frac_that_leaves_two_points_or_more_is_taken(value):
    """Two points of the initial design are enough for the GP fit."""
    vbmc = _vbmc(options={"hpd_frac": value})
    assert vbmc.options["hpd_frac"] is value
