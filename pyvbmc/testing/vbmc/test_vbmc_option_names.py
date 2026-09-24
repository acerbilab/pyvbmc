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
        ({"log_file_level": None}, ValueError, "log_file_level"),
        (
            {"bounded_transform": np.str_("probit")},
            ValueError,
            "bounded transform",
        ),
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


# Options that only construction reads, of each kind: booleans, a number,
# strings and arrays, one of them holding NaN.
_BUILT_WITH = {
    "warmup": False,
    "uncertainty_handling": True,
    "k_warmup": 3,
    "gp_mean_fun": "const",
    "bounded_transform": "logit",
    "f_vals": np.array([np.nan]),
    "integer_vars": np.array([True, False]),
}


def _vbmc_built_with(options):
    """A run whose first variable takes integer values, which puts its hard
    bounds at half-integers."""
    return VBMC(
        log_joint,
        np.zeros((1, D)),
        np.array([[-10.5, -10.0]]),
        np.array([[10.5, 10.0]]),
        np.full((1, D), -1.0),
        np.full((1, D), 1.0),
        options=options,
    )


def _same_stored_values(loaded, built):
    for name, value in built.options.items():
        if name in _BUILT_WITH:
            assert type(loaded.options[name]) is type(value), name
            assert np.array_equal(
                np.asarray(loaded.options[name]),
                np.asarray(value),
                equal_nan=np.asarray(value).dtype.kind == "f",
            ), name


@pytest.mark.parametrize(
    "given_back",
    [
        {},
        # The same values in the other forms that construction reads alike.
        {
            "uncertainty_handling": 1,
            "k_warmup": 3.0,
            "f_vals": [np.nan],
            "integer_vars": [0],
        },
    ],
    ids=["as_built", "other_forms"],
)
def test_load_takes_back_the_options_a_run_was_built_with(
    tmp_path, given_back
):
    """The options a run was built with can be given back to ``load``
    together with a new budget: a value of an option that only
    construction reads is refused only where it differs from the one the
    run stores, read as construction reads it. The values given back
    change nothing, and the budget takes effect."""
    built = _vbmc_built_with(dict(_BUILT_WITH))
    saved = tmp_path.joinpath("run.pkl")
    built.save(saved)

    new_options = {**_BUILT_WITH, **given_back, "max_fun_evals": 321}
    loaded = VBMC.load(saved, new_options=new_options)

    assert loaded.options["max_fun_evals"] == 321
    _same_stored_values(loaded, built)


@pytest.mark.parametrize(
    "changed",
    [
        {"warmup": True},
        {"uncertainty_handling": False},
        {"k_warmup": 4},
        {"gp_mean_fun": "negquad"},
        {"f_vals": np.array([-1.0])},
        {"integer_vars": np.array([False, True])},
        {"bounded_transform": "probit"},
        {"bounded_transform": 12},
    ],
)
def test_load_refuses_a_value_other_than_the_one_the_run_was_built_with(
    tmp_path, changed
):
    """A value of an option that only construction reads that differs from
    the one the run stores is refused with the message that says to
    construct a new ``VBMC`` object, also among the other options the run
    was built with."""
    saved = tmp_path.joinpath("run.pkl")
    _vbmc_built_with(dict(_BUILT_WITH)).save(saved)

    new_options = {**_BUILT_WITH, **changed, "max_fun_evals": 321}
    with pytest.raises(ValueError) as execinfo:
        VBMC.load(saved, new_options=new_options)

    message = execinfo.value.args[0]
    assert message.startswith("VBMC.load cannot change the option ")
    assert "construct a new VBMC object" in message
    (name,) = changed
    assert repr(name) in message
    for other in _BUILT_WITH:
        if other != name:
            assert repr(other) not in message


@pytest.mark.parametrize(
    "built, given_back",
    [
        ({"bounded_transform": "probit"}, {"bounded_transform": "norminv"}),
        ({"bounded_transform": "probit"}, {"bounded_transform": 12}),
        ({"bounded_transform": "norminv"}, {"bounded_transform": 12.0}),
        ({"bounded_transform": 3}, {"bounded_transform": "logit"}),
        ({"warmup": True}, {"warmup": 2}),
        ({"warmup": False}, {"warmup": []}),
        ({"entropy_switch": True}, {"entropy_switch": 2}),
        ({"fitness_shaping": True}, {"fitness_shaping": 5}),
    ],
)
def test_load_takes_back_a_value_that_construction_reads_as_the_stored_one(
    tmp_path, built, given_back
):
    """Construction reads ``bounded_transform`` through the map of names and
    numbers of ``ParameterTransformer``, where ``"probit"``, ``"norminv"``
    and ``12`` name one transform, and ``warmup``, ``entropy_switch`` and
    ``fitness_shaping`` by their truth. A value that it reads as the stored
    one is taken back by ``load``, which keeps the stored value, together
    with a new budget."""
    saved = tmp_path.joinpath("run.pkl")
    _vbmc(options=built).save(saved)

    loaded = VBMC.load(saved, new_options={**given_back, "max_fun_evals": 321})

    assert loaded.options["max_fun_evals"] == 321
    for name, value in built.items():
        assert type(loaded.options[name]) is type(value)
        assert loaded.options[name] == value


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
    "built, stored, restated",
    [
        (
            {"uncertainty_handling": True},
            {"uncertainty_handling": [1]},
            {"uncertainty_handling": True, "specify_target_noise": False},
        ),
        (
            {"uncertainty_handling": True},
            {"uncertainty_handling": [0]},
            {"uncertainty_handling": True, "specify_target_noise": False},
        ),
        (
            {"uncertainty_handling": True},
            {"uncertainty_handling": np.array([1])},
            {"uncertainty_handling": True, "specify_target_noise": False},
        ),
        (
            {"specify_target_noise": True},
            {"specify_target_noise": "yes"},
            {"uncertainty_handling": [], "specify_target_noise": True},
        ),
        (
            {"specify_target_noise": True},
            {"specify_target_noise": [0], "uncertainty_handling": False},
            {"uncertainty_handling": True, "specify_target_noise": True},
        ),
        (
            {},
            {"specify_target_noise": None},
            {"uncertainty_handling": [], "specify_target_noise": False},
        ),
    ],
    ids=["list_one", "list_zero", "array", "string", "list_and_off", "none"],
)
def test_load_restates_the_noise_handling_of_release_1_0_4(
    tmp_path, built, stored, restated
):
    """Release 1.0.4 read ``specify_target_noise`` by its truth, and it
    turned the noise handling on for any ``uncertainty_handling`` of nonzero
    length, which it did not read when ``specify_target_noise`` was on. A
    run it saved can store forms that construction refuses, such as ``[1]``,
    or a pair that it refuses, such as ``uncertainty_handling=False`` with
    a target that returns its noise. ``load`` gives each of the two options
    the form that states the uncertainty handling level the run's state
    holds, so that the options read as the run was made and the run's own
    options can be given back with a new budget."""
    vbmc = _vbmc_with_noise(built)
    for name, value in stored.items():
        vbmc.options.__setitem__(name, value, force=True)
    level = vbmc.optim_state["uncertainty_handling_level"]
    saved = tmp_path.joinpath("run.pkl")
    vbmc.save(saved)

    loaded = VBMC.load(saved)

    for name, value in restated.items():
        assert type(loaded.options[name]) is type(value), name
        assert np.array_equal(loaded.options[name], value), name
    assert loaded.optim_state["uncertainty_handling_level"] == level
    assert loaded.options.uncertainty_handling_on() == (level > 0)
    again = VBMC.load(saved, new_options={**restated, "max_fun_evals": 321})
    assert again.options["max_fun_evals"] == 321


def test_load_refuses_the_uncertainty_handling_form_of_release_1_0_4(
    tmp_path,
):
    """The ``[1]`` that a run saved by release 1.0.4 can store is restated
    as ``True`` when the run is loaded, and given to ``load`` it is refused
    with the message construction gives for it, which names the forms the
    option takes."""
    vbmc = _vbmc_with_noise({"uncertainty_handling": True})
    vbmc.options.__setitem__("uncertainty_handling", [1], force=True)
    saved = tmp_path.joinpath("run.pkl")
    vbmc.save(saved)

    with pytest.raises(ValueError) as at_construction:
        _vbmc_with_noise({"uncertainty_handling": [1]})
    with pytest.raises(ValueError) as at_load:
        VBMC.load(saved, new_options={"uncertainty_handling": [1]})
    assert at_load.value.args[0] == at_construction.value.args[0]
    assert "True or False" in at_load.value.args[0]


@pytest.mark.parametrize(
    "built, given_back",
    [
        ({"specify_target_noise": True}, {"uncertainty_handling": True}),
        ({"specify_target_noise": True}, {"uncertainty_handling": 1}),
        (
            {"specify_target_noise": True},
            {"uncertainty_handling": True, "specify_target_noise": True},
        ),
        (
            {"specify_target_noise": True, "uncertainty_handling": True},
            {"uncertainty_handling": []},
        ),
        (
            {"specify_target_noise": True, "uncertainty_handling": True},
            {"uncertainty_handling": None},
        ),
        ({}, {"uncertainty_handling": False}),
        ({}, {"uncertainty_handling": 0, "specify_target_noise": False}),
        ({"specify_target_noise": False}, {"uncertainty_handling": False}),
    ],
    ids=[
        "true_for_empty",
        "one_for_empty",
        "both_for_empty",
        "empty_for_true",
        "none_for_true",
        "false_for_empty",
        "zero_and_false_for_empty",
        "false_for_empty_beside_false",
    ],
)
def test_load_reads_the_noise_handling_pair_as_construction_does(
    tmp_path, built, given_back
):
    """Construction reads ``uncertainty_handling`` and
    ``specify_target_noise`` together, as the uncertainty handling level
    they select, and an empty ``uncertainty_handling`` follows
    ``specify_target_noise``. A value given to ``load`` for either is read
    in the pair it forms with the stored value of the other, so a pair that
    selects the run's level is taken, together with a new budget, and
    changes nothing."""
    vbmc = _vbmc_with_noise(built)
    stored = {
        name: vbmc.options[name]
        for name in ("uncertainty_handling", "specify_target_noise")
    }
    level = vbmc.optim_state["uncertainty_handling_level"]
    saved = tmp_path.joinpath("run.pkl")
    vbmc.save(saved)

    loaded = VBMC.load(saved, new_options={**given_back, "max_fun_evals": 321})

    assert loaded.options["max_fun_evals"] == 321
    for name, value in stored.items():
        assert type(loaded.options[name]) is type(value), name
        assert np.array_equal(loaded.options[name], value), name
    assert loaded.optim_state["uncertainty_handling_level"] == level


@pytest.mark.parametrize(
    "built, changed, named",
    [
        (
            {"specify_target_noise": True},
            {"specify_target_noise": False},
            ["specify_target_noise"],
        ),
        (
            {"specify_target_noise": True},
            {"specify_target_noise": False, "uncertainty_handling": True},
            ["uncertainty_handling", "specify_target_noise"],
        ),
        (
            {"specify_target_noise": True},
            {"uncertainty_handling": False},
            ["uncertainty_handling"],
        ),
        (
            {"uncertainty_handling": True},
            {"specify_target_noise": True},
            ["specify_target_noise"],
        ),
        (
            {"uncertainty_handling": True},
            {"uncertainty_handling": []},
            ["uncertainty_handling"],
        ),
        (
            {},
            {"uncertainty_handling": True},
            ["uncertainty_handling"],
        ),
        (
            {},
            {"uncertainty_handling": [], "specify_target_noise": True},
            ["specify_target_noise"],
        ),
    ],
    ids=[
        "none_for_given",
        "inferred_for_given",
        "refused_pair",
        "given_for_inferred",
        "none_for_inferred",
        "inferred_for_none",
        "given_for_none",
    ],
)
def test_load_refuses_a_noise_handling_pair_of_another_level(
    tmp_path, built, changed, named
):
    """A value given for ``uncertainty_handling`` or
    ``specify_target_noise`` whose pair with the stored value of the other
    selects another uncertainty handling level than the run's, or a pair
    that construction refuses, is refused with the message that names the
    option given another value and says to construct a new ``VBMC``
    object."""
    saved = tmp_path.joinpath("run.pkl")
    _vbmc_with_noise(built).save(saved)

    with pytest.raises(ValueError) as execinfo:
        VBMC.load(saved, new_options={**changed, "max_fun_evals": 321})

    message = execinfo.value.args[0]
    assert message.startswith("VBMC.load cannot change the option")
    assert "construct a new VBMC object" in message
    for name in ("uncertainty_handling", "specify_target_noise"):
        assert (repr(name) in message) == (name in named), name


@pytest.mark.parametrize(
    "options",
    [
        {"acq_hedge": True},
        {"search_cache_frac": 0.5},
        {"search_cache_frac": "a quarter"},
        {"search_optimizer": "Nelder-Mead"},
        {"cache_frac": -0.1},
        {"warp_cov_reg": True},
        {"min_iter": 2.5},
        {"min_iter": np.inf},
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


_NOT_NOISE_SIZES = [
    0,
    0.0,
    -1,
    np.float64(-0.5),
    np.nan,
    np.inf,
    True,
    np.bool_(True),
    [0.1, 0.2],
    np.array([0.1]),
    "0.1",
]
_NOT_NOISE_SIZE_IDS = [
    "zero",
    "zero-float",
    "negative",
    "negative-float64",
    "nan",
    "inf",
    "bool",
    "numpy-bool",
    "list",
    "one-entry-array",
    "str",
]


@pytest.mark.parametrize("value", _NOT_NOISE_SIZES, ids=_NOT_NOISE_SIZE_IDS)
def test_a_noise_size_that_is_not_a_positive_number_is_refused(
    tmp_path, value
):
    """``noise_size``, if given, needs to be a positive number
    (``misc/setupoptions_vbmc.m:131-132``): the GP fit starts the noise
    from its logarithm. Any other value, save an empty one, is refused at
    construction and by ``load`` with the same message, which names the
    option, where the first GP fit would otherwise fail on it or replace
    it without a word."""
    with pytest.raises(ValueError) as at_construction:
        _vbmc_with_noise({"noise_size": value})
    assert "noise_size" in at_construction.value.args[0]

    saved = tmp_path.joinpath("run.pkl")
    _vbmc_with_noise({}).save(saved)
    with pytest.raises(ValueError) as at_load:
        VBMC.load(saved, new_options={"noise_size": value})
    assert at_load.value.args[0] == at_construction.value.args[0]


@pytest.mark.parametrize(
    "options",
    [{"uncertainty_handling": True}, {"specify_target_noise": True}],
    ids=["level_1", "level_2"],
)
def test_a_noise_size_is_checked_at_every_uncertainty_level(caplog, options):
    """The check does not depend on the uncertainty level. With
    ``specify_target_noise`` the value is refused, as MATLAB VBMC refuses
    it before it warns that the option has no effect
    (``misc/setupoptions_vbmc.m:131-140``), and it is not also named as a
    value that is accepted and ignored."""
    caplog.set_level(logging.WARNING)
    with pytest.raises(ValueError) as execinfo:
        _vbmc_with_noise({**options, "noise_size": 0})
    assert "noise_size" in execinfo.value.args[0]
    assert _warnings_naming(caplog, "noise_size") == []


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        np.array([]),
        0.1,
        np.float64(0.1),
        np.float32(0.1),
        2,
        np.int64(2),
        np.array(0.1),
        1e-12,
    ],
    ids=[
        "None",
        "empty-list",
        "empty-array",
        "float",
        "float64",
        "float32",
        "int",
        "int64",
        "0-d-array",
        "below-tol-gp-noise",
    ],
)
def test_a_noise_size_that_is_empty_or_a_positive_number_is_taken(
    tmp_path, value
):
    """An empty value leaves the option unset, and a positive finite
    number of any numeric type is taken, at construction and by
    ``load``."""
    vbmc = _vbmc_with_noise({"noise_size": value})
    assert vbmc.options["noise_size"] is value

    saved = tmp_path.joinpath("run.pkl")
    _vbmc_with_noise({}).save(saved)
    loaded = VBMC.load(saved, new_options={"noise_size": value})
    assert loaded.options["noise_size"] is value


@pytest.mark.parametrize(
    "options",
    [{}, {"uncertainty_handling": True}],
    ids=["level_0", "level_1"],
)
def test_a_saved_noise_size_of_zero_loads_as_the_refusal_says(
    tmp_path, options
):
    """Release 1.0.4 took a ``noise_size`` that is not positive and ran it
    as ``tol_gp_noise``. Such a value is refused when a saved run that
    carries it is loaded. The refusal leads with the argument of ``load``
    that continues the run as 1.0.4 ran it, the run's ``tol_gp_noise``,
    and names the one that leaves the option unset."""
    vbmc = _vbmc_with_noise(options)
    vbmc.options.__setitem__("noise_size", 0, force=True)
    saved = tmp_path.joinpath("run.pkl")
    vbmc.save(saved)
    tol_gp_noise = float(vbmc.options["tol_gp_noise"])
    as_released = (
        f"VBMC.load(file, new_options={{'noise_size': {tol_gp_noise!r}}})"
    )
    unset = "VBMC.load(file, new_options={'noise_size': []})"

    with pytest.raises(ValueError) as at_load:
        VBMC.load(saved)
    message = at_load.value.args[0]
    assert as_released in message and unset in message
    assert message.index(as_released) < message.index(unset)
    loaded = VBMC.load(saved, new_options={"noise_size": tol_gp_noise})
    assert loaded.options["noise_size"] == tol_gp_noise
    loaded = VBMC.load(saved, new_options={"noise_size": []})
    assert loaded.options["noise_size"] == []


@pytest.mark.parametrize("value", [np.nan, "0.1", [0.1]])
def test_a_noise_size_that_is_not_a_number_says_how_to_leave_it_unset(value):
    """A value that is not a number is refused with the argument of
    ``load`` that leaves the option unset."""
    with pytest.raises(ValueError) as execinfo:
        _vbmc_with_noise({"noise_size": value})
    message = execinfo.value.args[0]
    assert "VBMC.load(file, new_options={'noise_size': []})" in message
    assert "tol_gp_noise" not in message


_MCMC_SAMPLES = "active_importance_sampling_mcmc_samples"


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(True),
        np.nan,
        np.inf,
        "100",
        None,
        [100],
        np.array([100]),
    ],
    ids=[
        "bool",
        "numpy-bool",
        "nan",
        "inf",
        "str",
        "None",
        "list",
        "one-entry-array",
    ],
)
def test_an_importance_sample_count_that_is_not_a_finite_number_is_refused(
    tmp_path, value
):
    """``active_importance_sampling_mcmc_samples`` is a finite number, or a
    function of ``K``, ``n_vars`` and ``D`` that returns one, which both
    branches of the importance sampling of the noisy acquisitions round up.
    A value that is neither is refused at construction and by ``load``,
    whatever the acquisition function, with the same message, which names
    the option, where it would otherwise be refused at the first importance
    sampling, after the initial design has been evaluated."""
    with pytest.raises(ValueError) as at_construction:
        _vbmc(options={_MCMC_SAMPLES: value})
    assert _MCMC_SAMPLES in at_construction.value.args[0]

    saved = tmp_path.joinpath("run.pkl")
    _vbmc().save(saved)
    with pytest.raises(ValueError) as at_load:
        VBMC.load(saved, new_options={_MCMC_SAMPLES: value})
    assert at_load.value.args[0] == at_construction.value.args[0]


@pytest.mark.parametrize(
    "value",
    [
        100,
        0,
        -5,
        2.5,
        np.float64(50.0),
        np.int64(10),
        np.array(20),
        lambda K, n_vars, D: 3 * K,
    ],
    ids=[
        "int",
        "zero",
        "negative",
        "float",
        "float64",
        "int64",
        "0-d-array",
        "function",
    ],
)
def test_an_importance_sample_count_that_is_a_number_or_a_function_is_taken(
    tmp_path, value
):
    """A finite number of any numeric type is taken, zero and a negative
    one included, which leave out the MCMC step of ``AcqFcnIMIQR``, and so
    is a function, whose result is checked where it is evaluated."""
    options = {"uncertainty_handling": True, _MCMC_SAMPLES: value}
    vbmc = _vbmc_with_noise(options)
    assert vbmc.options[_MCMC_SAMPLES] is value

    saved = tmp_path.joinpath("run.pkl")
    _vbmc_with_noise({"uncertainty_handling": True}).save(saved)
    loaded = VBMC.load(saved, new_options={_MCMC_SAMPLES: value})
    assert loaded.options[_MCMC_SAMPLES] is value


def test_a_saved_importance_sample_count_of_true_loads_as_the_refusal_says(
    tmp_path,
):
    """Release 1.0.4 took ``True`` for the number of importance samples and
    read it as 1 with ``AcqFcnVIQR``. Such a value is refused when a saved
    run that carries it is loaded, and the refusal names the argument of
    ``load`` that replaces it."""
    vbmc = _vbmc_with_noise({"uncertainty_handling": True})
    vbmc.options.__setitem__(_MCMC_SAMPLES, True, force=True)
    saved = tmp_path.joinpath("run.pkl")
    vbmc.save(saved)
    remedy = f"VBMC.load(file, new_options={{'{_MCMC_SAMPLES}': 100}})"

    with pytest.raises(ValueError) as at_load:
        VBMC.load(saved)
    assert remedy in at_load.value.args[0]
    loaded = VBMC.load(saved, new_options={_MCMC_SAMPLES: 100})
    assert loaded.options[_MCMC_SAMPLES] == 100


@pytest.mark.parametrize("ns_gp_max", [80, 0])
@pytest.mark.parametrize(
    "value",
    [0, -1, 2.5, np.nan, np.inf, True, np.bool_(True), "5", None, np.array(5)],
    ids=[
        "zero",
        "negative",
        "fractional",
        "nan",
        "inf",
        "bool",
        "numpy-bool",
        "str",
        "None",
        "0-d-array",
    ],
)
def test_a_gp_sample_thin_that_is_not_a_whole_number_above_zero_is_refused(
    tmp_path, value, ns_gp_max
):
    """The GP fit keeps one hyperparameter sample in ``gp_sample_thin``, a
    whole number greater than zero, of an integer or a floating-point type,
    as the fit of gpyreg takes it. Any other value is refused at
    construction and by ``load``, whatever ``ns_gp_max``, with the same
    message, which names the option, where it would otherwise fail at the
    first GP fit, after the initial design has been evaluated."""
    with pytest.raises(ValueError) as at_construction:
        _vbmc(options={"gp_sample_thin": value, "ns_gp_max": ns_gp_max})
    assert "gp_sample_thin" in at_construction.value.args[0]

    saved = tmp_path.joinpath("run.pkl")
    _vbmc(options={"ns_gp_max": ns_gp_max}).save(saved)
    with pytest.raises(ValueError) as at_load:
        VBMC.load(saved, new_options={"gp_sample_thin": value})
    assert at_load.value.args[0] == at_construction.value.args[0]


@pytest.mark.parametrize(
    "value",
    [1, 5, 5.0, np.int64(3), np.float64(2.0)],
    ids=["one", "int", "whole-float", "int64", "whole-float64"],
)
def test_a_gp_sample_thin_that_is_a_whole_number_above_zero_is_taken(
    tmp_path, value
):
    """A whole number greater than zero is taken, of an integer or a
    floating-point type, at construction and by ``load``."""
    vbmc = _vbmc(options={"gp_sample_thin": value})
    assert vbmc.options["gp_sample_thin"] is value

    saved = tmp_path.joinpath("run.pkl")
    _vbmc().save(saved)
    loaded = VBMC.load(saved, new_options={"gp_sample_thin": value})
    assert loaded.options["gp_sample_thin"] is value


def test_a_saved_gp_sample_thin_of_true_loads_as_the_refusal_says(tmp_path):
    """Release 1.0.4 took ``True`` for ``gp_sample_thin`` and ran it as 1,
    and took any value with ``ns_gp_max=0``, which fits the GP without
    sampling. Such a value is refused when a saved run that carries it is
    loaded, and the refusal names the argument of ``load`` that replaces
    it."""
    vbmc = _vbmc(options={"ns_gp_max": 0})
    vbmc.options.__setitem__("gp_sample_thin", True, force=True)
    saved = tmp_path.joinpath("run.pkl")
    vbmc.save(saved)
    remedy = "VBMC.load(file, new_options={'gp_sample_thin': 5})"

    with pytest.raises(ValueError) as at_load:
        VBMC.load(saved)
    assert remedy in at_load.value.args[0]
    loaded = VBMC.load(saved, new_options={"gp_sample_thin": 5})
    assert loaded.options["gp_sample_thin"] == 5
