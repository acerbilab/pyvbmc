import copy
import logging
import re
from math import ceil
from pathlib import Path

import numpy as np
import pytest

from pyvbmc import VBMC
from pyvbmc.acquisition_functions import AcqFcnLog, AcqFcnVIQR
from pyvbmc.vbmc import Options
from pyvbmc.vbmc.options import INERT_OPTIONS

options_path = Path(__file__).parent.parent.parent.joinpath(
    "vbmc", "option_configs"
)
basic_options_path = options_path.joinpath("basic_vbmc_options.ini")
advanced_options_path = options_path.joinpath("advanced_vbmc_options.ini")


def _declared_option_names():
    """The option names declared by the two shipped ini files."""
    names = set()
    for path in (basic_options_path, advanced_options_path):
        for line in path.read_text().splitlines():
            line = line.strip()
            if line == "" or line.startswith("#") or line.startswith("["):
                continue
            names.add(line.split("=", 1)[0].strip())
    return names


def _shipped_options(user_options, D=2):
    """Build the shipped options as ``VBMC.__init__`` does."""
    options = Options(basic_options_path, {"D": D}, user_options)
    options.load_options_file(advanced_options_path, {"D": D})
    options.update_defaults()
    options.validate_option_names([basic_options_path, advanced_options_path])
    return options


def test_options_no_user_options():
    default_options_path = options_path.joinpath("test_options.ini")
    options = Options(default_options_path, {"D": 2})
    assert options.get("bar") == 40
    assert len(options.get("useroptions")) == 0
    assert options.get("foo") == "iter"
    assert options.get("fooD") == 4


def test_options_user_options():
    default_options_path = options_path.joinpath("test_options.ini")
    user_options = {"foo": "iter2"}
    options = Options(default_options_path, {"D": 2}, user_options)
    assert options.get("bar") == 40
    assert len(options.get("useroptions")) == 1
    assert options.get("foo") == "iter2"
    assert options.get("fooD") == 4
    assert "foo" in options.get("useroptions")


def test_init_from_existing_options():
    default_options_path = options_path.joinpath("test_options.ini")
    user_options = {"foo": "iter2"}
    options_1 = Options(default_options_path, {"D": 2}, user_options)
    options_2 = Options.init_from_existing_options(
        default_options_path, {"D": 2}, options_1
    )
    assert options_1 == options_2
    assert len(options_1) == len(options_2)


def test_init_from_existing_options_modified():
    default_options_path = options_path.joinpath("test_options.ini")
    user_options = {"foo": "iter2"}
    options_1 = Options(default_options_path, {"D": 2}, user_options)
    options_1["bar"] = 80
    options_2 = Options.init_from_existing_options(
        default_options_path, {"D": 2}, options_1
    )
    assert options_1 != options_2
    assert options_1.get("bar") == 80
    assert options_2.get("bar") == 40
    assert options_1.get("foo") == "iter2"
    assert options_2.get("foo") == "iter2"
    assert options_1.get("fooD") == 4
    assert options_2.get("fooD") == 4


def test_init_from_existing_options_without_user_options():
    default_options_path = options_path.joinpath("test_options.ini")
    options_1 = Options(default_options_path, {"D": 2})
    options_1["bar"] = 80
    options_2 = Options.init_from_existing_options(
        default_options_path, {"D": 2}, options_1
    )
    assert options_1 != options_2
    assert options_1.get("bar") == 80
    assert options_2.get("bar") == 40
    assert options_1.get("foo") == "iter"
    assert options_2.get("foo") == "iter"
    assert options_1.get("fooD") == 4
    assert options_2.get("fooD") == 4


def test_init_from_existing_options_without_other_options():
    default_options_path = options_path.joinpath("test_options.ini")
    options_1 = Options.init_from_existing_options(
        default_options_path, {"D": 2}
    )
    options_2 = Options(default_options_path, {"D": 2})
    assert options_1 == options_2
    assert len(options_1) == len(options_2)


def test_init_with_specify_target_noise():
    """Turning on specify_target_noise should adjust defaults."""
    D = 1
    options = {
        "specify_target_noise": True,
        "active_sample_gp_update": "foo",  # But don't touch user options!
    }
    vbmc1 = VBMC(
        lambda x, y: (x + y, y),
        np.zeros((1, D)),
        -np.ones((1, D)),
        np.ones((1, D)),
        -0.5 * np.ones((1, D)),
        0.5 * np.ones((1, D)),
    )
    vbmc2 = VBMC(
        lambda x, y: (x + y, y),
        np.zeros((1, D)),
        -np.ones((1, D)),
        np.ones((1, D)),
        -0.5 * np.ones((1, D)),
        0.5 * np.ones((1, D)),
        options=options,
    )

    # Check that default options are changed:
    assert vbmc1.options != vbmc2.options
    assert vbmc2.options["max_fun_evals"] == ceil(
        vbmc1.options["max_fun_evals"] * 1.5
    )
    assert vbmc2.options["tol_stable_count"] == ceil(
        vbmc1.options["tol_stable_count"] * 1.5
    )
    assert len(vbmc2.options["search_acq_fcn"]) == 1
    assert isinstance(vbmc2.options["search_acq_fcn"][0], AcqFcnVIQR)
    assert vbmc2.options["active_sample_vp_update"] == True

    # Check that user-specified option is unchanged:
    assert vbmc2.options["active_sample_gp_update"] == "foo"

    # Check defaults for non-noisy target:
    assert len(vbmc1.options["search_acq_fcn"]) == 1
    assert isinstance(vbmc1.options["search_acq_fcn"][0], AcqFcnLog)
    assert vbmc1.options["active_sample_vp_update"] == False
    assert vbmc1.options["active_sample_gp_update"] == False


@pytest.mark.parametrize(
    "user_options",
    [{"specify_target_noise": True}, {"uncertainty_handling": True}],
)
def test_noisy_defaults_apply_at_either_noise_level(user_options):
    """``misc/setupoptions_vbmc.m:143-163`` changes five defaults whenever
    the noise handling is on, which covers a noise level VBMC infers as
    well as one the target supplies."""
    noiseless = _shipped_options({})
    noisy = _shipped_options(user_options)
    assert noisy["max_fun_evals"] == ceil(noiseless["max_fun_evals"] * 1.5)
    assert noisy["tol_stable_count"] == ceil(
        noiseless["tol_stable_count"] * 1.5
    )
    assert noisy["active_sample_gp_update"] is True
    assert noisy["active_sample_vp_update"] is True
    assert len(noisy["search_acq_fcn"]) == 1
    assert isinstance(noisy["search_acq_fcn"][0], AcqFcnVIQR)


def test_noisy_defaults_leave_the_values_the_user_set():
    """Only a default is changed, as the ``updated`` list of
    ``misc/setupoptions_vbmc.m`` requires."""
    noisy = _shipped_options(
        {
            "uncertainty_handling": True,
            "max_fun_evals": 17,
            "search_acq_fcn": [AcqFcnLog()],
        }
    )
    assert noisy["max_fun_evals"] == 17
    assert isinstance(noisy["search_acq_fcn"][0], AcqFcnLog)
    assert noisy["active_sample_vp_update"] is True


def test_a_noiseless_run_keeps_the_noiseless_defaults():
    noiseless = _shipped_options({})
    assert noiseless["active_sample_gp_update"] is False
    assert noiseless["active_sample_vp_update"] is False
    assert isinstance(noiseless["search_acq_fcn"][0], AcqFcnLog)


@pytest.mark.parametrize(
    "D, expected",
    [(1, 10), (2, 10), (9, 10), (10, 20), (15, 20), (19, 20), (20, 30)],
)
def test_fun_eval_start_default(D, expected):
    """The initial design holds 10 evaluations up to nine dimensions and
    the next multiple of 10 above ``D`` from there on, as in MATLAB VBMC
    (``10*ceil((D+1)/10)``)."""
    vbmc = VBMC(
        lambda x: -0.5 * np.sum(x**2),
        np.zeros((1, D)),
        np.full((1, D), -np.inf),
        np.full((1, D), np.inf),
        np.full((1, D), -1.0),
        np.full((1, D), 1.0),
    )
    assert vbmc.options["fun_eval_start"] == expected


def test_inert_options_are_the_declared_options_nothing_reads():
    """``INERT_OPTIONS`` lists exactly the declared options that no module
    of the package reads, so that a newly dead option, or a newly read
    registered one, fails here."""
    package_path = options_path.parent.parent
    # The option machinery itself is not a consumer, and it spells out
    # every registered name.
    registry_parts = ("vbmc", "options.py")
    sources = []
    for path in package_path.rglob("*.py"):
        parts = path.relative_to(package_path).parts
        if "testing" in parts or "option_configs" in parts:
            continue
        if parts == registry_parts:
            continue
        sources.append(path)
    text = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    unread = {
        name
        for name in _declared_option_names()
        if re.search("['\"]" + re.escape(name) + "['\"]", text) is None
    }
    assert unread == set(INERT_OPTIONS)


def test_inert_option_away_from_its_default_warns(caplog):
    """A value supplied for an inert option is named as having no effect."""
    caplog.set_level(logging.WARNING)
    options = _shipped_options({"double_gp": True})
    assert options["double_gp"] is True
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "double_gp" in message and "no effect" in message
        for message in messages
    )


def test_inert_option_at_its_default_does_not_warn(caplog):
    """Repeating the default of an inert option changes nothing and is
    silent, so that an option dictionary recorded by an earlier run loads
    without notices."""
    caplog.set_level(logging.WARNING)
    _shipped_options({"double_gp": False, "noise_shaping_threshold": 20})
    messages = [record.getMessage() for record in caplog.records]
    assert not any(
        "double_gp" in message or "noise_shaping_threshold" in message
        for message in messages
    )


def test_every_inert_option_at_its_default_is_silent(caplog):
    """An option dictionary recorded by an earlier run repeats every
    default, including the one that is a lambda, and passes through
    without notices."""
    defaults = _shipped_options({})
    repeated = {name: defaults[name] for name in INERT_OPTIONS}
    assert callable(repeated["annealed_gp_mean"])

    caplog.set_level(logging.WARNING)
    _shipped_options(repeated)
    messages = [record.getMessage() for record in caplog.records]
    assert not any(
        name in message for name in INERT_OPTIONS for message in messages
    )


def test_option_that_is_read_does_not_warn(caplog):
    """An option the algorithm reads has an effect and is not reported."""
    caplog.set_level(logging.WARNING)
    options = _shipped_options({"max_iter": 3})
    assert options["max_iter"] == 3
    messages = [record.getMessage() for record in caplog.records]
    assert not any("max_iter" in message for message in messages)


def test__str__and__repr__():
    default_options_path = options_path.joinpath("test_options.ini")
    options = Options(default_options_path, {"D": 2})
    one_option_str = "bar: 40 (Bar description)"
    assert one_option_str in options.__repr__()
    assert "None (use default options)." in options.__str__()
    options.__repr__()


def test_del():
    default_options_path = options_path.joinpath("test_options.ini")
    options = Options(default_options_path, {"D": 2})
    options.pop("foo")
    assert "foo" not in options


def test_eval_callable():
    default_options_path = options_path.joinpath("test_options.ini")

    def bar_function(T, S):
        return S, T

    user_options = {"foo": lambda Y, K: (Y, K), "bar": bar_function}
    options = Options(default_options_path, {"D": 2}, user_options)
    assert (2, 3) == options.eval("foo", {"K": 3, "Y": 2})
    assert (2, 3) == options.eval("foo", {"Y": 2, "K": 3})
    assert (3, 2) == options.eval("bar", {"T": 2, "S": 3})
    assert (5, 10) == options.eval("bar", {"S": 5, "T": 10})


def test_eval_constant():
    default_options_path = options_path.joinpath("test_options.ini")
    user_options = {"ns_ent": (5, 3)}
    options = Options(default_options_path, {"D": 2, "Y": 3}, user_options)
    assert (5, 3) == options.eval("ns_ent", {"K": 2})


def test_eval_callable_args_missing():
    default_options_path = options_path.joinpath("test_options.ini")
    user_options = {"ns_ent": lambda Y, K: (Y, K)}
    options = Options(default_options_path, {"D": 2}, user_options)
    with pytest.raises(TypeError):
        options.eval("ns_ent", {})


def test_eval_callable_too_many_args():
    default_options_path = options_path.joinpath("test_options.ini")
    user_options = {"bar": lambda Y, K: (Y, K)}
    options = Options(default_options_path, {"D": 2}, user_options)
    with pytest.raises(TypeError):
        options.eval("bar", {"U": 2, "S": 2, "T": 4})


def test_load_options_file():
    evaluation_parameters = {"D": 2}
    user_options = {"foo": "testuseroptions", "foo2": "testuseroptions2"}
    basic_test_options = options_path.joinpath("test_options.ini")
    options = Options(basic_test_options, evaluation_parameters, user_options)
    advanced_test_options = options_path.joinpath("test_options2.ini")
    options.load_options_file(advanced_test_options, evaluation_parameters)
    assert options.get("bar") == 40
    assert len(options.get("useroptions")) == 2
    assert options.get("foo") == "testuseroptions"
    assert options.get("fooD") == 4
    assert options.get("bar2") == 80
    assert options.get("foo2") == "testuseroptions2"
    assert options.get("fooD2") == 200


def test_validate_option_names():
    evaluation_parameters = {"D": 2}
    user_options = {"foo": "testuseroptions", "foo2": "testuseroptions2"}
    basic_test_options = options_path.joinpath("test_options.ini")
    options = Options(basic_test_options, evaluation_parameters, user_options)
    advanced_test_options = options_path.joinpath("test_options2.ini")
    options.load_options_file(advanced_test_options, evaluation_parameters)
    # should go fine
    options.validate_option_names([basic_test_options, advanced_test_options])
    # raise error
    with pytest.raises(ValueError) as execinfo1:
        options.validate_option_names([basic_test_options])


def test_validate_option_names_unknown_user_options():
    evaluation_parameters = {"D": 2}
    user_options = {"failoption": "testuseroptions"}
    basic_test_options = options_path.joinpath("test_options.ini")
    options = Options(basic_test_options, evaluation_parameters, user_options)
    advanced_test_options = options_path.joinpath("test_options2.ini")
    options.load_options_file(advanced_test_options, evaluation_parameters)
    with pytest.raises(ValueError) as execinfo1:
        options.validate_option_names(
            [basic_test_options, advanced_test_options]
        )
    assert "The option failoption does not exist." in execinfo1.value.args[0]


def test_load_options_invalid_path():
    evaluation_parameters = {"D": 2}
    basic_test_options = options_path.joinpath("test_options.ini")
    options = Options(basic_test_options, evaluation_parameters)
    non_existing_path = options_path.joinpath("does_not_exist.ini")
    with pytest.raises(ValueError) as execinfo1:
        options.load_options_file(non_existing_path, evaluation_parameters)
    assert "does not exist." in execinfo1.value.args[0]


def test_options_copy():
    default_options_path = options_path.joinpath("test_options.ini")
    test_list = [1, 2, 3, 4]
    user_options = {"foo": test_list}
    options = Options(default_options_path, {"D": 2}, user_options)

    options_copy = copy.copy(options)
    # Check that we have a copy:
    assert options == options_copy
    assert options_copy.get("foo") == test_list
    # Check that the copy is not deep:
    assert options.get("foo") is options_copy.get("foo")

    assert options_copy.get("bar") == 40
    assert len(options_copy.get("useroptions")) == 1
    assert options_copy.get("fooD") == 4
    assert "foo" in options_copy.get("useroptions")


def test_options_deepcopy():
    default_options_path = options_path.joinpath("test_options.ini")
    test_list = [1, 2, 3, 4]
    user_options = {"foo": test_list}
    options = Options(default_options_path, {"D": 2}, user_options)

    options_copy = copy.deepcopy(options)
    # Check that we have a copy:
    assert options == options_copy
    assert options_copy.get("foo") == test_list
    # Check that the copy is deep:
    assert options.get("foo") is not options_copy.get("foo")

    assert options_copy.get("bar") == 40
    assert len(options_copy.get("useroptions")) == 1
    assert options_copy.get("fooD") == 4
    assert "foo" in options_copy.get("useroptions")


def test_prevent_option_set_post_init():
    evaluation_parameters = {"D": 2}
    user_options = {"foo": "testuseroptions", "foo2": "testuseroptions2"}
    basic_test_options = options_path.joinpath("test_options.ini")
    options = Options(basic_test_options, evaluation_parameters, user_options)
    advanced_test_options = options_path.joinpath("test_options2.ini")
    options.load_options_file(advanced_test_options, evaluation_parameters)
    # Validate option names to complete initialization:
    options.validate_option_names([basic_test_options, advanced_test_options])
    options.__setitem__("bar", 1, force=True)
    try:
        # Should fail with AttributeError and not AssertionError:
        options["bar"] = 2
        assert False
    except AttributeError:
        pass
    try:
        # Ditto:
        options.__setitem__("bar", 2)
        assert False
    except AttributeError:
        pass
