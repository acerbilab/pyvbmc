import copy
from math import ceil
from pathlib import Path

import numpy as np
import pytest

from pyvbmc import VBMC
from pyvbmc.acquisition_functions import AcqFcnLog, AcqFcnVIQR
from pyvbmc.vbmc import Options

options_path = Path(__file__).parent.parent.parent.joinpath(
    "vbmc", "option_configs"
)


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
