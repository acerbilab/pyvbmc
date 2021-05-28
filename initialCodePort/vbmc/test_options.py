from vbmc import Options
import numpy as np


def test_options_no_user_options():
    default_options_path = "./vbmc/option_configs/test_options.ini"
    options = Options(default_options_path, {"D": 2})
    assert options.get("sgdstepsize") == 0.005
    assert len(options.get("useroptions")) == 0
    assert options.get("display") == "iter"


def test_options_user_options():
    default_options_path = "./vbmc/option_configs/test_options.ini"
    user_options = {"display": "off"}
    options = Options(default_options_path, {"D": 2}, user_options)
    assert options.get("display") == "off"
    assert options.get("sgdstepsize") == 0.005
    assert len(options.get("useroptions")) == 1
    assert "display" in options.get("useroptions")


def test_init_from_existing_options():
    default_options_path = "./vbmc/option_configs/test_options.ini"
    user_options = {"display": "off"}
    options_1 = Options(default_options_path, {"D": 2}, user_options)
    options_2 = Options.init_from_existing_options(
        default_options_path, {"D": 2}, options_1
    )
    assert options_1 == options_2
    assert len(options_1) == len(options_2)


def test_init_from_existing_options_modified():
    default_options_path = "./vbmc/option_configs/test_options.ini"
    user_options = {"display": "off"}
    options_1 = Options(default_options_path, {"D": 2}, user_options)
    options_1["sgdstepsize"] = 0.3
    options_2 = Options.init_from_existing_options(
        default_options_path, {"D": 2}, options_1
    )
    assert options_1 != options_2
    assert options_1.get("sgdstepsize") == 0.3
    assert options_2.get("sgdstepsize") == 0.005
    assert options_1.get("display") == "off"
    assert options_2.get("display") == "off"


def test_init_from_existing_options_without_user_options():
    default_options_path = "./vbmc/option_configs/test_options.ini"
    options_1 = Options(default_options_path, {"D": 2})
    options_1["sgdstepsize"] = 0.3
    options_2 = Options.init_from_existing_options(
        default_options_path, {"D": 2}, options_1
    )
    assert options_1 != options_2
    assert options_1.get("sgdstepsize") == 0.3
    assert options_2.get("sgdstepsize") == 0.005
    assert options_1.get("display") == "iter"
    assert options_2.get("display") == "iter"


def test_init_from_existing_options_without_other_options():
    default_options_path = "./vbmc/option_configs/test_options.ini"
    options_1 = Options.init_from_existing_options(
        default_options_path, {"D": 2}
    )
    options_2 = Options(default_options_path, {"D": 2})
    assert options_1 == options_2
    assert len(options_1) == len(options_2)