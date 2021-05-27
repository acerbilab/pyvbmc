from vbmc import Options
import numpy as np


def test_options_no_user_options():
    default_options_path = "./vbmc/option_configs/advanced_options.ini"
    options = Options(default_options_path, {"D": 2})
    assert options.get("warptolreliability") == 3
    assert len(options.get("useroptions")) == 0

def test_options_user_options():
    default_options_path = "./vbmc/option_configs/advanced_options.ini"
    user_options = {"display": "off"}
    options = Options(default_options_path, {"D": 2}, user_options)
    assert options.get("display") == "off"
    assert options.get("warptolreliability") == 3
    assert len(options.get("useroptions")) == 1
    assert "display" in options.get("useroptions")