from vbmc import Options
import numpy as np


def test_options_no_user_options():
    default_options = {"FunEvalsPerIter": 5, "Display": "iter"}
    options = Options(default_options)
    assert options.get("Display") == "iter"
    assert options.get("FunEvalsPerIter") == 5
    assert len(options.get("UserOptions")) == 0

def test_options_user_options():
    default_options = {"FunEvalsPerIter": 5, "Display": "iter"}
    user_options = {"Display": "off"}
    options = Options(default_options, user_options)
    assert options.get("Display") == "off"
    assert options.get("FunEvalsPerIter") == 5
    assert len(options.get("UserOptions")) == 1
    assert "Display" in options.get("UserOptions")