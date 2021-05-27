from vbmc import (
    OptionsVBMC,
    get_default_options_advanced,
    get_default_options_fixed,
)
import numpy as np


def test_options_advanced_options():
    D = 2
    options = OptionsVBMC(D)
    assert set(get_default_options_advanced(D)) < set(options)
    assert options.get("FunEvalStart") == 10
    assert options.get("TolsKL") == 0.01 * np.sqrt(D)
    assert len(options.get("UserOptions")) == 0


def test_options_fixed_options():
    options = OptionsVBMC(2)
    assert set(get_default_options_fixed()) < set(options)
    assert options.get("FeatureTest") == False
    assert len(options.get("UserOptions")) == 0


def test_options_basic_options():
    options = OptionsVBMC(2)
    assert options.get("Display") == "iter"
    assert "Plot" in options
    assert "MaxIter" in options
    assert "MaxFunEvals" in options
    assert "FunEvalsPerIter" in options
    assert "TolStableCount" in options
    assert "RetryMaxFunEvals" in options
    assert "MinFinalComponents" in options
    assert "SpecifyTargetNoise" in options
    assert options.get("Display") == "iter"
    assert len(options.get("UserOptions")) == 0

def test_options_set_user_options():
    D = 2
    user_options = {"Display": "off"}
    options = OptionsVBMC(D, user_options)
    assert options.get("Display") == "off"
    assert len(options.get("UserOptions")) == 1
    assert "Display" in options.get("UserOptions")