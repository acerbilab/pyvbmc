from vbmc import (
    OptionsVBMC,
    get_default_options_advanced,
    get_default_options_fixed,
)
import numpy as np


def test_vbmc_advanced_options():
    D = 2
    options = OptionsVBMC(D, 2)
    assert set(get_default_options_advanced(D)) < set(options)
    assert options.get("FunEvalStart") == 10
    assert options.get("TolsKL") == 0.01 * np.sqrt(D)


def test_vbmc_fixed_options():
    options = OptionsVBMC(2, 2)
    assert set(get_default_options_fixed()) < set(options)
    assert options.get("FeatureTest") == False

def test_vbmc_basic_options():
    options = OptionsVBMC(2, 2)
    assert options.get("Display") == "iter"
    assert "Plot" in options
    assert "MaxIter" in options
    assert "MaxFunEvals" in options
    assert "FunEvalsPerIter" in options 
    assert "TolStableCount" in options
    assert "RetryMaxFunEvals" in options
    assert "MinFinalComponents" in options
    assert "SpecifyTargetNoise" in options