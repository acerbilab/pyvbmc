from vbmc import OptionsVBMC, default_options_advanced, default_options_fixed


def test_vbmc_advanced_options():  
    options = OptionsVBMC(2, 2)
    assert set(default_options_advanced) < set(options)

def test_vbmc_fixed_options(): 
    options = OptionsVBMC(2, 2)
    assert set(default_options_fixed) < set(options)
