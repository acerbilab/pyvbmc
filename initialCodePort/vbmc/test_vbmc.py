import numpy as np
import pytest

from vbmc import VBMC

non_noisy_function = lambda x: np.sum(x + 2)

def test_vbmc():
    """
    just to test importing issues for now
    """    
    vbmc = VBMC(non_noisy_function, np.ones((2, 2)))
    assert isinstance(vbmc, VBMC)

def test_vbmc_init_no_x0_PLB_PUB():
    with pytest.raises(ValueError):
        VBMC(non_noisy_function)
