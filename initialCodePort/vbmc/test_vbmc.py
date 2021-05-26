from vbmc import VBMC



def test_vbmc():
    """
    just to test importing issues for now
    """    
    vbmc = VBMC()
    assert isinstance(vbmc, VBMC)