import numpy as np

from pyvbmc.vbmc.variational_optimization import _soft_bound_loss

def test_soft_bound_loss():
    D = 3
    x1 = np.zeros((D,))
    slb = np.full((D,), -10)
    sub = np.full((D,), 10)
    
    L1 = _soft_bound_loss(x1, slb, sub)
    assert np.isclose(L1, 0.0)
    
    L2, dL2 = _soft_bound_loss(x1, slb, sub, compute_grad=True)
    assert np.isclose(L2, 0.0)
    assert np.allclose(dL2, 0.0)
    
    x2 = np.zeros((D,))
    x2[0] = 15.0
    x2[1] = -20.0
    L3 = _soft_bound_loss(x2, slb, sub)
    assert np.isclose(L3, 156250.0)
    
    L4, dL4 = _soft_bound_loss(x2, slb, sub, compute_grad=True)
    assert np.isclose(L4, 156250.0)
    assert np.isclose(dL4[0], 12500.0)
    assert np.isclose(dL4[1], -25000.0)
    assert np.allclose(dL4[2:], 0.0)
