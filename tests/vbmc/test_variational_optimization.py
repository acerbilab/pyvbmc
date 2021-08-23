import numpy as np

import gpyreg as gpr

from pyvbmc.vbmc import Options
from pyvbmc.variational_posterior import VariationalPosterior
from pyvbmc.vbmc.variational_optimization import _vp_bound_loss, _soft_bound_loss, _negelcbo, _gplogjoint, update_K

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
    
def test_update_K():
    D = 2
    
    # Dummy option state and iteration history.
    optim_state = { 
        "vpK" : 2,
        "n_eff" : 10,
        "warmup": True
    }
    iteration_history = {}
    
    # Load options
    options = Options(
        "./pyvbmc/vbmc/option_configs/basic_vbmc_options.ini",
        evaluation_parameters={"D": D},
    )
    options.load_options_file(
        "./pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini",
        evaluation_parameters={"D": D},
    )

    # Check right after start up we do nothing.
    assert update_K(optim_state, iteration_history, options) == 2
    
    # Similar, but check new branch runs.
    optim_state["warmup"] = False
    optim_state["iter"] = 0
    assert update_K(optim_state, iteration_history, options) == 2
    
    # Check that nothing is done right after warmup.
    optim_state["iter"] = 5
    optim_state["recompute_var_post"] = True
    iteration_history = {
        "elbo": np.array([-0.0996, 0.0094, -0.021, -0.0280, 0.0368]),
        "elbo_sd": np.array([0.0081, 0.0043, 0.0009, 0.0011, 0.0012]),
        "warmup": np.array([True, True, True, True, False]),
        "pruned" : np.zeros((5, )),
        "rindex" : np.array([np.inf, np.inf, 0.1773, 0.1407, 0.3420])
    }
    assert update_K(optim_state, iteration_history, options) == 2
    
    # Check adding a new component.
    optim_state["iter"] = 7
    optim_state["recompute_var_post"] = True
    iteration_history = {
        "elbo": np.array([-0.0996, 0.0094, -0.021, -0.0280, 0.0368, 0.0239, 0.0021]),
        "elbo_sd": np.array([0.0081, 0.0043, 0.0009, 0.0011, 0.0012, 0.0008, 0.0000]),
        "warmup": np.array([True, True, True, True, False, False, False]),
        "pruned" : np.zeros((7, )),
        "rindex" : np.array([np.inf, np.inf, 0.1773, 0.1407, 0.3420, 0.1422, 0.1222])
    }
    assert update_K(optim_state, iteration_history, options) == 3
    
    # Check that allowing bonus works.
    optim_state["recompute_var_post"] = False
    assert update_K(optim_state, iteration_history, options) == 5
    
    # Check that if we pruned recently we don't do anything.
    iteration_history["pruned"][-1] = 1
    assert update_K(optim_state, iteration_history, options) == 2
    
def test_gplogjoint():
    D = 2
    K = 2
    vp = VariationalPosterior(D, K)
    vp.mu = np.loadtxt(open("./tests/variational_posterior/mu.dat", "rb"), delimiter=",")
    vp.eta = vp.eta.flatten()
    vp.sigma = vp.sigma.flatten()
    vp.lambd = vp.lambd.flatten()
    
    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    X = np.loadtxt(open("./tests/vbmc/X.dat", "rb"), delimiter=",")
    y = np.loadtxt(open("./tests/vbmc/y.dat", "rb"), delimiter=",").reshape((-1, 1))
    hyp = np.loadtxt(open("./tests/vbmc/hyp.dat", "rb"), delimiter=",")
    gp.update(X_new=X, y_new=y, hyp=hyp)
    
    F, dF, varF, dvarF, varss, I_sk, J_sjk = _gplogjoint(vp, gp, False, True, True, True, True)

    assert np.isclose(F, -0.461812484952867)
    assert dF is None
    assert np.isclose(varF, 6.598768992700180e-05)
    assert dvarF is None
    assert np.isclose(varss, 1.031705745662353e-04)
    
    F, dF, varF, dvarF, varss = _gplogjoint(vp, gp, True, True, True, False, False)
    matlab_dF = np.loadtxt(open("./tests/vbmc/dF_gplogjoint.dat", "rb"), delimiter=",")
    assert np.allclose(dF, matlab_dF)
    
def test_negelcbo():
    D = 2
    K = 2
    vp = VariationalPosterior(D, K)
    vp.mu = np.loadtxt(open("./tests/variational_posterior/mu.dat", "rb"), delimiter=",")

    gp = gpr.GP(
        D=D,
        covariance=gpr.covariance_functions.SquaredExponential(),
        mean=gpr.mean_functions.NegativeQuadratic(),
        noise=gpr.noise_functions.GaussianNoise(constant_add=True),
    )
    X = np.loadtxt(open("./tests/vbmc/X.dat", "rb"), delimiter=",")
    y = np.loadtxt(open("./tests/vbmc/y.dat", "rb"), delimiter=",").reshape((-1, 1))
    hyp = np.loadtxt(open("./tests/vbmc/hyp.dat", "rb"), delimiter=",")
    gp.update(X_new=X, y_new=y, hyp=hyp)
    
    options = {
        "tolconloss" : 0.01,
        "tolweight" : 1e-2,
        "weightpenalty": 0.1,
        "tollength" : 1e-6
    } 
    theta_bnd = None # vp.get_bounds(gp.X, options, K)
    theta = vp.get_parameters()
    
    F, dF, G, H, varF, dH, varGss, varG, varH, I_sk, J_sjk = _negelcbo(theta, gp, vp, 0.0, 0, False, True, theta_bnd, 0.0, True)
    
    assert np.isclose(F, 11.746298071422430)
    assert dF is None
    assert np.isclose(G, -0.461812484952867)
    assert np.isclose(H, -11.284485586469563)
    assert np.isclose(varF, 6.598768992700180e-05)
    assert dH is None
    assert np.isclose(varG, 6.598768992700180e-05)
    assert varH == 0.0
    
    F, dF, G, H, varF = _negelcbo(theta, gp, vp, 0.0, 0, True, False, theta_bnd, 0.0, False)
    matlab_dF = np.loadtxt(open("./tests/vbmc/dF.dat", "rb"), delimiter=",")

    assert np.allclose(dF, matlab_dF)
    
def test_vp_bound_loss():
    D = 2
    K = 2
    vp = VariationalPosterior(D, K)
    vp.mu = np.loadtxt(open("./tests/variational_posterior/mu.dat", "rb"), delimiter=",")
    
    options = {
        "tolconloss" : 0.01,
        "tolweight" : 1e-2,
        "weightpenalty": 0.1,
        "tollength" : 1e-6
    } 
    X = np.loadtxt(open("./tests/vbmc/X.dat", "rb"), delimiter=",")
    theta = vp.get_parameters()
    theta_bnd = vp.get_bounds(X, options, K)
    
    L, dL = _vp_bound_loss(vp, theta, theta_bnd)
    
    assert L == 0.0
    assert np.all(dL == 0.0)
    
    theta[-1] = 1.0
    L, dL = _vp_bound_loss(vp, theta, theta_bnd, tol_con=0.01)
    
    assert np.isclose(L, 178.1123635679098)
    assert np.isclose(dL[-1], 356.2247271358195)
    assert np.all(dL[:-1] == 0.0)
