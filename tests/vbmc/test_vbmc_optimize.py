import numpy as np

import scipy as sp
import scipy.stats

from pyvbmc.vbmc import VBMC


def _test_vbmc_optimize_rosenbrock():
    D = 2
    def llfun(x):
        if x.ndim == 2:
            return -np.sum((x[0, :-1]**2.0 - x[0, 1:])**2.0 + (x[0, :-1] - 1)**2.0 / 100) 
        else:
            return -np.sum((x[:-1]**2.0 - x[1:])**2.0 + (x[:-1] - 1)**2.0 / 100) 
        
    prior_mu = np.zeros((1, D)) 
    prior_var = 3**2 * np.ones((1, D))
    lpriorfun = lambda x: -0.5*(np.sum((x-prior_mu)**2 / prior_var) + np.log(np.prod(2*np.pi*prior_var)))
        
    f = lambda x: llfun(x) + lpriorfun(x)
    plb = prior_mu - 3*np.sqrt(prior_var)
    pub = prior_mu + 3*np.sqrt(prior_var)
    x0 = prior_mu.copy()

    vbmc = VBMC(f, x0, None, None, plb, pub)
    vbmc.optimize()
    
def run_optim_block(f, x0, lb, ub, plb, pub, ln_Z, mu_bar, options=None, noise_flag=False):
    if options is None:
        options = {}
        
    # options["maxfunevals"] = 100
    options["plot"] = False
    if noise_flag:
        options["specifytargetnoise"] = True
        
    vbmc = VBMC(f, x0, lb, ub, plb, pub, user_options=options)
    vp, elbo, _ = vbmc.optimize()
    
    vmu = vp.moments()
    err_1 = np.sqrt(np.mean((vmu-mu_bar)**2))
    err_2 = np.abs(elbo-ln_Z)
    
    return err_1, err_2
        
def test_vbmc_multivariate_normal():
    D = 6
    x0 = -np.ones((1, D))
    # Be careful about -2 and -2.0!
    plb = np.full((1, D), -2.0*D)
    pub = np.full((1, D), 2.0*D)
    lb = np.full((1, D), -np.inf)
    ub = np.full((1, D), np.inf)
    lnZ = 0
    mu_bar = np.zeros((1, D))
    f = lambda x : np.sum(-0.5*(x/np.array(range(1, np.size(x)+1)))**2) - np.sum(np.log(np.array(range(1, np.size(x)+1)))) - 0.5*np.size(x)*np.log(2*np.pi)

    err_1, err_2 = run_optim_block(f, x0, lb, ub, plb, pub, lnZ, mu_bar)
    
    assert err_1 < 0.5
    assert err_2 < 0.5
  
def test_vbmc_multivariate_half_normal():
    D = 2
    x0 = -np.ones((1, D))
    plb = np.full((1, D), -6.0)
    pub = np.full((1, D), -0.05)
    lb = np.full((1, D), -D*10.0)
    ub = np.full((1, D), 0.0)
    lnZ = -D * np.log(2)
    mu_bar = -2/np.sqrt(2*np.pi)*np.array(range(1, D+1))
    f = lambda x : np.sum(-0.5*(x/np.array(range(1, np.size(x)+1)))**2) - np.sum(np.log(np.array(range(1, np.size(x)+1)))) - 0.5*np.size(x)*np.log(2*np.pi)

    err_1, err_2 = run_optim_block(f, x0, lb, ub, plb, pub, lnZ, mu_bar)
    
    assert err_1 < 0.5
    assert err_2 < 0.5
    
def test_vbmc_correlated_multivariate_normal():
    D = 3
    x0 = 0.5*np.ones((1, D))
    plb = np.full((1, D), -1.0)
    pub = np.full((1, D), 1.0)
    lb = np.full((1, D), -np.inf)
    ub = np.full((1, D), np.inf)
    lnZ = 0.0
    mu_bar = np.reshape(np.linspace(-0.5, 0.5, D), (1, -1))

    err_1, err_2 = run_optim_block(cigar, x0, lb, ub, plb, pub, lnZ, mu_bar)
    
    assert err_1 < 0.5
    assert err_2 < 0.5
    
def test_vbmc_correlated_multivariate_normal_2():
    D = 3
    x0 = 0.5*np.ones((1, D))
    plb = np.full((1, D), -1.0)
    pub = np.full((1, D), 1.0)
    lb = np.full((1, D), -4.0)
    ub = np.full((1, D), 4.0)
    lnZ = 0.0
    mu_bar = np.reshape(np.linspace(-0.5, 0.5, D), (1, -1))

    err_1, err_2 = run_optim_block(cigar, x0, lb, ub, plb, pub, lnZ, mu_bar)
    
    assert err_1 < 0.5
    assert err_2 < 0.5
      
def test_vbmc_uniform():
    D = 1
    x0 = 0.5*np.ones((1, D))
    plb = np.full((1, D), 0.05)
    pub = np.full((1, D), 0.95)
    lb = np.full((1, D), 0.0)
    ub = np.full((1, D), 1.0)
    lnZ = 0
    mu_bar = 0.5*np.ones((1, D))
    f = lambda x : 0

    err_1, err_2 = run_optim_block(f, x0, lb, ub, plb, pub, lnZ, mu_bar)
    
    assert err_1 < 0.5
    assert err_2 < 0.5
    
def cigar(x):
    """
    Benchmark log pdf -- cigar density.
    """
    
    if x.ndim == 1:
        x = np.reshape(x, (1, -1))

    D = np.size(x)
    mean = np.reshape(np.linspace(-0.5, 0.5, D), (1, -1))
    if D == 1:
        R = 10.0
    elif D == 2:
        R = np.array([[0.438952107785021, -0.898510460190134], 
                      [0.898510460190134, 0.438952107785021]])
    elif D == 3:
        R = np.array([[-0.624318398571926, -0.0583529832968072, -0.778987462379818],
                      [0.779779849779334, 0.0129117551612018, -0.625920659873738], 
                      [0.0465824331986329, -0.998212510399975, 0.0374414342443664]])
    elif D == 4:
        R = np.array([[0.530738877213611, -0.332421458771, -0.617324087669642, 0.476154584925358],
                      [-0.455283846255008, -0.578972039590549, 0.36136334497906, 0.57177314523957],
                      [-0.340852417262338, -0.587449365484418, -0.433532840927373, -0.592260203353068],
                      [-0.628372893431681, 0.457373582657411, -0.548066400653999, 0.309160368031855]])
    elif D == 5:
        R = np.array([[-0.435764067736038, 0.484423029373161, 0.0201157836447536, -0.195133090987468, 0.732777208934001],
                      [-0.611399063990897, -0.741129629736756, -0.0989871013229956, 0.187328571370887, 0.178962612288509],
                      [-0.0340226717227732, 0.234931965418636, -0.886686220684869, 0.394077369749064, -0.0462601570752127], 
                      [0.590564625513463, -0.400973629067102, -0.304445169104938, -0.33203681171552, 0.536207298122059],
                      [0.293899205038366, 0.00939792298771627, 0.333012903768423, 0.813194727889496, 0.375967653898291]])
    elif D == 6:
        R = np.array([[-0.254015072891056, -0.0684032463717124, -0.693077090686521, 0.249685438636409, 0.362364372413356, -0.506745230203393],
                      [-0.207777316284753, 0.369766206365964, 0.57903494069884, -0.0653147667578752, 0.122468089523108, -0.682316367390556],
                      [-0.328071435400004, 0.364091738763865, -0.166363836395589, 0.380087224558118, -0.766382695507289, -0.0179075049327736],
                      [-0.867667731196277, -0.0332627076729128, 0.069771482765022, -0.25333031036676, 0.206274179664928, 0.366678275010647],
                      [-0.0206482741639122, -0.229074515307431, -0.237811602709101, -0.777821502080957, -0.426607324185607, -0.321782626441153],
                      [-0.177201636197285, -0.820030251267824, 0.308647597698151, 0.346038046252171, -0.204470893859678, -0.198332931405751]])
    else:
        raise Exception("Unsupported dimension!")

    ell = np.ones((D,)) / 100
    ell[-1] = 1
    cov = np.dot(R.T, np.dot(np.diag(ell**2), R))

    y = mvnlogpdf(x, mean, cov) # + mvnlogpdf(x, prior_mean, prior_cov)
    s = 0
    return y
    
def mvnlogpdf(x, mu, sigma):
    d = np.size(x)
    X0 = x - mu
    # print(X0)
    
    # Make sure sigma is a valid covariance matrix.
    R = sp.linalg.cholesky(sigma)
    # print(R.shape)

    # Create array of standardized data, and compute log(sqrt(det(Sigma)))
    xRinv = np.linalg.solve(R.T, X0.T).T
    # print(xRinv)
    log_sqrt_det_sigma = np.sum(np.log(np.diag(R)))
    # print(log_sqrt_det_sigma)

    # The quadratic form is the inner product of the standardized data.
    quad_form = np.sum(xRinv**2)
    # print(quad_form) 

    y = -0.5*quad_form - log_sqrt_det_sigma - d * np.log(2*np.pi)/2
    return y
