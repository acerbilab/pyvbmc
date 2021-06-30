import numpy as np
from pyvbmc.vbmc import VBMC


def create_vbmc(
    D: int,
    x0: float,
    lower_bounds: float,
    upper_bounds: float,
    plausible_lower_bounds: float,
    plausible_upper_bounds: float,
    user_options: dict = None,
):
    fun = lambda x: np.sum(x + 2)
    lb = np.ones((1, D)) * lower_bounds
    ub = np.ones((1, D)) * upper_bounds
    x0_array = np.ones((2, D)) * x0
    plb = np.ones((1, D)) * plausible_lower_bounds
    pub = np.ones((1, D)) * plausible_upper_bounds
    return VBMC(fun, x0_array, lb, ub, plb, pub, user_options)


def test_determine_best_vbmc_last_stable():
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4)
    vbmc.stats["iter"] = np.arange(0, 3)
    vbmc.stats["stable"] = np.full((3), True)
    vbmc.stats["vp"] = np.arange(0, 3)
    vbmc.stats["elbo"] = np.arange(0, 3)
    vbmc.stats["elbo_sd"] = np.arange(0, 3)
    vp, elbo, elbo_sd, idx_best = vbmc._determine_best_vp()
    assert idx_best == 2
    assert vp == 2
    assert elbo == 2
    assert elbo_sd == 2


def test_determine_best_vbmc_rankcriterion_elbo():
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4)
    n_iterations = 3000
    vbmc.stats["iter"] = np.arange(0, n_iterations)
    vbmc.stats["stable"] = np.full((n_iterations), False)
    vbmc.stats["vp"] = np.arange(0, n_iterations)
    vbmc.stats["elbo"] = np.arange(0, n_iterations)
    vbmc.stats["elbo_sd"] = np.zeros(n_iterations)
    vbmc.stats["rindex"] = np.arange(0, n_iterations)
    vp, elbo, elbo_sd, idx_best = vbmc._determine_best_vp(
        rank_citerion_flag=True
    )
    assert idx_best == n_iterations - 1
    assert vp == n_iterations - 1
    assert elbo == n_iterations - 1
    assert elbo_sd == 0


def test_determine_best_vbmc_rankcriterion_max_idx():
    vbmc = create_vbmc(3, 3, 1, 5, 2, 4)
    n_iterations = 3000
    vbmc.stats["iter"] = np.arange(0, n_iterations)
    vbmc.stats["stable"] = np.full((n_iterations), False)
    vbmc.stats["vp"] = np.arange(0, n_iterations)
    vbmc.stats["elbo"] = np.arange(0, n_iterations)
    vbmc.stats["elbo_sd"] = np.zeros(n_iterations)
    vbmc.stats["rindex"] = np.arange(0, n_iterations)
    vp, elbo, elbo_sd, idx_best = vbmc._determine_best_vp(
        rank_citerion_flag=True, max_idx=1000
    )
    assert idx_best == 1000
    assert vp == 1000
    assert elbo == 1000
    assert elbo_sd == 0