"""Variational optimization / training of variational posterior"""

import math
import numpy as np

def update_K(optim_state, iteration_history, options):
    """
    Update number of variational mixture components.
    """
    K_new = optim_state["vpK"]

    # Compute maximum number of components
    K_max = math.ceil(
        options.eval(
            "kfunmax", {"N": optim_state["n_eff"]}
        )
    )

    # Evaluate bonus for stable solution.
    K_bonus = round(options.eval("adaptivek", {"unkn" : K_new}))

    if not optim_state["warmup"] and optim_state["iter"] > 0:
        recent_iters = math.ceil(
            0.5
            * options["tolstablecount"]
            / options["funevalsperiter"]
        )

        # Check if ELCBO has improved wrt recent iterations
        lower_end = max(0, optim_state["iter"] - recent_iters)
        elbos = iteration_history["elbo"][lower_end:]
        elboSDs = iteration_history["elbo_sd"][lower_end:]
        elcbos = elbos - options["elcboimproweight"] * elboSDs
        warmups = iteration_history["warmup"][lower_end:]
        elcbos_after = elcbos[~warmups]
        # Ignore two iterations right after warmup.
        elcbos_after[0 : min(2, optim_state["iter"])] = -np.inf
        elcbo_max = np.max(elcbos_after)
        improving_flag = elcbos_after[-1] >= elcbo_max and np.isfinite(
            elcbos_after[-1]
        )

        # Add one component if ELCBO is improving and no pruning in last iteration
        if iteration_history["pruned"][-1] == 0 and improving_flag:
            K_new += 1

        # Bonus components for stable solution (speed up exploration)
        if (
            iteration_history["rindex"][-1] < 1
            and not optim_state["recompute_var_post"]
            and improving_flag
        ):
            # No bonus if any component was very recently pruned.
            new_lower_end = max(
                0, optim_state["iter"] - math.ceil(0.5 * recent_iters)
            )
            if np.all(
                iteration_history["pruned"][new_lower_end:] == 0
            ):
                K_new += K_bonus

        K_new = max(optim_state["vpK"], min(K_new, K_max))

    return K_new

def optimize_vp(vp, gp, Nfastopts, Nslowopts, K=None):
    """
    Optimize variational posterior.
    """
    # use interface for vp optimzation?
    if K is None:
        K = vp.K
    varss = []
    pruned = 0
    return varss, pruned
