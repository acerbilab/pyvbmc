"""Module that contains a version of ADAM for function minimization."""

import math

import numpy as np


def minimize_adam(
    f: callable,
    x0: np.ndarray,
    lb: np.ndarray = None,
    ub: np.ndarray = None,
    tol_fun: float = 0.001,
    max_iter: int = 10000,
    master_min: float = 0.001,
    master_max: float = 0.1,
    master_decay: float = 200,
    use_early_stopping: bool = True,
):
    """
    Minimize a function with ADAM.

    Parameters
    ----------
    f : callable
        The function to minimize. Should return both the value of the
        function and its gradient.
    x0 : np.ndarray, shape (D,)
        Initial starting point.
    lb : np.ndarray, shape (D,), optional
        Optional lower bounds. If not given we won't have lower bounds.
    ub : np.ndarray, shape (D,), optional
        Optional upper bounds. If not givven we won't have upper bounds.
    tol_fun : float, defaults to 0.001
        ?
    max_iter : int, defaults to 10000
        Maximum number of iteration to perform.
    master_min : float, defaults to 0.001
        ?
    master_max : float, defaults to 0.1
        ?
    master_decay : float, defaults to 200
        ?
    use_early_stopping : bool, defaults to True
        Whether to complete all iterations or use early stopping
        when little improvement is made.

    Returns
    -------
    x : np.ndarray, shape (D,)
        The optimized point.
    y : float
        The value of the given function at the optimized point.
    x_tab : np.ndarray, shape (D, iterations)
        Intermediate values for ADAM
    y_tab : np.ndarray, shape (iterations,)
        Intermediate values for ADAM.
    iterations : int
        The amount of iterations we performed during the optimization.
    """
    fudge_factor = np.sqrt(np.spacing(1))
    beta_1 = 0.9
    beta_2 = 0.999
    batch_size = 20
    tol_x = 0.001
    tol_x_max = 0.1
    tol_fun_max = tol_fun * 100

    min_iter = batch_size * 2

    n_vars = np.size(x0)
    if lb is None:
        lb = np.full((n_vars,), -np.inf)
    if ub is None:
        ub = np.full((n_vars,), np.inf)

    m = 0
    v = 0
    x_tab = np.zeros((n_vars, max_iter))

    x = x0
    y_tab = np.full((max_iter,), np.nan)

    for i in range(0, max_iter):
        is_minibatch_end = math.remainder(i + 1, batch_size) == 0

        y_tab[i], grad = f(x)

        m = beta_1 * m + (1 - beta_1) * grad
        v = beta_2 * v + (1 - beta_2) * grad**2
        m_hat = m / (1 - beta_1 ** (i + 1))
        v_hat = v / (1 - beta_2 ** (i + 1))

        step_size = master_min + (master_max - master_min) * np.exp(
            -(i + 1) / master_decay
        )
        x -= step_size * m_hat / (np.sqrt(v_hat) + fudge_factor)  # update
        x = np.minimum(ub, np.maximum(lb, x))

        # Store x
        x_tab[:, i] = x

        if use_early_stopping and is_minibatch_end and i + 1 >= min_iter:
            xxp = np.linspace(
                -(batch_size - 1) / 2, (batch_size - 1) / 2, batch_size
            )
            p, V = np.polyfit(
                xxp, y_tab[i - batch_size + 1 : i + 1], 1, cov=True
            )

            # Highest power first
            slope = p[0]
            slope_err = np.sqrt(V[0, 0] + tol_fun**2)
            slope_err_max = np.sqrt(V[0, 0] + tol_fun_max**2)

            # Check random walk distance as termination condition.
            dx = np.sqrt(
                np.sum(
                    (
                        np.mean(x_tab[:, i - batch_size + 1 : i + 1], axis=1)
                        - np.mean(
                            x_tab[
                                :, i - 2 * batch_size + 1 : i + 1 - batch_size
                            ],
                            axis=1,
                        )
                    )
                    ** 2
                    / batch_size,
                    axis=0,
                )
            )

            if (dx < tol_x and np.abs(slope) < slope_err_max) or (
                np.abs(slope) < slope_err and dx < tol_x_max
            ):
                break

    x = np.mean(x_tab[:, i - batch_size + 1 : i + 1], axis=1)
    y = np.mean(y_tab[i - batch_size + 1 : i + 1])

    x_tab = x_tab[:, 0 : i + 1]
    y_tab = y_tab[0 : i + 1]

    return x, y, x_tab, y_tab, i + 1
