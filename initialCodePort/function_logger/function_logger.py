import sys
import numpy as np
from vbmc.timer_vbmc import Timer


class Function_Logger(object):
    """
    Function_Logger Evaluates a function and caches results
    """
    def __init__(self, fun, nvars: int, noise_flag: bool):
        """
        __init__

        Parameters
        ----------
        fun : callable
            The function to be logged
            FUN must take a vector input and return a scalar value and,
            optionally, the (estimated) SD of the returned value (if the
            function fun is stochastic)
        nvars : int
            number of dimensions that the function takes as input
        noise_flag : bool
            whether the function fun is stochastic
        """
        self.fun = fun
        self.nvars: int = nvars
        self.noise_flag: bool = noise_flag

        nmax = 500

        self.func_count: int = 0
        self.cache_count: int = 0
        self.x_orig = np.empty((nmax, self.nvars))
        self.y_orig = np.empty((nmax, 1))
        self.x = np.empty((nmax, self.nvars))
        self.y = np.empty((nmax, 1))

        if noise_flag:
            self.S = np.empty((nmax, 1))

        self.Xn: int = 0  # Last filled entry
        self.X_flag = np.full((nmax, 1), True, dtype=bool)
        self.y_max = float("-Inf")
        self.fun_evaltime = np.empty((nmax, 1))
        self.total_fun_evaltime = 0

    def iter(self, x, uncertaintyHandlingLevel):
        """
        iter evaluates function FUN at X and caches values

        Parameters
        ----------
        x : nd.array
            The point at which the function will be evaluated
        uncertaintyHandlingLevel : int
            uncertainty handling level
            (0: none; 1: unkown noise level; 2: user-provided noise)
        """
        # optimstate uncertaintyHandlingLevel
        self.noise_flag = uncertaintyHandlingLevel > 0
        timer = Timer()

        # missing:
        # x_orig = warpvars_vbmc(x,'inv',optimState.trinfo);
        # Convert back to original space 

        try:
            timer.start_timer("funtime")
            if self.noise_flag and uncertaintyHandlingLevel == 2:
                fval_orig, fsd = fun(x)
            else:
                fval_orig = fun(x)
                if self.noise_flag:
                    fsd = 1
                else:
                    fsd = None
            timer.stop_timer("funtime")

            # Check function value
            if (
                not np.isscalar(fval_orig)
                or not np.isfinite(fval_orig)
                or not np.isreal(fval_orig)
            ):
                sys.exit(
                    "Function_Logger:InvalidFuncValue"
                    + "The returned function value must be a finite real-valued scalar"
                    + "(returned value: "
                    + str(fval_orig)
                    + ")"
                )

            # Check returned function SD
            if noise_flag and (
                not np.isscalar(fsd)
                or not np.isfinite(fsd)
                or not np.isreal(fsd)
            ):
                sys.exit(
                    "Function_Logger:InvalidNoiseValue"
                    + "The returned estimated SD (second function output)"
                    + "must be a finite, positive real-valued scalar (returned SD: "
                    + str(fsd)
                    + ")."
                )

        except:
            sys.exit(
                "Function_Logger:FuncError "
                + "Error in executing the logged function: "
                + str(fun)
                + "with input: "
                + str(y)
            )

        self.func_count += 1
        self.total_fun_evaltime += timer.get_duration["funtime"]

        self.Xn += 1
        if self.Xn > self.x_orig.shape[0] - 1:
            self._expand_arrays()

        self.x_orig[
            self.Xn,
        ] = x_orig
        self.x[
            self.Xn,
        ] = x
        self.y_orig[self.Xn] = fval_orig
        self.y[self.Xn] = fval
        if fsd:
            self.S[self.Xn] = fsd
        self.X_flag[self.Xn] = True
        self.fun_evaltime[self.Xn] = timer.get_duration["funtime"]
        self.nevals = max(1, self.nevals[self.xn] + 1)
        self.ymax = np.amax(self.y[self.X_flag])

        optimstate.N = self.Xn
        optimstate.Neff = np.sum(self.nevals[self.X_flag])

        return fval, fsd

    def _expand_arrays(self, resize_amount=500):
        """
        _expand_arrays a private function to extend the rows of the object attribute arrays

        Parameters
        ----------
        resize_amount : int, optional
            additional rows, by default 500
        """

        self.x_orig = np.append(
            self.x_orig, np.empty((resize_amount, self.nvars)), axis=0
        )
        self.y_orig = np.append(
            self.y_orig, np.empty((resize_amount, 1)), axis=0
        )
        self.x = np.append(
            self.x, np.empty((resize_amount, self.nvars)), axis=0
        )
        self.y = np.append(self.y, np.empty((resize_amount, 1)), axis=0)

        if noise_flag:
            self.S = np.append(self.S, np.empty((resize_amount, 1)), axis=0)
        self.X_flag = np.append(
            self.X_flag, np.full((resize_amount, 1), True, dtype=bool), axis=0
        )
        self.fun_evaltime = np.append(
            self.fun_evaltime, np.empty((resize_amount, 1)), axis=0
        )

    def add(self):
        pass
