from functools import wraps

import numpy as np


def handle_0D_1D_input(
    patched_kwargs: [], patched_argpos: [], return_scalar=False
):
    """
    A decorator that handles 0D, 1D inputs and transforms them to 2D.

    Parameters
    ----------
    kwarg : list of str
        The names of the keyword arguments that should be handeled.
    argpos : list of int
        The positions of the arguments that should be handeled.
    return_scalar : bool, optional
        If the input is 1D the function should return a scalar,
        by default False.
    """

    def decorator(function):
        @wraps(function)
        def wrapper(self, *args, **kwargs):
            for idx, patched_kwarg in enumerate(patched_kwargs):
                if patched_kwarg in kwargs:
                    # for keyword arguments
                    input_dims = np.ndim(kwargs.get(patched_kwarg))
                    kwargs[patched_kwarg] = np.atleast_2d(
                        kwargs.get(patched_kwarg)
                    )

                elif len(args) > patched_argpos[idx]:
                    # for positional arguments
                    arg_list = list(args)
                    input_dims = np.ndim(args[patched_argpos[idx]])
                    arg_list[patched_argpos[idx]] = np.atleast_2d(
                        args[patched_argpos[idx]]
                    )
                    args = tuple(arg_list)

            res = function(self, *args, **kwargs)

            # return value 1D or scalar when boolean set
            if input_dims == 1:
                # handle functions with multiple return values
                if type(res) is tuple:
                    returnvalues = list(res)
                    returnvalues = [o.ravel() for o in returnvalues]
                    if return_scalar:
                        returnvalues = [o[0] for o in returnvalues]
                    return tuple(returnvalues)

                elif return_scalar and np.ndim(res) != 0:
                    return res.ravel()[0]
                elif np.ndim(res) != 0:
                    return res.ravel()

            return res

        return wrapper

    return decorator
