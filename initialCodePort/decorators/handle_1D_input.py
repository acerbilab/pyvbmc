import numpy as np
from functools import wraps

def handle_1D_input(kwarg: str, argpos: int, return_scalar=False):
    """
    handle_1D_input a decorator to handle 1D inputs

    Parameters
    ----------
    kwarg : str
        name of the keyword argument that should be handeled
    argpos : int
        position of the argument that should be handeled
    return_scalar : bool, optional
        if 1D the function should return a scalar, by default False
    """

    def decorator(function):
        @wraps(function)
        def wrapper(self, *args, **kwargs):

            if kwarg in kwargs:
                # for keyword arguments
                input_array = kwargs.get(kwarg)
                input_dims = np.ndim(input_array)
                if input_dims == 1:
                    kwargs[kwarg] = np.reshape(
                        input_array, (1, input_array.shape[0])
                    )

            elif len(args) > 0:
                # for positional arguments
                input_array = args[argpos]
                input_dims = np.ndim(input_array)
                if input_dims == 1:
                    arg_list = list(args)
                    arg_list[argpos] = np.reshape(
                        input_array, (1, input_array.shape[0])
                    )
                    args = tuple(arg_list)

            res = function(self, *args, **kwargs)

            # return value 1D or scalar when boolean set
            if input_dims == 1:
                # handle functions with multiple return values
                if type(res) is tuple:
                    returnvalues = list(res)
                    returnvalues = [o.flatten() for o in returnvalues]
                    if return_scalar:
                        returnvalues = [o[0] for o in returnvalues]
                    return tuple(returnvalues)

                elif return_scalar and np.ndim(res) != 0:
                    return res.flatten()[0]
                elif np.ndim(res) != 0:
                    return res.flatten()

            return res

        return wrapper

    return decorator