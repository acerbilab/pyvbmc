import numpy as np


def tile_inputs(*args, size=None, squeeze=False):
    """Tile scalar inputs to have the same dimension as array inputs.

    If all inputs are given as scalars, returned arrays will have shape `size`
    if `size` is a tuple, or shape `(size,)` if `size` is an integer.

    Parameters
    ----------
    *args : [Union[float, np.ndarray]]
        The inputs to tile.
    size : Union[int, tuple], optional
        The desired size/shape of the output, default `(1,)`.
    squeeze : bool
        If `True`, then drop 1-d axes from inputs. Default `False`.

    Raises
    ------
    ValueError
        If the non-scalar arguments do not have the same shape, or if they do
        not agree with `size`.
    """
    if type(size) == int:
        size = (size,)
    shape = None

    # Check that all non-scalar inputs have the same shape
    args = list(args)
    for i, arg in enumerate(args):
        if not (np.isscalar(arg)):
            if squeeze:
                arg = args[i] = np.atleast_1d(np.squeeze(np.array(arg)))
            else:
                arg = args[i] = np.array(arg)
            if shape is None:
                shape = arg.shape
            elif arg.shape != shape:
                raise ValueError(
                    f"All inputs should have the same shape, but found inputs with shapes {shape} and {arg.shape}."
                )

    if size is None:
        if shape is None:
            # Default to shape (1,)
            size = (1,)
        else:
            # Or use inferred shape
            size = shape

    for i, arg in enumerate(args):
        if np.isscalar(arg):
            args[i] = np.full(size, arg)
        else:
            args[i] = args[i].reshape(size)

    return args
