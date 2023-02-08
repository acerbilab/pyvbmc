from pyvbmc.priors import Prior, Product, SciPy, UserFunction


def convert_to_prior(prior, sample_prior=None, D=None):
    """Convert an object to a pyvbmc Prior instance.

    Parameters
    ----------
    prior
        The object to convert to a PyVBMC ``Prior``. May be one of:

        #. a function of a single argument which returns the log-density of the
           prior given a point, and will be converted to a ``UserFunction``
           prior,
        #. a ``PyVBMC`` prior, which will remain unchanged,
        #. a frozen SciPy multivariate normal, multivariate t, or
           one-dimensional continuous distribution, which will be converted to
           a PyVBMC ``SciPy`` prior, or
        #. a list of one-dimensional continuous SciPy distributions and/or
           PyVBMC ``Prior`` objects, which will be treated as independent
           marginals of a ``Product`` prior.

    sample_prior : callable, optional
        A function of a single argument `n` which returns `n` samples from
        the prior distribution. Optional, used only if ``prior`` is a function.
    D : int, optional
        The dimension of the prior distribution. Optional, used only if
        ``prior`` is a function.
    """
    if isinstance(prior, list):
        prior = Product(prior)
    elif callable(prior):
        prior = UserFunction(prior, sample_prior, D)
    elif isinstance(prior, Prior):
        pass
    else:
        try:
            prior = SciPy(prior)
        except TypeError as err:
            raise TypeError(
                f"Optional keyword `prior` should be a subclass of `pyvbmc.priors.Prior`, an appropriate `scipy.stats` distribution, a list of these, or a function. ({err})"
            ) from err
    if sample_prior is not None and sample_prior != prior.sample:
        raise ValueError(
            "If `prior` is provided then `sample_prior` should be `None` or `prior.sample`."
        )
    return prior


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
