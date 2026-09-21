import ast

import pyvbmc.acquisition_functions

from .abstract_acq_fcn import AbstractAcqFcn


def string_to_acq(string):
    """Build the acquisition function that a string names.

    The string is read as Python reads a call: the name of one of the
    acquisition function classes of :mod:`pyvbmc.acquisition_functions`,
    on its own or followed by parentheses holding literal positional and
    keyword arguments. ``"AcqFcnLog"``, ``"AcqFcnLog()"``,
    ``"AcqFcnVIQR(0.9)"`` and ``"AcqFcnVIQR(quantile=0.9,
    loss='iqr_reduction')"`` are all accepted. The arguments must be
    literals: no name among them is looked up and nothing is evaluated.

    Parameters
    ----------
    string : str
        The class name, with the arguments to construct it with.

    Returns
    -------
    acq_fcn : AbstractAcqFcn
        The acquisition function the string names.

    Raises
    ------
    ValueError
        If the string is not the name of an acquisition function class
        followed by an optional call with literal arguments.
    """
    try:
        expression = ast.parse(string.strip(), mode="eval").body
    except (SyntaxError, ValueError) as exc:
        raise ValueError(
            f"Cannot read {string!r} as an acquisition function: it is "
            "not a Python expression."
        ) from exc

    if isinstance(expression, ast.Name):
        name, arg_nodes, keyword_nodes = expression.id, [], []
    elif isinstance(expression, ast.Call) and isinstance(
        expression.func, ast.Name
    ):
        name = expression.func.id
        arg_nodes, keyword_nodes = expression.args, expression.keywords
    else:
        raise ValueError(
            f"Cannot read {string!r} as an acquisition function: it is "
            "not the name of an acquisition function class, with or "
            "without a call."
        )

    args = [_literal(node, string) for node in arg_nodes]
    kwargs = {}
    for keyword in keyword_nodes:
        if keyword.arg is None:
            raise ValueError(
                f"Cannot read {string!r} as an acquisition function: its "
                "arguments must be given one by one."
            )
        kwargs[keyword.arg] = _literal(keyword.value, string)

    acq_fcn = getattr(pyvbmc.acquisition_functions, name, None)
    if not (isinstance(acq_fcn, type) and issubclass(acq_fcn, AbstractAcqFcn)):
        raise ValueError(
            f"Cannot read {string!r} as an acquisition function: "
            f"{name!r} is not one of the acquisition function classes of "
            "pyvbmc.acquisition_functions."
        )
    return acq_fcn(*args, **kwargs)


def _literal(node, string):
    """The value of one argument, which has to be a literal."""
    try:
        return ast.literal_eval(node)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(
            f"Cannot read {string!r} as an acquisition function: its "
            "arguments must be literal values."
        ) from exc
