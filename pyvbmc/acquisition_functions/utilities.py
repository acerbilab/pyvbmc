from ast import literal_eval

import pyvbmc.acquisition_functions


def string_to_acq(string):
    """Safely convert an input string to an acquisition function."""
    # Remove whitespace and trailing ")", if present
    string = string.rstrip()
    string = string.removesuffix(")")
    # Split on "(" to get class name and parameters
    parts = string.partition("(")
    acq_fcn = parts[0]
    args_string = parts[2]
    args = []
    kwargs = {}
    for arg in args_string.split(","):
        arg = arg.rstrip().split("=")
        if len(arg) == 1 and arg != [""]:
            args.append(literal_eval(arg[0]))
        elif len(arg) == 2:
            kwargs[arg[0]] = literal_eval(arg[1])

    acq_fcn = getattr(pyvbmc.acquisition_functions, acq_fcn)
    return acq_fcn(*args, **kwargs)
