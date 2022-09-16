from textwrap import indent

import numpy as np


def summarize(
    obj,
    arr_size_thresh=10,
    add_prefix=True,
    precision=4,
):
    """Construct a string summary of an object.

    If the object is an array and the array is small enough, print the full
    array and type. Otherwise, just print the size and type. If the passed
    object is not an array, just return its usual string summary.
    """
    string = ""
    prefix = ""

    if not isinstance(obj, np.ndarray):
        # Stringify anything that's not an array.
        prefix = " = "
        string = get_repr(obj)
    elif np.prod(obj.shape) < arr_size_thresh:
        # Print the full (but abbreviated precision) array on one line.
        prefix = " = "
        array_string = np.array2string(
            obj, precision=precision, suppress_small=True, separator=", "
        )
        string = array_string + f": {type(obj).__name__}"
    else:
        # Print the shape of the array.
        prefix = ": "
        string = f"{obj.shape} {type(obj).__name__}"

    if add_prefix:
        return prefix + string
    else:
        return string


def format_dict(d, **kwargs):
    """Pretty-print a dictionary.

    Summarize possible array values with ``summarize()``.
    """
    if d is None:
        string = "None"
    else:
        string = "{\n"
        body = ""
        for key, val in d.items():
            if (
                type(key) == str or type(key) == np.str_
            ):  # Enclose string keys in quotes
                body += repr(key) + ": "
            else:
                body += str(key) + ": "
            if type(val) == dict:  # Recursively format nested dictionaries
                val_string = format_dict(val, **kwargs)
            else:  # Format possible array values
                val_string = summarize(val, **kwargs)
            body += val_string + ",\n"
        body = indent(body, "    ")
        string += body
        string += "}"

    return string


def get_repr(obj, expand=False, full=False, **kwargs):
    if expand:  # Expand child elements
        if type(obj) == dict:
            return format_dict(obj, **kwargs)
        else:
            try:
                return obj.__repr__(expand=True, full=full)
            except TypeError:
                return repr(obj)
    else:  # Abbreviated representation
        if hasattr(obj, "_short_repr"):
            return obj._short_repr()
        elif type(obj) == dict:  # Just print type and memory location
            return object.__repr__(obj)
        else:
            return repr(obj)


def full_repr(obj, title, **kwargs):
    body = ""
    try:
        for key, val in sorted(obj.__dict__.items()):
            body += f"self.{key} = {get_repr(val, **kwargs)}\n"
    except TypeError:  # Keys cannot be sorted
        for key, val in obj.__dict__.items():
            body += f"self.{key} = {get_repr(val, **kwargs)}\n"
    return title + ":\n" + indent(body, "    ")
