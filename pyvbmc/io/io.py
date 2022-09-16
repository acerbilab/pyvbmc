from textwrap import indent

import numpy as np


def summarize(obj, thresh=10, add_prefix=True, precision=4):
    """Construct a string summary of an array.

    If the size of the array is small enough, print the full array and type.
    Otherwise, just print the size and type. If the passed object is not an
    array, just return its usual string summary.
    """
    string = ""
    prefix = ""

    if not isinstance(obj, np.ndarray):
        # Stringify anything that's not an array.
        prefix = " = "
        if (
            type(obj) == str or type(obj) == np.str_
        ):  # Enclose string values in quotes
            string = "'" + str(obj) + "'"
        else:
            string = str(obj)
    elif np.prod(obj.shape) < thresh:
        # Print the full (but abbreviated precision) array on one line.
        prefix = " = "
        array_string = np.array2string(
            obj, precision=precision, suppress_small=True, separator=", "
        ).replace("\n", "")
        string = array_string + f": {type(obj).__name__}"
    else:
        # Print the shape of the array.
        prefix = ": "
        string = f"{obj.shape} {type(obj).__name__}"

    if add_prefix:
        return prefix + string
    else:
        return string


def format_dict(d, arr_size_thresh=np.inf):
    """Pretty-print a dictionary.

    Summarize possible array values with ``sumarrize()``.
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
                body += "'" + str(key) + "'" + ": "
            else:
                body += str(key) + ": "
            if type(val) == dict:  # Recursively format nested dictionaries
                val_string = format_dict(val, arr_size_thresh)
            else:  # Format possible array values
                val_string = summarize(val, arr_size_thresh, add_prefix=False)
            body += val_string + ",\n"
        body = indent(body, "    ")
        string += body
        string += "}"

    return string
