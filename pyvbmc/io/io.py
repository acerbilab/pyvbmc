import numpy as np


def summarize(arr, thresh=10, add_prefix=True, precision=4):
    """Construct a string summary of an array.

    If the size of the array is small enough, print the full array and type.
    Otherwise, just print the size and type. If the passed object is not an
    array, just return its usual string summary.
    """
    string = ""
    prefix = ""

    if not isinstance(arr, np.ndarray):
        # Stringify anything that's not an array.
        prefix = " = "
        if (
            type(arr) == str or type(arr) == np.str_
        ):  # Enclose string values in quotes
            string = "'" + str(arr) + "'"
        else:
            string = str(arr)
    elif np.prod(arr.shape) < thresh:
        # Print the full (but abbreviated precision) array on one line.
        prefix = " = "
        array_string = np.array2string(
            arr, precision=precision, suppress_small=True, separator=", "
        ).replace("\n", "")
        string = array_string + f": {type(arr).__name__}"
    else:
        # Print the shape of the array.
        prefix = ": "
        string = f"{arr.shape} {type(arr).__name__}"

    if add_prefix:
        return prefix + string
    else:
        return string


def format_dict(d):
    """Pretty-print a dictionary.

    Summarize possible array values with ``sumarrize()``.
    """
    if d is None:
        string = "None"
    else:
        string = "{\n"
        for key, val in d.items():
            if (
                type(key) == str or type(key) == np.str_
            ):  # Enclose strings in quotes
                string += "'" + str(key) + "'" + ": "
            else:
                string += str(key) + ": "
            string += summarize(val, add_prefix=False) + ",\n"
        string += "}"

    return string
