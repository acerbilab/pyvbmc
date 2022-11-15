import copy
from collections.abc import MutableMapping
from textwrap import indent

import numpy as np

from pyvbmc.formatting import format_dict


class IterationHistory(MutableMapping, dict):
    """
    This class is responsible for the VBMC iteration history.

    Parameters
    ----------
    keys : list
        The keys that can be recorded with this object.
    """

    def __init__(self, keys: list):
        super().__init__()
        self.check_keys = False
        for key in keys:
            self[key] = None
        self.check_keys = True

    def __setitem__(self, key: str, val: object):
        """
        Set the value of a given key to the given value.

        Parameters
        ----------
        key : str
            The key for which the value should be stored.
        val : object
            The value which should be stored.

        Raises
        ------
        Raised if the key has not been specified on initialization of the
            object.
        """
        if self.check_keys and key not in self:
            raise ValueError(
                """The key has not been specified
                on initialization of the object"""
            )
        else:
            dict.__setitem__(self, key, copy.deepcopy(val))

    def __getitem__(self, key):
        return dict.__getitem__(self, key)

    def __iter__(self):
        yield from sorted(dict.__iter__(self))

    def __len__(self):
        return dict.__len__(self)

    def __delitem__(self, key):
        return dict.__delitem__(self, key)

    def __getstate__(self):
        return (self.check_keys, dict(self))

    def __setstate__(self, state):
        self.check_keys, data = state
        self.update(data)

    def __reduce__(self):
        return (
            self.__class__,
            (list(dict(self).keys()),),
            self.__getstate__(),
        )

    def record(self, key: str, value: object, iteration: int):
        """
        Store a value for a key in a given iteration.

        Parameters
        ----------
        key : str
            The key for which the value should be stored.
        value : object
            The value which should be stored.
        iteration : int
            The iteration for which the value should be stored, must be >= 0.

        Raises
        ------
        ValueError
            Raised if the value of the iteration is < 0.
        ValueError
            Raised if a key has not been specified on initialization of the
            object.
        """
        if iteration < 0:
            raise ValueError("The iteration must be >= 0.")
        if key not in self:
            raise ValueError(
                """The key has not been specified
                on initialization of the object"""
            )
        else:
            if self[key] is None:
                self[key] = np.full([1], None)
            if len(self[key]) <= iteration:
                self._expand_array(key, iteration + 1 - len(self[key]))
            self[key][iteration] = copy.deepcopy(value)

    def _expand_array(self, key: str, resize_amount: int):
        """
        A private method to expand the array for a given key by a resize_amount.
        """
        self[key] = np.append(
            self[key], np.full([resize_amount], None), axis=0
        )

    def record_iteration(
        self,
        key_value: dict,
        iteration: int,
    ):
        """
        Convenience method to record multiple key-values for a given iteration.

        Parameters
        ----------
        key_value : dict
            The keys and values that should be recorded. They keys must have
            been specified on initialization of the object.
        iteration : int
            The iteration for which the value should be stored, must be >= 0.

        Raises
        ------
        ValueError
            Raised if the value of the iteration is < 0.
        ValueError
            Raised if a key has not been specified on initialization of the
            object
        """
        if iteration < 0:
            raise ValueError("The iteration must be >= 0.")

        for key, value in key_value.items():
            if key not in self:
                raise ValueError(
                    """The key has not been specified
                on initialization of the object"""
                )
            else:
                self.record(key, value, iteration)

    def __str__(self):
        """Construct a string summary.

        Returns
        -------
        string : str
            The string summarizing the IterationHistory object.
        """
        return "IterationHistory:\n" + indent(
            f"num. iterations = {len(self)}\nkeys = \n"
            + indent(",\n".join(key for key in self.keys()), "    "),
            "    ",
        )

    def __repr__(self, full=False, arr_size_thresh=10, expand=False):
        """Construct a detailed string summary.

        Parameters
        ----------
        arr_size_thresh : float, optional
            If ``obj`` is an array whose product of dimensions is less than
            ``arr_size_thresh``, print the full array. Otherwise print only the
            shape. Default `10`.
        full : bool, optional
            If ``full`` is `False`, print only the relevant object attributes.
            Otherwise print all attributes.
        expand : bool, optional
            Unused.

        Returns
        -------
        string : str
            The string representation of ``self``.
        """
        if full:  # Output every class attribute (for debugging)
            return "IterationHistory:\n" + indent(
                "self.check_keys: {self.check_keys},\ndict: "
                + format_dict(self, arr_size_thresh=arr_size_thresh),
                "    ",
            )
        else:  # Summary
            return str(self)

    def _short_repr(self):
        """Returns abbreviated string representation with memory location."""
        return object.__repr__(self)
