from collections.abc import MutableMapping
import numpy as np
import copy


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
        """
        Returns the iteration history in a format key: value.

        Returns
        -------
        str
            The str to describe an IterationHistory object.
        """
        return "".join(["{}: {} \n".format(k, v) for (k, v) in self.items()])
