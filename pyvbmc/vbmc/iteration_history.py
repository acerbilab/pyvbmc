from collections.abc import MutableMapping
import numpy as np


class IterationHistory(MutableMapping, dict):
    """
    This class is responsible for the VBMC iteration history.
    """

    def __setitem__(self, key, val):
        dict.__setitem__(self, key, val)

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
        """        
        if iteration < 0:
            raise ValueError("The iteration must be >= 0.")
        if key in self:
            if len(self[key]) <= iteration:
                self._expand_array(key, iteration + 1 - len(self[key]))
        else:
            self[key] = np.full([iteration + 1], np.nan)
        self[key][iteration] = value

    def _expand_array(self, key: str, resize_amount: int):
        """
        A private method to expand the array for a given key by a resize_amount.
        """
        self[key] = np.append(
            self[key], np.full([resize_amount], np.nan), axis=0
        )

    def record_iteration(self, *args, **kwargs):
        pass

    def __str__(self):
        """
        Returns the iteartion history in a format key: value.

        Returns
        -------
        str
            The str to describe an IterationHistory object.
        """
        return "".join(["{}: {} \n".format(k, v) for (k, v) in self.items()])
