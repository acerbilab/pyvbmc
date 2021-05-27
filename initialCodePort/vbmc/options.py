from collections.abc import MutableMapping

class Options(MutableMapping, dict):
    """
    This class is responsible for Options.



    Parameters
    ----------
    UserOptions : set
        This set contains all options that have set by the user,
        if there are none is empty.
    """

    def __init__(self, default_options : dict, user_options : dict = None):
        """
        Initialize the options using default options and specified options from 
        the user.

        Parameters
        ----------
        default_options : dict
            The default options that can be overwritten by the user.
        user_options: dict
            User defined values to overwrite default options.
        """
        self.update(default_options)

        # User options
        if user_options is not None:
            self.update(user_options)
            self.__setitem__("UserOptions", set(user_options.keys()))
        else:
            self.__setitem__("UserOptions", set())

    def __setitem__(self, key, val):
        dict.__setitem__(self, key, val)

    def __getitem__(self, key):
        return dict.__getitem__(self, key)

    def __iter__(self):
        yield from sorted(dict.__iter__(self))

    def __len__(self):
        return dict.__len__(self)