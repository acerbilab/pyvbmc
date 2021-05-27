from collections.abc import MutableMapping
import configparser
import numpy as np


class Options(MutableMapping, dict):
    """
    This class is responsible for Options.


    Parameters
    ----------
    UserOptions : set
        This set contains all options that have set by the user,
        if there are none it is empty.
    """

    def __init__(
        self,
        default_options_path: str,
        evalutation_parameters: dict = None,
        user_options: dict = None,
    ):
        """
        Initialize the options using default options and specified options from
        the user.

        Parameters
        ----------
        default_options_path : str
            The path to default options that can be overwritten by the user.
        evalutation_parameters : dict
            Parameters used to evaluate the options.
        user_options : dict
            User defined values to overwrite default options.
        """

        # evaluation_parameters
        for key, val in evalutation_parameters.items():
            exec(key + "=val")

        # default_options
        conf = configparser.ConfigParser()
        conf.read(default_options_path)
        for section in conf.sections():
            for (k, v) in conf.items(section):
                self.__setitem__(k, eval(v))

        # User options
        if user_options is not None:
            self.update(user_options)
            self.__setitem__("useroptions", set(user_options.keys()))
        else:
            self.__setitem__("useroptions", set())

    def __setitem__(self, key, val):
        dict.__setitem__(self, key, val)

    def __getitem__(self, key):
        return dict.__getitem__(self, key)

    def __iter__(self):
        yield from sorted(dict.__iter__(self))

    def __len__(self):
        return dict.__len__(self)