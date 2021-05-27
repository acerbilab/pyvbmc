from collections.abc import MutableMapping
from .default_options_advanced import get_default_options_advanced
from .default_options_fixed import get_default_options_fixed


class OptionsVBMC(MutableMapping, dict):
    """
    This class is responsible for the options of the VBMC algorithm.

    The options can be divided into three types of options:
        - **Basic default options:** We expect that these options are changed
          by many users. You can find a description of those options in the
          Parameters section below.
        - **Advanced options:** These options are for advanced users of VBMC.
          Do not modify them unless you *know* what you are doing. You can find
          them below the documentation of the class in the VBMC documentation.
        - **Advanced options for unsupported/untested features:** These are the
          unsupported/untested features options of VBMC which you
          should *never* modify. You can find them below the documentation of
          the class in the VBMC documentation.


    Parameters
    ----------
    Display : {"iter", "notify", "final", "off"}
        This specifies the level of display for log messages.
    FunEvalsPerIter: int
        The number of target function evaluations per iteration, by default 5.
    MaxIter : int
        The maximum number of iterations of VBMC, by default 50 * (2 + D).
    MaxFunEvals : int
        The maximum number of target function evaluations,
        which is by default 50 * (2 + D).
    MinFinalComponents : int
        The Number of variational components to refine posterior at termination,
        by default 50.
    ModifiedOptions : set
        If RecordModifiedOptions == 1 this set contains all options that have
        been modified after the initialization, otherwise it is always empty.
    Plot : bool
        Plot marginals of variational posterior at each iteration,
        by default False.
    RecordModifiedOptions : bool
        Record the modification of options after the initialization,
        by default True.
    RetryMaxFunEvals : int
        The maximum number of target functions evals on retry,
        where 0 means no retry and the default is 0.
    SpecifyTargetNoise : bool
        The Target log joint function returns noise estimate (SD) as second
        output, by default this is False.
    TolStableCount : int
       The required number of stable function evals for termination,
       by default this is 60.
    """

    def __init__(self, D: int, *args, **kwargs):
        r"""
        Initialize the options of VBMC using default options and specified
        options from the user.

        Parameters
        ----------
        D : int
            The number of dimensions of the data.
        *args, **kwargs
            User defined values to overwrite default VBMC options.

        Examples
        --------
        The options can be modified with `\*\*kwargs` as shown below.

        >>> D = 2
        >>> user_options = {"Display": "off"}
        >>> options = OptionsVBMC(D, user_options)
        >>> print(options.get("Display"))
        "off"

        """
        self.__setitem__("RecordModifiedOptions", False)
        # Advanced options (do not modify unless you *know* what you are doing)
        self.update(get_default_options_advanced(D))
        # Advanced options for unsupported/untested features (do *not* modify)
        self.update(get_default_options_fixed())
        # Basic default options
        self.__setitem__("Display", "iter")
        self.__setitem__("FunEvalsPerIter", 5)
        self.__setitem__("MaxFunEvals", 50 * (2 + D))
        self.__setitem__("MaxIter", 50 * (2 + D))
        self.__setitem__("MinFinalComponents", 50)
        self.__setitem__("Plot", False)
        self.__setitem__("RetryMaxFunEvals", 0)
        self.__setitem__("SpecifyTargetNoise", False)
        self.__setitem__("TolStableCount", 60)
        self.update(*args, **kwargs)
        self.__setitem__("ModifiedOptions", set())
        self.__setitem__("RecordModifiedOptions", True)

    def __setitem__(self, key, val):
        dict.__setitem__(self, key, val)
        if (
            dict.__getitem__(self, "RecordModifiedOptions")
            and key != "RecordModifiedOptions"
            and key != "ModifiedOptions"
        ):
            modified_options = dict.__getitem__(self, "ModifiedOptions")
            modified_options.add(key)
            dict.__setitem__(self, "ModifiedOptions", modified_options)

    def __getitem__(self, key):
        return dict.__getitem__(self, key)

    def __iter__(self):
        yield from sorted(dict.__iter__(self))

    def __len__(self):
        return dict.__len__(self)