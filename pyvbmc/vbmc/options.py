# for annotating Options as input of itself
from __future__ import annotations

import configparser
import copy
import logging
from collections.abc import MutableMapping
from math import ceil
from pathlib import Path
from textwrap import indent

import numpy as np

from pyvbmc.acquisition_functions import *
from pyvbmc.formatting import full_repr

#: Options declared in the ``.ini`` files that no PyVBMC module reads. They
#: are kept so that option dictionaries recorded by earlier runs still load,
#: and the value given for one of them has no effect. The ``# description``
#: line of each says why it is there.
INERT_OPTIONS = frozenset(
    {
        "acq_hedge_decay",
        "acq_hedge_iter_window",
        "active_sample_fess_thresh",
        "active_variational_samples",
        "adaptive_entropy_alpha",
        "annealed_gp_mean",
        "best_frac_back",
        "best_safe_sd",
        "constrained_gp_mean",
        "double_gp",
        "empirical_gp_prior",
        "gp_stochastic_step_size",
        "integrate_gp_mean",
        "noise_shaping_factor",
        "noise_shaping_threshold",
        "nonlinear_scaling",
        "optimistic_variational_bound",
        "output_fcn",
        "rank_criterion",
        "sample_extra_vp_means",
        "scale_lower_bound",
        "search_cmaes_best",
        "variational_init_repo",
        "variational_sampler",
        "warmup_options",
    }
)


class Options(MutableMapping, dict):
    """
    This class is responsible for Options.

    Parameters
    ----------
    default_options_path : str
        The path to the default options. May be absolute or relative to this
        directory.
    evaluation_parameters : dict
        Parameters used to evaluate the options.
    user_options : dict
        User defined values to overwrite default options.

    Attributes
    ----------
    useroptions : set
        This set contains all options that have set by the user,
        if there are none it is empty. These ``useroptions`` are immutable to
        changes using :py:meth:`load_options_file`.
    """

    def __init__(
        self,
        default_options_path: str,
        evaluation_parameters: dict = None,
        user_options: dict = None,
    ):
        """
        Initialize the options using default options and specified options from
        the user.
        """
        # Flag initialization as in-progress
        # (completed in self.validate_option_names)
        self.is_initialized = False
        super().__init__()
        self.descriptions = {}
        self["useroptions"] = set()

        self.default_options_path = default_options_path
        self.evaluation_parameters = evaluation_parameters
        self.user_options = user_options

        # load options from file
        self.load_options_file(default_options_path, evaluation_parameters)

        # User options
        if user_options is not None:
            self.update(user_options)
            self["useroptions"].update(user_options.keys())

    def update_defaults(self):
        """Change defaults as needed based on values of other options."""
        if self.get("specify_target_noise"):
            updates = {
                "max_fun_evals": ceil(self["max_fun_evals"] * 1.5),
                "tol_stable_count": ceil(self["tol_stable_count"] * 1.5),
                "active_sample_gp_update": True,
                "active_sample_vp_update": True,
                "search_acq_fcn": [AcqFcnVIQR()],
            }
            for key, val in updates.items():
                if key not in self["useroptions"]:
                    self[key] = val

    @classmethod
    def init_from_existing_options(
        cls,
        default_options_path: str,
        evaluation_parameters: dict = None,
        other: Options = None,
    ):
        """
        Initialize an options instance using default options and another options
        instance.

        Only the user-definied options from the other object will overwrite the
        default options. Everything else will come from the default options.

        Parameters
        ----------
        default_options_path : str
            The path to the default options. May be absolute or relative to
            this directory.
        evaluation_parameters : dict
            Parameters used to evaluate the options.
        other : Options
            User defined values to overwrite default options.

        Returns
        -------
        new_options : Options
            The new options object with the values merged as described above.
        """
        if other is None:
            user_options = None
        else:
            user_option_keys = other.get("useroptions")
            user_options = {k: other.get(k) for k in user_option_keys}
        new_options = cls(
            default_options_path, evaluation_parameters, user_options
        )
        return new_options

    def load_options_file(
        self, options_path: str, evaluation_parameters: dict = None
    ):
        """
        Load options from an ini file and evaluate them using the specified
        ``evaluation_parameters``.

        Note that strings starting with # in the .ini file act as description to
        the option in the following line.

        Parameters
        ----------
        options_path : str
            The path to an ``options.ini`` file that should be loaded. May be
            absolute or relative to this directory.
        evaluation_parameters : dict, optional
            Parameters used to evaluate the options.
        """
        options_list = _read_config_file(options_path)
        for key, value, description in options_list:
            if key not in self.get("useroptions") and key != "useroptions":
                self[key] = eval(value, globals(), evaluation_parameters)
                self.descriptions[key] = description

    def validate_option_names(self, options_paths: list):
        """
        Check that ini files specified by the list of ``options_paths`` contain
        the option names from this options object at least once.

        Note that this method checks not if there are option names in files that
        are not in the object. After option names are validated, initialization
        is flagged as complete and `self.is_initialized` is set to `True` to
        prevent further modification of options.

        Parameters
        ----------
        options_paths : list of str
            A list of paths to ini files can contain the allowed option names.

        Raises
        ------
        ValueError
            Raised when an option exists in this object but not in one of the
            specified ini files.
        """
        # create set of option names from all ini files
        file_option_names = set()
        for options_path in options_paths:
            file_option_names.update(
                _read_config_file(options_path)[:, 0].flatten()
            )

        for key in self.keys():
            if key != "useroptions" and key not in file_option_names:
                raise ValueError("The option {} does not exist.".format(key))

        self._warn_inert_options(options_paths)

        # After initialzation is complete prevent changes to options:
        self.is_initialized = True

    def _warn_inert_options(self, options_paths: list):
        """
        Warn about the options of :data:`INERT_OPTIONS` that the user set to
        a value other than the default declared in the ini files.

        Parameters
        ----------
        options_paths : list of str
            A list of paths to the ini files that declare the defaults.
        """
        supplied = set(self.get("useroptions")) & INERT_OPTIONS
        if len(supplied) == 0:
            return

        default_values = {}
        for options_path in options_paths:
            for key, value, __ in _read_config_file(options_path):
                if key in supplied:
                    default_values[key] = value

        for key in sorted(supplied):
            if key not in default_values:
                continue
            try:
                default = eval(
                    default_values[key], globals(), self.evaluation_parameters
                )
            except Exception:
                continue
            if _equals_default(self[key], default):
                continue
            logging.warning(
                "The option %s has no effect in PyVBMC: the value %s is "
                "accepted and ignored.",
                key,
                self[key],
            )

    def __setitem__(self, key, val, force=False):
        # Prevent user from attempting to modify options after initialization
        if (
            hasattr(self, "is_initialized")
            and self.is_initialized
            and not force
        ):
            raise AttributeError(
                "Warning: Cannot set options after initialization. Please re-initialize with `options = {...}`"
            )
        else:
            dict.__setitem__(self, key, val)

    def __getitem__(self, key):
        return dict.__getitem__(self, key)

    def __iter__(self):
        yield from sorted(dict.__iter__(self))

    def __len__(self):
        return dict.__len__(self)

    def __delitem__(self, key):
        return dict.__delitem__(self, key)

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        # Copy class properties:
        result.__dict__.update(self.__dict__)
        # Copy options dict:
        for k, v in dict.items(self):
            result.__setitem__(k, v, force=True)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)

        # Avoid infinite recursion in deepcopy
        memo[id(self)] = result
        # Copy class properties:
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        # Copy options dict:
        for k, v in dict.items(self):
            result.__setitem__(k, copy.deepcopy(v, memo), force=True)
        return result

    def eval(self, key: str, evaluation_parameters: dict):
        """
        Evaluate an option using `evaluation_parameters` if it is a callable,
        otherwise return the value of the option.

        Parameters
        ----------
        key : str
            The name of the option.
        evaluation_parameters : dict
            Parameters for the options in case it is a callable. These have to
            match the key arguments of the callable and are ignored if it is not
            a callable.

        Returns
        -------
        val : object
            Value of the object which has been evaluated if it is a callable.
        """
        if callable(self.get(key)):
            return self.get(key)(**evaluation_parameters)
        else:
            return self.get(key)

    def __str__(self):
        """
        Return the user options in a format key: value (description).

        Returns
        -------
        str
            The str to describe an options object.
        """
        user_options = "\n".join(
            [
                f"{key}: {self[key]} ({self.descriptions.get(key)})"
                for key in self["useroptions"]
            ]
        )
        if user_options == "":
            user_options = (
                "None (use default options).\n"
                + "View current defaults with `options` or `repr(options)`."
            )
        return "User Options:\n" + indent(user_options, "    ")

    def __repr__(self, full=False, expand=False):
        """
        Return the options in a format key: value (description).

        Returns
        -------
        string : str
            The str to describe the Options object.
        full : bool, optional
            If ``full`` is `False`, print only the relevant object attributes.
            Otherwise print all attributes.
        expand : bool, optional
            If ``expand`` is `False`, then describe any complex child
            attributes of the object by their name and memory location.
            Otherwise, recursively expand the child attributes into their own
            representations. Default `False`.
        """
        if full:  # Output every class attribute (for debugging)
            return full_repr(self, "Options", expand=expand)
        else:  # Output relevant class attributes in meaningful format
            return "Options:\n" + indent(
                "\n".join(
                    [
                        f"{key}: {value} ({self.descriptions.get(key)})"
                        for (key, value) in self.items()
                    ]
                ),
                "    ",
            )


def _equals_default(value, default):
    """
    Private helper method to compare an option value against its default,
    for values of any type an ini file can produce.
    """
    if value is default:
        return True
    try:
        return bool(
            np.array_equal(
                np.asarray(value, dtype=object),
                np.asarray(default, dtype=object),
            )
        )
    except Exception:
        return False


def _read_config_file(options_path: str):
    """
    Private helper method to read a config file and return the options as a
    list of tuples (key, value, description).

    Note that strings starting with # in the .ini file act as description to
    the option in the following line.
    """
    path = Path(options_path)
    if not path.is_absolute():
        path = Path(__file__).parent.joinpath(path)

    if not path.exists():
        raise ValueError(f"{path.resolve()} does not exist.")
    conf = configparser.ConfigParser(comment_prefixes="", allow_no_value=True)
    # do not lower() both values as well as descriptions
    conf.optionxform = str
    conf.read(path)

    option_list = []
    description = ""
    for section in conf.sections():
        for key, value in conf.items(section):
            if "#" in key:
                description = key.strip("# ")
            else:
                option_list.append([key, value, description])
                description = ""

    if len(option_list) == 0:
        raise ValueError(
            "The option file at {} does not contain options.".format(
                options_path
            )
        )

    return np.array(option_list)
