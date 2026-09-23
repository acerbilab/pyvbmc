# for annotating Options as input of itself
from __future__ import annotations

import configparser
import copy
import inspect
import logging
import re
from collections.abc import MutableMapping
from math import ceil
from numbers import Real
from pathlib import Path
from textwrap import indent

import numpy as np

from pyvbmc.acquisition_functions import *
from pyvbmc.formatting import full_repr
from pyvbmc.parameter_transformer.parameter_transformer import (
    _bounded_transform_type,
)
from pyvbmc.whitening.whitening import _is_finite_real_number

#: Options declared in the ``.ini`` files that no PyVBMC module reads. They
#: are kept so that option dictionaries recorded by earlier runs still load,
#: and the value given for one of them has no effect. The ``# description``
#: line of each says why it is there.
INERT_OPTIONS = frozenset(
    {
        "acq_hedge_decay",
        "acq_hedge_iter_window",
        "active_importance_sampling_fess_thresh",
        "active_sample_fess_thresh",
        "active_variational_samples",
        "adaptive_entropy_alpha",
        "annealed_gp_mean",
        "constrained_gp_mean",
        "cov_sample_thresh",
        "diagnostics",
        "double_gp",
        "empirical_gp_prior",
        "gp_int_mean_fun",
        "gp_stochastic_step_size",
        "integrate_gp_mean",
        "noise_shaping_factor",
        "noise_shaping_threshold",
        "nonlinear_scaling",
        "optimistic_variational_bound",
        "output_fcn",
        "proposal_fcn",
        "sample_extra_vp_means",
        "scale_lower_bound",
        "search_cmaes_best",
        "separate_search_gp",
        "temperature",
        "variational_init_repo",
        "variational_sampler",
        "warmup_options",
    }
)


#: The ``.ini`` files that declare every option PyVBMC accepts. A name
#: outside them is not an option, wherever it was supplied.
SHIPPED_OPTIONS_PATHS = (
    "option_configs/basic_vbmc_options.ini",
    "option_configs/advanced_vbmc_options.ini",
)


def declared_option_names(options_paths=SHIPPED_OPTIONS_PATHS):
    """
    The option names the given ini files declare.

    Parameters
    ----------
    options_paths : iterable of str, optional
        Paths to ini files, absolute or relative to this directory.
        Default the two files PyVBMC ships.

    Returns
    -------
    names : set of str
        Every option name declared in those files.
    """
    names = set()
    for options_path in options_paths:
        names.update(_read_config_file(options_path)[:, 0].flatten())
    return names


# How the integer_vars option may be written, named in the errors raised
# for any other value.
_INTEGER_VARS_FORMS = (
    "a boolean array with one entry per variable, or an array of the "
    "0-based indices of the variables that take only integer values"
)

# How the uncertainty_handling option may be written, named in the error
# raised for any other value.
_UNCERTAINTY_HANDLING_FORMS = (
    "True or False (the integers 1 and 0 and their NumPy equivalents are "
    "also accepted), or an empty value ([], an empty array or None) to "
    "leave the choice to specify_target_noise"
)

# How the specify_target_noise option may be written, named in the error
# raised for any other value.
_SPECIFY_TARGET_NOISE_FORMS = (
    "True or False (the integers 1 and 0 and their NumPy equivalents are "
    "also accepted)"
)

# How the noise_size option may be written, named in the error raised for
# any other value.
_NOISE_SIZE_FORMS = (
    "a positive finite number (a Python or NumPy integer or floating-point "
    "number, or a 0-d array that holds one; not a boolean), or an empty "
    "value ([], an empty array or None) to leave it unset"
)


def _is_positive_integer_valued(value):
    """
    Whether a limit on iterations or evaluations is a positive integer.

    A floating value that lands on an integer counts, as MATLAB VBMC's
    ``round(x) ~= x`` test lets one through, and so does positive
    infinity, which stands for no limit.
    """
    if not isinstance(value, Real):
        return False
    if not value > 0:
        return False
    return bool(np.isinf(value)) or float(value).is_integer()


def _is_finite_non_negative_integer_valued(value):
    """
    Whether a count that may be zero is a finite non-negative integer.

    A floating value that lands on an integer counts, as for
    `_is_positive_integer_valued`; infinity and NaN do not.
    """
    if not isinstance(value, Real):
        return False
    if not value >= 0 or np.isinf(value):
        return False
    return float(value).is_integer()


def _stated_boolean(value):
    """
    The boolean that a value states, or `None` when it states none.

    A boolean, the integers 1 and 0 and their NumPy equivalents state one.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    return None


def _specify_target_noise_flag(value):
    """
    Read the ``specify_target_noise`` option as a boolean.

    Parameters
    ----------
    value : object
        The value of the option.

    Returns
    -------
    flag : bool
        Whether the target returns its own noise estimate.

    Raises
    ------
    ValueError
        When the value is not a boolean.
    """
    flag = _stated_boolean(value)
    if flag is None:
        raise ValueError(
            "The option specify_target_noise must be "
            + _SPECIFY_TARGET_NOISE_FORMS
            + f"; got {value!r}."
        )
    return flag


def _is_empty_value(value):
    """Whether an option holds an empty value (``None``, ``[]``, an empty
    tuple or an empty array), which states nothing."""
    if value is None:
        return True
    return isinstance(value, (list, tuple, np.ndarray)) and np.size(value) == 0


def _noise_size_reading(value):
    """
    Read the ``noise_size`` option.

    An empty value leaves the option unset. Any other value has to be a
    positive finite number, as ``misc/setupoptions_vbmc.m:131-132``
    requires, since the GP fit starts its noise from the logarithm of the
    value. A 0-d array counts as the number it holds, as it does for the
    other options that take one number; an array with an axis, even of one
    entry, does not.

    Parameters
    ----------
    value : object
        The value of the option.

    Returns
    -------
    noise_size : float or None
        The value as a float, or `None` when the option is empty.

    Raises
    ------
    ValueError
        When the value is neither empty nor a positive finite number.
    """
    if _is_empty_value(value):
        return None
    if _is_finite_real_number(value) and value > 0:
        return float(value)
    message = (
        "The option noise_size must be "
        + _NOISE_SIZE_FORMS
        + f"; got {value!r}. A saved run that carries such a value is "
        "continued with VBMC.load(file, new_options={'noise_size': []})."
    )
    if _is_finite_real_number(value):
        message += (
            " Release 1.0.4 ran a value that is not positive as the value "
            "of the option tol_gp_noise, which new_options can give "
            "instead."
        )
    raise ValueError(message)


def _uncertainty_handling_flag(value):
    """
    Read the ``uncertainty_handling`` option as a boolean.

    Parameters
    ----------
    value : object
        The value of the option.

    Returns
    -------
    flag : bool or None
        `True` or `False` when the value states the choice, and `None` when
        the option is empty, which leaves the choice to
        ``specify_target_noise``.

    Raises
    ------
    ValueError
        When the value is neither a boolean nor empty.
    """
    if value is None:
        return None
    flag = _stated_boolean(value)
    if flag is not None:
        return flag
    if isinstance(value, (list, tuple, np.ndarray)) and np.size(value) == 0:
        return None
    raise ValueError(
        "The option uncertainty_handling must be "
        + _UNCERTAINTY_HANDLING_FORMS
        + f"; got {value!r}."
    )


def _integer_vars_mask(value, D):
    """
    Read a value of the ``integer_vars`` option as a mask over the
    variables; :py:meth:`Options.integer_vars_mask` describes the forms.
    """
    mask = np.full(D, False)
    if value is None:
        return mask
    array = np.asarray(value)
    if array.size == 0:
        return mask
    if array.ndim != 1 or not (
        array.dtype == bool or np.issubdtype(array.dtype, np.integer)
    ):
        raise ValueError(
            "The option integer_vars must be "
            + _INTEGER_VARS_FORMS
            + f"; got {value!r}."
        )
    if array.dtype == bool:
        if array.size != D:
            raise ValueError(
                "The option integer_vars, written as a boolean mask, "
                f"needs one entry per variable, that is {D}; got "
                f"{array.size}."
            )
        mask[array] = True
        return mask
    if array.size == D and np.all((array == 0) | (array == 1)):
        raise ValueError(
            f"The option integer_vars holds {D} integers, each of them "
            "zero or one, which reads both as a mask and as a list of "
            "indices. Write a boolean array to give a mask."
        )
    if np.any(array < 0) or np.any(array >= D):
        raise ValueError(
            "The option integer_vars, written as indices, needs "
            f"0-based indices of the {D} variables; got {value!r}."
        )
    if np.unique(array).size != array.size:
        raise ValueError(
            "The option integer_vars, written as indices, names a "
            f"variable twice; got {value!r}."
        )
    mask[array] = True
    return mask


#: The options that construction reads by their truth.
_OPTIONS_READ_BY_TRUTH = ("warmup", "entropy_switch", "fitness_shaping")


def _construction_reading(name, value, D):
    """
    The value of an option as construction reads it.

    ``uncertainty_handling`` and ``specify_target_noise`` are read as the
    choices they state, ``integer_vars`` as its mask over the `D`
    variables, ``f_vals`` as a flat array, ``bounded_transform`` as the
    number of the transform it names, which ``ParameterTransformer`` gives
    ``"probit"`` and ``"norminv"`` alike, and the options of
    :data:`_OPTIONS_READ_BY_TRUTH` by their truth; any other option is read
    as it is.

    Raises
    ------
    ValueError
        When construction refuses the value.
    """
    if name == "uncertainty_handling":
        return _uncertainty_handling_flag(value)
    if name == "specify_target_noise":
        return _specify_target_noise_flag(value)
    if name == "integer_vars":
        return _integer_vars_mask(value, D)
    if name == "f_vals":
        return np.array(value).ravel()
    if name == "bounded_transform":
        return _bounded_transform_type(value)
    if name in _OPTIONS_READ_BY_TRUTH:
        return bool(value)
    return value


def _same_reading(value, other):
    """
    Whether two readings of an option are the same.

    Two strings are the same when they are equal. Numbers, booleans and
    arrays of them are the same when their values are equal and their
    shapes agree, NaN matching NaN, so that a boolean and the number it
    equals are the same. ``None`` is the same only as ``None``, and other
    objects are compared as :func:`_equals_default` compares a value with
    its default.
    """
    if value is other:
        return True
    if isinstance(value, str) or isinstance(other, str):
        return (
            isinstance(value, str)
            and isinstance(other, str)
            and value == other
        )
    if value is None or other is None:
        return False
    try:
        value_array = np.asarray(value, dtype=np.float64)
        other_array = np.asarray(other, dtype=np.float64)
    except (TypeError, ValueError):
        return _equals_default(value, other)
    return value_array.shape == other_array.shape and bool(
        np.array_equal(value_array, other_array, equal_nan=True)
    )


def _states_the_stored_value(name, value, stored, D):
    """
    Whether a value of an option that only construction reads states what
    the value a run stores for it states.

    Both are read as construction reads the option
    (:func:`_construction_reading`) and compared by
    :func:`_same_reading`. A stored value that construction would refuse,
    which a run saved by an earlier release can hold, states nothing that a
    value given now can match.

    Parameters
    ----------
    name : str
        The name of the option.
    value : object
        The value given for it.
    stored : object
        The value the run stores for it.
    D : int
        The number of variables of the run.

    Returns
    -------
    same : bool
        Whether the two values state the same.

    Raises
    ------
    ValueError
        When construction refuses ``value``.
    """
    reading = _construction_reading(name, value, D)
    try:
        stored_reading = _construction_reading(name, stored, D)
    except ValueError:
        return False
    return _same_reading(reading, stored_reading)


def _takes_keyword(function, name):
    """
    Whether a callable takes an argument by the keyword `name`.

    It does when it has a parameter of that name that is not
    positional-only, or ``**kwargs``. A callable whose signature
    :func:`inspect.signature` cannot read is taken not to.
    """
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return False
    for parameter in parameters:
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == name and parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return True
    return False


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

        # User options. They may be the options of another run, as an
        # `Options` object or a dict copied from one, whose set of user
        # options is left out: taken over, it would be shared between the
        # two, and the names added here would change the other run's.
        if user_options is not None:
            supplied = {
                k: v for k, v in user_options.items() if k != "useroptions"
            }
            self.update(supplied)
            self["useroptions"].update(supplied.keys())

    def integer_vars_mask(self, D: int):
        """
        Read the ``integer_vars`` option as a mask over the variables.

        A boolean array with one entry per variable is that mask. An array
        of integers holds the 0-based indices of the integer variables,
        which have to be distinct and within range. An array of `D`
        integers that are all zero or one reads as either and is refused.

        Parameters
        ----------
        D : int
            The number of variables.

        Returns
        -------
        mask : np.ndarray
            A boolean array of length `D`, `True` at the variables that
            take only integer values.

        Raises
        ------
        ValueError
            When the value is neither a boolean mask of length `D` nor an
            array of distinct indices within range.
        """
        return _integer_vars_mask(self.get("integer_vars"), D)

    def validate_run_limits(self):
        """
        Check the limits on iterations and function evaluations.

        ``max_fun_evals`` and ``max_iter`` have to be positive integers, as
        ``misc/setupoptions_vbmc.m:109-114`` requires, or infinity for no
        limit. ``min_iter`` has to be a finite non-negative integer, 0 for a
        run without a minimum: the minimum holds back every termination,
        the one on the budget of evaluations included, so a run under an
        infinite one would never stop. A floating value that lands on an
        integer counts. A ``max_iter`` below ``min_iter`` is raised to it,
        as ``misc/setupoptions_vbmc.m:115-119`` does.

        Raises
        ------
        ValueError
            When ``max_fun_evals`` or ``max_iter`` is not a positive
            integer, or ``min_iter`` is not a finite non-negative integer.
        """
        for key in ("max_fun_evals", "max_iter"):
            value = self.get(key)
            if not _is_positive_integer_valued(value):
                raise ValueError(
                    f"The option {key} needs to be a positive integer; "
                    f"got {value!r}."
                )
        min_iter = self.get("min_iter")
        if not _is_finite_non_negative_integer_valued(min_iter):
            raise ValueError(
                "The option min_iter needs to be a finite non-negative "
                f"integer (0 for no minimum); got {min_iter!r}."
            )
        if self.get("max_iter") < min_iter:
            logging.warning(
                "The option max_iter cannot be smaller than min_iter. "
                "Raising max_iter to %s.",
                self.get("min_iter"),
            )
            self.__setitem__("max_iter", self.get("min_iter"), force=True)

    def uncertainty_handling_on(self):
        """
        Whether the run treats the target log-density as noisy.

        It does when the target returns its own noise estimate
        (``specify_target_noise``) or when ``uncertainty_handling`` asks for
        the noise level to be inferred.

        Returns
        -------
        on : bool
            Whether uncertainty handling is on.

        Raises
        ------
        ValueError
            When ``uncertainty_handling`` holds a value that is neither a
            boolean nor empty, when ``specify_target_noise`` holds a value
            that is not a boolean, or when ``uncertainty_handling`` is off
            while ``specify_target_noise`` is set.
        """
        requested = _uncertainty_handling_flag(
            self.get("uncertainty_handling")
        )
        if _specify_target_noise_flag(self.get("specify_target_noise")):
            if requested is False:
                raise ValueError(
                    "A target that returns its own noise estimate is a "
                    "noisy target: with specify_target_noise set, "
                    "uncertainty_handling cannot be turned off. Leave it "
                    "empty or set it to True."
                )
            return True
        return bool(requested)

    def update_defaults(self):
        """Change defaults as needed based on values of other options."""
        if self.uncertainty_handling_on():
            # Each default is computed only if it is going to be used: a
            # value the user set need not admit the computation (an
            # infinite budget has no ceiling).
            updates = {
                "max_fun_evals": lambda: ceil(self["max_fun_evals"] * 1.5),
                "tol_stable_count": lambda: ceil(
                    self["tol_stable_count"] * 1.5
                ),
                "active_sample_gp_update": lambda: True,
                "active_sample_vp_update": lambda: True,
                "search_acq_fcn": lambda: [AcqFcnVIQR()],
            }
            for key, default in updates.items():
                if key not in self["useroptions"]:
                    self[key] = default()

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
        self,
        options_path: str,
        evaluation_parameters: dict = None,
        as_user_options: bool = False,
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
        as_user_options : bool, optional
            Whether the file states the user's choices rather than the
            defaults PyVBMC ships. The options it sets then join
            ``useroptions``, so that they survive a later load and are not
            replaced by :py:meth:`update_defaults`. Default `False`.
        """
        options_list = _read_config_file(options_path)
        loaded = set()
        for key, value, description in options_list:
            if key == "useroptions":
                continue
            if key not in self.get("useroptions"):
                self[key] = eval(value, globals(), evaluation_parameters)
                loaded.add(key)
                if description or key not in self.descriptions:
                    self.descriptions[key] = description
            elif key not in self.descriptions:
                # The description belongs to the option, whoever set its
                # value.
                self.descriptions[key] = description
        if as_user_options:
            self["useroptions"].update(loaded)

    def validate_supplied_option_names(self, names):
        """
        Check that every name of ``names`` is one PyVBMC declares.

        Parameters
        ----------
        names : iterable of str
            The option names a caller supplied.

        Raises
        ------
        ValueError
            Raised when a name is declared by neither of the shipped ini
            files.
        """
        declared = declared_option_names()
        for key in sorted(names):
            if key not in declared:
                raise ValueError("The option {} does not exist.".format(key))

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
        file_option_names = declared_option_names(options_paths)

        for key in self.keys():
            if key != "useroptions" and key not in file_option_names:
                raise ValueError("The option {} does not exist.".format(key))

        self._warn_inert_options(options_paths)
        self._warn_ignored_noise_size()

        # After initialzation is complete prevent changes to options:
        self.is_initialized = True

    def _warn_inert_options(self, options_paths: list, names=None):
        """
        Warn about the options of :data:`INERT_OPTIONS` that the user set to
        a value other than the default declared in the ini files.

        An option whose declared default is a callable is left alone: two
        functions cannot be told apart by value, and repeating such a
        default (as an option dictionary recorded by an earlier run does)
        would otherwise look like a change.

        Parameters
        ----------
        options_paths : list of str
            A list of paths to the ini files that declare the defaults.
        names : iterable of str, optional
            The names of the options to weigh. Default the options the user
            set (``useroptions``).
        """
        if names is None:
            names = self.get("useroptions")
        supplied = set(names) & INERT_OPTIONS
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
            if callable(default) or _equals_default(self[key], default):
                continue
            logging.warning(
                "The option %s has no effect in PyVBMC: the value %s is "
                "accepted and ignored.",
                key,
                self[key],
            )

    def _warn_ignored_noise_size(self, names=None):
        """
        Warn when the user set ``noise_size`` for a target that returns its
        own noise estimates.

        With ``specify_target_noise`` on, the GP takes the noise of each
        observation from the target and does not read ``noise_size``, as
        MATLAB VBMC warns (``misc/setupoptions_vbmc.m:139-140``). An empty
        value states no noise size and is left alone, and so is a value
        that is not a noise size, which the check of the option values
        refuses (:func:`_noise_size_reading`).

        Parameters
        ----------
        names : iterable of str, optional
            The names of the options to weigh. Default the options the user
            set (``useroptions``).
        """
        if names is None:
            names = self.get("useroptions")
        if "noise_size" not in names:
            return
        noise_size = self.get("noise_size")
        try:
            if _noise_size_reading(noise_size) is None:
                return
        except ValueError:
            return
        if _stated_boolean(self.get("specify_target_noise")) is not True:
            return
        logging.warning(
            "The option noise_size has no effect with specify_target_noise, "
            "because the target returns its own noise estimates: the value "
            "%s is accepted and ignored.",
            noise_size,
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

    def __delitem__(self, key, force=False):
        # Options cannot be removed after initialization, as they cannot be
        # set; ``pop``, ``popitem`` and ``clear`` all come through here.
        if (
            hasattr(self, "is_initialized")
            and self.is_initialized
            and not force
        ):
            raise AttributeError(
                "Warning: Cannot remove options after initialization. Please re-initialize with `options = {...}`"
            )
        else:
            dict.__delitem__(self, key)

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

        A callable evaluated with a single parameter receives its value by
        keyword when it takes an argument of that name (a parameter so
        named that is not positional-only, or ``**kwargs``), and by
        position otherwise, as MATLAB VBMC's ``misc/evaloption_vbmc.m``
        calls ``option(N)``, so the name of a single parameter does not
        matter. A callable whose signature cannot be read receives it by
        position. With several parameters it receives them as keyword
        arguments, and its parameters have to carry their names.

        Parameters
        ----------
        key : str
            The name of the option.
        evaluation_parameters : dict
            Parameters for the option in case it is a callable, by name.
            They are ignored if it is not a callable.

        Returns
        -------
        val : object
            Value of the object which has been evaluated if it is a callable.
        """
        value = self.get(key)
        if not callable(value):
            return value
        if len(evaluation_parameters) == 1:
            ((name, parameter),) = evaluation_parameters.items()
            if _takes_keyword(value, name):
                return value(**{name: parameter})
            return value(parameter)
        return value(**evaluation_parameters)

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


#: An option line of an ini file, up to the delimiter that ends its name.
_OPTION_LINE = re.compile(r"\s*([^#;=:\s][^=:]*?)\s*[=:]")


def _read_descriptions(path: Path):
    """
    Private helper method to read the description of each option of an ini
    file, that is the comment line above it.

    The description is taken from the raw line, because a config parser
    would split it at the first ``=`` or ``:`` it contains.
    """
    descriptions = {}
    description = ""
    with open(path, encoding="utf-8") as config_file:
        for line in config_file:
            stripped = line.strip()
            if stripped == "" or stripped.startswith("["):
                continue
            if stripped.startswith("#") or stripped.startswith(";"):
                description = stripped.lstrip("#;").strip()
                continue
            match = _OPTION_LINE.match(line)
            if match is not None:
                descriptions[match.group(1)] = description
                description = ""
    return descriptions


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
    conf = configparser.ConfigParser(allow_no_value=True)
    # do not lower() both values as well as descriptions
    conf.optionxform = str
    conf.read(path)

    descriptions = _read_descriptions(path)
    option_list = []
    for section in conf.sections():
        for key, value in conf.items(section):
            option_list.append([key, value, descriptions.get(key, "")])

    if len(option_list) == 0:
        raise ValueError(
            "The option file at {} does not contain options.".format(
                options_path
            )
        )

    return np.array(option_list)
