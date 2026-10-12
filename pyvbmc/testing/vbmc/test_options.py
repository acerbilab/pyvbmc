import ast
import copy
import functools
import logging
import re
import tokenize
from math import ceil
from pathlib import Path

import numpy as np
import pytest

from pyvbmc import VBMC
from pyvbmc.acquisition_functions import AcqFcnLog, AcqFcnVIQR
from pyvbmc.vbmc import Options
from pyvbmc.vbmc.options import INERT_OPTIONS
from pyvbmc.vbmc.vbmc import _CONSTRUCTION_ONLY_OPTIONS

options_path = Path(__file__).parent.parent.parent.joinpath(
    "vbmc", "option_configs"
)
basic_options_path = options_path.joinpath("basic_vbmc_options.ini")
advanced_options_path = options_path.joinpath("advanced_vbmc_options.ini")


def _declared_option_names():
    """The option names declared by the two shipped ini files."""
    names = set()
    for path in (basic_options_path, advanced_options_path):
        for line in path.read_text().splitlines():
            line = line.strip()
            if line == "" or line.startswith("#") or line.startswith("["):
                continue
            names.add(line.split("=", 1)[0].strip())
    return names


def _shipped_options(user_options, D=2):
    """Build the shipped options as ``VBMC.__init__`` does."""
    options = Options(basic_options_path, {"D": D}, user_options)
    options.load_options_file(advanced_options_path, {"D": D})
    options.validate_run_limits()
    options.update_defaults()
    options.validate_option_names([basic_options_path, advanced_options_path])
    return options


def test_options_no_user_options():
    default_options_path = options_path.joinpath("test_options.ini")
    options = Options(default_options_path, {"D": 2})
    assert options.get("bar") == 40
    assert len(options.get("useroptions")) == 0
    assert options.get("foo") == "iter"
    assert options.get("fooD") == 4


def test_options_user_options():
    default_options_path = options_path.joinpath("test_options.ini")
    user_options = {"foo": "iter2"}
    options = Options(default_options_path, {"D": 2}, user_options)
    assert options.get("bar") == 40
    assert len(options.get("useroptions")) == 1
    assert options.get("foo") == "iter2"
    assert options.get("fooD") == 4
    assert "foo" in options.get("useroptions")


@pytest.mark.parametrize("as_dict", [False, True])
def test_options_built_from_another_options_leave_it_alone(as_dict):
    """The options of one run can be given as the user options of another,
    as an `Options` object or a dict copied from one. Each keeps a set of
    user options of its own: building the second leaves the first's set as
    it was, and the name `useroptions` is not an option the user set."""
    default_options_path = options_path.joinpath("test_options.ini")
    options_1 = Options(default_options_path, {"D": 2}, {"foo": "iter2"})
    before = set(options_1["useroptions"])

    source = dict(options_1) if as_dict else options_1
    options_2 = Options(default_options_path, {"D": 2}, source)

    assert options_1["useroptions"] == before
    assert options_2["useroptions"] is not options_1["useroptions"]
    assert "useroptions" not in options_2["useroptions"]
    assert options_2.get("foo") == "iter2"


def test_init_from_existing_options():
    default_options_path = options_path.joinpath("test_options.ini")
    user_options = {"foo": "iter2"}
    options_1 = Options(default_options_path, {"D": 2}, user_options)
    options_2 = Options.init_from_existing_options(
        default_options_path, {"D": 2}, options_1
    )
    assert options_1 == options_2
    assert len(options_1) == len(options_2)


def test_init_from_existing_options_modified():
    default_options_path = options_path.joinpath("test_options.ini")
    user_options = {"foo": "iter2"}
    options_1 = Options(default_options_path, {"D": 2}, user_options)
    options_1["bar"] = 80
    options_2 = Options.init_from_existing_options(
        default_options_path, {"D": 2}, options_1
    )
    assert options_1 != options_2
    assert options_1.get("bar") == 80
    assert options_2.get("bar") == 40
    assert options_1.get("foo") == "iter2"
    assert options_2.get("foo") == "iter2"
    assert options_1.get("fooD") == 4
    assert options_2.get("fooD") == 4


def test_init_from_existing_options_without_user_options():
    default_options_path = options_path.joinpath("test_options.ini")
    options_1 = Options(default_options_path, {"D": 2})
    options_1["bar"] = 80
    options_2 = Options.init_from_existing_options(
        default_options_path, {"D": 2}, options_1
    )
    assert options_1 != options_2
    assert options_1.get("bar") == 80
    assert options_2.get("bar") == 40
    assert options_1.get("foo") == "iter"
    assert options_2.get("foo") == "iter"
    assert options_1.get("fooD") == 4
    assert options_2.get("fooD") == 4


def test_init_from_existing_options_without_other_options():
    default_options_path = options_path.joinpath("test_options.ini")
    options_1 = Options.init_from_existing_options(
        default_options_path, {"D": 2}
    )
    options_2 = Options(default_options_path, {"D": 2})
    assert options_1 == options_2
    assert len(options_1) == len(options_2)


def test_init_with_specify_target_noise():
    """Turning on specify_target_noise should adjust defaults."""
    D = 1
    options = {
        "specify_target_noise": True,
        "active_sample_gp_update": "foo",  # But don't touch user options!
    }
    vbmc1 = VBMC(
        lambda x, y: (x + y, y),
        np.zeros((1, D)),
        -np.ones((1, D)),
        np.ones((1, D)),
        -0.5 * np.ones((1, D)),
        0.5 * np.ones((1, D)),
    )
    vbmc2 = VBMC(
        lambda x, y: (x + y, y),
        np.zeros((1, D)),
        -np.ones((1, D)),
        np.ones((1, D)),
        -0.5 * np.ones((1, D)),
        0.5 * np.ones((1, D)),
        options=options,
    )

    # Check that default options are changed:
    assert vbmc1.options != vbmc2.options
    assert vbmc2.options["max_fun_evals"] == ceil(
        vbmc1.options["max_fun_evals"] * 1.5
    )
    assert vbmc2.options["tol_stable_count"] == ceil(
        vbmc1.options["tol_stable_count"] * 1.5
    )
    assert len(vbmc2.options["search_acq_fcn"]) == 1
    assert isinstance(vbmc2.options["search_acq_fcn"][0], AcqFcnVIQR)
    assert vbmc2.options["active_sample_vp_update"] == True

    # Check that user-specified option is unchanged:
    assert vbmc2.options["active_sample_gp_update"] == "foo"

    # Check defaults for non-noisy target:
    assert len(vbmc1.options["search_acq_fcn"]) == 1
    assert isinstance(vbmc1.options["search_acq_fcn"][0], AcqFcnLog)
    assert vbmc1.options["active_sample_vp_update"] == False
    assert vbmc1.options["active_sample_gp_update"] == False


@pytest.mark.parametrize("key", ["max_fun_evals", "max_iter"])
@pytest.mark.parametrize("value", [0, -1, 7.5, -np.inf, np.nan, None])
def test_a_run_limit_must_be_a_positive_integer(key, value):
    """``misc/setupoptions_vbmc.m:109-114`` rejects a MaxFunEvals or a
    MaxIter that is not a positive integer."""
    with pytest.raises(ValueError) as execinfo:
        _shipped_options({key: value})
    assert f"The option {key} needs to be a positive integer" in (
        execinfo.value.args[0]
    )


@pytest.mark.parametrize("key", ["max_fun_evals", "max_iter"])
@pytest.mark.parametrize("value", [1, 40, 40.0, np.int64(40), np.inf])
def test_a_run_limit_accepts_an_integer_value(key, value):
    """MATLAB compares the value with its rounding, so a floating value
    that lands on an integer passes, and so does an infinite limit."""
    options = _shipped_options({key: value, "min_iter": 0})
    assert options[key] == value


@pytest.mark.parametrize(
    "value", [-1, 2.5, np.inf, -np.inf, np.nan, None, "3"]
)
def test_min_iter_must_be_a_finite_non_negative_integer(value):
    """``min_iter`` is a number of iterations, 0 when a run has no
    minimum. Any other value is refused where it is given, before it can
    reach ``max_iter``, which is raised to ``min_iter`` when it is lower.
    An infinite minimum is refused as well: the minimum holds back every
    termination, the one on the budget of evaluations included (MATLAB
    VBMC, ``private/vbmc_termination.m:98-99``) unless the run accounts for
    evaluations made before it (``precomputed_evaluations`` or an
    ``initialization_cost``), so a run under it would never stop, or would
    stop only at its budget."""
    with pytest.raises(ValueError) as execinfo:
        _shipped_options({"min_iter": value, "max_iter": 2})
    assert "The option min_iter needs to be a finite non-negative integer" in (
        execinfo.value.args[0]
    )


@pytest.mark.parametrize("value", [0, 3, 3.0, np.int64(3)])
def test_min_iter_accepts_a_non_negative_integer_value(value):
    """As for the other limits on a run, a floating value that lands on an
    integer passes; 0 sets no minimum."""
    options = _shipped_options({"min_iter": value})
    assert options["min_iter"] == value


def test_the_default_min_iter_is_accepted():
    D = 3
    options = _shipped_options({}, D=D)
    assert options["min_iter"] == D  # `min_iter = D` in the shipped file


def test_min_iter_takes_a_boolean_as_max_iter_does():
    """The two limits on iterations agree on `True`: both accept it, or
    both refuse it."""

    def accepted(user_options):
        try:
            _shipped_options(user_options)
        except ValueError:
            return False
        return True

    assert accepted({"min_iter": True}) == accepted(
        {"max_iter": True, "min_iter": 0}
    )


def test_max_iter_below_min_iter_is_raised_to_it(caplog):
    """``misc/setupoptions_vbmc.m:115-119`` raises MaxIter to MinIter and
    says so."""
    caplog.set_level(logging.WARNING)
    options = _shipped_options({"max_iter": 2, "min_iter": 7})
    assert options["max_iter"] == 7
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "max_iter" in message and "7" in message for message in messages
    )


def test_max_iter_at_min_iter_is_left_alone(caplog):
    caplog.set_level(logging.WARNING)
    options = _shipped_options({"max_iter": 7, "min_iter": 7})
    assert options["max_iter"] == 7
    messages = [record.getMessage() for record in caplog.records]
    assert not any("max_iter" in message for message in messages)


@pytest.mark.parametrize(
    "user_options",
    [{"specify_target_noise": True}, {"uncertainty_handling": True}],
)
def test_noisy_defaults_apply_at_either_noise_level(user_options):
    """``misc/setupoptions_vbmc.m:143-163`` changes five defaults whenever
    the noise handling is on, which covers a noise level VBMC infers as
    well as one the target supplies."""
    noiseless = _shipped_options({})
    noisy = _shipped_options(user_options)
    assert noisy["max_fun_evals"] == ceil(noiseless["max_fun_evals"] * 1.5)
    assert noisy["tol_stable_count"] == ceil(
        noiseless["tol_stable_count"] * 1.5
    )
    assert noisy["active_sample_gp_update"] is True
    assert noisy["active_sample_vp_update"] is True
    assert len(noisy["search_acq_fcn"]) == 1
    assert isinstance(noisy["search_acq_fcn"][0], AcqFcnVIQR)


def test_noisy_defaults_leave_the_values_the_user_set():
    """Only a default is changed, as the ``updated`` list of
    ``misc/setupoptions_vbmc.m`` requires."""
    noisy = _shipped_options(
        {
            "uncertainty_handling": True,
            "max_fun_evals": 17,
            "search_acq_fcn": [AcqFcnLog()],
        }
    )
    assert noisy["max_fun_evals"] == 17
    assert isinstance(noisy["search_acq_fcn"][0], AcqFcnLog)
    assert noisy["active_sample_vp_update"] is True


@pytest.mark.parametrize(
    "user_options",
    [{"specify_target_noise": True}, {"uncertainty_handling": True}],
)
def test_noisy_defaults_accept_an_unlimited_budget(user_options):
    """A noisy run without a limit on its evaluations is a valid
    configuration: the limit is the user's, so the noisy default that
    would replace it is not needed, and the other defaults still apply."""
    noiseless = _shipped_options({})
    noisy = _shipped_options({"max_fun_evals": np.inf, **user_options})
    assert noisy["max_fun_evals"] == np.inf
    assert noisy["tol_stable_count"] == ceil(
        noiseless["tol_stable_count"] * 1.5
    )
    assert isinstance(noisy["search_acq_fcn"][0], AcqFcnVIQR)


def test_a_noiseless_run_keeps_the_noiseless_defaults():
    noiseless = _shipped_options({})
    assert noiseless["active_sample_gp_update"] is False
    assert noiseless["active_sample_vp_update"] is False
    assert isinstance(noiseless["search_acq_fcn"][0], AcqFcnLog)


@pytest.mark.parametrize(
    "D, expected",
    [(1, 10), (2, 10), (9, 10), (10, 20), (15, 20), (19, 20), (20, 30)],
)
def test_fun_eval_start_default(D, expected):
    """The initial design holds 10 evaluations up to nine dimensions and
    the next multiple of 10 above ``D`` from there on, as in MATLAB VBMC
    (``10*ceil((D+1)/10)``)."""
    vbmc = VBMC(
        lambda x: -0.5 * np.sum(x**2),
        np.zeros((1, D)),
        np.full((1, D), -np.inf),
        np.full((1, D), np.inf),
        np.full((1, D), -1.0),
        np.full((1, D), 1.0),
    )
    assert vbmc.options["fun_eval_start"] == expected


def _code_without_comments(path):
    """The module's source with its comments dropped."""
    pieces = []
    with tokenize.open(path) as handle:
        for token in tokenize.generate_tokens(handle.readline):
            if token.type != tokenize.COMMENT:
                pieces.append(token.string)
    return "".join(pieces)


# The forms of a read of an option, matched in the code with its whitespace
# removed: the name quoted and subscripted (other than as the target of an
# assignment), or fetched with ``get`` or ``eval``, from a mapping whose
# name ends in ``options`` (``options``, ``self.options``, ``new_options``)
# or from ``self``; or the name tested for membership in such a mapping,
# with ``in`` or ``not in``.
_OPTION_READS = (
    r"(?:options|self)"
    r"(?:\[['\"](\w+)['\"]\](?!=(?!=))|\.(?:get|eval)\(['\"](\w+)['\"])",
    r"['\"](\w+)['\"](?:not)?in[\w.]*options",
)

# Names read through an options mapping that are not options: an
# ``Options`` object keeps the names the user set under ``useroptions``.
_NAMES_THAT_ARE_NOT_OPTIONS = {"useroptions"}


def _names_read_as_options():
    """The names that the package, outside its tests, reads as options, in
    one of the forms of ``_OPTION_READS``. A mention in a comment is not a
    read."""
    package_path = options_path.parent.parent
    sources = [
        path
        for path in package_path.rglob("*.py")
        if "testing" not in path.relative_to(package_path).parts
    ]
    text = re.sub(
        r"\s+", "", "\n".join(_code_without_comments(path) for path in sources)
    )
    names = set()
    for pattern in _OPTION_READS:
        for match in re.finditer(pattern, text):
            names.update(group for group in match.groups() if group)
    return names


def test_inert_options_are_the_declared_options_nothing_reads():
    """``INERT_OPTIONS`` lists exactly the declared options that no module
    of the package reads, so that a newly dead option, or a newly read
    registered one, fails here.

    A read is one of the forms of ``_OPTION_READS``; inside
    :class:`Options` the mapping is ``self``. A key of another mapping that
    happens to carry an option's name, such as an entry of ``optim_state``,
    is not a read of the option, and neither is a mention in a comment."""
    unread = _declared_option_names() - _names_read_as_options()
    assert unread == set(INERT_OPTIONS)


def test_every_option_the_package_reads_is_declared():
    """Every name that the package reads as an option is declared in the
    shipped files. Option names are checked against those files wherever
    they are given, so no run can set an undeclared one: a read of it
    always finds it absent, and the code that it guards never runs."""
    undeclared = (
        _names_read_as_options()
        - _declared_option_names()
        - _NAMES_THAT_ARE_NOT_OPTIONS
    )
    assert not undeclared, (
        "read as options, declared in neither shipped file: "
        f"{sorted(undeclared)}"
    )


# The functions that run while a ``VBMC`` object is built, and the value
# checks and warnings that construction and ``load`` share. An option whose
# every read lies in them is read at construction alone.
_CONSTRUCTION_SITES = {
    "VBMC.__init__",
    "VBMC._init_optim_state",
    "VBMC._initialize_precomputed_evaluations",
    "Options.update_defaults",
    "Options._warn_ignored_noise_size",
}
# The two ``Options`` methods that read an option on their caller's behalf:
# a call of one is a read of these options at the site of the call.
_READS_BY_METHOD = {
    "uncertainty_handling_on": (
        "uncertainty_handling",
        "specify_target_noise",
    ),
    "integer_vars_mask": ("integer_vars",),
}
# Reads that no run acts on, by option and site: ``active_sample`` reads
# the option into two locals that nothing uses, and ``load`` reads the
# stored ``integer_vars``, ``uncertainty_handling`` and
# ``specify_target_noise`` to rewrite the forms that release 1.0.4 wrote.
_READS_THAT_DO_NOT_COUNT = {
    ("active_search_bound", "active_sample"),
    ("integer_vars", "VBMC.load"),
    ("uncertainty_handling", "VBMC.load"),
    ("specify_target_noise", "VBMC.load"),
}


def _option_read_sites(names):
    """Map each option name to the functions of the package that read it.

    A read is ``options[name]``, ``options.get(name)`` or
    ``options.eval(name, ...)`` with a quoted name, on a mapping whose name
    contains ``options``, the same on ``self`` inside :class:`Options`, or
    a call of one of the methods of ``_READS_BY_METHOD``. A read written
    otherwise, through a variable key, an alias whose name lacks
    ``options`` or ``__getitem__``, is not seen. The site is the function
    that holds the read, as ``Class.method``, or the name of a module-level
    function."""
    package_path = options_path.parent.parent
    sites = {name: set() for name in names}

    def is_options(node, in_options_class):
        if isinstance(node, ast.Name):
            return "options" in node.id or (
                in_options_class and node.id == "self"
            )
        if isinstance(node, ast.Attribute):
            return "options" in node.attr
        return False

    for path in package_path.rglob("*.py"):
        if "testing" in path.relative_to(package_path).parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))

        def visit(node, class_name, site):
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    visit(child, node.name, None)
                return
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and site is None
            ):
                site = f"{class_name}.{node.name}" if class_name else node.name
                if class_name == "Options" and node.name in _READS_BY_METHOD:
                    # The method's own reads are counted at its callers.
                    return
            in_options_class = class_name == "Options"
            read = None
            if isinstance(node, ast.Subscript) and is_options(
                node.value, in_options_class
            ):
                read = node.slice
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.args
            ):
                if node.func.attr in ("get", "eval") and is_options(
                    node.func.value, in_options_class
                ):
                    read = node.args[0]
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _READS_BY_METHOD
                and site is not None
            ):
                for name in _READS_BY_METHOD[node.func.attr]:
                    sites[name].add(site)
            if (
                isinstance(read, ast.Constant)
                and read.value in sites
                and site is not None
            ):
                sites[read.value].add(site)
            for child in ast.iter_child_nodes(node):
                visit(child, class_name, site)

        visit(tree, None, None)
    return sites


def test_construction_only_options_are_the_options_only_construction_reads():
    """``_CONSTRUCTION_ONLY_OPTIONS``, the options of which ``VBMC.load``
    takes only the value the run stores, lists exactly the declared options
    whose every read is made while a ``VBMC`` object is built or by a value
    check: an option that a later iteration starts to read, or a new option
    read at construction alone, fails here."""
    names = _declared_option_names() - set(INERT_OPTIONS)
    sites = _option_read_sites(names)
    construction_only = set()
    for name, readers in sites.items():
        readers = {
            site
            for site in readers
            if (name, site) not in _READS_THAT_DO_NOT_COUNT
        }
        sites[name] = readers
        if readers and all(
            site in _CONSTRUCTION_SITES or site.startswith("VBMC._validate_")
            for site in readers
        ):
            construction_only.add(name)
    differing = construction_only ^ set(_CONSTRUCTION_ONLY_OPTIONS)
    assert not differing, "; ".join(
        f"{name}: {'in' if name in _CONSTRUCTION_ONLY_OPTIONS else 'not in'}"
        f" the tuple, read at {sorted(sites.get(name, ()))}"
        for name in sorted(differing)
    )


def test_inert_option_away_from_its_default_warns(caplog):
    """A value supplied for an inert option is named as having no effect."""
    caplog.set_level(logging.WARNING)
    options = _shipped_options({"double_gp": True})
    assert options["double_gp"] is True
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "double_gp" in message and "no effect" in message
        for message in messages
    )


def test_separate_search_gp_is_inert(caplog):
    """The separate search GP with a constant mean is a development option
    of MATLAB VBMC that PyVBMC does not implement, so a value given for it
    has no effect and is reported as having none."""
    caplog.set_level(logging.WARNING)
    options = _shipped_options({"separate_search_gp": True})
    assert options["separate_search_gp"] is True
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "separate_search_gp" in message and "no effect" in message
        for message in messages
    )


@pytest.mark.parametrize(
    "name, value", [("gp_int_mean_fun", 1), ("proposal_fcn", print)]
)
def test_the_integrated_mean_and_the_proposal_function_are_inert(
    caplog, name, value
):
    """The integrated mean function of MATLAB VBMC's GP is not ported, and
    its proposal function for the search is an option that MATLAB VBMC
    stores and never reads, so a value given for either has no effect and
    is reported as having none."""
    caplog.set_level(logging.WARNING)
    options = _shipped_options({name: value})
    assert options[name] is value
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        name in message and "no effect" in message for message in messages
    )


def test_inert_option_at_its_default_does_not_warn(caplog):
    """Repeating the default of an inert option changes nothing and is
    silent, so that an option dictionary recorded by an earlier run loads
    without notices."""
    caplog.set_level(logging.WARNING)
    _shipped_options({"double_gp": False, "noise_shaping_threshold": 20})
    messages = [record.getMessage() for record in caplog.records]
    assert not any(
        "double_gp" in message or "noise_shaping_threshold" in message
        for message in messages
    )


def test_every_inert_option_at_its_default_is_silent(caplog):
    """An option dictionary recorded by an earlier run repeats every
    default, including the one that is a lambda, and passes through
    without notices."""
    defaults = _shipped_options({})
    repeated = {name: defaults[name] for name in INERT_OPTIONS}
    assert callable(repeated["annealed_gp_mean"])

    caplog.set_level(logging.WARNING)
    _shipped_options(repeated)
    messages = [record.getMessage() for record in caplog.records]
    assert not any(
        name in message for name in INERT_OPTIONS for message in messages
    )


def test_option_that_is_read_does_not_warn(caplog):
    """An option the algorithm reads has an effect and is not reported."""
    caplog.set_level(logging.WARNING)
    options = _shipped_options({"max_iter": 3})
    assert options["max_iter"] == 3
    messages = [record.getMessage() for record in caplog.records]
    assert not any("max_iter" in message for message in messages)


def test_description_keeps_the_whole_comment_line(tmp_path):
    """The description of an option is the comment line above it, in full,
    delimiters of the ini format included."""
    path = tmp_path.joinpath("described.ini")
    path.write_text(
        "[Described]\n"
        "# Explicit noise handling (0: none; 1: unknown; 2: provided)\n"
        "max_iter = 3\n"
        "# Number of GP samples when GP is stable (0 = optimize)\n"
        "min_iter = 1\n"
    )
    options = Options(path, {"D": 2})
    assert options.descriptions["max_iter"] == (
        "Explicit noise handling (0: none; 1: unknown; 2: provided)"
    )
    assert options.descriptions["min_iter"] == (
        "Number of GP samples when GP is stable (0 = optimize)"
    )


@pytest.mark.parametrize(
    "name, description",
    [
        (
            "search_optimizer",
            'Local optimizer of the acquisition search: "cmaes" (CMA-ES, '
            "replaced by a bounded scalar search where the problem has one "
            'variable) or "none" (no local search)',
        ),
        (
            "stable_gp_samples",
            "Number of GP samples when GP is stable (0 = optimize)",
        ),
        (
            "upper_gp_length_factor",
            "Upper bound on GP input lengths based on plausible box "
            "(0 = ignore)",
        ),
    ],
)
def test_shipped_descriptions_are_stored_in_full(name, description):
    """The descriptions users read come from the ini files as written."""
    options = _shipped_options({})
    assert options.descriptions[name] == description


@pytest.mark.parametrize(
    "name",
    [
        "heavy_tail_search_frac",
        "mvn_search_frac",
        "hpd_search_frac",
        "box_search_frac",
        "search_cache_frac",
    ],
)
def test_the_search_fractions_are_described_with_their_range_and_sum(name):
    """The comment above an option is its user documentation, and the
    five fractions of the acquisition search are refused outside [0, 1]
    or claiming more than the whole search set together."""
    description = _shipped_options({}).descriptions[name]
    assert "[0, 1]" in description
    assert "at most 1" in description
    assert "variational posterior" in description


def test_integer_variables_are_described_with_the_prior_they_need():
    """A prior given with ``prior=`` has to cover the hard bounds, which
    for an integer variable sit half an integer outside its range."""
    description = _shipped_options({}).descriptions["integer_vars"]
    assert "prior=" in description
    assert "UniformBox(-0.5, 10.5)" in description


def test_an_option_the_user_set_keeps_its_description():
    """The description belongs to the option, whoever set its value.

    ``print(options)`` lists the options the user set, so those are the
    descriptions a user reads, for the options of either shipped file.
    """
    defaults = _shipped_options({})
    options = _shipped_options({"max_fun_evals": 120, "tol_skl": 0.02})
    for name in ("max_fun_evals", "tol_skl"):
        assert defaults.descriptions[name]
        assert options.descriptions[name] == defaults.descriptions[name]
        assert f"({defaults.descriptions[name]})" in str(options)
    assert "(None)" not in str(options)


def test_a_user_file_without_comments_keeps_the_shipped_descriptions(
    tmp_path,
):
    """An options file of the user need not repeat the descriptions."""
    path = tmp_path.joinpath("mine.ini")
    path.write_text("[Mine]\ntol_skl = 0.02\n")
    defaults = _shipped_options({})
    options = Options(basic_options_path, {"D": 2})
    options.load_options_file(advanced_options_path, {"D": 2})
    options.load_options_file(path, {"D": 2}, as_user_options=True)
    assert options["tol_skl"] == 0.02
    assert options.descriptions["tol_skl"] == defaults.descriptions["tol_skl"]


def test__str__and__repr__():
    default_options_path = options_path.joinpath("test_options.ini")
    options = Options(default_options_path, {"D": 2})
    one_option_str = "bar: 40 (Bar description)"
    assert one_option_str in options.__repr__()
    assert "None (use default options)." in options.__str__()
    options.__repr__()


def test_del():
    default_options_path = options_path.joinpath("test_options.ini")
    options = Options(default_options_path, {"D": 2})
    options.pop("foo")
    assert "foo" not in options


def test_initialized_options_refuse_removal():
    """Options are fixed after initialization: a key can no more be
    removed than it can be set, by whichever method of a mapping."""
    default_options_path = options_path.joinpath("test_options.ini")
    options = Options(default_options_path, {"D": 2})
    options.validate_option_names([default_options_path])
    assert options.is_initialized
    n_options = len(options)

    with pytest.raises(AttributeError, match="after initialization"):
        options.pop("foo")
    with pytest.raises(AttributeError, match="after initialization"):
        del options["foo"]
    with pytest.raises(AttributeError, match="after initialization"):
        options.popitem()
    with pytest.raises(AttributeError, match="after initialization"):
        options.clear()
    assert len(options) == n_options
    assert options["foo"] is not None

    # The override that setting an option has.
    options.__delitem__("foo", force=True)
    assert "foo" not in options


def test_eval_callable():
    default_options_path = options_path.joinpath("test_options.ini")

    def bar_function(T, S):
        return S, T

    user_options = {"foo": lambda Y, K: (Y, K), "bar": bar_function}
    options = Options(default_options_path, {"D": 2}, user_options)
    assert (2, 3) == options.eval("foo", {"K": 3, "Y": 2})
    assert (2, 3) == options.eval("foo", {"Y": 2, "K": 3})
    assert (3, 2) == options.eval("bar", {"T": 2, "S": 3})
    assert (5, 10) == options.eval("bar", {"S": 5, "T": 10})


def _positional_only_k(K, /):
    return 10 * K


def test_eval_passes_a_single_parameter_by_position():
    """A callable option evaluated with one parameter, and without a
    parameter that takes it by its name, receives it by position, as
    ``misc/evaloption_vbmc.m`` calls ``option(N)``, so the function's
    parameter may have any name: ``lambda n: ...`` for ``ns_ent``, and
    ``lambda unkn: ...`` for ``adaptive_k``, the name that release 1.0.4
    passed. A parameter that takes no keyword, and a callable whose
    signature cannot be read, take the value by position too."""
    options = _shipped_options(
        {
            "ns_ent": lambda n: 100 * n,
            "adaptive_k": lambda unkn: unkn + 1,
            "k_fun_max": lambda n_eff: n_eff / 2,
            "ns_elbo": _positional_only_k,
            "ns_ent_fine": int,
        }
    )
    assert options.eval("ns_ent", {"K": 3}) == 300
    assert options.eval("adaptive_k", {"K": 4}) == 5
    assert options.eval("k_fun_max", {"N": 10}) == 5
    assert options.eval("ns_elbo", {"K": 4}) == 40
    assert options.eval("ns_ent_fine", {"K": 4.0}) == 4


def _two_parameters(a, K):
    return (a, K)


@pytest.mark.parametrize(
    "function, value",
    [
        (lambda scale=100, K=1: scale * K, 400),
        (functools.partial(_two_parameters, a=1), (1, 4)),
        (lambda *, K: K + 1, 5),
        (lambda **kwargs: kwargs["K"] + 2, 6),
    ],
    ids=["later-parameter", "partial", "keyword-only", "var-keyword"],
)
def test_eval_passes_a_single_parameter_by_the_name_it_is_taken_by(
    function, value
):
    """A callable option that takes the parameter by its name, as a
    parameter so named or through ``**kwargs``, receives it by keyword, as
    release 1.0.4 passed it, so that it computes what it computed there,
    also when the parameter is not its first."""
    options = _shipped_options({"ns_ent": function})
    assert options.eval("ns_ent", {"K": 4}) == value


def test_eval_passes_several_parameters_by_keyword():
    """With several parameters the names decide which value goes where,
    whatever the order of the function's parameters."""
    options = _shipped_options(
        {
            "active_importance_sampling_mcmc_samples": (
                lambda D, K, n_vars: (D, K, n_vars)
            )
        }
    )
    assert options.eval(
        "active_importance_sampling_mcmc_samples",
        {"K": 1, "n_vars": 2, "D": 3},
    ) == (3, 1, 2)


def test_eval_constant():
    default_options_path = options_path.joinpath("test_options.ini")
    user_options = {"ns_ent": (5, 3)}
    options = Options(default_options_path, {"D": 2, "Y": 3}, user_options)
    assert (5, 3) == options.eval("ns_ent", {"K": 2})


def test_eval_callable_args_missing():
    default_options_path = options_path.joinpath("test_options.ini")
    user_options = {"ns_ent": lambda Y, K: (Y, K)}
    options = Options(default_options_path, {"D": 2}, user_options)
    with pytest.raises(TypeError):
        options.eval("ns_ent", {})


def test_eval_callable_too_many_args():
    default_options_path = options_path.joinpath("test_options.ini")
    user_options = {"bar": lambda Y, K: (Y, K)}
    options = Options(default_options_path, {"D": 2}, user_options)
    with pytest.raises(TypeError):
        options.eval("bar", {"U": 2, "S": 2, "T": 4})


def test_load_options_file():
    evaluation_parameters = {"D": 2}
    user_options = {"foo": "testuseroptions", "foo2": "testuseroptions2"}
    basic_test_options = options_path.joinpath("test_options.ini")
    options = Options(basic_test_options, evaluation_parameters, user_options)
    advanced_test_options = options_path.joinpath("test_options2.ini")
    options.load_options_file(advanced_test_options, evaluation_parameters)
    assert options.get("bar") == 40
    assert len(options.get("useroptions")) == 2
    assert options.get("foo") == "testuseroptions"
    assert options.get("fooD") == 4
    assert options.get("bar2") == 80
    assert options.get("foo2") == "testuseroptions2"
    assert options.get("fooD2") == 200


def test_validate_option_names():
    evaluation_parameters = {"D": 2}
    user_options = {"foo": "testuseroptions", "foo2": "testuseroptions2"}
    basic_test_options = options_path.joinpath("test_options.ini")
    options = Options(basic_test_options, evaluation_parameters, user_options)
    advanced_test_options = options_path.joinpath("test_options2.ini")
    options.load_options_file(advanced_test_options, evaluation_parameters)
    # should go fine
    options.validate_option_names([basic_test_options, advanced_test_options])
    # raise error
    with pytest.raises(ValueError) as execinfo1:
        options.validate_option_names([basic_test_options])


def test_validate_option_names_unknown_user_options():
    evaluation_parameters = {"D": 2}
    user_options = {"failoption": "testuseroptions"}
    basic_test_options = options_path.joinpath("test_options.ini")
    options = Options(basic_test_options, evaluation_parameters, user_options)
    advanced_test_options = options_path.joinpath("test_options2.ini")
    options.load_options_file(advanced_test_options, evaluation_parameters)
    with pytest.raises(ValueError) as execinfo1:
        options.validate_option_names(
            [basic_test_options, advanced_test_options]
        )
    assert "The option failoption does not exist." in execinfo1.value.args[0]


def test_load_options_invalid_path():
    evaluation_parameters = {"D": 2}
    basic_test_options = options_path.joinpath("test_options.ini")
    options = Options(basic_test_options, evaluation_parameters)
    non_existing_path = options_path.joinpath("does_not_exist.ini")
    with pytest.raises(ValueError) as execinfo1:
        options.load_options_file(non_existing_path, evaluation_parameters)
    assert "does not exist." in execinfo1.value.args[0]


def test_options_copy():
    default_options_path = options_path.joinpath("test_options.ini")
    test_list = [1, 2, 3, 4]
    user_options = {"foo": test_list}
    options = Options(default_options_path, {"D": 2}, user_options)

    options_copy = copy.copy(options)
    # Check that we have a copy:
    assert options == options_copy
    assert options_copy.get("foo") == test_list
    # Check that the copy is not deep:
    assert options.get("foo") is options_copy.get("foo")

    assert options_copy.get("bar") == 40
    assert len(options_copy.get("useroptions")) == 1
    assert options_copy.get("fooD") == 4
    assert "foo" in options_copy.get("useroptions")


def test_options_deepcopy():
    default_options_path = options_path.joinpath("test_options.ini")
    test_list = [1, 2, 3, 4]
    user_options = {"foo": test_list}
    options = Options(default_options_path, {"D": 2}, user_options)

    options_copy = copy.deepcopy(options)
    # Check that we have a copy:
    assert options == options_copy
    assert options_copy.get("foo") == test_list
    # Check that the copy is deep:
    assert options.get("foo") is not options_copy.get("foo")

    assert options_copy.get("bar") == 40
    assert len(options_copy.get("useroptions")) == 1
    assert options_copy.get("fooD") == 4
    assert "foo" in options_copy.get("useroptions")


def test_prevent_option_set_post_init():
    evaluation_parameters = {"D": 2}
    user_options = {"foo": "testuseroptions", "foo2": "testuseroptions2"}
    basic_test_options = options_path.joinpath("test_options.ini")
    options = Options(basic_test_options, evaluation_parameters, user_options)
    advanced_test_options = options_path.joinpath("test_options2.ini")
    options.load_options_file(advanced_test_options, evaluation_parameters)
    # Validate option names to complete initialization:
    options.validate_option_names([basic_test_options, advanced_test_options])
    options.__setitem__("bar", 1, force=True)
    try:
        # Should fail with AttributeError and not AssertionError:
        options["bar"] = 2
        assert False
    except AttributeError:
        pass
    try:
        # Ditto:
        options.__setitem__("bar", 2)
        assert False
    except AttributeError:
        pass
