"""Settles row G1-X: ``VBMC.load(new_options=...)`` runs neither
``Options.update_defaults`` nor ``_init_optim_state``.

``pyvbmc/vbmc/vbmc.py: load`` (``:3196-3208``) validates the supplied
names, updates the options, checks the run limits and calls
``_validate_option_values``.  ``VBMC.__init__`` additionally calls
``self.options.update_defaults()`` (``:430``) and
``self.optim_state = self._init_optim_state()`` (``:539``).  An option that
only those two read is therefore accepted by ``load`` and takes no coherent
effect.

The script builds a VBMC instance on a trivial target, saves it WITHOUT
running ``optimize()``, and loads it with a series of ``new_options``,
printing for each the option value, the pieces of ``optim_state`` that
``_init_optim_state`` derives from it, the five noisy-target defaults that
``update_defaults`` sets, and the function logger's noise flag.  It writes
its pickle to a temporary directory of its own and deletes it.

Run from anywhere.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np

import pyvbmc
from pyvbmc import VBMC

print("pyvbmc:", pyvbmc.__file__)

D = 2
LB = np.full((1, D), -5.0)
UB = np.full((1, D), 5.0)
PLB = np.full((1, D), -2.0)
PUB = np.full((1, D), 2.0)
x0 = np.zeros((1, D))


def log_joint(x):
    x = np.atleast_2d(x)
    return -0.5 * np.sum(x**2, axis=1)


FIELDS = [
    "uncertainty_handling_level",
    "gp_noise_fun",
    "gp_mean_fun",
    "warmup",
    "vp_K",
    "entropy_switch",
    "tol_gp_var",
    "lb_search",
]
NOISY_DEFAULTS = [
    "max_fun_evals",
    "tol_stable_count",
    "active_sample_gp_update",
    "active_sample_vp_update",
    "search_acq_fcn",
]


def show(tag, vbmc):
    print(f"  [{tag}]")
    print(
        f"    options['uncertainty_handling'] = "
        f"{vbmc.options['uncertainty_handling']!r}"
        f"   specify_target_noise = "
        f"{vbmc.options['specify_target_noise']!r}"
    )
    print(
        f"    options.uncertainty_handling_on() = "
        f"{vbmc.options.uncertainty_handling_on()}"
    )
    for f in FIELDS:
        v = vbmc.optim_state.get(f)
        if isinstance(v, np.ndarray):
            v = np.round(v, 4)
        print(f"    optim_state[{f!r}] = {v!r}")
    for f in NOISY_DEFAULTS:
        v = vbmc.options[f]
        if f == "search_acq_fcn":
            v = [type(a).__name__ for a in v]
        print(f"    options[{f!r}] = {v!r}")
    print(
        f"    function_logger.noise_flag = {vbmc.function_logger.noise_flag}"
        f"   .uncertainty_handling_level = "
        f"{vbmc.function_logger.uncertainty_handling_level}"
    )


tmp = Path(tempfile.mkdtemp(prefix="G1_X_"))
try:
    print("\n=== A. a noiseless instance, constructed ===")
    v0 = VBMC(
        log_joint,
        x0,
        LB,
        UB,
        PLB,
        PUB,
        options={"display": "off", "max_iter": 2},
    )
    show("constructed, noiseless", v0)
    f = tmp / "run.pkl"
    v0.save(f)

    print(
        "\n=== B. loaded with new_options={'uncertainty_handling': True} ==="
    )
    v1 = VBMC.load(f, new_options={"uncertainty_handling": True})
    show("loaded, uncertainty_handling=True", v1)

    print(
        "\n=== C. loaded with new_options={'specify_target_noise': True} ==="
    )
    try:
        v2 = VBMC.load(f, new_options={"specify_target_noise": True})
        show("loaded, specify_target_noise=True", v2)
    except Exception as exc:  # noqa: BLE001
        print(f"  {type(exc).__name__}: {exc}")

    print("\n=== D. loaded with new_options={'gp_mean_fun': 'const'} ===")
    v3 = VBMC.load(f, new_options={"gp_mean_fun": "const"})
    print(
        f"    options['gp_mean_fun'] = {v3.options['gp_mean_fun']!r}"
        f"   optim_state['gp_mean_fun'] = "
        f"{v3.optim_state['gp_mean_fun']!r}"
    )

    print(
        "\n=== E. loaded with new_options={'k_warmup': 7, 'warmup': False,"
        " 'entropy_switch': True, 'active_search_bound': 4} ==="
    )
    v4 = VBMC.load(
        f,
        new_options={
            "k_warmup": 7,
            "warmup": False,
            "entropy_switch": True,
            "active_search_bound": 4,
        },
    )
    for f2 in ("vp_K", "warmup", "entropy_switch", "lb_search"):
        v = v4.optim_state.get(f2)
        if isinstance(v, np.ndarray):
            v = np.round(v, 4)
        print(
            f"    options-driven optim_state[{f2!r}] = {v!r}"
            f"   (option now {v4.options[{'vp_K': 'k_warmup', 'warmup': 'warmup', 'entropy_switch': 'entropy_switch', 'lb_search': 'active_search_bound'}[f2]]!r})"
        )

    print(
        "\n=== F. for contrast, an instance CONSTRUCTED with"
        " uncertainty_handling=True ==="
    )
    v5 = VBMC(
        log_joint,
        x0,
        LB,
        UB,
        PLB,
        PUB,
        options={
            "display": "off",
            "max_iter": 2,
            "uncertainty_handling": True,
        },
    )
    show("constructed, uncertainty_handling=True", v5)

    print(
        "\n=== F2. the validity check that _init_optim_state performs is"
        " skipped too ==="
    )
    try:
        vbad = VBMC(
            log_joint,
            x0,
            LB,
            UB,
            PLB,
            PUB,
            options={"display": "off", "gp_mean_fun": "nonsense"},
        )
        print("    construction with gp_mean_fun='nonsense' -> accepted")
    except Exception as exc:  # noqa: BLE001
        print(
            f"    construction with gp_mean_fun='nonsense' -> "
            f"{type(exc).__name__}: {str(exc)[:90]}"
        )
    try:
        vbad = VBMC.load(f, new_options={"gp_mean_fun": "nonsense"})
        print(
            "    load with gp_mean_fun='nonsense' -> accepted, "
            f"options={vbad.options['gp_mean_fun']!r}, "
            f"optim_state={vbad.optim_state['gp_mean_fun']!r}"
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"    load with gp_mean_fun='nonsense' -> "
            f"{type(exc).__name__}: {str(exc)[:90]}"
        )
    try:
        vbad = VBMC.load(f, new_options={"integer_vars": [True, False]})
        print(
            "    load with integer_vars=[True, False] -> accepted, "
            f"optim_state['integer_vars'] = "
            f"{vbad.optim_state['integer_vars']!r}"
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"    load with integer_vars -> {type(exc).__name__}: "
            f"{str(exc)[:90]}"
        )

    print(
        "\n=== G. an option a continued run does re-read every iteration"
        " (max_iter, max_fun_evals) ==="
    )
    v6 = VBMC.load(f, new_options={"max_iter": 9, "max_fun_evals": 123})
    print(
        f"    options['max_iter'] = {v6.options['max_iter']}"
        f"   options['max_fun_evals'] = {v6.options['max_fun_evals']}"
        f"   optim_state['max_fun_evals'] = "
        f"{v6.optim_state['max_fun_evals']}"
    )
finally:
    shutil.rmtree(tmp, ignore_errors=True)
    print("\n(temporary directory removed)")
