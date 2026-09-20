"""Which routes to a noisy run receive the noisy-target defaults?

Wave-2 findings P1b comparison F1 and P1b internal F4. MATLAB
(misc/setupoptions_vbmc.m:127-163) applies five changed defaults whenever
UncertaintyHandling is on, which covers both noise levels: inferred noise
(level 1, UncertaintyHandling set by the user) and user-provided noise
(level 2, SpecifyTargetNoise, which switches UncertaintyHandling on).
PyVBMC's Options.update_defaults tests specify_target_noise only, and
VBMC.__init__ calls it before the options file of options_path= is read.
"""

import os
import sys
import tempfile

import numpy as np

import pyvbmc
from pyvbmc import VBMC

print("pyvbmc:", pyvbmc.__file__, flush=True)
scratch = tempfile.mkdtemp(prefix="wave2_noisy_defaults_")
D = 3


def f(x):
    return -0.5 * np.sum(x**2), 1.0


def build(options=None, options_path=None):
    return VBMC(
        f,
        np.zeros((1, D)),
        -10 * np.ones((1, D)),
        10 * np.ones((1, D)),
        -np.ones((1, D)),
        np.ones((1, D)),
        options=dict(display="off", **(options or {})),
        options_path=options_path,
        seed=0,
    )


ini_level2 = os.path.join(scratch, "noisy_level2.ini")
with open(ini_level2, "w") as fh:
    fh.write("[AdvancedOptions]\n# Noise provided by the target\n")
    fh.write("specify_target_noise = True\n")
ini_level1 = os.path.join(scratch, "noisy_level1.ini")
with open(ini_level1, "w") as fh:
    fh.write("[AdvancedOptions]\n# Inferred noise\n")
    fh.write("uncertainty_handling = [1]\n")

routes = {
    "noiseless (default)": build(),
    "level 2, options= dict": build({"specify_target_noise": True}),
    "level 2, options_path= file": build(options_path=ini_level2),
    "level 1, options= dict": build({"uncertainty_handling": [1]}),
    "level 1, options_path= file": build(options_path=ini_level1),
}
matlab = {
    "noiseless (default)": (50 * (D + 2), 60, False, False, "acqf"),
    "level 1 or 2": (int(np.ceil(1.5 * 50 * (D + 2))), 90, True, True, "viqr"),
}
print(
    f"\nD = {D}. MATLAB: noiseless {matlab['noiseless (default)']}; "
    f"either noise level {matlab['level 1 or 2']}\n"
)
print(
    f"{'route':30s} level  max_fun_evals  tol_stable_count  "
    "gp_update  vp_update  search_acq_fcn"
)
for name, v in routes.items():
    o = v.options
    print(
        f"{name:30s} {v.optim_state['uncertainty_handling_level']:5d}  "
        f"{o['max_fun_evals']:13d}  {o['tol_stable_count']:16d}  "
        f"{str(o['active_sample_gp_update']):9s}  "
        f"{str(o['active_sample_vp_update']):9s}  "
        f"{[type(a).__name__ for a in o['search_acq_fcn']]}"
    )
sys.stdout.flush()
