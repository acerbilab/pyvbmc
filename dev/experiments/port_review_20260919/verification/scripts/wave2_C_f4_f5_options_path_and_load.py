"""Findings 4 and 5 (P1b internal F12 and F10).

F12: option names are validated for the `options=` dict but not for the
`options_path=` route nor for `load(new_options=)`.

F10: after `load`, `vbmc.parameter_transformer`, `vp.parameter_transformer`
and `function_logger.parameter_transformer` are three distinct objects, and
`vbmc.parameter_transformer` is the pickled live one, not the selected
iteration's.  Checked on a fresh round trip in this scratch directory and on
the committed run `pyvbmc/testing/vbmc/test_vbmc_save_static.pkl`, which is
opened read-only.
"""

import logging
import os
import tempfile
from pathlib import Path

import numpy as np

import pyvbmc
from pyvbmc import VBMC

logging.getLogger("VBMC").setLevel(logging.ERROR)
print("pyvbmc:", pyvbmc.__file__)

SCRATCH = Path(tempfile.mkdtemp(prefix="wave2_C_f4_f5_"))
REPO = Path(__file__).resolve().parents[5]

D = 2
f = lambda x: -0.5 * np.sum(np.atleast_2d(x) ** 2, axis=1)
lb, ub = np.full((1, D), -10.0), np.full((1, D), 10.0)
plb, pub = np.full((1, D), -1.0), np.full((1, D), 1.0)
x0 = np.zeros((1, D))

print("\n=== F12: option-name validation ===")
# (a) options= dict with a typo
try:
    VBMC(f, x0, lb, ub, plb, pub, options={"max_fun_eval": 123})
    print("options={'max_fun_eval': 123}  -> accepted (no error)")
except Exception as exc:
    print(f"options={{'max_fun_eval': 123}}  -> {type(exc).__name__}: {exc}")

# (b) the same typo through options_path=
ini = SCRATCH / "typo_options.ini"
ini.write_text(
    "[TypoOptions]\n"
    "# Max number of target fcn evals (typo of max_fun_evals)\n"
    "max_fun_eval = 123\n"
    "# An option that exists nowhere\n"
    "completely_made_up = 7\n",
    encoding="utf-8",
)
v = VBMC(f, x0, lb, ub, plb, pub, options_path=str(ini))
print(
    f"options_path with 'max_fun_eval'/'completely_made_up' -> accepted; "
    f"options['max_fun_eval']={v.options['max_fun_eval']}, "
    f"options['completely_made_up']={v.options['completely_made_up']}, "
    f"max_fun_evals still {v.options['max_fun_evals']}"
)

print("\n=== F10: transformer sharing across save/load ===")
v2 = VBMC(f, x0, lb, ub, plb, pub, options={"max_iter": 2})
print(
    "fresh instance: vbmc.pt is vp.pt ->",
    v2.parameter_transformer is v2.vp.parameter_transformer,
    "| vbmc.pt is fl.pt ->",
    v2.parameter_transformer is v2.function_logger.parameter_transformer,
)
pkl = SCRATCH / "fresh_roundtrip.pkl"
if pkl.exists():
    os.remove(pkl)
v2.save(pkl)
v3 = VBMC.load(pkl)
print(
    "fresh round trip: vbmc.pt is vp.pt ->",
    v3.parameter_transformer is v3.vp.parameter_transformer,
    "| vbmc.pt is fl.pt ->",
    v3.parameter_transformer is v3.function_logger.parameter_transformer,
    "| vp.rng is vbmc.rng ->",
    v3.vp.rng is v3.rng,
    "| iteration =",
    v3.iteration,
)

# (c) load(new_options=) with a typo
v4 = VBMC.load(pkl, new_options={"max_fun_eval": 999})
print(
    "load(new_options={'max_fun_eval': 999}) -> accepted; "
    f"options['max_fun_eval']={v4.options['max_fun_eval']}, "
    f"max_fun_evals={v4.options['max_fun_evals']}"
)

static = REPO / "pyvbmc" / "testing" / "vbmc" / "test_vbmc_save_static.pkl"
print(f"\n--- stored finished run {static.name} (read-only) ---")
s = VBMC.load(static)
print("last iteration:", s.iteration)
print(
    "vbmc.pt is vp.pt ->",
    s.parameter_transformer is s.vp.parameter_transformer,
    "| vbmc.pt is fl.pt ->",
    s.parameter_transformer is s.function_logger.parameter_transformer,
    "| vp.pt is fl.pt ->",
    s.vp.parameter_transformer is s.function_logger.parameter_transformer,
)


def pt_sig(pt):
    return (
        np.round(np.atleast_1d(pt.mu).ravel(), 6).tolist(),
        np.round(np.atleast_1d(pt.delta).ravel(), 6).tolist(),
        (
            None
            if pt.R_mat is None
            else np.round(np.asarray(pt.R_mat).ravel(), 6).tolist()
        ),
        (
            None
            if pt.scale is None
            else np.round(np.asarray(pt.scale).ravel(), 6).tolist()
        ),
    )


print("vbmc.pt signature:", pt_sig(s.parameter_transformer))
print("vp.pt   signature:", pt_sig(s.vp.parameter_transformer))
print(
    "equal by value:",
    pt_sig(s.parameter_transformer) == pt_sig(s.vp.parameter_transformer),
)

# Are there warps in this stored run?  Compare every recorded VP's transformer.
sigs = []
for i, vp in enumerate(s.iteration_history["vp"]):
    sigs.append((i, pt_sig(vp.parameter_transformer)))
distinct = {repr(sig) for _, sig in sigs}
print(
    f"\nrecorded iterations: {len(sigs)}; distinct transformers among them: {len(distinct)}"
)
if len(distinct) > 1:
    for i, sig in sigs:
        print("  iter", i, sig)

for it in (0, s.iteration):
    si = VBMC.load(static, iteration=it)
    same = pt_sig(si.parameter_transformer) == pt_sig(
        si.vp.parameter_transformer
    )
    print(
        f"load(iteration={it}): vbmc.pt is vp.pt -> "
        f"{si.parameter_transformer is si.vp.parameter_transformer}; "
        f"equal by value -> {same}"
    )
    print(
        f"  __str__ x0 line: {si.parameter_transformer.inverse(si.x0)}"
        f"   (vp.pt would give {si.vp.parameter_transformer.inverse(si.x0)})"
    )
