"""W6-A2, the window of the burn-in statistics of the slice sampler.

`gplite/private/slicesamplebnd.m:362` accumulates the sums behind the adapted
widths over `ii > burn/2` (`ii` from 1) and divides them by `floor(burn/2)`
(`:367-368`); `slice_sample.py:562` accumulates over `burn/2 <= i` (`i` from
0) and divides by the same number (`:568`). Part 1 counts the terms on both
sides. Part 2 runs one recorded GP fit per stored oracle state (the `gp_fit`
oracle) with gpyreg's sampler as it is and with MATLAB's window and MATLAB's
answer to a negative estimate (a complex square root, after which `:371`
keeps the old widths of every coordinate), installed in this process alone
by editing the source of `SliceSampler.sample`, at the burn-in the state
recorded (15 in every state: `thin * 3`, the burn-in of a PyVBMC fit that
does not recompute the variational posterior) and at 16. With the even
burn-in the two must agree bit for bit, which checks that the edit changes
nothing else. Part 2 also reports the smallest width gpyreg adapts to, for
the floor at zero that gpyreg puts under the estimate (`:574`). Part 3 is the
case in which that floor is reached by construction: with a burn-in of 2 or
3 gpyreg's window holds one iteration, the estimate is exactly zero, every
width is set to zero and the chain stops moving.

Run from anywhere with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1.
"""

import inspect
import textwrap
from pathlib import Path

import gpyreg
import gpyreg.slice_sample as ss_module
import numpy as np
from gpyreg.slice_sample import SliceSampler

import pyvbmc
from pyvbmc.testing.oracles._oracles import ORACLES
from pyvbmc.testing.oracles._state import (
    build_state,
    load_snapshot,
    snapshot_names,
)

print("pyvbmc:", pyvbmc.__file__)
print("gpyreg:", gpyreg.__file__, flush=True)
np.set_printoptions(precision=3, suppress=False, linewidth=130)

# Part 1: the terms that each side accumulates.
print("\nburn: terms MATLAB, terms gpyreg, divisor")
for burn in range(1, 21):
    matlab = sum(1 for ii in range(1, burn + 1) if burn / 2 < ii <= burn)
    python = sum(1 for i in range(burn) if burn / 2 <= i < burn)
    print(f"{burn:4d}: {matlab:2d} {python:2d} {int(np.floor(burn / 2)):2d}")

# Part 2: MATLAB's window and its answer to a negative estimate, as an edit
# of gpyreg's method.
source = textwrap.dedent(inspect.getsource(SliceSampler.sample))
EDITS = [
    ("if burn / 2 <= i < burn:", "if burn / 2 < i + 1 <= burn:"),
    (
        textwrap.dedent(
            """\
            new_widths = np.fmin(
                5
                * np.sqrt(
                    np.maximum(
                        xx_sq_sum / burn_stored
                        - (xx_sum / burn_stored) ** 2,
                        0,
                    )
                ),
                self.UB_out - self.LB_out,
            )
            if not np.all(np.isreal(new_widths)):
                new_widths = self.widths
            """
        ),
        textwrap.dedent(
            """\
            w6_est = xx_sq_sum / burn_stored - (xx_sum / burn_stored) ** 2
            self._w6_negative = int(np.sum(w6_est < 0))
            if self._w6_negative:
                new_widths = self.widths
            else:
                new_widths = np.fmin(
                    5 * np.sqrt(w6_est), self.UB_out - self.LB_out
                )
            """
        ),
    ),
]


def edit(src, old, new):
    # The second edit is matched at the indentation it has in the method.
    for pad in range(0, 40, 4):
        old_p = textwrap.indent(old, " " * pad)
        if src.count(old_p) == 1:
            return src.replace(old_p, textwrap.indent(new, " " * pad))
    raise AssertionError(f"not there once: {old[:40]!r}")


matlab_source = source
for old, new in EDITS:
    matlab_source = edit(matlab_source, old, new)
# Inside a class of the same name, so that the private names of the method
# are mangled as they are in gpyreg.
namespace = dict(vars(ss_module))
exec(
    "class SliceSampler:\n" + textwrap.indent(matlab_source, "    "), namespace
)
matlab_sample = namespace["SliceSampler"].sample
gpyreg_sample = SliceSampler.sample

record = []
force_burn = {"value": None}


def recording(sample):
    def wrapped(self, N, thin=1, burn=None):
        if force_burn["value"] is not None:
            burn = force_burn["value"]
        self._w6_negative = None
        out = sample(self, N, thin, burn)
        record.append(
            {
                "burn": burn,
                "widths": self.widths.copy(),
                "negative": self._w6_negative,
                "fixed": self.LB_out == self.UB_out,
            }
        )
        return out

    return wrapped


fixtures = (
    Path(pyvbmc.__file__).resolve().parent / "testing" / "oracles" / "fixtures"
)
oracle = ORACLES["gp_fit"]
for forced in (None, 16):
    force_burn["value"] = forced
    print(f"\n##### burn-in {'as recorded' if forced is None else forced}")
    for name in snapshot_names(fixtures):
        snap = load_snapshot(fixtures / name)
        results = {}
        for label, sample in (
            ("gpyreg", gpyreg_sample),
            ("matlab", matlab_sample),
        ):
            SliceSampler.sample = recording(sample)
            record.clear()
            out = oracle(build_state(snap))
            results[label] = (out["hyp"], [dict(r) for r in record])
        SliceSampler.sample = gpyreg_sample
        (hyp_g, rec_g), (hyp_m, rec_m) = results["gpyreg"], results["matlab"]
        if not rec_g:
            print(f"=== {name}: no sampling")
            continue
        g, m = rec_g[0], rec_m[0]
        free = ~g["fixed"]
        print(f"=== {name}: burn {g['burn']}, samples {hyp_g.shape[0]}")
        print(
            "  same samples:",
            np.array_equal(hyp_g, hyp_m),
            "| coordinates with a negative MATLAB estimate:",
            m["negative"],
            "of",
            int(free.sum()),
        )
        print("  widths gpyreg:", g["widths"][free])
        print("  widths MATLAB:", m["widths"][free])
        ratio = m["widths"][free] / g["widths"][free]
        print(
            f"  MATLAB over gpyreg: {ratio.min():.3g} to {ratio.max():.3g};"
            f" smallest gpyreg width {g['widths'][free].min():.3g}"
        )
        if hyp_g.shape == hyp_m.shape:
            print(
                "  largest difference of a sample's coordinate:",
                f"{np.abs(hyp_g - hyp_m).max():.3g};",
                "of the means over the samples:",
                f"{np.abs(hyp_g.mean(0) - hyp_m.mean(0)).max():.3g}",
                flush=True,
            )

# Part 3: a window of one iteration.
print("\n##### a standard normal in two dimensions, gpyreg's sampler as it is")
for burn in (2, 3, 4, 5, 15):
    sampler = SliceSampler(
        lambda x: -0.5 * np.sum(x**2),
        np.zeros(2),
        options={"display": "off", "diagnostics": False},
        rng=np.random.default_rng(1),
    )
    out = sampler.sample(6, burn=burn)
    distinct = np.unique(out["samples"], axis=0).shape[0]
    print(
        f"burn {burn:2d}: widths after the burn-in {sampler.widths},"
        f" distinct samples {distinct} of 6"
    )
