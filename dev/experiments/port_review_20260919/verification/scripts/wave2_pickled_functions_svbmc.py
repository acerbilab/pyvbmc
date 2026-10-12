"""Which functions does dill store by value when it pickles an SVBMC object?"""

import io
import logging
import re

import dill
import dill.detect

logging.disable(logging.CRITICAL)
from pyvbmc import SVBMC
from pyvbmc.testing.svbmc._fixtures import fixture_names, load_vp

names = fixture_names()
bounded = [n for n in names if "bound" in n or "half" in n] or names
loaded = [load_vp(n) for n in names]
vps = [item[0] if isinstance(item, tuple) else item for item in loaded]
import pyvbmc

print("pyvbmc:", pyvbmc.__file__)
from collections import defaultdict

groups = defaultdict(list)
for n, v in zip(names, vps):
    groups[
        (
            v.D,
            tuple(v.parameter_transformer.lb_orig.ravel()),
            tuple(v.parameter_transformer.ub_orig.ravel()),
        )
    ].append((n, v))
key, members = max(groups.items(), key=lambda kv: len(kv[1]))
print("stacking:", [n for n, _ in members], "bounds", key[1:])
s = SVBMC([v for _, v in members])
stream = io.StringIO()
handler = logging.StreamHandler(stream)
logger = logging.getLogger("dill")
logging.disable(logging.NOTSET)
logger.addHandler(handler)
dill.detect.trace(True)
data = dill.dumps(s)
dill.detect.trace(False)
lines = stream.getvalue().splitlines()
by_value = sorted(
    {
        re.sub(r" at 0x[0-9A-Fa-f]+", "", l.split("F1:")[1].strip())
        for l in lines
        if "F1:" in l
    }
)
print(
    f"SVBMC object: {len(data)} bytes; distinct functions by value: {len(by_value)}"
)
for name in by_value:
    print("   F1", name[:120])
