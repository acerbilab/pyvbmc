"""Finding 11, two minor observations that need the right logger name.

`VBMC.__init__` builds its logger as "VBMC_init"; `optimize()` rebuilds it
as "VBMC", the process-wide name.  This checks (a) that `_init_logger`
raises `AttributeError` when a non-file handler is attached to that logger
and a log file is configured, and (b) which `display` values set which
level.
"""

import logging
from io import StringIO

import numpy as np

import pyvbmc
from pyvbmc import VBMC

print("pyvbmc:", pyvbmc.__file__)
D = 2
f = lambda x: -0.5 * np.sum(np.atleast_2d(x) ** 2, axis=1)
lb, ub = np.full((1, D), -10.0), np.full((1, D), 10.0)
plb, pub = np.full((1, D), -1.0), np.full((1, D), 1.0)
x0 = np.zeros((1, D))

print("\n--- display values (the level `_init_logger` sets on 'VBMC') ---")
names = {logging.WARN: "WARN", logging.INFO: "INFO", logging.DEBUG: "DEBUG"}
for value in ("off", "iter", "full", "notify", "final", "anything else"):
    v = VBMC(f, x0, lb, ub, plb, pub, options={"display": value})
    lg = v._init_logger()
    print(f"    display={value!r:16s} -> {names.get(lg.level, lg.level)}")
    lg.setLevel(logging.ERROR)

print("\n--- a StreamHandler on the process-wide 'VBMC' logger ---")
v = VBMC(
    f,
    x0,
    lb,
    ub,
    plb,
    pub,
    options={"log_file_name": "unused.log", "log_file_level": logging.INFO},
)
h = logging.StreamHandler(StringIO())
logging.getLogger("VBMC").addHandler(h)
try:
    v._init_logger()
    print("    accepted")
except Exception as exc:
    print(f"    {type(exc).__name__}: {exc}")
finally:
    for handler in list(logging.getLogger("VBMC").handlers):
        logging.getLogger("VBMC").removeHandler(handler)

print("\n--- two FileHandlers for the same file: the removal loop ---")
lg = logging.getLogger("VBMC")
import os

for _ in range(2):
    lg.addHandler(logging.FileHandler("unused.log", mode="a"))
print("    handlers before:", len(lg.handlers))
v._init_logger()
print(
    "    handlers after :",
    len(lg.handlers),
    "(one stale handler survives; the new one was added)",
)
for handler in list(lg.handlers):
    handler.close()
    lg.removeHandler(handler)
if os.path.exists("unused.log"):
    print("    note: 'unused.log' was created in", os.getcwd())
