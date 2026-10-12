"""Finding 8 (P1b internal F9) and finding 11 (the minor observations).

F9: `VBMC(seed=None)` draws four uint32 from NumPy's global legacy state,
which advances it, against the `seed` docstring's "NumPy's global random
state is never written".

Minor observations: `__str__` always printing `Gaussian process = None` and
`None` for the two prior lines; `_init_logger` rejecting a non-standard
numeric level and reaching for `baseFilename` on any handler attached to
the process-wide "VBMC" logger; `Options.pop` after the freeze; the
unformatted `ValueError` of `_bounds.py`; the plausible-bound warning when
`x0` sits exactly on a plausible bound; `x0` as a list; one non-finite
entry of `x0`; `Options.eval` calling a user callable by keyword; the
accepted `display` values; `iteration_history["data_trim_list"]`.
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

logging.getLogger("VBMC").setLevel(logging.ERROR)

print("\n=== F9: seed=None and the global state ===")
np.random.seed(0)
before = np.random.get_state()
VBMC(f, x0, lb, ub, plb, pub)
after = np.random.get_state()
print("bit generator:", before[0])
print("position before / after:", before[2], "/", after[2])
print("key changed:", not np.array_equal(before[1], after[1]))
print(
    "state unchanged overall:",
    before[2] == after[2] and np.array_equal(before[1], after[1]),
)
# the four draws
np.random.seed(0)
print(
    "the four draws consumed:",
    np.random.randint(0, 2**32, size=4, dtype=np.uint32),
)
# a seeded construction
np.random.seed(0)
b2 = np.random.get_state()
VBMC(f, x0, lb, ub, plb, pub, seed=42)
a2 = np.random.get_state()
print(
    "seed=42: global state untouched:",
    b2[2] == a2[2] and np.array_equal(b2[1], a2[1]),
)
# an explicit Generator
np.random.seed(0)
b3 = np.random.get_state()
VBMC(f, x0, lb, ub, plb, pub, seed=np.random.default_rng(3))
a3 = np.random.get_state()
print(
    "seed=Generator: global state untouched:",
    b3[2] == a3[2] and np.array_equal(b3[1], a3[1]),
)

print("\n=== minor: __str__ ===")
v = VBMC(f, x0, lb, ub, plb, pub)
s = str(v)
for line in s.splitlines():
    if any(
        k in line
        for k in (
            "Gaussian process",
            "log-prior",
            "prior sampler",
            "log-density",
        )
    ):
        print("   ", line.strip())
print("    VariationalPosterior has a `gp` attribute:", hasattr(v.vp, "gp"))
print(
    "    VBMC has `log_prior` /`sample_prior`:",
    hasattr(v, "log_prior"),
    hasattr(v, "sample_prior"),
)
v2 = VBMC(
    f,
    x0,
    lb,
    ub,
    plb,
    pub,
    log_prior=lambda x: 0.0,
    sample_prior=lambda n: np.zeros((n, D)),
)
print(
    "    with log_prior= and sample_prior= given:",
    hasattr(v2, "log_prior"),
    hasattr(v2, "sample_prior"),
)
for line in str(v2).splitlines():
    if "log-prior" in line or "prior sampler" in line:
        print("   ", line.strip())

print("\n=== minor: _init_logger levels and handlers ===")
try:
    VBMC(
        f,
        x0,
        lb,
        ub,
        plb,
        pub,
        options={
            "log_file_name": "unused.log",
            "log_file_level": logging.DEBUG + 5,
        },
    )
    print("    log_file_level = DEBUG+5 (15) accepted")
except Exception as exc:
    print(f"    log_file_level = DEBUG+5 (15) -> {type(exc).__name__}: {exc}")
stream_handler = logging.StreamHandler(StringIO())
logging.getLogger("VBMC").addHandler(stream_handler)
try:
    VBMC(
        f,
        x0,
        lb,
        ub,
        plb,
        pub,
        options={
            "log_file_name": "unused.log",
            "log_file_level": logging.INFO,
        },
    )
    print("    a StreamHandler on the 'VBMC' logger: accepted")
except Exception as exc:
    print(
        f"    a StreamHandler on the 'VBMC' logger -> {type(exc).__name__}: {exc}"
    )
finally:
    logging.getLogger("VBMC").removeHandler(stream_handler)
    for h in list(logging.getLogger("VBMC").handlers):
        logging.getLogger("VBMC").removeHandler(h)

print("\n=== minor: Options.pop after the freeze ===")
v3 = VBMC(f, x0, lb, ub, plb, pub)
print("    options.is_initialized:", v3.options.is_initialized)
try:
    v3.options["max_iter"] = 5
    print("    __setitem__ after freeze: accepted")
except Exception as exc:
    print(f"    __setitem__ after freeze -> {type(exc).__name__}")
popped = v3.options.pop("max_iter")
print(
    f"    options.pop('max_iter') -> {popped}; 'max_iter' in options: {'max_iter' in v3.options}"
)

print("\n=== minor: x0 forms ===")
try:
    VBMC(f, [0.0, 0.0], lb, ub, plb, pub)
    print("    x0 as a list: accepted")
except Exception as exc:
    print(f"    x0 as a list -> {type(exc).__name__}: {exc}")
try:
    VBMC(f, 0.0, lb, ub, plb, pub)
    print("    x0 as a float: accepted")
except Exception as exc:
    print(f"    x0 as a float -> {type(exc).__name__}: {exc}")
v4 = VBMC(
    f, np.array([[0.2, 0.3], [0.4, np.nan], [-0.5, 0.1]]), lb, ub, plb, pub
)
print(
    f"    x0 with one NaN among three rows -> x0 shape {v4.x0.shape}, "
    f"cache x_orig {v4.optim_state['cache']['x_orig']}"
)

print("\n=== minor: x0 exactly on a plausible bound ===")
records = []
handler = logging.Handler()
handler.emit = lambda record: records.append(record.getMessage())
lg = logging.getLogger("VBMC")
old_level = lg.level
lg.setLevel(logging.WARNING)
lg.addHandler(handler)
v5 = VBMC(f, np.array([[-1.0, 0.0]]), lb, ub, plb, pub)  # x0[0] == plb[0]
lg.removeHandler(handler)
lg.setLevel(old_level)
print("    warnings:", [r for r in records if "plausible" in r.lower()])
print(
    f"    plb after: {v5.plausible_lower_bounds}, pub after: {v5.plausible_upper_bounds}"
)

print("\n=== minor: Options.eval calls a callable by keyword ===")
try:
    VBMC(f, x0, lb, ub, plb, pub, options={"ns_ent": lambda n: 100 * n})
    print(
        "    a user callable with parameter `n` instead of `K`: construction OK "
        "(Options.eval is called later)"
    )
except Exception as exc:
    print(f"    construction -> {type(exc).__name__}: {exc}")
v6 = VBMC(f, x0, lb, ub, plb, pub, options={"ns_ent": lambda n: 100 * n})
try:
    v6.options.eval("ns_ent", {"K": 2})
    print("    options.eval('ns_ent', {'K': 2}) -> OK")
except Exception as exc:
    print(
        f"    options.eval('ns_ent', {{'K': 2}}) -> {type(exc).__name__}: {exc}"
    )

print("\n=== minor: display values and data_trim_list ===")
for value in ("off", "iter", "full", "notify", "final"):
    vv = VBMC(f, x0, lb, ub, plb, pub, options={"display": value})
    print(
        f"    display={value!r:9s} -> logger level {logging.getLogger('VBMC').level}"
    )
print(
    "    iteration_history['data_trim_list'] on a fresh instance:",
    v.iteration_history["data_trim_list"],
)
print(
    "    'data_trim_list' declared:",
    "data_trim_list" in v.iteration_history.keys(),
)
print("    optim_state['data_trim_list']:", v.optim_state["data_trim_list"])
