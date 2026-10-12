"""Finding 6 (P1a internal F9, P1b internal F8, P1b comparison F6/F10).

Recomputes, independently of `INERT_OPTIONS`, the set of options declared in
the two shipped `.ini` files that no module of `pyvbmc/` (tests excluded)
reads through an options mapping, and the set of names read through an
options mapping that no `.ini` declares.  Also reports, for each declared
name, whether the bare quoted string occurs anywhere in the package text --
the rule that `test_inert_options_are_the_declared_options_nothing_reads`
uses -- so the gap between the two rules is visible.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
PKG = REPO / "pyvbmc"
INI = [
    PKG / "vbmc" / "option_configs" / "basic_vbmc_options.ini",
    PKG / "vbmc" / "option_configs" / "advanced_vbmc_options.ini",
]

import sys

sys.path.insert(0, str(REPO))
from pyvbmc.vbmc.options import INERT_OPTIONS, _read_config_file

declared = []
for path in INI:
    for key, _value, _desc in _read_config_file(str(path)):
        declared.append(key)
assert len(declared) == len(set(declared)), "duplicate declaration"
declared = set(declared)
print(f"declared options: {len(declared)}")

sources = [
    p
    for p in PKG.rglob("*.py")
    if "testing" not in p.parts and "__pycache__" not in p.parts
]
print(f"source files scanned: {len(sources)} (pyvbmc/, tests excluded)")

# Reads through an options mapping: <something>options[ "name" ] /
# .get("name") / .eval("name", ...).  Whitespace is flattened first so that
# a subscript broken over lines is still seen.
read_pat = re.compile(
    r"""(?<![\w.])(?:\w+\.)*options\s*(?:\[\s*|\.\s*get\s*\(\s*|\.\s*eval\s*\(\s*)["'](\w+)["']"""
)
# Any quoted occurrence of the name anywhere (the guard test's rule).
quoted = {}
options_reads = {}
for path in sources:
    text = path.read_text(encoding="utf-8", errors="replace")
    flat = " ".join(text.split())
    for name in read_pat.findall(flat):
        options_reads.setdefault(name, []).append(path)
    for name in declared:
        if f'"{name}"' in text or f"'{name}'" in text:
            quoted.setdefault(name, []).append(path)

unread = sorted(declared - set(options_reads))
undeclared_reads = sorted(set(options_reads) - declared)

print(f"\ndeclared but never read through an options mapping: {len(unread)}")
for name in unread:
    in_inert = "registered" if name in INERT_OPTIONS else "NOT REGISTERED"
    q = quoted.get(name, [])
    where = ", ".join(str(p.relative_to(REPO)) for p in q) if q else "-"
    print(f"  {name:38s} {in_inert:15s} quoted in: {where}")

print(
    f"\nINERT_OPTIONS entries that ARE read through an options mapping: "
    f"{sorted(INERT_OPTIONS & set(options_reads))}"
)
print(
    f"INERT_OPTIONS size: {len(INERT_OPTIONS)}; recomputed unread: {len(unread)}"
)

print(
    f"\nread through an options mapping but declared in no .ini: "
    f"{len(undeclared_reads)}"
)
for name in undeclared_reads:
    where = ", ".join(
        sorted({str(p.relative_to(REPO)) for p in options_reads[name]})
    )
    print(f"  {name:38s} {where}")

# Which of the unread names are copied into optim_state, and is that copy read?
print("\n--- optim_state keys named after an option, and their readers ---")
os_pat = re.compile(
    r"""optim_state\s*(?:\[\s*["'](\w+)["']\s*\](?!\s*[-+*/]?=[^=])|\.\s*get\s*\(\s*["'](\w+)["'])"""
)
os_write = re.compile(r"""optim_state\s*\[\s*["'](\w+)["']\s*\]\s*=""")
for name in (
    "temperature",
    "proposal_fcn",
    "lcb_max_vec",
    "int_mean_fun",
    "entropy_force_switch",
    "diagnostics",
    "recompute_lcb_max",
):
    readers, writers = [], []
    for path in sources:
        flat = " ".join(
            path.read_text(encoding="utf-8", errors="replace").split()
        )
        if name in {g for m in os_pat.findall(flat) for g in m if g}:
            readers.append(str(path.relative_to(REPO)))
        if name in os_write.findall(flat):
            writers.append(str(path.relative_to(REPO)))
    print(
        f"  optim_state[{name!r}]: written in {writers or '-'}; read in {readers or '-'}"
    )
