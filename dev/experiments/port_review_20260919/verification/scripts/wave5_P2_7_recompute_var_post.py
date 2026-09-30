"""P2i F8: every writer and reader of ``recompute_var_post`` on both sides.

Settles: the complete site list in the PyVBMC package and in MATLAB VBMC,
split into writers and readers, so that the save/restore pair of
``active_sample`` can be judged.  Read-only: it walks the two source trees.
"""
import pathlib
import re
import sys

sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from _common import banner  # noqa: E402

banner()

PY = pathlib.Path(r"C:\Users\luigi\Documents\GitHub\pyvbmc\pyvbmc")
ML = pathlib.Path(r"C:\Users\luigi\Documents\GitHub\vbmc")


def scan(root, pattern, suffixes, skip_dirs=()):
    out = []
    for p in sorted(root.rglob("*")):
        if p.suffix not in suffixes or not p.is_file():
            continue
        if any(part in skip_dirs for part in p.parts):
            continue
        for i, line in enumerate(
            p.read_text(encoding="utf8", errors="replace").splitlines(), 1
        ):
            if re.search(pattern, line):
                out.append((str(p.relative_to(root)), i, line.strip()))
    return out


print("\n=== PyVBMC (package, tests and fixtures excluded) ===")
for path, i, line in scan(
    PY, r"recompute_var_post", {".py"}, skip_dirs=("testing",)
):
    kind = "WRITE" if re.search(r'recompute_var_post"\]\s*=', line) else "read"
    if "recompute_var_post_old =" in line:
        kind = "read (save)"
    print(f"  [{kind:11s}] {path}:{i}: {line}")

print("\n=== MATLAB VBMC ===")
for path, i, line in scan(ML, r"RecomputeVarPost", {".m"}):
    stripped = line.lstrip("%").strip()
    commented = line.strip().startswith("%")
    kind = (
        "WRITE"
        if re.search(r"optimState\.RecomputeVarPost\s*=", stripped)
        else "read"
    )
    if "RecomputeVarPost_old =" in stripped:
        kind = "read (save)"
    if commented:
        kind += " [COMMENTED OUT]"
    print(f"  [{kind:25s}] {path}:{i}: {line.strip()}")
