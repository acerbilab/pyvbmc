"""P9-14. Which statements of `pyvbmc/testing/vbmc/test_vbmc_init.py` compute a
comparison and discard it?

Settles: the exact present line numbers of every bare `np.isclose` /
`np.allclose` expression statement in that file, the enclosing test
function, and which of them compare the log joint with likelihood + prior.
"""
import ast
import pathlib

import pyvbmc

print("pyvbmc.__file__ =", pyvbmc.__file__)
path = (
    pathlib.Path(pyvbmc.__file__).parent
    / "testing"
    / "vbmc"
    / "test_vbmc_init.py"
)
src = path.read_text(encoding="utf-8")
tree = ast.parse(src)
lines = src.splitlines()
for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
    for node in ast.walk(fn):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            f = node.value.func
            name = getattr(f, "attr", getattr(f, "id", ""))
            if name in ("isclose", "allclose"):
                text = " ".join(
                    l.strip() for l in lines[node.lineno - 1 : node.end_lineno]
                )
                print(
                    f"{fn.name}  lines {node.lineno}-{node.end_lineno}: {text}"
                )
print()
print("--- real assertions of log_joint == likelihood + prior ---")
for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
    for node in ast.walk(fn):
        if isinstance(node, ast.Assert):
            text = " ".join(
                l.strip() for l in lines[node.lineno - 1 : node.end_lineno]
            )
            if "log_joint" in text and ("prior" in text or "log_pdf" in text):
                print(f"{fn.name}  line {node.lineno}: {text}")
