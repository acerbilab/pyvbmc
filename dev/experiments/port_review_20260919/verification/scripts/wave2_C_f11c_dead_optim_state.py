"""Finding 11, two more minor observations.

(a) Which `optim_state` keys are written by `pyvbmc/` (tests excluded) and
read nowhere -- the reviewer's list of fourteen.
(b) `IterationHistory.__setitem__` deep-copies, so `load`'s truncation loop
copies the whole history a second time.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
PKG = REPO / "pyvbmc"
sources = [
    p
    for p in PKG.rglob("*.py")
    if "testing" not in p.parts and "__pycache__" not in p.parts
]

write = re.compile(
    r"""optim_state\s*\[\s*["'](\w+)["']\s*\]\s*(?:\[[^\]]*\]\s*)?=(?!=)"""
)
read = re.compile(
    r"""optim_state\s*(?:\[\s*["'](\w+)["']\s*\](?!\s*(?:\[[^\]]*\]\s*)?=(?!=))"""
    r"""|\.\s*get\s*\(\s*["'](\w+)["'])"""
)
writes, reads = {}, set()
for path in sources:
    flat = " ".join(path.read_text(encoding="utf-8", errors="replace").split())
    for name in write.findall(flat):
        writes.setdefault(name, set()).add(str(path.relative_to(REPO)))
    for a, b in read.findall(flat):
        reads.add(a or b)

dead = sorted(set(writes) - reads)
print(f"optim_state keys written in pyvbmc/ (tests excluded): {len(writes)}")
print(f"written and never read: {len(dead)}")
for name in dead:
    print(f"  {name:32s} written in {sorted(writes[name])}")

print("\n--- IterationHistory.__setitem__ deep-copies ---")
import sys

sys.path.insert(0, str(REPO))
from pyvbmc.vbmc.iteration_history import IterationHistory

h = IterationHistory(["vp"])
marker = {"payload": [1, 2, 3]}
h.record("vp", marker, 0)
stored = h["vp"][0]
print("    record() stored a copy:", stored is not marker)
h["vp"] = h["vp"][:1]
print("    a second __setitem__ copies again:", h["vp"][0] is not stored)
