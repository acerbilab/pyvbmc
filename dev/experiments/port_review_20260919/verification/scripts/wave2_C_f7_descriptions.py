"""Finding 7 (P1b internal F11): option descriptions truncated at `=` or `:`.

`_read_config_file` (`pyvbmc/vbmc/options.py:401-437`) reads the `.ini`
files with `configparser.ConfigParser(comment_prefixes="",
allow_no_value=True)`, so a `# description` line survives as an entry -- but
configparser still splits every line at its first delimiter (`=` or `:`),
and only the part before it becomes the key, hence the description.  This
script lists every option of the two shipped files whose stored description
is shorter than the `# ` line in the file.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
import pyvbmc
from pyvbmc.vbmc.options import _read_config_file

print("pyvbmc:", pyvbmc.__file__)

for rel in (
    "pyvbmc/vbmc/option_configs/basic_vbmc_options.ini",
    "pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini",
):
    path = REPO / rel
    # The descriptions as the file writes them: the `# ` line preceding
    # each non-comment, non-section line.
    full = {}
    pending = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("["):
            continue
        if line.startswith("#"):
            pending = line.lstrip("# ").strip()
        else:
            key = line.split("=", 1)[0].strip()
            full[key] = pending
            pending = None

    stored = {k: d for k, _v, d in _read_config_file(str(path))}
    print(f"\n### {rel}: {len(stored)} options")
    n = 0
    for key, text in stored.items():
        if full.get(key) != text:
            n += 1
            print(f"  {key}")
            print(f"    stored: {text!r}")
            print(f"    file  : {full.get(key)!r}")
    print(f"  -- {n} truncated")
