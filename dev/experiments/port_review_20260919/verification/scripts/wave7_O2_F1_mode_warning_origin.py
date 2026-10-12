"""F1: where the RuntimeWarning printed by mode() on a narrow posterior
comes from, for both values of orig_flag (warnings turned into errors to
get the traceback), and what a default warning filter shows."""
import os
import subprocess
import sys
import traceback
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from wave7_O2_common import banner, build_vp

banner()
rs = np.random.default_rng(3)
base_mu = rs.normal(0, 0.6, (2, 3))
for flag in (False, True):
    vp = build_vp(
        base_mu * 0.003, [1.0, 1.3, 0.8], np.full(2, 0.003), [0.4, 0.33, 0.27]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            vp.mode(orig_flag=flag)
            print(f"orig_flag={flag}: no RuntimeWarning", flush=True)
        except RuntimeWarning as e:
            fr = traceback.extract_tb(e.__traceback__)[-1]
            print(
                f"orig_flag={flag}: RuntimeWarning '{e}' raised at "
                f"{fr.filename.split('site-packages')[-1]}:{fr.lineno}",
                flush=True,
            )
# a fresh interpreter with Python's default filters: what reaches stderr
code = (
    "import sys; sys.path.insert(0, %r)\n"
    "import numpy as np\n"
    "from wave7_O2_common import build_vp\n"
    "rs = np.random.default_rng(3); m = rs.normal(0, 0.6, (2, 3))\n"
    "vp = build_vp(m * 0.003, [1.0, 1.3, 0.8], np.full(2, 0.003), [0.4, 0.33, 0.27])\n"
    "print('mode:', vp.mode())\n"
) % os.path.dirname(os.path.abspath(__file__))
r = subprocess.run(
    [sys.executable, "-c", code],
    capture_output=True,
    text=True,
    env=dict(os.environ),
)
print("default filters, vp.mode() stdout:", r.stdout.strip(), flush=True)
print("default filters, stderr:", r.stderr.strip()[-400:], flush=True)
