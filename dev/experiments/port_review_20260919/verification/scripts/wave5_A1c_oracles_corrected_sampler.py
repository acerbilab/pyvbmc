"""W5, which oracle outputs the corrected balanced draw moves.

Runs `dev/scripts/make_oracle_fixtures.py --check --exact` in a process in
which `VariationalPosterior.sample` is replaced by the same method with
`np.sum` in the remainder weights of the balanced draw
(`variational_posterior.py:643`). The generator compares every oracle output
with the committed reference bit for bit, so the oracles it reports as moved
are the references that the corrected line would change.

Run from the repository root with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1.
"""

import inspect
import runpy
import sys
import textwrap
from pathlib import Path

import pyvbmc
import pyvbmc.variational_posterior.variational_posterior as vp_module
from pyvbmc.variational_posterior import VariationalPosterior

print("pyvbmc:", pyvbmc.__file__, flush=True)

source = textwrap.dedent(inspect.getsource(VariationalPosterior.sample))
old = "repeats_extra - sum(w_extra)"
assert source.count(old) == 1, "the line is not as the finding describes it"
namespace = dict(vars(vp_module))
exec(source.replace(old, "repeats_extra - np.sum(w_extra)"), namespace)
VariationalPosterior.sample = namespace["sample"]
print(
    "VariationalPosterior.sample replaced by the corrected method", flush=True
)

root = Path(pyvbmc.__file__).resolve().parents[1]
script = root / "dev" / "scripts" / "make_oracle_fixtures.py"
sys.argv = [str(script), "--check", "--exact"]
runpy.run_path(str(script), run_name="__main__")
