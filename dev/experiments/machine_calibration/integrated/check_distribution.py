import json
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent
wheels = list((root / "dist").glob("*.whl"))
sdists = list((root / "dist").glob("*.tar.gz"))
assert len(wheels) == len(sdists) == 1, (wheels, sdists)
required = [
    "pyvbmc/calibration/__init__.py",
    "pyvbmc/calibration/profile.py",
    "pyvbmc/calibration/_api.py",
    "pyvbmc/calibration/_cache.py",
    "pyvbmc/calibration/_campaign.py",
]
with zipfile.ZipFile(wheels[0]) as z:
    names = z.namelist()
    assert all(p in names for p in required)
    assert not any("/testing/" in p for p in names)
    metadata = z.read(
        next(p for p in names if p.endswith(".dist-info/METADATA"))
    ).decode()
    for dependency in ("filelock", "platformdirs", "threadpoolctl"):
        assert "Requires-Dist: " + dependency in metadata
with tarfile.open(sdists[0]) as archive:
    names = archive.getnames()
    for p in required + [
        "pyvbmc/testing/calibration/test_cache_process.py",
        "pyvbmc/testing/calibration/test_campaign.py",
    ]:
        assert any(name.endswith("/" + p) for name in names), p
site = root / "wheel_site"
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--no-compile",
        "--upgrade",
        "--target",
        str(site),
        str(wheels[0]),
    ],
    check=True,
)
env = os.environ.copy()
env["PYTHONPATH"] = str(site)
env["PYVBMC_CACHE_DIR"] = str(root / "wheel_unused_cache")
code = """import inspect,json,sys
from pathlib import Path
import numpy as np
import pyvbmc
from pyvbmc.entropy.entmc_vbmc import entmc_vbmc
assert 'wheel_site' in pyvbmc.__file__,pyvbmc.__file__
assert 'pyvbmc.calibration._campaign' not in sys.modules
assert 'pyvbmc.calibration._cache' not in sys.modules
assert str(inspect.signature(pyvbmc.calibrate)) == '(*, verbose=True)'
profile=pyvbmc.CalibrationProfile(pdf_chunk_elements=16384,entropy_grad_chunk_elements=16384,entropy_value_chunk_elements=16384)
vp=pyvbmc.VariationalPosterior(2,3,calibration=profile,rng=123)
assert vp.pdf(np.zeros((2,2)),orig_flag=False).dtype==np.float64
value,gradient=entmc_vbmc(vp,10)
assert np.isfinite(value) and gradient.dtype==np.float64
assert vp.calibration_profile is profile
assert profile.source=='explicit'
assert 'pyvbmc.calibration._campaign' not in sys.modules
print(json.dumps({'source':pyvbmc.__file__,'calibrate_signature':str(inspect.signature(pyvbmc.calibrate)),'profile':profile.to_dict(),'kernel_import_check':'passed'}))
"""
r = subprocess.run(
    [sys.executable, "-c", code],
    cwd=root,
    env=env,
    text=True,
    capture_output=True,
)
print(r.stdout)
print(r.stderr)
r.check_returncode()
(root / "distribution_check.json").write_text(
    json.dumps(
        {
            "wheel": wheels[0].name,
            "sdist": sdists[0].name,
            "clean_install": json.loads(r.stdout),
            "status": "passed",
        },
        indent=2,
    )
)
