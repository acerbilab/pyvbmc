import json
import sys
import time
from pathlib import Path

root = Path("dev/scripts/runs/calibration_integration_20260909")
label = sys.argv[1]
started = time.perf_counter()
import pyvbmc

imports = time.perf_counter() - started
if label == "lookup":
    started = time.perf_counter()
    from pyvbmc.calibration._cache import resolve_cached_profile

    profile = resolve_cached_profile()
    cold = time.perf_counter() - started
    repeats = []
    for _ in range(12):
        started = time.perf_counter()
        profile = resolve_cached_profile()
        repeats.append(time.perf_counter() - started)
    result = {
        "imports_seconds": imports,
        "cold_cached_lookup_seconds": cold,
        "repeated_lookup_seconds": repeats,
        "profile": profile.to_dict(),
    }
    (root / "cached_startup.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
else:
    started = time.perf_counter()
    profile = pyvbmc.calibrate()
    elapsed = time.perf_counter() - started
    result = {
        "imports_seconds": imports,
        "api_elapsed_seconds": elapsed,
        "profile": profile.to_dict(),
    }
    (root / f"campaign_{label}_profile.json").write_text(
        json.dumps(result, indent=2)
    )
    assert profile.status == "complete", result
    assert profile.cache_path, result
    (root / f"campaign_{label}.json").write_bytes(
        Path(profile.cache_path).read_bytes()
    )
    print(json.dumps(result, indent=2))
