# Package-integrated calibration verification

These artifacts describe the PyVBMC package implementation on
`dev-machine-calibration`, based on `f7f0ce1`. They are separate from the
historical source-cloning experiment in the parent directory. See the
[results](../../../results/2026-09-09-machine-local-calibration.md) and
[implementation plan](../../../plans/machine-local-calibration.md).

Run the explicit public API on a quiet machine, with no concurrent heavy
process. For comparable measurements, set all three BLAS variables below.
This is a measurement environment, not a requirement imposed on library users.

```powershell
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYVBMC_CACHE_DIR = "$PWD/dev/scripts/runs/calibration_reproduce/cache"
.venv/Scripts/python.exe -c "import pyvbmc; p = pyvbmc.calibrate(); print(p.to_dict())"
```

Run the same command again for a fresh campaign. Copy the JSON at the returned
`cache_path` before rerunning if both full reports are needed. Cache records
contain timing rounds, held-out controls, all numerical/memory gates and
machine/backend identity. Profiles contain compact provenance and the three
fixed settings. A complete all-default profile is a successful measurement.

`kernel_overhead_memory.json` compares baseline/current public calls in one
process using twelve balanced four-arm rounds (old/new plus same-version
controls). Inputs are preallocated; private RNG setup is outside timing.
Allocation tracing is a separate diagnostic, never mixed into selection
timings. The reported peaks are incremental allocations tracked by
`tracemalloc`, not process RSS. Analytical bounds include temporary overlap,
reduction workspace and allocation metadata. The retained script requires a
prechange source checkout at `baseline_source` under the local run directory;
`baseline_manifest.json` identifies its source. Large timing variation and
controls prevent treating unpaired medians as speedup claims.

The five-case default and nondefault replay summaries retain exact comparisons,
initial-design checks, finite finals and historical population envelopes.
The normal_D5 seed-0 gsKL envelope failure already occurs in the prechange
baseline; the nondefault replay changes that gsKL by only 2.22e-16.
Raw trajectories and full source snapshots remain ignored local artifacts;
these compact reports preserve the conclusions and source provenance.

Preliminary 36.80/46.78-second campaigns are excluded because review found
imbalanced discovery controls and held-out pair directions. Their retained
JSON is explicitly prefixed `preliminary_`. The first probe with a coarse
Windows monotonic timing clock failed before persistence; kernel durations
now use `perf_counter`, independently of the watchdog clock.
