# GP box-sampler comparison

Paired runs of `rosenbrock_D2_noise3_svbmc` at seeds 1000–1002 with the
original normal proposal and the corrected uniform proposal. The
[report](../../results/2026-09-16-gp-box-sampler.md) describes the source
comparison, experiment and limitations.

- `manifest.json`: the baseline and MATLAB commits, dependency versions,
  source archive hash, configuration, seed selection and thread setting.
- `normal/` and `uniform/`: each run's complete diagnostic JSON, including
  per-iteration GP condition numbers, source hashes and final metrics.
- `summary.json`: the six runs' scalar results, without filtering.
- `verification.json`: exact agreement of each pair's initial design and
  the list of oracle arrays changed by the sampler correction, package
  test counts and golden replay results.
- `replay/`: the standard five-configuration golden replay's report and
  JSON sidecars. The full NPZ traces remain with the raw artifacts.

Raw target-call arrays, VBMC saves, logs and the frozen source are listed
in `dev/scripts/runs/LOCAL.md`. The probe at execution is copied there
and identified by the `sources.probe` hash in each diagnostic; the tracked
runner subsequently moved its diagnostic-helper import after the source
check so import sorting preserves the selected baseline package. The
numerical procedure is unchanged.

To reproduce, archive `pyvbmc/` from commit `69a7a184` into a temporary
source directory, and use a gpyreg checkout at tag `v1.2.1`. Run each seed
in a separate process with both source roots, using distinct output
directories:

```console
python dev/scripts/gp_box_probe.py --source-root BASELINE_ROOT --gpyreg-source GPYREG_ROOT --seed 1000 --out NORMAL_OUT
python dev/scripts/gp_box_probe.py --source-root . --gpyreg-source GPYREG_ROOT --seed 1000 --out UNIFORM_OUT
```

Repeat for seeds 1001 and 1002, with one process at a time. The probe sets
the BLAS thread variables before importing NumPy and verifies both
package import paths. Run the uniform arm from the box-sampler fix's
source; later numerical changes produce a different comparison.
