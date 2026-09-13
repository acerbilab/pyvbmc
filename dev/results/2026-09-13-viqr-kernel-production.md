# VIQR prediction-kernel reuse: production validation

The production implementation reuses gpyreg's training-to-candidate kernel
matrices within a standard VIQR call. The captured late noisy sieve is
1.205x faster, with exact acquisition outputs on this platform. The retained
matrix payload is capped at 128 MiB; larger calls use ordinary evaluation.
The 18-run replay is exact; release validation is tracked in the
[execution plan](../plans/noisy-acquisition-efficiency.md#kernel-reuse-implementation-plan).

## Sources and method

The before pair is PyVBMC `15a14cc` with gpyreg `a2f8ddc` (v1.1.0).
The candidate pair is PyVBMC `9e8d2f2` with gpyreg `355d754`.
The [production runner](../scripts/validate_viqr_kernel_reuse.py) loads these
actual packages in separate warmed workers, with one BLAS thread and both
workers pinned to the same logical CPU. Only one worker computes at a time.
Timing excludes imports, restoration, IPC and instrumentation; it includes
the complete public acquisition call or ordinary GP prediction.

Coverage comprises the seven noisy captures and supplementary Gaussian
oracle from the [feasibility probe](2026-09-13-viqr-kernel-reuse.md), each at
one candidate, a CMA-ES population of 6-10 candidates, and 8192 candidates.
The four methods are ordinary VIQR, VIQR forced onto its memory fallback,
ordinary GP prediction, and the non-VIQR `AcqFcnLog`: 96 comparisons.
Each uses nine alternating rounds, with 31 pairs per round for small calls
and one pair per round for sieve calls. Speedups below are medians of paired
before/after time ratios. No targets are evaluated or integration points redrawn.

The [raw production artifact](../experiments/noisy-acquisition-efficiency/viqr_kernel_reuse_production.json)
records every repeat, capture hashes, normalized source-text hashes, imported
paths, revisions, environment, CPU affinity, kernel requests, numerical
comparisons, and peak traced allocations. Capture archives retain raw byte
hashes. Preflight runs under the ignored run directory are excluded from the
performance evidence.

## Captured-state results

| State | N / GP samples | Full sieve speedup |
|---|---:|---:|
| Logistic regression D5, noise 3, early | 10 / 8 | 1.045x |
| Multisensory D6, noise 1.3, early | 10 / 8 | 1.026x |
| Rosenbrock D2, noise 1, early | 10 / 8 | 1.006x |
| Rosenbrock D2, noise 3, early | 10 / 8 | 1.014x |
| Rosenbrock D2, noise 3, late | 150 / 6 | 1.205x |
| Student D8, noise 3, early | 10 / 8 | 1.036x |
| Timing D5, noise 2.2, early | 10 / 8 | 1.013x |
| Supplementary Gaussian D2 oracle | 65 / 1 | 1.124x |

The late-sieve gain is present in all nine pairs (1.154-1.304x), exceeding
the 1.10x acceptance target. Small noisy VIQR calls improve by about 5-9%.
Across all captured sizes, fallback VIQR ratios are 0.975-1.015x, ordinary
GP prediction 0.968-1.153x, and `AcqFcnLog` 0.981-1.062x. None crosses the
5% slowdown threshold. Unchanged GP arithmetic makes apparent GP gains a
measure of timing variability; these timings do not estimate whole-run gain.

Every compared output is byte-exact, every stored full-sieve baseline is
reproduced, and state/RNG digests remain unchanged. Request tracing confirms
that only the standard VIQR path requests retained kernels. All captures are
below the real cap; focused tests lower the cap to exercise its exact boundary
and above-cap fallback without allocating a large fixture.

The late call's peak traced allocation rises from 64.44 to 102.01 MiB;
retained kernels occupy 56.25 MiB. Early captures retain 5 MiB and increase
the peak by about 4.38 MiB. The supplementary oracle retains 4.06 MiB while
its measured peak remains 48.50 MiB. These are temporary traced allocations,
not total process memory. Only one late noisy state is represented.

## Correctness and integration gates

Local checks completed before replay:

- Gpyreg full suite: 198 passed; its focused API module passed 50 checks
  after two missing-factor checks were added following independent review.
- PyVBMC acquisitions and importance sampling: 123 passed.
- Full PyVBMC suite: 1277 passed, 39 skipped, 2 successful reruns in 334 s,
  with BLAS single-threaded and workspace-owned pytest temporary/cache paths.
- Normal oracle suite: 143 passed, 15 platform/coverage skips.
- Exact oracle generator comparison against frozen before outputs:
  all 11 fixture groups passed, without changing any references.
- Both repositories' formatting hooks pass. Gpyreg's Sphinx build succeeds
  with two pre-existing underline warnings in unrelated method docstrings.
- Independent static reviews of the API, PyVBMC integration and comparison
  harness found no implementation defect. The missing-factor test gap was fixed.

Both the [gpyreg candidate matrix](https://github.com/acerbilab/gpyreg/actions/runs/34759637059)
and [PyVBMC candidate matrix](https://github.com/acerbilab/pyvbmc/actions/runs/34759664051)
passed all nine OS/Python cells. Gpyreg passed all 200 tests in every cell.

## Bounded replay

The six noisy configurations in the plan, seeds 0-2, completed in 45.9
minutes with 18 successful finite results and zero flags. Every stored
non-timer array and semantic final field equals the immediate guarded-sinh
baseline exactly; all initial designs match. These historical traces omit
the returned posterior's transformer, so equality of that omitted state is
not certifiable. The promoted golden population remains the accuracy reference.

Replay used PyVBMC `16dcc3f`, a documentation successor of the measured
`9e8d2f2`, with gpyreg `355d754`. The wrapper required the measured production
source hashes before launch; package, runner, target/data and baseline hashes
match before and after replay. Documentation tracker edits during the campaign
can make individual sidecars report a dirty checkout. The
[validation artifact](../experiments/noisy-acquisition-efficiency/viqr_kernel_reuse_validation.json)
contains all replay comparisons, the source/input manifest, output hashes,
candidate CI records and local-check outcomes. Raw traces remain under
`dev/scripts/runs/viqr_kernel_20260913/replay/`.

The release-backed dependency rollout remains required before integration.
