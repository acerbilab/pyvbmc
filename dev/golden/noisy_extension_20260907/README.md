# Noisy reference extension, 2026-09-07

The reduced extension adds **60 runs**: `rosenbrock_D2_noise3` and
`student_D8_noise3`, each at seeds 0–29. The combined reference has **870
JSON/NPZ pairs across 19 configurations**, including **160 noisy runs across
four configurations**. Its allocation is the original 16 × 50 plus the
D15 exhaust × 10, with two new configurations × 30. The original 810-run
[execution record](../extension_20260907/README.md) remains unchanged.

The PI reduced the original 150-run proposal after discussing runtime.
`lumpy_D10_noise3` remains registered in the benchmark suite but was not run
and is not part of this reference. No numerical, boost, or Q1/Q4 treatment
was changed to generate these additions.

## Results and limitations

All 60 new runs produced complete traces and finite final comparison metrics.
The frozen launcher validated each result and the completed set. All 1,620
original files were hash-verified byte-identical. Combined traces were copied
into a new directory, preserving both input populations; all 870 NPZ archives
passed their ZIP integrity checks.

| New configuration | Runs | Converged | Reached budget | Usable | Mean optimizer time |
|---|---:|---:|---:|---:|---:|
| Rosenbrock D2, noise SD 3 | 30 | 23 | 7 | 23/30 | 2.262 min |
| Student D8, noise SD 3 | 30 | 30 | 0 | 14/30 | 6.180 min |

The evaluation budgets are 200 and 500. “Usable” retains the existing summary
criterion: evidence error < 1, gsKL < 1 and MMTV < 0.2. It is distinct from
convergence: six of the seven budget-limited Rosenbrock runs meet that
accuracy criterion, whereas some converged runs do not. Student's median
absolute evidence error is 1.08. The reference retains every seed and outcome;
no difficult run was discarded or rerun. Runtime warnings about logarithms of
zero occurred in the noisy acquisition, but no run produced an error file.

The combined reference has 853 converged runs and 17 budget terminations:
the original ten deliberate D15 exhaust runs plus the seven new Rosenbrock
runs. These outcomes record the frozen algorithm's behavior, rather than
establishing a final boost tolerance or a claim of uniform inference accuracy.
The [baseline summary](../baseline/summary.md) contains the full distributions.

## Source and timing provenance

All additions used reference commit
`623f5cd91925c6e6b6a23d2985a66bcb1d6bea86`, one additive benchmark-registration
commit above `7314a6afe158c5673775ca79601b88b3913ae383`. Numerical source stayed
frozen. Actual worker imports were checked against this isolated checkout and
gpyreg `a2f8ddce867f502e29717959cf0ff3529f598618`; editable installation metadata
alone was not used to establish source identity. The original environment was
Python 3.12.6, NumPy 2.5.2, SciPy 1.18.1, gpyreg 1.1.0 and cma 4.4.4.
Workers used `vectorized_target=False`, one process and one BLAS thread.

The two seed-0 runs were timed first, starting at 16:29:51 and 16:33:15 local
UTC+03. Their original JSON/NPZ/runtime records were copied byte-identically
into the batch and reused. The remaining 58 runs executed from 17:02:17 to
21:08:49: **4 h 6 min 32 s**. Summed optimizer time across all 60 was 4.221 h.
These laptop timings are observations, not a controlled performance comparison.

| Cohort | Pairs | Recorded code | Dirty flag |
|---|---:|---|---|
| Historical original | 280 | `18a236c` | false |
| Earlier first extension | 420 | `7314a6a` | false |
| Earlier second extension | 110 | `7314a6a` | true; historical documentation changes |
| New noisy extension | 60 | `623f5cd` | false |

All original sidecar provenance is retained. The new
[SHA256 manifest](sha256_manifest.json) covers the 870 pairs and summary,
with JSON/Markdown line endings normalized to LF and NPZ files hashed as raw
bytes. [Validation metadata](validation.json) records allocation, provenance,
checks and runtime-record hashes for the 60 additions.

## Completion gates and artifacts

The [even/odd comparison](even_vs_odd.md) passed: no configuration flagged
across 76 KS tests, Holm alpha 0.05. The new configurations have 15 seeds per
half. This null check does not establish equivalence or eliminate low-power
limitations of the smaller cohorts.

The [final five-case replay](final_replay.md) passed in 4.74 minutes: all
stored non-timer loop/final arrays, all 11 semantic final fields and initial
designs matched exactly, with zero flags. Numerical imports resolved to the
clean pinned reference `623f5cd`; the stronger development comparator was
used without changing numerical source. The five cases are the existing
default Gaussian, banana, half-normal, cigar and noise-SD-1 Rosenbrock seed-0
cases. This does not claim a fresh replay of all 870 runs or of the two new
configurations. Historical traces lack the returned posterior's transformer,
so that state remains explicitly uncertifiable. The machine-readable
[replay report](final_replay.json) retains this limitation for every case.

The focused harness tests (`test_golden_replay.py` and
`test_reference_noisy_extension.py`) passed: 74 tests, retries disabled,
11.69 seconds. The one-line default replay-path change passed pinned Black
23.3.0 formatting verification.
Independent Sol review of the publication and provenance passed with no
remaining findings; the review was static and did not duplicate numerical runs.

The canonical local traces are under
`dev/scripts/runs/golden/reference_870_20260907/`. The previous
`dev/scripts/runs/golden/item7_20260906/` remains unchanged. Traces stay
gitignored; tracked sidecars and the summary are in `dev/golden/baseline/`.
No release asset is published by this extension task.

Local preparation, worker import records, logs and seed-0 adoption evidence
are under `dev/scripts/runs/noisy_reference_20260907/reduced_60/`.
Its `integration/` directory contains the combined-copy check, summary/null
logs, final replay driver and runtime provenance. The frozen launcher's
`validate --manifest <reduced_60/preparation.json>` checks the 60-run input
set. Summary and null checks used the pinned `golden_trace.py` with
`summary <combined>` and `compare --split <combined>`. Final replay uses the
current stronger comparator with verified pinned numerical imports, explicit
`vectorized_target=False`, and the combined directory as both baseline and
sidecar population. The task plan records the exact local paths and commands.
