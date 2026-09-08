# Paired boost timing pilot

Update 2026-09-08: the PI has now authorized the full paired campaign.
Current execution is recorded in [the campaign note](2026-09-08-boost-campaign.md).
The parked instructions below preserve the original restart contract.

## Parked experiment restart

**PI direction, 2026-09-08: defer the full boost experiment; proceed with
main-loop fixes.** Checkpoint `764a177` on branch `dev-final-boost` preserves
the experimental code and harness. No batch or watcher is scheduled.

To resume later:

1. Use an isolated checkout of this parked checkpoint, not the evolving
   main-loop branch. Use the original `.venv` dependencies (Python 3.12.6,
   NumPy 2.5.2, SciPy 1.18.1, gpyreg
   `a2f8ddce867f502e29717959cf0ff3529f598618`), one BLAS thread, and explicitly
   verify actual PyVBMC/gpyreg imports resolve to the intended sources.
2. Read this note and `2026-09-08-boost-reconstruction.md`. The immutable
   inputs are `dev/scripts/runs/golden/reference_870_20260907/`; authentic
   snapshots are under `dev/scripts/runs/latent_fixes/boost_20260907/`
   in `revised_seven/captures/` and `noisy_counterexamples/captures/`.
   Raw traces/captures are gitignored local artifacts, not in the checkpoint.
3. Extend `dev/scripts/boost_penalty_pilot.py` from its three fixed cases
   to the 870-case manifest. This is remaining work: the pilot command below
   does not launch the population. Reuse existing pilot pairs only if the
   final protocol matches. Use a new output directory and retain every pair.
4. Redo BOTH boosts from one shared state and identical independent RNGs,
   with `weight_penalty=0.1` versus `0`, `tol_weight=0` and
   `tol_elcbo_boost=None`. Keep full pre/candidate VPs, transformers, GP,
   actual options, RNG and ELBO/SD. Use authentic snapshots where available.
   Retain historical pre-scores separately; consistent diagnostic rescoring
   on a reconstructed GP must use a separate RNG and must not advance the
   optimization stream.
5. Schedule the single-worker campaign when the PI returns to it (about
   3 hours, 4-hour planning allowance). Compare the newly generated paired
   arms statistically; historical golden outcomes are context. Select the
   boost penalty and acceptance rule/default from the evidence. Acceptance
   thresholds can be applied offline without rerunning candidate generation.

Pilot results remain in
`dev/scripts/runs/latent_fixes/boost_20260907/penalty_pilot_20260908/`.
No production penalty/default was selected by parking this work.

The PI authorized a timing pilot for the proposed 870-state comparison of
boosts with and without the small-weight penalty. The three fixed seed-0
cases were noisy Student D8, Lumpy D10 and budget-exhausting Cigar D15.
Each state was reconstructed once; both arms used independent copies of
identical VP/GP inputs and identical fresh RNG states. No main optimization
loops or target evaluations ran.

## Measured cost

| Configuration, seed 0 | Penalty 0.1 boost | Penalty 0 boost | Diagnostics for pre + both candidates | Whole pair |
|---|---:|---:|---:|---:|
| Student D8, noise SD 3 | 2.689 s | 2.661 s | 1.921 s | 7.405 s |
| Lumpy D10 | 3.264 s | 3.268 s | 2.372 s | 8.960 s |
| Cigar D15 exhaust | 6.116 s | 6.101 s | 3.372 s | 15.648 s |

Six optimizer calls took **24.100 seconds**. The full pilot took **32.018
seconds**, including 7.665 seconds of diagnostics, 0.128 seconds of
reconstruction and 0.098 seconds of dill serialization. Optimizer time is
nested inside final-boost time and is not counted twice. Per-case wall time
ends before writing the small JSON report; the overall summary includes
those report writes. Its own final JSON write and interpreter startup are
outside the measured total.

Straight-line extrapolation is `32.018 / 3 * 870 / 3600 = 2.579 hours` for
870 paired cases, or 1,740 boosts. A practical estimate is **about 3 hours,
with a 4-hour planning allowance**, using one sequential process and
single-threaded BLAS on this machine. These three seed-0 observations are
timing coverage, not a confidence interval or upper bound. Cigar D15 is
overrepresented in this pilot relative to its 10/870 population share;
other configurations and seeds can nevertheless take longer than these
observations. Any additional diagnostic rescoring in the eventual campaign
adds work not measured here.

## Pairing and retained evidence

Both arms set `tol_elcbo_boost=None`, disabling the acceptance guard and
preventing the experimental guard path from overriding the chosen penalty.
The actual options at entry to `optimize_vp` were recorded and checked:
`weight_penalty` was 0.1 or 0 respectively, and `tol_weight` was zero in
both. Each arm made exactly one optimizer call and returned a finite
50-component candidate. The two arms had identical numeric input hashes
and RNG entry states, with distinct generators to avoid shared-stream
advancement through VP deepcopy.

All three complete pre-boost states and both raw candidate VPs are retained,
including GP factors, options, scores, transformers and RNG states.
Returned VPs are also saved: the raw optimizer candidate has its temporary
`stable=False`, while final boost restores the pre-boost stability flag.
The captures total 23.6 MB across the three cases. They can be reused when
the full paired campaign uses the same protocol.

Pre-boost scores are explicitly marked as historical trace scores, not
rescored under the reconstructed GP. This is a timing and candidate-generation
pilot, with no acceptance-rule analysis or statistical penalty-effect claim.
Both treatments use the same reconstructed state; exact historical boost
reproduction is not required for this paired comparison. See the
[reconstruction check](2026-09-08-boost-reconstruction.md) for the distinction.

Artifacts:
`dev/scripts/runs/latent_fixes/boost_20260907/penalty_pilot_20260908/`
contains three JSON reports, three dill captures, `summary.json`, and
`capture_validation.json`. Actual module paths, versions, source hashes and
thread settings are in the summary. PyVBMC imported from this experimental
checkout; sibling gpyreg was separately verified clean at
`a2f8ddce867f502e29717959cf0ff3529f598618`.

```powershell
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:MPLBACKEND = 'Agg'
.venv/Scripts/python.exe dev/scripts/boost_penalty_pilot.py --out dev/scripts/runs/latent_fixes/boost_20260907/penalty_pilot_repeat
```

The initial attempt stopped before any optimizer call because the pilot's
hash check encoded large Python RNG integers as object-array pointer bytes.
That check was corrected to deterministic scalar encoding before the six
measured calls. No candidate was discarded or rerun to obtain these timings.

Independent Sol review checked pairing, actual options, source provenance,
artifacts and timing accounting, with no remaining findings. A separate
load of all three dill captures verified the retained pre-GP, both entry
RNG states, one optimizer call per arm, finite scores and normalized weights.
Python compilation and repository whitespace checks passed. This pilot
does not launch the full population campaign or select production defaults.
