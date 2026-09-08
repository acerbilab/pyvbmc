# Paired final-boost campaign

Status: computation complete, 2026-09-08 at 10:11:53 UTC;
[paired analysis](2026-09-08-boost-analysis.md) now available.
This resumes the [parked experiment](2026-09-08-boost-penalty-pilot.md),
after the bounded eta-bound comparison completed. The
[live plan](plans/latent-bug-fixes.md) owns execution tracking.

## Completion

All **870 paired endpoints / 1,740 boost arms** completed with zero errors.
The worker generated 868 pairs and reused the two smoke pairs. Its full
invocation took **5,577.435 s (1h32m57s)**, including the final capture-hash
verification; the earlier smoke invocation took 12.479 s. Completion was
10:11:53 UTC (13:11:53 local). Both worker and environment launcher exited;
no lock, pending generated-arm checkpoint or pair error file remains.

The status audit checked all 870 reports against the immutable manifest:
861 reconstructed and nine authentic inputs, both actual penalty settings,
disabled guard/pruning, finite returned scores and finite common-GP
diagnostic ELBO/SD/ELCBO values. The runner verified all capture hashes at
completion; this lightweight status audit did not independently rehash the
large dill files. Evidence: `summary.json` and `completion_status_check.json`.

Scientific paired-effect analysis and acceptance-rule comparisons are in
the linked analysis report. They do not select a penalty or threshold.
The following launch record is retained for provenance and possible resume.

## Protocol and scope

Run both final boosts for every endpoint in the immutable 870-run reference.
The paired settings are `weight_penalty=0.1` and `0`, both with `tol_weight=0`
and `tol_elcbo_boost=None`. No main-loop runs, target evaluations, reference
updates or production default selection are part of this campaign.

Each pair uses identical VP/GP/options/state inputs and independent generators
cloned from the same fresh RNG state. Use the nine authentic captures with
their original GP factors; reconstruct the other 861 from the stored traces.
This is a paired comparison conditional on the shared endpoint; reconstructed
states need not reproduce historical numerics exactly. See the
[reconstruction evidence](2026-09-08-boost-reconstruction.md).

Retain the full pre-state, raw candidates, returned VPs, actual optimizer
options and RNG states. Keep historical pre-boost scores separate from
consistent rescoring of pre and both candidates under the same GP, using a
separate diagnostic RNG. Save ELBO, GP uncertainty and beta-5 ELCBO, plus
available ground-truth posterior/evidence metrics. Entropy Monte Carlo error
is not included in the GP SD. Different component counts limit common-draw
alignment. Acceptance thresholds remain an offline analysis and PI decision.

The original three timing-pilot pairs are retained as historical evidence;
the campaign generates fresh pairs with the common-GP rescoring protocol.
Completed pairs are written independently and verified before reuse on
resume. One detached worker uses single-threaded BLAS.

## Code, inputs and evidence

- Isolated checkout: `dev/scripts/runs/latent_fixes/boost_campaign_worktree/`,
  branch `dev-boost-campaign`, runner commit `8c7919f`, numerical base `764a177`.
- Python: original `.venv/Scripts/python.exe`; actual imported PyVBMC source
  must resolve inside the isolated checkout, regardless of editable metadata.
- gpyreg: clean sibling at `a2f8ddce867f502e29717959cf0ff3529f598618`.
  Python 3.12.6, NumPy 2.5.2, SciPy 1.18.1; OMP/OpenBLAS/MKL threads all one.
- Reference: `dev/scripts/runs/golden/reference_870_20260907/`.
- Authentic captures: `dev/scripts/runs/latent_fixes/boost_20260907/`,
  under `revised_seven/captures/` and `noisy_counterexamples/captures/`.
- Evidence root: `dev/scripts/runs/latent_fixes/boost_campaign_20260908/`.
  `before.json` records actual imports and versions before launch.

All 870 JSON/NPZ pairs and nine authentic captures were found. Actual imports
resolve to the isolated PyVBMC checkout and pinned sibling gpyreg. No Python
worker was running before preparation; approximately 43 GB disk space was
available. Raw artifacts remain local and gitignored.

The timing pilot extrapolated to 2h35m for 870 pairs; practical expectation
was about three hours with a four-hour allowance. Campaign common-GP rescoring
adds work beyond that pilot. Update the estimate from campaign throughput;
the user's 3h30 absence is a planning window, not a forced cutoff.

Pre-launch checks re-audited all 870 traces in 1.54 s: all are structurally
eligible, including 13 earlier-best-iteration endpoints. Selected component
counts range from 5 to 38, so every endpoint requires a boost to 50.
Timing the additional 100,000-draw unpenalized rescoring on the three saved
pilot pairs took 0.638 s (Cigar D15), 0.452 s (Lumpy D10), and 0.370 s
(noisy Student D8) per pre/two-candidate triple: approximately seven extra
minutes when extrapolated to 870 pairs. No new boost fits ran in these checks.
Evidence: `reference_audit.json` and `scoring_timing.json` in the evidence root.

## Launch and verification

Independent Sol launch review passes. Whole-tree comparison confirms all
`pyvbmc/` files equal the pinned numerical base; tracked PyVBMC/gpyreg trees
are clean. The runner checks this at startup and verifies gpyreg's exact HEAD.
Its configuration freezes all 870 input paths/hashes and the measured runner
source; committing the runner does not invalidate the configuration.

Zero-fit preflight passed. The first two campaign pairs were authentic
`banana_D2_seed0` and reconstructed `student_D8_noise3_seed0`: four boosts,
two successful completions, zero errors, 12.479 s total invocation work.
Capture hashes, paired input/RNG equality, actual options, finite normalized
50-component candidates, retained restart fields and all six diagnostic
scores passed inspection. Repeating the command skipped both with no new fits.
A separate recovery check injected a diagnostic failure, verified generated
candidates remained saved, then recovered their scores with the optimizer
disabled by a test stub. Three invalid-score checks (NaN ELBO, negative or
infinite GP variance) were rejected. Diagnostic rescoring did not advance
candidate RNGs. Evidence: `smoke.log`, `smoke_validation.json`,
`smoke_validation.log` and `recovery_check/`. Repository hooks passed.

The detached launch began at **08:38:54 UTC**. The Python environment launcher
PID is **31008**, and the numerical worker PID recorded by the lock is
**14272**. One worker runs both arms sequentially, with all three BLAS thread
limits set to one. `launch.json` preserves the executable, full argument list,
working directory, timestamp, source commits and output log paths.

Initial health check at 08:41:11 UTC confirms the worker is alive and
progressing: 19 complete pair reports, including the two smoke pairs, zero
errors and no stderr output. `launch_health.json` retains that snapshot.

Live files in the evidence root:

- `worker.log` and `worker.stderr.log`: progress and diagnostic messages.
- `progress.json`: full worker invocation counts and elapsed time, updated
  after each attempted new pair. A prefix of skipped pairs does not update it.
- `<tag>.dill` plus `<tag>.json`: completed pair, with capture SHA in the JSON.
- `<tag>.generated.dill`: both generated arms saved before diagnostics; reused
  if scoring must be retried. Removed once the complete pair is saved.
- `<tag>.error.json`: any failed pair, preserved for follow-up; other pairs
  continue. Do not interpret an attempted pair as a successful completion.
- `summary.json`: end-of-invocation totals. While the full job runs, this may
  still describe the earlier smoke/resume invocation; use the live log and
  `progress.json` for current progress.

Per-pair `wall_s` is generation time only; it excludes diagnostic scoring and
serialization. Use worker `progress.json`/`summary.json` `wall_s` for the full
invocation duration. The two smoke pairs are included in the population and
will be skipped when the worker reaches their tags, not regenerated.

For resume, first confirm the recorded worker has exited. A hard kill can
leave `.campaign.lock`; remove that file only after confirming the process is
dead. Run the same command with the same source and environment. Complete
pairs are hash-verified and skipped; saved generated arms are rescored.
No watcher or automatic restart is scheduled. The output is local/gitignored,
and the runner branch is a local checkpoint until explicitly pushed.

From the main repository root, the foreground equivalent is:

```powershell
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:MPLBACKEND = 'Agg'
.venv/Scripts/python.exe -u dev/scripts/runs/latent_fixes/boost_campaign_worktree/dev/scripts/boost_penalty_campaign.py --reference dev/scripts/runs/golden/reference_870_20260907 --capture-root dev/scripts/runs/latent_fixes/boost_20260907 --out dev/scripts/runs/latent_fixes/boost_campaign_20260908
```

The subsequent full capture audit and paired analysis are complete; see the
linked report. Historical golden outcomes remain separate context. Production
penalty choice and the eta-bound follow-up remain future work; this campaign
makes none of those decisions.
