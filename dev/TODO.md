Pickup (2026-09-07): resume dev/plans/latent-bug-fixes.md with $task.
Latest PI update after CI 135: reduce the noisy reference addition because
the estimated day-long 150-run batch is infeasible. The reduced proposal is
30 seeds each of noisy Rosenbrock D2 and Student D8 (60 additions, 870 total).
The PI authorized one timed seed-0 run of each on pinned reference code,
then explicitly authorized launching the remaining 58 runs. The prepared
150-run manifest/launch command below are superseded for future execution.
The two timed seed-0 runs are now complete and validated: Rosenbrock 201.96 s
(190/200 evaluations), Student 344.74 s (255/500), both converged. Including
worker startup, these observations extrapolate to 4.600 h for all 60, or
4.447 h for the remaining 58; one seed per target does not measure seed
variability. Reusable traces, runtime records and validation are under
`dev/scripts/runs/noisy_reference_20260907/timing_20260907/`. All 1,620
original files remain byte-identical; published reference count is still 810.
RUNNING since 17:02:17 UTC+03, 2026-09-07: the reduced frozen launcher
reported 2 complete/58 pending and began Rosenbrock seed 1 with verified
imports. Canonical manifest/logs/output are under
`dev/scripts/runs/noisy_reference_20260907/reduced_60/`; launcher PID 6380
(venv shim PID 3712), stdout `run.stdout.log`, stderr `run.stderr.log`.
The development branch remains `dev-latent-neutral-fixes`; workers use source
`623f5cd` and gpyreg `a2f8ddc`. Seven focused tests and independent review passed.
Only one heavy computation: no tests/replays or CI-triggering push during
the batch. Push the pending launcher/test changes after the worker finishes.
Next: inspect completion, run the frozen launcher's `validate` with its
manifest, then combined-reference checks before publishing 870. Exact
resume/validation instructions are in the plan's reduced-batch launch record.
Astra orchestrates scientific decisions; Sol implements and reviews bounded
groups. The agreed fixes and experiment designs are approved; Q1's final
main-loop penalty treatment and Q4's final default remain evidence-dependent.
The approved 150-run reference addition is prepared, but NOT RUN
(explicit PI instruction, 2026-09-07). Phase 0's preparation subset is complete
on `dev-noisy-reference-prep`: pinned source `623f5cd` over `7314a6a`, isolated
gpyreg, imports verified, 7 focused tests passed, independent Sol review clear.
Canonical manifest: `dev/scripts/runs/noisy_reference_20260907/preparation.json`.
The plan has exact future launch/validation commands. All 810 original pairs
remain byte-identical; the extension directory is empty. No launch or scheduling
was performed. Phase 0's regression gate and Phase 1's approved neutral fixes
are complete on `dev-latent-neutral-fixes` (PI resumed with $task). Commits:
`ae5d871` replay gate, `c8a284e` neutral state/dtype, `3e67d9e` utility/API.
Exact oracles passed all eight fixtures before and after; both five-case
replays match every stored loop/final output under the stronger comparator.
Focused tests, SciPy 1.15/current prior checks, docs/examples and independent
Sol review passed. Historical returned transformers remain uncertifiable.
CI follow-up is complete: `514bad7` preserves the live transformer after
variational optimization; `686a75f` also restores sharing after active-sampling
rollback. CI run 135 passed (1,043 passed, 49 skipped, one separate acquisition
test retry); both previously failing integration cases passed first attempt.
The affected 25 local tests passed with retries disabled, with eight exact
oracles and five unchanged replays. Details and retry limitations are in the
plan's CI follow-up. Current Phase 0/1 work is complete.
Next development group: Phase 2's approved boost comparison; final Q1/Q4
choices remain evidence-dependent. No moving production fix was applied.
Evidence: `dev/scripts/runs/latent_fixes/neutral_20260907/` and the active
execution checklist in the plan. The 150-run extension remains parked.
Reference preparation and developer notes accompany the implementation commits.
The PI authorized committing the remaining files and pushing this branch on
2026-09-07; this does not authorize launching the parked 150-run batch.
Parallel development on isolated checkouts is approved; verify actual imports
and run only one heavy computation at a time. No job/watcher needs reattaching.
The broader calibration proposal is in roadmap pickup 12, outside this plan.

Earlier reference extension complete and verified (2026-09-07, 08:55 UTC+03).
Results and documentation are committed and pushed in 2a09fcd.
The checkout is back on dev-next, preserving its newer release decisions.
Frozen reference/stage3-20260906 remains at 7314a6a; it was used with the
original .venv for all 530 new runs, with explicit vectorized_target=False.

Result: 810 complete JSON/NPZ pairs (16 configurations at seeds 0-49,
cigar_D15_exhaust at seeds 0-9), no error files, original 280 pairs
hash-verified unchanged. Batch 1 added 420 runs and passed both null checks
(56 KS tests each). Batch 2 added 110 and passed the final even/odd null
check (68 tests). Fresh preflight and final replays: all five cases identical,
including initial designs. All 10 exhaust runs reached their intended budget;
all 800 other runs reported convergence. Total chain time including checks:
12 h 54 min, under light laptop use; timing is not a controlled speed test.

Durable records: dev/golden/extension_20260907/README.md, three statistical
comparison reports, preflight/final replay reports, validation.json and
sha256_manifest.json. Baseline: dev/golden/baseline (810 sidecars + summary).
Historical sidecars retain 18a236c, dirty=false (numerical code bdaf322).
New sidecars retain 7314a6a: 420 dirty=false; 110 dirty=true because batch-1
publication changed baseline documentation. Numerical source stayed frozen.

Next: resume dev/plans/latent-bug-fixes.md, the partially approved plan for
roadmap pickup 9. D1–D5 are approved (2026-09-07): neutral fixes first, then
boost after settling its rule, then other trajectory-moving fixes. Q2 is
approved: reject original-space PDF gradients (current S-VBMC does not use
them), with possible future autodiff support. Q3 is approved: preserve fixed
widths and copy for new components. Q1's final penalty choice and Q4's final
boost tolerance remain open; the comparison designs below are agreed.
Prepare and test those designs before choosing production defaults.
The PI approved three additional authentic-history GP-fit fixtures
(early, later changing-Ns, noisy) alongside the legacy and controlled-history
tests; capture/replay requirements are in phase 3. Independent plan review is
recorded in the plan itself.
Q1 now explicitly questions the purpose of the eta bounds, not only MATLAB
parity. Current max-shift bounds penalize weight ratios below 1/200 at the
defaults; a bounded probe also confirmed the returned gradient is wrong when
that lower bound is active. The plan records the evidence and leaves the
main-loop penalty design open. The PI selected a three-way comparison: no
eta-bound penalty, MATLAB's correctly implemented historical formula (also a
diagnostic, not the presumed solution), and current behavior as diagnostic
reference only. Hold the separate shrinkage term fixed; if the
current inconsistent implementation wins, investigate why, never retain it
as the solution. Phase 6 specifies source checks and paired comparisons.
The PI selected disabling the remaining
small-weight shrinkage penalty during final boost, paired with the planned
acceptance gate; record this as a boost-only override (`weight_penalty=0`).
Q4 is now under discussion: require the score difference to exceed minus the
tolerance for all b in [0,5] (checking the endpoints suffices). The PI selected
tol_elcbo_boost=0.1 and 0.2 for comparison, with neither selected as the
default; broader noisy-target coverage is needed before
settling the default. The existing 100 noisy runs cover only two configurations
and none would be rejected at either threshold; 15 of the full 810 would be
rejected at 0.1 and seven at 0.2.
The PI approved extending the reference
itself with rosenbrock_D2_noise3, student_D8_noise3 and lumpy_D10_noise3,
each at seeds 0–49 (150 new runs, directly superseding the 10/20-seed pilot).
All three join the standard benchmark set: 960 reference runs across 20
configurations, including 250 noisy runs across five configurations, once
validated. Preserve the existing 810 pairs unchanged and append the new cases
on the same reference numerical implementation, excluding latent/boost fixes.
The PI approved parallel development: use a pinned reference checkout and
separate development branches/checkouts, verifying actual editable-install
import paths. Neutral fixes and boost/test tooling preparation can proceed
while the reference runs; only one heavy compute process runs at a time.
The numbered phases describe dependencies and validation order, not a serial
development schedule. Scientific choices still await their evidence.
The plan's Phase 0 checklist records preparation, pinned checkout locations,
verification evidence, and the separate future launch/validation procedure.
No new campaign has been launched or scheduled.
Verify claims against current source and the existing derivation/bug notes,
separate trajectory-neutral from trajectory-moving changes, define the gates,
and surface unresolved bug-versus-algorithm decisions for the PI. The existing
final-boost decision is in dev/2026-09-04-final-boost-failure.md and pickup 9.
Complete S-VBMC integration and connections to upcoming extensions (pickup 10),
resolve the PyTorch feasibility decision below, and validate the final 1.5 code
as a population against this extended reference before dev-next -> main.
No latent fix or solver implementation was made during the reference work.

The PI's new broader proposal is recorded: machine-local calibration of
implementation tuning knobs, cached
after a short first-use sweep and explicitly rerunnable. Roadmap pickup 12
records concrete PDF/entropy chunk-size candidates, timing evidence,
reproducibility considerations and open cache/API/budget choices. This is
outside the latent-fix plan; no calibration implementation has started.

The first Stage 4 step is a PyTorch feasibility prototype tested on both CPU
and GPU in float64 against the modernized NumPy CPU baseline, including
transfer costs, followed by an explicit decision on the full port
(PI decision, 2026-09-06). Inclusion in 1.5 is preferred if feasible, subject
to numerical reliability, acceptable CPU performance, manageable installation
friction, and simpler future method development. Immediate speedup is not
required. PI clarification (2026-09-07): 3× runtime was an example of clearly
unacceptable performance; concern starts at substantially smaller slowdowns.
Around 1.2× runtime may be acceptable given the other benefits, but is not
an agreed cutoff. The full port is not yet a
commitment. Settle the NumPy transition, core dependencies, and Python floor
in the design. Algorithmic extensions remain separate from the port. See
the roadmap Stage 4 entry and the 1.5 overview.

No campaign process, scheduler, or session-bound watcher remains. The launcher
finished both batches, published the local baseline, passed final replay,
returned to dev-next, and released its temporary keep-awake request.
Local operational state/logs: dev/scripts/runs/reference_extension_20260906/,
dev/scripts/runs/golden_{grow,extend}_20260906.log and
.venv/reference_extension_20260906.*.log. The permanent reports above contain
the outcomes; no old job needs reattachment. The new 150-run addition above
is prepared; launching remains a separate next action.

The full NPZ traces remain gitignored on this laptop under
C:/Users/luigi/Documents/GitHub/pyvbmc/dev/scripts/runs/golden/item7_20260906/.
Copy that directory for exact replay elsewhere; publish the traces as a 1.5
release asset later. The original .venv is the reference environment:
Python 3.12.6, NumPy 2.5.2, SciPy 1.18.1, gpyreg 1.1.0, cma 4.4.4.
The separate pyvbmc-stage3 worktree has its own optional-dependency environment;
its runs directory is a junction to these artifacts. Do not overwrite work in
that checkout. Stage 3 integration and CI evidence remain in
plans/stage3-pipeline-features.md.

Read first: dev/plans/latent-bug-fixes.md, this reminder,
dev/2026-09-06-pyvbmc-1.5-overview.md, dev/golden/README.md, and roadmap
pickups 3f, 9, 10, 11, 12.
For bug work also read dev/2026-09-02-modernization-discussion.md section 9.
Use dev/golden/extension_20260907/README.md for execution/provenance details.
Keep dev/golden/README.md a human-facing description of the current runs
(purpose, coverage, metrics, files and usage); batch chronology and release
stage history belong in the execution record, not that README (PI request).
