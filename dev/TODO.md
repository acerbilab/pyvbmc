Pickup (2026-09-08): resume dev/plans/latent-bug-fixes.md with $task.

Remaining-work map (2026-09-08; roadmap is authoritative):
- Full supported matrix: green, all nine jobs in 34263042113 at `24cf369`.
  The acquisition oracle now accounts for variance-penalty conditioning;
  a separate Python 3.10 test mock target is explicit. Both fixes preserve
  production code and reference values. See
  dev/2026-09-08-acquisition-oracle-conditioning.md and the live latent-fix plan.
- Release work: S-VBMC compatibility/integration and connections to upcoming
  extensions; PyTorch CPU/GPU float64 feasibility, then the explicit port
  decision and resulting dependency/Python-floor/NumPy-transition work.
- User-facing agent skill: skills/pyvbmc/SKILL.md with FAQ/reference material,
  helpers and packaging/installation design, once the API settles.
- Proposed improvement, release placement open: machine-local calibration
  of kernel chunk sizes, caching and explicit recalibration (roadmap 12).
- Final validation/release: freeze the final candidate, run the 870-case
  population with retained boost candidates and compare accuracy/usability;
  complete supported-matrix gates, final docs/release records, one PR to main,
  and publish identifiable reference NPZ archives as release assets.
  Release notes must explain changed RNG streams and compact-history save
  compatibility. A lab-server backup of the local reference traces was
  suggested as optional insurance; it is not recorded as completed.
- Deferred research: variational stopping rules; per-component lambda;
  gradient acquisition optimization; parallel target acquisition; multi-chain
  GP sampling; larger-N inference; log-space mixture sums; diagonal log-joint
  variance/gradient (compute_var == 2); noise shaping. gpyreg #44 is separate
  deferred maintenance. These are not current release prerequisites.

CI follow-up: e2639a1 was pushed, but run 34240597505 collected historical
eta-experiment tests against changed production code. Package test discovery
is corrected, with the historical source guard intact. Full local suite:
1,049 passed, 35 skipped, no reruns; independent review passes. Repair
96a8e3a is pushed; CI 34246862361 passed (1,122 passed, 60 skipped, no reruns).
Merged and pushed to dev-next at 89ac5f0; integration CI 34248221965 passed
(1,122 passed, 60 skipped, no reruns, 541.28 s).
PI deferred the optional step-out repair to gpyreg issue #44:
https://github.com/acerbilab/gpyreg/issues/44
No gpyreg repair or pin bump is required for this release. Final
latent-fix population validation remains pending; the full supported matrix
passed all nine jobs in 34263042113 after the two test corrections.
The final 870-run population follows the remaining S-VBMC compatibility/
integration work and the agreed PyTorch feasibility decision and resulting
release work; do not launch it prematurely. The next broader workstream is
S-VBMC compatibility/integration.
Also retain the user-facing coding-agent skill in the remaining work:
dev/2026-09-02-user-agent-skill.md proposes skills/pyvbmc/SKILL.md, supporting
references/helpers and distribution with the library. Prepare the guidance
after the 1.5 user-facing API settles; FAQ/reference drafting can start
earlier. See the remaining-work map above.
PI-selected A is implemented on dev-eta-bound-fix: no eta-bound penalty or
caller-theta mutation. Other regularizers and optimizer settings remain.
64 focused tests pass; all 11 numerical fixtures exact. Six paired replays
converged and remained usable; retain one Normal D5 gsKL fence flag for
the final population assessment. Independent core/artifact reviews pass;
see dev/2026-09-08-eta-bound-fix.md.
Stopping-rule investigation is deferred research/improvement outside this
fix campaign, not a release blocker. The gpyreg step-out repair is likewise
deferred; Phase7 retains the remaining integrated validation work.

Completed: equal-iteration eta comparison on dev-eta-equal-budget;
independent scientific/artifact review passed. The subsequent PI selection
and production integration of A are recorded above.
Two saved noisy states x three optimizer RNGs x A/B, 400 Adam iterations,
checkpoints 40/100/200/400 before pruning, ten diagnostic scoring RNGs.
See dev/2026-09-08-eta-equal-budget.md. No production selection or offset fits.
All 12 fits / 480 scores complete in 25.261 s across successful invocations;
all historical trajectory prefixes exact. At equal effort B's sizable gains
disappear; remaining ELBO differences tiny, ELCBO mixed. A is supported for
PI consideration; production and stopping-policy choices remain separate.

Completed pickup on dev-final-boost-default: PI-selected final-boost defaults: zero boost-only
weight penalty and joint ELBO/ELCBO(beta=5) tolerance 0.1. Main-loop penalty
unchanged. Retest with the integrated final 870-run benchmark against the
expanded reference; no population launch in this pickup. This supersedes
the pending boost decision in historical records below.
Validation: 125 focused tests passed, 11 exact numerical fixtures passed,
independent Sol review and repository hooks passed. Final integrated suite
and population validation remain pending.
Boost-default commit 4ab2003 was pushed and CI passed:
https://github.com/acerbilab/pyvbmc/actions/runs/34229085644

Completed pickup: paired boost analysis on dev-boost-analysis; statistical
and scientific independent review passed. See dev/2026-09-08-boost-analysis.md and
dev/experiments/boost_campaign_20260908/ for full effects, all rejected cases,
golden comparison and scoring sensitivity. The penalty has mostly tiny
directional effects plus large tail exceptions; guard choice changes the
returned comparison. Production penalty and threshold remain undecided.

Completed: full 870-endpoint paired boost computation and analysis.
Finished 10:11:53 UTC (13:11:53 local), 870 pairs / 1,740 arms, zero errors;
worker invocation 1h32m57s. Worker/launcher exited and lock removed. Report
audit confirms actual options, 861 reconstructed/9 authentic inputs and finite
scores; runner verified all capture hashes. No penalty/threshold selected.
Isolated dev-boost-campaign runner `8c7919f`, numerical base `764a177`.
Former worker PID 14272 (launcher 31008). Two smoke pairs, source/import checks,
independent review and no-new-fit resume/recovery tests passed. Paired
penalties 0.1/0, pruning/guard disabled, common-GP pre/candidate rescoring,
retained VPs. See dev/2026-09-08-boost-campaign.md for execution and evidence.
Historical parked status below is superseded by this explicit restart.

Completed: Phase 6 bounded local eta-bound experiment on dev-eta-bound-comparison,
based on 03650a2. Approved mathematical checks, a timed three-arm pilot,
then at most 72 local fits (8 saved states x 3 treatments x 3 paired RNGs).
All 72 fits completed in 22.12 s excluding interpreter startup; zero failures.
Variants pass 35 mathematical/mutation/diagnostic checks; artifact audit,
resume verification and independent scientific review pass. Production
treatment remains open: B's noisy gains coincide with longer optimization;
C's penalty never activates here. Proposed next scope is equal-iteration
A/B fits with common eta shifts and repeated scoring; no new fits scheduled.
Manifest: dev/experiments/eta_bound_20260908.json. Design and evidence:
dev/2026-09-08-eta-bound-comparison.md. No full trajectories or production
penalty choice in this pickup; boost remains parked.

Completed Phases 3-5 follow:
Current PI direction: PARK Phase 2's full paired boost campaign until later;
main-loop fixes started from dev-next 3e879b6 and completed in order:
weighted GP covariance (Phase 3), acquisition regularization (Phase 4),
GP sampling termination (Phase 5), with separate numerical gates.
Current branch: dev-main-loop-fixes. Boost code is preserved at local commit
764a177 on dev-final-boost (not pushed). Its exact restart scope, local
artifact paths and remaining harness work are in
dev/2026-09-08-boost-penalty-pilot.md, "Parked experiment restart".
No boost job/watcher is running or scheduled. Q1/Q4 remain evidence-dependent.

Phase 3 covariance implementation and independent review complete: 146 tests
passed, 15 skipped, 11 exact fixtures passed. All six replay initial designs
match. Cigar seed 0 is worse and exceeds accuracy fences (still usable);
seed 1 and the other four default cases pass. Keep this flag for population
assessment, not a claim of all-green covariance-only trajectories. Phase 4
is now integrated at 90d08d3 and verified: 34 acquisition formula checks,
11 exact fixtures and a five-case replay against Phase 3 with zero flags.
Cumulative Cigar seed 0 is back inside every fence. Phase 5 is complete:
five default replays are exact against Phase 4; a forced threshold crossing
switches the next GP fit to stable samples. Independent review passed.
Final combined gate: 296 tests passed, 15 skipped, all 11 fixtures exact.
Next unresolved items: Phase 6 eta-bound comparison/PI choice, Phase 7
upstream gpyreg repair and final integration/population gates, and the parked
boost campaign when scheduled. Branch pushed; CI 138 passed on b0d3437:
full suite on Ubuntu/Python 3.12, 1,090 passed, 60 skipped, no retries.
The full OS/Python matrix and integrated population assessment remain pending.
See dev/2026-09-08-main-loop-fixes.md.

Historical Phase 2 progress follows:
Paired penalty timing pilot complete: seed 0 of noisy Student D8, Lumpy D10
and Cigar D15 exhaust; six boosts took 24.10 seconds and the complete pilot
32.018 seconds. Both arms started from identical reconstructed states/RNGs;
actual penalties 0.1/0 and disabled pruning/guard were verified. Full paired
captures are saved. Estimate about 3 hours for 870 pairs, with a 4-hour
planning allowance; see dev/2026-09-08-boost-penalty-pilot.md. Independent
review passed. The full population campaign has not been launched.

Reconstruction check completed numerically: all 870 traces are structurally eligible;
nine exact snapshots and three common-RNG boost pairs were compared. Normal
reproduces exactly, Logistic nearly matches, and noisy Rosenbrock seed 7
changes materially. An all-870 GP/SD screen took 17.6 seconds and found four
absolute SD differences above 0.001 nats (descriptive, not a validity gate).
See dev/2026-09-08-boost-reconstruction.md; independent artifact review is complete.
No full population optimization has been launched.
The PI deferred the acceptance threshold: retain both full VPs and ELBO/SD,
then assess acceptance rules post hoc. Rejected improvements are a tradeoff,
not grounds by themselves to disqualify a threshold. A paired penalty-on/off
boost experiment is the proposed next step, subject to reconstruction results.

Initial Phase 2 findings (2026-09-07):
Current work: Phase 2's initial final-boost comparison is implemented and
evaluated on dev-final-boost, based on 3e879b6. The completed neutral/reference work was
fast-forwarded into dev-next; CI 137 passed (1,043 tests, 49 skipped, no retries).
118 focused tests and eight exact oracle fixtures passed. All nine revised
trajectories (seven specified plus two targeted noisy counterexamples)
preserved exact main loops and initial designs, with raw restart state and
candidates retained. Both tolerances made identical decisions on those nine.
The guard fixes the severe logistic case but worsens noisy Rosenbrock seed17
from reference gsKL 0.352 to 6.402 by rejecting an improved candidate.
The production default remains deferred pending the broader comparison;
None retains legacy behavior on this experimental branch. Q1 is unchanged.
Evidence: dev/2026-09-07-final-boost-comparison.md and
dev/scripts/runs/latent_fixes/boost_20260907/.
Phase 2's live checklist is at the top of the latent-bug plan.

The reduced noisy reference extension is complete and integrated: 870 pairs
across 19 configurations, including 160 noisy runs across four configurations.
Allocation: original 16 x 50 plus D15 exhaust x 10, with noisy Rosenbrock D2
and Student D8 x 30 each. This supersedes the abandoned 150-run proposal;
no noisy Lumpy runs were included. The two timing seeds were reused.

All 60 additions passed validation; all 1,620 original files and the 810
historical baseline sidecars remain byte-identical. All 870 archives passed
integrity checks. The 76-test even/odd comparison had no flags; the final
five-case replay matched stored loop/final values and initial designs exactly.
Historical returned transformers remain uncertifiable. New runs: 53 converged,
7 Rosenbrock budget terminations; usability is 23/30 Rosenbrock and 14/30
Student, distinct from convergence. No run was discarded or rerun.

Current reference traces: dev/scripts/runs/golden/reference_870_20260907/.
Tracked sidecars/summary: dev/golden/baseline/.
Completion, validation, hash manifest and reports:
dev/golden/noisy_extension_20260907/README.md.
The historical 810 execution record and input population remain unchanged.
Workers used isolated reference 623f5cd over 7314a6a, gpyreg a2f8ddc,
original Python 3.12.6/NumPy 2.5.2/SciPy 1.18.1, and one BLAS thread.
The remaining 58 ran 17:02:17-21:08:49 UTC+03; no worker or watcher remains.

Completed development branch: dev-latent-neutral-fixes. Phase 0's stronger replay gate
and Phase 1's neutral fixes are complete (including CI 135's transformer
sharing corrections). Reference publication is committed and pushed in
b2ea859. CI 136 passed: 1,043 tests, 49 skipped, no retries (10m27s job).
Independent Sol review passed; all reference-extension checklist items are complete.
Phase 2 followed the reference completion; its current findings are above.
Final Q1 treatment and Q4's acceptance rule/default remain evidence-dependent;
preserve existing PyTorch feasibility
and modernization decisions. Astra orchestrates; Sol implements/reviews.

Historical pickup notes follow; current reference paths/counts above govern.

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
C:/Users/luigi/Documents/GitHub/pyvbmc/dev/scripts/runs/golden/reference_870_20260907/.
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
Use dev/golden/noisy_extension_20260907/README.md for current execution/provenance
details; dev/golden/extension_20260907/README.md preserves the earlier 810-run record.
Keep dev/golden/README.md a human-facing description of the current runs
(purpose, coverage, metrics, files and usage); batch chronology and release
stage history belong in the execution record, not that README (PI request).
