# Existing noisy-acquisition efficiency

Created 2026-09-13. Status: guarded-sinh and kernel-reuse implementation,
validation, independent review and integration into `dev-next` are complete.
The guarded-sinh numerical revision is `6734817`, from `83692ac`; the
[kernel-reuse execution record](#kernel-reuse-implementation-plan) records
its released gpyreg dependency and completed gates.

Experimental extension drafted 2026-09-16: **E2, E3 AND E5 COMPLETE; E6 ASSESSMENT RECORDED; FOLLOW-UPS PROPOSED, NOT APPROVED**. Branch:
`dev-noisy-acquisition-efficiency`, created from `dev-next` at `9cc6882`.
The [integration and search experiment](#integration-and-search-experiment)
below specifies the investigation. E0-E4 execution was authorized within
the recorded windows; the exploratory E5 pilot and its continuation are
separately authorized under the 2026-09-17 amendment and the continuation
launch clearance of the same day.
The PI accepted the benefit criteria, paired allocation and screening gates
on 2026-09-16. The revised experimental scope omits Bayesian quadrature.

## Pickup point and deferred measurements

The usage-limit interruption occurred after all 864 development-panel
treatment selections and 96 production baselines completed, before full
independent judging. Morning recovery on 2026-09-17 verified every terminal
record and bound payload hash and passed all 133 developer tests. No Python
experiment process remained running. The
[completion record](../experiments/noisy-acquisition-efficiency/integration-search/panel_completion.json)
identifies the saved allocation. The continuation completed panel judging,
MC diagnostics and all 108 paired panel timing cells. All 48 full-sieve
timing cells also completed; the four retained settings fail the necessary
cost gate. E2 closes without an integration replacement. E3 uses the
prescribed MC1600 fallback: all 384 S0-S3 development selections succeeded.
The additional 336 selections completed in 12.7 minutes of worker time,
against 12.2 minutes predicted and 30.1 minutes conservative. Independent
judging reached its 65536-node cap with all 576 scheduled comparisons
observed. A reviewed measurement adapter repairs reporting for paired
timing and memory without changing numerical sources. All 36 timing and
72 memory cells succeeded. S2 is frozen for holdout under the improved-choice
branch: its median of state-median timing ratios is 0.43387. The holdout
draft passed independent review. Following the recorded approval interruption,
the user explicitly authorized its 192 frozen-state selections and independent
judging at approximately 12:50 UTC. All 192 holdout selections succeeded
in 441.33 seconds against
443.69 seconds predicted and 766.64 seconds conservative. No target calls
or GP fits occurred. The holdout RQMC ladder reached its cap with 54
beneficial, 17 tied, 13 harmful and 12 unresolved comparisons out of 96;
11 material raw losses received the prescribed cross-method investigation.
Search-specific ordinary-MC checks are complete with no resolved disagreement;
the five-case saved-point loss investigation is also complete. It found no
confirmed harmful refinement against its own starting winner, and localized
several losses before refinement. The mixed holdout evidence does not support
a general replacement recommendation. E4's development entry gates are not
established. The user subsequently authorized the exploratory E5 pilot
specified in the amendment below. Production defaults are unchanged.
The 24-fit E5 pilot, its excluded exact reproducibility repeat and the
96-fit continuation for seeds 2002-2009 are complete: 120 of 120 fits
succeeded and all 60 initial-design pairs match. The user's launch decision
of 2026-09-17 (about 19:47 UTC, with a nine-hour window) is recorded in the
continuation's `launch_clearance.json`. Two bounded batches ran from 19:51
to 01:56 UTC, 6.02 hours of worker time against the 4.04-hour estimate, on
a machine whose seconds per target call drifted from 0.8 to 2.2 times the
pilot's over the schedule. The combined ten-seed summary, closure record and
independent review are published, and the [report](../results/2026-09-16-noisy-acquisition-integration-search.md#ten-seed-outcomes)
records the outcomes: S2 halves fit time at unchanged call counts but is not
shown to be free of target-specific accuracy loss (five convergence losses
against none, one usability loss and gsKL increases up to 103 times on
high-noise Rosenbrock, worse posterior-shape metrics on logistic regression)
against six usable gains on Student-t. The [E6 assessment](#e6-assessment-2026-09-18)
recommends against adoption and proposes targeted follow-ups that await a
user decision. No E5 process is running.

The 2026-09-17 continuation resumed from `4145832` in the existing checkout
and environment. The user authorized a further 2–2.5-hour window starting
at approximately 09:03 UTC. At 10:07 UTC the user confirmed approximately
90 minutes of remaining availability. That window's compute cutoff was
11:22 UTC, with new cell launches stopping by 11:12 UTC and about 15 minutes
reserved before the availability ends. Earlier allocation records retain
their original 11:03 UTC cutoff. The saved capture and panel artifacts
are reused. The 4096-node late Student-t judge pilot succeeded in 3.247 seconds
excluding imports. All five judge budgets, the independent MC diagnostic
and panel timing subsequently completed; the execution report records
measured costs and forecast errors. Further allocations use observed
per-state costs and the remaining window.
All completed-phase source locks and scientific gates remain in force;
the later E5 amendment records its separate authorization.

The pre-pilot planning estimate was E2 2-4 hours, E3 1-2 hours,
conditional E4 up to 1-3 additional hours if its gate passes, and 30-60
minutes for final review and reporting, totaling approximately 6-8 hours.
Measured pilots supersede these provisional estimates for each allocation.
Resume in the existing checkout and local
environment. The report identifies the local-only artifacts; no background
work requires reattachment.

The completed computational optimizations require no further campaign or
release action. The proposed next work compares integration methods on
frozen states, then tests fixed-budget search and, conditionally, adaptive
integration. GP fitting, initialization and retraining policy are excluded
from this experiment by the PI's scope decision of 2026-09-16.

A controlled estimate of the combined **whole-VBMC runtime speedup** from
guarded sinh and kernel reuse is deferred. The saved replay campaigns ran
under different laptop loads; their elapsed times cannot isolate the effect
of these changes. The validated speedups apply to matched acquisition calls.
A later runtime comparison should pair repeated before/after runs on the same
quiet machine and numerical environment, using identical configurations and
seeds and checking evaluation counts and trajectories. Suitable frozen pairs
are PyVBMC `83692ac` with gpyreg `a2f8ddc` before both changes and PyVBMC
`9c7d1b8` with gpyreg `39536b0` after both changes. This measurement is an
optional follow-up, not a prerequisite for the other experiments.

## Integration and search experiment

### Question, ownership and scope

Can numerical integration and search choose equally good or better VIQR
evaluation points at lower computational cost, with acceptable inference
performance across noisy synthetic and real-data problems?

This section owns the experimental design, allocation, decisions and eventual
execution checklist. The historical sections below retain the completed
arithmetic optimizations. The existing acquisition note and search report
own the earlier measurements; their scratch scripts and performance claims
are not assumed to reproduce on the current code.

In scope are ordinary Monte Carlo (MC), component-stratified MC, randomized
quasi-Monte Carlo (RQMC), fixed-budget sieve/re-scoring/refinement,
conditional adaptive integration, and evaluation
of the resulting decisions and inference. Candidate-side linear algebra may
be tested if integration setup remains a measured bottleneck.

Out of scope are changes to GP fitting, hyperparameter initialization or
sampling, retraining schedules, noise estimation, the variational objective,
the VIQR criterion, acquisition regularization, final boost, repeat policy,
IMIQR, Bayesian quadrature (BQ), new acquisitions and public defaults.
No HPC connection is required. Production adoption is a later decision
supported by this experiment, not a consequence of an arm winning locally.

### Live checklist and execution roles

2026-09-17 extended measurement window from approximately 11:49 UTC
(another 2.5 hours):

- [x] Complete the prescribed E3 judge ladder, escalating only unresolved
  primary or component comparisons under the frozen predicate. Freeze each
  complete budget summary before allocating the next budget.
- [x] Implement and independently review a separate measurement-reporting
  adapter, then run focused checks and freeze its source identity.
- [x] Pilot and cost the repaired paired timings and separate memory cells;
  allocate the remaining prescribed cells when measured costs fit.
- [x] Assess E3 development gates and the conditional E4 entry gates before
  freezing any further scientific allocation or opening holdout.
- [x] Review and freeze the S2-only holdout manifest, then execute its
  bounded 192 selections and fresh independent judge ladder. Leave the
  generic manifest's 12 timing and 24 memory cells scientifically unallocated.
  The user explicitly confirmed execution after the recorded approval
  interruption. Preserve that record and bind the new authorization in a
  successor record before unlock and launch.
- [x] Publish reviewed evidence, update the pickup point and stop workers.
  Development evidence, locked holdout draft, approval-pause record and
  available wrapper sources are preserved. Independent review verified the
  numerical evidence, source bindings and publication records. Holdout,
  ordinary-MC and loss-diagnostic evidence are complete and preserved.
- [x] Complete E3's independent ordinary-MC diagnostic using the committed
  search-crosscheck harness. Freeze development extrema, comparisons at
  least ten practical bands from zero and material raw losses; add explicit
  holdout catastrophe tags after its RQMC ladder. Evaluate fresh eight-rule
  MC budgets 32768 and 65536 with the same frozen bands. Investigate
  disagreement and retain unresolved diagnostics without retuning S2.
- [x] If the MC check confirms or cannot resolve material holdout losses
  in accepted-refinement outputs, investigate those saved choices with a
  bounded diagnostic comparing S0, the arm's pre-refinement winner and its
  selected point. Prepare and independently review the diagnostic adapter
  while the prescribed MC checks run; execute only eligible saved cases.
  Preserve the primary allocation and frozen configuration.

New launches stop by 14:04 UTC and compute stops by 14:09 UTC. This window
governs the original holdout RQMC allocation and supersedes the earlier
operational deadlines below; scientific gates are unchanged.
E5 remains unapproved.

At approximately 13:04 UTC the user extended availability by another
90 minutes and authorized further work within the E3/E4 gates, including
resumable allocations. Existing short judge allocations retain their frozen
deadlines. Successor MC allocations use guards of 14:29 UTC for new launches
and 14:34 UTC for stopping compute. Independent review
identified that the completed E2 MC diagnostic does not cover E3 search
comparisons; the existing `noisy_acq_search_crosscheck.py` provides the
prescribed search-specific check. Both budgets and the conditional loss
diagnostic completed with the frozen finalist unchanged. E4's scientific
entry criteria are unchanged.

The conditional loss diagnostic reconstructs the pre-refinement winner from
the authenticated selection NPZ: take the first minimum of `shortlist_scores`,
map it through `shortlist_indices`, and read the corresponding row of
`coarse_candidates`. Verify that this is a newly generated row using
`len(coarse_candidates) - generated_count`; repeated rows undergo a separate
coordinate substitution and cannot use this reconstruction. Accepted
refinement cases should pass this check because repeat winners return before
refinement. Bind all source artifacts and reconstructed coordinate
hashes before evaluation. Its eligible cases are the final holdout's material
raw losses with an accepted refined point and no fallback; inspect only those
still confirmed or unresolved by the prescribed MC diagnostic. For each case,
evaluate S0, the pre-refinement winner and the selected point together using
eight fresh ordinary-MC rules at 32768 and 65536 nodes, with a separate seed
domain and the original frozen practical band. Compare selected versus own
winner, own winner versus S0 and selected versus S0, applying the existing
precision and consecutive-budget rules. Record raw reduction as well as the
complete score. Keep these diagnostic contrasts outside the 96 primary
comparisons. A harmful selected-versus-own-winner contrast identifies an
accepted refinement regression when the direct selected-versus-S0 loss is
also resolved. A resolved non-harmful selected-versus-own-winner contrast
together with a resolved own-winner-versus-S0 loss locates the direct loss
before refinement. If both stage contrasts are within their bands but the
direct contrast is harmful, report accumulated sub-band differences rather
than absence of a loss. Raw localization uses the separate 10% material-loss
threshold and is inapplicable when any compared point has an active penalty.
Mixed or unresolved contrasts do not establish a unique cause. These observations can explain
limitations and guide a later development proposal; they cannot tune S2,
change the holdout verdict or establish an E4 entry condition from holdout.

The reporting adapter uses new source files and a new measurement namespace.
Its manifest binds the unchanged parent search manifest, the completed
384-selection snapshot, frozen numerical sources, adapter source, original
measurement cells/seeds and the failed timing evidence. It calls the existing
timing and allocation routines unchanged. After clocks and memory tracing
stop, it replaces the full selection result with a validated compact record:
selected-coordinate digest, seeds, arm, counters, nullable cache index,
target-call flag and fallback reasons. Raw diagnostic arrays are omitted.
Focused tests cover nonfinite discarded diagnostics, preserved measurements,
state/identity guards and strict publication/reuse. The original failed
terminal and partial JSON remain evidence. No selection or numerical source
identity changes as part of this reporting transition.

The completed development judge supports a conditional S2 holdout decision
under the improved-choice branch: 54 beneficial, 19 tied, 5 harmful and
18 unresolved comparisons out of 96. The 5% harmful ceiling belongs to
the faster-selection branch; the improved-choice branch requires comparable
cost, within 10%. The two material Student-t raw losses have an identified
mechanism: exhaustion of the shared row budget returns the re-scored
shortlist fallback. Independent review accepts this as satisfying the
investigation requirement while retaining both losses in the evidence.
This explanation establishes neither harmlessness nor noninferiority.
If the median of the twelve state-median S2/S0 paired timing ratios is at
most 1.10, freeze S2's
unchanged MC1600 configuration and this gate assessment before opening
holdout. If it exceeds 1.10, S2 fails both benefit branches. Any holdout
allocation needs a separate manifest, bounded cost estimate, fresh split
streams and reviewed measurement provenance; it may not retune the arm.

The completed timing ratios are S1 0.18845, S2 0.43387 and S3 0.83991.
The S2 decision and unchanged configuration were frozen in
`search_development_gate.json` and `search_holdout_selection.json` before
holdout inspection. Prepare a locked holdout manifest with twelve seed-1
states, eight repetitions, and tags S0/finalist. Independently review its
configuration, source identity, split streams and exact 192-cell allocation
before freezing the unlocked executable manifest. Reuse the existing
per-cell selection and judging APIs; do not run the generic controller that
would also launch timing and memory. Allocate selections from matched
development S0/S2 worker costs, with a conservative allowance no greater
than 25 minutes and the existing window cutoffs. Then cost the initial
4096-node judge round and allocate consecutive budgets only for pending
states, preserving an immutable summary at each rung. The holdout supplies
96 primary comparisons, no retuning, and no duplicate MC1600 control.

Before holdout judging, independent review confirmed the interpretation of
the improved-choice branch. Development timing supplies its frozen
comparable-cost evidence; no holdout runtime claim is inferred. Use all
96 scheduled comparisons and report each trajectory and resolved-only
counts. More than 20% unresolved prevents a positive claim. Reproducible
numerical failures or unexplained material raw losses block advancement.
The plan specifies no minimum beneficial fraction, benefit-to-harm ratio or
per-trajectory pass threshold for this branch; do not introduce one after
inspection. Assess whether the independently judged pattern supports an
improvement across trajectories, retain mixed evidence as inconclusive,
and keep explained regressions visible. E5 remains a separate decision.

2026-09-17 additional window from approximately 11:24 UTC (30+ minutes):

- [x] Validate and reuse all 48 completed E3 selections; execute the
  remaining 336 frozen selections with per-cell checkpoints and deadline
  guards. Stop new launches by 11:51 UTC and compute by 11:54 UTC.
- [x] Re-estimate timing and independent judging from a bounded pilot if
  selections finish with sufficient time remaining.
- [x] Complete a bounded independent-judge round and assess the next budget.
- [x] Paired timing required a reporting repair: the frozen timing pilot
  cannot serialize NaNs inside its diagnostic `cache_indices` array. Keep
  the failed terminal record and preserve all frozen selection identities.
- [x] Verify saved evidence, update the pickup point and stop all workers.
  The extended window below owns subsequent measurements and closeout.

This user-authorized window superseded the earlier compute cutoff without
changing sources, settings, seeds or scientific gates. The conservative
selection estimate is approximately the entire window, so unfinished cells
remain resumable if measured costs prevent completion. E5 is unapproved.

After all 384 selections finish, bind their terminal hashes and run two
bounded cost pilots on late logistic regression: the 4096-node independent
judge and S3's prescribed seven-round paired timing against S0. Both pilots
reuse their original frozen seeds and count toward their respective full
matrices. Estimate remaining costs before allocating either matrix. If only
one fits, prioritize the 36 paired timing cells. The initial judge budget
cannot resolve complete-score quality under the consecutive-budget rule;
it supplies setup and evidence for the next prescribed budget. Complete a
judge round before freezing its summary or allocating the next budget.
Keep memory measurements and holdout separate; E5 remains unapproved.

2026-09-17 bounded continuation from `4145832` (2–2.5-hour availability):

- [x] Run the saved E2 4096-node development judge pilot and estimate costs.
- [x] Allocate resumable E2 judging and timing from measured costs, stopping
  compute by the amended 11:22 UTC cutoff and preserving a 15-minute margin.
- [x] Enforce E3's shared 50-iteration limit, verify the correction, and
  complete a bounded pilot followed by repetition zero on all twelve states.
- [x] Verify saved results, independently review the continuation evidence,
  and record the next pickup point. E5 remains unapproved.

The bounded continuation applies the existing E2 gate to full-sieve timing
before allocating fresh full-sieve quality comparisons. Freeze MC2048,
stratified-MC2048 and stratified-RQMC512/2048 from the completed panel
evidence. These retain the MC/component-allocation controls and the two
RQMC accuracy/cost settings without confirmed material raw loss. The other
five panel settings have confirmed material losses and exceed the pooled
harmful-selection screen. MC2048 and RQMC512 retain explicitly unresolved
severe early Rosenbrock-noise-3 raw-loss checks; this shortlist is for
investigation, not promotion.

Run a bounded 8192-candidate timing pilot, then estimate and allocate the
prescribed seven paired rounds per state/setting, including preparation.
If a setting's median of state-median treatment/baseline ratios exceeds
1.10, neither approved benefit branch can pass; end that arm without
separate selection, judging or holdout allocations. A setting at or below
1.10 continues through the existing quality gates. Report every state's
ratio and range;
timed selections do not substitute for independent quality comparisons.
This ordering changes no scientific threshold, method or budget and was
independently reviewed before allocation. Incomplete timing leaves the
setting pending, not failed. No new full-sieve setting may be added from
these results.

If the four full-sieve settings all fail the cost gate, E3 uses the
prescribed ordinary-MC1600 fallback. Before numerical allocation, enforce
the shared 50-iteration cap across local starts and verify it independently.
Freeze the full twelve-state, eight-repetition S0-S3 development manifest
(384 selection cells), then launch only repetition zero of all four arms
on early Rosenbrock noise 3, late logistic regression and late Student-t.
These twelve pilot cells cover adverse selection geometry, multiple GP
hyperparameter samples and higher dimension. Bind the subset, E2 decision,
source identity and deadlines in a prelaunch allocation record.

The pilot assesses operation, numerical calibration and cost. Check state
and RNG preservation, absence of target calls/GP fitting, shared iteration
and row limits, every available step-halving diagnostic, fixed positive
objective scaling, solver outcomes and fallback reasons. Retain finite-
difference failures as prescribed fallbacks. Decide whether the frozen
defaults support further development testing; do not infer quality or
choose a search arm from these unjudged selections. If settings change,
freeze a new identity and allocation rather than reusing incompatible
cells. Estimate remaining work from measured state/arm costs before
allocating another bounded subset. Use individual cell commands: the
unbounded search controller also launches timing, memory and judging work.

- [x] E0: freeze sources, state allocation and measurement protocol.
- [x] E1: build and verify developer integration and judging tools.
- [x] E2: compare MC, stratified MC and RQMC; panel selections, prescribed
  RQMC judge ladder, MC crosscheck and timing complete. All four retained
  full-sieve settings fail the necessary cost gate; no integration finalist.
- [x] E3: compare fixed-budget re-scoring and local search; shared-iteration
  correction independently reviewed and 25 focused tests passed. The
  MC1600 development manifest is frozen; all 384 selections succeeded.
  Development judging, 36 paired timing cells and 72 memory cells are
  complete. S2 is frozen under the improved-choice branch; all 192 holdout
  selections and its RQMC judge ladder succeeded. Search-specific
  ordinary-MC diagnostics and the five-case loss investigation are complete.
  Mixed holdout evidence does not support a general replacement recommendation.
- [x] E4: entry gates assessed; neither is established. No adaptive arm is
  allocated. Full-sieve E2 quality comparisons were cost-screened out;
  accepted harmful E3 refinements occur on only one development trajectory.
- [x] E5: the 24-fit pilot and the 96-fit continuation are complete and
  independently reviewed: 120 fits, 60 complete pairs, no failure. The
  ten-seed comparison is recorded in the report.
- [x] E6: the [assessment](#e6-assessment-2026-09-18) recommends retaining
  the production search and proposes frozen-state follow-ups before any
  further inference allocation. No default changes.

Astra (`gpt-6-astra`, high) orchestrates E0, scientific decisions in E2-E5,
and E6. Sol (`gpt-5.6-sol`, high) implements the developer harness and
executes the specified E1-E5 checks; use separate Sol agents for independent
review. At most one agent runs compute, including tests and timing, at once.
Other agents perform static review. `$task` execution began on 2026-09-16;
E0 and the independent E1 harness work proceed together. No additional
approval is needed for phases within the authorized window and gates.
A failed gate ends that arm or yields an
inconclusive result; it does not authorize expanding the experiment.

### Scientific and software contracts

**Two point sets.** A candidate is a possible next target evaluation. An
integration node estimates a candidate's utility using the fitted GP. The
frozen-state experiments make no target calls and no GP fits. State capture
and E5 inference do make target calls under the unchanged fitting policy.

For a fixed VP, let `R(x)` be the posterior-weighted residual IQR after a
hypothetical observation at candidate `x`, averaged over the existing GP
hyperparameter samples. Let `R0` be the corresponding current IQR. Compute
both using normalized integration weights. The usual VIQR score differs
from `log(R(x))` by a candidate-independent constant for uniform MC nodes.
Normalize those constants when comparing different node counts; do not
compare raw unnormalized sums from different budgets.

**The complete acquisition must be preserved.**
`AbstractAcqFcn.__call__` maps integer coordinates, applies a
candidate-dependent variance penalty and rejects points near hard bounds.
The judged score is `F(x) = log(R(x)) + P(x)` for valid continuous candidates,
with the existing penalty `P` and masks. Its equivalent positive score is
`J(x) = exp(P(x)) * R(x)`; evaluate comparisons stably in log space.
The GP noise estimate, hyperparameter averaging and variance clipping must
match the frozen source. Record raw VIQR and the complete score separately.

Minimizing residual IQR and maximizing IQR reduction share a minimizer for
fixed nodes and weights before candidate-dependent penalties. Adding the
same penalty to their logarithms need not preserve that equivalence.
Therefore local search initially uses a positive affine rescaling of the
complete `F`, fixed for that optimization. Test `iqr_reduction` only where
the penalty is identically inactive throughout the searched domain and
equivalence has been verified. Merely observing zero penalty at the start
is insufficient. Do not disable or redesign regularization for an arm.

**Weights require an explicit evaluator.** Standard VIQR's `iqr` path
currently assumes constant weights and does not consume `ln_weights` in
its sum. Replacing nodes or populating that field is insufficient for
component-stratified rules. Build a private developer evaluator that
supports explicit weights, preserving all hyperparameter contributions and
the full wrapper semantics. Establish equal-weight parity first. Reuse the
current production path as the timed baseline, including its kernel reuse;
do not handicap it with the experimental evaluator's overhead.

All integration is in the VP's internal coordinates, where component `k`
has mean `mu[:, k]` and diagonal standard deviation
`sigma[0, k] * lambd[:, 0]`. Preserve the transformer and use its original
coordinate bounds through the existing wrapper. Freeze the GP, VP, logger,
options, observation-noise inputs, cache and search bounds together.
Capture object/RNG digests before and after every frozen-state treatment.
`VariationalPosterior.__deepcopy__` shares its generator: install an
independent generator on experiment copies before drawing. Pass candidate
copies through integer snapping so shared panels cannot be modified.

Keep evaluation nodes fixed throughout a local optimization. Frozen-state
candidate generation, integration replicates, local starts, acceptance
checks and final judging use explicitly separate deterministic streams.
Common nodes across candidates within one replicate reduce noise in their
differences. Development judging may select among the predeclared bounded
method/settings grid, but must never optimize a candidate or decide a
runtime fallback. Holdout and final judging are evaluation-only and cannot
tune methods or settings. Independent RQMC scrambles, not
the scatter within a scramble, provide its replicate uncertainty estimate.

### E0: freeze the baseline and representative states

**Executor:** Astra orchestrator; Sol may implement capture plumbing.

Freeze the exact branch base, gpyreg revision, Python/NumPy/SciPy versions,
BLAS/thread configuration, options, target/data/truth hashes and seed
allocation in a versioned manifest before measurements. The starting
gpyreg pin is `9e70e6ba53f7607d05c2d9cc2fa9f41cd12b8f3b` (1.2.1); verify
the imported source rather than relying on a package version string.
Later `dev-next` changes do not silently change the baseline. If integration
onto a newer base becomes necessary, record the difference and rerun the
affected checks before using old measurements to support it.

Use these six existing golden configurations, with their existing bounds,
budgets, noise and initialization conventions:

| Configuration | Coverage |
| --- | --- |
| `rosenbrock_D2_noise1` | Lower noise and low dimension |
| `rosenbrock_D2_noise3` | Same geometry at higher noise |
| `logreg_D5_noise3` | Bounded synthetic target |
| `student_D8_noise3` | Heavy tails and higher dimension |
| `multisensory_s1_D6_noise1.3` | Real-data likelihood |
| `timing_D5_noise2.2` | Real-data likelihood with larger target cost |

Target 24 states: two trajectories per configuration (seeds 0 and 1),
two acquisition checkpoints per trajectory. The early checkpoint is the
first acquisition after at least `max(20, 2*D)` charged target evaluations;
the late checkpoint is the first after 60% of the configured budget.
For these ordinary runs initialization cost is zero. If a run terminates
before the late threshold, use its last acquisition checkpoint, provided
it is distinct from the early one; otherwise mark the missing cell. Do not
force extra iterations or discard failed trajectories to fill the matrix.
Capture after the current GP/VP update and before candidate selection.

Seed-0 trajectories are development states; seed-1 trajectories are held
out. Never split early/late states from the same trajectory across those
groups. Add constructed stress checks for tiny mixture weights, narrow
components, warped/bounded transforms, nearly zero acquisition reductions,
penalty-active candidates and both GP factor representations. These are
correctness/stress tests, not extra independent target observations.

Audit existing captures using `validate_viqr_sinh.py:restore` and
`probe_viqr_kernel_reuse.py:captured_states`; consult the ignored
`dev/scripts/runs/LOCAL.md` for availability. Historical captures are useful
supplementary states, but are not automatically current-base trajectories.
The local inventory at planning contains six early captures (`N=10`) and
one late Rosenbrock-noise-3 capture (`N=150`). It cannot supply the proposed
24-state matrix or late higher-dimensional/real-data coverage. The original
mid-run search scratch scripts and pickles were not committed; rebuilding
the experiment must not depend on finding them.
Check complete capture contents and source provenance; do not attempt to
reconstruct omitted importance samples from an ordinary lean history.
If the current-base matrix is unavailable, capture it in at most 12 baseline
fits. Archive acquisition inputs using the existing plain-array state
machinery where possible. Preserve any necessary additional metadata in
an explicit schema; never silently discard unsupported state.
Preserve live GP posterior factors and `temporary_data` using the existing
capture extension: ordinary state reconstruction recomputes factors and can
move acquisition rounding. Save the generator state at the capture point
explicitly; supplying a seed to `build_state` does not restore a mid-run
stream. Keep provenance for whether candidates/nodes were captured or drawn
afresh by the experiment.

Freeze candidate panels with the current `_get_search_points` and existing
repeat/cache rules. E2 uses a common 8192-candidate pool per state. Its
512-candidate development panel includes the baseline's top 32 and 480
seeded candidates from the remaining pool. Record this selection bias;
panel results alone cannot establish full-sieve selection quality. E3
generates each sieve size through the existing generator with recorded
seeds; taking the first rows of a larger sieve can distort its source mix.

**Acceptance:** complete manifest and coverage table; current public VIQR
reproduced on captured nodes/candidates; missing states and unverified
historical claims explicitly identified. Check data and truth availability
before scheduling real-data runs. Missing truth prevents inference-quality
claims for that target even if acquisition comparisons can proceed.

### E1: developer tools and independent judging

**Executor:** Sol implementation; Astra checks the measurement contract.

Implement the following developer-only modules (names are proposed paths):

- `dev/scripts/noisy_acq_experiment.py`: manifest-driven commands for state
  inventory/capture, integration comparisons, search, inference and summaries.
- `dev/scripts/noisy_acq_quadrature.py`: node rules, explicit-weight VIQR,
  stable residual/reduction calculations and replicate judging.
- `dev/scripts/test_noisy_acq_experiment.py`: protocol, isolation, resumption,
  full-score parity and focused numerical checks.

Reuse `benchmark_targets.py`, `golden_trace.py`, `population_run.py`'s
manifest-checked completion records, the existing captured-state loader,
and the sequential warmed-worker timing pattern in
`validate_viqr_kernel_reuse.py`. Do not modify their historical outputs or
copy a complete numerical solver into a second implementation. If a small
shared helper is needed, extract it with value/RNG regression coverage;
experimental weighted behavior stays behind the developer harness.

Concrete implementation steps:

1. Add a state schema check and digests, load the real baseline callable,
   and reproduce its equal-weight results after accounting for constant
   normalization. Cover single/multiple hyperparameter samples, both GP
   factor representations, penalties, masks, finite and zero reductions.
2. Implement explicit positive weights and test constants and analytic
   Gaussian moments under the VP. Keep all positive-weight components;
   invalid weighted evaluations return structured failure records rather
   than a clipped number or silently substituted method. Check stable
   positive-weight residual and reduction sums against direct dot products
   in well-conditioned cases and higher-precision references in
   cancellation stress cases.
3. Add a deterministic fixed-node cache builder and compare it with the
   captured MC cache; ordinary `active_importance_sampling` draws nodes
   and cannot be used as a fixed-node constructor. Implement randomized
   node streams without changing the captured VP's
   generator. Repeated calls on the same manifest must reproduce results;
   draw ordering in one arm must not change another arm or its judge.
4. Build the independent judge below, retaining raw replicate estimates,
   candidate provenance, uncertainty and unresolved status.
5. Add atomic per-cell output, manifest/hash checks on resume, and summaries
   that count failed/missing/unresolved cells in their denominators.

Judge all methods' selected candidates together with the baseline candidate
and a fixed reference shortlist. Use eight independent component-stratified
RQMC replicates initially, each with an actual budget near 4096 nodes.
Double per-replicate budgets to 8192, 16384, 32768 and at most 65536 only for
unresolved comparisons. Use the same judge nodes for paired candidates.
Check a separate ordinary-MC estimate for the development extremes and
any apparent catastrophic win/loss. Disagreement triggers investigation,
not automatic preference for RQMC.

The MC diagnostic includes each development state's most positive and most
negative mean complete-score difference, every comparison whose magnitude
is at least ten practical bands, and every confidently flagged material
raw loss. Deduplicate candidate pairs and evaluate their union with eight
fresh MC replicates at 32768 and 65536 nodes. Retain the original pilot
band and the same uncertainty and consecutive-budget checks. These two
budgets are diagnostic allocations; disagreement or unresolved MC evidence
cannot certify the RQMC verdict. Add apparent holdout catastrophes explicitly
without using them to tune a treatment.

For candidates A and B, form paired per-replicate complete-score differences
`d_r = F_r(A) - F_r(B)` using the same nodes within each replicate. Use
the paired mean and a two-sided 95% Student-t interval with seven degrees
of freedom for eight replicates. These are approximate operational
intervals; nonlinearity of the logarithm and heavy-tailed contributions
require the budget-stability and cross-method checks. Also report scores
computed from the pooled residual estimates. If averaging replicate log
scores versus taking the log of pooled residuals changes the classification,
increase the budget or leave the comparison unresolved.

Define a per-state practical band before comparing treatments:
`eps_F = max(log1p(1e-6), 0.01 * abs(log(R0) - log(Rbaseline)))`, using
independently judged raw residuals for the production winner. Compute the
band once per state from a dedicated pilot judge stream. The absolute
floor prevents near-flat surfaces from demanding rounding-level accuracy;
the second term scales with the baseline's look-ahead benefit. If the
baseline gain is unresolved, use the absolute floor and mark that state.
This band is an engineering choice, not a numerical precision claim; report
unthresholded differences as well. It is never passed to a runtime policy.

Relative to B, A is beneficial if the upper interval endpoint is below
`-eps_F`, harmful if the lower endpoint is above `+eps_F`, a practical tie
only if the entire interval lies inside the band, and unresolved otherwise.
Require the same classification at two consecutive node budgets; for a
resolved directional result also require the interval half-width to be at
most one quarter of its magnitude. Compare adjacent budgets up to the cap;
there is no requirement for a budget beyond 65536. Retain unresolved cells
at the cap. Check the raw-reduction material-loss gate below even when the
complete scores are practically tied. These are resolution checks, not
simultaneous hypothesis-test claims.
An unseen narrow contribution can defeat every replicate; analytic stress
cases and the cross-method check address that risk without claiming proof.

The primary score is the complete acquisition `F`; also report stable raw
IQR reduction `R0-R(x)`. Evaluate reductions directly where possible,
rather than subtracting nearly equal exponentials. Report regret relative
to the best assessed candidate in the recorded union, explicitly not a
global optimum. Do not report reduction ratios when their denominator is
unresolved or effectively zero. A true-GP pathological flat surface is an
uninformative selection case to record, not permission to change GP fitting.

For implementation, assess baseline-gain resolution from the eight paired
relative reductions `(R0-Rbaseline)/R0`: the 95% Student-t lower endpoint
must exceed `1e-12`. This numerical floor is fixed before inspecting
treatment outcomes. Assess a material raw loss through paired differences
`reduction_arm - 0.9 * reduction_baseline`, using a common log-derived
scale to avoid overflow. A confidently negative difference must satisfy
the same consecutive-budget and directional-precision checks as the
complete-score comparison. Retain unresolved raw-loss checks separately
from complete-score ties.

Use quiet-machine sequential timings, single-threaded BLAS, warmed code,
alternating treatment order and seven paired timing rounds. Report total
node generation/preparation plus acquisition/search cost, separately from
judge time and capture/import time. Count cache construction once per
acquisition selection; no amortization across subsequent GP/VP updates.
Measure memory separately from timed rounds and bound allocations by
chunking nodes/candidates; never compare an instrumented arm to an
uninstrumented baseline.

**Acceptance:** focused developer tests pass; equal-weight objective and
full-wrapper parity pass; manifests isolate source, seeds and holdout data;
timing includes preparation; independent judge can decline a verdict.
Only after these checks may performance measurements count as evidence.

### E2: integration methods at fixed budgets

**Executor:** Sol implements/runs; Astra chooses at most two finalists.

Keep candidate locations fixed. Compare the exact production MC baseline
at its ordinary 100 nodes, ordinary MC at budgets 128/512/2048,
component-stratified MC at those budgets, and component-stratified scrambled
Sobol integration at those budgets. The stratified-MC control separates
the gain from component allocation from the gain from low-discrepancy nodes.

For stratified rules allocate at least one node to each positive-weight
component, then assign further nodes according to weight, without inspecting
the candidate integrands. For Sobol, preserve power-of-two blocks per
component: start at one each and double the component allocation with
largest `w_k/n_k` that fits within the total budget, with a fixed tie rule.
Record actual counts and unused budget. Use the same counts for the
stratified-MC control. Every node in component `k` has weight `w_k/n_k`.
If the budget is smaller than the number of positive-weight components,
mark the rule unavailable at that budget; do not drop components. Protect
inverse-normal transforms at machine endpoints and test the handling.

Use eight independent integration repetitions on development panels. Retain
one fresh production-MC100 selection for each state and repetition, paired
with that repetition's treatment selections. The captured production
winner remains a fixed reference and supplies the practical-band pilot.
Retain the non-dominated accuracy/time settings, at most two settings per method,
for the full 8192-candidate development sieve. Timing finalists receive
the seven paired rounds from E1. Choose at most two integration methods
and one fixed setting per method before opening held-out trajectories.
Evaluate those fixed choices with eight fresh repetitions on the holdout
full sieves. If holdout fails, report the failure; do not retune on it.

**Screening gate:** an integration treatment should either reduce median
selection time by at least 20% with no more than 5% confidently worse
selections, or improve independently judged choices at comparable cost
(within 10%). Count unresolved comparisons separately; if more than 20%
are unresolved, make no positive quality claim. A reproducible loss of
over 10% of a resolved positive raw reduction when penalties are inactive,
or a repeated adverse complete-score selection when penalties are active,
requires investigation and blocks promotion until explained. Report every
state/trajectory and the worst losses; pooled medians cannot hide a failing
target. Fractions use all scheduled state-by-integration-repetition cells
for that treatment, including failed cells; missing or failed method cells
cannot count as ties or wins. Also report resolved-only results separately.
Report each trajectory's cell counts and harmful/unresolved fractions;
replicates and early/late states are clustered observations, not independent
inference runs. A reproducible numerical failure blocks promotion even if
its pooled frequency is low. Thresholds are proposed engineering screens,
not statistical proof
of noninferiority. If no method passes, retain ordinary MC for E3.

### E3: fixed-budget search

**Executor:** Sol implementation/run; Astra assesses tradeoffs.

Use the best eligible integration rule from E2, retaining a plain-MC
version of the strongest search arm to separate integration and search
effects. S1-S3 all use the production ordinary-MC 100-node coarse rule.
Their accurate re-scoring and refinement use the single E2-selected rule
and its frozen node budget. The ordinary-MC control uses exactly 1600
accurate nodes, with every other search setting unchanged. Couple candidate-
generation streams across arms where possible and record actual candidates
when different sieve sizes consume different streams. Use eight paired
selection repetitions per state, with a shared candidate-generation seed
across arms and a shared accurate-rule seed across S1-S3. Derive a separate
integration stream for the ordinary-MC control. Compare these development
treatments:

| Arm | Candidate search |
| --- | --- |
| S0 | Unmodified production search, including CMA-ES and current setup |
| S1 | 1024 candidates, ordinary coarse integration, top 8 accurately re-scored |
| S2 | S1 plus one L-BFGS-B refinement from the re-scored best |
| S3 | S1 plus up to four spatially separated local starts |

Judge every arm against S0. Also retain the paired contrasts S2 versus S1,
S3 versus S2 and S3 versus S1, using the same repetition's candidate stream.
These contrasts separate refinement and multiple-start effects and identify
harmful refinement for E4's entry gate. They use the existing candidate
union and independent judge streams; unresolved component contrasts remain
eligible for budget escalation even when an arm's comparison with S0 has
resolved. Keep their denominators and classifications separate from the
primary S0 screening counts.

After choosing the strongest development search arm, compare its E2 rule
directly with the same arm using MC1600 on common independent judge nodes.
The control allocation includes S0 and both versions of that arm; reproduce
the original version with its frozen development settings and seeds. Retain
this direct contrast on holdout and through unresolved-contrast escalation,
with a separate denominator. If the selected rule is already ordinary MC1600,
the duplicate control is unnecessary. Timings paired separately with S0 are
descriptive for the direct integration contrast and do not establish a paired
runtime effect between the two versions.

Use the E2-selected accurate budget; include the historical ordinary-MC
1600-node refinement as one development comparator, rather than assuming
it remains optimal. Test a 2048-candidate sieve only if 1024 shows coverage
loss on development states. Do not turn this into a Cartesian sweep over
sieve sizes, node counts, start counts and stopping thresholds.

Re-score the original coarse winner alongside the shortlist and compare
all refined points using the same accurate integration rule. Always retain
that winner as a possible return value. Such a fallback protects only the
arm's own smaller sieve; it does not reproduce S0's 8192-point search.

Use batched finite differences in internal coordinates, with a step tied to
coordinate scale and checked by step halving on development cases. Start
with at most 50 iterations and 1000 candidate-row evaluations total across
all starts, including finite differences. Calibrate numerical tolerances
and fixed affine objective scaling on development states, then freeze them.
Use existing search bounds, with explicit shape validation. Preserve hard
bounds, integer handling, repeat rows and cache indices. Integer variables
and nonsmooth/invalid refinement cases return the arm's re-scored candidate;
log the reason and include fallback cost and outcome. Nearest-neighbor
noise estimates can make the objective nonsmooth even for continuous
parameters, so a solver's success flag is not a quality certificate.

If matrix setup dominates, permit one separately labelled candidate-side
evaluation variant. Reuse current GP factors; verify values for both factor
representations, multiple hyperparameter samples, noise scaling and chunks.
No GP-fitting code changes are permitted. Include kernel storage and all
preparation costs; retire the variant if it does not improve total time.

Choose a search finalist on development results under E2's quality/time
screens. Freeze it before holdout. Use fresh independent streams for E3's
judge; holdout results are evaluation only and cannot tune later arms.
If E2's holdout has already been inspected, E3 is a staged development
comparison on those same geometries, not a second independent confirmation.
The fresh E5 trajectories supply the final out-of-trajectory assessment.

**Acceptance:** independent complete-score quality and timing pass the
screens; accurate re-scoring/refinement gains are separated; fallback and
multiple-start costs are included; baseline default behavior remains exact.

### E4: conditional adaptive integration

**Executor:** Astra specifies the rule from development evidence; Sol runs it.

Enter only if development evidence meets one of these prespecified gates:

- Within one integration method, at least two fixed budgets are needed:
  a lower budget is tied/beneficial
  under E2's classification on some cells but harmful/unresolved on others,
  and retrospectively choosing each cell's cheapest passing fixed budget
  would save at least 20% median selection time versus the cheapest fixed
  rule that passes globally. This is an optimistic upper bound that excludes
  adaptive-check overhead, not a runtime policy with access to the judge.
- The best fixed search rule accepts a harmful refinement on at least two
  development trajectories, and a higher tested fixed integration budget
  removes those same errors. This supports testing an accuracy gate even
  if a global fixed rule has not passed.

To verify the second gate, if component-contrast harm appears on at least
two development trajectories, freeze a confirmation allocation for the
affected state/repetition cells. Re-run both members of each implicated
contrast at the next higher already-tested E2 budget of the same integration
method, keeping search settings and paired candidate-generation streams
fixed. Test only that one higher budget. Include the original selected and
reference points in the independent judge union, so a changed reference
cannot by itself establish that the original error was removed. Require
the higher-budget contrast to pass and its selected point to avoid the
original harmful loss against the original reference. If no higher tested
budget exists, this gate is unavailable. The MC1600 control separates
integration methods and search effects; it does not establish this gate.

Evaluate the first gate on the full-sieve development results, using the
retained budgets within the integration method that the adaptive prototype
would use. A passing fixed rule satisfies the quality requirements; assess
its speed relative to production separately. Use each state's warmed median
time for the relevant method/budget when evaluating the
retrospective cell choices. A cell with no passing budget contributes no
time saving and remains failed or unresolved in the quality accounting.
The independently drawn fixed-budget rules make this an optimistic screen;
it supplies no evidence that a runtime policy can identify the same cells.

If neither gate passes, record E4 as unnecessary and continue with the
fixed rule. Adaptive sieve size is deferred: keep the selected sieve fixed
to isolate this experiment.

Prototype one rule with at most two attempts and a cumulative ceiling of
2048 integration nodes for refinement and acceptance. The ordinary
100-node coarse sieve is additional and charged separately:

| Attempt | Fixed search nodes | Fresh acceptance nodes | Maximum cumulative nodes |
| --- | ---: | ---: | ---: |
| 1 | 128 | Four independent replicates of at most 96 each | 512 |
| 2, if needed | 512 fresh | Four independent replicates of at most 256 each | 2048 |

Use independent RQMC scrambles or MC draws for the acceptance replicates.
Record actual component allocations and unused budget. If any replicate
cannot cover all positive-weight components, skip to the next feasible
attempt within the cap or fall back. Nodes discarded by a failed attempt
still count against the ceiling. There is no free intermediate check at
128 nodes and no recycling of acceptance nodes into a later proposal.

Freeze search nodes within each attempt, propose one candidate from the
shortlist/refinement, then compare it with the arm's coarse winner on the
fresh acceptance nodes. Use a paired Student-t 97.5% interval with three
degrees of freedom for the four complete-score differences. Accept only
when its upper endpoint is below `-log1p(1e-6)`. This fixed runtime threshold
does not use the external judge's baseline-dependent band. Otherwise move
to the second attempt or return the coarse winner. A log-mean versus
pooled-residual ordering disagreement also prevents acceptance.
This is an empirical decision rule whose calibration must be measured,
not a guaranteed confidence sequence or simultaneous error bound.

Cap candidate-row evaluations at 1000 cumulatively across both attempts,
including finite differences, re-scoring and acceptance. Charge all cache
preparation, discarded attempts and fallback costs. Never add a hidden
full production sieve as a free fallback. Record which coarse winner is
returned; the external judge scores the actual final choice.

Compare the resulting decisions with the independent judge, especially
false acceptance of worse points and missed tail contributions. Use E2's
screening gates and compare against the best fixed rule as well as S0.
Construct and select this rule using development states only; the previously
inspected seed-1 trajectories cannot choose fixed versus adaptive. Any
additional seed-1 measurements are labelled staged evaluation, and E5
provides the fresh out-of-trajectory comparison.
The adaptive arm replaces the fixed search finalist only if it earns its
complexity; it does not create an additional unbounded family of E5 arms.

### E5: bounded paired inference comparison

**Executor:** Sol prepares/runs; Astra assesses inference evidence.

#### Approved exploratory S0/S2 amendment (2026-09-17)

The user authorized preparation and execution of the two-seed pilot after
agreeing to ten paired seeds per configuration in total. This amendment
admits the frozen S2 as an exploratory treatment despite E3's mixed outcome;
it does not classify S2 as a passed replacement or reopen E4. Historical
records stating E5 was unapproved retain the status at their creation.

Use the six E0 configurations and exactly two arms: unmodified production
S0 and the frozen S2 in `search_holdout_selection.json` (raw SHA256
`98c5d773bd83c6b1954e7d5d5f79b83a47d196c6916925f335897350d66aab40`).
The full design has seeds 2000-2009: 120 fits, comprising 24 pilot fits
(2000-2001) and 96 subsequent fits (2002-2009). The current executable
allocation is only the pilot. Retain its results in the full comparison.
After the pilot, complete verification and independent review, then report
the findings and measured cost estimate to the user. Launching seeds
2002-2009 requires a subsequent explicit user decision.
Continuation depends on verified execution and affordable measured cost,
not favorable early scientific results. No treatment tuning or replacement
of adverse seeds is permitted. Any implementation correction requires a
declared successor allocation and explicit handling of invalidated fits.

The twelve recent production capture fits completed full optimizations in
33.55 minutes. Equal-cost extrapolation gives 67.1 minutes for 24 fits and
335.5 minutes for 120. Reserve 90-120 minutes for pilot computation and
7-10 hours for the full design, subject to measured pilot costs. These are
planning allowances, not timing claims. Preparation and verification are
additional. Resume between completed fits; an interrupted fit may need to
restart and must not be mistaken for completed evidence.
The supervisor has a three-hour operational ceiling per pilot invocation
and a twenty-minute per-fit timeout. It stops launching when less than one
full per-fit allowance remains, preserving completed fits for resumption.

For ten complete pairs, use nine degrees of freedom in the planned runtime
and target-call log-ratio intervals. Report all ten paired accuracy differences
and transition outcomes. A worse median with at least seven of ten pairs
worse on the same target/metric triggers scientific review, together with
every new failure or usability loss. The two-seed pilot provides descriptive
results and operational checks only; it cannot establish quality equivalence.
This amendment supersedes the six-seed/108-fit maximum and the E5
eligibility and authorization statements in the original design below.

Pilot preparation and execution checklist:

- [x] Add a scoped S2 selection adapter and frozen-state parity checks.
  Reuse the existing shortlist scoring and refinement implementation.
  Preserve the sampling loop, target logger, repeats, cache consumption,
  GP/VP updates and default random stream. Use a private module callback
  only if process-local instrumentation cannot preserve those semantics.
- [x] Add a manifest-bound, sequential two-arm runner with atomic terminal
  records, per-fit subprocesses, immutable payload hashes, failure records,
  interrupted-fit rejection, bounded scheduling and resume validation.
- [x] Record initial-design equality, per-selection RNG state digests,
  accurate-rule seeds, search diagnostics, complete golden traces and
  paired final metrics. Define acquisition timing separately from full
  active-sampling time, which includes GP and VP updates.
- [x] Independently review implementation and scientific allocation; run
  focused tests, the package suite and exact default-path oracle checks.
- [x] Freeze source/configuration/data/environment identities and reviewed
  24-fit pilot manifest. Alternate arm order across seed blocks and stop
  after the pilot; seeds 2002-2009 remain outside its executable allocation.
- [x] Execute the pilot, summarize all outcomes and observed costs, update
  the result report and local inventory, and review before further allocation.

The [pilot report](../results/2026-09-16-noisy-acquisition-integration-search.md#pilot-outcomes)
records all 24 valid fits, the exact excluded replay, the independent review
and the measured continuation cost. All twelve initial-design pairs match,
and all 5,365 selection records verify. The operational continuation criteria
were satisfied; the continuation below was launched on the user's decision
of 2026-09-17 and is complete.

Continuation preparation checklist (2026-09-17):

- [x] Add a continuation manifest kind that makes exactly the 96 remaining
  fits executable. It binds the pilot manifest, its campaign and its four
  immutable records by hash, and admits only the runner's own source hash
  to differ from the pilot identity; every other source, data archive and
  environment entry must be equal, re-derived by each worker at start.
- [x] Add sequential, resumable batches with an explicit wall-time limit and
  optional fit cap. Validated success terminals are reused, failed fits are
  never rerun, and a failure ends the batch unless told to continue.
- [x] Add the combined ten-seed summary: pilot pairs read from the pilot
  campaign, nine degrees of freedom for ten complete pairs, the seven-of-ten
  accuracy trigger, transition tables, reference-envelope exceedances and a
  consistency check against the immutable pilot summary.
- [x] Focused checks (26 tests), exact default-path oracles, byte-identical
  regeneration of the pilot summary, independent static review and its
  should-fix corrections; freeze the reviewed manifest without a launch
  clearance and record the preparation.
- [x] Launch seeds 2002-2009 in bounded batches: the user's decision of
  2026-09-17 is recorded in `launch_clearance.json` before the first batch.
  Two batches (91 and 5 fits) completed all 96 cells with no failure. The
  combined summary, closure record, review of every trigger and independent
  review are complete; the E6 assessment records the outcome.

Implementation contract for the scoped policy:

- Keep S2's 1024-candidate coarse sieve, ordinary MC100 scoring, top-eight
  shortlist without a diversity constraint, independent MC1600 re-scoring,
  and one bounded L-BFGS-B start. Reuse the frozen refinement code and every
  tolerance and fallback. The user explicitly retained this configuration
  after discussing shortlist clustering.
- Install the policy only inside a process-local context. The default
  callback is absent. The callback receives the current GP, VP, logger,
  options, optimization state and full scored candidate set on every
  selection, including selections after in-loop GP/VP updates. It returns
  the selected coordinate, starting-cache index and exact-repeat flag.
  Production target evaluation, cache removal and update bookkeeping remain
  in the existing sampling loop. Restore all installed hooks in `finally`.
- Candidate generation and the production coarse rule retain the live VP
  random stream. Derive each S2 accurate-rule seed from a separate stable
  domain, configuration, run seed and selection ordinal. No instrumentation
  or S0 policy setup may consume a random draw. Record RNG state digests
  around selection and the independent accurate-rule seeds. Do not promise
  identical post-initialization streams between arms.
- Measure search wall time from candidate generation through point selection,
  ending before the logger evaluates or records the point. Apply the same
  boundaries to both arms; exclude target evaluation and intervening GP/VP
  updates. Report this alongside total optimizer and target-evaluation time,
  with the existing active-sampling stage timers available in the traces.
- Validate default-path exactness with the stored oracles, adapter S0 parity
  on frozen states and cleanup checks. Validate S2 against the frozen-state
  selector with matched search and accurate-rule streams, plus repeat and
  starting-cache cases. Check each complete pilot pair's initial design and
  initial noisy observations for exact equality before interpreting it.

One operational validation repeat is predeclared before opening pilot
results: repeat Rosenbrock D2, noise 1, seed 2000, S2 in a separate
`validation_replay` output directory under the same frozen identity.
Compare every stored non-timer trace array, semantic final field and
per-selection record (excluding timing), including the initial design.
The first pilot fit remains the sole scientific and timing observation;
the repeat contributes no extra pair, seed or quality evidence. Charge
its execution to validation overhead. Any mismatch stops operational
clearance and is investigated without replacing the original result.

Freeze no more than two candidate pipelines: an integration-only treatment
with production search and a selected search treatment (fixed or adaptive).
The comparator is the frozen current production code. If only one treatment
passes, use two arms; if none passes, end without changing the package.

Use the same six configurations as E0 and six fresh seeds 2000-2005:
at most **108 fits**, including all three arms. Seeds 2000-2001 form a
36-fit maximum operational pilot; run the remaining seeds only if the
pipeline is reproducible, finite and free of unexplained failures. A
scientific loss is evidence, not a reason to replace a seed. Do not tune
the frozen method after seeing these results. A revised treatment requires
a new declared allocation; do not merge its results with the old arm.

Use existing per-target budgets, initial design, noise generation, stopping
rules, GP fitting and final boost. Pair by configuration and seed, not by
assuming identical trajectories or identical noise at different locations.
Record acquisition RNG consumption; altered search can shift subsequent
shared-stream draws. The unmodified arm must retain its exact stream;
experimental arms must remain reproducible and preserve the initial design.
No special RNG repartitioning of the baseline is allowed to improve pairing.

The developer runner needs an explicit scoped search-policy seam, since
`active_sample.py` currently recognizes CMA-ES/Nelder-Mead/none only.
Prefer process-local injection confined to that runner; if a small private
package hook is necessary, isolate it and verify the default path exactly.
Do not monkeypatch GP training or duplicate the active-sampling loop. Restore
all temporary hooks in `finally` and test exception cleanup. An integration-
only arm uses its new integration rule throughout the existing search.

Collect complete traces and sidecars through `golden_trace.py` conventions.
Use the validated completion-manifest pattern in `population_run.py` for
resume, not the presence of a trace file alone. Reuse the paired summaries
from `analyze_population_run.py` where compatible, binding their metric and
failure definitions in the new manifest.
Assess absolute evidence error, gsKL, MMTV, convergence, usable fraction,
target calls, total optimizer time and acquisition time. Keep the existing
usable criteria (evidence error <1, gsKL <1, MMTV <0.2), and report failures
and nonfinite metrics explicitly. Compare with the promoted reference's
per-target envelopes as a supplementary regression diagnostic; fresh paired
baseline runs, not historical laptop timing, are the primary comparator.
The historical noisy multisensory-subject-1 usable fraction is only 0.17;
retain that difficult configuration and report continuous errors alongside
the binary usable flag. Its low success rate is not a reason to filter runs.

Sequentially alternate arm order within seed blocks on a quiet machine;
record interruptions and target evaluation time. Do not infer acquisition
speedup from total runtime alone. Predeclare these per-target summaries:

- All six paired observations for every metric, including failures.
- For positive elapsed times and target-call counts, the mean paired log
  ratio and its two-sided 95% Student-t interval (five degrees of freedom
  for six complete pairs), exponentiated for display. If a pair fails,
  retain its failure and report the interval only for available complete
  pairs with the corresponding degrees of freedom and explicit counts.
  Such a conditional runtime interval cannot establish overall efficiency.
- For continuous accuracy errors, all paired differences and their median
  and range. These small-sample descriptive summaries do not establish
  equivalence; no post hoc choice among significance tests.
- For convergence and usability, paired gain/loss counts and the full
  transition table. Show nonfinite/error cases separately rather than
  assigning arbitrary numerical errors.

Pooled tables weight configurations equally and remain descriptive;
heterogeneous targets, iterations and quadrature repetitions are not
additional independent inference runs. A worse median accuracy error with
at least four of six pairs worse on the same target/metric triggers review,
as does any new failure or usable-to-unusable transition. These triggers
are deliberately descriptive investigation gates, not calibrated tests.

**Decision gate:** no automatic adoption. Investigate every new failure,
usability loss or reference-envelope exceedance. Recommend continuation only
if the acquisition speed/quality benefit survives inference, with no coherent
target-specific accuracy loss and no unexplained failure mechanism. Six
seeds per target cannot establish a tight noninferiority bound; if that is
the remaining uncertainty, report it and propose a targeted extension.
Do not interpret a nonsignificant test as equivalence. The final release
campaign and any default change require a separate decision.

### Resource bounds, artifacts and verification

The proposed allocation is at most 12 state-capture fits plus 108 inference
fits; frozen-state calculations make no target calls. The PI authorized
an **11-hour first window** beginning 2026-09-16 18:34:41 UTC and ending
2026-09-17 05:34:41 UTC for E0-E4, including implementation, capture,
experiments, verification and the progress report. Reserve the final hour
for checks and consolidation; do not launch work expected to overrun it.
E5 may receive a separate 11-12-hour window tomorrow if needed; that second
window is not yet activated. The first window does not launch E5 fits.
These are ceilings, not expected durations or a reason to spend the
remaining allowance. Record a pilot cost estimate before each matrix and
stop cleanly with partial results if a ceiling would be exceeded. Never
extend by silently adding seeds, methods or nodes beyond the stated caps.

Store versioned manifests and compact per-cell results under
`dev/experiments/noisy-acquisition-efficiency/integration-search/`.
Large captures, worker logs and traces go under ignored
`dev/scripts/runs/noisy_acq_efficiency_20260916/`; list their local locations
in `LOCAL.md` only when they exist. The manifest identifies source revisions,
normalized source hashes, raw artifact hashes, target/data/truth versions,
state split, all node/candidate/acceptance/judge seeds, numerical settings,
failures and timing method. Summaries must regenerate without rerunning fits.

The developer tools separate capture, integration, search and judging.
`noisy_acq_experiment.py` captures and inventories states;
`noisy_acq_integration.py` runs fixed-budget selections;
`noisy_acq_search_experiment.py` runs search selections;
`noisy_acq_campaign.py` advances independent judging budgets; and
`noisy_acq_timing.py` measures integration cost. Conditional adaptation
and E5 inference are implemented only when their entry conditions apply.
Run commands with the frozen dependency path and thread environment
recorded in the experiment manifest. For example:

```console
python -m pytest dev/scripts/test_noisy_acq_experiment.py -q
python dev/scripts/noisy_acq_experiment.py inventory --manifest <manifest> --out <out>
python dev/scripts/noisy_acq_integration.py run --manifest <integration-manifest> --out <selections>
python dev/scripts/noisy_acq_campaign.py integration --manifest <integration-manifest> --results <selections> --out <judging>
python dev/scripts/noisy_acq_timing.py run --manifest <integration-manifest> --out <timing>
```

Run relevant acquisition/importance-sampling tests whenever their helpers
or hooks change. Before E5, run the package suite and exact oracle comparison
for the default path on its generating platform. Follow repository gates
for any default-path numerical change; do not regenerate fixtures to hide
an experimental regression. Targeted synthetic analytic and finite-difference
checks belong in the developer tests, with no new full `VBMC.optimize()`
runs added to the shipped unit suite. Experimental fits are the allocation
above. Apply repository formatting hooks to changed files.

### E6: interpretation, records and independent review

**Executor:** Astra orchestrator, independent Sol reviewers.

Consolidate results into one report under
`dev/results/2026-09-16-noisy-acquisition-integration-search.md`, created
when measurements exist. It owns scientific comparisons and limitations,
while this plan owns execution status and decisions; machine-readable
artifacts own reproducible values. Do not create separate session journals.
Update the existing acquisition note with conclusions and links, and update
`dev/TODO.md`, `dev/README.md` and the modernization roadmap's scope/status.
Public API documentation changes belong to a later production integration.

Use `$doublecheck` on protocol, numerical implementation and final evidence.
Independent reviewers check objective preservation, weights, holdout use,
failure accounting, cost fairness and whether the conclusion follows from
the measured uncertainty. The compute-owning agent runs tests and reproduces
summaries; reviewers remain static. Preserve a null or negative result as a
valid completed outcome. Return a recommendation to retain the current
method, pursue one specified treatment, or collect narrowly identified
missing evidence. No merge, publication or default switch is part of this
experimental plan.

#### E6 assessment (2026-09-18)

The experiment asked whether numerical integration and search can choose
equally good or better VIQR points at lower cost with acceptable inference.
The [report](../results/2026-09-16-noisy-acquisition-integration-search.md)
holds the measurements; this is the reading.

- **Integration (E2).** Better integration rules choose better points:
  stratified RQMC at 512 or 2048 nodes was judged better than the
  production 100-node MC estimate in about half of the panel comparisons
  and worse in almost none, and even RQMC at 128 nodes was better in 45 and
  worse in 10 of 96. No rule passed the 10 percent cost gate as a drop-in
  replacement for the 8192-candidate sieve, because selection cost scales
  with the node count. The predeclared grid had no matched-cost RQMC
  setting near 100 nodes, so that direct comparison does not exist.
- **Search (E3).** S2 (1024-candidate sieve, eight re-scored on MC1600, one
  L-BFGS-B refinement) selects at 0.43 of production time and improved the
  judged choice on development states, but the holdout was mixed: 54
  beneficial, 17 tied, 13 harmful and 12 unresolved of 96, with material
  raw losses concentrated where the refinement guard fired or the shortlist
  winner was already worse than S0's choice.
- **Inference (E5, ten paired seeds per configuration).** S2 reduces fit
  time to 0.57 and search time to 0.37 of S0's at unchanged target-call
  counts (0.97); the fit-time saving is the search saving, so it shrinks
  toward nothing on likelihoods expensive enough for target evaluation to
  dominate the run. Usable fits rise from 41 to 49 of 60, nine gains (one
  marginal) against one loss; the gains concentrate on Student-t D8 (3 to 9
  usable, all three metrics better in eight of ten pairs). All 60 S0 fits
  converge against 55 of 60 for S2, all five at the evaluation ceiling. The
  usability loss and four of the five convergence losses are on high-noise
  Rosenbrock D2, where gsKL increases 3.5 to 103 times within pairs on six
  of ten seeds. Logistic regression shows worse gsKL and MMTV in eight of
  ten pairs with a better evidence error. The four seven-of-ten accuracy
  triggers are at the count a symmetric null predicts (3.1 of 18), and no
  accuracy comparison reaches a two-sided sign-test probability below 0.11.
  Multisensory is unusable in both arms.

**Recommendation: retain the production search as the default, do not
adopt S2, and collect the narrowly identified missing evidence below.**
The decision gate recommends continuation only if the speed benefit
survives inference with no coherent target-specific accuracy loss and no
unexplained mechanism. The speed benefit survives. The accuracy
precondition is not met: ten paired seeds neither establish nor exclude a
target-specific loss, and what weighs against S2 is not the trigger count
but the binary asymmetries (five convergence losses against none, all at
the evaluation ceiling; one usability loss with gsKL 103 times its pair)
and the unexplained opposite effects on Student-t and logistic regression.
The high-noise Rosenbrock accuracy result also reverses sign between the
two pilot seeds and the eight continuation seeds, which shows how unstable
a target-level verdict is at this size. Two observations shape the
follow-ups. First, the frozen-state acquisition judge predicted the
inference outcomes in neither direction: S2's worst holdout trajectory
(low-noise Rosenbrock, 8 harmful of 16) is clean in inference with the
largest speedup, and its near-clean holdout trajectory (high-noise
Rosenbrock, 1 harmful) is the one that degraded. Frozen-state work can rank
variants and cost them; the inference comparison remains the arbiter.
Second, the S2 design changed sieve size, coarse-score noise handling and
refinement at once, and E3's loss diagnostic could not separate
insufficient sieve coverage from coarse mis-ranking.

Limitations of the evidence: ten seeds per configuration are descriptive
and establish neither noninferiority nor outcome frequencies; timings are
laptop measurements on cheap likelihoods, and the continuation ran on a
progressively slowing machine (within-pair bias about 1 to 3 percent), so
only paired ratios are interpretable; the promoted reference envelopes were
exceeded on gsKL by two fits of each arm and do not register the high-noise
Rosenbrock usability loss, whose gsKL of 1.56 lies under that target's
envelope of 2.02. The whole-VBMC runtime measurement of the guarded-sinh
and kernel-reuse changes remains deferred as recorded in the pickup point.

#### Proposed follow-ups (2026-09-18, not approved)

These extend the completed experiment with narrowly identified missing
evidence. None is authorized; each needs a user decision, a frozen
manifest and a bounded allocation before any computation. Because they
are designed after seeing the E5 results, any variant they produce is a new
treatment under a new declared allocation whose results are never merged
with the completed S2 arm; the amendment forbids tuning the frozen method
itself. F1 to F3 are frozen-state or trace analyses with no target calls
and no GP fits, and they can run in any order; F4 is conditional on them.

- **F1. Isolate the sieve size.** Run the frozen S2 selection with the
  coarse sieve at 2048, 4096 and 8192 candidates, holding the eight-point
  shortlist, MC1600 re-scoring, single refinement and every tolerance
  fixed, on the 24 captured states with eight repetitions, paired with S0
  and judged with the existing independent RQMC ladder and ordinary-MC
  checks. Report, per state and sieve size, the beneficial/harmful counts,
  material raw losses and the paired timing ratio. The question is where
  the holdout losses disappear and what each step costs; the E3 timing
  suggests a 2048 sieve keeps about half of S2's saving and 4096 about a
  quarter. Estimated cost: selections took about two seconds per cell in
  E3, so three variants on 24 states and eight repetitions are under an
  hour of selections plus one to two hours of judging and timing.
- **F2. Matched-cost RQMC.** Implement a production-grade
  component-stratified scrambled-Sobol rule that reuses the kernel path,
  at 96 and 128 nodes, and evaluate it two ways on frozen states: as a
  drop-in for the production 8192-candidate sieve, timed against MC100 on
  the full sieve and judged; and as the coarse scorer inside S2 and its F1
  variants, to test whether low-discrepancy nodes at the same budget remove
  the coarse mis-ranking that E3 could not separate from coverage loss.
  E2's panel result for RQMC128 (45 beneficial, 10 harmful of 96 at 1.49
  times panel cost) is the motivation; the full-sieve cost ratio and the
  benefit at matched cost are the missing numbers. Estimated cost: one to
  two hours of frozen-state compute after implementation and focused tests.
- **F3. Diagnose the high-noise Rosenbrock trajectories.** Read-only
  analysis of the stored golden traces and selection records of the ten
  S0/S2 pairs on `rosenbrock_D2_noise3`, starting with seed 2008: where the
  trajectories diverge, whether S2's selections cluster near the current
  mode or in the tails, how often the refinement guard and row-budget
  fallbacks fired, how the GP hyperparameters and the variational
  components evolve, and why four S2 runs failed the stability criterion
  within 200 evaluations. The aim is a mechanism for the degradation that
  F1 and F2 can then be checked against. About an hour of analysis; no new
  computation on the targets.
- **F4 (conditional). Inference comparison of a repaired variant.** Only if
  F1 or F2 identifies a variant that removes the Rosenbrock and logistic
  regression losses on frozen states at a worthwhile cost, allocate a new
  E5-style paired inference comparison under a new manifest, with the same
  six configurations and ten seeds, treating the present S2 results as a
  completed arm rather than merging them. The cost is that of the completed
  E5 design, about six worker hours on this laptop.

Executor roles follow the experiment's convention: Astra decides scope and
reads evidence, Sol implements and runs, separate Sol agents review, and at
most one agent runs compute at a time.

### Decisions and unresolved empirical questions

- **Integration before search, fixed rules before adaptation.** This order
  separates estimation error from optimizer behavior. A combined adaptive
  rewrite could show a gain without explaining which part earned it.
- **Stratified MC accompanies RQMC.** It isolates the benefit of component
  coverage. Plain MC versus RQMC alone would conflate two changes.
- **BQ is excluded from this experiment (PI scope review, 2026-09-16).**
  Analytic kernel means under the VP make integration convenient but do
  not determine a suitable kernel for the candidate-dependent VIQR
  integrands. The original proposed kernel/length-scale grid was a
  heuristic. Learning or deriving an auxiliary integration kernel would
  require a separate modelling and validation effort. MC/RQMC comparisons
  address the present efficiency question without that additional model.
- **Use the complete acquisition score.** Unregularized reduction alone
  can change the selected point under the existing variance penalty.
- **Small paired allocation with explicit uncertainty.** A reference-sized
  campaign is unnecessary for screening. This allocation cannot certify
  rare-failure rates and does not automatically select a release default.
- **Developer prototypes precede a public interface.** An experimental win
  may suggest a simpler implementation than exposing every tested control.

The PI accepted the benefit criteria, six-seed paired allocation and
screening thresholds on 2026-09-16. BQ was removed following that scope
review. The resource ceilings remain limits rather than runtime estimates.
Which integration rule wins, whether refinement pays, whether adaptation is
needed, and whether inference benefits are empirical questions addressed by
the phases rather than assumptions. Plan approval authorizes the bounded
experiment; it does not authorize changing GP policy or public defaults.

### Plan verification (2026-09-16)

Independent Sol review through `$doublecheck` checked the complete-score
contract, integration rules, judging, holdout separation, resource allocation
and inference analysis against the source and historical evidence. Review
corrections made practical ties and interval construction explicit, fixed
adaptive node accounting and its entry gate, and specified the paired
inference summaries. The initial review also covered the BQ design before
its removal from the experimental scope.
The final review reported no remaining must-fix or should-fix findings.

Read-only feasibility exploration identified the reusable state, timing and
completion-manifest tools and the limited historical state coverage. Local
checks confirmed relative links, preservation of the historical evidence
and execution records below, repository formatting and `git diff --check`.
No experimental implementation, numerical probe, test suite or campaign ran
during planning. The subsequent first-window authorization is recorded
in the resource bounds above.

The BQ removal and first-window timing amendment were checked directly
against the PI's scope and scheduling instructions. This narrow revision
used the doublecheck skill's faithful-record self-review exception; the
scientific design had already received independent review.

### Execution record

- 2026-09-16 18:34:41 UTC: first-window clock started. The deadline is
  2026-09-17 05:34:41 UTC; verification/reporting reserve begins at 04:34 UTC.
- E0 inventory and E1 developer harness implementation began in parallel.
  The root owns all tests and numerical runs; Sol workers own disjoint
  numerical-module and capture/runner files. The production package is
  frozen at `9cc6882` for the experiment baseline.
- Environment preflight resolved the frozen gpyreg source to commit
  `9e70e6b`; installed distribution metadata still says 1.2.0, so source
  hashes and the import path identify the dependency. The three BLAS thread
  environment variables are fixed at one. Five data/truth archives were
  hashed in `integration-search/environment_preflight.json`.
- All seven historical captures reproduce their stored full-sieve VIQR
  scores exactly without changing state or RNG. They support harness
  checks; current early/late trajectory coverage still requires fresh
  capture. Single-call preflight timings are feasibility measurements only.
- The first eight capture-protocol checks pass. The initial test command
  encountered permissions on pytest's shared temporary directory; using
  a new workspace-local temporary directory resolved that environment
  issue. Numerical and snapshot-roundtrip checks remain in progress.
- The combined developer checks reached 44 passes before first capture
  launch. These cover explicit weighted rules, live-factor replay, exact
  restored RNG, failed-trajectory accounting, source isolation, full-score
  parity and target-free search instrumentation. Independent review found
  and verified fixes to capture-state digests and failure persistence;
  numerical stress coverage and the E2 driver remain in progress.
- Search instrumentation lives in `dev/scripts/noisy_acq_search.py`; it
  invokes the production active-sampling controller and stops at selection
  before any target call. E2 orchestration lives in the separate
  `dev/scripts/noisy_acq_integration.py`, allowing the capture source to stay
  locked during the baseline trajectories.
- Capture harness commit `3fe616c` passed its commit hooks. The reviewed
  capture manifest is stored in the tracked `integration-search/` directory
  and the raw `noisy_acq_efficiency_20260916/` directory. The first allocated
  case (`rosenbrock_D2_noise1`, seed 0) completed in 57.2 seconds: early
  state at 20 charged evaluations, late fallback at 114, natural termination
  at 115. Both full 8192-point scores replay exactly. The remaining eleven
  cases were launched sequentially after 49 focused checks passed.
- 2026-09-16 19:38 UTC: all twelve allocated capture runs completed without
  execution errors. All twenty-four early/late states are present and pass
  exact public full-sieve replay. Integration and timing tools passed
  independent static review after fixes to pooled-score regret, grouped
  denominators and artifact/source identity checks. The expanded numerical
  and protocol checks are running before E2 launch.
- Pre-launch verification passed: 77 focused developer checks and 234
  existing acquisition/oracle checks, with 15 skips.
  The developer tests include exact S0 agreement with direct production
  CMA-ES selection. Warnings came from deliberate invalid numerical inputs
  and existing VP-density gradient cases. Production source remains
  identical to the numerical baseline. E2 begins with a one-cell smoke
  before the full development allocation.
- Harness commit `8a13215` freezes integration, quadrature, search and
  timing tools. The paired E2 smoke succeeded; its separate judge evaluated
  34 candidate coordinates under eight independent replicates without
  invalid scores. The reviewed development panel manifest allocates 864
  treatment selections and 96 fresh production baselines across twelve
  states. The full sequential panel run began at 19:43 UTC. Holdout
  treatment evaluation remains locked. Fourteen additional search-runner
  and ordinary-MC crosscheck protocol tests passed; their independent
  static review continues alongside the panel run.
- E3 campaign tools are being verified while E2 runs. They include a
  separate development comparison of the strongest search arm with
  MC1600, before finalist selection; the fixed captured reference
  shortlist; paired selection and judging streams; retention of valid
  cells alongside failed/missing ones; and separate memory measurements.
  No E3 numerical allocation has been launched before selection of its
  accurate integration rule.
- Independent static review of E3 and the judging controllers is complete.
  The MC1600 control includes a direct comparison with the same frozen
  search configuration; primary, component and direct-control comparisons
  retain separate denominators. Judging ladders bind all terminal selection
  artifacts across budgets. The ordinary-MC diagnostics retain the primary
  verdict and report agreement, disagreement and unresolved checks separately.
  Runtime verification of these additional tools follows completion of the
  E2 panel, preserving the single-compute-process rule.
- 2026-09-17 morning recovery: the usage limit stopped work after all panel
  cells completed and during the expanded tool tests. Fixture corrections
  aligned exact-byte zero rows, synthetic state identifiers and required
  provenance fields with the tested contracts. All 133 developer checks
  pass, with two expected warnings from deliberately invalid inputs.
  Recovery revalidated all 960 panel terminal records and their bound
  payload hashes. Production source is unchanged from `9cc6882`.
  Full judging, timing, E3 numerical runs and holdout treatment evaluation
  remain unstarted; the earlier one-cell judge smoke is separate evidence.
  Independent recovery review cleared the fixture repairs, completion
  record and bounded-pilot restart instructions with no open findings.
- Recovery checkpoint `07b00c6` preserves the reviewed tools and records.
  The branch incorporates `dev-next` at `953c88b`, including teaching
  material and release-audit documentation. That integration changes no
  numerical experiment source; the frozen source identity and all 960 panel
  terminal records remain valid after the merge.

## Guarded-sinh execution checklist (completed)

- [x] Freeze before-change source and oracle outputs; implement guarded sum
  and focused tests.
- [x] Run numerical gates and classify changed outputs.
- [x] Capture matched states during the 18-run replay and assess results.
- [x] Measure complete acquisition calls against the frozen implementation.
- [x] Complete full tests, CI and independent review; record outcome.

## Goal and scope of the completed computational work

Reduce the cost of evaluating existing noisy acquisitions while preserving
their criterion, importance samples, random draws and search settings.
The first implementation targets standard VIQR's sum over importance points.
Sieve size, importance-sample count, optimizer, frequent GP retraining and
default acquisition remain unchanged. New acquisitions and the planned
experimental-option removal are separate work.

This plan owns implementation decisions and validation for the efficiency
work. The [noisy-acquisition note](../2026-09-08-noisy-acquisitions.md) and
its linked reports retain the historical investigations. The
[TODO](../TODO.md) owns release scope and workstream status.

## Evidence and candidate changes

The historical measurements identify large-sieve acquisition evaluation and
frequent GP training as the main costs. Their proposed search changes need
separate validation: optimizing a small importance set more aggressively
can overfit its Monte Carlo error. They do not establish a new default.

Before this optimization, standard VIQR performed a log-sinh followed by
log-sum-exp over each candidate's importance points in
`pyvbmc/acquisition_functions/acq_fcn_viqr.py`. For finite nonnegative
`a = u * s_pred`, this computes `log(2 * sum(sinh(a)))`. Evaluating that
sum directly avoids two elementwise transcendental sweeps. An overflow
guard is required; the log-space calculation supplies the fallback for
large values.
The direct expression also retains extremely small positive contributions
that the old `1 - exp(-2a)` calculation can round to zero.

The bounded [probe](../scripts/probe_viqr_sum.py) uses the committed
`rosenbrock_D2_noise1_viqr` oracle state (D=2, N=25, eight GP samples,
8192 candidates, 100 importance points). In the initial probe, direct
evaluation took 91 ms versus 338 ms for the existing sum, with maximum
absolute difference 1.78e-15. A scaled exp/expm1 identity took 266 ms.
These are timings of the sum alone across the eight GP samples, on one
early state; they do not estimate complete-run speedup. Exact timings,
versions, hashes and limitations are in
[the probe output](../experiments/noisy-acquisition-efficiency/viqr_sum_probe.json).

Standard VIQR reuses prediction kernels through gpyreg 1.2.0, with a
128 MiB retained-payload cap and ordinary-evaluation fallback. The
[production report](../results/2026-09-13-viqr-kernel-production.md) records
the measured 1.205x late-sieve speedup and numerical gates. IMIQR kernel
reuse remains separate from this completed optimization.

Additional opportunities remain for subsequent steps:

- Optional IMIQR cleanup: it recomputes triangular solves in every acquisition call although
  `active_importance_sampling.py` already prepares `C_tmp`. Reusing it
  requires checking both GP factor representations and the placement of
  the noise scaling, which can change rounding. Keep IMIQR's weighted
  criterion and MCMC stream intact.
- Adaptive sieve/search remains a main efficiency experiment. Compare a
  smaller coarse sieve, more accurate re-scoring, and L-BFGS-B from one or
  several spatially separated candidates. Investigate sieve sizing and
  uncertainty-based integration budgets/refinement gates; the initial
  experimental sieve size is about 1024, not an established new default.
  Judge candidates on independent integration points to detect Monte Carlo
  overfitting, then validate inference end to end. The
  [search analysis](../results/2026-09-09-acquisition-search-analysis.md)
  records the mixed evidence on additional starts. Kernel reuse and
  quadrature are independent opportunities, not prerequisites for this
  experiment.
- More efficient integration of the existing VIQR criterion, including
  shared-weight Bayesian quadrature (PI, 2026-09-13). For each fixed VP and
  quadrature kernel, prepare nodes and weights once and reuse them across
  candidate locations. Gaussian-kernel means under the Gaussian-mixture VP
  are analytic, but the nonlinear VIQR integrand itself remains numerical.
  Compare with ordinary VP sampling and mixture-stratified randomized
  quasi-Monte Carlo on matched saved states, measuring total preparation
  and evaluation costs and candidate rankings under a larger independent
  integration set. Investigate quadrature-kernel choice, negative weights,
  positivity and difficult integrands before choosing an implementation.
  This is a possible integration-efficiency experiment within this work,
  preserving the VIQR criterion; it is not part of the guarded-sinh change.
  See the [VIQR definition](../../papers/acerbi2020variational_main.md#33-integrated-median--variational-interquantile-range-imiqr--viqr)
  and [Gaussian integration identities](../../papers/acerbi2019exploration_appendix.md#appendix-a-expected-log-joint-via-bayesian-quadrature).

## Phase 1: implement the VIQR sum optimization

Executor: Sol implementation agent, with Astra orchestrating design and
integration. Only the main thread runs heavy checks or benchmarks; reviewers
use static analysis.

1. Start a feature branch from the settled `dev-next` source. Read
   `dev/2026-09-02-modernization-discussion.md` and the acquisition note.
   Freeze the before-change revision and dump current oracle outputs using
   `python dev/scripts/make_oracle_fixtures.py --dump-outputs <before-dir>`.
   Set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` to 1
   for numerical checks on the fixture-generating machine.
2. Modify only standard VIQR's sum, preserving the hyperparameter-sample
   aggregation and normalization. Use a direct-sinh path on safe finite
   nonnegative arguments, with the existing log-space calculation as the
   fallback. Derive the overflow bound from float64 and the number of
   importance points, with margin for the sum. Keep the fallback's handling
   of unsupported/nonfinite arguments and old serialized VIQR instances.
   Do not apply the unweighted identity to IMIQR or reduction variants.
   Leave `is_log_added` and the importance proposal helpers unchanged:
   their values feed IMIQR's MCMC target and can alter its sampling stream.
3. Add focused tests in `pyvbmc/testing/acquisition_functions/` for small
   and large arguments, zero rows, mixed zero/positive entries, overflow
   guard boundaries, alternative quantiles, and one/multiple GP samples.
   Compare with independent analytic or high-precision values where the
   old expression loses precision; use the existing full-covariance
   reference in `_look_ahead.py` for acquisition-level checks.
4. Check public-call scalar/batch agreement, dtype, variance penalties and
   hard bounds; require unchanged RNG state during evaluation and unchanged
   importance samples/weights. Test reductions and IMIQR as unaffected
   paths. Avoid adding full `VBMC.optimize()` runs to the unit suite.

Acceptance: same mathematical criterion and all existing public behavior,
apart from documented floating-point rounding; no sampling or search-policy
change. If the fast path needs broader API/state changes, stop and revise
the design rather than expanding this implementation silently.

## Phase 2: numerical and performance checks

Executor: Astra main thread runs checks; Sol independently reviews source
and evidence. Model roles follow the repository's Astra/Fable and Sol/Opus
equivalences.

- Run `python -m pytest pyvbmc/testing/acquisition_functions
  pyvbmc/testing/vbmc/test_active_importance_sampling.py -q` and the oracle
  suite. Run the generator with `--check --exact --against <before-dir>`
  to classify every changed output, alongside the normal tolerance gate.
  Expect possible rounding in VIQR and movement in its stochastic search;
  investigate any other changes. Do not regenerate fixtures to make the
  optimization pass. Any sanctioned targeted rebaseline comes only after
  the acquisition value gates and assessment.
- Extend the probe to time complete public acquisition calls at candidate
  counts 1, a CMA-ES population, and 8192. Include setup costs separately.
  Use early and late noisy Rosenbrock, noisy Student D8 and noisy
  multisensory states where available, covering one/multiple GP samples.
  Inventory reconstructible saved states first: golden trace sidecars alone
  do not supply a GP/VP snapshot. Capture missing states during the approved
  replay allocation instead of launching additional exploratory runs.
  Add a developer-only capture wrapper around the VIQR sieve call in the
  replay runner, before acquisition evaluation. For seed 0 of the selected
  targets, capture the first sieve and the first sieve with either one GP
  sample or N at least 75% of the evaluation budget. Record an unreached
  late checkpoint as a coverage gap. Use `_state.snapshot_from_objects`
  and `save_snapshot`, adding the exact candidates through `_state.encode`;
  retain the current importance samples, weights and caches in `optim_state`.
  Snapshot copies must consume no random draws or mutate live inputs;
  assert RNG-state identity around capture. Serialize outside the measured
  acquisition interval. Record the live acquisition outputs for validation.
  Rebuild the same snapshots under frozen before/after checkouts using
  `build_state` and `prepare_gp_for_acq`, without redrawing importance points.
  Verify the after-code reconstruction against the captured candidate-run
  outputs. Then compare before and after on that same reconstructed state,
  classifying the expected rounding differences. Resolve any missing
  factor/state information before accepting timings.
- Compare before and after on identical states, candidates and importance
  samples in alternating timing order, single BLAS thread and one heavy
  process. Require a repeatable public-call gain beyond timing variation
  and no material small-batch slowdown. Record ranges, allocation overhead
  and fallback frequency; do not extrapolate the D2 sum-only ratio to runs.
- Approved bounded replay allocation: seeds 0-2 for
  `rosenbrock_D2_noise1`, `rosenbrock_D2_noise3`, `logreg_D5_noise3`,
  `student_D8_noise3`, `multisensory_s1_D6_noise1.3`, and
  `timing_D5_noise2.2` (18 candidate runs). Use
  `dev/scripts/golden_replay.py --configs <comma-separated-labels>
  --seeds 0-2 --out <candidate-dir>` against the promoted
  `reference_990_20260913`. Assess changed trajectories, initial designs,
  posterior/evidence errors, evaluations and completion/usable outcomes.
  These runs are a regression screen, not a powered equivalence study.
  Investigate flags before deciding whether a targeted extension is needed;
  a fresh 990-run campaign is not the default validation allocation.
  Acceptance requires no unresolved replay flags, identical initial designs,
  finite final metrics and population-envelope compliance for changed runs.
- Once the candidate is settled, run the required full suite and CI matrix
  for numerical changes, formatting hooks, and independent doublecheck.
  Existing occasional inference failures are not a new repair objective.

## Decisions and alternatives

- Start with the local VIQR sum: it has measured promise and a small
  implementation boundary. Kernel reuse could yield additional gains, but
  requires a gpyreg interface or duplicated prediction logic; it is a later
  design decision. Production monkey-patching is not an acceptable interface.
- Prefer a guarded direct sum with the current fallback. The scaled
  exp/expm1 alternative avoids overflow for finite nonnegative arguments,
  but was substantially slower in the probe. An unconditional direct sum
  would overflow on sufficiently uncertain GP states.
- Leave sample budgets and refinement unchanged in this step. A smaller
  sieve with better refinement is promising, but the historical experiments
  expose Monte Carlo overfitting and do not yet establish end-to-end gains.
- Treat IMIQR cache reuse as a separate subsequent change so its numerical
  and timing effects can be assessed independently.

## Documentation, review and rollback

Update this plan with accepted decisions and validation, link it from
`dev/TODO.md` and `dev/README.md`, and update the existing acquisition note
only when measurements warrant revised conclusions. No public option or
tutorial change is expected for the computational optimization.

Revert the isolated implementation commit to restore the previous sum;
preserve the before-change source and any accepted reference changes
together. The active 990-run baseline stays intact during investigation.

Independent Sol doublecheck found no must-fix mathematical or scope defect.
The plan incorporates its requested snapshot-capture procedure and explicit
replay acceptance criteria. The bounded probe, provenance hashes, relative
links and repository formatting hooks were checked during exploration.
Implementation and the 18-run allocation were subsequently approved by the
user. Execution status is tracked above.

## Validation record

The [validation artifact](../experiments/noisy-acquisition-efficiency/viqr_sinh_validation.json)
contains replay results, matched timing repeats, source hashes, coverage and
verification records. The [replay table](../experiments/noisy-acquisition-efficiency/viqr_sinh_replay.md)
shows each of the 18 comparisons.

- Before-change checkout: `dev/scripts/runs/viqr_sinh_20260913/before`,
  detached at `83692ac`; oracle dump in the adjacent `oracles_before/`.
- Acquisition and importance-sampling tests: 100 passed. Exact comparison
  against the dump changed only `acq_AcqFcnVIQR` on the noisy Rosenbrock
  fixture, by at most 8.88e-16. All other outputs, including the seeded
  active-sampling step and GP fits, matched exactly (10/11 fixture groups
  completely exact). The normal oracle suite passed: 143 passed, 15 skipped.
- The developer capture preflight rebuilt an acquisition output exactly
  after the live VP had been deliberately mutated, confirming snapshot
  isolation. The runner retains live GP factors and temporary data as well
  as the normal snapshot fields. Captured sets have per-file hashes;
  timing uses only checkpoints declared by the current status record.
- Independent Sol code review found no issues in the guarded calculation,
  its integration or focused tests. A complete-call preflight on the early
  noisy Rosenbrock state measured a 1.34x sieve speedup; single-point and
  six-point calls were 1-3% slower in that measurement. The broader timings
  below informed the final guard refinement. Full-suite execution initially
  encountered an inaccessible Windows temporary directory; a rerun used a dedicated
  workspace test directory and passed.
- Full suite passed on candidate `0b3c7b8`: 1254 passed, 39 skipped in
  399.50 seconds. The approved replay ran via
  `python -u dev/scripts/validate_viqr_sinh.py replay --out
  dev/scripts/runs/viqr_sinh_20260913/replay`; raw logs, traces and snapshots
  remain under the ignored run directory. The numerical source and runner
  were frozen at `0b3c7b8` throughout replay. Uncommitted status documentation
  accounts for the dirty-tree flag. The only intervening commit, `b2d7d72`,
  corrects a test mock and changes no numerical or replay source.
- Full-matrix run `34749673691` started after dispatch retries. Its three
  Python 3.10 cells exposed a pre-existing ambiguous mock target in
  `test_ns_gp_max_active_caps_the_in_loop_refits` (introduced in `5b857dde`):
  Python 3.10 resolves the dotted path through the exported `active_sample`
  function. Explicitly importing the module and using `patch.object`
  corrects the test (`b2d7d72`), with independent static review and formatting
  passed. Replacement full matrix `34750419101` passed all nine OS/Python
  cells; smoke `34750413117` also passed. The final full-suite run includes
  this corrected test.
- Static review of the capture runner led to v2 manifests that bind the
  runner, allocation, production/dependency sources, numerical-library
  versions and benchmark data. These were introduced after the campaign;
  its original v1 manifest remains intact.
- The 18-run replay completed in 49.0 minutes: 17 exact stored-loop/final
  matches and no flags. Every initial design matched, and all 18 runs
  terminated successfully with finite, consistent final outputs. The sole
  changed trajectory was noisy Rosenbrock (noise 3), seed 1: both versions
  used 180 evaluations; its evidence error, gsKL and MMTV remained inside
  the promoted population envelopes. This is a bounded regression screen.
  The historical archives omit the returned transformer's state, so exact
  identity applies to the recorded arrays and semantic final fields.
- Capture coverage is six early states and one late state (Rosenbrock,
  noise 3, N=150, six GP samples). The other five seed-0 runs converged
  before the late checkpoint. The existing `normal_D2_singlesample` oracle
  supplies a separate single-GP-sample timing check; it is not a noisy-run
  checkpoint. No extra target evaluations were used for timing.
- The first guard incurred single-point allocation overhead. Revision
  `6734817` uses scalar extrema for an entirely safe array and caches the
  logarithmic constants, retaining the same arithmetic and row fallback.
  All 11 oracle fixtures matched the first guard exactly, as did 162
  boundary/mixed/random helper arrays and every captured public output.
  Independent review confirmed this preserves the completed campaign's
  numerical behavior; a second campaign is unnecessary.
- Complete 8192-candidate calls on the seven noisy states improved by
  1.113-1.371x. The supplementary single-GP-sample sieve improved by 1.260x.
  Short timing blocks showed substantial variation with laptop load, so
  small calls were rechecked with nine rounds of 101 individually
  interleaved before/after pairs. Singleton speedups were 0.991-1.009x;
  CMA-population speedups were 1.035-1.060x. No material small-call slowdown
  remains. Peak traced allocations and all raw timing repeats are recorded;
  no fallback rows occurred in these measured states. These are acquisition
  timings, not estimates of whole-run speedup.
- The final full suite passed on `6734817`: 1254 passed, 39 skipped,
  three existing warnings in 567.46 seconds. The
  [full CI matrix](https://github.com/acerbilab/pyvbmc/actions/runs/34752259774)
  passed all nine cells, and the
  [smoke run](https://github.com/acerbilab/pyvbmc/actions/runs/34752255560)
  passed. The runner's v2 manifests bind source,
  allocation, environment and benchmark-data hashes. Existing v1 manifests
  remain intact; explicit `--equivalent-viqr` timing records both the
  capture and successor hashes and requires exact captured outputs.
- The completed replay's report-only check exited 0 with no flags. Metadata
  checks accepted an unchanged v2 resume and rejected eight changes covering
  source revision, VIQR/runner/production code, seed allocation, numerical
  dependency versions, GP sampler source and real-data archives. Default
  timing rejected the successor source until equivalence mode was explicit.
- Final independent Sol review of the source, recorded evidence and
  documentation found no remaining issues. The review checked the revision
  equivalence, timing claims, replay outcomes and stated coverage limits.

## Kernel-reuse implementation plan

Created 2026-09-13. **Status: COMPLETE.** This section owns the
cross-repository API, integration, validation and rollout decisions for
kernel reuse. The
[feasibility report](../results/2026-09-13-viqr-kernel-reuse.md) owns the
prototype measurements.

### Kernel-reuse execution checklist

- [x] K1: freeze before sources/oracles; implement and verify gpyreg API.
- [x] K2: implement bounded VIQR reuse and compatibility checks.
- [x] K3: numerical/performance gates, 18-run replay, full suites and CI.
- [x] K4: release validated gpyreg, update dependency/pin, integrate and archive.
- [x] Final independent doublecheck.
- [x] Closeout: evidence archived and implementation integrated into `dev-next`.

Execution baseline: PyVBMC `15a14cc`, feature branch
`dev-noisy-viqr-kernel`; gpyreg `a2f8ddc`/v1.1.0, feature branch
`dev-predict-cross-covariance`. Isolated before checkouts and the gpyreg
candidate are under `dev/scripts/runs/viqr_kernel_20260913/`.
Before-source worktrees and oracle-output dumps are preserved. The candidate
pair is PyVBMC `9e8d2f2` with gpyreg `355d754`; the dependency change is in
[gpyreg PR #45](https://github.com/acerbilab/gpyreg/pull/45).

Initial validation: gpyreg's full suite passed (198 tests); the focused API
module then passed 50 tests after adding the independent review's missing-factor
coverage. API coverage is consolidated in `test_predict_cross_covariance.py`.
PyVBMC acquisition/importance checks passed (123 tests), the normal oracle
suite passed (143 tests, 15 skips), and all 11 fixture groups passed the exact
comparison against the frozen outputs. Independent code reviews found no
implementation defect. Production measurements pass: all 96 comparisons are
exact; the late sieve is 1.205x faster; no control or fallback exceeds the 5%
slowdown threshold. Full details and raw measurements are in the
[production report](../results/2026-09-13-viqr-kernel-production.md).
Both nine-cell candidate CI matrices passed. The full local PyVBMC suite
passed (1277 tests, 39 skips, 2 successful reruns). The bounded 18-run replay
passed in 45.9 minutes from `16dcc3f` (documentation successor of `9e8d2f2`)
and gpyreg `355d754`: all 18 stored trajectories and final results are exact,
all initial designs match, and no runs are flagged. Before/after source and
input hashes match. Tracker edits during replay affect documentation only.
Independent replay review confirmed all output/input hashes and reported no
findings. Gpyreg PR #45 is merged as `39536b0`, whose tree equals `355d754`;
both repositories' merged-source matrices passed all nine cells before release.
Gpyreg's standalone tests workflow was re-enabled after GitHub disabled it
for inactivity. Gpyreg v1.2.0 is published from the validated merge; its PyPI
wheel passes API/version/source checks. The final dependency floor is 1.2.0
and CI pin is `39536b0`. The independent rollout doublecheck found no issues
in the release, dependency, provenance or integration ancestry.
The [final PyVBMC matrix](https://github.com/acerbilab/pyvbmc/actions/runs/34763665956)
passed all nine cells at `9c7d1b8` with the release pin and floor.
[PyVBMC PR #174](https://github.com/acerbilab/pyvbmc/pull/174) integrated the
implementation into `dev-next` at `5a5c17c`; the merge tree equals the reviewed
PR head. PyVBMC 1.5 publication remains a separate release task.

### Objective and boundaries

Compute the training-to-candidate covariance once per GP hyperparameter
sample and reuse it in standard VIQR. Preserve the criterion, integration
points and weights, random stream, candidate set, search settings, GP fits,
noise estimates, variance penalties, integer mapping and bound checks.
The accepted numerical scope is floating-point rounding; aim for exact
outputs on the generating platform, as in all 24 probe cases.

This change covers gpyreg's supported prediction API and the built-in
standard VIQR path. IMIQR cache reuse, experimental-loss removal, adaptive
sieve/search, multistart L-BFGS-B and BQ/QMC remain separate tasks. In
particular, none of those tasks depends on this change.

### Proposed API and integration

**gpyreg:** append a keyword-only `return_cross_covariance=False` argument
to `GP.predict`. When true, append a tuple with one entry per GP
hyperparameter sample to the existing return values:

- Ordinary prediction: `(mu, s2, cross_covariance)`.
- With `return_lpd=True`: `(mu, s2, lpd, cross_covariance)`.
- Each trained-GP entry contains the `(N_training, N_candidates)` latent
  kernel matrix `Ks = K(X, X_star)` already computed by the predictor.
  This is the unconditioned kernel, not posterior predictive covariance.
  Preserve sample order and values; do not stack or transpose for the return.
  Retain known unchanged bundled kernel implementations' fresh matrices
  directly. For custom or overridden covariance methods, snapshot each
  matrix immediately after computation: a custom kernel may reuse a scratch
  buffer across calls. This copying is confined to the requested extra
  output and does not change which matrix prediction uses. Do not impose
  new lifetime requirements on existing custom covariance implementations.
- `separate_samples=False` still averages `mu`, `s2` and any `lpd` as
  before; the kernel tuple remains per hyperparameter sample. There is no
  useful averaged kernel for the downstream calculation.
- For a prior-only GP with posterior hyperparameters but `self.y is None`,
  return one `None` entry per hyperparameter sample: the predictor has not
  computed a training cross-covariance. A GP with training data but missing
  factors after `clean()` or `compute_posterior=False` retains its existing
  unsupported prediction behavior; the new flag does not rebuild it.
- `add_noise`, `y_star`, `s2_star`, custom covariance implementations and
  both posterior factor representations retain their existing behavior.
  The extra output contains the kernel values used by that predictor.
- The default false path retains its return arity and does not retain
  matrices from earlier samples. No new GP attributes, persistent caches,
  serialization fields or random draws are introduced. Returned kernels
  are for read-only use by consumers; do not mutate flags on arrays owned
  by a custom covariance implementation. The returned tuple contains stable
  per-sample values; custom-kernel entries may be defensive copies.

**PyVBMC:** keep `_compute_acquisition_function`'s existing extension
contract. Introduce two protected hooks in `AbstractAcqFcn` for prediction
with an optional call-local context, and computation using that context.
The default hooks call the existing predictor and existing acquisition
method with their previous arguments. `__call__` retains its arithmetic
and pointwise checks, passes the context between the hooks, and releases
it after acquisition evaluation.

VIQR's hooks request and consume the kernel tuple only for the built-in
`AcqFcnVIQR` type with `getattr(self, "loss", "iqr") == "iqr"`, the standard
SE-ARD covariance implementation, and the unmodified gpyreg predictor.
Custom acquisition subclasses, custom/overridden covariance calculations
and GP prediction overrides use the original path. Recognize bound method
implementations, including instance overrides, rather than assuming
`isinstance` proves compatibility. This avoids replacing VIQR's hard-coded
SE calculation with a custom kernel's potentially different result.
Also require VIQR's `_compute_acquisition_function` to resolve to its
original implementation: an exact-type instance with an overridden method
must still call that override. Keep original method references outside
serialized instance state so later class or instance overrides are detected.

Move VIQR's common arithmetic into one private helper accepting an optional
kernel tuple. Its existing `_compute_acquisition_function` delegates with
no tuple; the fast hook delegates with the tuple. Both paths retain the
same loop and formulas. The fast path uses `cross_covariance[s].T` as a
view; it computes the candidate-to-integration-point kernel as before.
No output, context or matrix is stored on the acquisition, GP, VP or
`optim_state`. Legacy saved VIQR objects require no new attributes.

**Memory:** use an internal constant of **128 MiB** for the retained
cross-kernel payload. Before requesting the extra output, calculate
`8 * N_training * N_candidates * N_GP_samples` with Python integers on
the supported float64 path. Allow reuse at or below the cap; use ordinary
prediction and VIQR evaluation above it. Apply this check after candidate
normalization and integer mapping. This is a cap on retained matrix bytes,
not total process memory or the complete call's peak allocation. Empty or
untrained states follow the existing path. The check neither chunks nor
discards candidates and has no user-facing option.

### Phase K1: freeze the baseline and implement gpyreg

**Executor:** Astra orchestrator freezes provenance; Sol implementation
agent implements the API and focused tests. Only Astra runs heavy checks.

1. Inspect both repositories' instructions and Git state. Preserve unrelated
   changes. Start a PyVBMC feature branch from the settled guarded-sinh work
   (`a98dac0` at planning time), and a separate gpyreg feature branch from
   the checked source (`a2f8ddc`, v1.1.0, at planning time). Use isolated
   checkouts if either branch has concurrent work. Record exact source SHAs,
   installed module paths, library versions and hashes before measuring.
2. Read `dev/2026-09-02-modernization-discussion.md` before numerical edits.
   Dump before-change oracle outputs with
   `python dev/scripts/make_oracle_fixtures.py --dump-outputs <before-dir>`.
   Preserve the seven noisy captures, supplementary oracle capture, and
   18 guarded-sinh replay traces under
   `dev/scripts/runs/viqr_sinh_20260913/`. Verify their recorded hashes.
3. In gpyreg's `gpyreg/gaussian_process.py`, implement the optional output
   at the existing `Ks` assignment and return sites. Preserve numerical
   evaluation order and the false path. Do not use the feasibility probe's
   source substitutions in production.
   For zero-copy eligibility, snapshot the original compute methods of the
   five fresh-output bundled classes in a private module-level mapping:
   `SquaredExponential`, `Matern`, `RationalQuadraticARD`,
   `MaternIsotropic` and `SquaredExponentialIsotropic`. Require exact class
   identity and the original bound method on that instance. Unknown classes
   and class/instance overrides use
   `np.array(Ks, copy=True, order="K", subok=False)` for the returned entry.
   Evaluate eligibility only when the new output is requested.
4. Add focused checks in `gpyreg/testing/test_gaussian_process.py` for
   false/true output agreement; return arity
   with/without `lpd`; separate/averaged samples; observation-noise options;
   one/multiple samples; trained/prior-only GP; `(N,M)` orientation; custom
   kernels; and both `L_chol` representations. Count kernel evaluations to
   prove the option does not recompute `Ks`. Verify direct retention of
   known fresh built-in matrices, and independent snapshots from a custom
   kernel that deliberately reuses one scratch buffer across samples.
   Include an isotropic case in `test_gaussian_process_isotropic.py` and
   a Matern/RQ case to exercise the generic API. Check no matrices remain on
   the GP after a call. Use small explicit GP updates, not new fitting runs.
5. Update the `GP.predict` docstring, rendered by
   `docsrc/source/gaussian_process.rst` in gpyreg,
   with shapes, tuple order, lifetime and per-sample behavior. Correct the
   existing `add_noise` documentation mismatch while touching this
   docstring (the code default is false). Keep other GP APIs unchanged.

**Acceptance:** default prediction outputs and errors retain their previous
behavior; requested covariance entries equal the predictor's actual
matrices. Every supported return combination has a clear contract and a
focused check. Unexpected numerical differences require investigation.

### Phase K2: integrate bounded reuse in VIQR

**Executor:** Sol implementation agent; Astra reviews the cross-repository
boundary and runs checks. Depends on K1's API, not on a published release.

1. Implement the two protected hooks in
   `pyvbmc/acquisition_functions/abstract_acq_fcn.py`, preserving total
   variance aggregation, regularization and masking order.
2. Implement eligibility, memory fallback and shared arithmetic in
   `pyvbmc/acquisition_functions/acq_fcn_viqr.py`. Keep reduction losses
   and other acquisitions on their existing paths. Add no constructor
   arguments or serialized state.
3. Extend `test_abstract_acquisition_function.py` with an acquisition
   subclass implementing only the old method signature. In focused VIQR
   checks, cover custom VIQR subclasses and overridden acquisition, GP and
   kernel methods, including instance overrides on an exact-type VIQR
   object; verify those methods are still called.
4. Add fast/fallback equivalence checks at one, several and sieve-sized
   candidate counts; one/multiple GP samples; alternative quantiles; both
   factor representations; integer mapping, bounds, penalties and saved
   objects without `loss`. Use `_look_ahead.py` and the existing
   `test_acq_fcn_viqr_losses.py` setup for independent value checks.
5. Test the memory decision below, at and above the cap by lowering the
   private constant on small fixtures. Above the cap, assert prediction
   is called without requesting kernels, and the acquisition values and
   RNG state match ordinary evaluation. Exercise repeated calls after GP
   and candidate changes to catch accidental stale reuse. Check float64,
   input/state immutability, and absence of retained context after errors.
6. Update the existing acquisition API reference/docstrings as necessary
   to describe unchanged public behavior and custom-subclass compatibility;
   private hooks need developer docstrings, not new public API pages.

**Acceptance:** only eligible standard-VIQR calls request kernels. The
existing subclass signature and all fallback behavior remain usable.
Kernel matrices have one-call lifetime, and allocation of the retained
payload is guarded before prediction. No duplication of VIQR formulas.

### Phase K3: numerical, performance and replay gates

**Executor:** Astra main thread performs all compute. Independent Sol
reviewers use static analysis of code and recorded evidence.

1. Run focused acquisition/importance checks and the normal oracle suite:
   `python -m pytest pyvbmc/testing/acquisition_functions
   pyvbmc/testing/vbmc/test_active_importance_sampling.py -q`, then
   `python -m pytest pyvbmc/testing/oracles -q`. Run the generator with
   `--check --exact --against <before-dir>`. Require unchanged default
   GP predictions and classify every VIQR difference; do not regenerate
   references to make a change pass. Check unrelated oracle outputs too.
2. Keep `probe_viqr_kernel_reuse.py` and its recorded source hashes as the
   historical feasibility recipe. Add a production comparison mode or
   developer runner that evaluates the actual before/after packages on
   identical restored captures. It must support different gpyreg sources;
   do not disable the existing runner's dependency checks wholesale.
   Record both package revisions, hashes and imported module paths.
3. Use isolated, warmed before/after worker processes when comparing the
   two package pairs. Give each explicit source paths and the same numerical
   environment; time only complete public acquisition calls inside the
   workers. Alternate requests, keeping the other worker idle, to avoid
   concurrent heavy compute. Restore captured live factors and importance
   arrays without redrawing. Verify the baseline against captured outputs
   before accepting comparisons. Assert runtime BLAS thread counts of one.
4. Repeat the 24 captured cases and both fast/fallback paths. Record complete
   call timings, peak traced allocations, retained payload and fallback
   counts. Use nine alternating rounds, with at least 31 pairs per round
   for small calls. Include ordinary GP prediction with the flag false
   and representative non-VIQR acquisition calls to check hook overhead.
   Recheck only unresolved timing differences; no new target evaluations
   are needed for these measurements.
5. Performance acceptance: preserve a repeatable late-sieve gain (target
   at least 1.10x on the N=150 capture), with no repeatable slowdown above
   5% on small/default/fallback calls. Small early-state gains need not be
   significant. Account for timing variation and report individual states.
   Verify the payload cap; report total peak memory separately. A failure
   calls for revising the implementation, not silently raising tolerances
   or changing search settings.
6. Run the bounded 18-candidate replay, seeds 0-2 for the same six noisy
   configurations as the guarded-sinh campaign. Use
   `python dev/scripts/golden_replay.py --configs
   rosenbrock_D2_noise1,rosenbrock_D2_noise3,logreg_D5_noise3,student_D8_noise3,multisensory_s1_D6_noise1.3,timing_D5_noise2.2
   --seeds 0-2 --baseline dev/scripts/runs/viqr_sinh_20260913/replay
   --sidecars dev/golden/baseline --out <kernel-candidate-dir>`.
   The immediate trajectory baseline is the validated guarded-sinh
   campaign; accuracy fences remain the promoted 990-run population.
   Its use relies on the documented equivalence between the replayed
   guarded-sinh source and the final before-change numerical source.
7. Require identical initial designs, successful finite final results and
   no unresolved replay flags. Investigate trajectory changes against the
   immediate baseline and promoted population. Historical exact claims
   cover only stored state, not omitted returned-transformer information.
   If traces are missing, restore the preserved artifacts first; final-only
   comparisons do not replace the intended trajectory check. Any needed
   extension beyond this allocation requires a reason tied to a finding.
8. Run both repositories' full suites and CI matrices, repository formatting
   hooks, and independent doublecheck. On Windows use workspace-owned
   pytest temporary/cache directories when required. Record exact commands,
   versions, outcomes and unresolved limits. gpyreg's current matrix covers
   Python 3.9-3.11 on three OSes; PyVBMC's candidate matrix also covers
   Python 3.12. No fresh 990-run campaign or
   repair of historical inference failures is part of this change.

### Phase K4: dependency rollout and closeout

**Executor:** Sol prepares repository changes and reviewable PRs; Astra
coordinates verification, release sequencing and final integration.

1. During development, install the gpyreg candidate editable before running
   PyVBMC. Use its exact commit in development CI. Keep PyVBMC's current
   minimum requirement during this pre-release validation so dependency
   resolution does not request a nonexistent release. Do not merge or ship
   the new PyVBMC path against the old gpyreg pin.
2. After both-package validation and review, merge the gpyreg PR and record
   its merge SHA. Verify gpyreg CI and PyVBMC integration CI against that
   merged source are green; use the manual `gpyreg-ref` dispatch to check
   gpyreg main without waiting for the scheduled run. Fetch origin and tags
   and choose the next available additive release (proposed `v1.2.0`).
   Create the version tag and GitHub release at exactly the validated merge
   SHA, following gpyreg's release workflow. Verify publication and that
   the PyPI artifact exposes the new API before raising PyVBMC's minimum
   to that version and updating `GPYREG_PIN` in
   `.github/workflows/test-matrix.yml` to the released commit. Reinstall the
   editable sibling after fetching the release tag so setuptools_scm and
   the requirement agree.
3. Run the final PyVBMC nine-cell matrix with the final pin and dependency
   floor; verify installation resolves the intended gpyreg source/version.
   Keep gpyreg and PyVBMC commits separate and traceable. Integrate into the
   active 1.5 development branch after checks, without publishing PyVBMC 1.5.
4. Update this plan's execution status, `dev/TODO.md`, and the existing
   feasibility report with a link to production evidence. Store the
   consolidated raw results under
   `dev/experiments/noisy-acquisition-efficiency/`; use a new artifact for
   production validation so prototype measurements remain distinguishable.
   Update repository setup/pin notes where their minimum-version statement
   changes. Do not add a separate completion journal.

**Rollback:** revert the PyVBMC integration/pin change to the frozen
guarded-sinh source. The opt-in gpyreg API may remain released: ordinary
calls retain their old behavior. Preserve the before checkouts, captures,
oracle dumps and validation artifacts. Do not change the promoted golden
population as part of rollback.

### Decisions and approval

- **Optional gpyreg return, with a release-backed dependency floor:** a
  supported API avoids duplicated prediction logic and temporary monkey
  patches. Rejected permanent runtime feature detection against old gpyreg:
  it adds a second installed-dependency path for a capability both
  repositories can release together. Development validation precedes the
  floor bump so an unpublished dependency cannot block installation.
- **Transpose views, retained up to 128 MiB:** the probe's 56.25 MiB late
  kernel payload fits, preserving its measured gain. Rejected unconditional
  retention (unbounded extra memory), mandatory copies (lost much of the
  late gain), and streaming/chunking in this change (broader interfaces or
  changes to batching and arithmetic). The cap bounds kernel payload,
  not total memory, and remains an internal implementation choice.
- **Conservative fast-path eligibility:** built-in standard VIQR and
  standard GP/kernel implementations receive the optimization. Rejected
  automatically enabling it for arbitrary subclasses: VIQR's hard-coded
  SE calculation and custom prediction/kernel overrides may differ.
  At the public gpyreg boundary, unknown covariance implementations receive
  defensive output snapshots; changing the existing custom-kernel ownership
  contract would make an additive optimization unnecessarily disruptive.
- **Existing extension contract through private hooks:** avoids changing
  every acquisition method signature or retaining per-call state on objects.
- **Bounded 18-run validation against the immediate baseline:** separates
  this change from guarded-sinh trajectory differences and uses existing
  captures. Rejected a new reference-sized campaign without a specific
  unresolved regression question.

No scientific choice blocks this plan. API tuple semantics, the 128 MiB
cap, the release-backed dependency floor and the bounded replay allocation
were approved for execution on 2026-09-13. Exact release naming is confirmed
against gpyreg's tags at execution time. Plan approval covers implementation,
validation, the prerequisite gpyreg release and integration into the 1.5
development branch; PyVBMC 1.5 publication remains separate.

Independent Sol plan doublecheck completed on 2026-09-13 with no remaining
findings. Review corrections specify defensive snapshots for custom kernel
scratch buffers, fallback for acquisition-method overrides on exact-type
instances, and release from the merged, validated gpyreg commit. The review
covered source/API branches, probe evidence, tests, documentation and both
repositories' dependency/release workflows. This was a static plan review;
implementation tests and campaigns remain the work described above.
