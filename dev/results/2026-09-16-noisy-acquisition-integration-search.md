# Noisy-acquisition integration and search experiment

**Status:** E2 completed during the bounded 2026-09-17 continuation.
All four retained full-sieve integration settings failed the necessary
cost gate. E3's prescribed MC1600 fallback has all 384 development selections
saved: all four arms and eight repetitions on twelve states. Independent
judging, 36 paired timing cells and 72 separate memory cells are complete.
S2 is frozen for holdout under the improved-choice branch. After the recorded
approval interruption, the user explicitly authorized its 192 frozen-state
selections and independent judging at approximately 12:50 UTC. Holdout
selections all succeeded in 441.33 seconds, with no target calls or GP fits.
The holdout RQMC ladder is complete: 54 beneficial, 17 tied, 13 harmful and
12 unresolved comparisons out of 96. Independent ordinary-MC cross-checks
are complete with no resolved disagreement. The five-case saved-point loss
diagnostic is complete: no confirmed harmful refinement against its own
starting winner, with several losses localized before refinement. The mixed
holdout evidence does not support a general replacement recommendation.
E4's development entry gates are unmet. E5 was unapproved at the E3
checkpoint; the user subsequently authorized the exploratory pilot described
below, then its continuation. All 120 fits of the ten-seed S0/S2 design are
complete: the 24-fit pilot, its excluded reproducibility repeat, and the
96-fit continuation for seeds 2002-2009, which ran in two bounded batches on
2026-09-17/18 with no failed fit. S2 roughly halves total fit time at
unchanged evaluation counts on these cheap likelihoods, but it is not shown
to be free of target-specific accuracy loss: all 60 S0 fits converge against
55 of 60 for S2, high-noise Rosenbrock has one usability loss and gsKL
increases up to 103 times, logistic regression's posterior-shape metrics are
worse in eight of ten pairs, and Student-t gains six usable fits. The
ten-seed result does not support adopting S2 as a default; the plan's E6
assessment proposes narrowly targeted follow-ups. Production defaults are
unchanged.

This experiment evaluates the cost and selection quality of positive-weight
integration rules and smaller candidate searches for standard VIQR. The
[experimental plan](../plans/noisy-acquisition-efficiency.md) specifies the
allocation, independent judging, timing protocol and promotion criteria.
Changes to GP fitting and Bayesian quadrature are outside its scope.

## Sources and environment

The numerical baseline is PyVBMC commit
`9cc6882768ff682ae892a6b453c42e0f2d03d5fa`. The capture harness is committed
as `3fe616c477130625bb01d30d29785773fe2e0e66`. E1-E3 leave production
`pyvbmc/` source unchanged. E5 source checkpoint `d8eb7af` adds a private,
inactive-by-default selection callback; exact default-path oracle checks pass.

The frozen gpyreg source is commit
`9e70e6ba53f7607d05c2d9cc2fa9f41cd12b8f3b`, imported from the archived
`gpyreg_1.2.1` checkout. Its installed distribution metadata reports 1.2.0;
the source hashes and resolved import path identify the actual dependency.
The environment uses Python 3.12.6, NumPy 2.5.2, SciPy 1.18.1 and CMA 4.4.4
on Windows. `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS`
are all fixed at one. Numerical processes run sequentially.

The [environment preflight](../experiments/noisy-acquisition-efficiency/integration-search/environment_preflight.json)
records the dependency paths, build information and five data/truth archive
hashes. Raw states and per-cell evidence are stored under the local artifact
directory `dev/scripts/runs/noisy_acq_efficiency_20260916/`, indexed in
`dev/scripts/runs/LOCAL.md`.

## State coverage and verification

All seven historical VIQR captures reproduced their stored full-sieve
scores exactly, with no change to their state or RNG. The
[historical-state preflight](../experiments/noisy-acquisition-efficiency/integration-search/historical_state_preflight.json)
records these checks. Its single-call timings are feasibility measurements,
not performance evidence.

The [capture manifest](../experiments/noisy-acquisition-efficiency/integration-search/capture_manifest.json)
allocates six target/noise configurations, development seed 0 and holdout
seed 1, with early and late checkpoints for each trajectory. A trajectory
that terminates before the late threshold contributes its last distinct
eligible state. Captures retain live GP factors and the complete acquisition
setup; exact public-score replay is required before a state is accepted.
All twelve allocated capture runs completed without execution errors, and
all twenty-four states passed exact replay. The [capture inventory](../experiments/noisy-acquisition-efficiency/integration-search/capture_inventory.json)
records coverage and each checkpoint's charged evaluation count.

The expanded developer suite passed 133 checks covering capture identity,
snapshot reconstruction, weighted VIQR parity, numerical stress cases,
judge uncertainty, split locks, grouped denominators, target-free search,
exact CMA-ES baseline agreement, search/crosscheck provenance, failure
denominators, judging escalation and timing boundaries. Two warnings came
from deliberately invalid inputs in the failure-reporting test.
The existing acquisition/oracle suite passed 234 checks, with 15 skips and
three existing VP-density gradient warnings. The initial numerical tools
are committed as `8a13215`; the additional campaign, crosscheck and evidence
tools accompany this report's recovery record.

Independent static reviews covered the numerical protocol and tools before
the interruption, then the fixture repairs, saved-result provenance and
restart instructions during recovery. Recovery review findings were closed.
The bounded continuation adds independent checks of artifact hashes,
scheduled denominators, timing records and scientific gates.

## Experimental results

Development-panel selections, their prescribed RQMC judge ladder,
independent MC diagnostics and paired timing are complete. The four
full-sieve cost failures end those arms before fresh quality comparisons.
Search development and holdout subsequently completed, as recorded below.
Holdout treatment choices must be frozen before their evaluation. Failed,
missing and unresolved comparisons remain in the allocated denominators.

The [development panel manifest](../experiments/noisy-acquisition-efficiency/integration-search/panel_manifest.json)
allocates 864 treatment selections and 96 paired fresh production MC100
baselines. All 960 cells succeeded, with no failed, missing or partial
records. Morning recovery validated every terminal record and its bound
payload hashes against the frozen manifest and source identity; the
[completion record](../experiments/noisy-acquisition-efficiency/integration-search/panel_completion.json)
records those counts and hashes. The first paired cell passed the smoke, including
eight independent judge replicates on its 34-coordinate union. This smoke
establishes pipeline operation and provides no method-selection conclusion.

The authorized first execution window was 2026-09-16 18:34:41 UTC through
2026-09-17 05:34:41 UTC. A usage limit interrupted active work after screening.
The 04:46 UTC recovery check found no running Python experiment process.
Recovery completed verification and preserved the restart point; the user
needed the laptop approximately 45 minutes after that check. At that
recovery point, E2-E4 were incomplete. E5 requires authorization for a
separate execution window.

## Bounded continuation on 2026-09-17

The continuation started from `4145832`, reusing the existing environment
and saved captures/selections. E2's numerical sources remain unchanged.
The authorized
availability was initially 2–2.5 hours from approximately 09:03 UTC. A
10:07 UTC update confirmed approximately 90 minutes remaining: compute
stops by 11:22 UTC, with new launches stopping by 11:12 UTC and about
15 minutes reserved before the availability ends. Original allocation
records retain the initial cutoff; a successor record binds the amendment. E5 remains
unapproved. Work proceeds through completed state/budget cells so another
session can reuse every saved result.

The initial late Student-t pilot used 54 candidate coordinates, eight
4096-node comparison replicates and eight separate one-candidate
practical-band replicates. It succeeded in **3.247 seconds excluding
imports**. The remaining first round and full second round were estimated
conservatively at 8.8 minutes, accounting for training-set size,
hyperparameter-sample count, candidate count and worker startup. Subsequent
rounds were allocated from measured per-state costs, with allowance for
streaming when the node cache exceeds its memory cap.

All five prescribed budgets completed without failed or missing cells.
Completed comparisons are carried forward; only unresolved score or raw-loss
checks advance. The count requiring further resolution fell from 864 to
237, 175, 165 and finally 151. At the cap, 73 complete-score comparisons and
151 raw-loss checks remain unresolved. These are retained in all scheduled
denominators.

| Integration rule | Nodes | Beneficial | Harmful | Practical tie | Unresolved | Material raw loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MC | 128 | 28 | 25 | 35 | 8 | 12 |
| MC | 512 | 41 | 11 | 35 | 9 | 4 |
| MC | 2048 | 48 | 4 | 36 | 8 | 0 |
| Stratified MC | 128 | 29 | 26 | 33 | 8 | 13 |
| Stratified MC | 512 | 46 | 11 | 30 | 9 | 3 |
| Stratified MC | 2048 | 53 | 2 | 33 | 8 | 0 |
| Stratified RQMC | 128 | 45 | 10 | 33 | 8 | 3 |
| Stratified RQMC | 512 | 52 | 1 | 35 | 8 | 0 |
| Stratified RQMC | 2048 | 53 | 0 | 36 | 7 | 0 |

Each row contains 96 scheduled state-by-repetition comparisons against a
paired fresh production MC100 selection. The twelve development states
come from six trajectories; repetitions and checkpoints are clustered
observations. Material raw loss is a separate check for losing more than
10% of a resolved positive raw IQR reduction; its counts overlap the score
classifications. The panels contain the captured baseline's top 32 candidates
plus 480 seeded candidates, so these results do not establish full-sieve
selection quality. The [panel evidence](../experiments/noisy-acquisition-efficiency/integration-search/panel_evidence_20260917.json)
records scheduled denominators, per-trajectory counts, worst losses and
source hashes for the judge, MC diagnostic and paired timing summaries.

The independent ordinary-MC diagnostic selected 273 comparison tags under
the prespecified extrema/material-loss rule, deduplicated to 183 candidate
pairs. Both 32768- and 65536-node budgets completed on all twelve states.
Complete-score classifications agreed with RQMC for 166 pairs, disagreed
for none, and remained unresolved for 17. Raw-loss classifications agreed
for 154 pairs, disagreed for none, and remained unresolved for 29. Unresolved
diagnostics do not certify the corresponding RQMC verdicts.

Two early high-noise Rosenbrock selections need particular caution: MC2048
repetition 3 and RQMC512 repetition 6 have harmful complete-score verdicts,
with mean score increases about 0.0256 and 0.0249. Their RQMC raw-reduction
ratios are about 0.548 and 0.555. The independent MC diagnostic includes both
pairs and points in the same adverse direction, with ratios about 0.531 and
0.539. Its intervals indicate loss but fail the prescribed relative-precision
test; the raw-loss classifications remain unresolved. These cases prevent
a blanket quality claim for either setting despite low pooled harmful
fractions. They remain part of the full-sieve investigation record.

Its largest predicted-workload pilot, late logistic regression with 29
candidates and six GP hyperparameter samples, took 24.930 seconds excluding
imports at 32768 nodes. Recorded MC cell durations totaled 513.99 seconds,
including imports for fresh workers but excluding them for the pilot.
This exceeded the 420-second planning estimate by 93.99 seconds. The
training-count/sample-count/candidate-count scaling model missed quadratic
GP solve cost and fixed setup costs; subsequent estimates use measured
per-state costs. The allocation decision was communicated before execution;
its local JSON record explicitly identifies its later reconstruction.

All 108 warmed timing cells succeeded. Each state/setting has seven paired
rounds with alternating order, including node generation, cache preparation
and selection against the production MC100 path. The following ratios are
medians of the twelve state-median treatment/production ratios:

| Integration rule | 128 nodes | 512 nodes | 2048 nodes |
| --- | ---: | ---: | ---: |
| MC | 1.279 | 6.204 | 24.908 |
| Stratified MC | 1.374 | 6.476 | 27.147 |
| Stratified RQMC | 1.485 | 6.922 | 25.265 |

Recorded timing-worker wall time was 407.12 seconds including imports,
below the conservative 957.93-second allocation estimate. Worker wall time
is a scheduling measurement; the ratios above use only the warmed selection
operations. None meets the benefit gate on the 512-candidate panels.

The [frozen selection](../experiments/noisy-acquisition-efficiency/integration-search/full_cost_selection_20260917.json)
and [full-sieve manifest](../experiments/noisy-acquisition-efficiency/integration-search/full_cost_manifest_20260917.json)
allocate four settings for an **8192-candidate cost screen**: MC2048,
stratified-MC2048 and stratified-RQMC512/2048. These retain
the integration controls and the two RQMC accuracy/cost settings without
confirmed material raw loss. The other five panel settings exceed 5%
harmful selections and have MC-confirmed material losses. The shortlist
retains the unresolved Rosenbrock limitations above; stratified-MC2048 also
has an MC-agreed harmful late timing-target selection, with a RQMC raw
reduction ratio about 0.939 and an unresolved MC raw-loss check.

Full-sieve paired timing precedes fresh quality comparisons. A setting with
a median state ratio above 1.10 cannot satisfy either the 20% speedup branch
or the better-quality-at-comparable-cost branch. Such an arm ends at this
necessary cost gate without allocating fresh selection/judging or holdout
work. Any setting at or below 1.10 must still pass the original quality
gates. This ordering was independently reviewed before allocation; it
changes no threshold or numerical setting. Timed selections supply runtime
evidence only. Incomplete timing leaves the corresponding setting pending.

The full-sieve late-logistic-regression RQMC2048 timing pilot succeeded in
69.22 seconds, versus 98.18 seconds predicted from its panel measurements.
RQMC512 subsequently completed all twelve states: its median state ratio
is 3.8154, with a range of 2.0192–4.8987. It fails the cost gate and receives
no fresh full-sieve quality allocation. Its recorded worker time was
246.89 seconds versus 150.51 seconds predicted. The updated remaining-work
estimate exceeded the window at its conservative bound, so further work
is allocated one complete setting at a time, subject to the cutoff.
RQMC2048 also completed all twelve states and failed the cost gate: median
15.0508, range 7.2013–19.8370. Its scientific timing records are complete.
The outer scheduling duration for one cell is unavailable because the
orchestration process was replaced after that worker completed; its seven
paired rounds and hashed terminal record were preserved.
MC2048 completed all twelve states and also failed the cost gate: median
14.2562, range 7.4105–18.4508. Its measured worker time was 885.88 seconds.
The final stratified-MC2048 allocation uses a per-state forecast calibrated
to these MC timings: 1029.42 seconds predicted and 2118.83 seconds
conservative, against 2511.08 seconds remaining to the soft stop when
allocated. Successor records preserve each revision to the forecast.
Stratified-MC2048 completed all twelve states in 1087.86 seconds of worker
time and failed the cost gate. The complete full-sieve screen is:

| Integration rule | Nodes | Median state cost ratio | State ratio range | Decision |
| --- | ---: | ---: | ---: | --- |
| MC | 2048 | 14.2562 | 7.4105–18.4508 | Cost failure |
| Stratified MC | 2048 | 15.3588 | 6.4406–18.3680 | Cost failure |
| Stratified RQMC | 512 | 3.8154 | 2.0192–4.8987 | Cost failure |
| Stratified RQMC | 2048 | 15.0508 | 7.2013–19.8370 | Cost failure |

All 48 timing cells succeeded, with seven paired rounds each. No setting
meets the necessary cost condition, so E2 closes without an integration
finalist. No fresh full-sieve quality, memory or holdout cells were launched.
These cost failures do not establish full-sieve selection quality.
The [full-sieve cost screen](../experiments/noisy-acquisition-efficiency/integration-search/full_cost_screen_20260917.json)
records every state ratio and terminal/report hash. The
[E2 continuation record](../experiments/noisy-acquisition-efficiency/integration-search/continuation_20260917_final.json)
binds the judging, diagnostic, timing and allocation evidence.
Forty-seven outer worker durations total 2857.46 seconds. Adding the
explicitly estimated 59.02 seconds for the unavailable outer duration
gives an estimated total of 2916.48 seconds, versus an initial 1852.34-second
prediction and 3947.09-second conservative estimate. The one environment
preflight failure is recorded separately: an incorrect dependency import
path was rejected before numerical work, then corrected for the unstarted
cell. All successful scientific cells used the frozen dependency identity.

Before E3 allocation, independent review identified that its developer
harness applied the 50-iteration limit separately to each local start.
The approved plan specifies a shared limit across starts. The correction
counts completed iterations through SciPy's callback, passes only the
remaining allowance to each subsequent start, and retains that count if
a row-budget exception interrupts the solver. The shared 1000-row cap and
re-scored fallback remain in force. This changes only the experimental
search harness; E2 does not import or bind that module. E3 manifests must
bind the corrected source. Focused verification covers cumulative limits,
interrupted runs and agreement with SciPy's returned iteration counts.
Both focused modules passed: **25 tests in 3.72 seconds**, using the frozen
dependency and single-thread environment. An initial run passed the nine
search-core tests but could not create the default temporary directory for
the campaign tests; the successful rerun used a workspace-local temporary
directory. Black's direct formatter check, isort and `git diff --check`
passed. The Black CLI's Windows process pool stalled, so the same installed
formatter API checked the files directly. Standalone Pycln was unavailable
during preflight. All repository pre-commit hooks subsequently passed,
including Black and Pycln in their hook environments.

The twelve-cell MC1600 search pilot completed successfully in 25.57 seconds
of outer worker time including imports (1.91–2.48 seconds per cell). All
four arms ran on early Rosenbrock noise 3, late logistic regression and
late Student-t. State/RNG and source identities were preserved, and no
target evaluations or GP fits occurred. The fourteen step-halving
discrepancies were at most 3.70e-8, well below the frozen 0.25 threshold;
objective scales were positive and fixed across each state's S2/S3 runs.
Completed solver iteration counts matched callback counts.

S2 used 6, 13 and 21 iterations and 59, 196 and 451 accurate candidate rows
on those three states. S3 used 36, 49 and 44 iterations and 357, 727 and 996
rows. Student-t S3 reached the candidate-row cap and returned the prescribed
re-scored fallback. Independent review accepted these operational checks
for further development testing with unchanged settings. They provide no
independent quality comparison or choice of search arm.

The [pilot outcome](../experiments/noisy-acquisition-efficiency/integration-search/search_pilot_outcome_20260917.json)
records its calibration diagnostics and per-cell costs. A
[successor allocation](../experiments/noisy-acquisition-efficiency/integration-search/search_rep0_allocation_20260917.json)
completed repetition zero on the remaining nine states, adding exactly 36
successful cells. Its forecast was 74.82 seconds predicted and 180 seconds
conservative, with unchanged launch and compute cutoffs. All 48 scheduled
repetition-zero cells succeeded; at this checkpoint the other 336 selections,
timing, memory, judging and holdout remained unallocated. Numerical work stopped after these
48 cells, before the soft launch cutoff.
The [combined repetition-zero outcome](../experiments/noisy-acquisition-efficiency/integration-search/search_rep0_outcome_20260917.json)
records 104.51 seconds of total outer worker time, including 78.94 seconds
for the additional 36 cells. Six selections used the prescribed row-budget
fallback. Every cell respected the 50-iteration and 1000-row limits.

The remaining **336 selections alone** were estimated at **12.2 minutes
predicted, 30.1 minutes conservative**, using each state/arm's measured
repetition-zero cost for its seven remaining repetitions. These are
scheduling allowances, not statistical bounds or paired performance
measurements. The 36 paired timing cells, 72 memory/allocation cells and
independent judging/escalation required additional estimates and allocation.
No search quality or speedup conclusion was available at this checkpoint.
Its process check found no Python processes or jobs needing reattachment.

## Additional selection window on 2026-09-17

At approximately 11:24 UTC, the user authorized another 30+ minutes from
checkpoint `250460b`. This extension supersedes the earlier operational
cutoff: new launches stop by 11:51 UTC and compute stops by 11:54 UTC.
The allocation reuses all 48 validated repetition-zero cells and runs
up to the exact 336 remaining selections, repetitions 1–7, from the same
frozen manifest. Each completed cell is saved independently. Failure,
identity mismatch or insufficient remaining time stops further launches.
Timing and independent judging require bounded cost pilots before any
further matrix allocation. Sources, scientific gates and E5's unapproved
status are unchanged.

At approximately 11:49 UTC the user extended availability by another
2.5 hours. The successor window stops new launches by 14:04 UTC and compute
by 14:09 UTC. It covers the remaining prescribed judging and measurements,
with E4 conditional on its original entry gates and E5 still unapproved.

All 336 additional selections succeeded in 763.63 seconds of worker time,
giving 384 successful development selections with no failed or missing
cells. Independent review verified every terminal/report/NPZ hash and the
reused repetition-zero records. Among the additional cells, 26 used the
prescribed candidate-row fallback; no other fallback reason occurred. The
observed maxima were 50 local iterations and 999 accurate candidate rows.

The late-logistic-regression 4096-node judge pilot succeeded in 8.83 seconds
of outer worker time on a 63-coordinate union. Its eight comparison rules
and eight separate band-pilot rules used the frozen 4096-node allocation.
The paired S3 timing pilot completed its numerical rounds but failed when
publishing JSON, after 11.05 seconds of worker time. It supplies no usable
timing ratio or performance verdict.

Independent static review identified a deterministic reporting defect:
`_timing_factory` retains the selection result's diagnostic `cache_indices`
array, which contains NaN sentinels for newly generated candidates.
`canonical()` converts an array with `tolist()` without recursively
encoding its nonfinite entries, and strict JSON serialization rejects it.
The failure terminal and partial temporary JSON are preserved. Further
timing cells were paused pending a reviewed reporting repair and fresh
measurement allocation that preserves the frozen selection evidence.
The judge does not consume timing artifacts and its successful pilot can
be reused; its remaining initial-budget cells were separately allocated
at 60 seconds predicted and 120 seconds conservative.

The complete 4096-node round took 76.02 seconds of outer worker time.
All 576 scheduled comparisons were observed: 288 primary comparisons with
S0 and 288 component contrasts. All remained unresolved under the required
consecutive-budget rule. The 8192-node round completed in 84.57 seconds;
the 16384-node round evaluated the eight remaining states in 79.45 seconds.
The 32768-node round took 149.16 seconds for eight states; the 65536-node
round took 189.93 seconds for six states. Every escalation used the frozen
pending-state set. There were no failed or missing judge comparisons.
The cap retains 76 unresolved complete-score and 99 unresolved raw-loss
comparisons across the 576 primary and component comparisons.
The [completed ladder](../experiments/noisy-acquisition-efficiency/integration-search/search_judge_ladder_20260917.json)
binds all budget summaries, allocations and terminal payloads. One
orchestration-provenance limitation remains: the wrapper used for budgets
8192-32768 was corrected in place before 65536, so its earlier exact bytes
are unavailable. Independent review verified that every executed state set
matches the frozen escalation predicate, and all 46 judge terminals and
their numerical payloads remain authenticated. Numerical source files were
unchanged.

Primary comparisons with S0, each out of all 96 scheduled pairs, are:

| Arm | Beneficial | Practical tie | Harmful | Unresolved | Material raw loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| S1: re-scoring | 38 | 20 | 25 | 13 | 4 |
| S2: one refinement | 54 | 19 | 5 | 18 | 2 |
| S3: multiple starts | 53 | 18 | 8 | 17 | 2 |

These are development screening counts, with repetitions and early/late
states clustered within six trajectories. The investigated material losses
remain limitations in the eligibility assessment below.
S1's worst confirmed raw losses retain 66.89% of S0's reduction on early
Rosenbrock noise 1, repetition 3, and 67.67% on early Rosenbrock noise 3,
repetition 0. Its other two material losses are the Student-t cases below.

A separate versioned [measurement adapter](../scripts/noisy_acq_search_measurements.py)
passed independent static review.
It calls the frozen timing/allocation routines and projects selection
diagnostics into small JSON-safe provenance records only after measurement
ends. Its new manifest and namespace bind the unchanged selection evidence,
original measurement seeds, failed pilot and new adapter source. The same
reporting defect affects the original memory-cell path, which also retains
the full selection result; both measurement types require the adapter.
All five focused adapter tests passed in 1.28 seconds. Black, isort, Pycln
and whitespace checks passed before the adapter source was frozen. The
adapter and its tests use CRLF working-copy bytes to match this checkout's
Git configuration; manifests bind their raw file hashes.

The repaired three-cell pilot succeeded: one timing cell in 10.46 seconds
of worker time and two memory cells in 2.77 and 2.93 seconds. These cells
count toward the full matrices. The remaining 35 timing and 70 memory cells
all succeeded in 430.63 seconds, versus 388.97 predicted and 777.95
conservative. This completes all 36 timing and 72 memory cells without
measurement failures. Judge/import time is excluded from paired selection
timings; memory instrumentation runs in separate processes.

Each table entry is the median of seven paired treatment/S0 selection-time
ratios. The final row is the median of the twelve state medians.

| Development state | S1 | S2 | S3 |
| --- | ---: | ---: | ---: |
| Rosenbrock noise 1, early | 0.101 | 0.173 | 0.184 |
| Rosenbrock noise 1, late | 0.245 | 0.639 | 1.504 |
| Rosenbrock noise 3, early | 0.107 | 0.172 | 0.163 |
| Rosenbrock noise 3, late | 0.246 | 0.335 | 0.883 |
| Logistic regression, early | 0.075 | 0.175 | 0.483 |
| Logistic regression, late | 0.319 | 0.441 | 0.869 |
| Student-t, early | 0.132 | 1.155 | 1.154 |
| Student-t, late | 0.333 | 0.546 | 0.888 |
| Multisensory, early | 0.061 | 0.293 | 0.511 |
| Multisensory, late | 0.301 | 0.464 | 0.908 |
| Timing, early | 0.088 | 0.716 | 0.775 |
| Timing, late | 0.281 | 0.426 | 0.811 |
| **Median** | **0.188** | **0.434** | **0.840** |

The [measurement outcome](../experiments/noisy-acquisition-efficiency/integration-search/search_measurement_outcome_20260917.json)
binds all timing records and lists the separate memory cells.
Treatment traced peaks range from 7.39 to 41.63 MiB; each arm's median
statewise traced-peak ratio to its paired S0 measurement is about 0.201.
S2's maximum process-lifetime resident peak is 177.99 MiB. Tracemalloc
excludes some native allocations, and resident peaks include process setup;
these measures describe different quantities and are not timing evidence.

The development gate was frozen before the search-specific ordinary-MC
diagnostic was run. The completed E2 panel diagnostic does not cover E3
search comparisons. The 2026-09-17 continuation identified this outstanding
cross-check and retained the frozen S2 choice while allocating the existing
search-crosscheck harness. This is a sequencing deviation: search-specific
MC confirmation follows the frozen finalist and holdout evaluation. Neither
its results nor the holdout results can revise the treatment choice. Its
outcome must accompany any positive E3 claim.

The [development gate assessment](../experiments/noisy-acquisition-efficiency/integration-search/search_development_gate_20260917.json)
freezes S2 under the **improved choices at comparable cost** branch:
54 beneficial comparisons out of 96, 18.75% unresolved, and a timing ratio
of 0.43387, below 1.10. The faster-selection branch fails its distinct
5% harmful ceiling because 5/96 is 5.21%. The raw-loss investigation
identified the exact fallback mechanism, satisfying the plan's requirement
to explain reproducible material losses while retaining them as observed
regressions. Early Student-t is the worst joint limitation: its state-median
timing ratio is 1.15487, its round range is 1.13708-1.84151, and it has the
two material raw losses described below. The whole Student-t trajectory
has 13 beneficial and 3 harmful comparisons. These clustered observations
do not establish noninferiority or inference performance.

S1 has substantially more harmful selections. S3 adds time and harmful
fallbacks without improving the overall quality counts. The
[holdout selection](../experiments/noisy-acquisition-efficiency/integration-search/search_holdout_selection_20260917.json)
therefore freezes only S2, with unchanged MC1600, 1024 candidates, eight
re-scored points, one local start, and the shared 50-iteration/1000-row caps.
It was written before opening holdout for search evaluation. E0's capture
and exact-replay validation covered both splits; holdout search outcomes
did not inform this selection. Its schema requires
`include_mc1600_control=true`; because S2 itself uses MC1600, the manifest
omits a duplicate control and allocates only S0 and the finalist.

The independently resolved raw-loss cases shared by S2 and S3 occur on
early Student-t, repetitions 1 and 4. Their raw reductions retain 85.63%
and 85.97% of S0's reduction. Both arms exhaust the shared candidate-row
budget at 994 rows and 40 completed iterations, then return the same
accurately re-scored shortlist fallback. This is a limitation of the bounded
search and its fallback. It does not demonstrate acceptance of a harmful
refinement or establish that a larger integration budget would remove the
loss. These cells remain in the quality-gate denominator.

Independent review found that E4's entry conditions are not established.
The first condition requires full-sieve quality evidence at multiple fixed
integration budgets; E2 ended those arms at its necessary cost screen.
For the second condition, four harmful component comparisons are accepted
refinements: S2 versus S1 and S3 versus S1 on late timing, repetitions 2
and 3. They belong to one trajectory, below the required two. The other ten
harmful component comparisons are S3 versus S2 on late Rosenbrock noise 1
(two repetitions) and late Student-t (eight repetitions). Every S3 outcome
is a row-budget fallback to its shortlist, while the S2 reference retains
a refined point. They do not demonstrate accepted harmful refinement on
additional trajectories. No higher-budget confirmation or adaptive arm is
allocated. The present comparisons also change both sieve and refinement
behavior, so they do not isolate a sieve-coverage defect that a larger sieve
would address.

## Frozen S2 holdout evaluation

All 192 selections succeeded in 441.33 seconds of outer worker time,
against 443.69 predicted and 766.64 conservative. Independent review
authenticated every terminal, report and selected-coordinate archive.
The maximum accurate-candidate count was 876, below the frozen 1000-row
cap. There were no target evaluations or GP fits.
The [selection outcome](../experiments/noisy-acquisition-efficiency/integration-search/holdout_selection_outcome_20260917.json)
binds its completion snapshot, costs and fallback diagnostics.

The RQMC judge used 12, 12, 6, 5 and 4 state cells at budgets 4096,
8192, 16384, 32768 and 65536. Their worker times were 62.93, 79.82,
58.62, 83.79 and 100.47 seconds, respectively. Each escalation used
the frozen predicate and an immutable completed previous summary.
All 96 primary comparisons were observed without a failed or missing cell.
The [judge outcome](../experiments/noisy-acquisition-efficiency/integration-search/holdout_judge_outcome_20260917.json)
records grouped counts and every material loss; the
[completed ladder](../experiments/noisy-acquisition-efficiency/integration-search/holdout_judge_ladder_20260917.json)
binds each allocation, cost record and immutable summary. The exact executed
holdout wrappers were archived before use, as recorded in their
[archive index](../experiments/noisy-acquisition-efficiency/integration-search/holdout_wrapper_archive_20260917.json).

| Holdout trajectory | Beneficial | Practical tie | Harmful | Unresolved | Material raw loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 12 | 1 | 2 | 1 | 2 |
| Multisensory | 8 | 1 | 0 | 7 | 0 |
| Rosenbrock noise 1 | 5 | 3 | 8 | 0 | 7 |
| Rosenbrock noise 3 | 5 | 10 | 1 | 0 | 1 |
| Student-t | 15 | 1 | 0 | 0 | 0 |
| Timing | 9 | 1 | 2 | 4 | 1 |
| **All 96 scheduled** | **54** | **17** | **13** | **12** | **11** |

Among the 84 score-resolved comparisons, 64.29% are beneficial, 20.24%
tied and 15.48% harmful. Fractions of all scheduled comparisons are
56.25%, 17.71%, 13.54% and 12.50% unresolved. The separate raw-loss
assessment has 77 no-material-loss, 11 material-loss and eight unresolved
results. Early/late states and repetitions remain clustered within six
trajectories; these are not independent inference runs.

All eight late Rosenbrock-noise1 repetitions triggered the prescribed
finite-difference instability guard. Each used 18 accurate rows and zero
local iterations, then returned a finite, accurately re-scored shortlist
winner. Six have material raw losses, retaining 77.10%-88.80% of S0's
reduction. The guard behaved as specified, but its fallback does not
recover the larger production search's choices. The other five material
losses occurred in accepted-refinement outputs: early Rosenbrock noise 1
repetition 5, late Rosenbrock noise 3 repetition 0, early logistic regression
repetitions 0 and 4, and late timing repetition 6. Their raw reductions
retain approximately 72.88%, 68.42%, 79.89%, 79.46% and 87.43% of S0's,
respectively. Each refined point improved its own fixed MC1600 score;
that internal comparison does not establish independent improvement over
its shortlist or explain the loss against S0. The completed cross-method
checks and saved-point diagnostic below assess these cases with S2 unchanged.

The holdout allocation uses twelve seed-1 states, eight fresh repetitions
and S0 versus the unchanged finalist: 192 selections and 96 primary judge
comparisons. The generic manifest also declares twelve timing and 24 memory
cells; these are scientifically unallocated because the finalist was chosen
using completed development timings and memory measurements. Holdout
evaluates the fixed selection rule with fresh independent judge streams.
It cannot tune the arm or change the E4 decision.

The locked draft passed independent review for source/configuration identity,
split seeds, state coverage and exact cell allocation. Matched development
S0/S2 workers took 443.69 seconds
for 192 cells; the holdout conservative allowance is 766.64 seconds, using
1.5 times the largest observed worker cost per cell. Judging is allocated
separately from its pilot and successive complete budget summaries.
The original RQMC allocation retained its 14:04 UTC launch cutoff and
14:09 UTC hard stop. The user's further 90-minute window at approximately
13:04 UTC supports successor MC allocations with 14:29/14:34 UTC guards.

Execution was paused at manifest unlock. Automatic approval review twice
classified the action as unapproved E5; the second request also incorrectly
called it E4. The corrected request identified the plan's E3 frozen-state
holdout, its source-level guards against target calls and GP fits, and the
frozen gate/selection/draft hashes. That request returned "aborted by user"
after 5.1 seconds. No executable holdout manifest was written and no holdout
worker started. Explicit confirmation was requested before retrying this
action. This is an approval interruption, not an experimental failure.
At approximately 12:50 UTC the user answered "Run the E3 holdout" to the
explicit request for 192 frozen-state S0/S2 selections and independent
judging, with no target evaluations, GP fits or E5 inference. This authorizes
the reviewed manifest unlock and execution; the earlier pause record remains
immutable. That RQMC allocation retained the 14:04/14:09 UTC cutoffs.
The [authorization successor](../experiments/noisy-acquisition-efficiency/integration-search/search_holdout_authorization_20260917.json)
binds this confirmation to the unchanged finalist and reviewed draft.
Independent review verified that the
[executable holdout manifest](../experiments/noisy-acquisition-efficiency/integration-search/search_holdout_manifest_20260917.json)
changes only the readiness and holdout-lock flags. The
[192-cell allocation](../experiments/noisy-acquisition-efficiency/integration-search/search_holdout_allocation_20260917.json)
and [launch clearance](../experiments/noisy-acquisition-efficiency/integration-search/search_holdout_clearance_20260917.json)
bind every scheduled tag and the executed wrapper.
The [pause record](../experiments/noisy-acquisition-efficiency/integration-search/search_holdout_approval_pause_20260917.json)
binds the prepared state. The local wrapper archive preserves fourteen
available scripts, including the finalist-freeze script, with hashes in its
[archive index](../experiments/noisy-acquisition-efficiency/integration-search/search_wrapper_archive_20260917.json).
That development-checkpoint process check found no Python workers. Two workspace-local pytest
temporary directories remain because Windows denied their removal; neither
is tracked or contains experiment evidence.

## Search-specific independent MC checks

The prescribed development diagnostic selected 183 comparison tags from
the completed RQMC summary: each state's minimum and maximum mean score
difference, every comparison at least ten practical bands from zero, and
every material raw loss. Exact-coordinate deduplication gives 137 pairs
across twelve states. Eight fresh ordinary-MC rules at 32768 and 65536
nodes use the original bands and consecutive-budget criteria. All cells
succeeded. Worker times were 167.94 and 362.73 seconds, respectively.

Of the 137 pairs, complete-score checks agree with RQMC on 118, disagree
on none and remain unresolved on 19. Raw-loss checks agree on 108,
disagree on none and remain unresolved on 29. The two S2 Student-t early
losses have harmful MC complete scores, agreeing with RQMC; their MC
10% raw-loss checks remain unresolved. These outcomes confirm the score
direction while leaving that threshold independently unconfirmed.

The holdout [explicit allocation](../experiments/noisy-acquisition-efficiency/integration-search/search_mc_holdout_catastrophes_20260917.json)
contains 48 distinct pairs across eight states: every apparent ten-band
gain or loss and all eleven material raw losses. Holdout extrema alone
do not qualify. This diagnostic uses fresh streams and cannot override
the frozen primary RQMC classifications or tune S2.
All eight state cells succeeded at each budget. The 32768-node round took
276.64 seconds and the 65536-node round 450.21 seconds. Score checks agree
on 47 of 48 pairs, disagree on none and leave one unresolved; raw-loss
checks agree on 44, disagree on none and leave four unresolved. All eleven
RQMC material-loss cases have harmful MC complete scores. MC independently
confirms seven of their 10% raw-loss classifications; the two early logistic
regression cases, late Rosenbrock-noise1 repetition 7 and late timing
repetition 6 remain unresolved on that threshold.

Pilot-based forecasting underestimated the seven remaining holdout
32768-node cells: 234.45 seconds observed versus 51.55 predicted and 103.09
conservative. The successor round used twice each state's actual previous
cost, predicting 553.28 seconds with a 1106.56-second conservative allowance.
Its 450.21 seconds fit both. Every cell remained within the bounded worker
allocation; no scientific setting or denominator changed because of cost.
The [combined MC outcome](../experiments/noisy-acquisition-efficiency/integration-search/search_mc_outcome_20260917.json)
binds the [development outcome](../experiments/noisy-acquisition-efficiency/integration-search/search_mc_development_outcome_20260917.json)
and [holdout outcome](../experiments/noisy-acquisition-efficiency/integration-search/search_mc_holdout_outcome_20260917.json),
including the original material-loss tags and their MC classifications.
The [development ladder](../experiments/noisy-acquisition-efficiency/integration-search/search_mc_development_ladder_20260917.json)
and [holdout ladder](../experiments/noisy-acquisition-efficiency/integration-search/search_mc_holdout_ladder_20260917.json)
preserve both budget allocations, summaries, worker costs and source bindings.

The separate [loss-diagnostic adapter](../scripts/noisy_acq_search_loss_diagnostic.py)
compares a saved accepted refined point with its accurately re-scored
pre-refinement winner and S0. It is a post-hoc explanation of confirmed
or unresolved material losses, outside the 96 primary comparisons.
Independent static review cleared candidate reconstruction, source bindings,
fresh common streams, consecutive-budget checks and conservative
interpretation of all three contrasts. A real-data preflight caught a
raw-file versus semantic-digest comparison error before any diagnostic
evaluation. The corrected adapter and a regression test preserve the
distinct parent hash contracts; the earlier source archive and failed
preflight remain recorded. Twelve focused tests passed in 1.46 seconds;
Black, isort, Pycln and whitespace checks passed before the corrected
source was frozen.

Both diagnostic budgets completed: four states, five cases and fifteen
contrasts per budget, with eight fresh ordinary-MC rules at 32768 and 65536
nodes. Worker times were 48.09 and 92.07 seconds. The final round used a
185.90-second prediction and 371.80-second conservative allowance based on
matched full-state MC costs. All cases succeeded; no target calls or GP fits
were made. The [diagnostic outcome](../experiments/noisy-acquisition-efficiency/integration-search/search_loss_outcome_20260917.json)
and [completed ladder](../experiments/noisy-acquisition-efficiency/integration-search/search_loss_ladder_20260917.json)
bind the source versions, failed preflight, exact allocations and results.
A wrapper-identity check was recorded after the first budget's freeze;
independent review verified that the frozen, executed and archived bytes
all matched the reviewed wrapper. An append-only clearance preserves this
sequencing exception without changing any numerical evidence.

Here, "own winner" is S2's accurately re-scored shortlist winner before
local refinement. Score classifications use the frozen practical bands
and consecutive-budget rule; no penalty is active in these cases.

| Holdout case | Own winner vs S0 | Selected vs own winner | Selected vs S0 | Direct raw-loss check |
| --- | --- | --- | --- | --- |
| Logistic regression early, repetition 0 | Harmful | Beneficial | Harmful | Unresolved |
| Logistic regression early, repetition 4 | Harmful | Beneficial | Harmful | Unresolved |
| Rosenbrock noise 1 early, repetition 5 | Harmful | Practical tie | Harmful | Material loss |
| Rosenbrock noise 3 late, repetition 0 | Harmful | Unresolved | Harmful | Material loss |
| Timing late, repetition 6 | Harmful | Beneficial | Unresolved | Unresolved |

The score loss is localized before refinement in both logistic-regression
cases and the early Rosenbrock-noise1 case. The Rosenbrock-noise3 score
comparison against its own winner remains unresolved, so its score-loss
stage remains unresolved; its material raw loss is already present before
refinement. Both Rosenbrock cases have independently resolved material raw
losses before refinement and after selection. The timing case's direct
comparison remains unresolved under this fresh diagnostic. There is no
confirmed harmful selected-versus-own-winner contrast. These observations
do not distinguish insufficient sieve coverage from inaccurate re-scoring,
and do not override the primary holdout or earlier MC classifications.

The [final assessment](../experiments/noisy-acquisition-efficiency/integration-search/e3_final_assessment_20260917.json)
records mixed evidence: S2's development timing ratio is 0.43387 and many
held-out choices improve, but target-specific regressions prevent a general
replacement recommendation. The ordinary-MC checks have no resolved
disagreement with the primary judge; unresolved checks remain unresolved.
No paired holdout timings or inference runs were allocated. E4 cannot use
holdout losses to establish its development entry gates, and the diagnostic
does not supply the missing multi-budget development comparisons or harmful
accepted refinements on two development trajectories.

## Exploratory E5 pilot

Following the mixed E3 result, the user authorized a focused S0-versus-S2
inference pilot on 2026-09-17. The question is whether the faster selection
rule improves complete-run efficiency without a coherent loss of inference
accuracy. S2 retains its frozen 1024-candidate sieve, MC100 coarse scores,
eight-candidate shortlist without a diversity filter, MC1600 accurate rule
and single bounded refinement. This exploratory allocation does not certify
S2 as a replacement and does not use holdout data to design a new treatment.

The agreed design contains ten fresh paired seeds, 2000-2009, on all six
configurations: 120 fits across two arms. The authorized pilot executes only
seeds 2000-2001, giving 24 fits. Its outcomes remain part of the full design.
The first batch checks execution, reproducibility and cost; favorable early
accuracy is not a condition for continuing. Remaining seeds require a
reviewed continuation allocation and explicit user authorization after the
pilot report. The
[plan amendment](../plans/noisy-acquisition-efficiency.md#approved-exploratory-s0s2-amendment-2026-09-17)
specifies the exact scope, metrics, safeguards and preparation checklist.

The twelve existing production capture fits took 33.55 minutes, including
capture instrumentation. Equal-cost extrapolation predicts 67.1 minutes
for the pilot and 335.5 minutes for the full design. Planning allowances
are 90-120 minutes and 7-10 hours, respectively, before measuring actual S2
inference costs. Adapter implementation and validation are additional.
Completed fits can be resumed by identity-checked records; interrupted fits
are not assumed to have a usable intermediate checkpoint.

The adapter, runner and tests are committed at `d8eb7af`. Validation passed
18 focused developer tests, 1,614 package tests (45 skipped), and all 11
exact numerical oracle fixtures. The three package warnings are the existing
VP-density derivative warnings in oracle cases. The existing environment
lacked the declared `threadpoolctl` dependency; installing version 3.6.0 with
`--no-deps` repaired collection without changing numerical dependencies.
Initial focused-test failures were fixture/stub errors, including a consumed
search RNG during VP reconstruction. Matching the frozen selector's fresh
post-reconstruction stream restores exact S2 parity without changing the
adapter or relaxing tolerance. Original and final validation logs are retained.

Independent Sol review verified the implementation, all 99 source/data/config
hashes, 23 gpyreg source hashes, 220 promoted reference sidecars and 18
reference envelopes. The locked pilot manifest has semantic digest
`98a0f1cd64c6b17137ed623ae7b28e0e7da207c354eb932fa094018d0a939134`.
Its executable allocation contains only the 24 pilot fits. One separately
recorded S2 replay of Rosenbrock D2/noise 1/seed 2000 checks every stored
non-timer trace array, semantic final field and selection record before the
remaining pilot fits proceed. The replay is validation overhead and contributes
no additional scientific or timing observation.
The predeclared repeat passed exact comparison with no differing field.

### Pilot outcomes

All 24 fits completed without execution failure. All 12 paired initial designs
and initial noisy observations match exactly; all reported quality metrics
are finite. No fit exceeds the frozen promoted-reference quality envelopes.
The [paired summary](../experiments/noisy-acquisition-efficiency/integration-search/e5_summary_20260917.json)
retains every observation, paired difference and interval, including adverse
outcomes. The replay is excluded from those denominators.
The [independent post-pilot review](../experiments/noisy-acquisition-efficiency/integration-search/e5_independent_review_20260917.json)
passed without remaining findings. It verified all 24 terminal digests, 72
bound payload hashes, 5,365 selection records and 2,780 independently derived
S2 accurate-rule seeds, including every frozen search-budget cap.

The table gives geometric means of the two paired S2/S0 ratios for each
configuration. Values below one indicate lower S2 time or fewer calls.
Usability applies the predeclared evidence-error, gsKL and MMTV thresholds;
the convergence flag is a separate outcome.

| Configuration | Fit time S2/S0 | Search time S2/S0 | Calls S2/S0 | Usable S0 → S2 (of 2) | Converged S0 → S2 (of 2) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rosenbrock D2, noise 1 | 0.521 | 0.313 | 0.983 | 2 → 2 | 2 → 2 |
| Rosenbrock D2, noise 3 | 0.467 | 0.337 | 1.000 | 2 → 2 | 2 → 0 |
| Logistic regression D5, noise 3 | 0.749 | 0.483 | 1.012 | 1 → 2 | 2 → 2 |
| Student-t D8, noise 3 | 1.070 | 0.782 | 1.152 | 1 → 2 | 2 → 2 |
| Multisensory subject 1 D6, noise 1.3 | 0.592 | 0.377 | 1.012 | 0 → 0 | 2 → 2 |
| Timing D5, noise 2.2 | 0.747 | 0.389 | 1.158 | 2 → 2 | 2 → 1 |

Equal-weight geometric means across the six configurations are 0.6646 for
fit time and 0.4246 for search time: descriptive reductions of 33.5% and
57.5%. These are laptop measurements on two seeds per configuration, not
quality-equivalence claims. The stored per-target timing intervals have one
degree of freedom and are generally very wide.

Usable outcomes increase from 8/12 to 10/12, with gains on logistic-regression
seed 2001 and Student-t seed 2000 and no usable-to-unusable transition.
Convergence falls from 12/12 to 9/12: both high-noise Rosenbrock S2 fits stop
at 200 evaluations, and timing seed 2001 stops at 350. All three still satisfy
the usable-quality thresholds. For timing seed 2001, S0 converges at 250 calls;
S2 therefore uses 100 additional evaluations. These are scientific outcomes
under the unchanged stopping rule and budgets, not execution failures.

Student-t uses more evaluations with S2 on both seeds (370 versus 305, and
410 versus 375), offsetting cheaper search and leaving the paired geometric
fit-time ratio above one. Multisensory remains unusable in both arms for both
seeds. Its seed-2001 S2 result is worse on all three quality metrics, including
gsKL 4.252 versus 2.068 and MMTV 0.266 versus 0.205; the other seed improves
these two posterior metrics. Low-noise Rosenbrock also has higher MMTV on
both S2 seeds, while remaining well within the usable threshold. These
target-specific concerns remain visible in any ten-seed continuation.

The pilot's summed worker time is 60.58 minutes; elapsed time from the first
worker start to the last finish is 62.11 minutes, including the 45.53-second
validation repeat and launch gaps. The observed-rate estimate for the
remaining 96 fits is 4.04 hours. Reserve approximately 5-6 hours, using a
25-50% allowance over that small-sample extrapolation. Preparation and review
are additional. The [completion record](../experiments/noisy-acquisition-efficiency/integration-search/e5_completion_20260917.json)
records the exact arithmetic and per-target worker times.

The pilot clears the amendment's operational continuation criteria through
verified execution, reproducibility and measured cost. Scientific outcomes
remain descriptive and do not serve as a favorable-results screen.
Convergence losses, target-call increases and multisensory quality remain
explicit concerns for the complete comparison. Two seeds cannot establish
noninferiority or the frequency of those outcomes. Seeds 2002-2009 are
allocated by the continuation manifest below; their execution and the
ten-seed outcomes follow it.

### Continuation preparation

The 96 remaining fits, seeds 2002-2009 on all six configurations, are
allocated by a separate continuation manifest prepared and independently
reviewed on 2026-09-17 and frozen without a launch clearance. Its semantic
digest is `c4d2c7265afb278aa2e7bdcef724ac8d0656bf78888c343730fee56e3175e335`.
The manifest binds the pilot manifest, its campaign and its four immutable
records by hash. The runner's own source hash is part of the recorded
identity, so the continuation records a provenance comparison instead of
reusing the pilot identity: all 98 other source and data hashes and the
entire environment equal the pilot's, and only the runner differs. Each
worker re-derives this comparison before fitting. The fit path itself
(worker, selection policy, golden trace) is unchanged, the default-path
oracle fixtures remain exact, and the refactored reporting regenerates the
immutable pilot summary byte-for-byte.

Execution proceeds in sequential batches in manifest order: target-major,
with arm order alternating by seed block as in the pilot. Each batch
declares a wall-time limit of at most six hours, the upper planning
allowance, and optionally a fit cap. It launches no fit when less than one
20-minute per-fit timeout remains, reuses validated success terminals,
never reruns a failed fit, and by default stops at a failure so it can be
investigated before more compute is spent. Every batch writes an
append-only record with its stop reason, including an explicit aborted
record if it is interrupted. The runner must not be modified once a
continuation fit has run; a correction needs a declared successor
allocation with explicit handling of the fits it invalidates.

The combined summary reports all ten paired seeds per configuration. Pilot
pairs are read from the pilot campaign under the pilot manifest and
continuation pairs from the continuation campaign. It gives the planned
runtime and target-call log-ratio intervals with nine degrees of freedom for
ten complete pairs, all paired accuracy differences with the seven-of-ten
review trigger applied only when all ten pairs are complete, convergence
and usability transition tables, reference-envelope exceedances, and
failures and missing fits in every denominator. It also checks the
recomputed pilot pairs against the immutable pilot summary. Equal-weight
pooled geometric means across configurations remain descriptive.

Validation passed 26 focused tests, all 11 exact oracle fixtures and the
byte-identical pilot-summary regeneration. The independent static review
found no must-fix defect; its should-fix items (abort-safe batch records,
one terminal validation pass per batch, tests for the runtime provenance
gate and for a batch leaving the pilot directory untouched, the
source-freeze rule and these records) were addressed before freezing. The
[continuation manifest](../experiments/noisy-acquisition-efficiency/integration-search/e5_continuation_manifest_20260917.json),
[preparation record](../experiments/noisy-acquisition-efficiency/integration-search/e5_continuation_preparation_20260917.json)
and [independent review](../experiments/noisy-acquisition-efficiency/integration-search/e5_continuation_review_20260917.json)
are published as redacted review copies bound by the
[continuation publication index](../experiments/noisy-acquisition-efficiency/integration-search/publication_index_e5_continuation_20260917.json).
The measured-rate estimate for the 96 fits is 4.04 hours with a 5-6-hour
planning allowance.

### Continuation execution

The user took the launch decision at approximately 19:47 UTC on 2026-09-17,
with nine hours of wall clock available, and it is recorded in the
continuation directory's `launch_clearance.json` together with the manifest
digests and the authorized batch limits. Pre-launch checks re-derived the
runtime identity and the runner-only provenance under the documented
environment, verified both manifest hashes, and found no Python process
running. Two batches then ran from the repository root with the documented
environment:

| Batch | Started (UTC) | Wall allowance | Launched | Stop reason | Wall time |
| --- | --- | ---: | ---: | --- | ---: |
| 1 | 2026-09-17 19:51:01 | 21600 s | 91 | Less than one fit timeout left | 20456 s |
| 2 | 2026-09-18 01:32:47 | 7200 s | 5 | All cells complete | 1406 s |

All 96 cells have validated success terminals; no fit failed, timed out or
was interrupted, and the second batch reused the 91 earlier terminals without
relaunching them. Summed worker time was 21661.8 s (6.02 hours), 1.49 times
the 14538 s point estimate and inside the 18173-21807 s planning range at its
upper end. Elapsed time from the first worker start to the last finish was
21909 s. The runner writes nothing to its own stdout, so the two batch logs
are empty; the batch records under `campaign/batches/` are the execution
evidence.

The machine slowed progressively during the continuation. Target-call
counts per fit were in the pilot's range, but the median seconds per target
call, continuation over pilot, ranged from 0.82 (low-noise Rosenbrock S2)
and 1.01 (its S0) through 1.2 on high-noise Rosenbrock and 1.3 to 1.6 on
logistic regression and Student-t, to 2.0 to 2.2 on the two real-data
targets, which ran last in the target-major schedule; the factor grows
roughly monotonically with position in the schedule, about 18 percent per
hour. An operator observation at 00:08 UTC, recorded in
`machine_load_observation.json` beside the manifest, found total CPU load
of 5 to 24 percent on 22 logical processors against under 5 percent for the
single-threaded worker, a reported processor clock of 1.4 GHz against a
3.8 GHz maximum, and a screensaver process with over two hours of
accumulated CPU time; the record attributes no cause. The two arms of a seed
run back-to-back, and within each target block the first arm alternates by
seed parity, so a drift of that size biases a paired ratio by about 1
percent at the median arm-start separation of three minutes and about 3
percent at the largest; the independent review found no systematic order
direction. Paired ratios therefore remain interpretable, absolute times are
not comparable with the pilot's, and the laptop was not a quiet machine in
the sense of the timing protocol. Accuracy metrics, convergence and
usability are deterministic given the seed and unaffected.

The closure record `completion.json` binds the launch clearance, both batch
records, all 96 terminal digests, the continuation-only summary (48 pairs) and
the combined ten-seed summary (60 pairs). Every runtime identity check at
worker start passed; the runner source was not modified after the first
continuation fit.

### Ten-seed outcomes

All 120 fits succeeded, all 60 paired initial designs and initial noisy
observations match exactly, all reported metrics are finite, and the twelve
pilot pairs recomputed from the pilot campaign equal the immutable pilot
summary. The [combined summary](../experiments/noisy-acquisition-efficiency/integration-search/e5_combined_summary_20260918.json)
holds every observation, paired difference and interval. Each configuration
has ten complete pairs; the intervals are two-sided 95% Student-t intervals
on the mean paired log ratio with nine degrees of freedom, exponentiated.

| Configuration | Fit time S2/S0 | Search time S2/S0 | Calls S2/S0 | Usable S0→S2 (of 10) | Converged S0→S2 (of 10) |
| --- | --- | --- | --- | --- | --- |
| Rosenbrock D2, noise 1 | 0.416 [0.356, 0.487] | 0.269 [0.229, 0.317] | 0.929 [0.869, 0.993] | 10 → 10 | 10 → 10 |
| Rosenbrock D2, noise 3 | 0.487 [0.422, 0.563] | 0.339 [0.295, 0.391] | 1.019 [0.924, 1.125] | 10 → 9 | 10 → 6 |
| Logistic regression D5, noise 3 | 0.566 [0.473, 0.677] | 0.368 [0.308, 0.440] | 0.903 [0.807, 1.010] | 9 → 10 | 10 → 10 |
| Student-t D8, noise 3 | 0.843 [0.710, 1.001] | 0.615 [0.524, 0.722] | 1.059 [0.948, 1.184] | 3 → 9 | 10 → 10 |
| Multisensory subject 1 D6, noise 1.3 | 0.593 [0.488, 0.720] | 0.384 [0.315, 0.468] | 0.964 [0.870, 1.067] | 0 → 1 | 10 → 10 |
| Timing D5, noise 2.2 | 0.599 [0.494, 0.725] | 0.347 [0.291, 0.413] | 0.972 [0.856, 1.104] | 9 → 10 | 10 → 9 |

Equal-weight geometric means across the six configurations are 0.570 for fit
time, 0.375 for search time, 0.973 for target calls and 1.095 for
target-evaluation time. The fit-time saving is the search saving:
acquisition search is 57 to 77 percent of an S0 fit on these targets, and
the wall ratio predicted from the search ratio alone matches the observed
one within 0.02 to 0.06 on every configuration. Two intervals deserve
mention. Student-t's fit-time interval includes 1, so its saving is not
distinguished from zero at the predeclared level, and its call count is
about 6 percent higher with an interval that includes 1; low-noise
Rosenbrock uses about 7 percent fewer calls with an interval that excludes
1. Target-evaluation time is below 0.2 percent of the fit on five targets,
so its ratio there is timer granularity; on the timing model, where it is
5 to 8 percent of the fit, the ratio is 0.894 [0.757, 1.057]. These are
laptop measurements on cheap likelihoods: on a likelihood expensive enough
for target evaluation to dominate the run, the fit-time ratio moves toward
1 and the headline saving shrinks to the search share.

Paired accuracy differences, S2 minus S0, with the median, the range and the
number of pairs in which S2 is worse. An asterisk marks the predeclared
seven-of-ten review trigger (worse median with at least seven of ten pairs
worse).

| Configuration | Evidence error | gsKL | MMTV |
| --- | --- | --- | --- |
| Rosenbrock D2, noise 1 | −0.034 [−0.161, 0.127], 4 | +0.004 [−0.427, 0.218], 6 | +0.010 [−0.052, 0.069], 6 |
| Rosenbrock D2, noise 3 | +0.044 [−0.177, 0.339], 7* | +0.152 [−0.328, 1.543], 6 | +0.003 [−0.061, 0.176], 5 |
| Logistic regression D5, noise 3 | −0.111 [−0.866, 0.367], 4 | +0.079 [−0.445, 0.275], 8* | +0.035 [−0.148, 0.084], 8* |
| Student-t D8, noise 3 | −0.612 [−1.586, 0.557], 2 | −0.113 [−0.978, 0.177], 2 | −0.033 [−0.073, 0.022], 2 |
| Multisensory subject 1 D6, noise 1.3 | +0.098 [−0.547, 0.402], 7* | +0.050 [−1.016, 2.184], 5 | −0.005 [−0.034, 0.061], 4 |
| Timing D5, noise 2.2 | −0.036 [−0.613, 0.230], 4 | +0.020 [−0.740, 0.505], 5 | −0.005 [−0.126, 0.085], 4 |

Usable fits (evidence error below 1, gsKL below 1, MMTV below 0.2) rise from
41 of 60 under S0 to 49 of 60 under S2: nine usable gains against one loss.
Six gains are on Student-t and one each on logistic regression, multisensory
and timing; the multisensory gain (seed 2009) is marginal, crossing the gsKL
and MMTV thresholds narrowly while its evidence error worsens from 0.22 to
0.52, on a configuration that otherwise fails in both arms. The loss is
high-noise Rosenbrock seed 2008, where S2 stops at the 200-call budget with
gsKL 1.56 and MMTV 0.22 while S0 converges at 195 calls with gsKL 0.015 and
MMTV 0.048. Convergence is the sharpest asymmetry in the design: all 60 S0
fits converge, 55 of 60 S2 fits do, and all five non-converged S2 fits
(high-noise Rosenbrock seeds 2000, 2001, 2004 and 2008, timing seed 2001)
stop at their evaluation budget. A five-to-nothing discordance has two-sided
probability 0.0625 under a symmetric null, tighter than any accuracy
comparison in the campaign. Three of the five were already in the pilot; the
continuation added two. Four fits exceed the promoted reference envelope on
gsKL: two S0 fits (low-noise Rosenbrock seed 2007 at three times the
envelope, timing seed 2006) and two S2 fits (low-noise Rosenbrock seeds 2008
and 2009), all still usable except the timing S0 fit. The envelope check is
insensitive where the concern lies: high-noise Rosenbrock's gsKL envelope is
2.02, so the usability-loss fit at 1.56 does not trip it. There is no new
failure and no nonfinite metric.

The seven-of-ten triggers are descriptive gates, not tests, and their count
carries little weight on its own: under a symmetric null the probability of
at least seven worse in ten is 0.17 per target and metric, so about 3.1 of
the 18 combinations would trigger by chance against the 4 observed, and no
combination reaches a two-sided sign-test probability below 0.11. What is
informative is the clustering, the effect sizes and the binary outcomes. On
high-noise Rosenbrock, gsKL is worse in six of ten pairs with within-pair
increases of 3.5 to 103 times: the four converged S2 fits on seeds 2002,
2003, 2006 and 2007 have gsKL between 0.41 and 0.58 against 0.03 to 0.13 for
S0, and the usability-loss fit has 1.56 against 0.015; S2 spends its full
200-call budget on four seeds and still ends worse. This is a
continuation-seed phenomenon: both pilot seeds favoured S2 on all three
metrics, which shows how unstable such a target-level verdict is at ten
seeds. On logistic regression S2 improves the evidence error in six of ten
pairs while gsKL and MMTV are worse in eight of ten, with two S2 fits at MMTV
0.19 against the 0.2 usability threshold. On Student-t S2 is better on all
three metrics in eight of ten pairs and turns six of its seven S0
evidence-error failures into usable fits, with about 6 percent more target
calls and a fit-time interval that includes 1. Multisensory remains unusable
in both arms in nine of ten seeds, with gsKL between 0.9 and 4.5; its
evidence-error trigger fires inside that failing regime.

The frozen-state acquisition judge predicted none of this, in either
direction. S2's worst holdout trajectory was low-noise Rosenbrock (8 harmful
of 16, seven material raw losses), which in inference shows no trigger, no
convergence or usability loss and the largest speedup; its near-clean holdout
trajectory, high-noise Rosenbrock (1 harmful of 16), is the one that
degraded. Selection quality judged on frozen states is therefore not a usable
screen for a search change on its own, which bears on how the E2 to E4
evidence is weighted and on the design of any follow-up.

The plan's decision gate recommends continuation only if the speed benefit
survives inference with no coherent target-specific accuracy loss and no
unexplained failure mechanism. The speed benefit survives. The second
condition is a precondition to be established, and ten paired seeds neither
establish nor exclude it: S2 is not shown to be free of target-specific
accuracy loss. The convergence asymmetry, the usability loss against nine
gains, the gsKL increases on high-noise Rosenbrock and the opposite effects
on Student-t and logistic regression are unexplained, so the ten-seed
comparison does not support adopting S2 as a replacement default. The
evidence does not demonstrate a loss either; it is descriptive and
establishes neither noninferiority nor the frequency of the observed
outcomes. The [plan](../plans/noisy-acquisition-efficiency.md#e6-assessment-2026-09-18)
records the E6 assessment, which takes the form of collecting narrowly
identified missing evidence under new declared allocations. The
[independent results review](../experiments/noisy-acquisition-efficiency/integration-search/e5_results_review_20260918.json)
verified every provenance, batch and summary quantity above and supplied the
sign-test, drift-bias and envelope observations.

## Resuming the saved experiment

Run from the existing `pyvbmc-stage3` checkout on
`dev-noisy-acquisition-efficiency`, reusing its `.venv`.
No background job or session needs reattachment.
The ignored `dev/scripts/runs` junction points to the shared local artifact
directory in the sibling checkout, `../pyvbmc/dev/scripts/runs`.
The `noisy_acq_efficiency_20260916/` results, frozen dependency at
`svbmc_pool_20260913/gpyreg_1.2.1/`, and `.venv` are local by design and
are not included in the Git push. They remain available on this laptop;
moving machines requires transferring these artifacts and matching the
recorded environment. The capture and panel runs do not need to be repeated.
Published JSON records redact user-directory paths. Execute with the
original manifests under the local artifact directory, whose hashes remain
recorded in the published summaries. Historical local checkpoint IDs are
retained on the local-only `local/noisy-acquisition-pre-redaction-20260917`
branch; the public feature branch consolidates the same experiment sources.
Use:

```powershell
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYVBMC_GPYREG_SOURCE = (Resolve-Path 'dev/scripts/runs/svbmc_pool_20260913/gpyreg_1.2.1').Path
$env:PYTHONPATH = "$PWD;$env:PYVBMC_GPYREG_SOURCE;$PWD\dev\scripts"
$runRoot = 'dev/scripts/runs/noisy_acq_efficiency_20260916'
```

E2 is complete. Its final local evidence is `panel_evidence.json`,
`full_cost_screen.json` and `continuation_20260917_final.json`. The panel
judge and MC ladders are bound by their respective `completed_ladder.json`
files; completed pilots and cells must not be repeated for cost measurement.

E3's [frozen selection](../experiments/noisy-acquisition-efficiency/integration-search/search_selection_20260917.json)
uses ordinary MC1600. The [development manifest](../experiments/noisy-acquisition-efficiency/integration-search/search_manifest_20260917.json)
contains 384 paired selection cells: twelve states, eight repetitions and
S0-S3. The [pilot allocation](../experiments/noisy-acquisition-efficiency/integration-search/search_pilot_allocation_20260917.json)
authorized twelve repetition-zero cells on three development states; its
completed successor covered the other nine states. A further allocation
completed repetitions 1-7. All 384 cells are saved and must be reused.
Execution uses the original local `search_manifest.json`, with outputs in
`search_development/` and scheduling/diagnostic records in `search_pilot/`.

All development selections, judging, timing and memory measurements are
complete. Their terminal records bind `search_manifest.json`,
`search_remaining/selection_completion_snapshot.json`, the summaries under
`search_judging/`, and the separate `search_measurements_v2/manifest.json`.
Do not repeat them. An individual completed selection tag can be validated
and reused with:

```powershell
.venv/Scripts/python.exe dev/scripts/noisy_acq_search_experiment.py cell --manifest "$runRoot/search_manifest.json" --out "$runRoot/search_development" --tag $cellTag
```

Set `$cellTag` from the frozen manifest. The command reuses a valid completed
terminal and refuses incompatible artifacts. The S2 holdout is complete
under `search_holdout_manifest.json`, with outputs in
`search_holdout/` and scheduling records in `search_holdout_selection/`.
Its development gate, frozen selection and reviewed locked draft remain
immutable. Do not use the development manifest for holdout cells, repeat
completed cells, or overwrite an existing judge summary. Development and
holdout MC checks and the saved-point loss diagnostic are complete;
their `completed_ladder.json` records bind every executed budget.
Do not use the unbounded `run` controller: it also launches the structurally
declared timing and memory cells that this holdout does not allocate.
E4's entry gates are not established. E5 is complete in both phases;
E3 artifacts remain immutable. The pilot's raw
manifest, launch clearance and validation logs are in
`noisy_acq_efficiency_20260916/e5_pilot_20260917/`; fit artifacts are under
`campaign/`, with the excluded repeat under `campaign/validation_replay/`.
The immutable `summary.json` and `completion.json` bind the 24-fit result.
The continuation directory
`noisy_acq_efficiency_20260916/e5_continuation_20260917/` holds the frozen
`manifest.json` (semantic digest
`c4d2c7265afb278aa2e7bdcef724ac8d0656bf78888c343730fee56e3175e335`, raw
SHA-256 `cb4bfb70a0c5c368da3c3e06bc440243a3d30e37bfbbd4eb7545ccb8dc3a5228`),
the user's `launch_clearance.json`, the two batch logs, and under
`campaign/` the 96 fit artifacts, terminal and selection records, the two
batch records, `summary.json` (continuation only) and
`summary_combined.json` (all ten seeds); `completion.json` beside the
manifest binds them, and `validation/close_continuation.py` is the script
that wrote it. All 96 cells have success terminals, so the commands below
are complete and need not be repeated: `run-continuation` reuses every
terminal and exits 0, and both summary commands refuse to overwrite their
outputs (an interim continuation-only summary needs a fresh `--report`
path). Do not modify `noisy_acq_inference.py` against these artifacts: its
hash is part of the manifest identity, and a correction would need a
declared successor allocation.

```powershell
$cont = "$runRoot/e5_continuation_20260917"
.venv/Scripts/python.exe -u dev/scripts/noisy_acq_inference.py run-continuation --manifest "$cont/manifest.json" --out "$cont/campaign" --timeout 1200 --max-wall-seconds 21600
.venv/Scripts/python.exe dev/scripts/noisy_acq_inference.py summary --manifest "$cont/manifest.json" --out "$cont/campaign"
.venv/Scripts/python.exe dev/scripts/noisy_acq_inference.py summary-combined --manifest "$cont/manifest.json" --out "$cont/campaign"
```

`--max-wall-seconds` is required and capped at 21600; `--max-fits N` bounds
a batch further, and `--on-failure continue` lets a batch proceed past a
recorded failure after it has been investigated. Exit status 0 means every
continuation cell has a success terminal, 1 that a failure was encountered
or skipped, and 2 that the batch ended cleanly with cells still missing.
The first batch ran with the 21600-second allowance and stopped at its
launch cutoff with five cells left; the second ran with 7200 seconds and
completed them.
The [E5 publication index](../experiments/noisy-acquisition-efficiency/integration-search/publication_index_e5_20260917.json)
binds the raw manifest, launch clearance, replay, paired summary, completion
record and independent review to their redacted review copies.

The [publication index](../experiments/noisy-acquisition-efficiency/integration-search/publication_index_20260917.json)
maps raw local artifact hashes to the redacted review copies and specifies
the line-ending convention for checking the published bytes.
The [extended-window index](../experiments/noisy-acquisition-efficiency/integration-search/publication_index_20260917T1124Z.json)
adds the completed E3 development evidence, reporting-adapter transition,
frozen finalist decision and reviewed locked holdout draft.
The [holdout and diagnostic index](../experiments/noisy-acquisition-efficiency/integration-search/publication_index_20260917T1425Z.json)
adds holdout execution, independent MC checks, the loss investigation and
the final scientific assessment.
The [compute closure](../experiments/noisy-acquisition-efficiency/integration-search/search_compute_closure_20260917.json)
and its [final successor](../experiments/noisy-acquisition-efficiency/integration-search/search_compute_closure_20260917_final.json)
bind the archived orchestration scripts, completed evidence and local
inventory. The final experiment-process check found no numerical workers.
The [cleanup successor](../experiments/noisy-acquisition-efficiency/integration-search/search_compute_process_cleanup_20260917.json)
records removal of owned formatter processes and the final empty check for
experiment or owned formatter workers. Exact orchestration scripts were
archived before their workspace scratch copies were removed.
