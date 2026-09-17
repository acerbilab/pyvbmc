# Noisy-acquisition integration and search experiment

**Status:** E2 completed during the bounded 2026-09-17 continuation.
All four retained full-sieve integration settings failed the necessary
cost gate. E3's prescribed MC1600 fallback has 48 saved selections: all four
arms at repetition zero on twelve development states. Another 336
selections, paired timing and independent judging remain. No treatment or
production-default recommendation has been established.

This experiment evaluates the cost and selection quality of positive-weight
integration rules and smaller candidate searches for standard VIQR. The
[experimental plan](../plans/noisy-acquisition-efficiency.md) specifies the
allocation, independent judging, timing protocol and promotion criteria.
GP fitting and Bayesian quadrature are outside its scope.

## Sources and environment

The numerical baseline is PyVBMC commit
`9cc6882768ff682ae892a6b453c42e0f2d03d5fa`. The capture harness is committed
as `3fe616c477130625bb01d30d29785773fe2e0e66`. Production `pyvbmc/` source
is unchanged by these experiments.

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
Search development remains incomplete.
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
repetition-zero cells succeeded; the other 336 selections, timing, memory,
judging and holdout remain unallocated. Numerical work stopped after these
48 cells, before the soft launch cutoff.
The [combined repetition-zero outcome](../experiments/noisy-acquisition-efficiency/integration-search/search_rep0_outcome_20260917.json)
records 104.51 seconds of total outer worker time, including 78.94 seconds
for the additional 36 cells. Six selections used the prescribed row-budget
fallback. Every cell respected the 50-iteration and 1000-row limits.

The remaining **336 selections alone** are estimated at **12.2 minutes
predicted, 30.1 minutes conservative**, using each state/arm's measured
repetition-zero cost for its seven remaining repetitions. These are
scheduling allowances, not statistical bounds or paired performance
measurements. The 36 paired timing cells, 72 memory/allocation cells and
independent judging/escalation require additional estimates and allocation.
No search quality or speedup conclusion is available. The final process
check found no Python processes; no job needs reattachment.

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
completed successor covered the other nine states. All 48 repetition-zero
cells are saved and must be reused.
Execution uses the original local `search_manifest.json`, with outputs in
`search_development/` and scheduling/diagnostic records in `search_pilot/`.

Before resuming, validate completed terminal records against that manifest,
inspect `search_rep0_outcome.json` and allocate repetitions 1–7 from the
measured costs and available window. Keep the remaining scheduled cells visible;
unlaunched cells do not supply quality evidence. An individual approved tag
can be executed with:

```powershell
.venv/Scripts/python.exe dev/scripts/noisy_acq_search_experiment.py cell --manifest "$runRoot/search_manifest.json" --out "$runRoot/search_development" --tag $cellTag
```

Set `$cellTag` from the frozen manifest's approved subset. If the tag
already has a valid terminal record, the command reuses it; otherwise run
it only after a separate bounded allocation. Do not use the unbounded
`run` controller in a short window: it also launches all timing, memory and
initial judging work. E3 quality judging must include S0 comparisons and
the prescribed component contrasts before selecting a finalist. E4 remains
conditional on its original entry gates; E5 remains unapproved.

The [publication index](../experiments/noisy-acquisition-efficiency/integration-search/publication_index_20260917.json)
maps raw local artifact hashes to the redacted review copies and specifies
the line-ending convention for checking the published bytes.
