# Machine-local calibration integrated into PyVBMC

Created/updated: 2026-09-09, roadmap pickup 12.
Status: **IMPLEMENTED — local verification complete** (PI: go ahead, use task skill).
Implementation branch: `dev-machine-calibration`; planning was on `dev-next`.
Release scope: **include in PyVBMC 1.5**, confirmed by the PI.
Astra orchestrates design and integration; Sol implements and reviews.
Plan review: independent Sol static double-check complete; no findings remain.
Review clarified continuation ordering, load overrides and the exact timing
aggregation/held-out/control rules.

## Live execution checklist

- [x] Preserve a prechange source snapshot and capture oracle/five-case baselines.
- [x] Phase 1: immutable profile, VP state and shared PDF/entropy budget plumbing (Sol).
- [x] Phase 2: explicit campaign/cache/API and focused tests (Sol; main integration fixes).
- [x] Phase 3: lifecycle/options/docs verified; full docs, wheel/sdist and isolated install pass.
- [x] Phase 4: default/nondefault replays, two corrected campaigns and full suite complete.
- [x] Independent Sol cross-component double-check; all findings resolved and checked.
- [x] Finalize package/full-suite evidence and durable status records.
- [x] Apply PI-approved user-facing wording; completion explicitly names the saved JSON path.
- [x] Commit/push all work and verify feature-branch CI (PI-requested delivery).
- [x] Reduce `dev/TODO.md` to current pickup/actions; keep history in the existing plans/reports.
- [~] Merge into `dev-next`, push, and delete the merged feature branch locally/remotely.
- [ ] Run the full three-OS/three-Python CI matrix on `dev-next` and record its outcome.

Delivery: feature commit `f2f99e3` passed [branch CI](https://github.com/acerbilab/pyvbmc/actions/runs/34397131792)
(1,215 passed, 60 skipped). Repository hooks passed; formatting changes preserve
Python syntax trees, and historical measurement runners are excluded from Black
to retain their recorded source. TODO now contains current pickup/actions only;
the prior accumulated handoffs remain in git history and linked durable records.

UX follow-up (PI): routine messages use plain-language outcomes and the actual
absolute saved-report path, including cache-root overrides. Detailed budgets
and kernel timings remain in the profile/report. The Quickstart introduces
optional calibration in two sentences with a link to the guide, which also
explains Jupyter reuse. All 83 calibration checks, 11 existing seeded-run
checks and the Sphinx build pass. Sol's focused wording review has no
remaining findings. This follow-up changes presentation and documentation;
the earlier measured campaign artifacts retain their original wording.

Baseline evidence: `dev/scripts/runs/calibration_integration_20260909/` holds
the 1,140-file tracked-source snapshot at `f7f0ce1`, eight oracle output sets,
and five completed seed-0 replays (3.5 minutes, one BLAS thread). Prechange
normal_D5 has gsKL 0.000491 versus the old population fence 0.000429; the
baseline report therefore exits 1 despite all fits completing. Use these
fresh traces for exact default integration comparisons; do not attribute
this pre-existing historical-envelope discrepancy to calibration.
The snapshot also passes `make_oracle_fixtures.py --check --exact`: all
11 fixtures, including the three GP fit-history fixtures.
Integration gates so far: the shared default kernels also pass all 11 exact
oracle fixtures; all five default-profile replays match the snapshot's stored
loops, initial designs and final results exactly. Recorded replay wall times
are not a paired performance comparison.
The 67 focused calibration checks passed, including real process locks. Two
preliminary campaigns took 36.80/46.78 s and retained all three defaults;
both persisted successfully, but their timings are excluded after independent
review found discovery-control and held-out pair-order imbalances. Corrected
campaigns took 42.39/43.53 s including persistence, retaining all three defaults.
Cold cached lookup including lazy imports took 0.141 s; twelve subsequent
lookups had median 0.00633 s. The first real
probe exposed coarse Windows `monotonic()` durations: kernel timing now uses
`perf_counter()` independently of the monotonic watchdog, with regression tests.
No partial/failed probe was persisted. Fourteen focused lifecycle tests pass,
including whitening/undo and accepted/rejected/skipped boost. Sphinx builds
the full documentation including examples and options (one pre-existing Plotly
notebook MIME warning).
Measured incremental allocation peaks are below 26 MiB; review corrected
the analytical memory estimate to include overlapping temporaries, reduction
workspace and allocation metadata. The campaign remains below its 64 MiB
estimate threshold for every workload/budget. Cross-component Sol reviewers inspected
code they did not write; final static review has no remaining findings. Cache
validation now reconciles complete coverage, numerical/RNG validity and held-out
acceptance with the final selected settings. All 82 calibration tests pass; 93 calibration/seed checks pass after the final
manual-profile provenance correction. The full package suite passed 1,140 tests
with 35 skips (337.94 s, no reruns). Wheel/sdist and isolated wheel import/PDF/
entropy checks pass. Packaging now excludes test files from the runtime wheel
while retaining them in the sdist, as the repository documents.
The session's agent-thread limit prevented an additional fresh-context reviewer.
Retained integration evidence lives in
`dev/experiments/machine_calibration/integrated/`; the result report distinguishes
preliminary exclusions, paired controls and the pre-existing replay flag.

Build a user-facing calibration facility shipped with PyVBMC. It measures
PDF and Monte Carlo entropy chunk budgets on the user's machine, stores a
reusable profile, and applies fixed settings to subsequent runs. This file
owns the integration design; the completed developer experiment is evidence,
not the finished feature. Package implementation is locally verified. Other-platform CI and another
CPU/stack remain release follow-ups; neither was available/run locally.
This durable plan remains the task record; no transient TASK file was created.

## User decisions and public workflow

Settled with the PI: this belongs in the package; choices stay fixed per
run; about 30 seconds for an occasional machine/configuration campaign is
acceptable as an indicative duration, not a time limit; users receive feedback
and can initiate or rerun the campaign. Complete the prescribed campaign even
when it takes longer than the estimate. Stop for excessive runtime only at a
generous watchdog threshold (initial proposal: five minutes).
The recommended workflow implements these decisions as follows:

```python
profile = pyvbmc.calibrate()                 # run a new ~30 s campaign
# Call pyvbmc.calibrate() again whenever you want to recalibrate.
vbmc = pyvbmc.VBMC(..., options={"performance_calibration": profile})
# Normal VBMC construction can omit the option: use a valid cached profile.
# options={"performance_calibration": "off"} pins historical defaults.
```

Public signature: `calibrate(*, verbose=True)`.
It returns an immutable `CalibrationProfile` with three settings, provenance
and completion/source status. Every call requests new measurements, even when
a compatible cache record exists. There is no `force` flag: normal optimization
already handles cache reuse. A busy or failed campaign reports that outcome
explicitly, including any fallback to a previous profile.
The report and cache path are inspectable; detailed timings live in the cache
report, not in every posterior/history copy. A small profile summary remains
available even when persistence fails. Invalid arguments raise normally.

Only this explicit API runs the campaign. Import, constructors, optimization,
PDF/entropy calls and save/load never initiate timing work. Normal VBMC uses
`performance_calibration="cached"`: read a compatible profile or use 65,536
defaults. Missing/stale records produce a short suggestion to call
`pyvbmc.calibrate()`, at most once per process/configuration, respecting
`display="off"`. A normal startup summary identifies cached, explicit or
default settings. This replaces the earlier automatic first-use proposal;
there is no surprise 30-second delay after an upgrade or cache deletion.

Explicit calibration shows an initial time estimate, progress through setup,
PDF, entropy and validation, then elapsed time, a plain-language outcome and
the actual saved-report path or unsaved status. Detailed budgets and held-out
measurements remain inspectable in the profile/report.
Use plain log lines suitable for notebooks and terminals, not a required
progress-bar dependency. `verbose=False` suppresses routine text; returned
status remains inspectable. Report timeout, numerical rejection, busy lock
and unwritable cache plainly. Catch expected measurement/resource failures;
do not hide programming errors. Ctrl-C cleans up and propagates, preserving
the previous cache. Recalibration never modifies a resolved run.

## Settings and run lifetime

Recommend three positive integer element budgets: `pdf_chunk_elements`,
`entropy_grad_chunk_elements`, `entropy_value_chunk_elements`. Each defaults
to `2**16`. PDF uses one setting for values and gradients; entropy chooses
between its two fixed settings by whether any gradient is requested. Shape
changes still determine block sizes from those constants, as today; they
do not trigger timing or change the constants. Sample counts, estimator,
optimizer tolerances, float64 dtype and RNG algorithms stay unchanged.

One authoritative immutable profile travels with each VP; VBMC exposes it
through its VP and includes a compact snapshot in results. For a new VBMC,
resolve explicit profile, compatible process/cache record, or defaults once
at `optimize()` entry before kernel work. On completed-run continuation,
restore the selected historical VP first, then resolve only if that VP is
unresolved; a resolved historical profile takes precedence without cache
lookup. If the user calls a kernel on `vbmc.vp` earlier, resolve there instead
and preserve that choice at optimize entry. The VP carries the requested
mode until resolution, so `off` and explicit settings work in that case.
Standalone VPs default to historical settings; add a trailing keyword
`calibration=None` accepting an explicit profile or `"cached"` to opt in.
No repeated filesystem/environment reads in numerical hot paths.

Deepcopy, new sieve candidates, pruning, whitening, undo, final boost and
returned posteriors preserve the profile. Save/load preserves it even on a
different machine or after recalibration. Loading an earlier iteration uses
that recorded VP's profile. For a resolved save, no override preserves it;
an explicit profile with the same supported schema and three budget values
is allowed but retains the saved provenance; `off` is allowed only when all
three saved budgets are historical defaults. Reject explicit `cached` and
unequal profile overrides without reading the cache. A new run is needed for
new settings. For an unresolved pre-run save, a load-time mode/profile override
may replace the pending request without resolving it. Test this complete
resolved/unresolved by cached/off/explicit-profile matrix.
Legacy saves missing the attribute receive the historical default profile,
including historical VPs. New saves made before resolution retain their
explicit unresolved state, then resolve cheaply at first use. Save/load
itself never resolves from cache or runs calibration.
Machine compatibility controls automatic cache reuse, not the validity of an
explicitly supplied or restored profile. Those keep their recorded settings
across hosts; validate their schema/settings without replacing them.

Entropy chunk sizes can change floating-point addition order. Check tight
numerical agreement and identical RNG advancement, record the fixed settings,
and document that identical seeds across different profiles are not a claim
of identical trajectories. Same-profile reproducibility under the same
numerical environment remains required. No additional user policy decision
is needed for this ordinary reproducibility contract.

## Storage, compatibility and concurrency

Use `platformdirs.user_cache_dir("pyvbmc", appauthor=False, opinion=False)`
plus `calibration/v1/<fingerprint>.json`. On Windows this is normally
`%LOCALAPPDATA%\pyvbmc\calibration\v1\`; macOS uses `~/Library/Caches/pyvbmc/`
and Linux uses the XDG cache root. A single `PYVBMC_CACHE_DIR` override replaces
the application cache root for notebooks, containers and tests; changing it
cannot affect an already resolved run. Keep data outside source/install trees.

Fingerprint schema/workload/kernel revision, CPU model and architecture,
hashed host identity (shared home directories), OS, Python major/minor,
NumPy/SciPy versions, numerical backend/version/architecture, reported native
thread limits and relevant thread/SIMD environment, CPU device and float64.
Record missing identity fields explicitly; insufficient machine/backend
identity disables persistent reuse and reports why. Recompute compatibility
for new runs, never during a run. Record the PyVBMC version for provenance;
invalidate on explicit kernel/recipe revision rather than every docs commit.
No expiry by elapsed age alone. A changed environment requires an explicit
campaign; optimization uses defaults meanwhile.

JSON separates settings/identity/status from bounded measurement data. Validate
schema, supported candidate integers (reject bool), finite numbers, required
fields, fingerprint and a bounded file size before reuse. A complete campaign
that retains defaults is a valid reusable result. Write a temporary file in
the same directory and atomically replace only a fully validated record.
Malformed files are cache misses, never executable data. Failed or interrupted
recalibration preserves the previous valid record. An incomplete
campaign returns the previous compatible profile or defaults with an explicit
status; do not persist a partial winner as a successful calibration.

Use a nonblocking OS-backed `filelock` lock per machine campaign in the cache
root (not per numerical fingerprint, since different thread configurations
still contend for the same CPU). A second explicit caller reports busy and
returns a valid existing profile or defaults, without launching another sweep.
Normal optimization reads only complete records and never waits for calibration.
Use native lock backends on the supported three OSes; no blind stale-lock
deletion. For an unwritable cache, explicit calibration can run in memory,
with a process-local guard and a notice that cross-process exclusion/persistence
is unavailable. The caller remains responsible for avoiding other heavy work.
Successful profiles are reusable in that process and may be passed explicitly.

Recommend small runtime dependencies `platformdirs`, `threadpoolctl` and
`filelock`, with Python 3.10-compatible bounds verified during implementation.
They handle [OS cache paths](https://platformdirs.readthedocs.io/en/latest/api.html),
[native thread-limit inspection](https://github.com/joblib/threadpoolctl) and
[process locks](https://py-filelock.readthedocs.io/en/latest/).
Inspect thread settings without changing them. No background workers, GPU,
user-target calls, network access or BLAS thread changes in the campaign.

## What the shipped campaign checks

Use the same production numerical helpers with an explicit private budget
argument. Replace the prototype's AST/global-dictionary clones; do not mutate
module globals or duplicate numerical formulas. Create fixed synthetic
float64 inputs and private seeded generators independent of inference RNGs.

Start with candidates `2**14, 2**15, 2**16, 2**17, 2**18` and these regimes:

| Group | Synthetic workloads |
| --- | --- |
| PDF | Small 8-row calls; D4/K20/N8192 sieve, values and gradients; D15/K26/N100000 density |
| Entropy with gradients | D4/K20/Ns37; D4/K50/Ns55 boost; D15/K50/Ns55; D4/K20/Ns200 active scoring |
| Entropy values | D4/K20/Ns4096 and D15/K26/Ns4096 fine scoring |

Specify the small PDF shape as D4/K20. Entropy Ns is per component and keeps
the current even-number rounding. These cases cover regimes, not measured
solver call-frequency weights. Verify PDF gradients' performance before
selecting a shared PDF setting; the prior timing experiment measured values.

Warm candidates; use balanced interleaved timing cycles including the default,
rotate every candidate through each position, and reserve independent held-out
timings. Initial recipe: five discovery rounds and four paired held-out rounds,
with reversed pair order and same-budget controls. Define aligned rounds
within each group: `q[c,w,r] = t_default[w,r] / t_candidate[c,w,r]`, then
`G[c,r] = exp(mean_w(log(q[c,w,r])))`. The group mean includes equally weighted
workloads whose effective chunk partitions differ from the default for that
candidate. Detect equivalent layouts deterministically from block partitions,
collapse their timing arms, and keep them in numerical coverage/control
reporting; they cannot contribute evidence of speedup. If none differ, retain
the default. For R complete rounds require `median_r(G[c,r]) >= 1.10`, at
least `ceil(0.8*R)` rounds with `G[c,r] > 1`, and for every affected workload
`median_r(q[c,w,r]) >= 1/1.05` (at most 5% slowdown). Select the passing
candidate with highest median G, breaking exact ties toward smaller budgets;
if none passes, retain the default. Apply precisely the same formula and
gates to independent held-out rounds for the selected candidate. Compute
same-budget paired controls over the same affected workloads; require their
median group ratio in `[1/1.05, 1.05]` in both stages, otherwise retain defaults
for that group. This specifies the observation population rather than pooling
different shapes' raw ratios. Report per-case timings, uncertainty/spread and
held-out savings; never turn noise into amortization.

Numerical gates precede acceptance: default helper exactly matches existing
production results; PDF values and input gradients are exact across chunks;
entropy values/gradients agree with common random numbers at initial
`rtol=1e-12, atol=1e-12` (verify against the broader existing gates, do not
silently relax failures). Cover partial component/sample/row blocks,
log density, transformed coordinates, entropy gradient blocks and both
Jacobian modes. All raw outputs stay float64; inputs and VP parameters are
unchanged; private entropy RNG advancement is identical for all budgets;
VP/global RNG state is untouched. Calibration states use explicit defaults
to prevent recursive profile resolution. Candidate failures disqualify that
candidate; a failed baseline or incomplete required coverage invalidates the
campaign. Keep larger finite-difference matrices in developer tests.

Roughly 30 seconds is an indicative runtime based on initial measurements,
not a deadline or a selection budget. Run all prescribed setup, warmup,
discovery, numerical checks and held-out validation to completion. Passing
30 seconds must not reduce rounds, omit cases, reject a winner or return
defaults. Measure setup/first-call/steady timings separately and keep progress
visible when the campaign takes longer than estimated.

Use a generous watchdog only for unexpectedly excessive runtime: propose
five minutes elapsed from API entry for this recipe. Check elapsed monotonic
time between kernel/diagnostic calls, stopping further work only once that
threshold is reached, rather than skipping work based on projected duration.
A running NumPy call is allowed to return; this does not add worker processes
or claim hard interruption of a stuck native call. Report a watchdog stop as
incomplete and preserve the previous valid cache. The watchdog is an internal
guard against runaway work, separate from the displayed runtime estimate;
validate its scale across machines rather than treating the PI's example as
an exact universal cutoff. No public 30-second `time_budget` option is needed.

Bound synthetic allocations separately from the chunk element count. Start
with a conservative 64 MiB estimate of incremental live NumPy storage for
each case, including entropy's full draw arrays; validate estimates against
allocator traces during development. This is neither process RSS nor a hard
memory guarantee. If required cases cannot fit the memory cap, or the generous
watchdog stops the campaign, return an incomplete/default result. The existing
23.74-second complete experiment supports an estimate in the tens of seconds,
not a completion limit or portable promise. Do not require a cheaper
proxy campaign merely to keep first use under two seconds.

## Implementation phases and acceptance

All phases below are approved. No planning branch was created. Source work
uses the existing calibration feature branch.
Astra runs heavy checks sequentially; delegated Sol review is read-only.

Before Phase 1 source edits, Astra preserves an immutable local copy of current
tracked source, then records current default oracle outputs and
a bounded five-config/seed-0 replay into a fresh local directory using current
that source snapshot while Sol can implement in the main working tree. This
creates the comparison baseline without requiring old ignored
artifacts. Use `normal_D5,banana_D2,halfnormal_D2,cigar_D4,rosenbrock_D2_noise1`;
preserve the command, source revision and thread environment with the output.

### 1. Profile plumbing and unchanged defaults — Sol implementation

1. Read `dev/2026-09-02-modernization-discussion.md` before numerical edits.
   Add `pyvbmc/calibration/` with `profile.py` and private helpers, keeping
   imports independent of VBMC/VP to avoid eager-import cycles.
2. In `variational_posterior/variational_posterior.py`, replace the PDF literal
   with an explicit budget passed to shared computation; in
   `entropy/entmc_vbmc.py`, replace `_MAX_TENSOR_ELEMENTS` selection with the
   fixed profile's gradient/value budget. Preserve existing public entropy
   signature and exact default arithmetic. Do not introduce source cloning.
3. Add profile validation/serialization and backward-compatible VP attributes.
   Cover integer validation, deepcopy, standalone saves and old fixture loads.
   Adapt existing entropy tests that monkeypatch the global to explicit budgets.

Acceptance: default-budget existing numerical/finite-difference tests and exact
oracles pass with calibration explicitly off; nondefault checks meet the
numerical/RNG contract. If default arithmetic moves, investigate the helper
extraction; do not rebaseline references to make it pass.

### 2. Explicit campaign and durable cache — Sol implementation

1. Add private `_campaign.py`/`_cache.py` under `pyvbmc/calibration/`, adapting
   the validated experiment's timing/diagnostic logic to shared kernels and
   three aggregate settings. Expose `calibrate` and `CalibrationProfile`
   through `pyvbmc/__init__.py` without cache I/O or timing on import.
2. Implement the fingerprint, JSON schema, atomic writes, lock, in-memory
   fallback, progress and final report above. Add dependencies in
   `pyproject.toml`; verify compatible versions rather than guessing minima.
3. Add `pyvbmc/testing/calibration/__init__.py` and focused tests for obvious
   timing winners, noisy ties, regression vetoes, held-out rejection, completion
   past the 30-second estimate, excessive-runtime watchdog stops, incomplete
   coverage, cache invalidation/corruption/permissions,
   concurrent callers, cancellation and failed recalibration. Use fake
   clocks/kernels for scheduler tests and small real cases for numerical tests.

Acceptance: mocked campaigns deterministically select/reject as specified;
cache-only lookup never invokes timing; every explicit calibration call attempts
a fresh campaign even with a valid record; repeated calls and
feedback work with writable and unwritable caches. Light subprocess tests
verify lock/atomic-read behavior on supported platforms, without benchmarks.

### 3. VBMC lifetime, options and public docs — Sol implementation

1. Declare `performance_calibration` with comment documentation in
   `option_configs/advanced_vbmc_options.ini`; validate `cached`, `off` or a
   profile. Integrate resolution in `vbmc/vbmc.py` at optimize entry and the
   exposed VP's first kernel use, preserving options frozen after init.
2. Preserve/share the immutable profile in `_candidate_vp`, the shell used by
   `_vb_init` in `vbmc/variational_optimization.py` (its generic attribute copy
   already preserves new state); check all deepcopy/whitening/undo and
   final-boost paths in that file, `vbmc/vbmc.py` and `whitening/whitening.py`.
3. In VBMC/VP save/load, migrate missing legacy profiles to historical defaults;
   preserve selected history profiles and reject conflicting load overrides.
   Add compact provenance to `_create_result_dict`; avoid new history keys
   or raw timing data in per-iteration records.
4. Extend existing seed/lifecycle tests using their existing run fixtures and
   small mocked state transitions. Cover pre-optimize PDF pinning, load at
   earlier iteration, restoration before resolution on completed-run
   continuation, boost, cache replacement
   and different-host resume. Do not add full optimize runs to unit tests.
5. Add handwritten `docsrc/source/api/functions/calibrate.rst` and
   `api/classes/calibration_profile.rst`, linked through the respective
   `functions.rst`/`classes.rst` toctrees and root `index.rst` as appropriate.
   Update VBMC/VP API pages and options docs; put the practical workflow in
   existing documentation, including explicit rerun, off, cache override,
   feedback, fixed settings, save/resume and entropy rounding explanation.
   New API pages own reference contracts; do not create another status report.

Acceptance: no lifecycle path recalibrates or changes a resolved setting;
display-off remains quiet; old saves load; docs accurately describe shipped
behavior. Build wheel/sdist and verify the new package/API from a clean install
(explicit setuptools package configuration may require adjustment).

### 4. Numerical/performance verification — Astra orchestration, Sol review

1. Main runs focused calibration, existing entropy/PDF gradients, options,
   save/load and seed tests in one process at a time. Isolate all test caches;
   pin historical settings for exact oracles and golden harnesses independent
   of a developer's real cache. Use the project's single-threaded BLAS oracle
   environment where required.
2. Run `dev/scripts/make_oracle_fixtures.py --check --exact` and
   `dev/scripts/golden_replay.py` for unchanged-default integration. Use the
   existing bounded replay mechanism for a fixed nondefault profile as an
   additional trajectory/quality check; compare against a separately recorded
   prechange default baseline, not a promise of identical cross-profile steps.
   Require unchanged initial designs, finite finals and the existing replay's
   population accuracy fences (Q3 + 3 IQR for changed trajectories). Compare
   only the five named configurations/seed 0; do not regenerate a population.
3. Measure one fresh ~30-second campaign, cached startup cost, and explicit rerun
   sequentially; report setup, selection, held-out results, numerical deltas,
   memory observations and total elapsed. Validate caching even when all three
   choices remain defaults. Repeat on a different available CPU/stack to assess
   portability; if none is available, report that limitation without inventing
   gains or blocking the cache/lifecycle correctness checks.
4. Sol independently reviews the full package change and test evidence by
   static analysis. Main resolves findings, then runs affected checks and the
   required project suite/3-OS CI when the implementation reaches PR stage.
   Update this plan, TODO, roadmap and overview to match delivered scope.

Do not launch the 870-case benchmark, S-VBMC jobs or a Torch solver effort.
Rollback is `performance_calibration="off"`; leave historical kernel defaults
and saved profiles interpretable. Calibration is in scope for PyVBMC 1.5;
complete its numerical, lifecycle, packaging and platform checks before release.

## Approved design decisions

- Explicit campaign; normal runs only load cached settings. This gives users
  predictable startup and lets them choose a quiet machine for measurement;
  unattended first-use timing could reach more users but adds surprise cost.
- Three fixed settings rather than per-shape adaptive tuning. This covers the
  distinct entropy value/gradient workloads with compact reproducible state;
  finer shape tables could improve some cases but add coverage and lifecycle
  complexity without current evidence.
- Per-user machine/environment JSON cache rather than project configuration
  or saves as the discovery store. Runs still embed their actual settings so
  portability does not depend on that cache. No automatic age-based reruns.
- Small platform helper dependencies rather than custom OS path/lock/backend
  inspection. They add dependencies but avoid maintaining three OS variants.
- Conservative aggregate selection with complete held-out/numerical gates.
  Defaults are a successful result when tuning offers no reliable gain.

The PI approved implementation. Follow the contracts above; investigate
evidence that contradicts implementation assumptions and record justified
adjustments here without reopening settled user decisions.

## Existing evidence retained

The [first measurement report](../results/2026-09-09-machine-local-calibration.md)
and [reproduction artifacts](../experiments/machine_calibration/README.md)
retain the developer experiment and its exclusions. The guarded prototype
uses source-derived private clones solely for pre-integration measurement.
Two balanced sweeps retained 65,536 in all nine regimes, costing 23.74/44.49 s
after imports. PDF was exact; entropy's maximum difference was 7.11e-15,
with identical RNG advancement. Eleven developer and eight existing numerical
tests passed; no optimizer trajectories were run. Other-machine gains remain
unmeasured. Those limits guide validation and do not redefine package
integration as a developer benchmark.
