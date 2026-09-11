# Staged PyVBMC 1.5 population benchmark

## Purpose

Assess the integrated numerical changes against the existing 870-run golden
population, starting with an overnight subset of its configurations' seeds.
Compare posterior accuracy,
evidence error, convergence, evaluation counts, runtime and final-boost
decisions. Changes to the algorithm can legitimately move these quantities;
the assessment must distinguish improvements, tradeoffs and regressions.

The first stage supplies evidence for deciding whether to accept the changes,
investigate particular configurations, or collect more seeds. Equal sample
counts with the golden population are unnecessary. The PI accepts a staged
assessment with decisions informed by earlier results; subsequent reports
should identify each stage and the data used for its decisions. A full
870-run candidate population is an available extension, not a prerequisite.

Generating a golden reference and checking a candidate against it have
separate compute budgets. A large reference can take a long time to build
and then support many smaller candidate checks. Choose each check's target
coverage and seed count for its scientific question and available compute;
retain paired comparisons for overlapping seeds and add runs when useful.
The reference size does not set a minimum candidate run count.

Promotion of the candidate population to a new reference follows assessment
and an explicit scientific decision. Preserve the previous population and its
identity so the release comparison remains reproducible.

The candidate's numerical code and environment must stay fixed during the
campaign. Agent-skill work, documentation and S-VBMC integration that leaves
ordinary VBMC inference unchanged can proceed independently.

## Population and execution settings

Reference: [reference_870_20260907](../golden/noisy_extension_20260907/README.md).
Tracked sidecars are in `dev/golden/baseline/`; complete local traces are in
`dev/scripts/runs/golden/reference_870_20260907/`.

| First-stage cohort | Seeds | Runs |
| --- | --- | ---: |
| Original 16 configurations | 0–14 | 240 |
| `cigar_D15_exhaust` | 0–2 | 3 |
| `rosenbrock_D2_noise3`, `student_D8_noise3` | 0–14 | 30 |
| Total | | 273 |

Compare against all available reference seeds: 50 per original configuration,
10 exhaust cases and 30 per added noisy configuration. Preserve same-seed
comparisons for diagnosing individual changes. Three exhaust runs retain
coverage of the long-budget path within the overnight allocation.

The exact labels and seeds are recorded in the
[preparation manifest](../experiments/final_population_20260910/preparation.json).
This allocation excludes `lumpy_D10_noise3`, which is registered in the suite
but absent from the reference. Do not launch every registered configuration
with a uniform seed range.

Use one numerical worker and one BLAS thread, with
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` and `MPLBACKEND=Agg`.
Keep target budgets and definitions identical to the reference. Set
`vectorized_target=False` and `performance_calibration="off"` explicitly;
the latter preserves the historical chunk settings and avoids dependence on
the local calibration cache. This campaign measures performance with those
settings. It does not measure the benefit of optional machine calibration.

Use the production joint boost guard, `tol_elcbo_boost=0.1`. The production
code disables weight shrinkage inside the boost. Do not set the global
`weight_penalty` option to zero: the main variational fit retains its penalty.
Record requested and effective options, including those used inside the boost.

Pin actual source imports and commit identities in a separate checkout so
ongoing development cannot alter a running worker. The initial candidate is
`de686f712c6822968f0b98779469ec9e6152d0f4`; record the final harness commit
separately if preparation adds tooling. The checked environment is Python
3.12.6, NumPy 2.5.2, SciPy 1.18.1, gpyreg 1.1.0 and cma 4.4.4, with gpyreg
at `a2f8ddce867f502e29717959cf0ff3529f598618`. Editable package version
metadata is stale; establish source identity from worker import paths and
git/source hashes rather than that version string alone.

## Runtime estimate

The selected 273 reference runs took **7.139 optimizer hours**. Weighting the
full reference cohort means by the first-stage allocation estimates **7.336
hours**. Allow **8–10 hours elapsed** for changed trajectories, startup,
capture, serialization and machine load. These are scheduling estimates
derived from historical observations; a current-code timing check will refine
them.

| Candidate allocation | Runs | Time of matching reference seeds |
| --- | ---: | ---: |
| 15 seeds per ordinary configuration, 3 exhaust seeds | 273 | 7.139 h |
| 20 seeds per ordinary configuration, 3 exhaust seeds | 363 | 9.273 h |
| Full historical allocation | 870 | 21.638 h |

The first stage uses 15 seeds to leave overnight headroom. An extension to
20 seeds would add 90 runs and about 2.1 historical optimizer hours; decide
whether it is useful after reviewing the first stage.

After the capture mechanism is validated, time a small selection of actual
campaign cases, including a noisy and an exhaust case. Retain complete pilot
results as population members if their code, settings and artifacts match
the final campaign. Update the estimate from completed cases as the run
progresses. Statistical investigation after the campaign is additional work;
its duration depends on the findings.

## Artifact requirements

The existing `golden_trace.py` writes loop traces, returned-posterior arrays
and final metrics. It does not retain all information needed for the new
boost assessment. Add capture in the developer harness around the production
boost, preserving the production calculation and random stream:

- Pre-boost posterior, raw candidate even if rejected, and returned posterior,
  including their parameter transformers.
- Stored ELBO and SD values actually used for the decision, tolerance,
  acceptance outcome, and whether a boost was attempted.
- Sufficient posterior state to compute candidate and fallback accuracy and
  compare alternative guard thresholds offline without another boost.
- Candidate source, dependency versions and commits, worker import paths,
  thread settings, options, timings and artifact hashes.

Use fresh output directories. Resume only cases whose full required artifact
set passes integrity, identity and completeness checks. The current runner's
NPZ-exists shortcut is insufficient: an interrupted write can leave a trace
without its JSON or boost capture. Retain failed cases and diagnostics rather
than selecting replacement seeds or rerunning until favorable outcomes appear.

## Assessment

Validate the first-stage 273-case allocation, readable artifacts, finite comparison
metrics and 750 evaluations for each deliberate exhaust run. Distinguish
execution errors, budget terminations and the separate accuracy-based
usability criterion.

Run the existing population comparison against `dev/golden/baseline`: four
metrics across 19 configurations, giving 76 KS tests with Holm correction at
alpha 0.05. Unequal sample sizes are supported by the existing comparator.
Report effect sizes and distributions alongside flags; investigate adverse
changes. Interpret the three exhaust outcomes primarily as individual cases.
An even/odd split is optional descriptive context at this stage; the exhaust
halves are too small for the harness's minimum of three observations per
sample. No flag does not establish equivalence, and a flagged improvement is
not a regression.

Report convergence and usability rates, runtime and memory, and boost
acceptance/rejection counts with pre/candidate/returned scores and accuracy.
Examine the retained Normal D5 gsKL flag from the
[eta-bound fix](../results/2026-09-08-eta-bound-fix.md) in the population context.
Keep all outcomes, including difficult noisy cases, visible in the report.

Review the completed first stage before selecting follow-up work. Possible
outcomes are accepting the evidence, investigating a specific change, adding
seeds for selected configurations, or extending the whole population. Keep
existing cases when extending an unchanged candidate. If a numerical repair
changes the candidate, record it as a separate treatment and assess the
affected results accordingly.

## Preparation status — 2026-09-10

- [x] Establish allocation and historical runtime from all 870 sidecars.
- [x] Verify the reference pair hashes, NPZ integrity and agreement between
  local and tracked sidecars; record the preparation manifest.
- [x] Verify the local dependency stack and pinned gpyreg source.
- [x] Set the overnight allocation to 273 cases and estimate its duration
  from the matching reference seeds and full-cohort means.
- [x] Implement complete boost capture and validated resume behavior.
- [x] Check capture against ordinary execution for identical outputs and RNG
  state, covering accepted, rejected and skipped boosts.
- [x] Freeze the launcher and verify actual worker imports in isolation.
- [x] Run a bounded recording/timing check using the first Gaussian case;
  retain its artifacts as part of the population. Noisy and exhaust timing
  cases run next in the same campaign.
- [x] Start the authorized overnight first stage after preparation passes.
- [x] Assess the completed first stage and recommend whether further sampling
  is useful.
- [x] Independently verify the assessment and paired statistical analysis.
- [x] Run the recommended seeds 15–29 extension of the two noisy
  configurations and assess it with the pooled population (2026-09-11).
- [ ] Accept the evidence for reference promotion, or select further
  sampling.

### Launcher and verification

`dev/scripts/population_run.py` uses one fresh subprocess per case and wraps
the production boost to retain independent VP snapshots. Auxiliary JSON and
completion records live in `records/`, outside the golden comparator's
sidecar discovery. Complete records hash the trace, sidecar, boost snapshot
and boost metrics; partial or mismatched artifacts stop resumption for
inspection. A failed worker stops the supervisor with a case log.

The supervisor verifies the manifest's dependency versions and gpyreg commit,
clean frozen checkouts, actual import paths, and unchanged numerical source
relative to the prepared candidate. It requests Windows idle-sleep prevention
for its lifetime while allowing the display to sleep. `status.json` records
the active case and completed cases; full completion generates `summary.md`,
`comparison.md` and `finished.json`.

Focused validation: `test_population_run.py` passes all 13 tests, including
accepted/rejected/skipped production boost paths, RNG preservation, hook
restoration after exceptions, and invalid resume artifacts. A live paired
`normal_D5` seed-0 run has identical non-timer trace arrays, semantic final
fields and post-optimization RNG state with and without recording. Evidence:
`dev/scripts/runs/population_capture_preflight_20260910/verification.json`.

Independent Sol static review found no remaining launch blockers after
corrections to helper-import provenance, dependency recording, boost-report
validation, exclusive locking and comparison error handling. The summary
uses the golden harness's single writer. The runtime manifest records
`filelock==3.32.5` and `dill==0.4.1` alongside the numerical stack.

### Overnight launch — 2026-09-10

The supervisor started at **22:59:07 UTC+03**, Windows launcher PID **26080**,
supervisor PID **18876**. One Gaussian case completed and passed full artifact
validation before unattended launch; it is reused. The next case is
`rosenbrock_D2_noise1_seed0`, followed by the seed-0 D15 exhaust case.

Campaign root: `dev/scripts/runs/population_overnight_20260910/`.
The frozen PyVBMC/tooling checkout is `source/` at `68a43db`; numerical code
is unchanged from `de686f7`. The separate `gpyreg/` checkout is at `a2f8ddc`.
`launch_manifest.json` contains the ready, fixed 273-case manifest and
verified dependency versions. `source_verification.json` records the actual
import paths. `process.json` records the Windows launcher.

Read `results/status.json` for the active case, completed list and failures;
`launcher.stdout.log` and `launcher.stderr.log` hold supervisor output, with
one log per case under `results/`. Completion produces `results/finished.json`,
`results/summary.md` and `results/comparison.md`. The job requests idle-sleep
prevention for its lifetime; normal display sleep remains available.

To resume an interrupted job after confirming that no supervisor remains,
use the main repository's `.venv/Scripts/python.exe`, set
`PYVBMC_GPYREG_SOURCE` to the absolute campaign `gpyreg/` path, and invoke
the frozen `source/dev/scripts/population_run.py run` with `--manifest`
pointing to `launch_manifest.json` and `--out` pointing to `results/`.
Preserve these files and source checkouts. The launcher verifies complete
cases before skipping them and stops on partial attempts for inspection.

### Execution complete — 2026-09-11

All 273 cases completed with validated completion records and no execution
errors. The supervisor finished at **05:48:58 UTC+03**, after **6 h 49 min
50 s**; summed optimizer time across the 273 cases was **6.555 h**. The
supervisor is no longer running.

The automatic comparison reports **no configurations flagged across 76 KS
tests, Holm alpha 0.05**. Of the 273 runs, 266 reported convergence. The
remaining seven reached their evaluation budget: all three deliberate D15
exhaust cases and noisy Rosenbrock SD3 seeds 3, 5, 8 and 13. These budget
terminations produced complete results, not execution errors.

The completion record, summary and comparison are under `results/` at the
campaign paths above.

### Assessment — 2026-09-11

The [results report](../results/2026-09-11-overnight-population.md) records
artifact validation, accuracy, convergence, timing and boost decisions.
Usability is 259/273 versus 253/273 on matching reference seeds and 825/870
across the full reference. The final boost accepts 268 candidates and rejects
five; all 273 decisions match the production guard. Usability increases from
257 pre-boost to 259 returned, with no boost-induced usability losses.

The supplementary within-configuration paired analysis uses exact signed-rank
tests for the four continuous/count metrics and exact McNemar tests for
usability. None of its 95 tests rejects after one Holm correction. All paired
p-values were independently reproduced with discrete rank-sum enumeration
and binomial arithmetic.

Independent Sol review verified raw counts, matching, statistical methods,
boost decisions and report interpretation with no material findings. The
analysis script rejects Python's optimized mode so that its assertion-based
artifact checks cannot be silently disabled.

The recommended extension is 15 additional seeds (15–29) each for noisy
Rosenbrock SD3 and noisy logistic SD3, approximately 80 minutes, to assess
their posterior-error and usability shifts. Report seeds 15–29 separately
alongside cumulative summaries of seeds 0–29. Further sampling and reference
promotion remain open decisions.

### Follow-up extension — 2026-09-11

Seeds 15–29 of `logreg_D5_noise3` and `rosenbrock_D2_noise3` (30 cases) run
as a separate campaign of the same treatment under
`dev/scripts/runs/population_extension_20260911/`, reusing the overnight
campaign's frozen `source/` (`68a43db`) and `gpyreg/` (`a2f8ddc`) checkouts
unchanged. The launcher's recorded identity (commits, import paths,
dependency versions, thread settings and options) is identical to the
overnight campaign's, so the two campaigns pool as one candidate population.

The manifest fixes the confirmatory analysis before the run: seeds 15–29
paired by seed with the reference, exact two-sided signed-rank tests of
evidence error, gsKL and MMTV and exact McNemar tests of usability for each
configuration, eight tests under one Holm correction at alpha 0.05, with
noisy Rosenbrock gsKL and MMTV as the primary outcomes. Seeds 0–14 generated
the hypothesis; seeds 15–29 test it. The pooled seeds 0–29 are then reported
with the recomputed 19-configuration KS screen and the 95-test paired family.

The same reference seeds took **96.1 optimizer minutes**; the planning
estimate is about 90 minutes elapsed. The supervisor started at
**12:35:47 UTC+03**, PID **25292**, from the main repository's `.venv`
with `PYVBMC_GPYREG_SOURCE` set to the frozen `gpyreg/` checkout.
`process.json`, `launcher.*.log`, `results/status.json` and per-case logs
follow the overnight layout. `dev/scripts/analyze_population_run.py` now
pools the two campaigns (`--extension`), replaces the 20-seed cap on the
exact signed-rank enumeration with a dynamic program over midranks that
reproduces the committed first-stage results exactly, and reports the
extension separately.

#### Outcome

All 30 cases completed with validated completion records and no execution
errors; the supervisor finished at **14:41:23 UTC+03** after **2 h 5 min
36 s**. The [follow-up report](../results/2026-09-11-noisy-follow-up-seeds-15-29.md)
records the results: the seeds 0–14 shifts did not replicate (every
Rosenbrock metric moved in the candidate's favor on seeds 15–29, and
logistic gained five usable runs against one loss), none of the eight
confirmatory tests rejects, and the pooled 303-run population has 287
usable runs against 273 on matching reference seeds with no rejection in
the 76-test KS screen or the 95-test paired family. The pooled assessment
is under `dev/experiments/population_extension_20260911/`
(`assessment.json`, `comparison.md`, both campaign manifests);
`python dev/scripts/analyze_population_run.py` reproduces it.

Summed optimizer time was 123.0 minutes against 96.1 for the same reference
seeds: the machine ran slower throughout this campaign, including its last
45 minutes when nothing else ran, so the timing is not a speed comparison
and the trajectories are unaffected. No further sampling is recommended;
reference promotion remains the PI's decision.

#### Direction under consideration — 2026-09-11

Since the integrated changes look likely to stay, the PI is considering
completing the remaining 567 seeds of the same frozen treatment (the full
870-run allocation) and promoting that candidate population to the new
golden reference in place of `reference_870_20260907`. PI position
(2026-09-11): the first-stage evidence is accepted tentatively, with the
confirmation at equal sample sizes to come from this extension itself; the
extension is agreed in principle but is not to be started without an
explicit go-ahead, and the shipping defaults (the noisy-acquisition
decisions) must be settled before it runs. Recorded for that decision:

- A reference generated from the shipping code makes the golden replay
  gate (`golden_replay.py`) bit-exact again; against the current reference
  every replay parts early because the numerics changed, and only the
  accuracy envelopes remain.
- The launcher, manifests and pooled assessment need no change: further
  seeds are more campaigns against the same frozen checkouts, and the
  exact signed-rank enumeration covers 50 pairs.
- The shipping defaults must be settled first. The frozen candidate equals
  current `dev-next` on the default path only because the noisy-acquisition
  work is opt-in; a later default change would make the new reference
  stale for the noisy configurations.
- Run on the original benchmark machine: an exact replay baseline depends
  on platform-bound BLAS rounding and CMA-ES decisions, so a cluster run
  would give a valid population but not an exact replay baseline.
- Preserve the previous reference with its identity and README, and run
  the even/odd null check on the new population as previous references did.
- New targets (the benchflow candidates, the unrun `lumpy_D10_noise3`) fit
  the existing extension pattern later and need not block or wait for this.

| Remaining 567 runs | Optimizer hours |
| --- | ---: |
| Reference time for the same seeds | 12.9 |
| Candidate at first-stage speed (ratio 0.92) | 11.8 |
| Candidate at follow-up speed (ratio 1.28) | 16.5 |

Two overnight sessions, or one long unattended run, plus about 4%
supervisor overhead.
