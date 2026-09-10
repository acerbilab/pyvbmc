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
- [~] Freeze the launcher and verify actual worker imports in isolation.
- [ ] Run bounded timing/recording checks and update the estimate.
- [ ] Start the authorized overnight first stage after preparation passes.
- [ ] Assess the completed first stage and decide whether further sampling
  is useful.

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
