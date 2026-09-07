# Plan: PyVBMC 1.5 latent bug fixes

Created: 2026-09-07
Status: PARTIALLY APPROVED — D1–D5, Q2/Q3, Q1/Q4 comparison designs,
the 150-run reference addition and parallel development approved on
2026-09-07; final Q1 treatment and Q4 default remain open.
Reference-extension preparation is complete (2026-09-07). The PI resumed
Phase 0 and the approved Phase 1 neutral fixes later that day; both are now
complete and verified. The 150-run noisy extension remains parked and must
not be launched or scheduled without a separate PI instruction.
PI update (2026-09-07, after CI 135): the estimated day-long runtime is
infeasible; reduce the reference addition. The 150-run allocation and launch
command below are historical preparation, superseded for future execution.
The PI selected 30 seeds each of `rosenbrock_D2_noise3` and
`student_D8_noise3` for the reduced proposal (60 additions, 870 combined).
Its runtime remains unmeasured: the earlier 5–6.5 hour estimate was an
extrapolation from other targets, not a Student D8 noisy measurement.
The PI subsequently authorized a couple of timed runs: exactly seed 0 of
each selected configuration, sequentially, on the pinned reference code.
This timing check does not launch the remaining 58 runs or the old 150.

- [x] Select a smaller allocation: 30 seeds each of noisy Rosenbrock D2 and
  Student D8; preserve pinned numerical source and original 810.
- [x] Measure one full seed-0 reference run of each selected configuration,
  verifying actual imports and retaining reusable JSON/NPZ results; report
  measured runtimes, termination and a revised batch estimate.
- [ ] Update the existing preparation and coverage records for that allocation.

Timing evidence: `dev/scripts/runs/noisy_reference_20260907/timing_20260907/`.
The ignored one-off driver reuses the frozen launcher's isolation and
validation helpers and calls the pinned `golden_trace.run_task` for the two
fixed seed-0 tasks. Independent Sol review of this invocation found no issues.
Both runs completed and passed the existing record/trace validator:

| Configuration (seed 0, noise SD 3) | Optimizer wall | Whole subprocess | Evaluations | Termination |
|---|---:|---:|---:|---|
| Rosenbrock D2 | 201.96 s | 204.21 s | 190/200 | Converged |
| Student D8 | 344.74 s | 347.82 s | 255/500 | Converged |

Worker times including trace/metric work were 202.38 and 345.84 s;
worker CPU times were 197.02 and 336.34 s. The recorded active-sampling
timers were 179.91 and 267.87 s (89% and 78% of optimizer wall; GP/VP
timers nest inside this bucket). Target evaluation took only 0.030 and
0.085 s. Student's convergence flag does not imply exact inference:
its evidence error was 1.603, gsKL 0.654 and MMTV 0.171; Rosenbrock's
were 0.297, 0.00580 and 0.0202. These are two reference observations,
not a tolerance selection or a population-quality conclusion.

Multiplying the measured subprocess times by 30 gives 4.600 h for the
proposed 60 runs, or 4.447 h for the remaining 58 if these two are reused.
This replaces the earlier cross-target extrapolation but is still based
on only one seed per configuration; it is not a measured population mean
or a runtime guarantee. No further runs are queued or running.
Both standard JSON/NPZ pairs and per-worker import records remain under
`timing_20260907/output/`, with `timing.json` there and `validation.json`
one level above. Actual imports matched the canonical preparation;
reference `623f5cd`, gpyreg `a2f8ddc`, supporting source hashes and the
frozen launcher were unchanged. All 1,620 original files passed raw-byte
hash checks. The obsolete 150-run `extension/` remains empty. The two
timing pairs are retained for validation/adoption into the reduced batch;
the published reference remains 810 until that integration is performed.

Base inspected: `dev-next`, `edb59700bbb77034c877c71f8475994abfd40f66`.

## Purpose and boundaries

Repair the remaining implementation defects in modernization-roadmap pickup 9,
with each numerical change attributable and checked against the completed
reference (810 runs complete; extension to 960 approved below). This plan owns the candidate dispositions, implementation
contracts, PI questions, and execution gates for the next executor; the
roadmap remains the release tracker and `dev/golden/README.md` describes the
reference. Update this plan during execution; do not create a parallel worklog.

The earlier extension to 810 runs is complete; its sidecars, summary and
validation records are published. The approved 150-run addition below is
pending. The NPZ traces remain local/gitignored; release-asset
publication is pending. No jobs or watchers need
reattaching. Preserve `reference/stage3-20260906` at `7314a6a`, all reference
sidecars and NPZ traces, and their recorded provenance. Never regenerate the
reference to make a correctness change pass.

S-VBMC is already implemented in the separate `acerbilab/svbmc` repository.
Roadmap pickup 10 covers integration and compatibility of that existing
package with PyVBMC 1.5; it remains separate from this pickup 9 latent-bug
plan. Upcoming method extensions and tensor-solver work also have their own
scope. Preserve the existing PyTorch decision: first a float64 CPU/GPU
feasibility prototype against modernized NumPy CPU, including transfer costs;
then a PI decision on the full port, preferably for 1.5 if feasible. CPU
performance, numerical reliability, installation friction, and extensibility
matter. PI clarification (2026-09-07): 3× runtime was an example of clearly
unacceptable performance; concern starts at substantially smaller slowdowns.
Around 1.2× runtime may be acceptable given the other benefits, but is not
an agreed cutoff. Dependencies, Python floor, and the NumPy transition remain
that design's questions.

Noise shaping, `compute_var == 2`, log-space mixture sums, and new acquisition
or optimizer algorithms stay deferred. Approval of this plan authorizes the
accepted repairs and short gates, not an unspecified population campaign,
release, merge, or publication. The PI separately approved the 150-run noisy
reference extension below for tonight. The final 960-run candidate campaign is prepared here and runs
only on explicit PI instruction after the remaining release code is settled.

## Investigation and candidate dispositions

Read the 1.5 overview, `dev/TODO.md`, golden README and extension record,
roadmap pickups 3f/9, modernization discussion (especially §3–5 and §9),
and the final-boost finding before execution. Paths below are repository-relative.

“Neutral” means unchanged numerical outputs and RNG stream for existing
supported float64 reference runs. It does not mean that a broken optional
input or unsupported utility call must retain its old behavior. “Moving”
means the fix can change a seeded run's optimization trajectory or returned
result: for example, its sampled points, GP fits, posterior parameters, or
ELBO. “Final-output moving” means the main-loop trajectory stays unchanged
but the returned posterior or its statistics can change. Each moving fix
gets a separate replay to identify and assess those changes; movement alone
does not imply a regression.

### Numerical and state changes

| Candidate / current location | Verified status and proposed disposition | Gate |
|---|---|---|
| `_get_hyp_cov`, `pyvbmc/vbmc/gaussian_process_train.py:691–745` | Still broken: transposes sample rows, counts parameters as samples, adds scalar dot products to every covariance entry. Also has a decay-parentheses mismatch with MATLAB. Repair together, retaining the intended weighted estimator. | Moving, phase 3 |
| Acquisition regularization, `vbmc.py:762`, `acquisition_functions/abstract_acq_fcn.py:114–155` | Producer/reader spellings differ; activation also exposes `(M,1)` masking/broadcast failure. Canonical spelling, old-state compatibility, and output-shape repair belong in one change. | Moving, phase 4 |
| GP sampling stop, `vbmc.py:1341–1352,1965–1989` | Both guard and assignment use the wrong key. History `N`/`gp_sample_var` do not exist; `var_ss` already does. Implement existing MATLAB criterion using canonical state and recorded N bookkeeping. | Moving, phase 5 |
| `VBMC.final_boost`, `vbmc.py:2000–2099` | Unconditionally returns candidate. Test the PI-selected continuum b in [0,5] guard with tolerances 0.1 and 0.2, with boost shrinkage disabled; choose the default after expanded noisy validation (Q4). | Final-output moving, phase 2 |
| `_neg_elcbo`, `variational_optimization.py:1144–1148` | Confirmed caller mutation and, with an active lower eta bound, a gradient inconsistent with the evaluated objective. Absolute-eta versus weight-ratio penalty semantics remain an open PI design question; MATLAB parity alone does not decide it (Q1). | Moving if approved, phase 6 |
| `vp.pdf`, `variational_posterior.py:744–762` | Original-space non-log density gradient is silently wrong; log density already rejects it. MATLAB has the same inconsistency. No solver or current S-VBMC caller needs this gradient. Reject both consistently; Q2 approved. | Neutral for solver; API correction |
| `true_mean` / `true_cov`, `vbmc.py:1281–1303` | NumPy truth-value guard is broken. List-valued diagnostics also consume the solver RNG. Recommend fix presence/shape checks and compute diagnostic moments on a private copied generator. | Neutral by default; changes diagnostic-enabled runs |
| Logger finalization, `function_logger/function_logger.py:562–576` | Explicit `finalize()` omits `n_evals`; optimize never invokes it. Fix aligned trimming; recommend leave invocation explicit (decision D4). | Neutral |
| `results["rng_state"]`, `vbmc.py:2505` | Literal string remains. Return an independent snapshot with the existing `{"generator": ...}` format at return-time, after post-loop draws. | Neutral |

### Supporting fixes and closed entries

| Candidate | Disposition and acceptance |
|---|---|
| Float32/float16 constructor inputs (`vbmc.py:458–478`) | Widen x0 and all hard/plausible bounds to float64 at the input boundary, preserving validation/warnings and caller arrays; turn the strict xfail in `test_vbmc_init.py` into passing parametrized coverage. |
| Integer variance placeholders / `var_ss : int` | Use floating zeros for mathematical variance outputs, including no-variance and one-sample paths in `variational_optimization.py`; correct docstring and remove obsolete integer exceptions from the dtype canary. Counts stay integers. |
| Transformer equality (`parameter_transformer.py:405–419`) | Compare scale with the other object's scale, handling None/shape differences; use actual `is` for the integration tests that claim shared-transformer identity. |
| Logit Jacobian overflow (`parameter_transformer.py:312–318`) | Stable tail evaluation; test finite results at ±710 and beyond. Preserve the current ordinary-range arithmetic where finite if `logaddexp` changes its bits; no need to change the transform or inverse. |
| Rotation determinant assumption | Forward/inverse use transpose as inverse, so adding a log determinant alone is incorrect. Validate finite `(D,D)` orthogonal rotations (reflections allowed) and document the contract; do not add general affine support. Test FD Jacobian and round trip with rotation and scale. |
| `kl_div_mvn` method decorator | Remove the method-only decorator and normalize all four inputs locally; scalar, 1-D, 2-D, keyword and mixed calls must agree with existing 2-D results. Keep the KL formula unchanged. |
| Cubic closure (`gaussian_process_train.py:604–613`) | Replace lower-order uses of captured `x` with the argument `x_`; production already calls `f(x)`, so values stay identical. No new polynomial algorithm. |
| `_vb_init` type 3, frozen sigma, `K_new > K` | Still creates old-K zero widths. Preserve supplied widths and copy for new slots (Q3 approved). Default optimize-sigma=True path must retain draw order and values; the normal solver never disables this flag. |
| Pruned `J_sjk` (`variational_optimization.py:378–394`) | Additional unfixed §9 entry omitted from pickup's short list: prune both component axes, keep shape `(Ns,K,K)` and correspondence with `I_sk`. No PyVBMC runtime reader uses it, but S-VBMC's constructor filters VPs using its maximum; include that downstream behavior in compatibility coverage. |
| Unused `noisy_cigar` test helper | Delete the unused definition in `test_vbmc_optimize.py`; no new full optimize test. |
| Notebook 1 evidence constant / notebook 6 noise broadcast | Correct notebook sources and regenerate scripts through `examples/scripts/Makefile` (equivalent Windows pipeline if make/sed unavailable). Validate the example-1 one-dimensional quadrature and per-row noise shapes without running whole notebooks. |
| SciPy private imports in `priors/` | Remove unused Product imports. Consolidate frozen-distribution recognition through public distribution factories/types, preserving accepted types; do not broaden to arbitrary duck typing or change dependencies. See phase 1. |
| Multivariate SciPy prior constructor RNG draw | Additional related finding in `priors/scipy.py:49`: calls `rvs(1)` only to infer D. Use the frozen distribution's dimension metadata and test no global/frozen RNG mutation. No reference benchmark uses this constructor. |
| `compute_var_log_joint`, `mcmc_importance_sampling`, `search_cmaes_best` | Dormant for built-ins; custom acquisition objects can set the two flags and saved Options can contain the key. Recommend preserve compatibility hooks/accepted option and document current support status, rather than delete or implement new algorithms (D5). |
| `optimize_lambda` typo | Already repaired: `_vp_bound_loss` uses `optimize_lambd` at lines 566/588. Verify frozen-sigma combinations in focused coverage, do not redo the change. |
| `_compare_matlab.rand_int` | Already repaired (`res = 1`, lines 41–51); preserve its MATLAB 1..hi convention. |
| Runtime plotly/pytest dependencies | Already moved to extras; corner is lazy. No fresh plotting-dependency redesign. |
| Resume ELBO self-comparison | Already repaired: `test_vbmc_optimize.py:630` compares `elbo_1 == elbo_2`. Keep and reuse this test. |
| Earlier §9 GP-gradient, `separate_K`, history-factor, aliasing, plotting and fixture-name bugs | Already fixed in Stages 0–3 as marked in §9; preserve their tests and fixtures. |
| `_sq_dist` batch-dependent centering | Explicitly not a defect in §9: different floating-point centering can affect near-ties. Preserve the Stage 2 arithmetic; no change in this plan. |
| `_real2int` input mutation | Deliberately preserved by the Stage 2 batched CMA-ES objective. Retain its current behavior; changing the mutation contract is outside this fix pass. |
| gpyreg `step_out=True` stale coordinates | Confirm against sibling `gpyreg/slice_sample.py`; separate upstream repair/test and later validated pin bump (phase 7). PyVBMC keeps step_out off. |

### MATLAB evidence and bounded probes

Upstream examined at commit `396d649c3490f1459828ac85f552482869edf41c`:

- [GetHypCov](https://github.com/acerbilab/vbmc/blob/396d649c3490f1459828ac85f552482869edf41c/misc/get_GPTrainOptions.m)
  weights sample rows and accumulates outer products. Its decay divides sKL
  by `TolsKL * FunEvalsPerIter`. A local 2-sample/3-parameter probe currently
  produces a constant `(2,2)` covariance, not the required `(3,3)` result.
- [Sampling stop](https://github.com/acerbilab/vbmc/blob/396d649c3490f1459828ac85f552482869edf41c/private/vbmc_termination.m)
  uses all history after three iterations: the helper's `idx_stable` is
  literally 1, not a reliability-selected suffix. Half the weight is on
  the current variance; half is normalized `exp(-(N_last-N_i)/10)`.
  [MATLAB's main loop](https://github.com/acerbilab/vbmc/blob/396d649c3490f1459828ac85f552482869edf41c/vbmc.m)
  sets N from Xn, including logged rows made inactive by trimming, as Python
  does. This count must not be replaced by the number of active GP rows.
- [Negative ELCBO](https://github.com/acerbilab/vbmc/blob/396d649c3490f1459828ac85f552482869edf41c/misc/negelcbo_vbmc.m)
  keeps original theta for the bound loss. The Python stability subtraction
  is useful for softmax arithmetic but need not mutate theta.
- [PDF](https://github.com/acerbilab/vbmc/blob/396d649c3490f1459828ac85f552482869edf41c/vbmc_pdf.m)
  also leaves the non-log original gradient uncorrected, so MATLAB agreement
  cannot certify this API result.

An offline read of all 810 current JSON/NPZ pairs found **259** stored final
ELCBOs strictly below their corresponding `best_iter` ELCBO at coefficient 5.
The largest decrease is `logreg_D5` seed 5 (about 20.18); many are small.
This is the number the guard would reject on the stored inputs, not a claim
about reruns after subsequent fixes or the quality of every rejection.
`student_D4` seed 19 in the current reference is accurate (gsKL about 0.0104);
the old `baseline_20260903` failure trace remains locally available. Do not
carry forward the historical claim that only one current run changes.

**Boost-gate follow-up (PI discussion, 2026-09-07):** the PI proposed
requiring delta_ELBO - b*delta_SD > -tol_elcbo_boost for every b in [0,5].
This is affine in b, so the exact continuum check is
min(delta_ELBO, delta_ELBO - 5*delta_SD) > -tol_elcbo_boost; a grid is
unnecessary. Across all 810 finite stored pairs, the endpoint result matched
a 1001-point grid to absolute tolerance 1e-12. Tolerance 1 rejects only
logreg_D5 seed 5, but misses the original historical student_D4 seed 19
failure (drop 0.9403265375). Tolerance 0.5 rejects that historical failure
and still only one current reference run. The PI subsequently selected 0.1 and 0.2
for testing (see below); the final default is not yet approved. These are existing boosted candidates,
before the planned removal of boost shrinkage.

For the five largest continuum score drops (all at b=5), reconstructing
the selected `best_iter` and final VPs gives the following accuracy changes.
Delta means boosted minus pre-boost; positive means worse for both metrics.

| Run | ELCBO drop | Pre gsKL | Boost gsKL | Delta gsKL | Pre MMTV | Boost MMTV | Delta MMTV |
|---|---:|---:|---:|---:|---:|---:|---:|
| logreg_D5 seed 5 | 20.184524 | 0.068067 | 3.526090 | +3.458023 | 0.058863 | 0.320079 | +0.261216 |
| cigar_D8 seed 44 | 0.402476 | 0.024293 | 0.203778 | +0.179485 | 0.038365 | 0.068253 | +0.029888 |
| student_D8 seed 12 | 0.333870 | 1.859018 | 2.585095 | +0.726078 | 0.144950 | 0.177164 | +0.032215 |
| logreg_D5 seed 34 | 0.240717 | 0.010312 | 0.079130 | +0.068818 | 0.021758 | 0.050497 | +0.028739 |
| lumpy_D10 seed 11 | 0.219067 | 0.751231 | 0.664735 | -0.086495 | 0.084072 | 0.080923 | -0.003150 |

The reconstruction uses saved per-iteration VP arrays and transformer state
at `best_iter`, final VP arrays, and the existing `benchmark_targets.metrics`
with its fixed diagnostic seeds (100,000 samples for MMTV, 200,000 for bounded
moments; unbounded moments are analytic). All five reconstructed boosted
gsKL/MMTV/RMSE/evidence-error metrics matched the tracked sidecars at
rtol=1e-9, atol=1e-11. No optimization was rerun. Detailed local output:
`dev/scripts/runs/latent_fixes/boost_metrics_top5_20260907.json` (ignored).
These results show that tolerance 0.5 admits some measurable accuracy
regressions; an ELCBO drop also need not mean worsening posterior accuracy
(lumpy_D10 seed 11 improves on both estimated metrics).

**Noisy coverage before selecting the tolerance (2026-09-07):** the PI
suggested considering 0.25 and flagged limited noisy coverage. The reference
has 100 noisy runs but only two configurations, each with 50 seeds, both
using supplied homoskedastic Gaussian observation SDs. An offline scan found:

| Noisy configuration | Runs | Largest continuum ELCBO drop | Rejected at 0.25 |
|---|---:|---:|---:|
| rosenbrock_D2_noise1 | 50 | 0.0082534110 (seed 39) | 0 |
| logreg_D5_noise3 | 50 | 0.0226660557 (seed 36) | 0 |

Tolerance 0.25 rejects three of the full 810 reference runs (logreg_D5 seed 5,
cigar_D8 seed 44, student_D8 seed 12). Existing noisy cases therefore do not
argue against 0.25, but two configurations do not establish behavior across
noisy targets. Larger absolute ELBO uncertainty alone does not imply a larger
boost score drop: both posteriors use the same fitted GP, and the guard uses
the change in uncertainty. Noise can still change the surrogate landscape
and boost behavior, so this needs empirical coverage.

**Current thresholds to test (PI, 2026-09-07): `tol_elcbo_boost=0.1` and `0.2`.**
The PI considers even a 0.2-nat drop in the ELBO itself nontrivial. Keep the
continuum b in [0,5] criterion, including its b=0 plain-ELBO safeguard.
On the stored 810 runs, 0.2 rejects seven: logreg_D5 seed 5 (20.184524),
cigar_D8 seed 44 (0.402476), student_D8 seed 12 (0.333870), logreg_D5 seed 34
(0.240717), lumpy_D10 seed 11 (0.219067), student_D8 seed 8 (0.208928), and
lumpy_D10 seed 4 (0.201714). No current noisy run is rejected. All seven
worst drops occur at b=5. At 0.1, 15 of the stored 810 runs are rejected,
also with no current noisy rejections. Test both thresholds on the same
candidates, without treating either as the preferred default;
the final default remains subject to expanded noisy coverage and the revised
boost objective. Include metric checks of accepted and rejected candidates:
lumpy_D10 seed 11 is already a known case where rejection sacrifices a small
improvement in the two posterior accuracy metrics despite a score decrease.

**Approved reference extension (PI, 2026-09-07; current task: prepare only):**
add rosenbrock_D2_noise3 (isolate noise level), student_D8_noise3
(higher-dimensional broad tails), and lumpy_D10_noise3 (higher-dimensional
mixture structure), each at seeds 0–49, using the existing target wrapper and
paper-budget convention. The PI superseded the proposed 10/20-seed pilot with
50 seeds per configuration from the outset: 150 new runs. All three become
standard golden benchmark configurations. Extend the existing reference
itself to **960 runs across 20 configurations**, including **250 noisy runs
across five configurations**; retain the existing 810 pairs unchanged.
This is an additive reference expansion, not a separate diagnostic-only set.

Prepare the batch against the existing reference numerical implementation
(`7314a6a`, original reference environment, explicit vectorized_target=False,
single worker and single-threaded BLAS), with additive benchmark configuration
changes and truthful source provenance. Do not mix latent or boost fixes into
these new reference runs. Run from a dedicated checkout pinned to an additive
benchmark-only commit based on `7314a6a`; freeze it for the batch. Development
on other branches/checkouts may proceed before or during these runs, as
specified below. Validate all 150 new pairs, hash-check the original
810 unchanged, and extend the standard baseline sidecars, summaries and
reference manifest after validation. Update standard suite registration,
golden README, benchmark plan and release comparison coverage consistently;
the expanded population comparison has 80 KS tests. Keep the historical
810-run completion record as provenance for the subset, not as a competing
reference. No new batch has been launched or scheduled during this discussion.

Record pre-boost and boosted MMTV/gsKL/evidence error and score changes at
b=0 and b=5, reconstructing metrics from saved states as above if necessary.
Test tolerances 0.1 and 0.2 on the same reference candidates. Earlier
0.25 and 0.5 results remain supporting sensitivity evidence.
Subsequently evaluate the planned boost objective without shrinkage against
this expanded reference before fixing the default, retaining rejected raw
candidates for analysis. The reference extension is authorized; final
release-candidate population runs remain a later, separate launch decision.

Probes ran via the reference `.venv/Scripts/python.exe`, single-threaded
BLAS, without source edits or optimization runs. The first eta probe used
an unsupported VP `seed=` argument and failed before its diagnostic call;
the corrected probe used `rng=` and produced the result above. No pytest,
oracle generation, trajectory replay, or campaign was run while planning.
A separate constructor probe confirmed MVN and MVT `.dim` metadata matches
the inferred dimension and both current SciPy-prior constructors advance
global RNG.

## Execution roles and sequence

**Astra (`gpt-6-astra`, high), orchestrator:** owns scientific contracts,
  PI decisions, integration, and the acceptance of explained numerical changes.
**Sol (`gpt-5.6-sol`, high), implementation executor:** implements the bounded
phases below, one change group at a time. A separate fresh-context Sol reviews
each substantive group statically. No more than one agent runs heavy compute;
the orchestrator normally runs all tests/replays, with reviewers read-only.
These roles correspond to Fable and Opus in the companion workflow.

**Parallel development approved by the PI (2026-09-07):** the phase order
expresses dependencies and validation/integration order, not a requirement
to perform all work serially. In particular, phase 1 neutral fixes need not
wait for the reference extension to finish. Boost implementation, tests,
and diagnostic tooling may also be prepared in advance on isolated branches;
final scientific choices still depend on the agreed evidence.

- Use a dedicated reference checkout with the frozen numerical source and
  additive benchmark registrations, and a separate development checkout
  based on `dev-next`. Do not switch, edit, merge into, or repoint the
  reference checkout while its batch is running. Record the exact launch
  commit and benchmark-only diff from `7314a6a`.
- Verify actual imports in the reference worker, not just its working
  directory: the original environment uses editable installs, so a separate
  worktree alone does not guarantee source isolation. Check `pyvbmc.__file__`,
  benchmark module location, gpyreg source/version, interpreter, package
  versions and thread settings. Ensure every worker resolves the pinned
  source; do not reinstall or upgrade the reference environment or mutate
  its imported dependencies during the batch.
- Only one heavy computation runs at a time. While the reference worker
  runs, development may edit files, prepare tests, inspect sources, and do
  static review. Run numerical preflights before the batch or queue test
  suites, replays, and experiments for when the worker is idle. Parallel
  development does not authorize concurrent heavy suites or benchmarks.
- Keep separate commits for the agreed change groups. Preserve the
  neutral-before-boost-before-other-moving validation sequence and use the
  same base for each arm of a scientific comparison. Early implementation
  does not settle Q1's treatment or Q4's default. Integration and final
  claims wait for the corresponding checks and PI decisions.

Use `$task` on this approved file when execution begins. Work on a feature
branch from current `dev-next`; inspect status first and preserve other work.
If source contradicts a stated assumption, investigate and update the plan;
do not force an obsolete recipe. A newly discovered algorithm choice goes
to the PI, while independent authorized fixes may continue.

### Phase 0 — Establish the execution gates

**Executor:** Astra orchestrator, with Sol for bounded harness changes.

#### Active execution: regression gate and neutral fixes (2026-09-07)

Work on `dev-latent-neutral-fixes`, preserving the completed, uncommitted
reference preparation inherited from `dev-noisy-reference-prep`. The pinned
reference checkouts, manifest, and original 810 pairs remain untouched.

- [x] Record development imports/environment; run exact existing oracles and
  one fresh default five-case preflight before any numerical edit. Evidence
  goes under `dev/scripts/runs/latent_fixes/neutral_20260907/`.
- [x] Sol: repair the existing replay comparison to the Phase 0 contract below,
  with synthetic field-mutation coverage. Change only the existing replay
  tooling and its focused tests; no new supervisor or launch framework.
- [x] Re-render the saved preflight through the stronger comparison, preserving
  its original run provenance, and obtain independent static review.
- [x] Implement Phase 1 in bounded Sol-owned groups, starting with boundary
  float64 casts and floating variance outputs; preserve the detailed ten-step
  contracts below and keep Q1/Q4/PyTorch decisions unchanged.
- [x] Run the relevant focused checks, neutral exact-oracle/replay gates and
  independent review; record completed groups and any remaining requirements.

The preflight completed before numerical edits. Sol owns disjoint state/dtype
and utility/API groups; Astra runs all numerical checks sequentially.

Initial evidence: development imports resolve to this checkout and sibling
gpyreg `a2f8ddc`; original Python 3.12.6 / NumPy 2.5.2 / SciPy 1.18.1 /
gpyreg 1.1.0 / cma 4.4.4, all three BLAS thread settings 1, recorded in
`neutral_20260907/environment.json`. `make_oracle_fixtures.py --check --exact`
exited 0: all eight fixtures exact (`oracles_before.log`). The fresh preflight
executes the original `HEAD:dev/scripts/golden_replay.py` text with its normal
`__file__` and fresh `--out .../preflight`, so concurrent comparator edits
cannot change the baseline comparison midway through the five runs.
It exited 0: all five cases identical under the original comparison, zero
flags, 4.5 minutes (`preflight.log`).
The repaired comparator then passed 57 synthetic tests (`replay_tests.log`)
and `--report-only --out .../preflight` exited 0 with all five stored loops
and finals exact (`preflight_stronger.log`). Original code, elapsed time and
thread provenance were retained. Both historical and fresh traces omit the
returned VP transformer, explicitly reported as not certifiable; no reference
schema or files were changed. Independent review remains pending.
The separate Python 3.12 environment installed successfully with NumPy 2.2.6,
SciPy 1.15.0 and gpyreg 1.1.0 (`scipy115_install.log`). The original interpreter
was not modified.

Phase 1 validation so far: `pytest pyvbmc/testing/priors -q --reruns=5 -x`
passed 46 tests in the original SciPy 1.18.1 environment and 46 in the separate
SciPy 1.15.0 / NumPy 2.2.6 environment, without reruns (`priors_current.log`,
`priors_scipy115.log`). Both runs used workspace-local pytest temp/cache paths
and single-threaded BLAS. The remote CI matrix was inspected, not rerun here.
After the final frozen-univariate type check, the minimum-version run again
passed all 46 (`priors_scipy115_final.log`); current-version priors passed in
the 74-test GP-training/gradient/seed group (`remaining_tests.log`).
State/init, single-sample variance, oracle, logger, variational-optimization
and mocked-result coverage passed 251 tests, 15 skipped (`state_tests.log`).
Transformer/KL/posterior and posterior-FD coverage passed 293 tests, 18 skipped
(`utilities_tests3.log`), after correcting a new test's stale scale and updating
the old original-space-gradient expectation to Q2. No new full optimize test
was added. Example 1 quadrature gives -2.2598471241723845; example 6 returns
matching `(N,)` values/noise for N=1 and N=4 (`examples_checks.json`). Both
scripts were regenerated from notebook code cells with the Makefile-equivalent
source export/path rewrite and in-process isort/Black; they match exactly.
The complete Sphinx HTML build, including the Makefile's example-copy step,
exited 0 (`sphinx_with_examples.log`); its sole warning is the existing example
2 Plotly MIME output. The temporary copied examples were removed afterward.

Independent Sol review found three issues, now corrected: cast narrow inputs
before inferred-bound arithmetic; use the matching baseline sidecar for custom
replays; explicitly flag nonfinite returned values even when accuracy metrics
are finite. Review uses the two available Sol agents on each other's work;
the runtime's thread limit prevents additional fresh reviewer threads. No agent
reviews its own implementation. The corrected constructor passed all 66 init
tests (`bounds_final_tests.log`); the final replay harness passed 67 synthetic
tests (`replay_tests_closed.log`), including custom-baseline missing-sidecar
uncertainty, and re-rendered all five preflight cases exactly with zero flags.
Post-fix exact oracles passed all eight fixtures (`oracles_after.log`), with
no fixture changes. The post-fix five-case replay exited 0 in 3.3 minutes:
all stored loop/final outputs and initial designs exact, zero flags
(`postfix.log`, `postfix/replay.json`, `postfix/replay.md`). Final report-only
passes on both preflight and post-fix artifacts retained their original run
provenance (`preflight_closed.log`, `postfix_closed.log`). Historical returned
transformers remain explicitly uncertifiable; this is not a full-state claim
about information absent from the traces.

Reference preservation rechecked: all 1,620 original raw files are byte-identical,
the frozen launcher hash is unchanged, and the 150-run extension output directory
is empty (`reference_preservation.json`). Q1/Q4 and PyTorch decisions are unchanged.

**Outcome:** Phase 0's regression gate and all ten Phase 1 steps are accepted.
Boundary/floating variance, transformer, KL, cubic, both-axis pruning, Q3 fixed
widths, optional truth diagnostics, logger finalization, returned RNG snapshots,
Q2, public SciPy compatibility, examples and option comments are complete.
The already-repaired `optimize_lambd` item is closed by inspection and frozen-
sigma coverage; D5 hooks/serialized keys remain intact with honest comments.
No Phase 2–6 trajectory-moving production treatment was applied. The next
development group is Phase 2's approved boost comparison; final Q1 treatment
and Q4 tolerance still require their specified evidence.

Independent `$doublecheck` is complete: each Sol reviewed the other's code,
all confirmed findings were fixed and rechecked, with no remaining findings.
No fresh third reviewer could be spawned because of the runtime thread limit.
Checks were sequential in the parent; reviews were static. The three accepted
implementation commits on `dev-latent-neutral-fixes` are `ae5d871` (replay),
`c8a284e` (state/dtype) and `3e67d9e` (utility/API). Repository-pinned Black 23.3.0
was run in-process on all 29 changed Python/notebook files to avoid the known
Windows multiprocessing hang; syntax equivalence was checked for Python
formatting changes. The commit hooks for whitespace, final newlines, isort and
pycln passed; only the already-run Black hook was skipped. The PI subsequently
authorized committing the remaining reference-preparation files and developer
notes and pushing `dev-latent-neutral-fixes` (2026-09-07). The parked 150-run
batch remains a separate future action.

#### CI follow-up: run 133 (2026-09-07)

- [x] Diagnose the failed push run on `613ba72`: Ubuntu/Python 3.12 reached
  1,009 passing tests before the new live-transformer identity assertion failed
  in `test_vbmc_optimize_rosenbrock` at iteration 0. The previous focused checks
  omitted this end-to-end module; the exact replays do not assert object identity.
- [x] Repair live-transformer sharing without weakening the assertion or changing
  posterior deepcopy/history isolation; Sol implements, Astra runs checks.
- [x] Run the affected end-to-end module, focused regression and numerical gate;
  independently review the correction.
- [x] Commit/push the correction and verify its CI result.

The reference extension remains parked. Evidence is under
`dev/scripts/runs/latent_fixes/neutral_20260907/ci_133*`.
Local reproduction failed at the same assertion. `optimize_vp` now retains its
input transformer on the accepted return object after internal fine/pruned
copies; ordinary VP deepcopy and history isolation remain intact. The existing
pruning test covers both pruned/unpruned returns and the deepcopy contract.
Independent Sol static review found no issues. The complete
`test_vbmc_optimize.py` module, both new return-path cases, and the existing
deepcopy test passed: 12 tests, five reruns (312.28 s; `ci_133_after.log`).
All eight exact oracles passed (`ci_133_oracles.log`); all five replay cases
matched every stored loop/final output and initial design with zero flags
(3.0 minutes; `ci_133_replay.log`). No oracle or baseline references changed.
The correction was pushed as `514bad7`. CI run 134 passed the original case
but failed the same identity assertion in noisy half-normal active sampling
(1,016 passed, 49 skipped). The first correction was incomplete: accepting
the pre-update VP rollback in `active_sample` also adopts a deep-copied
transformer. The local module's five reruns were not investigated before
push and cannot be dismissed as statistical failures.

- [x] Preserve transformer identity when active sampling restores the old VP;
  cover that rollback deterministically without weakening the integration test.
- [x] Validate with reruns disabled first, inspect any rerun reasons explicitly,
  independently review, then push and verify the replacement CI run.

The second correction synchronizes the VP at `active_sample`'s sole return
with the logger's transformer. The deterministic regression forces a rejected
full update and verifies the restored parameters, shared live transformer,
and ordinary deepcopy isolation/shared RNG. It passed alone; the complete
active-sampling test module plus the failing noisy half-normal integration
test passed with retries disabled: 25 tests, 66.29 s (`ci_134_after.log`).
Independent Sol static review found no issues and checked the other live-loop
replacement paths. All eight exact oracles passed (`ci_134_oracles.log`), and
all five replay cases matched every stored loop/final output and initial
design with zero flags (`ci_134_replay.log`). No references changed.
The correction was pushed as `686a75f`. [CI run 135](https://github.com/acerbilab/pyvbmc/actions/runs/34124719311)
passed the Ubuntu/Python 3.12 suite: 1,043 passed, 49 skipped, one rerun,
629.41 s (`ci_135.log`). Both previously failing integration tests and the new
rollback regression passed on their first attempts. The sole retry was
`test_acq_fcn_imiqr.py::test_complex__call__`, which then passed; the workflow's
pytest flags do not retain its retry traceback, so its exact cause is
unconfirmed. It is a separate acquisition test, not another observed live
transformer assertion failure. The current Phase 0/1 scope is complete;
the 150-run reference extension remains parked and later phases remain pending.

#### Active subset: prepare the noisy reference extension (2026-09-07)

This subset implements the approved allocation above without running it.
The remaining numerical phases and scientific decisions stay pending.

- [x] Read the governing records, inspect the clean checkout, and create
  `dev-noisy-reference-prep` from `a165626ef7b0284b866ab0e27d9a8a764f912758`.
- [x] Sol: add the three golden configurations using the existing noisy
  wrapper and explicit paper budgets (200, 500, 600); prepare a safe command
  manifest/launcher and lightweight checks under `dev/scripts/`. Verify
  allocation, pinned source and dependency imports in a spawned worker,
  one-worker/thread settings, and preservation of the existing 810 pairs.
  Default invocation must not run optimization; launching requires an
  explicit separate invocation. Do not modify numerical source or the tracker.
- [x] Astra: prepare a dedicated local checkout with a benchmark-only commit
  based on `7314a6a`, retain the original environment without reinstalling,
  record its exact SHA/diff, and verify actual imports and original hashes.
  Isolated base and gpyreg checkouts are created under the ignored
  `dev/scripts/runs/noisy_reference_20260907/` directory; original 810 pairs
  and all published sidecars match the historical manifest. Raw hashes of
  the 1,620 original files are saved there for the final preservation check.
- [x] Update benchmark coverage and pickup documents, recording exact future
  launch/validation commands and distinguishing the current 810 from the
  pending 960 runs. Preserve the historical completion record.
- [x] Run lightweight preparation checks only, then a fresh-context Sol
  review through `$doublecheck`; record results here. No benchmark, replay,
  oracle regeneration, watcher, scheduler, or full test suite in this task.

Acceptance: a reviewable, pinned, import-verified 3 × 50 allocation with
explicit `vectorized_target=False`, one worker and single-threaded BLAS;
the existing 810 reference pairs unchanged; zero new optimization runs.

**Prepared source identity:** local `reference/noisy-20260907` is
`623f5cd91925c6e6b6a23d2985a66bcb1d6bea86`, a single benchmark-only commit
on `7314a6afe158c5673775ca79601b88b3913ae383`. Its entire diff changes
`dev/scripts/benchmark_targets.py` only (three golden registrations and their
comments); the profile suite and all numerical source remain unchanged.
The original `reference/stage3-20260906` still points to `7314a6a`.
The dedicated checkout is detached at the new pin in
`dev/scripts/runs/noisy_reference_20260907/source/`; its isolated gpyreg
neighbor is detached at `a2f8ddce867f502e29717959cf0ff3529f598618`.
Both checkouts are clean. Local Git bundles in their parent directory retain
both sources; no remote publication was performed.

An independent isolated-interpreter probe used the original
`.venv/Scripts/python.exe -I` with those source paths inserted before imports.
It verified `pyvbmc.__file__`, `gpyreg.__file__`, `benchmark_targets`,
`golden_trace`, and `profile_run` all resolve inside the pinned checkouts,
Python 3.12.6 / NumPy 2.5.2 / SciPy 1.18.1 / gpyreg 1.1.0 / cma 4.4.4,
and OMP/OPENBLAS/MKL thread environment values of 1. The golden registry has
20 configurations, with the three new noise-SD-3 budgets 200/500/600.
The probe made no target calls or optimizations. Its initial attempt to
inspect loaded thread pools found `threadpoolctl` absent; it was rerun
successfully checking the configured thread environment without installing
anything. Evidence: ignored `independent_import_probe.json` in the preparation
directory. The launcher will repeat its own worker checks before any future run.

The first end-to-end `prepare` tooling check exited 0 and produced all 150
tasks with correct isolated-worker provenance; its extension directory was
empty. This was preparation only, not an end-to-end solver test. It used
temporary `checks/prepare_probe/` artifacts. The canonical manifest below is
now frozen from the final reviewed helper; its `prepare` command exited 0.

**Outcome: preparation complete; nothing launched or scheduled.** The
canonical `preparation.json` contains all 150 tasks and actual worker import
provenance. Its frozen launcher SHA256 is
`c4a7f2381913debbe20c9c1afc602f4bb0a5a7cc36b68fee80d16e876b32e3a1`.
The new `extension/` directory is empty, no launch lock exists, and a final
raw-byte check confirms all 1,620 original JSON/NPZ files unchanged.

Verification: seven focused tests passed without reruns (final run 1.32 s):
`python -m pytest pyvbmc/testing/test_reference_noisy_extension.py -q`, using
the original venv, single-threaded BLAS, and workspace-local `--basetemp`
and `cache_dir` under `checks/`. The first attempt hit existing system-temp
ACL restrictions during four fixture setups; the workspace-temp run passed.
The schema validator accepted all 810 existing traces without optimization.
The worker environment also returned truthful pinned `profile_run.git_info()`
metadata (`623f5cd`, dirty=false). Syntax and sequential in-process Black
checks passed for the new helper/tests. Stalled multiprocessing Black checks
were stopped; the newer installed formatter's unrelated legacy benchmark
linewrap was left unchanged. Fresh-context Sol `$doublecheck` finished with
no remaining findings after source/manifest, worker authorization, resume,
trace-validation and task-tag corrections. No numerical suite, replay,
oracle, population run, Q1/Q4 experiment, or PyTorch work was performed.
The broader phases below remain pending; launching is a separate next action.

**Preparation and future commands (PowerShell, repository root):**

```powershell
.venv/Scripts/python.exe dev/scripts/reference_noisy_extension.py prepare `
  --python .venv/Scripts/python.exe `
  --reference-checkout dev/scripts/runs/noisy_reference_20260907/source `
  --expected-sha 623f5cd91925c6e6b6a23d2985a66bcb1d6bea86 `
  --gpyreg-checkout dev/scripts/runs/noisy_reference_20260907/gpyreg `
  --expected-gpyreg-sha a2f8ddce867f502e29717959cf0ff3529f598618 `
  --population dev/scripts/runs/golden/item7_20260906 `
  --historical-manifest dev/golden/extension_20260907/sha256_manifest.json `
  --out dev/scripts/runs/noisy_reference_20260907/extension `
  --manifest dev/scripts/runs/noisy_reference_20260907/preparation.json
```

`prepare` checks sources, allocation, original hashes, and actual imports in
an isolated child interpreter, then freezes the launcher beside its manifest.
It performs no optimization. The manifest and launcher paths above are
machine-local; retain the dedicated source checkouts and original environment.
The default invocation prints help. Re-preparation refuses to overwrite its
existing manifest or frozen launcher; use a fresh manifest directory if
preparation inputs must change after investigating the reason.

**Future launch, only after a separate PI instruction; not run in this task:**

At that point perform any reference preflight replay before this command,
using the pinned imports and fresh output paths. Numerical replay has been
deliberately excluded from preparation; no new replay pass is claimed here.

```powershell
.venv/Scripts/python.exe dev/scripts/runs/noisy_reference_20260907/reference_noisy_extension_frozen.py run `
  --manifest dev/scripts/runs/noisy_reference_20260907/preparation.json `
  --confirm-run-150
```

Each task uses the original interpreter in isolation, pinned source imports,
one BLAS thread, and explicit `vectorized_target=False`. Output goes into the
separate `extension/` directory; the 810-run source population is untouched.
No scheduler, background watcher, keep-awake request, or launch time is set.
Do not run a test suite, replay, or any other heavy computation alongside it.

**After all 150 runs, before appending or publishing anything:**

```powershell
.venv/Scripts/python.exe dev/scripts/runs/noisy_reference_20260907/reference_noisy_extension_frozen.py validate `
  --manifest dev/scripts/runs/noisy_reference_20260907/preparation.json
```

Investigate any integrity/provenance failures and convergence changes. Build
a fresh combined directory by copying the verified 810 and 150 JSON/NPZ pairs;
verify its exact allocation (19 × 50 plus exhaust × 10), readable traces,
finite metrics, and unchanged historical hashes. Against that combined set,
run `golden_trace.py summary <combined>` and
`golden_trace.py compare --split <combined>` using the pinned source and
reference interpreter: 80 KS tests, Holm alpha 0.05, with failures surfaced.
Perform the final reference replay when numerical runs are authorized,
recording its actual coverage; the stronger replay-harness work below has
not been implemented by this preparation subset.
Only after the checks and any flagged-outcome investigation, append the
validated new pairs to the local reference and extend the tracked sidecars,
summary, and reference hash manifest. Retain the historical 810-run execution
record unchanged and describe 960 as current only after that publication.
The final 960-run release-candidate comparison remains a separate later task.

- D1–D5, Q2 and Q3 are approved. The agreed Q1/Q4 experiments and isolated
  preparation may proceed; settle their final scientific choices before
  finalizing the dependent production treatment/default. Record environment,
  base SHA, gpyreg SHA, and threads.
- Run exact existing oracles and the default five-case replay once before
  numerical edits. Keep outputs in a fresh ignored `dev/scripts/runs/latent_fixes/`
  directory and record actual commands, exit codes and reports in this plan.
- Fix the replay's coverage gap in a separate harness change: currently
  `golden_replay.py` calls a run identical from main-loop ELBO/live points
  alone. Require exact equality (including shape/length, and equal NaNs only
  where the stored format intentionally permits them) for every existing
  non-timer NPZ array: iteration numbers, ELBO/SD, sKL, reliability, stability,
  warmup, Ns_gp, func_count, n_eff, pruned, K, N, warped, GP hyperparameters
  and iteration mapping, VP parameters and iteration mapping, transformer
  arrays, initial/live X/y, returned VP parameters and posterior moments.
  Only `timer` is excluded from the current NPZ schema. Keep toleranced
  diagnostics distinct from the exact identity verdict. Compare returned
  final VP arrays and semantic sidecar final fields: `elbo`, `elbo_sd`,
  `final_K`, `best_iter`, `success_flag`, `message`, `iterations`,
  `func_count`, `final_N`, `min_Ns_gp`, and `n_warps`. Duplicated counts must
  agree with the NPZ. Exclude timing/memory/provenance metadata from identity
  and keep accuracy metrics under their separate fences. Report
  “same loop, changed final” separately and applying accuracy fences to that
  case. Compare the transformer used by the returned VP where stored; retain
  backward-compatible “not certifiable” reporting if old traces lack it.
  Test this with synthetic copied trace/sidecar data, changing each required
  field individually (and array length), and assert non-identity; changes to
  timers alone must not part the run. No new solver runs for harness tests.
  Re-render the saved preflight with `--report-only` under the repaired
  harness to validate the stronger gate without repeating those solver runs.
- A neutral group must preserve final outputs as well as the main loop.
  A moving group must have an explained first difference, identical initial
  design, finite outputs, and no unexplained accuracy-fence failures.
- Keep reference population and oracle reference changes separate: only the
  narrowly named oracle outputs described below may be deliberately updated.

### Phase 1 — Neutral fixes, utility contracts, and compatibility

**Executor:** Sol implementation; Astra owns acceptance and all heavy checks.

1. Implement boundary dtype and floating-placeholder fixes first, using
   `test_vbmc_init.py`, the variance-path tests and `testing/_dtype.py` /
   `testing/oracles/test_oracles.py`. Remove obsolete canary exemptions.
   Verify the full state is float64, not just returned values.
2. Repair transformer equality, identity assertions, and logit tails. Validate
   orthogonal rotations at construction with a tolerance appropriate for
   float64 SVD rotations; include reflected/near-orthogonal examples and
   reject nonorthogonal/singular matrices. Keep old valid pickles loadable.
3. Repair local KL input normalization, the cubic typo, pruned `J_sjk`, and
   Q3's approved fixed-sigma case. Use existing modules under `testing/stats/`,
   `testing/parameter_transformer/`, and `testing/vbmc/`; test pruning with
   distinguishable entries in both axes so a shape-only test cannot pass.
4. Repair optional truth diagnostics: distinguish absent/empty values from
   arrays, validate `(D,)`/`(1,D)` means and `(D,D)` covariances, and retain
   the existing finite-value behavior. Give a diagnostic VP copy a deep copy
   of `vp.rng` explicitly (VP deepcopy shares RNG). Keep the million-sample
   estimator and count unchanged. Unit tests use mocked moments and compare
   generator state before/after; no million-draw pytest fixture.
5. Fix `finalize()` to trim every row array including `n_evals` consistently.
   Test empty, noisy, duplicate, inactive-row and repeated-finalize cases,
   then append/evaluate again. Keep `X_flag`, row indices and counts intact;
   do not compact inactive rows or invoke finalize from optimize under D4.
6. Set results RNG state from the return-time generator, as an independent
   deep copy in the existing snapshot format. Test reproducing subsequent
   draws and mutation isolation; extend existing mocked-result/seed tests.
7. Apply approved Q2. Test both density modes reject original gradients
   before doing work, while transformed density/log-density FD tests remain
   green. Update `docsrc/source/api/classes/variational_posterior.rst` and docstrings.
8. Remove Product's unused private import and unused Iterable import. Use
   `scipy.stats.rv_continuous` for the univariate frozen object's `dist`
   check, and obtain multivariate frozen types via public MVN/MVT factories
   in one private compatibility helper. Keep exact accepted categories;
   do not substitute arbitrary method presence checks. For MVN/MVT use
   `.dim` rather than drawing a sample to infer D. Test continuous univariate,
   MVN, MVT acceptance and discrete/unfrozen/unsupported rejection, plus
   unchanged global and frozen-distribution RNG states during construction.
   Update private-type assertions in `test_convert_to_prior.py`; verify
   against the actual resolved versions in the existing CI matrix. Also run
   a focused `testing/priors/` check in a separate Python 3.12 environment
   with `scipy==1.15.0` and a resolver-compatible NumPy, installing `.[test]`;
   the current CI matrix does not pin SciPy's lower bound. Record versions
   and results without modifying the reference venv. No dependency/floor change.
9. Correct notebooks 1/6 and regenerate only their generated scripts. For
   notebook 1 verify approximately -2.2598 by the existing analytic inner
   integral/1-D quadrature; for notebook 6 use rowwise norms and matching
   `(N,)` value/noise arrays, checking N=1 and N>1. Validate JSON and compile
   generated scripts. Remove unused `noisy_cigar`. Correct option comments
   for `display`/`log_file_level` to `off`, `iter`, `full`.
10. Record already-fixed items as closed and D5 hooks as retained, with honest
    comments/docs about built-in use. Avoid changing serialized names or
    deleting accepted options. Run relevant focused modules, then neutral
    exact-oracle and replay gates before phase 2.

### Phase 2 — Guard final boost before changing main-loop trajectories

**Executor:** Sol implementation; Astra scientific acceptance.

**Decision status:** the rule's form and comparison at tolerances 0.1/0.2
are selected. Implement and evaluate that comparison without presuming a
final default; obtain the PI's default selection after expanded noisy
validation before finalizing the production option value.

1. Save an independent pre-boost VP and its stored ELBO/SD before mutating
   candidate flags/state. Set `weight_penalty=0` only in the boost's copied
   options, retaining `tol_weight=0`: final refinement has neither weight
   penalty nor pruning (PI-selected direction, 2026-09-07). Main-loop weight
   regularization remains subject to Q1. Run boost exactly once with its
   existing entropy settings and shared RNG; optimization and draw counts
   may change with the objective. Do not re-evaluate the old VP, rewind
   the RNG, change beta/pruning, or tune component bounds.
2. PI-selected rule to test: for finite before/after ELBO and nonnegative
   finite SD, let dE=candidate_elbo-pre_elbo and dS=candidate_sd-pre_sd.
   Accept exactly when `min(dE, dE-5*dS) > -tol_elcbo_boost`, testing both
   0.1 and 0.2 without selecting a default in advance.
   This covers the full continuum b in [0,5]; no grid or new evaluations are
   needed. Document the tolerance option in the advanced configuration and
   validate it as finite and nonnegative. A drop exactly equal to the
   tolerance is rejected under the PI's strict inequality; identical scores
   pass at positive tolerance. Final default approval follows the expanded
   noisy comparison. Reject a failing or invalid candidate when the pre-boost score is valid,
   warn, and return the untouched pre-boost VP and its stats. If the pre-score
   is invalid, accept a valid candidate; if neither is valid after an attempted
   boost, raise a clear RuntimeError that no finite posterior can be returned.
   Define/test `changed_flag` as whether the returned VP was boosted, not
   whether an attempt ran. Preserve history and transformer/RNG identity rules.
3. Extend `test_vbmc_finalboost.py` with distinct mocked candidates for better,
   equal, worse and nonfinite cases, including the historical -10.340/0.044
   versus -9.031/0.494 example. Assert returned parameters, flags, stats,
   both possible worst endpoints, exact tolerance-boundary behavior, and
   equivalence to a dense b-grid for representative finite inputs. Assert
   warning and source nonmutation. Verify result ELBO, SD and K describe the
   selected VP; no extra model evaluations or acceptance-specific RNG draws.
   Assert the boost receives zero weight penalty and zero pruning threshold,
   while the original options retain their configured values. Check that
   the boost weight-regularization contribution and gradient are zero.
4. Run default replay plus current `logreg_D5` seed 5 and `student_D4` seed 19
   with unique output paths. The pre-boost loop should remain identical;
   assess returned posterior changes explicitly. Historical student seed 19
   is a separate stored-evidence regression, not a promised current failure.
   Record rejection counts from the eventual population, without a quota.

### Phase 3 — Repair weighted GP covariance and its oracle contract

**Executor:** Sol implementation; Astra validates estimator and evidence.

1. In `_get_hyp_cov`, take hyp_N from the current model's hyperparameter count
   (`hyp_dict["hyp"]` with its shape validated, or the GP count passed by the
   caller), not the newest historical block. Use `(Ns,hyp_N)` sample rows,
   allow ragged Ns across history, skip incompatible parameter dimensions,
   allocate each iteration
   total weight w, normalize row weights, and form the unbiased weighted
   outer-product covariance. Fix the sKL decay denominator to
   `tol_skl * fun_evals_per_iter`. Keep truncation and unweighted `run_cov`
   behavior. Return None for no usable samples or an undefined single-effective-
   sample covariance so existing default widths apply; no new jitter policy.
2. Replace commented-out weighted coverage in `test_gaussian_process_train.py`
   with hand-computed asymmetric data: ragged samples, Ns==hyp_N and unequal,
   multiple weights/decay, cutoff, a newer incompatible block followed by an
   older compatible block, and degenerate
   history. Assert shape, symmetry, values, and delivered sampler widths;
   a shape-only or constant matrix test is insufficient.
3. The current `gp_fit` oracle uses synthetic history and assumes widths are
   discarded. Preserve its old numerical contract explicitly: on a private
   options copy disable history covariance/widths (weighted_hyp_cov=False,
   run_cov=None in the copied hyp_dict, gp_sample_widths=0). Verify every old
   `gp_fit` output remains exact; if not, investigate before changing refs.
4. Add a distinct `gp_fit_history` oracle with documented deterministic ragged
   history built from existing snapshot hyperparameters, plus known offsets
   to exercise unequal variances/decay. Return widths alongside fit outputs.
   Its synthetic history is an explicit input, never described as the real
   historical fit. Unit-test that construction against the hand estimator;
   classify its sampling outputs as platform-bound. Add references with
   `--add-oracle gp_fit_history --reason ...` from stored snapshots, never
   rerun snapshot recipes. Update oracle inventory/platform handling/docs.
5. Add **three authentic-history fixtures** (PI-approved addition, 2026-09-07)
   alongside the legacy and controlled-history oracles: an early sampled fit,
   a later sampled fit with changing historical sample counts, and a noisy
   sampled fit with nonuniform historical weights. All three must actually
   use history-derived widths; an optimization-only fit does not cover this.
   Capture each call's matching inputs immediately before `train_gp`: logger
   data, options, optim_state, hyp_dict, the history entries consumed for
   covariance and warm starts, and the generator state. Copy these before
   the call can mutate them. Record the observed fit outputs and verify an
   independent rebuild/replay reproduces them, including delivered widths.
   Use the corrected covariance code after steps 1–4 pass, with code and
   dependency versions, seed, capture iteration and platform recorded.
   Multiple fixtures may come from one bounded capture run; stop capture
   runs after obtaining the needed states. These are one-time development
   runs, not new full optimize runs in pytest or a population campaign.
6. Store the three additive fixtures as plain JSON/NPZ in a dedicated
   `pyvbmc/testing/oracles/fixtures/gp_fit_history/` subset, with a focused
   replay test under `pyvbmc/testing/oracles/`. Reuse existing state encoding
   and GP reconstruction helpers; serialize only the history needed by the
   fit, not full Cholesky factors for every historical GP. Add a targeted
   capture mode to `dev/scripts/make_oracle_fixtures.py` that does not rewrite
   existing snapshots, and include this subset in its check workflow. Test
   history-derived covariance/widths across platforms; keep stochastic fit
   output comparisons under the existing generating-platform policy and
   float64 checks. Add the new directory's JSON/NPZ patterns to MANIFEST.in
   and document capture/replay in the existing oracle plan. Hash-check old
   fixture files unchanged by the additive capture; this extends coverage,
   not the 810-run reference or its saved histories.
7. Exact-check unrelated stage outputs and replay this group before any
   acquisition or stop repair. Record its own first divergence and metrics.

### Phase 4 — Activate acquisition regularization safely

**Executor:** Sol implementation; Astra reviews intended numerical changes.

1. Use `variance_regularized_acq_fcn` for new state. Read the legacy
   `variance_regularized_acqfcn` alias when canonical is absent; canonical
   wins, including False. Cover new instances, old saves/resumes, and fixture
   rebuilds: old snapshots must exercise the newly active behavior, not stay
   silently on the dead path. Preserve absence/False behavior for custom states.
2. Validate/flatten the acquisition result to `(M,)` before regularization
   and bounds masks. Accept the previously documented vector/row/column
   forms and reject incompatible matrices. Preserve the existing penalty
   equations and threshold. Handle zero variance as its limiting penalty,
   without NaNs or divide warnings; do not add a new variance floor.
3. Extend `test_abstract_acquisition_function.py` and real VIQR/IMIQR tests
   for one/multiple GP samples, M=1/M>1, mixed threshold masks, log/non-log
   functions, zero variance and invalid shapes. Expected penalized values
   come from the stated formula; test batch versus pointwise equivalence.
4. Update `test_vbmc_init.py`'s key assertion. Inspect every affected `acq_*`
   oracle discrepancy against that formula on the stored candidate points.
   Only then deliberately update those named references and
   `active_sample_step`, with audit reasons and `--expect-moving` naming only
   the other confirmed-moving oracles. No blanket fixture regeneration;
   all unrelated outputs remain exact. Replay separately before phase 5.

### Phase 5 — Restore variance-based GP sampling termination

**Executor:** Sol implementation; Astra verifies MATLAB semantics.

1. Use `stop_sampling` for the existing guard and assignment. Record existing
   `optim_state["N"]` each iteration, registering it in IterationHistory;
   reuse `var_ss` for GP sample variance. Despite the existing "training
   inputs" comment, N counts logged distinct locations (`Xn+1`), including
   inactive rows; it is neither live GP row count, evaluation count nor n_eff.
   Pinned MATLAB `vbmc.m` likewise sets N from its 1-based Xn. Preserve this
   bookkeeping for this repair; changing global N or hard-stop semantics
   would be a separate algorithm decision.
2. On iterations >=2 outside warmup while stop_sampling==0, slice populated
   history through the current iteration and use weights
   `0.5*one_hot_current + 0.5*normalize(exp(-(N_last-N_i)/10))`.
   Evaluate exponentials after subtracting their maximum to avoid overflow
   following trimming. Stop when weighted var_ss is strictly below the
   existing `tol_gp_var_mcmc`; missing/nonfinite data must not cause a
   premature stop. Do not introduce a reliability-based history suffix.
3. Add a compatibility helper for load and direct continuation of old
   objects: register/backfill N from each recorded `optim_state["N"]`, or
   that iteration's logger `Xn+1` if necessary. Do not substitute X_flag
   counts or gp.X row counts after trimming: they describe a different
   quantity. Leave unavailable entries explicitly missing and
   skip this optional criterion until its history is usable. Do not invent
   variances or replace missing history with zeros. Preserve hard N/K stop
   behavior and the option's ability to disable the variance criterion.
4. Replace the fake-key test in `test_vbmc_loop_termination.py` with populated
   real histories: low/high/latest-high variance, equality at threshold,
   warmup/early/already-stopped guards, large N, trimming with present N
   exceeding the live GP size, duplicate noisy
   observations and missing old records. Assert the next `_gp_hyp` call uses
   stable samples after stopping. Extend save/load compatibility coverage
   without regenerating static pickles or adding full optimize tests.
5. Replay this group separately, with a targeted case forced to cross the
   variance criterion if default cases do not. Existing pointwise oracle
   computations should stay exact; a replay moving earlier is expected.

### Phase 6 — Compare weight-bound treatments, then select the correction

**Executor:** Sol implementation; Astra mathematical review.

**PI-selected investigation (2026-09-07):** compare (a) no eta-bound penalty,
(b) MATLAB's historical formula with a consistent implementation, checked
against the original paper where documented, and (c) current behavior.
Arm (b) is a historical diagnostic, not the presumed correct design or an
approved production choice. Arm (c) is diagnostic only, never a shipping option.
The separate small-weight shrinkage term and pruning are held fixed across
these main-loop comparisons to isolate the eta-bound treatment. The final
main-loop choice remains open until the comparison is discussed with the PI.

**Design rationale to retain (PI discussion, 2026-09-07):** a correct
derivative does not establish that the objective is sensible. MATLAB's
absolute-eta bounds assign different penalties to eta vectors representing
the same posterior. They mix control of the redundant common offset with
restrictions on weight ratios; MATLAB parity does not justify that mixture.
Keep three concerns separate: stable softmax for numerical stability;
an explicit parameterization or centering convention, with consistent
derivatives, if the redundant offset needs handling; and scientifically
motivated preferences expressed through normalized weights. This is a design
principle, not approval to introduce a new parameterization in this fix.
Neither historical treatment has to survive. If removing the eta penalty
performs well, do not retain one without demonstrated benefit. If (b) or (c)
performs better, investigate which useful effect it provides before choosing
a justified treatment; empirical success alone does not validate the formula.

**Investigation before production edits:**

- Establish the mathematical contract for (b). The pinned MATLAB
  `misc/negelcbo_vbmc.m` at `396d649c3490f1459828ac85f552482869edf41c`
  passes original theta to `vpbndloss`; it does not subtract max(eta) first.
  The full chain was inspected on 2026-09-07: `misc/vpbounds.m` sets
  a=log(0.5*TolWeight), b=0; `misc/vpbndloss.m` extracts raw eta from theta
  and passes it unchanged to `utils/softbndloss.m`. With
  s=(b-a)*TolConLoss, each component contributes
  0.5*((a-eta)_+/s)^2 + 0.5*((eta-b)_+/s)^2. Its derivative is
  (eta-a)/s^2 below a, zero inside [a,b], and (eta-b)/s^2 above b.
  Those derivatives are mathematically consistent with the inspected loss;
  this source check did not execute MATLAB. Arm (b) therefore uses raw-eta
  bounds, with stable max-shifting confined to private softmax calculations.
  The separate capped-weight term is also present in MATLAB and remains
  fixed across the three arms. The original paper's
  local transcription, `papers/acerbi2018variational_appendix.md` A.3.1–A.3.3,
  describes softmax and its gradient Jacobian, but does not specify this
  penalty there; the main paper §3.6 describes pruning. Distinguish explicit
  paper statements from MATLAB implementation details, and surface any
  discrepancy rather than inventing paper support. In particular, merely
  differentiating the current relative-weight penalty correctly is not (b).
- Use isolated experimental variants: (a) removes only the eta-bound loss
  and gradient, (b) implements the source-verified loss and gradient, and
  both preserve caller theta with stable private softmax arithmetic; (c)
  preserves the existing loss, gradient, and mutation behavior exactly.
  Verify (a)/(b) by finite differences, including active bounds and common
  eta shifts, before interpreting performance. Do not rebaseline oracles
  for experiments or alter the frozen reference.
- Start with bounded, paired variational-optimization replays from authentic
  saved states covering early/later fits, noisy/noiseless targets, and both
  active and inactive eta bounds. Reconstruct identical GP/VP/options and
  independent generators initialized to the same captured state per arm;
  VP deepcopy alone does not isolate the generator. Hold candidate starts,
  optimizer settings, budgets, and all other fixes constant across arms.
- Record violation frequency/magnitude, eta offsets and ranges, supplied
  versus correct penalty gradients relative to the ELBO gradient, objective
  progress, stopping reasons, weight distributions, pruning, and runtime.
  Compare resulting posteriors using a common independent, unpenalized
  ELBO/uncertainty evaluation with paired draws; do not rank arms by their
  differently penalized training objectives. Follow promising local results
  with a bounded paired set of whole trajectories and available ground-truth
  posterior/evidence metrics. Scope that set explicitly before launch; this
  is not authorization to rerun the 810-run population. Analyze pre-boost
  outcomes separately, holding any boost treatment identical across arms.
- If (c) reproducibly outperforms both (a) and (b), investigate before choosing
  a replacement: separate mutation effects from the erroneous gradient,
  raw-eta versus relative-weight semantics, optimizer/stopping interactions,
  and stochastic evaluation noise. Any extra ablation serves that diagnosis;
  superior measurements do not authorize retaining the inconsistent pair.

**Prerequisite for the production correction:** discuss the comparison and
select the penalty treatment with the PI. Rewrite the chosen objective and
gradient contract explicitly before delegating production implementation.

1. Remove mutation of the caller's theta and retain stable private softmax
   calculations. Implement the PI-selected penalty and its consistent
   gradient: if applied to raw eta, use original theta; if applied to
   eta-max(eta), account for that normalization in the penalty gradient.
   A redesigned penalty or its removal requires an explicit PI decision.
   Preserve parameter layout and entropy sampling; change no unrelated
   optimizer settings.
2. Extend `test_variational_optimization_grad_fd.py`: assert theta unchanged,
   exercise upper and lower eta bounds and common eta shifts, and check the
   complete bound-augmented gradient with deterministic and common-draw MC
   entropy. With no bounds, the density/ELBO stays invariant to common eta
   shifts. With bounds, assert the shift behavior of the selected penalty:
   a relative-weight penalty stays invariant, while raw-eta bounds need not.
3. Inspect affected `neg_elcbo` oracle outputs analytically; no change to
   standalone GP/entropy/PDF oracles is expected. Deliberately rebaseline
   only confirmed affected names, preserving stored states and other refs.
   Run its own replay. If the gradient or MATLAB comparison contradicts the
   contract, stop this phase for Astra/PI adjudication, not tolerance changes.

### Phase 7 — Upstream fix and final integration

**Executor:** Sol for bounded gpyreg repair/docs; Astra for integration/gates.

1. On an isolated gpyreg feature branch, resync both step-out bracket vectors
   from accepted current coordinates before processing each dimension. Test
   a correlated target or instrumented log density that proves every bracket
   evaluation lies on the current coordinate line; keep step_out=False draw
   order unchanged. Run gpyreg checks as the sole heavy process. Prepare an
   upstream PR; publication requires the user's normal explicit authorization.
   Do not overwrite the sibling checkout or reference environment.
2. Advance `GPYREG_PIN` only after upstream integration and green checks on
   the candidate pin; validate PyVBMC exact oracles/replay and CI against it.
   If upstream is pending, report it as pending instead of claiming it fixed
   in the dependency used by PyVBMC. Do not bump the pin speculatively.
3. Run the full existing suite once with CI's reruns and one BLAS thread,
   after focused checks pass. Run the full supported OS/Python matrix for
   the integrated release branch as part of the existing release process.
   Preserve Stage 3 optional-export tests and static save/load fixtures.
4. Update this plan with actual evidence and remaining decisions; update
   roadmap pickup 9, `dev/TODO.md`, and `dev/README.md`'s index. Update affected
   API docs and option comments, and relevant oracle design records. Keep
   the golden README about its unchanged reference, not batch chronology.
   Correct stale status pointers without rewriting the historic findings
   or PyTorch feasibility decisions.

## Verification commands and population gate

Use the original reference venv for trajectory checks. Set PowerShell:

```powershell
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
.venv/Scripts/python.exe dev/scripts/make_oracle_fixtures.py --check --exact
.venv/Scripts/python.exe dev/scripts/golden_replay.py --out dev/scripts/runs/latent_fixes/preflight
```

Use a distinct replay output path per group. For the boost's targeted cases:

```powershell
.venv/Scripts/python.exe dev/scripts/golden_replay.py --configs logreg_D5 --seeds 5 --out dev/scripts/runs/latent_fixes/boost_logreg
.venv/Scripts/python.exe dev/scripts/golden_replay.py --configs student_D4 --seeds 19 --out dev/scripts/runs/latent_fixes/boost_student
```

Focused tests use `python -m pytest <affected modules> -q` first; deterministic
regression tests must pass without reruns. Run `python -m pytest --reruns=5 -x
-vv` for final integration. Run formatting on edited files, build relevant
Sphinx sources if API docs change, and inspect generated notebook scripts.
Do not repeatedly run the whole suite after passing without a new reason.

For neutral groups require exact oracles and identical default replay,
including final outputs. After a moving group record exactly which oracles
are expected to differ. Review direct math/unit evidence before targeted
rebaseline, use audit reasons, and exact-check every unaffected reference.
Existing MATLAB NPZ values are immutable; explain any intended disagreement
with a broken path rather than regenerate MATLAB data or loosen tolerances.

After the accepted fixes, integration/compatibility work with the existing
S-VBMC package, and the remaining PyTorch release decisions and resulting work
are complete, freeze the final candidate and prepare a single-worker campaign
matching **19 configurations × seeds 0–49 plus cigar_D15_exhaust × 0–9**
after the approved noisy reference extension is validated.
Use explicit `vectorized_target=False`, the allocation in
`dev/golden/extension_20260907/README.md`, and fresh output directories.
Do not use `regenerate_baseline.sh` or overwrite reference pairs. The
rough 12-hour estimate is historical, not a promised runtime after fixes.

On explicit PI launch instruction, run and validate 960 complete readable
JSON/NPZ pairs, no errors, finite metrics, correct configuration/seed coverage,
750 evaluations for every exhaust run, and truthful SHA/options/dependency/
thread metadata. Check convergence statuses of the other runs and investigate
changes. Compare with `golden_trace.py compare dev/golden/baseline <new_dir>`
(80 KS tests, Holm alpha 0.05) and run an even/odd null comparison. Examine
accuracy, usability and evaluation counts; a statistical flag requires
investigation, not automatic rejection or approval, and no flag is not proof
of equivalence. Record guard rejection counts and pre/post scores separately.
Final comparison reports and PI disposition precede any reference promotion
or release. Keep the reference used for release validation identifiable.
Hand off roadmap pickup 8 explicitly: at release, archive the NPZ traces of
the PI-designated reference used to validate that release, one zip per
reference, attach the archive(s) as release assets, and update the golden
README with the asset names and extraction path. Retain clear identities for
the expanded 960-run reference and the new candidate population; do not
silently substitute either for the other. Packaging/publication is a later
release action, not authorized by this planning task.

## Approved decisions

D1–D5 were approved by the PI on 2026-09-07. The agreed order is neutral
("no-op" for existing reference runs) fixes first, then the boost guard once
its rule is settled, then the other trajectory-moving fixes. Q2 and Q3 were
subsequently approved. Q1's treatment and Q4's default await the agreed
experiments; the comparison designs and 150-run reference addition are settled.

- **D1: Neutral fixes first, then boost after settling its rule, then broad
  trajectory changes.** The boost's effect can be
  isolated on the existing loop. Rejected: fixing all core keys together,
  which is faster to code but hides causes of changed results.
- **D2: Preserve the legacy gp_fit oracle and add an explicit history oracle.**
  This retains an exact fitter/sampler gate and honestly tests adaptive
  widths. Rejected: silently make repeated stand-in history authoritative,
  or rerun snapshot recipes to reconstruct history (moves unrelated states).
  **Approved addition (2026-09-07):** capture three authentic-history fixtures
  for representative integration coverage (early sampled, later changing-Ns,
  and noisy sampled fits). These supplement the legacy and controlled-history
  tests; they do not replace existing fixtures or their references. Phase 3
  specifies capture, provenance, packaging and replay requirements.
- **D3: Keep optional truth diagnostics off the solver RNG.** They should
  observe a run without perturbing it. Rejected: repair only array truthiness,
  which leaves diagnostics changing the inferred posterior. Default runs stay
  exact; diagnostic-enabled legacy trajectories intentionally change.
- **D4: Repair explicit finalize; do not invoke it automatically.** The live
  logger's allocation is useful for continuation, while compact histories
  already address the major memory cost. Rejected: auto-finalize now, which
  changes live-storage behavior without a demonstrated need. The alternative
  remains available if the PI wants automatic end-of-run trimming.
- **D5: Retain dormant compatibility hooks and narrow rotation/prior contracts.**
  Do not delete custom acquisition hooks or accepted saved options merely
  because built-ins do not use them. Reject invalid rotations instead of
  adding general affine support; recognize frozen SciPy types via public
  factories instead of broadening accepted priors. Rejected: wholesale dead-code/API
  deletion or feature expansion in this correctness pass.

## PI questions and resolutions

- **Q1 — Eta soft bounds (open; PI clarification 2026-09-07):** the purpose
  of the penalty needs discussion; restoring MATLAB behavior is not presumed
  desirable. The quadratic bounds are a=log(0.5*tol_weight), b=0, scaled by
  (b-a)*tol_con_loss. With current max-shift normalization, eta_k becomes
  log(w_k/w_max), so the lower bound penalizes w_k/w_max < 0.5*tol_weight
  (1/200 at defaults); the upper bound cannot fire. Absolute-eta bounds also
  restrict the arbitrary common offset, although some offset fits all bounds
  exactly when the same weight-ratio condition holds. The separate penalty
  weight_penalty*sum(min(w_k, max(1/(4K),tol_weight))) encourages tiny weights
  toward zero and is a different design choice. When tol_weight=0 the current
  infinite-width scaling makes the eta-bound penalty zero for finite eta.
  During final boost, `tol_weight=0` disables that bound and pruning, but
  currently leaves the shrinkage term active below 1/(4K). The PI selected
  disabling this remaining term in boost (`weight_penalty=0` in copied
  options), paired with the planned acceptance gate: let the ELBO determine
  final weights without an additional sparsity preference. Retaining it
  could suppress unnecessary components but also discourage useful small
  components despite pruning being disabled. This settles the boost-specific
  direction; main-loop regularization and the exact gate remain open.
  Distinguish the clear input-mutation bug from choosing raw-eta bounds,
  a properly differentiated relative-weight penalty, or another PI-selected
  treatment. A bounded deterministic-entropy probe on the existing fixture
  with eta=[0,-7] found an additional gradient defect: the objective was
  identical after a common +2 shift, but its returned eta-gradient summed
  to -606.1814; central differences (h=1e-5) gave [606.1621,-606.1621], sum 0,
  versus the returned [-0.01938,-606.1621]. The gradient omits the max-shift
  contribution when the lower bound is active. No implementation was changed.
  The PI selected the three-way investigation in phase 6: no eta-bound
  penalty, a correct MATLAB/paper-checked implementation, and current behavior
  as a diagnostic reference. Keeping the current inconsistent implementation
  is excluded; if it wins, investigate why rather than accepting it. This
  approves the comparison direction, not a final main-loop penalty choice.
  MATLAB's formula is also scientifically questioned: a consistent gradient
  does not justify penalizing arbitrary representations of the same posterior.
  Arm (b) is a historical diagnostic, not the default solution. Preserve the
  separation of numerical stability, redundant-offset handling, and actual
  weight regularization described in phase 6; neither (b) nor (c) must survive.
- **Q2 — Original-space PDF gradients (approved, 2026-09-07):** raise NotImplementedError
  for both original-space gradient modes, consistent with existing logpdf
  and the documented unsupported capability. Alternative: implement the full
  transform chain rule (affine and bounded transforms, rotations, density
  Jacobian derivative, boundary behavior), a larger API/mathematical task.
  This is inherited from MATLAB, so agreement is not a correctness argument.
  Reconsider full support with PyTorch autodiff if that port proceeds; this
  is not an additional port commitment. Compatibility check: all four source
  modules of `acerbilab/svbmc` at `13a78f6c4a3ffe9c4557fee4f4b8f67c98b29e01`
  were inspected. `SVBMC.stacked_entropy` evaluates component densities using
  SciPy plus transformer/Jacobian values and differentiates only stacking
  weights through torch; it does not request PyVBMC PDF gradients.
- **Q3 — Frozen sigma with component growth (approved default, 2026-09-07):** preserve supplied
  widths for existing components and cyclically copy those widths for new
  components in the dormant type-3 initialization, without new sigma draws;
  retain the existing center jitter. Alternative: explicitly reject component
  growth when sigma optimization is disabled, avoiding an initialization
  convention. Both fix the shape failure; initializing all widths randomly
  despite the frozen flag is not proposed. Cover K=1/K>1 and K_new==K/>K.
  The failure requires a manually/custom-set `vp.optimize_sigma=False`
  combined with type-3 candidate initialization; normal VP construction sets
  this flag True and no production assignment turns it off. Growth adds a
  shape failure; with unchanged K the existing zero widths are still invalid.
- **Q4 — Final-boost acceptance rule (reopened by PI, 2026-09-07):** the guard
  now has a PI-selected test rule: score loss strictly less than the tolerance
  (test both 0.1 and 0.2) for
  every b in [0,5], checked exactly at the endpoints. The final default is
  pending expanded noisy coverage and validation of the revised boost.
  The planned removal of boost weight shrinkage will change candidates;
  the historical counts below characterize the existing implementation,
  not the prospective rejection rate after that change.
  Offline, the 259 score decreases have median 0.01059 nats, 90th percentile
  0.06205; 31 exceed 0.05, 15 exceed 0.1, and only one exceeds 0.5. Those
  cutoffs describe the data, not calibrated acceptance thresholds. Even
  plain ELBO decreases on 236 stored boosts, so lowering beta alone will not
  remove most reversions. The PI has now chosen to test a tolerant continuum
  guard, rather than requiring strict nondecrease. Retain a separate
  invalid-result rejection. For the initial comparison use the existing
  stored scores as specified in phase 2; examine entropy-estimation precision
  on cases near the thresholds. Re-evaluating both posteriors at common
  precision is a possible follow-up if that noise changes decisions, not
  a prerequisite for starting the agreed comparison. The stored
  ELBO SD is GP uncertainty, not an independent standard error of the score
  difference; do not turn it into a significance test without accounting for
  the shared GP and entropy-estimation error. Subsequent PI discussion
  proposed the continuum b in [0,5] check (exactly reducible to its endpoints)
  and considered tolerances 1, 0.5, then 0.25. The PI selected both 0.1 and
  0.2 for testing, with neither preferred as the default,
  pending broader noisy-target evidence and validation without boost
  shrinkage; the offline findings and approved 150-run reference extension are recorded above.
  No final tolerance or acceptance rule has been approved.

D1–D5, Q2 and Q3 are approved; Q1 remains under investigation and Q4's
selected 0.1/0.2 comparison awaits validation before final default approval.

## Review and rollback

Independent `$doublecheck` was performed on this explicit untracked plan by
two fresh-context `gpt-5.6-sol` reviewers at high effort, one for numerical
contracts/gates and one for scope, compatibility, documentation and release
boundaries. Static review findings were reconciled against source and fixed:

- Preserve existing N bookkeeping, confirmed against pinned MATLAB, rather
  than silently switch the stop criterion to active GP row counts.
- Expand neutral replay identity to all stored non-timer algorithm arrays
  and semantic final outcomes, with explicit synthetic tests.
- Anchor covariance dimensions to the current model, not historical blocks.
- Specify a separate minimum-SciPy prior check; current CI alone does not
  cover the declared lower bound.
- Clarify NPZ publication remains pending, add the release-asset handoff,
  and explicitly retain §9's intentional centering/snapping behaviors.

Q1–Q4 were open at that review; the PI subsequently approved Q2 and Q3.
Later discussion settled the Q1/Q4 experiment designs, 150-run reference
addition, and parallel development. Their final scientific choices remain
open. Handoff verification checked those documentation updates against the
conversation and for consistency; it was not a second independent review
of the enlarged plan. Review did not
run numerical tests or replays. Planning verification consisted of the bounded
probes above, source/MATLAB inspection, Markdown structure/whitespace checks,
and `git diff --check`; no implementation-gate pass is claimed.

During implementation use separate conventional
commits per accepted group, including its tests and relevant narrow oracle
updates. If a group fails a gate, investigate or revert that group; do not
change the reference population, unrelated fixtures, thresholds, or later
algorithm choices to conceal the failure.

Next: prepare the approved reference extension and evaluate Q1/Q4 as above.
D1–D5, Q2 and Q3 need no further approval.
