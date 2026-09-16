# S-VBMC two-level shrinkage estimator

Created: 2026-09-16. Status: **implemented and verified on `dev-next-work`**.

## Live checklist

User approval and `$task` execution request: 2026-09-16. The primary agent
owns this checklist; implementers report outcomes without editing it.

- [x] Phase 1: package estimator, compatibility and focused tests.
- [x] Phase 2: campaign recording, summary compatibility and reference parity.
- [x] Phase 3: API documentation and development records.
- [x] Verification: S-VBMC suite, developer parity and harness checks,
  formatting, Sphinx build and rendered API inspection.
- [x] Independent implementation doublecheck and any resulting fixes.
- [x] Rebase onto the completed PyMC integration on `dev-next`; rerun
  focused tests and independently review compatibility.

## Purpose and scope

Expose the existing full-covariance two-level empirical-Bayes estimate as
`stacked.elbo_details["shrunk_two_level"]`. This plan owns its package
integration, numerical boundary cases and verification. The completed
[Phase 1 reporting plan](svbmc-elbo-reporting.md) owns the existing reporting
contract; the [headline note](../2026-09-15-svbmc-headline-shrinkage.md) and
[campaign plan](svbmc-benchmark-campaign.md) own estimator selection and the
release campaign.

The estimator is evaluated for both noisy and noiseless stacks at the
weights selected by the existing optimization, using the same final entropy
estimate. `elbo`, `headline_method`, `cap_amount`, `elbo_sd`, optimization,
sampling and every existing estimate keep their current behavior. A later
campaign decides whether to promote this estimator to the headline.

`VBMC.optimize()` keeps returning the posterior and results dictionary, and
`SVBMC(vp_list)` keeps accepting fitted posteriors alone. The existing
`I_sk`, `J_sjk`, component weights and parameter transformers provide all
inputs. No GP, full VBMC object, new VP statistic or save-format change is
required. The cross-run GP estimator investigated in the campaign is a
different estimator and is outside this integration.

No estimator-selection argument, new public function, additional shrinkage
variants, uncertainty formula for shrinkage, new VBMC runs, HPC work or
release campaign is part of this change.

Planning records belong on `dev-next`; package implementation uses a feature
branch from it (`dev-next-work`). The implementation and development-record
changes are prepared together in this checkout for integration. Reconcile
the records on `dev-next` without overwriting concurrent PyMC records; do
not modify the active `dev-pymc-adapter` worktree.

## Existing implementation and evidence

- `dev/scripts/svbmc_shrink_elbo.py`: `run_estimates`, `moments`,
  `shrink_full`, and the `two_level_full` branch in `score_cell` are the
  scientific reference. Preserve this historical runner and its results.
- `pyvbmc/svbmc/svbmc.py`: construction filters runs, computes deterministic
  Jacobian corrections and original-space component expectations;
  `optimize` reports several estimates after fresh entropy evaluation.
- `_expected_log_joint_variance` uses floors appropriate to the existing
  raw uncertainty calculation. Do not reuse that scalar calculation as a
  replacement for the reference's component covariance or change its
  arithmetic in this task.
- `pyvbmc/testing/svbmc/test_svbmc_references.py` checks 18 short seeded
  group/mode recipes and every existing report value. Its exact-key check
  needs to allow precisely the added fields; the fixture values stay fixed.
- `dev/scripts/svbmc_pool_stack.py` explicitly selects estimates to record.
  Adding a class field alone would omit it from future campaign outputs.

A read-only NumPy probe of all 39 shipped posterior snapshots found 21
total-covariance matrices with negative eigenvalues (minimum approximately
`-3.51e-10`); all original-weight run variances were positive (minimum
approximately `1.08e-9`). The prototype uses a symmetric linear solve, not
Cholesky. Requiring strictly positive-definite covariance would reject
existing inputs. This probe is diagnostic, not a parity test or evidence
that every possible input is numerically well-conditioned.

## Numerical and reporting contract

For retained run `m`, use its normalized original component weights `o_m`:

1. `I_m = mean_s(I_sk) - jacobian_m`, in original coordinates.
2. `C_m = mean_s(J_sjk) + cov_s(I_sk, ddof=1)`, symmetrized. The second
   term is zero for one hyperparameter sample. Preserve `(K_m, K_m)` shape
   for a one-component run.
3. Within a run, `mu_m = mean(I_m)`, `spread_m = var(I_m, ddof=1)` and
   `noise_m = (trace(C_m) - sum(C_m)/K_m)/(K_m-1)`.
   For `K_m=1`, spread and noise are zero.
   Set `tau2_m = max(spread_m - noise_m, 0)`.
4. With positive `tau2_m`, compute
   `within_m = mu_m + tau2_m * solve(tau2_m * eye(K_m) + C_m, I_m-mu_m)`.
   For zero `tau2_m`, use the constant `mu_m`. Solving for a vector avoids
   constructing an inverse or a dense matrix across runs.
5. Run levels are `E_m = o_m @ I_m`, with variances `V_m = o_m @ C_m @ o_m`.
   Apply the reference's moments rule across these levels with independent
   run errors (diagonal covariance), then its scalar shrinkage rule to get
   `E_shrunk_m`. Add `E_shrunk_m - E_m` to every `within_m` component.
6. Dot the concatenated corrected estimates with the selected global
   component weights and add `stacked.entropy`.

A single retained run has zero run-level shift. For a zero-variance level,
retain its observed value, including when excess population variance is
zero; avoid the prototype's `0/0` in that degenerate case. All-zero
estimation covariance must leave the original estimate unchanged. A run
with one component can still receive a between-run shift.

Use float64, no random draws and no mutation of posterior statistics.
Keep the reference covariance and symmetric solve, including its small
negative eigenvalues; do not introduce PSD projection, jitter, a pseudoinverse
or a new noise floor merely to obtain passing tests. An indefinite but
solvable covariance remains eligible when its run-level variance is
nonnegative and the calculated results are finite, matching the reference.

Existing validation permits finite covariance statistics that make the
shrinkage formula undefined. For a negative run-level variance, a singular
linear solve, or a nonfinite shrinkage result, return `None` for both new
fields and emit a `RuntimeWarning` identifying shrinkage as unavailable and
the numerical cause (and retained-run index when applicable). Preserve all
existing optimization/report results. Catch only these numerical conditions;
do not hide programming errors behind a broad exception handler. Do not
substitute the raw estimate or zero, tighten constructor validation, or
discard a retained run. This unavailable-estimate contract also covers
nonfinite intermediate calculations from finite but extreme inputs.

Add `elbo_details["shrinkage_noise_share"]`: the selected-run-mass-weighted
mean of `noise_m/spread_m` over runs with positive spread, renormalizing over
their mass, as in `score_cell`. Return `None` if no defined share has positive
mass; this alone does not invalidate the ELBO estimate. Do not clamp this
diagnostic to `[0, 1]`; it is an estimated noise to
spread ratio and can exceed one. It is available independently of the
`noisy` flag. No new uncertainty field accompanies `shrunk_two_level`.

## Execution phases

### 1. Package estimator and focused tests

**Executor:** Sol (`gpt-5.6-sol`, high), implementation; Astra orchestrator
owns scientific decisions and integration. One implementation agent owns the
package, tests and campaign adapter for this bounded change.

1. Add private NumPy helpers in `pyvbmc/svbmc/_elbo_shrinkage.py` for the
   two-level calculation and its noise-share diagnostic. Keep the interface
   private and limited to the retained statistics, Jacobian-corrected means,
   original run weights and selected component weights. Follow the existing
   S-VBMC license-header convention for code adapted from its sources.
2. Call the helper from `SVBMC.optimize` after weight selection. Reuse the
   final entropy and insert the two report fields. Do not modify
   `I_corrected`, `stacked_ELBO`, `maximize_ELBO`, `_expected_log_joint_variance`
   or input VPs. Repeated `optimize` calls must report the current weights.
3. Add `pyvbmc/testing/svbmc/test_elbo_shrinkage.py` for deterministic
   algebra and edge cases. Use independent closed-form expected answers:
   correlated versus diagonal component errors, between-hyperparameter
   covariance, unequal original run weights, one run, one component,
   `Ns=1`, zero covariance, zero excess variance, identical levels, zero
   selected run mass and defined/undefined noise shares. Include finite
   indefinite-but-solvable covariance, an exactly singular solve, negative
   run-level variance and overflow; check the unavailable-estimate warning
   and preservation of the existing report for the last three conditions.
4. Extend `test_elbo_reporting.py` to check both noise statuses, all three
   optimization modes, filtering to retained runs, bounded/warped Jacobian
   corrections, final-entropy reuse, repeat evaluation and unchanged RNG
   and input state. Use its fixed-evaluation helper where possible rather
   than adding full optimization runs. Include posteriors carrying only the
   currently required statistics, with no GP or VBMC references, to pin the
   unchanged input requirements.
5. Update `test_svbmc_references.py` to require the old key set plus exactly
   the two new fields, and continue comparing all old values to their
   stored references. Update `FIXTURES.md` to explain additive-field checks;
   do not regenerate `references.npz` or its sidecar.

**Acceptance:** numerical tests pass; the raw, capped and naive estimates,
headline, uncertainty, weights, entropy and RNG behavior retain their
existing checks; no core VBMC source or fixture baseline changes.

### 2. Campaign recording and reference parity

**Executor:** the same Sol implementation agent. Astra runs all numerical
commands sequentially, coordinated with the PyMC workstream; parallel
reviewers use static analysis only.

1. Extend the integrated arm of `dev/scripts/svbmc_pool_stack.py` to record
   available `shrunk_two_level` values in `elbos` and their automatically
   derived bias. Record `shrinkage_available` (a boolean) and the nullable
   `shrinkage_noise_share` outside the numeric-only `elbos` dictionary.
   When unavailable, omit the new numeric estimate from `elbos`; never pass
   `None` to its float conversion or to `sample_metrics`. The availability
   flag distinguishes a new failed estimate from an old record that never
   computed shrinkage. Preserve the original arm and headline criteria.
2. Ensure summary output exposes the new estimator's bias when present.
   Historical result files lacking it and mixed old/new summaries remain
   readable; any estimator summary must state its contributing cell count.
   The harness shares a bootstrap generator across summaries: compute the
   added estimator's intervals with a separate reproducible stream so
   inserting its bootstrap does not move existing intervals or paired
   checks. Test preservation of existing summaries on otherwise identical
   records with and without the added fields.
3. Add a bounded developer test for package/reference parity using existing
   fixture groups (unbounded, bounded, warped and noisy), fixed nonuniform
   selected weights and the historical script's actual functions. Keep the
   test under `dev/scripts/`, outside package discovery, so shipped tests
   have no dependency on developer scripts. Independently assemble the
   reference's `two_level_full` composition; include its Jacobian and
   noise-share rules. Expected absolute agreement: `1e-10` nats for ordinary
   fixture values, with relative tolerance `1e-12`. Investigate mismatches
   before considering a justified, documented tolerance adjustment.
4. Extend `dev/scripts/test_svbmc_pool_stack.py` for field recording,
   JSON-null diagnostics, unavailable estimates, old-result loading and
   summary behavior. Summary output must expose unavailable counts as well
   as contributing counts, including a group with no available estimates.
   Reuse its existing short fixture-mode comparison rather than launching
   a pool.

**Acceptance:** campaign output can score the package estimator directly;
historical results remain usable; no new pool, target evaluations or
baseline regeneration is required. Developer checks that need missing
machine-local artifacts must be reported as unavailable, never as passing.

### 3. Documentation, verification and independent review

**Executor:** Sol implements documentation; Astra orchestrator runs checks
and reconciles an independent Sol (`gpt-5.6-sol`, high) static review.

1. Update the `SVBMC` attribute docstrings and
   `docsrc/source/api/classes/svbmc.rst` with the new keys and a short usage
   example. Explain within-run shrinkage plus between-run level shifts,
   evaluation at the selected weights, and the same final entropy. Explain
   that shrinkage relies on GP uncertainty estimates and has no guaranteed
   bias removal or calibrated interval from `elbo_sd`. Document the nullable
   result and its warning for numerically undefined shrinkage. Existing
   headline documentation remains accurate. A new public-function RST page is not
   needed for private helpers.
2. On `dev-next`, update the existing headline decision note and TODO to
   separate availability as a diagnostic estimator from later selection
   as headline. Link this plan from `dev/README.md` and the relevant
   existing plan. Preserve the historical runner and scientific results.
   No separate completion report is needed.
3. Verify imports use this checkout. Read the local environment record
   for Torch; the base venv has no Torch and the recorded CPU overlay is
   the existing test environment. Keep BLAS threads at one and do not run
   heavy checks concurrently with the PyMC workstream.
4. Run `python -m pytest pyvbmc/testing/svbmc -q` with Torch present, then
   the new developer parity test and the affected pool-harness checks by
   explicit path. The shipped 18 reference cells must pass unchanged.
   Missing Torch skips are not acceptable evidence for S-VBMC verification.
5. Run repository formatting hooks on changed files. Build Sphinx when the
   machine is available; inspect the changed API page and distinguish
   pre-existing warnings from any introduced by this change.
6. Invoke `$doublecheck` on the implementation and this plan's acceptance
   criteria. Fix confirmed findings, rerun affected checks, and record
   outcomes here. Full core golden campaigns and a new whole-package suite
   are unnecessary for this isolated reporting addition; final integrated
   release validation remains separately scheduled.

## Approved decisions

- **One additive estimate:** use `shrunk_two_level` for the campaign's
  `two_level_full`, plus `shrinkage_noise_share`. The name describes the
  operation without asserting unbiasedness. Importing every research
  variant would expand the public contract beyond the selected candidate.
- **Report it for both noise statuses:** this permits campaign comparison
  on noiseless controls without conflating estimator availability with
  headline policy. No estimator-selection parameter is needed for access.
- **Preserve the historical implementation as the reference:** a private
  package helper supports normal distribution packaging; a developer test
  checks agreement without coupling package imports to `dev/scripts`.
- **Keep covariance arithmetic specific to this estimator:** refactoring
  the existing uncertainty calculation would risk changing `elbo_sd` and
  its numerical floors without helping the requested integration.
- **Record it in the campaign harness now:** otherwise a later headline
  decision could compare a research-only implementation while leaving the
  packaged result unverified. Existing headline gates keep their meaning.
- **Unavailable estimate on undefined numerics:** return `None` with an
  explanatory warning while preserving the existing report. New input
  restrictions would violate the unchanged-requirements contract; a silent
  raw fallback would mislabel the estimator, and covariance repairs would
  change the method relative to the campaign reference.

## Approval and review

The user requested planning before implementation, then approved the plan
and requested execution with `$task` on 2026-09-16. Planning changed no
package code.

Independent Sol plan review found one issue: failure behavior was unspecified
for covariance statistics accepted by the existing constructor. The plan
now defines unavailable results, warnings, unchanged legacy reporting and
campaign availability counts, with explicit synthetic tests. Independent
static follow-up confirmed the correction; no findings remain. The local
checks covered source contracts, the covariance probe and relative plan
links. No implementation tests or builds were run. Execute an approved plan with
`$task`, retaining the checklist and verification results here.

## Execution record

- Implementation runs on `dev-next-work`; the PyMC worktree is untouched.
  Development-record edits are prepared alongside the feature for separate
  integration into `dev-next`.
- Verification uses this checkout's Python 3.12 environment and the
  recorded CPU Torch 2.14.0 overlay, with BLAS threads set to one. The
  local venv also has Torch 2.7.0; the overlay requires sandbox escalation
  for read access. Imports were checked to resolve PyVBMC to this checkout.
- The first S-VBMC check passed 212 tests; three import tests had setup
  errors because the selected temporary directory's parent did not exist.
  Creating the workspace-local parent resolved those errors.
- The next S-VBMC plus historical-parity check passed 220 tests, including
  all 18 numerical reference cases and all four parity groups. One new
  warped-input test compared normalized final weights to the unnormalized
  test input with exact equality; it was corrected to compare against the
  final `stacked.w`.
- The final formatted S-VBMC suite plus historical parity passed all
  **222 tests in 35.93 seconds**. This includes all 18 unchanged reference
  recipes and all four historical-estimator parity groups. The new
  zero-covariance underflow test and corrected warped-input check passed.
- Independent static review requested explicit tests for warped inputs,
  filtered runs and minimal posterior statistics. These were added. The
  exact-zero-covariance case uses the component estimates directly and has
  an underflow regression; review confirmed this resolves the numerical
  boundary concern. The final independent static review covered package,
  campaign, tests and documentation and reported no remaining findings.
- The bounded campaign-harness suite passed all 22 tests in 90.15 seconds,
  including both original and integrated fixture arms, historical result
  compatibility, unavailable-estimate counts and preservation of legacy
  bootstrap values and generator state.
- All repository formatting hooks passed on the 15 changed files:
  whitespace, end-of-file, isort, black-jupyter and pycln. Formatting used
  one Black worker after a Windows multiprocess formatter stalled and its
  own process tree was stopped.
- Sphinx 9.1.0 built all 41 pages successfully. Its one warning is the
  unchanged Example 2 Plotly MIME output; the source and temporary copy
  were SHA-256-identical and that notebook has no Git diff. The updated
  S-VBMC page has both new keys, a valid two-column table, a resolved
  `optimize` cross-reference and no unresolved markup. Browser setup found
  no available connections, so inspection covered generated HTML rather
  than a visual browser rendering. The temporary examples copy was removed
  after verifying its resolved path stayed inside this workspace.

Final commands (with the recorded Torch overlay on `PYTHONPATH`, this
checkout first, and single-threaded BLAS):

```console
python -m pytest pyvbmc/testing/svbmc dev/scripts/test_svbmc_shrinkage_parity.py -q --basetemp=.cache/shrinkage-verified-pytest -o cache_dir=.cache/shrinkage-pytest-cache
python -m pytest dev/scripts/test_svbmc_pool_stack.py -q --basetemp=.cache/shrinkage-harness-pytest -o cache_dir=.cache/shrinkage-pytest-cache
python -m pre_commit run --files <the 15 changed files>
python -m sphinx -b html -j 1 --keep-going -w _build/shrinkage-warnings.log source _build/shrinkage-html
```

The Sphinx command ran from `docsrc/` with examples temporarily copied into
`source/_examples`, following the repository's build workflow. The final
result is an additive estimator with **244 passing tests**, unchanged
reference fixtures and historical research code, and no open implementation
review findings. The headline decision and release campaign remain outside
this task; no merge or publication was performed.

### Integration with the PyMC base (2026-09-16)

The feature was rebased onto `dev-next` at `95c5689`, which includes the
completed PyMC integration. The rebase had no conflicts; `git range-diff`
confirmed that the feature patch in `628064a` matches the original
`6630917`. Both development indexes retain the PyMC completion records and
the shrinkage entries. Core VBMC, posterior and PyMC code match the base.

The same focused commands above, with fresh temporary directories, passed
after rebase: **222 S-VBMC and historical-parity tests in 43.25 seconds**,
and **22 campaign-harness tests in 115.57 seconds**. No reference fixtures
were regenerated. The full PyMC suite and documentation build were not
repeated for this rebase; the feature patch and its API documentation were
unchanged.

An independent Sol review checked the rebased patch, shared interfaces,
posterior-statistics producer, lazy imports and combined development
indexes, with no must-fix or should-fix findings. This was a static
compatibility review; it did not repeat the scientific derivation or run
a fresh PyMC-to-S-VBMC workflow. The updated base remains an ancestor of
the feature branch; integration into `dev-next` is still pending.
