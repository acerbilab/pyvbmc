# Cross-run expected log joint of stacked posteriors: prototype on the pilot pool

The Phase 2 estimator of the
[S-VBMC ELBO optimism note](../2026-09-12-svbmc-elbo-optimism.md#phase-2-measuring-instead-of-capping-with-the-runs-surrogates)
is implemented as `dev/scripts/svbmc_honest_elbo.py` and scored on the
three-seed pilot pool of the
[run-pool campaign](../plans/svbmc-benchmark-campaign.md): for every
component of a stacked posterior, the GPs of the *other* runs that cover
it estimate its expected log joint, and the weighted sum replaces the
runs' own estimates, which the weight optimization selected on. The
prototype establishes the code path, the checks, the coverage rule and
the figures on real artifacts. Three runs per condition and one value of
`M` cannot settle the estimator's behaviour; what they do show, on the
five pilot conditions, is that the cross-run estimates are accurate where
the target is noiseless and systematically *low* on two of the four
noisy conditions, by about as much as the raw estimate is high, for a
reason the per-component diagnostics make visible. The full pools will
decide; this report records what the prototype measures and how.

Tracked outputs: [`experiments/svbmc_pool/pilot/phase2/`](../experiments/svbmc_pool/pilot/phase2/)
(`results.json`, `summary.md`, `summary.json`, `sources.json`,
`figures/`) and
[`pilot/phase2_newconds_selfcheck/`](../experiments/svbmc_pool/pilot/phase2_newconds_selfcheck/).
The per-cell component arrays stay with the raw pilot artifacts on the
machine that holds them (listed in its gitignored
`dev/scripts/runs/LOCAL.md`).

## The estimator, as implemented

The module docstring of `svbmc_honest_elbo.py` is the specification;
in short:

- **Cross-run estimates.** Every component contributes 100 draws from
  its Gaussian in its run's transformed space, mapped to the original
  space. Every run of the cell maps the draws into its own space and
  returns `predict mean − log-Jacobian`, averaged over the draws: that
  run's estimate of the component's expected log joint in the common
  space, with a Monte Carlo standard error, and the mean predictive
  variance as the run's self-reported uncertainty.
- **Coverage rule.** Run `r` covers component `k` when its mean
  predictive SD on the component is at most `ratio` times the own run's
  or below an absolute floor of 0.1 nats (on a noiseless target both SDs
  are thousandths of a nat and their ratio says nothing), and in either
  case at most `sqrt(5)` nats, the S-VBMC filter's scale. Ratios 1.5, 2,
  3 and 5, the cap alone (`cap_only`) and no rule at all (`none`) are
  evaluated on every cell; the headline is ratio 2.
- **Combination.** Median, precision weighting (by the mean predictive
  variances) or mean over the covering other runs; a component no other
  run covers keeps its own stored, Jacobian-corrected value. The honest
  expected log joint of the stack is the weighted sum over components.
  The `pooled` variant includes the own run and is not honest.
- **Scoring.** Every estimate of the expected log joint (`raw`, the two
  caps of the class, honest under every rule and method) is scored by its
  bias against the cell's `e_log_joint_mc`, so the entropy term drops
  out; adding `entropy_ref` gives the ELBOs with the same biases. The same
  draws give each component's true expected log joint, hence each run's
  own error, each covering run's error, and `z` statistics against the
  self-reported SDs (the run's `sqrt(J_kk)` for its own components, the
  GP's mean predictive SD for the cross-run estimates). Every pair of
  evaluating run and component also records `n_eff`, the run's training
  inputs weighted by the component's Gaussian kernel (a point at the mode
  counts one; not comparable across dimensions).
- **Checks.** Own-run Monte Carlo against the stored `I_corr` and Monte
  Carlo against deterministic Jacobian terms gate the mapping and the
  units; the recomputed raw expected log joint against the arm's
  `raw ELBO − entropy` gates the weights and the component order.

Inputs: the 15 pilot artifacts (gpyreg 1.2.0 from the worktree the pool's
manifest names, recreated at tag `v1.2.0`), the 25 cells of the pilot
comparison (`pilot_stack/results.json`: five cell seeds per condition at
`M = 3`, the integrated arm's weights), 100 draws per component, seed 0.
The whole scoring takes 39 s single-threaded. The two artifacts of the
later conditions (`student_D8_noise3_svbmc`, `multisensory_s1_D6_svbmc`,
gpyreg 1.2.1) went through `--self-check`. The test module
`test_svbmc_honest_elbo.py` (9 tests, 27 s, a two-run `normal_D2` pool
generated on the fly) passes.

## Checks

| Check | Result over the 15 runs and 25 cells |
|---|---|
| Own-run Monte Carlo against stored `I_corr` | weighted offset at most 2.9 SE; largest single-component `\|z\|` 5.8 over 3750 component-cells (100-draw t statistics on skewed draws) |
| Monte Carlo log-Jacobian against `expected_log_jacobian` | exact (below 1e-15) on the unbounded targets; offset at most 1.9 SE on multisensory, whose Jacobian varies over a component |
| Recomputed raw expected log joint against the arm's `raw − entropy` | at most 2.2e-16 |
| Stratified truth `sum w E_true` against `e_log_joint_mc` | median bias per condition between −0.02 and +0.01 nats, standard errors 0.01–0.02 |
| Self-check of the two later conditions under gpyreg 1.2.1 | pass (offsets within 1.4 SE) |

The two later conditions' own errors are +0.32 ± 0.06 (Student D8, noise
3) and −0.02 ± 0.03 (noiseless multisensory), in line with the pilot
conditions below.

## Results at `M = 3`

Median over the five cells of a condition, 95 % bootstrap interval in the
summary file; every value is a bias of the expected log joint against
`e_log_joint_mc`, in nats. `covered` is the stacked weight on components
at least one other run covers under the headline rule.

| Condition | raw | capped_I | capped_E | honest (ratio 2, median) | covered | honest GP sd (indep / corr) |
|---|---:|---:|---:|---:|---:|---:|
| GMM, noise 3 | +0.69 | −0.30 | 0.00 | −0.82 | 1.00 | 0.11 / 0.46 |
| GMM, noiseless | +0.03 | −0.02 | +0.03 | +0.03 | 0.30 | 0.01 / 0.02 |
| Multisensory, noise 3 | +0.56 | +0.27 | +0.46 | +0.10 | 0.78 | 0.12 / 0.50 |
| Ring, noise 3 | +0.11 | 0.00 | +0.11 | +0.05 | 0.30 | 0.07 / 0.30 |
| Rosenbrock, noise 3 | +0.12 | +0.03 | +0.04 | −0.23 | 1.00 | 0.04 / 0.22 |

Reading the table:

- **Noiseless GMM.** Where another run covers a component (30 % of the
  weight; the rest sits on clusters only one run found), its estimate is
  right to a few hundredths of a nat (`|z|` median 0.04 against the GP's
  own SD, which is far too conservative there). The honest value equals
  the raw one because the fallback dominates.
- **Multisensory.** The one covering pair (runs 1000 and 1002 cover
  each other; run 1001 sits elsewhere) estimates the other's components
  with an error of +0.10 nats, while the runs' own errors are +0.89,
  +0.33 and +0.37. The honest value removes most of the raw optimism and
  more than the cap does.
- **Ring.** Coverage is 30 % under ratio 2 (each run holds a different
  arc; run 1001 is covered by nobody), 51 % under ratio 5. The honest bias
  is small at every rule, +0.05 to −0.03.
- **Noisy GMM and Rosenbrock.** Every component is covered, and every
  covering run's estimate is *low*: on the GMM by 0.5–1.4 nats per pair
  and on Rosenbrock by 0.1–0.3, giving honest biases of −0.82 and −0.23
  against raw biases of +0.69 and +0.12. Median, precision weighting and
  the ratio (1.5 to 5, or the cap alone) change these by at most 0.02
  nats: with two other runs the combination barely matters, and coverage
  saturates. Without any rule (`none`) the mean function of a
  non-covering run takes over and the estimate is off by 8 (GMM), 19
  (multisensory) or 1000 nats (ring), so the rule is necessary; among
  covering runs it is not what sets the bias.

## Where the cross-run bias comes from

The per-component diagnostics on the heaviest component of GMM run 1001
(weight 0.12, true expected log joint −1.58) decompose the errors:

| Estimate of the component's expected log joint | Value | Error | Data behind it |
|---|---:|---:|---|
| Own run (1001), stored Bayesian quadrature | −0.32 | +1.26 | 36 training points within two SD of the component; their noisy values average +0.98 above the true log joint |
| Other run 1002 | −2.93 | −1.35 | 16 training points within two SD, averaging +0.02 above the truth; predictive SD 0.65; mean function −57 there |
| Other run 1000 | −28.3 | −26.7 | no training point within two SD; predictive SD 7.8 (excluded by every rule) |

Two effects, both systematic:

1. **The own run's optimism is partly in its data.** The component sits
   where the run's noisy evaluations came out high (the acquisition
   returns to regions the surrogate rates highly, and the variational
   optimization places components on the surrogate's maxima), so the
   training values around a heavy component average a noise realization
   of the same sign. This is the selection the honest estimator is meant
   to escape, and the other runs' data at the same place are unbiased.
2. **Another run's GP is tuned to its own region.** Run 1002's kernel
   length scales, signal variance and negative-quadratic mean were fitted
   to its own cluster; at a cluster 15 units away, with 16 noisy points,
   its posterior mean sits 1.35 nats below the truth although its
   predictive SD is 0.65 and of the same size as the own run's. The
   under-prediction does not disappear with more data: on the noisy GMM
   the covering runs have a weighted median of 8 effective training points
   on a component against the own run's 12, and the error stays negative
   across that range (`figures/sparsity.png`). On Rosenbrock, a unimodal
   target every run fitted over the same region, every one of the six
   cross-run pairs is 0.1–0.3 nats low while two of the three own errors
   are +0.17.

The calibration statistics say the same in aggregate. The runs' own
Bayesian-quadrature SD does not cover the run's own error where the
selection is strong (weighted median `z` +1.8 on the noisy GMM, +1.1 on
multisensory, 44 % and 18 % of the weight beyond `|z| = 2`). The GP's
predictive SD is about the size of the cross-run error but the error is
one-signed (`z` median −1.0 on the noisy GMM, −0.6 on Rosenbrock, +0.3 on
multisensory, 0.0 on the ring and the noiseless GMM), so precision
weighting cannot repair it and the honest value's GP-side SD (0.46 nats on
the GMM under full within-run correlation) understates its bias.

## What the full pools need to settle

The prototype scores any cell of the stacking comparison in seconds per
cell at `M = 3`; at `M = 32` a cell needs about 5 million GP predictions
(1600 components, 100 draws, 32 GPs), a few minutes, so the stage D grid
over eight conditions fits an evening at 100 draws and half of that at 50.
The questions the pilot leaves open, in the order the optimism note asks
them:

- **Growth with `M`.** The raw bias grows with the number of runs; the
  honest estimator was designed so that it cannot. Whether the honest
  bias is flat in `M` at whatever level it has, and whether that level is
  the pilot's −0.8 on the GMM once many runs cover every cluster, is the
  main measurement.
- **The level.** If the cross-run under-prediction is the rule on noisy
  targets, an honest estimate that is pessimistic by a stable amount is
  still a measured lower value and, with the raw one, a bracket; whether
  it can be the headline, or the cap stays, is the PI's decision on the
  full-pool numbers. Two variants the script does not implement would
  address the fit effect directly if it is confirmed: a leave-one-run-out
  GP refit on the pooled evaluations of the other runs, whose
  hyperparameters would be tuned over the whole posterior region, and a
  correction of each covering run's estimate by its measured offset on
  the components it shares with the evaluated run.
- **Coverage on the multimodal targets.** At `M = 3` no other run
  covers 70 % of the weight on the ring and the noiseless GMM, so the
  honest value there is mostly the raw one; with 40 to 100 runs every
  mode should be covered, and the ring's arcs will show whether the
  under-prediction depends on the local geometry.
- **The rule and the combination.** Ratio 1.5 to 5 with the cap gave
  the same answer on every pilot condition, and median, precision and
  mean were indistinguishable with two covering runs. With ten or more
  covering runs the median and the precision weighting will differ, and
  the precision weights can be checked against the measured cross-run
  errors cell by cell.

## Files

- `dev/scripts/svbmc_honest_elbo.py`: the estimator, its checks, the
  summaries and the figures (`--pool`, `--cells`, `--out`; `--self-check`
  without cells; `--summarize-only` to rebuild summaries and figures from
  a finished `results.json`).
- `dev/scripts/test_svbmc_honest_elbo.py`: the contracts, on a two-run
  Gaussian pool generated by the pool generator.
- `dev/experiments/svbmc_pool/pilot/phase2/summary.md`: every table of
  the run above, per condition and rule, with bootstrap intervals;
  `results.json` every cell row and every run's checks; `sources.json`
  commits, import paths, versions, the hashes of the script and of the
  cells file.
- `dev/experiments/svbmc_pool/pilot/phase2/figures/`:
  `bias_by_condition.png` (raw, capped and honest biases with bootstrap
  intervals), `component_errors.png` (own against honest error per
  component, marker area by stacked weight), `coverage.png` (covered
  weight and honest bias under every rule), `calibration.png` (`z` of
  the own and cross-run estimates against their self-reported SDs),
  `sparsity.png` (errors against effective training points).
