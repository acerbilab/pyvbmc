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
`M` cannot settle the estimator's behaviour. What they show, on the five
pilot conditions, is that the cross-run estimates are accurate where the
target is noiseless and systematically *low* on two of the four noisy
conditions, by more than the raw estimate is high, for a reason the
per-component records make visible; at `M = 3` the existing
component-median cap is closer to the reference than the honest estimate
on four of the five conditions. The full pools will decide; this report
records what the prototype measures and how.

Tracked outputs: [`experiments/svbmc_pool/pilot/phase2/`](../experiments/svbmc_pool/pilot/phase2/)
(`results.json`, `cells.jsonl`, `summary.md`, `summary.json`,
`sources.json`, `figures/`) and
[`pilot/phase2_newconds_selfcheck/`](../experiments/svbmc_pool/pilot/phase2_newconds_selfcheck/).
Every measurement below is in `summary.md` or `results.json`. The per-cell
component arrays (`cells/*.npz`) stay with the raw pilot artifacts on the
machine that holds them (listed in its gitignored
`dev/scripts/runs/LOCAL.md`); the three figures that read them are
regenerated only there.

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
  bias against the cell's `e_log_joint_mc`, the mean of the target's
  noiseless log density over the arm's own draws of the stacked
  posterior, so the entropy term drops out; adding `entropy_ref` gives
  the ELBOs with the same biases. The same draws give each component's
  true expected log joint, hence each run's own error, each covering
  run's error, and `z` statistics against the self-reported SDs (the
  run's `sqrt(J_kk)` for its own components, the GP's mean predictive SD
  for the cross-run estimates). Monte Carlo standard errors account for
  the covering runs and the truth averaging the same draws. Two GP-side
  standard deviations of the honest value are reported, `gp_sd`, under
  independence across components and under full correlation within each
  evaluating run.
- **Data behind each estimate.** Every pair of evaluating run and
  component records `n_eff` (the run's training inputs weighted by the
  component's Gaussian kernel; a point at the mode counts one; not
  comparable across dimensions), `n_within` (training inputs within two
  component SDs), `data_offset` (the mean over those inputs of the stored
  noisy value against the true log joint there) and `mean_fn` (the run's
  GP mean function alone on the component). For the heaviest component of
  every run of a cell these are written out with every run's estimate as
  a `decomposition` record.
- **Checks.** Own-run Monte Carlo against the stored `I_corr` and Monte
  Carlo against deterministic Jacobian terms gate the mapping and the
  units; the recomputed raw expected log joint against the arm's
  `raw ELBO − entropy` gates the weights and the component order.

Inputs: the 15 pilot artifacts (generated under gpyreg 1.2.0, scored here
under the campaign's pin 1.2.1 through `--gpyreg-source`; a scoring
under 1.2.0 gave every number identically, as the 1.2.1 release changed
nothing in GP prediction), the 25 cells of the pilot
comparison (`pilot_stack/results.json`: five cell seeds per condition at
`M = 3`, the integrated arm's weights; the tracked copy
`experiments/svbmc_pool/pilot/stack/results.json` is the same file), 100
draws per component, seed 0, BLAS single-threaded as `sources.json`
records. The whole scoring takes 41 s. The two artifacts of the later
conditions (`student_D8_noise3_svbmc`, `multisensory_s1_D6_svbmc`, gpyreg
1.2.1) went through `--self-check`. The test module
`test_svbmc_honest_elbo.py` (12 tests, 30 s, a two-run `normal_D2` pool
generated on the fly plus a synthetic two-transformer check of the
mapping) passes.

## Checks

| Check | Result over the 15 runs and 25 cells |
|---|---|
| Own-run Monte Carlo against stored `I_corr` | weighted offset at most 3 SE; largest single-component `abs z` 5.8 over 3750 component-cells (100-draw t statistics on skewed draws) |
| Monte Carlo log-Jacobian against `expected_log_jacobian` | exact (below 1e-15) on the unbounded targets; on multisensory, whose Jacobian varies over a component, weighted offset at most 1.5 SE and single components up to `abs z` 4.1 |
| Recomputed raw expected log joint against the arm's `raw − entropy` | at most 2.2e-16 |
| Stratified truth `sum w E_true` against `e_log_joint_mc` | median bias per condition between −0.03 and +0.02 nats, standard errors 0.01–0.02 |
| Self-check of the two later conditions under gpyreg 1.2.1 | pass (offsets within 1.5 SE) |

The two later conditions' own errors (the run's own weights) are
+0.32 ± 0.06 (Student D8, noise 3) and −0.02 ± 0.03 (noiseless
multisensory), in line with the pilot conditions below.

## Results at `M = 3`

Median over the five cells of a condition, 95 % bootstrap interval in the
summary file; every value is a bias of the expected log joint against
`e_log_joint_mc`, in nats. `covered` is the stacked weight on components
at least one other run covers under the headline rule; `within 2 SE` is
the fraction of cells whose honest bias is within twice the combined
Monte Carlo standard error of the estimate and of the reference.

| Condition | raw | capped_I | capped_E | honest (ratio 2, median) | covered | within 2 SE | median abs bias raw / capped_I / honest | honest `gp_sd` (indep / corr) |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| GMM, noise 3 | +0.69 | −0.30 | 0.00 | −0.82 | 1.00 | 0 % | 0.69 / 0.30 / 0.82 | 0.11 / 0.46 |
| GMM, noiseless | +0.03 | −0.02 | +0.03 | +0.03 | 0.30 | 40 % | 0.03 / 0.02 / 0.03 | 0.01 / 0.02 |
| Multisensory, noise 3 | +0.56 | +0.27 | +0.46 | +0.10 | 0.78 | 0 % | 0.56 / 0.27 / 0.10 | 0.12 / 0.50 |
| Ring, noise 3 | +0.11 | 0.00 | +0.11 | +0.05 | 0.30 | 0 % | 0.11 / 0.00 / 0.05 | 0.07 / 0.30 |
| Rosenbrock, noise 3 | +0.12 | +0.03 | +0.04 | −0.23 | 1.00 | 0 % | 0.12 / 0.03 / 0.23 | 0.04 / 0.22 |

Reading the table:

- **The honest estimate is not within its Monte Carlo error of the
  reference on any noisy condition** (0 % of cells; 40 % on the noiseless
  GMM), so whatever it measures is not sampling noise. Its median absolute
  bias is smaller than the raw estimate's on multisensory and the ring and
  larger on the noisy GMM and Rosenbrock; the component-median cap is
  closer than the honest estimate on every condition but multisensory.
- **Noiseless GMM.** Where another run covers a component (30 % of the
  weight; the rest sits on clusters only one run found), its estimate is
  right to a few hundredths of a nat (`abs z` median 0.18 against the
  GP's own SD). The honest value equals the raw one because the fallback
  dominates.
- **Multisensory.** One pair covers each other (runs 1000 and 1002),
  with cross-run errors of +0.10 and +0.18 nats in the two directions
  (medians over the five cells; +0.03 to +0.20 per cell), while the runs'
  own errors with the run's own weights are +0.89, +0.33 and +0.37. The
  honest value removes most of the raw optimism and more than the cap
  does. The rule is not watertight here: 0.5–5.4 % of run 1001's weight
  passes ratio 2 in every cell, and on that sliver the covering runs are
  off by −4.5 to −6.5 nats (`seen_by` in `results.json`), about −0.03 nats
  of the honest value.
- **Ring.** Coverage is 24 % under ratio 1.5, 30 % under ratio 2 and 51 %
  under ratio 5 (each run holds a different arc; run 1001 is covered by
  nobody), and the honest bias moves with it: +0.07, +0.05, −0.03.
- **Noisy GMM and Rosenbrock.** Every component is covered, and every
  covering run's estimate is *low*: all 30 directed run-pair errors on the
  GMM (−0.09 to −1.45 nats, weighted by the evaluated run's components)
  and all 30 on Rosenbrock (−0.004 to −0.30) are negative, giving honest
  biases of −0.82 and −0.23 against raw biases of +0.69 and +0.12.
- **Rule and combination.** Where coverage saturates (noisy GMM,
  Rosenbrock) the rule is immaterial from ratio 2 upward and ratio 1.5
  differs by at most 0.05 (GMM: −0.78 against −0.82), and the median,
  precision and mean combinations differ by at most 0.03 (two covering
  runs). Where coverage does not saturate (ring, noiseless GMM) the rule
  is what moves the honest value (noiseless GMM: +0.03 at ratio 2, −0.05
  under the cap alone with 65 % covered). Without any rule (`none`) the
  mean function of a non-covering run takes over and the estimate is off
  by 8 (GMM), 19 (multisensory) or 1100 nats (ring), so the rule is
  necessary; among covering runs it is not what sets the bias.

## Where the cross-run bias comes from

The `decomposition` record of the first cell of every condition (in
`results.json` and `summary.md`) lists, for the heaviest component of
each run, the truth, the run's own estimate, and every other run's
estimate next to what that run's data said at the component. On the
noisy GMM (cell repetition 0, run 1001's component 18, weight 0.077,
truth −0.93):

| Estimate of the component's expected log joint | Value | Error | Training points within 2 SD, their mean offset from the truth | Mean function there |
|---|---:|---:|---|---:|
| Own run 1001, stored Bayesian quadrature (BQ sd 0.43) | −0.03 | +0.91 | 23 points, +0.86 | −26 |
| Other run 1002 (predictive sd 0.60, covered) | −1.94 | −1.00 | 11 points, +0.63 | −59 |
| Other run 1000 (predictive sd 8.4, not covered) | −30.7 | −29.8 | no points | −31 |

The cross-run half of this pattern holds for the heaviest components of
the other two GMM runs (covering other runs −1.25, −1.16, −0.16 and
−0.37) and of all three Rosenbrock runs (−0.47, −0.38, −0.20, −0.22,
−0.06, −0.07): every one of the ten covering estimates is low. The own
half varies: the own errors of those components are −0.07 and +0.60 (GMM)
and +0.18, +0.30 and −0.53 (Rosenbrock), and each follows the sign of the
run's own data offset at the component (−0.01, +0.78; −0.43, +0.83,
−0.66). Two effects, the second systematic:

1. **The own run's optimism, where it occurs, is in its data.** A run's
   own error on a heavy component moves with the noise realization of its
   evaluations there: +0.91 with an offset of +0.86 over 23 points at the
   component above, and negative where the offset is negative. The
   positive cases carry the optimism (the acquisition returns to regions
   the surrogate rates highly, and the variational optimization places
   components on the surrogate's maxima, which favours regions where the
   noise came out high), and they are the selection the honest estimator
   is meant to escape.
2. **Another run's GP is tuned to its own region.** Run 1002 had 11
   points within two SDs of the same component whose values average +0.63
   *above* the truth, and its GP still predicts 1.0 nats *below* it with a
   predictive SD of 0.60: its kernel length scales, signal variance and
   negative-quadratic mean function (−59 at that component, against −26
   for the own run) were fitted to its own cluster, and away from it the
   fit falls toward the mean function. The under-prediction is not
   sparsity alone: on the noisy GMM the covering runs have a weighted
   median of 8 effective training points on a component against the own
   run's 12, and the weighted median cross-run `z` is −0.9 with 16 % of
   the weight beyond `abs z` 2 (`figures/sparsity.png` shows errors of
   both signs at every `n_eff`, with the negative side heavier).

The calibration statistics say the same in aggregate. The runs' own
Bayesian-quadrature SD does not cover the run's own error where the
selection is strong (weighted median `z` +1.8 on the noisy GMM, +1.1 on
multisensory, 44 % and 18 % of the weight beyond `abs z` 2). The GP's
predictive SD is about the size of the cross-run error but the error is
one-signed (`z` median −0.9 on the noisy GMM, −0.6 on Rosenbrock, +0.3 on
multisensory, 0.0 on the ring and the noiseless GMM), so precision
weighting cannot repair it and the honest value's GP-side SD (0.46 nats on
the GMM under full within-run correlation) understates its bias.

## What the full pools need to settle

The prototype scores a cell in about a second at `M = 3`; at `M = 32` a
cell needs about 5 million GP predictions (1600 components, 100 draws,
32 GPs), a few minutes, so the stage D grid over eight conditions fits an
evening at 100 draws and half of that at 50. The questions the pilot
leaves open, in the order the optimism note asks them:

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
- **The rule.** Ratio 2 and above with the cap agree where coverage
  saturates and differ where it does not, and on multisensory a few
  percent of one run's weight passes the rule with errors of several
  nats; with 40 to 100 runs every mode should be covered and the rule's
  false admissions can be counted against the `n_within` and
  `data_offset` records.
- **The combination.** Median, precision and mean were indistinguishable
  with two covering runs. With ten or more covering runs the median and
  the precision weighting will differ, and the precision weights can be
  checked against the measured cross-run errors cell by cell.

## Files

- `dev/scripts/svbmc_honest_elbo.py`: the estimator, its checks, the
  summaries and the figures (`--pool`, `--cells`, `--out`; `--self-check`
  without cells; `--summarize-only` to rebuild summaries and figures from
  a finished `results.json`; `--headline-ratio` to move the headline
  within `--ratios`).
- `dev/scripts/test_svbmc_honest_elbo.py`: the contracts, on a two-run
  Gaussian pool generated by the pool generator, plus synthetic checks of
  the combination rules, the checks and the mapping between two different
  transformers.
- `dev/experiments/svbmc_pool/pilot/phase2/summary.md`: every table of
  the run above, per condition and rule, with bootstrap intervals and the
  decomposition of the heaviest components; `results.json` every cell row
  (including `seen_by`, `calibration`, `sparsity`, `decomposition` and
  the checks) and every run's self-check; `cells.jsonl` the same cell rows
  as written during the sweep; `sources.json` commits, import paths,
  versions, thread settings, the hashes of the script and of the cells
  file.
- `dev/experiments/svbmc_pool/pilot/phase2/figures/`:
  `bias_by_condition.png` (raw, capped and honest biases with bootstrap
  intervals), `component_errors.png` (own against honest error per
  component, marker area by stacked weight), `coverage.png` (covered
  weight and honest bias under every rule), `calibration.png` (`z` of
  the own and cross-run estimates against their self-reported SDs),
  `sparsity.png` (errors against effective training points).
