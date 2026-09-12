# S-VBMC's ELBO optimism: what ships, where it comes from, and a two-phase plan

*Written 12 September 2026 from a discussion with the PI; the decisions are
recorded at the end. Neither phase has started.*

Stacking (Silvestrin, Li and Acerbi, 2025;
[`papers/silvestrin2025stacking_main.md`](../papers/silvestrin2025stacking_main.md)
and [its appendix](../papers/silvestrin2025stacking_appendix.md)) reweights
the components of several finished VBMC posteriors by maximizing the stacked
ELBO: the weighted sum of the per-component expected log-joints that every
run stored, plus the entropy of the stacked mixture. Section 5 of the paper
reports that on noisy targets the resulting ELBO estimate is optimistic, that
the optimism grows with the number of stacked runs until the estimate sits
above the true log marginal likelihood, and proposes a post-hoc cap. This
note records what `pyvbmc/svbmc/` does about it today, why the optimism is a
cross-run selection effect that a cap can bound but not remove, and the two
phases agreed with the PI: a first phase that works from the stored
posteriors alone, fixes the corrections and tells users what they are
looking at, and an exploratory phase that measures the expected log-joint
with the other runs' surrogates instead of capping it. The settled S-VBMC
decisions this builds on are in
[plans/svbmc-integration.md](plans/svbmc-integration.md).

## The effect, in the paper's numbers

The paper's Figure 6 compares the expected log-joint of the stacked mixture
as S-VBMC estimates it with a Monte Carlo estimate from many extra
evaluations of the true log-joint, on the three noisy benchmarks (Gaussian
noise of SD 3 on every log-likelihood). The values below are read off the
transcribed figure descriptions to one significant figure; the descriptions
of Figures 6 and 7 disagree on the multisensory value by about 0.4, hence
the range. The last column is the residual the paper reports for the
component-median cap of its Section 5.2.

| Noisy target | Optimism at 2 runs | Optimism at 40 runs | Residual after the cap |
| --- | ---: | ---: | ---: |
| GMM, D = 2 | about 0.1 | about 0.7 | none reported |
| Ring, D = 2 | about 0.3 | about 0.7 | about 0.15 |
| Multisensory, D = 6 | about 0.5 | 1 to 1.5 | about 0.3 |

The optimism at two runs is mostly VBMC's own: each run's variational
optimization already maximizes over a noisy surrogate. The growth with the
number of runs is what stacking adds. On the noiseless benchmarks (GMM,
ring, neuronal model) the paper reports no overshoot, and the effect is not
expected there: with exact log-likelihoods the surrogates' errors are small
and the Bayesian-quadrature estimates of the expected log-joints have little
noise to select on.

Tables A.3 and A.4 of the paper separate the contributions by how finely the
weights are selected. At 20 runs on the noisy targets, the distance of the
stacked ELBO from the true log marginal likelihood (Δ LML) and the
Gaussianized symmetrized KL divergence of the posterior (GsKL):

| Weights | GMM Δ LML | GMM GsKL | Ring Δ LML | Ring GsKL | Multisensory Δ LML | Multisensory GsKL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Naive stacking, equal weight per run, no optimization | 0.056 | 0.017 | 0.19 | 0.023 | 0.55 | 0.06 |
| Posterior-only, one optimized weight per run | 0.32 | 0.0082 | 0.37 | 0.0054 | 0.88 | 0.047 |
| All-weights, every component's weight optimized (the default) | 0.53 | 0.0049 | 0.68 | 0.0045 | 1.2 | 0.039 |

The marginal metric (MMTV) is similar across the three rows; GsKL improves
with the finer selection, by a factor of three to five on the synthetic
targets and by a third on the multisensory model. The optimism is part of
the price of that selection, not the whole of it. The paper's Δ LML values
come from the raw ELBO; the cap was applied only in Section 5.

## What the implementation ships

`SVBMC.optimize()` in `pyvbmc/svbmc/svbmc.py` stores `elbo` as a dictionary
with three entries: `"estimated"` (the optimized stacked ELBO), and the two
capped values of the paper's Section 5.2, `"debiased_I_median"` (the expected
log-joint at the optimized weights capped at the median of the corrected
per-component estimates, the paper's recommended variant) and
`"debiased_E_median"` (capped at the median of the runs' expected log-joints).
The cap is `min(E_stacked, E_cap)` with the entropy added back, on the
Jacobian-corrected estimates of Appendix A.2, so the medians are taken in the
common original space. `pyvbmc/testing/svbmc/test_svbmc.py` asserts the keys
and that each capped value is at most the raw one;
`fixtures/references.json` pins all three for the seeded short optimization
of every fixture group and mode.

Three terms are used throughout this note. The *raw* value is the
`"estimated"` entry. The *capped* value is `"debiased_I_median"` unless the
run-median variant is named. The *headline* is the value a user reads as
the stacked ELBO, whatever container it ends up in.

Four things around the cap are worth knowing.

- **The Jacobian corrections are Monte Carlo, and the caps rest on one
  draw.** The Jacobian term of every component, the expectation of the
  run's log-Jacobian under the component, is estimated from the same draws
  as the entropy, with the optimization-time sample count (20 per component
  by default). The expected log-joint term of the objective therefore
  carries fresh Monte Carlo noise at every Adam step, alongside the
  entropy's. The copies that feed the two medians are frozen at the first
  `stacked_ELBO` call, so both caps rest on one 20-draw estimate, and
  `stacked_ELBO` is stateful because of it.
- **The reported ELBO is the best noisy iterate.** `maximize_ELBO` returns
  the ELBO of the best step seen by Adam, with fresh draws every step and
  the optimization-time sample count. There is no final re-estimation at
  the returned weights; the paper's tables used 100 draws per component
  after convergence. Choosing the best of noisy iterates is a small
  selection effect of its own. In the executed Example 7 as it stands (four
  runs on a noiseless bimodal target with true log evidence 0), the raw
  stacked ELBO is 0.046 and the component-median capped one 0.019; a value
  above the truth on a noiseless target could come from this Monte Carlo
  selection.
- **No uncertainty is reported.** VBMC users know `elbo` together with
  `elbo_sd`; the stacked object reports neither the entropy's Monte Carlo
  standard deviation nor anything else.
- **Little tells the user which value to use or when it matters.**
  Example 7 says to prefer the debiased values. Beyond that, the object
  neither says whether the cap binds nor by how much, and nothing says the
  raw value is optimistic on noisy targets specifically. The `"ns"` mode of
  `maximize_ELBO` computes the naive-stacking ELBO, which involves no
  selection, but it is not reported alongside the optimized one.

## Where the optimism comes from, and why the objective stays as it is

Each run's surrogate carries its own error, independent of the other runs'
errors because every run trains its own GP on its own evaluations. The
optimizer, region by region, gives weight to whichever run's estimate of the
expected log-joint came out highest, so the reported value collects the
highest errors; the more runs cover a region, the higher the highest error.
The all-weights mode does this per component, which is why it collects more
optimism than one weight per run (second table above). Within one run the
components share the GP and so share most of their error; that is why the
number of runs, not the number of components, drives the growth. The paper's
Section 5.1 gives the same account with identical single-component
posteriors, where the optimization reduces to picking the largest of M noisy
estimates.

Two fixes inside the objective are available and not taken.

- **Penalizing the variance of the expected log-joint term** (a lower
  confidence bound, as VBMC uses to select its best iteration) would push
  the weights toward spreading over interchangeable runs, whose average is
  free of selection. It needs the per-run error scale, and the only one at
  hand is the runs' own Bayesian-quadrature covariance (`stats["J_sjk"]`),
  which is the GP's self-reported uncertainty: nobody has checked that it
  tracks the actual cross-run disagreement, and the change alters the
  weights and therefore the posterior.
- **A Stein-type estimate of the selection bias** (the covariance between
  each estimate and its weight, computed by implicit differentiation of the
  optimum) would leave the weights alone but rests on the same unverified
  error scale.

Decision: the optimizer and its objective do not change. The reported values
and the corrections around them can, and the error-scale question is
answered empirically in Phase 2 rather than assumed.

## Phase 1: from the stored posteriors alone, no change to the optimization

Everything in this phase works from the `VariationalPosterior` objects the
constructor already takes. The optimization is unchanged in kind. The
weights do move, because the expected log-joint term of the objective loses
its per-step Monte Carlo noise (item 1); how far is not known in advance,
so the regenerated references are reviewed for plausibility, not for
closeness (item 6). When the phase starts, its items, decisions and gates
move into a slug-named plan under `dev/plans/`, as
[plans/svbmc-integration.md](plans/svbmc-integration.md) did for the
integration; this note stays the narrative.

1. **Jacobian corrections computed once, exactly or nearly so.** A run's
   log-Jacobian at a transformed point is a sum over dimensions of
   constants (`log delta`, `log scale`, and the bound width for bounded
   dimensions) and, for bounded dimensions, a one-dimensional function of
   one linear projection of the point (`parameter_transformer.py`:
   `log_abs_det_jacobian` un-scales, un-rotates by `R_mat`, then applies
   the bounded transform's log-derivative coordinate by coordinate; no
   term couples coordinates nonlinearly). Under an axis-aligned Gaussian
   component every projection is a one-dimensional Gaussian, so each term
   is a one-dimensional Gauss-Hermite integral, exact to within rounding
   with a few dozen nodes. The alternative is many draws per component (a
   thousand, say), simpler to write, with a little noise left in. Either
   way the term is computed once at construction, which removes the
   per-step noise from the objective, removes the noise from the corrected
   expected log-joints, the run values and both medians, and removes the
   stateful first-call behaviour of `stacked_ELBO`; the quadrature version
   also consumes no random draws. Which of the two is open decision 2.
2. **Final evaluation at the returned weights.** After Adam, one fresh
   entropy estimate at the returned weights with a larger sample replaces
   the best noisy iterate; the count is a parameter, proposed as
   `n_samples_final` with default 100 per component, the paper's choice.
   Its sample variance gives the entropy's Monte Carlo variance; what the
   stacked `elbo_sd` should contain is open decision 5. One more evaluation
   at the naive-stacking weights gives a reference value that involves no
   selection: those weights depend on no estimate, so the value is a valid,
   if pessimistic, estimate of a lower bound on the evidence. Since the
   optimized weights should raise the true ELBO above the naive one, the
   naive value and the raw value roughly bracket the true ELBO of the
   optimized mixture; roughly, because the naive value still carries each
   run's own optimism and its own Monte Carlo noise.
3. **Reporting.** The capped value becomes the headline, the paper's
   recommendation, for every stacking or only for noisy ones (open
   decision 4); the raw value stays available under its own name; the
   standard deviation and the naive-stacking value are stored alongside.
   When the cap binds, the log states the amount (for example: expected
   log-joint capped by 0.7 nats). The clause that the raw value is
   optimistic follows the same noise condition as the tip of item 4, since
   the cap also binds by small amounts on noiseless targets (Example 7).
4. **Tips, in the style of VBMC's runtime tips.** VBMC shows one short tip
   occasionally at the start of a run (`pyvbmc/vbmc/_runtime_tips.py`,
   data-only catalog in `_tip_catalog.py`, printed through
   `pyvbmc/_user_hints.py`; policy in
   [plans/runtime-tips.md](plans/runtime-tips.md)). S-VBMC gets the same
   shape: a small data-only catalog, the shared emitter, and VBMC's policy
   (the first eligible `optimize()` call of a session, then every third;
   one shuffle per session; each tip at most once; whether the scheduler
   state is shared with VBMC's or duplicated is an implementation choice).
   S-VBMC has no `display` option and logs instead, so what "off" means
   is open decision 3. The PI edits the wording as for the VBMC catalog.
   One entry is conditional and takes precedence over the catalog order
   the first time it is eligible: it is shown only when at least one
   retained run was noisy, never for noiseless targets, where the effect
   has not been observed, and it says that the raw stacked ELBO is
   optimistic and grows with the number of stacked runs, that the capped
   `elbo` is the value to use for model comparison and is to be reported
   with `elbo_sd`, and that the stacked posterior itself is unaffected.
   Candidates for the rest, unconditional: stack about ten runs, where
   most of the gain in posterior quality is (the paper's recommendation),
   and start the runs from different points.

   The condition needs a fact the posterior does not record today. Its
   `stats` hold `I_sk`, `J_sjk`, `e_log_joint`, `e_log_joint_sd`, `elbo`,
   `elbo_sd`, `entropy`, `entropy_sd` and `stable`; the noise treatment
   of a run lives in `optim_state["uncertainty_handling_level"]` on the
   VBMC object (0 noiseless, 1 noisy without a supplied noise estimate,
   2 with one). `optimize_vp` in `variational_optimization.py` receives
   the state and is where the stats are set, so recording the level there
   is one line, and every posterior VBMC returns from then on carries it;
   adding a stats key breaks no saved posterior or static fixture. The
   stacked object reads it from each retained run. When the entry is
   missing (posteriors saved by earlier versions, the fixture corpus), the
   run's noise treatment is guessed from a proxy: the ELBO standard
   deviation the run stored (`stats["elbo_sd"]`). VBMC's stopping rule
   averages that SD over the tolerance `tol_sd` (default 0.1) with two
   other ratios and stops below 1, so a noiseless run ends near or below
   the tolerance; on a noisy target the rule inflates the tolerance to
   `min(max(tol_sd, sqrt(sn tol_sd)), 10 tol_sd)` with `sn` the noise SD
   in the high-posterior region (about 0.32 at noise SD 1 and 0.55 at SD
   3; `_compute_reliability_index` in `vbmc.py`), so a noisy run ends
   well above the default. Two corpora say how far apart the two end up:
   the golden reference population `reference_870_20260907` (sidecars in
   `dev/golden/baseline/`; 870 runs of 19 configurations including 160
   noisy runs; final values from the `final.elbo_sd`, `noise_sd` and
   `final.success_flag` fields, converged runs only, aggregated on
   2026-09-12 with a few lines of Python, no script kept; the counts
   describe that population and change when a new reference is
   promoted), and the paper's own runs in the S-VBMC fixture corpus.

   | Corpus | Runs | `elbo_sd` | Runs above 0.1 | Runs above 0.05 |
   | --- | ---: | --- | ---: | ---: |
   | Golden, noiseless, the 14 configurations with converged runs, D = 2 to 10 | 700 | per-configuration medians 0.0005 to 0.05; max 0.12 apart from one anomaly at 4.6 | 4 | 50 |
   | Golden, noise SD 1 (`rosenbrock_D2_noise1`) | 50 | 0.097 to 0.149, median 0.11 | 49 | 50 |
   | Golden, noise SD 3 (three configurations, D = 2 to 8) | 103 | 0.25 to 1.8 | 103 | 103 |
   | Paper corpus, `upstream_GMM`, noiseless | 10 | 0.0016 to 0.020 | 0 | 0 |
   | Paper corpus, `upstream_Ring`, noise status not recorded in `FIXTURES.md` | 10 | 0.0003 to 0.019 | 0 | 0 |
   | Paper corpus, `upstream_GMM_noisy`, noise SD 3 | 10 | 0.29 to 0.35 | 10 | 10 |

   The ring group's values match the noiseless GMM group, which is the
   only ground for reading it as noiseless. The largest per-component
   standard deviation from `stats["J_sjk"]` does not separate the paper's
   groups (noiseless GMM 0.017 to 1.0, ring 0.002 to 0.77, noisy GMM 0.71
   to 1.9), so it is not the proxy. At noise SD 3 the ELBO standard
   deviation sits a factor of two or more above 0.1 in every corpus. At
   noise SD 1 in D = 2 the runs end right at 0.1, and the hardest
   noiseless configurations (`lumpy_D10`, `student_D8`) reach 0.12 in a
   few runs, so the band from 0.05 to 0.15 is ambiguous: a threshold of
   0.1 misclassifies 4 of 700 noiseless runs and 1 of 50 low-noise runs,
   while 0.05 catches every noisy run but flags 50 noiseless ones, most of
   them in those two configurations. The guess is `elbo_sd > 0.1`, the
   constant (the stacked object holds posteriors, not the runs' options),
   because the tip must not appear on noiseless targets and because the
   runs the threshold misses are the low-noise ones, whose surrogate
   errors, and with them the optimism, are small in the same proportion.
   It is a guess: only the recorded level settles it. A run is treated as
   noisy when its recorded level says so, or, absent a record, when the
   proxy says so. The anomalous noiseless run in the table is a benchmark
   matter, recorded in `dev/TODO.md`.
5. **Documentation.** The class docstring, `docsrc/source/api/classes/svbmc.rst`
   and Example 7 (`examples/pyvbmc_example_7_stacking.ipynb`, regenerated
   script) state the same guidance and name the values. The existing VBMC
   tip `svbmc` links to the standalone package's GitHub README; once the
   docs with Example 7 are published it should link there.
6. **Tests, fixtures and gates.** `references.json` and `references.npz`
   under `pyvbmc/testing/svbmc/fixtures/` move by design, since the
   objective loses its per-step noise and the reported values change; they
   are regenerated with `dev/scripts/make_svbmc_fixtures.py references`
   under the Torch overlay, on purpose, with the reason and the new recipe
   arguments (the final sample count) recorded in `FIXTURES.md` and passed
   by `test_svbmc_references.py`. The review of the regenerated values is
   for plausibility: the optimized weights and ELBO of every group and
   mode within the old Monte Carlo spread, no mode ranking changed, and
   the same fixture groups passing the same tests. The existing assertion
   that each capped value is at most the raw one stays valid under either
   headline. The one-line change in `optimize_vp` must leave every number
   and the random stream untouched, gated as any change under
   `pyvbmc/vbmc/`: `python dev/scripts/make_oracle_fixtures.py --check
   --exact` and `python dev/scripts/golden_replay.py` reporting identical.
   New checks: the quadrature Jacobian term against a high-sample Monte
   Carlo estimate on the bounded and warped fixture groups; the final
   evaluation using the requested sample count; the cap message and its
   noise condition; the tips' cadence, quiet behaviour and noise
   condition, including the proxy on posteriors without the recorded
   level (the noisy fixture group is classed noisy, the GMM noiseless
   group is not). Rollback is the revert of the feature branch, which
   brings the old references back with it.

Open decisions for the PI before this phase starts:

1. `elbo` stays a dictionary with added keys, or becomes a float with
   separate attributes for the raw value, the standard deviation and the
   naive-stacking value. Existing code reads the dictionary; the package is
   new enough that either is defensible. With it: whether the run-median
   cap `"debiased_E_median"` survives, and whether the naive-stacking
   reference is computed by default (one more entropy evaluation at the
   final sample count).
2. Gauss-Hermite quadrature or many draws for the Jacobian term.
3. How S-VBMC's tips are silenced: a `show_tips` constructor argument, the
   logger level, or both.
4. Whether the headline is the capped value for every stacking or only
   when the runs were noisy, with the raw value as the headline otherwise.
   The paper applied the cap to its noisy experiments only, and on a
   noiseless target the cap can only lower a value that does not
   overshoot; the recorded noise level makes the distinction possible.
5. What the stacked `elbo_sd` contains: the entropy's Monte Carlo variance
   alone, or that plus the runs' own quadrature uncertainty of the
   expected log-joint at the stacked weights (the run blocks of
   `stats["J_sjk"]`, independent across runs). A single run's `elbo_sd` is
   mostly the latter, so only the second choice makes the two comparable.

What this phase does not do: it does not reduce the cross-run optimism. It
removes Monte Carlo noise from the objective and from the inputs of the
cap, removes the small best-iterate selection term, and tells the user what
happened. The ten noisy GMM posteriors of the fixture corpus
(`upstream_GMM_noisy_00` to `_09`, from the paper's own runs) allow a sanity
check of the raw, capped and naive-stacking values against the target's
true log evidence for up to ten runs without any new VBMC run, once that
evidence is computed from the target definition in the pinned upstream
checkout, which is gitignored and present on one machine
(`dev/scripts/runs/svbmc_compat_20260908/source/`, per the integration
plan).

## Phase 2: measuring instead of capping, with the runs' surrogates

The cap bounds the growth at a level chosen empirically, not derived. The
alternative is to measure the expected log-joint of the final mixture with
noise that is independent of the noise the weights were selected on, the
standard cure for a winner's curse; such an estimate is called *honest*
below. Component by component, the evaluating surrogates are the GPs of the
*other* runs that cover the component; the component's own run is left out.
The selection then has nothing to feed on, so the growth with the number of
runs disappears by construction, and so does the part VBMC's own
optimization introduced, since the evaluating GPs are independent of that
too. What remains is the systematic GP error shared by all runs (a smoothed
fit of noisy data), the components no other run covers, which keep their own
estimate and face no competition anyway, and Monte Carlo noise that shrinks
with the number of covering runs. If it works, the honest value is the
candidate headline, the difference from the raw value becomes a measured
quantity the user sees, and the cap stays as the fallback for posteriors
without their GP; the headline and the interface are decided on the results.

**Mechanics, without new closed forms.** The entropy code already maps every
component's draws into every run's transformed space and evaluates the
run's log-Jacobian there (Appendix A.2 of the paper). The same mapped draws
go to each run's `gp.predict`, mean and variance. The GP models the
log-joint in its run's transformed space with the log-Jacobian folded in
(`function_logger.py` adds it to every stored value), so the per-component
mean prediction minus the Jacobian term is that run's estimate of the
component's expected log-joint in the common space, and the predictive
variance says whether the run covers the component at all. The covering
runs, own run excluded, are combined by median or precision weighting. A
closed-form Bayesian quadrature would need the components rotated into
another run's whitened space, which are no longer axis-aligned and which
`_gp_log_joint` does not handle, so Monte Carlo on the surrogate is the
practical route; it needs no likelihood evaluations.

**Costs.**

- Keeping: the final GP of each run, meaning inputs, targets, noise
  estimates and hyperparameter samples, tens of kilobytes; the posterior
  factors are rebuilt with one Cholesky per sample, which `VBMC.get_gp`
  already does. The VBMC object holds it as `vbmc.gp`; a
  `VariationalPosterior` has no GP (the `__str__` of `VBMC` looks for a
  `vp.gp` attribute defensively, but nothing sets one). Saved VBMC objects
  keep their GP, so existing runs can be reused.
- Computing: GP predictions at every component's draws, once per run,
  after the optimization. With ten runs, 500 components, 100 draws and a
  few hundred training points this is seconds, well below the entropy
  Monte Carlo, which evaluates every component against every draw at every
  Adam step.

**Risks.** A badly fitted GP in another run predicts confidently and
wrongly, so the combination has to be robust and the coverage rule has to
compare a run's predictive variance with the component's own run. The
posterior is untouched, so nothing can get worse there.

**What the exploration has to settle.** The coverage rule; median versus
precision-weighted combination; the fraction of stacked weight that is
cross-covered on each target; whether the honest estimate sits on the
Monte Carlo ground truth within its own standard deviation at every number
of runs; and the multimodal case, where cross-coverage exists per mode
only. It also answers the error-scale question of the previous section for
free: with the GPs at hand, each run's actual offset against the other
runs' evaluations is measurable, so whether the runs' self-reported
uncertainty (`stats["J_sjk"]`, `stats["e_log_joint_sd"]`) is calibrated
becomes a plot rather than an assumption.

**Where and on what.** A script under `dev/scripts/` and a report under
`dev/results/`, with the package untouched until the result is in. Pools
of VBMC runs on noisy targets, each run's final GP kept alongside its
posterior; bootstrap over the number of stacked runs as the paper did;
at every number of runs the raw, capped, honest and Monte Carlo
ground-truth values. Targets: the paper's noisy ring and GMM (definitions
in the pinned upstream checkout named above, present on one machine; the
fixture corpus has posteriors but no GPs), and the multisensory
causal-inference model on subject 1, which is the paper's real-data noisy
problem and is in the benchmark suite at the 2020 paper's noise level of
1.3 (`benchmark_targets.py`, `multisensory_s1_D6` with the noise wrapper);
the paper's SD 3 is one added configuration line. The runs are the
expensive part: from the paper's timings a pool of a hundred runs is
several CPU hours per target. They are a campaign under the working rules
of `dev/README.md`, one heavy process at a time on the laptop and started
only on PI instruction, placed after the reference campaigns that
`dev/TODO.md` orders first; running them on the cluster depends on the
Slurm support listed there as a follow-up. The stacking and the estimates
are seconds.

**Interface, decided after the results.** VBMC objects, posterior-GP
pairs, or the GP attached to the posterior at the end of a run so that a
saved posterior is self-contained. With posteriors alone the object falls
back to Phase 1 behaviour.

## Decisions (PI, 2026-09-12)

- The stacking optimizer and its objective stay as they are. No variance
  penalty, no in-objective bias correction.
- Two phases. Phase 1 works from the stored posteriors alone, fixes the
  corrections and makes the values clear to users; Phase 2 explores and
  tests the cross-run honest estimate before any interface change, and the
  interface is decided on its results. The items of both phases, the
  uncertainty, the naive-stacking reference and the gates are the note's
  proposals within that framing.
- S-VBMC gets tips in the style of VBMC's runtime tips. The tip about the
  ELBO is shown only when the stacked runs were noisy, never for noiseless
  targets. When a posterior does not record whether its run was noisy, the
  stored ELBO standard deviation is the proxy. Recording the noise level on
  the posterior, and the threshold of 0.1, are the note's proposals.
- Neither phase has started. Phase 1 waits on the five open decisions.
