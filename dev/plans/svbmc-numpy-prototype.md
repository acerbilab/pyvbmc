# Bounded S-VBMC NumPy prototype

Completed 2026-09-09 on `dev-svbmc-numpy-prototype` from `dev-next` at
`48ed027`. The PI authorized this experiment and specifically requested an
optimized NumPy implementation. Astra orchestrates and runs computations
serially; Sol implements and reviews. PyVBMC's NumPy/SciPy solver decision
for 1.5 is settled. S-VBMC integration remains parked. Results are summarized
in the existing ecosystem proposal and detailed in
[the report](../results/2026-09-09-svbmc-numpy-prototype.md).

## Question and scope

Can an optimized NumPy implementation of S-VBMC's weight optimization match
its sampled objective and optimizer behavior and improve CPU runtime while
removing the numerical implementation's Torch requirement? Compare S-VBMC
directly; the completed PyVBMC Stage 4 results do not answer this question.

Use the pinned upstream `13a78f6c4a3ffe9c4557fee4f4b8f67c98b29e01`
and its three groups of ten saved posteriors. Keep their component parameters,
individual transforms and GP integrals fixed. Cover component-weight and
posterior-weight optimization. Do not run new VBMC fits, repeat the completed
compatibility campaign, change production APIs, or launch the final 870-case
population benchmark.

All three shipped groups are D=2, with ten K=50 posteriors each: 500 mixture
components and a 10,000 by 500 density matrix at the default sample count.
These include final-boost component counts but do not span higher dimensions
or a population of inference problems.

## Implementation and comparison

- A developer-only NumPy module implements the sampled entropy and both
  logit gradients analytically, including the outer stratum weights and
  the derivative of the mixture density. Use stable, vectorized reductions.
- Reuse transforms/Jacobians within each input posterior, evaluate diagonal
  Gaussian log densities directly in a reusable array, and bound its size.
  Reuse one exponential workspace for the entropy value and derivative.
  Preserve upstream component draw order and
  fresh Monte Carlo draws every iteration; freezing samples for a fit would
  change the algorithm.
- Match upstream initialization, Adam defaults and epsilon placement,
  five-decimal loss comparison, five unsuccessful iterations, best-iterate
  selection, and first-draw correction caches used by the ELBO debiasing.
- Run float64 in both arms explicitly. Upstream's default float32 output
  cast is a separate dtype-policy issue, not an arithmetic parity target.
- Compare complete optimized NumPy fits with unchanged upstream Torch CPU
  fits. Attribute preparation improvements separately from objective,
  backward and optimizer costs. A vectorized Torch fixed-matrix control
  distinguishes backend costs from upstream's Python component loops.
- Use three seeds (1701, 1702, 1703), both weight modes and all three groups:
  18 matched pairs, with upstream defaults of 20 samples per component,
  learning rate 0.1 and at most 500 steps. Pilot first; retain any failures
  and do not silently change the stopping rule to force agreement.
- Report import/setup and warmup separately from clean full-fit timing;
  rotate backend order and keep BLAS/Torch single-threaded. Instrumented
  diagnostics must be distinguished from clean timing observations.

The fixed-matrix control showed that vectorized Torch can be competitive
with NumPy even while unchanged upstream complete fits are slower. To answer
the PI's optimized-implementation question without conflating backend and
preparation improvements, add one bounded attribution control: give Torch
the same optimized NumPy preparation and vectorized entropy reduction, then
inherit upstream's ELBO, Adam, stopping and result handling. Compare it with
NumPy over the same 18 cells, with fresh paired timings (36 fits). This
supplement runs after the primary campaign, uses a separate frozen script
and artifact, and does not change either primary measured implementation.
No further tuning campaign is planned from its outcome.

This is a CPU study. It does not establish GPU speed or numerical reliability;
any later GPU comparison needs synchronized timings and transfer/setup costs.
There is no assumed 1.2x acceptance threshold. CPU performance, values and
gradients, numerical reliability, code simplicity and installation needs all
contribute to the recommendation.

## Checks and evidence

- [x] Verify pinned source, saved-posterior hashes, environment and imports.
- [x] Test analytic gradients against finite differences, including uneven
  posterior sizes, unnormalized weights and extreme logits.
- [x] Check preparation, objective/gradient, Adam and complete-fit parity
  against the upstream implementation on matched random streams.
- [x] Complete bounded timing comparisons and retain machine-readable data
  under `dev/experiments/svbmc_numpy/`; raw logs go under `dev/scripts/runs/`.
- [x] Independently review implementation and interpretation; fix confirmed
  issues and run only the checks needed for those changes.
- [x] Write one detailed report directly in `dev/results/` and summarize it
  in the existing ecosystem proposal and 1.5 overview. Update TODO/roadmap.

The public upstream methods returning Torch tensors, D=1 sampling defect,
exact requested sample counts, RNG ownership, canonical VP weight shapes,
dependency floors and posterior API remain integration decisions. A local
reshape needed to test D=1 is recorded as a prototype correction, not an
integrated fix. Preserve the user-facing skill and calibration follow-ups.

## Execution record

- The isolated upstream checkout is clean at the specified revision. The
  existing environment imports S-VBMC from that checkout and Torch from the
  compatibility overlay: Python 3.12, NumPy 2.5.2, SciPy 1.18.1 and
  Torch 2.14.0+cpu. No installation or dependency upgrade was needed. As in
  the compatibility check, execution needs access to the overlay directory;
  the restricted account cannot read its `torch.optim` package.
- Hardware identified through `Win32_Processor`: Intel Core Ultra 7 155H,
  16 physical cores and 22 logical processors. The experiment uses one
  BLAS thread and one Torch intra/inter-op thread, not the full CPU.
- Inspection of the saved files confirms ten K=50 posteriors in each of
  GMM, GMM_noisy and Ring. These are the experiment inputs; no saved posterior
  is regenerated or modified.
- Ten focused NumPy tests pass, covering direct-weight and logit finite
  differences, uneven posterior sizes, gauge invariance, transform reuse,
  D=1/single-draw shapes, the correction cache and stopping behavior. An
  initial exact equality assertion for Adam's first moment differed by
  1.4e-17 because `1 - 0.9` and literal `0.1` round differently; the test
  now uses a floating-point tolerance. The optimizer formula was unchanged.
- A preparation profile of the first optimized version attributed 0.417 s
  of a 0.579 s call to SciPy's generic normal-log-density wrapper. That
  motivated direct diagonal-normal arithmetic before the measured campaign.
  The reference implementation retains SciPy's formula for parity checks.
- Preliminary GMM smoke: ten matched iterations, maximum weight difference
  5.9e-16, ELBO difference 2.7e-15 and identical final NumPy random-stream
  state. Smoke timings are developmental checks, excluded from the final
  performance summary. The harness is reviewed before freezing its sources.
- Primary campaign completed: eighteen clean pairs, six diagnostic pairs
  (105 matched iterations) and three kernel/preparation cases, zero validation
  failures. Aggregate full-fit time is 287.75 s upstream versus 116.50 s
  NumPy (2.47x). All returned weights and ELBOs agree to about 1e-15;
  stopping, best iteration, rounded losses and RNG states agree.
- The shared-preparation Torch control completed eighteen fresh pairs,
  also with zero failures: 84.90 s NumPy versus 97.35 s Torch. Median paired
  speedup is 1.06x, aggregate 1.15x. The larger Ring advantage, one high-ratio
  observation and cross-campaign timing variation are preserved in the
  report. No general backend ranking or acceptance cutoff is inferred.
- Independent Sol reviews covered the core, both runners and interpretation.
  Root fixed the identified timing/trace issues before the relevant campaign;
  the final report checks found no numerical/table mismatch. Exact measured
  sources are archived with hashes, and live/archive hashes match after
  execution. Source formatting hooks pass. No production code changed and
  no computation or watcher remains.
