# Modernization roadmap (living plan)

Status tracker for the staged plan in
`dev/2026-09-02-modernization-discussion.md` §10. That devlog holds the
rationale and the decisions; this file holds only what is done, what is next,
and where to pick up. Update it in place; do not add status to the devlogs.

Branch: `dev-next`. No PR to `main` until the work here is done: one PR at
the end (decided 2026-09-02, superseding the per-stage PRs of devlog §11).
CI on `dev*` branches: pushes that touch the package run a reduced smoke
(Ubuntu, Python 3.12) of the `tests` workflow automatically (added
2026-09-03); dispatch the workflow by hand for the full matrix before
anything that changes numerics lands.

## Stages

**Population assessment, 2026-09-10:** prepare an overnight first stage of
273 runs (15 seeds for each of 18 ordinary configurations, 3 exhaust cases)
against all 870 golden reference cases. Review the results before deciding
on further sampling; a full 870-run candidate population is optional.
The [staged benchmark plan](final-population-benchmark.md) gives the allocation,
8–10-hour estimate and remaining preparation, superseding earlier requirements
below for a full candidate sweep before assessment.

- [x] **Stage 0 — test oracle** (devlog §10)
  - [x] finite-difference checks: `entmc`, `entlb` (pre-existing),
    `_gp_log_joint`, `_neg_elcbo`, `_vp_bound_loss`, `_soft_bound_loss`,
    `vp.pdf` (`plans/profile-and-gradient-checks.md` §6)
  - [x] finite-difference checks: parameter transformer Jacobian, gpyreg
    kernel/mean/noise derivatives (`compute_vargrad` no longer applies:
    the dead path was deleted with Stage 2 item 1 on 2026-09-04; the
    variance gradient is unimplemented and raises). Done 2026-09-06
    (`test_parameter_transformer_jacobian_fd.py`: the full `D × D`
    Jacobian of `inverse` by central differences against
    `log_abs_det_jacobian`, unbounded, centered, bounded, mixed and
    rotoscale-warped cases at D = 1, 2, 5, round trips and the sign
    convention; `test_gpyreg_derivatives_fd.py`: gpyreg's mean, noise (the
    six switch combinations PyVBMC selects) and SE kernel gradients; 248
    tests, no discrepancy; three incidental findings in devlog §9)
  - [x] fixture generator script (`dev/scripts/make_oracle_fixtures.py`,
    2026-09-04: 8 snapshots as plain arrays under
    `pyvbmc/testing/oracles/fixtures/`, 1.4 MB;
    `plans/fixture-generator-and-oracles.md`). The MATLAB `.mat` fixtures
    were converted on 2026-09-06 to plain `.npz` files with a `FIXTURES.md`
    per directory (every array bit-identical, the readers' assertions and
    tolerances unchanged, the orphan `whitening/vp_initialized_MATLAB.mat`
    deleted, `MANIFEST.in` covering every fixture directory): the MATLAB
    agreement they pin is kept, the opaque format is gone.
  - [x] stage-level oracles (`pyvbmc/testing/oracles/`, 16 oracles × 8
    snapshots since 2026-09-05 (`gp_nlZ`, the GP log marginal likelihood
    and log posterior with gradients, and `gp_fit`, a whole `train_gp`
    call, added for item 8), about 25 s; per-element tolerances with a
    robust floor, set from measured rounding floors on three BLAS builds
    (Windows, Ubuntu, macOS); the per-commit gate for Stage 2). Not built
    from the devlog's Stage 0 list: `GP.quad`, `kl_div_mvn`, `kde_1d`.
    Fixtures 1.5 MB.
  - [x] golden-trace harness over the benchmark target suite
    (`dev/scripts/golden_trace.py`; first reference population
    `baseline_20260903`: 20 seeds × 14 configs (D = 2–10, two noisy) on
    the code of `5020879`, run 2026-09-03/04 with the papers' procedure,
    280 of 280 succeeded, null check clean over 56 KS tests;
    `plans/benchmark-suite-and-golden-traces.md` §Results (regenerated).
    Replaced on 2026-09-06 by `item7_20260906`, the same suite and seeds
    on the end-of-Stage-2 code `bdaf322`, after it passed against the
    first (pickup 5); sidecars and `summary.md` in git under
    `dev/golden/baseline/`, traces gitignored under
    `dev/scripts/runs/golden/item7_20260906/`. Extended 2026-09-07 on
    frozen `7314a6a`: 810 pairs, 16 configs at 50 seeds plus the D15
    exhaust at 10, all null checks passed and final replay identical
    on all five cases; `dev/golden/extension_20260907/README.md`. The current
    combined reference is `reference_870_20260907`: 60 further runs from
    pinned source `623f5cd`, 30 each for noisy Rosenbrock D2 and Student D8,
    giving 870 runs across 19 configurations and 160 noisy runs;
    `dev/golden/noisy_extension_20260907/README.md`. All archives passed
    integrity checks, the 76-test even/odd comparison had no flags, and the
    default five-case replay matched stored loop/final values and initial
    designs with zero flags; the historical returned transformer is explicitly
    uncertifiable because those traces did not store it.)
  - [x] dtype canary (`plans/stage0-dtype-canary.md`, 2026-09-06 midday;
    tests only: every raw oracle output and every rebuilt oracle state
    checked for float64 inside `test_oracles.py`, a walk of a live run in
    `test_vbmc_seed.py` on a shared short run, a manifest of the
    load-bearing arrays (`pyvbmc/testing/_dtype.py`); float32/float16
    constructor inputs found to keep their dtype in `optim_state` and the
    transformer, pinned as a strict `xfail` until the boundary cast lands
    after the nights)
- [x] **Stage 1 — `seed=` and `Generator` threading**
  (`plans/stage1-rng-generator.md`). The remaining seam (gpyreg and the
  cma noise handler drawing from the global state) was removed on
  2026-09-05 with Stage 2 item 8: gpyreg's `fit` takes `rng=`
  (acerbilab/gpyreg#43), the noise-handler subclass draws from `vp.rng`,
  and a run never reads or writes NumPy's global state; the per-iteration
  `random_state` holds only the generator state.
- [x] **Benchmark target suite** (`dev/scripts/benchmark_targets.py`,
  2026-09-02/03, corrected to the papers' procedure 2026-09-03): banana,
  cigar, lumpy, Student-t at D = 4 (lumpy and banana also at D = 10, banana
  at D = 2 and 6), two noisy VIQR targets (Rosenbrock σ = 1, bounded logreg
  σ = 3), a bounded logistic regression, a 15-D cigar budget-exhausting run;
  profiled 2026-09-03 (`plans/benchmark-suite-and-golden-traces.md`
  §Results (regenerated)). D8 cigar and Student-t and the D15 exhaust
  are now in the golden set; their reference runs completed 2026-09-07
  (pickup 3f).
- [~] **Stage 2 — NumPy vectorization + memory fix.** Order measured
  2026-09-02 on the first suite and **confirmed 2026-09-03/04 on the
  regenerated suite** (papers' procedure; the shares below are the
  regenerated ones, `plans/benchmark-suite-and-golden-traces.md` §Results
  (regenerated)): (3) batched acquisition evaluation (`GP.predict` over
  `Ns`, `vp.pdf` over `K`, the CMA-ES population; active sampling 54–69 %
  of wall on every noiseless target, single-point `predict` 40–50 % of
  profiled time at D ≤ 10, `vp.pdf` 15 % at D = 15 with K ≈ 25) —
  **PyVBMC half done 2026-09-04** (`plans/stage2-batched-acquisition.md`:
  CMA-ES evaluates each generation in one call, `vp.pdf` broadcasts over
  K; noiseless targets 1.4–1.8× faster end to end, active sampling
  2.1–2.5× and down to 36–47 % of wall, noisy VIQR targets +6–9 %; six of
  the nine converging trajectories bit-identical to the baseline; gpyreg's
  `predict` loop over `Ns` and `sW` tiling moved to item 8), (8) gpyreg
  sampler overhead (GP training 13–20 % at D = 4, 22–28 % at D = 10, and
  41 % of an iteration in the late sampling regime at D = 15; the Cholesky
  is a small part of it; plus, from item 3, `predict`'s per-sample loop and
  `sW` tiling; gpyreg PR), (1) `_gp_log_joint` einsum (variational
  stage 9–20 % at D = 4, largest on ridged posteriors and 32 % of an
  optimize-only iteration; PyVBMC-local, may land before (8) for
  logistics), (2) `_eval_full_elcbo` multi-RHS solve (also shrinks
  `final_boost`, 3–12 % of a short run) — **(1) and (2) done 2026-09-04
  evening** (`plans/stage2-gp-log-joint-einsum.md`: `_gp_log_joint`
  vectorized over samples and components with `einsum` contractions, its
  variance from two multi-RHS solves per sample; per call 10–11× faster
  at D = 4–5 and 26–62× on the variance; end to end the noiseless D ≤ 10
  targets 1.13–1.39× faster with the variational fit 2.6–5× faster and
  down to 3–12 % of wall, the 15-D exhaust run 1.24× with its
  variational fit 2.2× faster; `_gp_log_joint` 18–24 % → ~2 % of a D = 4
  run, `final_boost` 10–17 % → 4–6 %; every trajectory changed at
  rounding level, all seed-0 finals inside the population fences; four
  latent defects of the function fixed), then `entmc_vbmc` (item 5; now
  the largest piece of the variational stage at 6–9 % of a D = 4 run, and
  11 % of the 15-D exhaust run in the 2026-09-03 profile), drop
  per-candidate deepcopy, `GP.clean()` / stop
  retaining full GPs (memory only: deepcopy is < 1 % of time). On noisy
  targets the active-sampling bucket is the per-sample GP refits and VP
  re-optimizations, so (8) and (1) are what speed those up. After items
  3, 1 and 2 the remaining time at D = 4 is active sampling (46–54 %,
  gpyreg's `predict` overhead 31–34 %) and GP training (31–42 %, the
  slice sampler 27–38 %): item 8. **(8) done 2026-09-05**
  (`plans/stage2-gpyreg-predict-and-sampler.md`;
  gpyreg branch `perf/predict-sampler-overhead`, draft PR
  acerbilab/gpyreg#43, merged into gpyreg `main` as `a2f8ddc` on
  2026-09-05, `GPYREG_PIN` at the merge commit): the three performance commits are identity-preserving (every
  oracle output checked is bit-identical to a dump of the pre-change
  code, the replay `identical` after each), `predict` 1.4–1.7× per
  CMA-ES-sized call, the sampler's log-posterior evaluation 1.4–1.9×, one
  `train_gp` call 2.1–2.5× (the Cholesky reused on mean-hyperparameter
  moves); generator support in gpyreg; PyBADS's suite passes against the
  branch (one metadata-dependent test deselected). The PyVBMC seam is
  removed (every draw through `vbmc.rng`), which is *not*
  identity-preserving: it shifts the stream of every existing seed, and
  the two stream-dependent oracles were re-baselined. End to end
  (profile campaign on the identity-preserving commit, identical
  trajectories, ten configs): 1.14–1.49× faster (noiseless 1.19–1.49×),
  GP training 2.0–2.6× (23–46 % of wall → 13–29 %), active sampling
  1.11–1.20× (now 46–85 % of wall, `GP.predict` alone 30–36 % of a
  profiled D = 4 run); the 15-D exhaust run 1041 → 777 s, 2.7× since
  2026-09-03 over items 3, 1, 2 and 8 together. **(5) done 2026-09-05
  (afternoon)** (`plans/stage2-entmc.md`): `entmc_vbmc` draws every
  component's antithetic samples in one call (the same stream, in the
  same order, so the `entmc` oracle holds without a re-baseline),
  evaluates the mixture density once as a broadcast over a `(components,
  samples, D, K)` tensor with `einsum` contractions and takes the
  reparameterization gradients from the same tensors, in 2^16-element
  blocks; per call 6.7–12.5× faster at the shape Adam sees (K = 14–50 at
  D ≤ 15; 3.8× at D = 20, K = 60) and 0.9–1.35× at the value-only shape
  of `_eval_full_elcbo`, which is arithmetic-bound (a GEMM expansion would
  give 2.8–6.9× there, but its error grows without bound with the
  components' width ratio; rejected, the plan's Open question 1); oracles
  green with no
  re-baseline, the exact check against a pre-change dump moves only the
  entropy-carrying outputs at 1e-15, replay 0 flagged of 5 with the
  design identical, full suite green. Profile campaign (17:11, idle
  machine, gpyreg v1.1.0; every trajectory differs from item 8's because
  the seam removal moved every stream): in situ the Adam-shape entropy
  calls are 5× faster per call (the exhaust run's 147 → 29 s over 20k
  calls), the value-only `_eval_full_elcbo` calls gain nothing and are now
  the entropy's dominant cost on large-K runs; `entmc_vbmc` 9–13 % → 3–6 %
  of a profiled D = 4 run, `final_boost` halved, the variational fit
  0.37–0.88 of its time on nine configs; end to end within trajectory
  noise (suite 1424 → 1384 s; the exhaust run 777 s on both, its
  variational fit 244 → 209 s on a higher-K path). **(6) done 2026-09-05
  (evening)** (`plans/stage2-memory.md`, shared with (7)): `_vb_init`
  builds each sieve candidate as a shell sharing the
  run's generator and transformer instead of `copy.deepcopy(vp)`;
  bit-identical candidates and stream (27,936 candidates checked against
  the old code, replay `identical` with finals equal to item 5's), the
  `_vb_init` step 31–43 → 12–20 µs per candidate, about 0.1 % of a run
  (cProfile had overstated the copy about 6×). **(7) done 2026-09-05
  (night)** (`plans/stage2-memory.md`; decided with the PI: what can be rebuilt
  from the record is never stored, what cannot is dropped by default and
  kept under a new option): the history grows without re-copying its
  stored past (the re-copy was quadratic, 1.3 % of the exhaust run under
  cProfile; on the noisy logreg run alone 254 → 208 s of wall), each
  iteration's GP is recorded without its posterior factors and the public
  `VBMC.get_gp(iteration)` restores them bit for bit where a full GP is
  needed (`final_boost`, `load`), and the recorded `optim_state` leaves
  out the importance samples of the noisy acquisitions unless
  `record_full_history_details` is set. Retained history 9.4 → 1.9 MB on
  `cigar_D4`, 25.8 → 1.3 MB on `rosenbrock_D2_noise1`, 117 → 4.6 MB on
  `logreg_D5_noise3` (RSS after that run 332 → 163 MB, peak 427 → 273);
  the 15-D exhaust run's 323 MB of factors (analytic) become about 8 MB
  of data and hyperparameters.
  Every step replay `identical` and full suite green; the resume test
  compares the two runs' ELBOs (it compared one with itself) and pins a
  `load(iteration=)` round trip on a file written by the new code; two
  latent defects fixed on the way (a continued run aliased the history's
  last entries; the in-loop plot passed no GP). Save format: files
  written by this code hold lean GP records, so an older PyVBMC cannot
  resume them; old files load unchanged. **The 20-seed population after
  items 8, 5, 6 and 7** (2026-09-06, pickup 3e): 280 of 280 succeeded, no
  config flagged against the reference over 56 KS tests nor against the
  item 1 population, summed run time 6.41 → 4.71 h against item 1 (9.92 h
  for the reference), the runner's high-water mark 474 → 402 MB; one
  descriptive shift, lumpy_D10's median evaluations 242 → 308 with its
  quality metrics unchanged or better.
- [~] **Latent bug fixes for 1.5** (pickup 9; PI decision 2026-09-06: 1.5
  leaves the VBMC algorithm as designed and fixes the latent bugs of
  devlog §9, the trajectory-moving ones included, each replayed and the
  set checked as a population against the extended reference).
- [x] **Stage 3: connect models and use posteriors downstream.** Merged
  into `dev-next` at `4bff1a5`. Branch smoke 34043031387, all nine jobs of
  full matrix 34043071150, and integrated smoke 34043979358 passed.
  Integrated local checks: exact oracles 8/8, all five replay cases
  identical, full base-install suite 858 passed (35 skipped, 2 xfailed).
  The feature-branch suite with optional exports passed 957 tests.
  Non-moving snapshot `reference/stage3-20260906` names the exact final
  checkout for both reference batches; its final commit adds only
  verification records to tested code `4bff1a5`. Keep batching off.
  Two workflows (PI confirmed 2026-09-06): bring torch/JAX models into
  PyVBMC through documented adapters and an opt-in batched initial design;
  take the fitted posterior into torch as a distribution or ArviZ as
  samples. The plan is `plans/stage3-pipeline-features.md` on
  `dev-next-stage3`. Current ArviZ DataTree support is agreed; the core
  stays Python >=3.10 and the ArviZ export requires >=3.12. Stage 4 later
  retained those version floors. Ships in 1.5 with Stages
  0–2 (PI decision 2026-09-06: the whole body of work in one release, for
  visibility, rather than a 1.5 followed by a 1.6 within days). Both
  trajectory-neutral and trajectory-moving latent bug fixes were held behind
  the reference boundary. That boundary completed on 2026-09-07 (pickup 3f).
  Pickup 9 is in progress: Phase 1 neutral fixes are complete; Q1/Q4 decisions
  and moving groups remain open.
- [x] **Stage 4 — PyTorch feasibility prototype, then a port decision**
  (preferred for 1.5 if feasible;
  PI decision, 2026-09-06). The purpose combines future method development,
  participation in the modern ML ecosystem, and performance opportunities.
  Comparable CPU performance can be acceptable if autodiff and the tensor
  architecture substantially simplify extension and integration. Acceptance
  requires numerical reliability, float64 robustness, acceptable CPU
  performance against the modernized NumPy core, and manageable installation
  friction. PI clarification (2026-09-07): 3× runtime was an example of
  clearly unacceptable performance; concern starts at substantially smaller
  slowdowns. Around 1.2× runtime may be acceptable given the other benefits,
  but is not an agreed cutoff.
  Immediate speedup or GPU advantage is not required. Preserve the existing
  method for the port; richer posterior families, gradient-based acquisition
  optimization, and larger-scale inference remain separate future work.
  The first step is feasibility prototyping; the proposed scope is a complete
  variational optimization
  step (GP integrals, entropy, parameter handling, and optimizer), checked
  for values, gradients, CPU and GPU performance, and code simplicity,
  including final-refinement shapes. Test both devices in float64 against
  the modernized NumPy CPU baseline, include transfer costs, and report
  hardware and timings separately. GPU timings must synchronize completed
  work; separate setup/warmup from repeated execution. GPU gains do not
  remove the CPU acceptance requirement. If GPU hardware is unavailable,
  record GPU feasibility as untested and obtain measurements before closing
  this investigation. Use that evidence for an explicit decision before
  committing to the full port; a stage timing alone does not establish the
  end-to-end runtime impact. Inclusion in 1.5 remains conditional on this
  decision and subsequent implementation and validation.
  Settle the NumPy transition, core dependencies, and Python floor in the
  implementation design. JAX model adapters remain part of Stage 3;
  PyTorch is the candidate solver backend if a port is authorized. Decision
  rationale: devlog §10 and the
  [1.5 overview](../2026-09-06-pyvbmc-1.5-overview.md).
  Prototype completed (2026-09-09): the bounded
  [variational-step feasibility results](stage4-torch-feasibility.md) are on
  `dev-stage4-torch-feasibility`, from `dev-next` at `bf43c20`.
  Float64 CPU/CUDA diagnostics pass. The 24 trace-disabled whole-fit controls
  show Torch CPU at 2.12-6.26x NumPy and synchronized CUDA at 3.35-23.34x on
  the tested laptop (one clean observation per workload/arm; 72 additional
  instrumented fits retained). Recommendation: retain NumPy for this release.
  **PI decision (2026-09-09): retain the modernized NumPy/SciPy solver for
  1.5; do not undertake a full Torch solver port for this release.** Optional
  Torch and ArviZ exports remain and the existing Python floors are unchanged.
  No full port or final population run was launched or authorized.
  The [compiled-Torch follow-up](../results/2026-09-09-torch-compile-follow-up.md)
  is complete: 80 fits, fixed-input float64 gates passed, no warm recompilation.
  CUDA gains about 1.28x/1.54x over NumPy on the two boost cases (paired warm
  medians), while compiled CPU stays roughly 2-4x the NumPy runtime by per-arm medians.
  Cold compilation costs and sampled-boost endpoint drift remain material.
  The 1.5 NumPy decision stands; this did not authorize a full port.

## Pickup point

0. ~~Regenerate the benchmark results~~ done 2026-09-03/04 in two sessions
   (profile campaign plain + cProfile, golden population 20 × 14, null check
   clean, sidecars published; `plans/benchmark-suite-and-golden-traces.md`
   §Results (regenerated)). The Stage 2 order above is confirmed; committed
   and pushed as `9206738`. The one wrong posterior in the population,
   `student_D4` seed 19, is a final-boost failure, written up in
   `../results/2026-09-04-final-boost-failure.md`; the guard is a 1.5 fix (pickup
   9, PI ruling 2026-09-06).
1. ~~Stage 0 oracles first~~ done 2026-09-04 (PI: an arithmetic-preserving
   refactor is gated by fixed-state oracles, not by the 10-hour statistical
   run, which is the end-of-stage check and the Stage 4 gate):
   `pyvbmc/testing/oracles/` and `dev/scripts/make_oracle_fixtures.py`,
   `plans/fixture-generator-and-oracles.md`; reviewed (three read-only
   Opus reviews, all findings folded in), committed and pushed 2026-09-04.
   Four smoke CI runs on Ubuntu each stopped at one oracle (the
   platform-bound step oracle, then three cross-BLAS floors on the
   ill-conditioned snapshots); each was measured and its tolerance class
   set from the measurement (plan tracker); the fifth run is green (510
   passed), and a full-matrix dispatch (33865996373) is green on all nine
   jobs, macOS included: the oracles hold on three BLAS builds.
2. ~~Stage 2 item 3~~ done 2026-09-04 for the PyVBMC half
   (`plans/stage2-batched-acquisition.md`, commits `50c1e50`, `7a07c0b`,
   `f441172`, `eca45ec`, `e923163`, `3033526`): the per-change gates now
   exist and were used — the oracles on every commit (the
   `active_sample_step` oracle re-baselined once from the stored state with
   `make_oracle_fixtures.py --rebaseline`, every `acq_*` oracle unchanged),
   the golden replay (`golden_replay.py`, 7 min: iteration 0 identical on
   every config, finals inside the population envelope except one chance
   excursion of `halfnormal_D2` seed 0's gsKL, not reproduced on seeds
   1–4) per step, the profile suite once
   (1.4–1.8× on noiseless targets at D ≤ 10, 1.65× on the 15-D exhaust
   run with active sampling 3.3× faster there; a first exhaust measurement
   on a throttling laptop was repeated on a cool one). The 20-seed
   population
   (`golden_trace.py run --suite golden --seeds 0-19 --workers 1 --out
   dev/scripts/runs/golden/<label>`, then `compare dev/golden/baseline
   dev/scripts/runs/golden/<label>`, about 7 h now) stays the once-per-stage
   check.
3. ~~Stage 2 items 1 and 2~~ done 2026-09-04 (evening;
   `plans/stage2-gp-log-joint-einsum.md`, commits `9d92c7f`, `5ce1bc6`,
   `f93ea5e`, `8943cec` and the records commit after them): `_gp_log_joint`
   vectorized over hyperparameter samples and components (item 1) and its
   variance from multi-RHS solves (item 2, done in the same function
   because the pair loop lived inside the loop being removed); oracles
   green on every commit with no re-baseline, replay 0 flagged of 5 (the
   ELBO arithmetic changed, so every config parts at iteration 0; the
   replay now certifies the initial design from the traces instead), full
   suite green twice; four latent defects of the function fixed. Measured
   speedup: the Stage 2 paragraph above and the plan's §Results. The
   replay gate learned two things here: a change to the ELBO arithmetic
   parts every trajectory at iteration 0 (on cigar's chaotic warm-up
   optimization a one-ulp gradient perturbation moves the iteration-0
   ELBO by 0.4 and can push a seed outside the population fence), so the
   initial design is now certified from the traces (`X_init`) rather than
   from an identical iteration-0 ELBO; and the envelope check on a single
   seed has a visible false-alarm rate on cigar. ~~Next: item 8 as one
   gpyreg PR~~ done 2026-09-05 except the profile, see point 3a below;
   then item 5 (`entmc_vbmc`), then the memory items.
   The 20-seed population against `dev/golden/baseline` (the end-of-stage
   check of items 3, 1 and 2 together) ran 2026-09-05 00:39–07:06 on the
   final code (`a39e5ec`): **280 of 280 succeeded, no config flagged over
   56 KS tests (Holm α 0.05)**, median evaluation ratios 0.98–1.10, the
   population's summed run time 9.92 → 6.41 h (noiseless configs
   1.56–2.03× faster, noisy VIQR 1.13–1.20×); the baseline's final-boost
   failure (`student_D4` seed 19) did not recur. Reports under
   `dev/scripts/runs/golden/item1_20260905/` (`summary_item1.md`,
   `compare_vs_baseline.md`; traces gitignored). The new population is
   not promoted to the reference: `dev/golden/baseline` stays the
   pre-Stage-2 reference until the stage ends (then re-baseline once, and
   grow to 50 seeds, pickup point 5).
3a. **Stage 2 item 8** (`plans/stage2-gpyreg-predict-and-sampler.md`,
   2026-09-05): the gpyreg work is merged into `acerbilab/gpyreg` `main`
   (PR #43, squash `a2f8ddc`, 2026-09-05; bit-identical outputs, `rng=`
   support, four review fixes folded in before the merge), `GPYREG_PIN`
   points at the merge commit, gpyreg v1.1.0 released to PyPI the same day
   and required by `pyproject.toml`), PyBADS's
   suite passes against it with one metadata-dependent test deselected,
   PyVBMC's seam is removed, and the profile campaign is recorded (plan
   §Results; `runs/profile_20260905_item8/`). **Open:** (i) whether
   to run
   the 20-seed population right after item 8 (the seam removal changed
   every stream) or at the end of the stage; (ii) Open question 8 of the
   plan: re-baseline the committed oracle references that items 3, 1, 2
   moved within tolerance, so `--check --exact` against the fixtures becomes
   the identity gate.
3b. ~~Stage 2 item 5, `entmc_vbmc`~~ done 2026-09-05 (afternoon;
   `plans/stage2-entmc.md`, commit `5a8e181` and the records commit after
   it). The profile of one call from the stored
   snapshots answered the pickup question: the `K × K` density loop, not
   the draws (draws 1–9 % of a call, the density loop 56–97 %, the
   gradient block the rest, and the gradient block recomputed the
   density). Vectorized as in the Stage 2 bullet above; gates green
   (`pytest pyvbmc/testing/oracles` 116 passed with no re-baseline, since
   the draw order is preserved; `make_oracle_fixtures.py --check --exact
   --against` a pre-change dump moves only `entmc` and the
   entropy-carrying `neg_elcbo` outputs, by 1e-15; replay 0 flagged of 5
   against the item 8 Step 8 traces, initial design identical on every
   config; 541 passed). Profile campaign 17:11–17:54
   (`runs/profile_20260905_item5/`, compared with item 8's through
   `profile_compare.py --control gp_train`; plan §Results): the Stage 2
   bullet above has the numbers. The GEMM expansion for the value-only
   entropy path (2.8–6.9× there) was rejected by the PI the same day: its
   error grows without bound with the components' width ratio (plan,
   Open question 1). Item 5 is complete; the population run of 3c is the
   statistical gate for items 8 and 5 together.
3c. ~~**Next**~~ (handoff 2026-09-05 18:30, code `a0e70fe`, tree clean,
   nothing running; track (ii) done the same night, see the Status line
   at the end of this point and 3d; track (i) is point 3e). Two tracks,
   the first needing the laptop free for
   about 6.5 h and started only when the PI says so, the second light
   enough to run beside it.
   (i) **The 20-seed population** after the seam removal and item 5 (the
   statistical gate for both, neither being identity-preserving):
   `python -u dev/scripts/golden_trace.py run --suite golden --seeds 0-19
   --workers 1 --out dev/scripts/runs/golden/item8_<date>`, then
   `golden_trace.py summary <out>` and `golden_trace.py compare
   dev/golden/baseline <out>` (the item 1/2 run, `runs/golden/
   item1_20260905/` and `runs/golden_item1_20260905.log`, shows the
   chain and the report layout). Read `compare_vs_baseline.md` first:
   expected no rejection over the 56 KS tests (Holm α 0.05); a rejection
   on one config's finals would be the first evidence of a change beyond
   rounding and would reopen item 8 or 5. Not promoted to the reference
   until the stage ends (pickup 5).
   (ii) **Stage 2 items 6 and 7 (memory)** and two small follow-ups.
   Item 6: `_vb_init` (`variational_optimization.py`) deep-copies the VP
   for every sieve candidate (5k–50k per `optimize_vp`); make the copies
   cheap or unnecessary without changing any output. It is
   identity-preserving, so the gate is `make_oracle_fixtures.py
   --dump-outputs DIR` before and `--check --exact --against DIR` after
   (every oracle output bit-identical), the replay `identical` against
   `runs/golden/replay_item5_step1` traces, and the full suite; no profile
   campaign (`copy.deepcopy` is 1–2 % of a profiled run). Item 7 needs a
   plan first (`plans/stage2-memory.md`, on the pattern of
   `plans/stage2-entmc.md`): `iteration_history` deep-copies every
   iteration's GP with all `Ns` Cholesky factors and `GP.clean()` is never
   called, so memory grows as `Σ_i Ns_i N_i²`; readers of the stored GPs
   are `final_boost` (the best iteration's GP), `train_gp`'s warm start
   (`iteration_history["gp"]`), the noisy path's per-sample full update
   in `active_sample.py`, save/load and resume (`vbmc.py`,
   `iteration_history.py`). First step: measure (RSS per iteration on
   `cigar_D15_exhaust` and a noisy config, `psutil`), then decide between
   `GP.clean()` on recorded copies, storing hyperparameters and data only
   and rebuilding on demand, or keeping only the best/last GPs; gate as
   item 6. The follow-ups: the `gp_nlZ` oracle should call gpyreg 1.1's
   public `log_likelihood` / `log_posterior` instead of the mangled
   private method (`plans/stage2-gpyreg-predict-and-sampler.md`
   §Follow-ups; references bit-identical), and Open question 8 of that
   plan (re-baseline the committed oracle references to the current
   numerics so `--check --exact` against the fixtures becomes the identity
   gate) is the PI's call.
   Reading list for a fresh session: `dev/README.md`; this file's Stage 2
   bullet and pickup points; devlog §2 (the last two measured paragraphs),
   §9, §10 (the item 8 and item 5 paragraphs); `plans/stage2-entmc.md`
   §Findings, §Decisions, §Results (the current template for a
   PyVBMC-local item and its gates); `plans/fixture-generator-and-oracles.md`
   (oracle table, the dump gate); `pyvbmc/vbmc/variational_optimization.py`
   (`_sieve`, `_vb_init`); for item 7 also `pyvbmc/vbmc/iteration_history.py`
   and the `final_boost` / save / load / resume paths of `pyvbmc/vbmc/vbmc.py`.
   Then the end-of-stage re-baseline of the golden population (pickup 5)
   and the Stage 0 leftovers (pickup 6).
   **Status 2026-09-05 (evening):** track (ii) done as far as it goes
   without a decision: item 6 done (`plans/stage2-memory.md`; the gates of
   the point above, replay `identical`, full suite green), the `gp_nlZ`
   oracle calls gpyreg 1.1's public `log_likelihood` / `log_posterior`
   (references bit-identical against the dump), and the item 7 plan is
   written in the same file with the readers of the stored GPs, the
   measurements and three identity-preserving steps. Track (i), the
   population, has not run.
3d. ~~**Next** (2026-09-05, late evening)~~ done the same night: item 7's
   four steps are committed (`564f53a`, `e357216`, `1aa1933`, `ffbd4b2`)
   with their gates (`plans/stage2-memory.md` §Verification (item 7),
   §Results); the population run is point 3e. The text of the decision
   stays below for the record. (i) The 20-seed population
   exactly as in 3c (i), when the PI says the laptop is free. (ii) Item 7,
   decided with the PI (`plans/stage2-memory.md` §Decisions): the
   retention rule (what can be rebuilt from the record, the GP posteriors,
   is never stored and rebuilt on demand; what cannot, the importance
   samples of the noisy acquisitions, is dropped by default and kept under
   the new option `record_full_history_details`), lean GP records with a
   public `VBMC.get_gp(iteration)` restoring a copy, the warm start left
   on the GP records (a switch to `gp_hyp_full` would move the `gp_fit`
   oracle through its history stand-in), and the existing resume test
   made real (`elbo_1 == elbo_2`) before any history change. Four steps in the
   plan's §Design with their gates (`mem_history.py` before and after; the
   replay's finals exercise what `final_boost` receives). Open question 8
   of `plans/stage2-gpyreg-predict-and-sampler.md` (re-baseline the
   committed oracle references) is still the PI's call.
3e. ~~**Next**~~ (2026-09-06, 00:20; code `bdaf322`, the ten commits since
   `2dcb51a` unpushed; all three tracks done the same morning, see 3f).
   (i) ~~**The 20-seed population is running**~~
   **Done 2026-09-06 00:16–05:01: 280 of 280 succeeded, no config flagged
   against the reference over the 56 KS tests (Holm α 0.05), none against
   the item 1 population either** (`runs/golden/item7_20260906/
   compare_vs_baseline.md`, `compare_vs_item1.md`, `summary.md`). Summed
   run time 6.41 → 4.71 h against the item 1 population (0.74; items 8, 5,
   6 and 7 together), 9.92 h for the reference. The runner's high-water
   mark fell from 474 to 402 MB (`peak_rss_vs_item1.md`: in item 1 the
   noisy logreg runs raised it from 438 to 474 MB, here nothing after the
   lumpy_D10 runs raised it at all). One descriptive shift worth the PI's
   eye: lumpy_D10's median evaluation count went 220 (reference) → 242
   (item 1) → 308 here (KS p 0.012 against the reference, 0.034 against
   item 1, neither near the Holm threshold; its quality metrics moved
   toward the truth, median shifts −0.036 ΔLML, −0.007 gsKL), so the
   multimodal 10-D target runs longer before the stability test is met.
   Every other config's median evaluations are within 10 %. Promoted to
   the reference the same morning (pickup 5). Run on the PI's
   word, one process, code `bdaf322`
   (`python -u dev/scripts/golden_trace.py run --suite golden --seeds
   0-19 --workers 1 --out dev/scripts/runs/golden/item7_20260906`, then
   `summary`, `compare dev/golden/baseline`, `compare` with the item 1
   population and a per-config table of the sidecars' `peak_rss_mb`
   medians against item 1's, all chained by one script into
   `runs/golden_item7_20260906.log`; 4.75 h elapsed). Its sidecars record
   `18a236c`, `dirty: false`: a documentation-only descendant of `bdaf322`.
   Git metadata is captured after each run; the numerical code is unchanged. It was the statistical gate for
   items 8, 5, 6 and 7 together (none identity-preserving against the
   item 1 population, whose code predates the seam removal), and it
   passed. (ii) ~~Push `dev-next` so the CI smoke runs on the package
   commits, and dispatch the full matrix once~~ done 2026-09-06 (PI's
   word): pushed `2dcb51a..4f77e1a` at 05:14 UTC; the smoke (run
   34013498552, Ubuntu / 3.12) green in 11 min, the dispatched full matrix
   (run 34014012295, three OSes × Python 3.10–3.12) green on all nine jobs
   in 12 min, so the re-baselined oracle references hold at their
   tolerances on three BLAS builds. (iii) Then the rest of pickup 5 (grow the reference
   population to 50 seeds, add the exhaust config) and pickup 6 (the
   Stage 0 leftovers). Open question 8 of
   `plans/stage2-gpyreg-predict-and-sampler.md` was decided and done on
   2026-09-06 (PI: option A): the eight oracle references that items 3, 1,
   2 and 5 had moved within tolerance were re-baselined to the current
   numerics from their stored states, so `--check --exact` against the
   committed fixtures is the identity gate from here on
   (`plans/fixture-generator-and-oracles.md`).
   Reading list for a fresh session: `dev/README.md`; this file's Stage 2
   bullet and pickup points 3c–3e; `plans/stage2-memory.md` (§Summary,
   §Decisions, §Results, the tracker's doublecheck entry); the population
   reports above.
3f. **Reference extension complete, 2026-09-07.** The PI authorized both
   batches as a sequential chain on September 6. The chain ran 20:01–08:55
   (UTC+03), including preflight/final replays and checks, on detached
   `7314a6a` (`reference/stage3-20260906`) and the original reference `.venv`.
   Stage 3's prior integrated exact oracles, replay, local suite and CI
   gates remain recorded in `plans/stage3-pipeline-features.md`.

   - Batch 1: 420 new runs, original 14 configurations at seeds 20–49,
     completed 20:04–03:26; 700 complete pairs. Summary and both null
     checks passed (even/odd; seeds 0–19 versus 20–49), 56 KS tests each,
     before baseline publication and batch 2 started.
   - Batch 2: `cigar_D8` and `student_D8` at seeds 0–49, then
     `cigar_D15_exhaust` at seeds 0–9; 110 new runs, completed 03:26–08:52.
     Final total 810 JSON/NPZ pairs; no errors, final even/odd null check
     passed over 68 KS tests (Holm alpha 0.05). The exhaust configuration
     reaches its intended 750 evaluations; its 10 seeds cover that regime.
   - Fresh preflight and final default five-case replays were identical,
     including initial designs; the final replay used the expanded fences.
     Completeness, NPZ integrity, finite metrics, sidecar SHA/options,
     dependency/thread metadata and publication copies were checked.
     SHA256 checks verified all 280 historical JSON/NPZ pairs unchanged.
   - Historical sidecars retain `18a236c`, dirty=false (numerical `bdaf322`).
     New records retain `7314a6a`: 420 dirty=false, 110 dirty=true because
     first-batch baseline publication changed tracked documentation.
     HEAD and numerical source stayed frozen; every population command
     explicitly passed `{"vectorized_target": false}`, with one worker and
     single-threaded BLAS. Light laptop use makes wall times observational.
   - All 810 sidecars plus `summary.md` are published under
     `dev/golden/baseline/`. Reports, provenance, exact command allocation,
     validation metadata and the full file-hash manifest are under
     `dev/golden/extension_20260907/`; traces remain local/gitignored under
     `dev/scripts/runs/golden/item7_20260906/`. The checkout returned to
     current `dev-next`, preserving the newer PyTorch feasibility decisions.
     The frozen reference branch is unchanged. No campaign process remains.

   **Next:** prepare the pickup 9 latent-fix plan and implement its approved
   sequence, with each trajectory-moving fix replayed against this reference.
   Complete the remaining S-VBMC/extension and PyTorch feasibility scope,
   then validate the final release code as a population before the 1.5 PR.
   Read `dev/golden/extension_20260907/README.md`, pickups 9–11, and devlog
   §9 before starting correctness work. Neither a latent fix nor a solver
   implementation was made as part of the reference extension.
4. ~~Run the `tests` workflow on `dev-next` for the package fix~~ done
   2026-09-03 (full matrix green, run 33715620257); pushes to `dev*` now
   run a smoke automatically.
5. ~~Re-baseline the golden population at the end of Stage 2~~ done
   2026-09-06 (PI: promote now, grow later): the population of that night
   (`item7_20260906`, code `bdaf322`) replaced `baseline_20260903` as the
   reference after passing against it (no config flagged over 56 KS
   tests; null check clean), sidecars and `summary.md` under
   `dev/golden/baseline/`, the comparison reports under
   `dev/golden/promotion_20260906/`, the replay's default trace directory
   moved to it (`golden_replay.py`), `dev/golden/README.md` updated.
   **Extension completed 2026-09-07** after Stage 3 integration, on frozen
   `7314a6a` (pickup 3f): 16 configurations at 50 seeds plus the D15 exhaust
   at 10, 810 complete pairs and 811 baseline files. Original seeds 0–19
   retain `18a236c` and unchanged traces; 530 new runs record the frozen
   integrated SHA with explicit batching=false. Both first-batch null
   checks and the final 68-test null check passed; both five-case replays
   were identical. Records and commands: `dev/golden/extension_20260907/`.
6. ~~Stage 0 remaining after the oracles: finite-difference checks for the
   transformer Jacobian and the gpyreg derivatives (`compute_vargrad`
   dropped from the list: the path was deleted with Stage 2 item 1); retire
   the `.mat` fixtures once the oracles cover what they pin.~~ Done
   2026-09-06 by two Opus agents from written specs, tests only (the
   reference population is unaffected): the finite-difference modules
   (Stage 0 bullet above; no discrepancy, three incidental findings in
   devlog §9) and the `.mat` fixtures converted to `.npz` rather than
   deleted (PI: keep the MATLAB agreement, drop the opaque format).
   The dtype canary, the last Stage 0 item, followed on 2026-09-06 midday
   (`plans/stage0-dtype-canary.md`). A first draft of that plan, written
   by an Opus orchestrator, was discarded because its verification section
   recorded gates that had not run; it was redone with Opus agents
   confined to read-only review.
7. ~~gpyreg generator support (`GP.fit`, `SliceSampler`, `f_min_fill`,
   `GP.random_function`) on a gpyreg branch when convenient.~~ Done
   2026-09-05 in acerbilab/gpyreg#43 (item 8, point 3a); the PyVBMC seam is
   removed on `dev-next`; gpyreg v1.1.0 released 2026-09-05 and required
   by `pyproject.toml`.
8. One PR `dev-next` → `main` when the work is done, Stage 3 included
   (decided with the PI 2026-09-06: Stages 0–3 ship together as 1.5).
   With the 1.5 release that follows it, attach the reference population's
   `.npz` traces as a release asset (PI decision 2026-09-06): one zip per
   reference, made from the traces of the population the released code
   was validated against (currently 870 traces across 19 configurations),
   unpacked to
   `dev/scripts/runs/golden/<population>/`
   for `golden_replay.py`'s per-iteration verdict; `dev/golden/README.md`
   names the asset. The sidecars stay in git, the traces stay out of it,
   and anyone working on the numerics can fetch them. Until the release
   the traces exist only on the machine that ran the population
   (regenerable from the sidecars' code SHA, seeds and options);
   a copy on the lab server is cheap insurance.
9. **Latent bug fixes for 1.5** (PI decision 2026-09-06: 1.5 does not
   change the VBMC algorithm, but it fixes the latent bugs; own plan under
   `plans/` before it starts). PI update (2026-09-08): the boost default is
   unpenalized final refinement with the joint ELBO/ELCBO(beta5) tolerance
   0.1 guard (implemented, CI passed). The PI also selected eta treatment A:
   remove eta bounds and caller-theta mutation, retaining the separate
   main-loop small-weight penalty and other bounds. Production integration
   passes 64 focused tests and all 11 exact fixtures. All six bounded paired
   replays are usable; one Normal D5 gsKL fence flag is retained for final
   population assessment. See [the fix note](../results/2026-09-08-eta-bound-fix.md).
   Stopping-rule improvements are explicitly deferred research outside this
   fix campaign. These decisions supersede the historical pending choices
   below. The [implementation plan](latent-bug-fixes.md)
   is partially approved (2026-09-07): D1–D5 and Q2/Q3 settled; Q1/Q4
   comparison designs agreed, final scientific choices await evidence.
   Compare no eta-bound penalty, MATLAB's historical formula with consistent
   implementation, and current Python as a diagnostic only. Test a boost
   score-loss tolerance of 0.1 and 0.2 for every b in [0,5] (the endpoints
   suffice), with boost weight shrinkage disabled. Phase 0 and all Phase 1
   trajectory-neutral fixes are complete and verified; CI run 135 passed
   1,043 tests with 49 skipped. The exact oracles and five-case replay retained
   their reference outputs. Q1 and Q4 remain open, and the boost and other
   trajectory-moving groups have not been selected by this result. The
   original validation sequence was neutral fixes, boost, then other moving
   groups. PI update (2026-09-08): park the boost experiment at `764a177`
   on `dev-final-boost` and proceed independently with weighted GP covariance,
   acquisition regularization and GP sampling termination on
   `dev-main-loop-fixes`. The boost restart instructions are in
   [the pilot note](../results/2026-09-08-boost-penalty-pilot.md#parked-experiment-restart).
   Each moving group still receives its own numerical gate; no boost batch
   is scheduled, and final Q1/Q4 choices remain evidence-dependent.
   The PI originally approved extending the reference itself by 150 runs: add
   rosenbrock_D2_noise3, student_D8_noise3 and lumpy_D10_noise3, each at seeds
   0–49, to the standard benchmark set. That proposal would have produced
   960 runs across 20 configurations (250 noisy runs across five configs).
   Preparation completed on 2026-09-07 with isolated reference source at
   `623f5cd` (benchmark-only addition to `7314a6a`), verified worker imports,
   a frozen 150-task manifest, seven passing focused tests, and independent
   Sol review. The existing 810 pairs remain byte-identical.
   This original preparation was superseded later that day: after timing one
   seed each, the PI selected and authorized launching 30 seeds each of noisy
   Rosenbrock D2 and Student D8, reusing the two timing pairs (58 new runs).
   The reduced batch completed: 870 runs now form the combined reference across
   19 configurations, including 160 noisy runs across four configurations,
   with 76 KS tests. All 60 additions produced complete pairs; 53 converged
   and seven noisy Rosenbrock runs reached their budget. No noisy lumpy runs
   are included, and the pre-existing 810 pairs remain byte-identical. The
   active traces are under
   `dev/scripts/runs/golden/reference_870_20260907/`; integration evidence is
   under `dev/golden/noisy_extension_20260907/`. All archive, allocation,
   provenance, 76-test even/odd and default five-case replay gates passed with
   no flags; the replay matched stored loop/final values and initial designs.
   The historical returned transformer remains explicitly uncertifiable
   because it was not stored. The plan owns preparation, source isolation,
   verification, experiment and per-change gate details.
   The original reference boundary was completed on
   2026-09-07 (pickup 3f). The agreed boundary was:
   Stage 3 merges first and that integrated code is frozen for both
   reference-extension nights. No latent fix lands before or between those
   nights, whether classified trajectory-neutral or trajectory-moving.
   Pickup 9 fixes land only after both nights and their records are complete,
   so that its final population night runs on the final code. The
   candidates are the unfixed entries of devlog §9; the split below is a
   first reading, to be confirmed in the plan. Where the line between a
   bug and an involuntary design decision is loose, the PI rules case by
   case: the plan lists such items as open questions, it does not decide
   them.
   Trajectory-moving, each replayed on its own and the set checked as a
   population against the extended reference, which then becomes the
   release's reference (a separately authorized population campaign): the
   `_get_hyp_cov` slips (production slice-samples with gpyreg's default
   widths); the misspelled `variance_regularized_acqfcn` key (the
   variance-regularization branch of every acquisition is dead and
   `tol_gp_var` has no effect; the dead block's batch-shape bug comes
   with it, and `test_vbmc_init.py` asserts the misspelling); the
   misspelled `stop_gp_sampling` key (`_is_gp_sampling_finished` and
   `tol_gp_var_mcmc` are dead, and the method reads undeclared history
   keys, so it is an implementation, not a one-liner); the `final_boost`
   guard (`../results/2026-09-04-final-boost-failure.md`, option 1: keep the
   pre-boost posterior when the boosted one's ELCBO is worse; PI ruling
   2026-09-06, a borderline bug rather than an algorithmic decision;
   alters one trace of the 280 in the reference). To verify against
   MATLAB before deciding: the in-place `eta` shift in `_neg_elcbo` that
   keeps the eta upper soft bound from firing, and `vp.pdf`'s uncorrected
   gradient in the original space (moves a trajectory only if a run path
   uses it).
   Trajectory-neutral, landing with the replay `identical`: the
   float32/float16 widening cast; the integer placeholders and the
   `var_ss` docstring; `optimize_lambda` → `optimize_lambd`;
   `ParameterTransformer.__eq__`'s self-comparison of `scale`; the
   `true_mean`/`true_cov` guard; the `display`/`log_file_level` option
   comments; `results["rng_state"]`; whether `optimize` calls
   `FunctionLogger.finalize()`; the unreachable code behind
   `compute_var_log_joint`, `mcmc_importance_sampling` and
   `search_cmaes_best`; the cubic's closure variable; `kl_div_mvn`'s
   decorator; the `_vb_init` shape at `vb_type = 3`; the logit Jacobian's
   overflow; the `|det R| = 1` assumption of `log_abs_det_jacobian`;
   `_compare_matlab.rand_int`; `noisy_cigar`; notebook 1's `lml_true`
   and notebook 6's noise broadcast; the runtime dependencies in
   `pyproject.toml`; the scipy private imports in `priors/`; gpyreg's
   `step_out` stale coordinates (inert for current PyVBMC; PI deferred to
   [gpyreg #44](https://github.com/acerbilab/gpyreg/issues/44) on 2026-09-08;
   no repair or pin bump required for this release).
   Stays deferred as algorithmic (devlog §12): `compute_var == 2`, noise
   shaping, log-space mixture sums.
10. **S-VBMC integration and compatibility** (PI, 2026-09-06: part of the
    1.5 work; how to integrate is still to be decided). The broader goal
    is to bring in or connect better to S-VBMC and other upcoming VBMC
    extensions; individual extension integration details remain open
    (PI clarification, 2026-09-06). `acerbilab/svbmc`
    (Stacking VBMC) is the lab's own package, separate on PyPI (`svbmc` 0.1.1 at a glance on
    2026-09-06; last commit 2025-11-10; depends on `pyvbmc>=1.0.4`,
    `GPyReg>=1.0.2` and `torch>=2.7`) that stacks the components of
    several `VariationalPosterior` objects, which its users pass in as
    pickles of finished runs, and ships thirty such pickles under
    `vbmc_runs/` for its tests and notebooks. Decide the form of the
    integration (a separate package as today, an optional extra, or a
    module of PyVBMC), check compatibility of `dev-next` with it (its
    test suite against this branch; the pickled posteriors it carries
    are instances of the current classes, so they are exposed to the same
    attribute changes as PyVBMC's own static pickles; the shared
    `torch` dependency meets Stage 3's `vp.to_torch()`), and release it
    in step with 1.5: its `pyvbmc` pin bumped, its tests green against the
    released code, its pickles regenerated if the classes changed.

    **Compatibility pickup (2026-09-08):** PI authorized the compatibility
    check first, on `dev-svbmc-compat` from `dev-next` at `feadf30`.
    PI clarification: favor making S-VBMC directly available in PyVBMC,
    and establish a common delivery approach for ecosystem additions rather
    than creating a separate package for each algorithm. The concrete API,
    dependency treatment and migration remain design questions; see
    [the integration proposal](../2026-09-08-ecosystem-integration.md).
    PI subsequently endorsed this direction and parked implementation for
    later; retain it in release planning. Stage 4 feasibility is the next
    proposed workstream, not an authorization to begin the full port.
    - [x] Pin an isolated S-VBMC checkout, inspect its test/load contracts,
      and verify actual PyVBMC/gpyreg/Torch imports and versions.
      S-VBMC `13a78f6`, PyVBMC `feadf30`, gpyreg `a2f8ddc`; local
      Python 3.12.6, NumPy 2.5.2, SciPy 1.18.1, Torch 2.14.0+cpu.
      Torch is isolated under the ignored compatibility run directory.
    - [x] Run its existing suite against current PyVBMC and check all 30
      shipped posterior pickles, including the operations stacking uses.
      Existing upstream suite: 32 passed in 3.53 s; its VP interfaces are
      mocks. All 30 real posteriors load/sample/evaluate densities; all three
      groups retain 10/10 inputs and pass entropy gradients, a 3-step weight
      optimization, sampling, and input parameter/transformer preservation.
    - [x] Distinguish compatibility regressions from upstream/environment
      issues; independently review and record findings and next steps.
      See [the compatibility record](../results/2026-09-08-svbmc-compatibility.md).
      Independent Sol review found no record/checker issues. The shipped
      D=2 corpus is compatible; a separate D=1 probe confirms an existing
      S-VBMC draw-shape bug to fix during integration. No core or pickle
      changes were needed. Compatibility check complete; integration API
      and ecosystem delivery design are the next work.
    PI follow-up (2026-09-09): inspect S-VBMC and consider a NumPy port after
    the poor eager-Torch Stage 4 results. Static inspection finds a narrow
    weights-only Torch boundary; NumPy/SciPy already handle sampling,
    transforms and density construction. Compare S-VBMC itself before making
    performance claims or choosing its backend. See the NumPy alternative in
    [the integration proposal](../2026-09-08-ecosystem-integration.md).
    The PI then authorized a bounded optimized NumPy prototype, now complete:
    eighteen primary fit pairs, eighteen shared-preparation control pairs,
    six diagnostic pairs and ten focused tests pass. NumPy is 2.47x faster
    than unchanged upstream; with preparation improvements shared with Torch,
    its median paired speedup is 1.06x (1.15x aggregate). Most of the original
    gain comes from preparation, while the remaining advantage depends on
    the workload. See the
    [results](../results/2026-09-09-svbmc-numpy-prototype.md) and
    [plan](svbmc-numpy-prototype.md).
    **Integration proposal (PI, 2026-09-09):** write the proposal for a human
    reviewer, assuming familiarity with both methods and briefly explaining
    concrete technical seams. S-VBMC already exists in its own repository;
    this is integration work, not algorithm implementation. The revised
    [proposal](../2026-09-08-ecosystem-integration.md) recommends retaining
    Torch and the current object workflow, optional imports/dependencies and
    a legacy migration route. Targeted fixes and preparation optimizations
    remain separate changes. PI clarification (2026-09-09): await review
    from S-VBMC's main developer before proceeding. Integration code remains
    parked and no production port is authorized.
    Root owns numerical execution and tracking; Sol may inspect compatibility
    contracts read-only in parallel. Run one heavy computation at a time.
11. **Stage 3 integrated before the reference extension** (PI,
    2026-09-06). Feature code `4ee612d` was fast-forwarded into `dev-next`
    at `4bff1a5`. Branch smoke 34043031387, all nine full-matrix jobs
    34043071150, and integrated smoke 34043979358 passed. Integrated
    local checks: exact oracles 8/8, all five replay cases identical,
    full suite 858 passed (35 skipped, 2 xfailed). Final reference snapshot:
    `reference/stage3-20260906` (only verification records after tested
    code `4bff1a5`). Keep every reference configuration at
    `vectorized_target=False`. Existing `18a236c` sidecars and traces stay
    intact; new sidecars identify the actual frozen commit. No code change,
    including a trajectory-neutral latent fix, lands between the freeze
    and completion of night 2. Heavy local checks do not run while either
    night is running. The plan (`plans/stage3-pipeline-features.md`) and
    these decisions were taken with the PI on 2026-09-06:
    (a) torch and ArviZ are optional extras, `pyvbmc[torch]` and
    `pyvbmc[arviz]`, imported lazily inside the methods that need them,
    with an error naming the extra to install; their tests skip when the
    extra is absent, and the Ubuntu newest-Python leg of CI installs both,
    torch from the CPU wheel index; conda-forge has no extras, so the
    install docs name the conda packages. Torch as a hard dependency of
    the core was Stage 4's question (devlog §13); the 2026-09-09 decision
    retains the NumPy core and keeps Torch optional for 1.5.
    (b) `vp.to_torch()` builds and returns a torch distribution; nothing
    torch is stored on a PyVBMC object, so the dtype canary stays as it is
    (its Decision 5; Stage 4 revises the canary when the state itself
    becomes tensors).
    (c) The batched target: a boolean option `vectorized_target`, default
    off, in the basic options. When on, the target always receives a 2-D
    array of shape (N, D), N ≥ 1, the single-point calls of active
    sampling included as (1, D), and returns an array of shape (N,), with
    (N, 1) accepted and squeezed, or for a noisy target the pair of such
    arrays that vectorizes today's pair of scalars (an (N, 2) array is
    refused with a message saying to return a pair). The batch path is
    the initial design only, through a `FunctionLogger.batch_call` that
    evaluates the batch once and records each row as the sequential loop
    does, in input order, the batch's wall time split equally across
    rows, the evaluation count raised by N; rows with provided values keep
    going through `add`. Every later evaluation stays sequential by
    construction (the five points of an iteration are chosen one after
    another with a rank-1 GP update in between), so the payoff is the
    `fun_eval_start` evaluations in one call, plus one code path for users
    with vectorized torch or JAX models and a path batched acquisition
    (deferred) would reuse. Rejected: detecting batch support by probing
    the target (costs an evaluation, misreads functions that broadcast);
    a process pool over the initial design for non-vectorized targets
    (pickles the user's closure, differs under Windows spawning,
    duplicates the user's framework). The default path is unchanged, so
    the replay reads `identical`; a unit test asserts batch and sequential
    agree on a vectorizable target.

12. **Machine-local performance calibration** (PI proposal, 2026-09-07;
    next workstream selected 2026-09-09, outside pickup 9's latent-fix plan).
    Implementation of the approved [package integration plan](machine-local-calibration.md)
    is merged into `dev-next`; the feature branch is deleted. The local suite passed
    1,140 tests (35 skips); docs/distributions and isolated wheel checks pass.
    The [full CI matrix](https://github.com/acerbilab/pyvbmc/actions/runs/34398191188)
    passed all nine OS/Python jobs on `46c16b7`.
    Two final campaigns took 42.39/43.53 s and retained all three defaults. Planning/status
    updates stayed on `dev-next` without creating a branch.
    Calibration is intended to ship in PyVBMC. The PI clarified that choices
    stay fixed per run, including final boost and save/resume; about 30 seconds
    for an occasional machine/configuration campaign is an acceptable estimate,
    not a limit. Complete the campaign, with a generous watchdog only for
    excessive runtime (e.g. five minutes). Users
    receive feedback and explicitly initiate or rerun the campaign.

    Implemented workflow: every `pyvbmc.calibrate()` call requests a new bounded
    campaign, with progress and a result summary. There is no `force` flag;
    normal optimization loads compatible cached settings or
    uses historical defaults and suggests calibration. It never starts a
    surprise campaign after first use, cache loss or a software upgrade.
    This supersedes the initial automatic-first-use/few-second suggestion.

    The plan specifies three fixed PDF/entropy budgets, a per-user JSON cache
    keyed by machine and numerical environment, atomic writes, concurrency,
    no-persistence fallback, numerical/RNG checks, conservative held-out
    selection and run provenance. It maps the package/API/options/VP lifecycle
    changes, docs and tests for Sol implementation with Astra orchestration.
    The approved API, helper dependencies and campaign thresholds are implemented;
    inclusion in 1.5 is confirmed by the PI. Another CPU/stack is useful validation,
    but a cheaper proxy campaign is not a prerequisite for this design.

    The bounded developer prototype and [first measurements](../results/2026-09-09-machine-local-calibration.md)
    are complete: two balanced sweeps retain 65,536 in all nine regimes.
    Discovery costs 15.01/27.14 s; full measured work 23.74/44.49 s after imports.
    PDF checks are exact; entropy differences are rounding-level with identical
    RNG advancement. These pre-integration results inform the package feature;
    its own measurements and verification are recorded in the linked plan.

    Starting knobs are `vp.pdf`'s `2**16` row budget and `entmc_vbmc`'s
    `_MAX_TENSOR_ELEMENTS`. Historical PDF timings in
    `stage2-batched-acquisition.md` favored 2^16 over 2^14 and 2^18 on this
    laptop; `stage2-entmc.md` records component/sample tiling. Preserve
    float64, sample counts and estimator semantics. PDF row chunks are
    independent; entropy chunks change addition order. Check tight numerical
    agreement and identical RNG advancement, pin settings for exact gates,
    and preserve profiles across save/resume. Different profiles need not
    produce identical seeded trajectories; no separate entropy policy gate
    remains. S-VBMC awaits its main developer's review, and the final 870-case
    benchmark remains deferred. Run only one heavy computation at a time.

13. **Optional runtime hints — implemented and verified** (2026-09-10).
    The [runtime-tips plan](runtime-tips.md) owns the approved policy, PI-edited
    wording and completed acceptance checks. Focused verification passed
    (151 tests), the full suite passed (1170 passed, 35 skipped), the docs built,
    and independent Sol review found no issues. Occasional startup tips now
    provide useful guidance with links.

    Start with a small, curated collection drawn from the FAQ and planned
    user-facing skill: comparing multiple runs and their diagnostics,
    interpreting `elbo_sd`, and discovering S-VBMC when available. Keep shared
    guidance consistent across hints, documentation and the skill; link to
    fuller explanations where useful.

    A limited share (PI suggests no more than 25-30%) may introduce complementary
    methods such as S-VBMC or PyBADS, explaining a useful task rather than promoting
    the software. Use a factual low-frequency category, which can hold other tips
    too. Shuffle so short sessions can encounter these tips; prefer an ordinary
    tip after a low-frequency tip, but make progress if only low-frequency tips
    remain. The initial catalog contains six ordinary and two low-frequency tips.

    Show at most one hint per optimization start, with conservative frequency
    and repetition suppression. PI decisions (2026-09-10): a calibration reminder
    takes the shared hint slot, so skip the tip when it is shown; tip history
    stays within the Python session, with no new persistent store. Reuse a small
    shared hint-output helper for calibration and tips. Prefer relevant guidance
    when the current configuration supplies enough context. Respect `display="off"` and
    provide a separate way to disable hints. Selection must not consume the
    inference RNG or change numerical results. Use the existing startup
    output location. The approved option is `show_tips=True`, with tips on eligible
    first starts 1/4/7 and no repeats within a Python session. Merged to `dev-next`
    as `fc64381` after a green nine-job feature matrix and package build; the
    post-merge full matrix and automatic smoke check both passed. The merged
    feature branch has been removed. The final benchmark schedule remains unchanged.

## Post-release follow-up

- [ ] After PyVBMC 1.5 is released, respond to
  [issue #138: finer control over random variate generation](https://github.com/acerbilab/pyvbmc/issues/138)
  with the released API/docs. The RNG modernization addresses its core request:
  `VBMC(seed=generator)` accepts a NumPy `Generator`, shared through inference,
  GP fitting and posterior operations; standalone VPs and built-in prior sampling
  also accept generators. Point to the reproducibility/global-state tests and
  saved generator state. Explain that `seed=None` derives a generator from the
  global legacy state for compatibility, while legacy `RandomState` objects
  are not directly accepted. User-supplied stochastic callbacks remain responsible
  for their own RNGs, and not every statistical test is deterministic. See
  [Stage 1 RNG work](stage1-rng-generator.md) and Stage 2 item 8 above.
  This is a post-release response reminder; no issue comment has been posted.

## Deferred (devlog §12)

Variational optimizer stopping-rule improvements (PI, 2026-09-08): the
[equal-iteration eta experiment](../results/2026-09-08-eta-equal-budget.md) found
that longer local fits improved surrogate scores on two saved noisy states.
Investigating broader accuracy/runtime tradeoffs is future research, outside
the latent-fix campaign and not a release blocker. Preserve current stopping
settings for the accepted fixes; no additional stopping experiment is scheduled.

Per-component `lambd`, gradient-based acquisition optimization, batched
acquisition (parallel target evaluations within an iteration via local
penalization or Kriging believer, a research item; not the batched
acquisition *evaluation* of Stage 2 item 3), multi-chain slice sampling,
scaling to `N ≈ 2k–5k`, log-space
mixture sums, user-facing agent skill (`2026-09-02-user-agent-skill.md`),
porting MATLAB's diagonal approximation of the log-joint variance and its
gradient (`compute_var == 2` in `_gp_log_joint`, which raises "not
implemented"; it would allow the ELCBO gradient with `beta ≠ 0`, which no
option enables today; the dead accumulators were deleted 2026-09-04).
The guard for `final_boost` (`../results/2026-09-04-final-boost-failure.md`) left
this list on 2026-09-06: the PI ruled it a borderline bug, and it is a
1.5 fix (pickup 9).
