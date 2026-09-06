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

- [~] **Stage 0 — test oracle** (devlog §10)
  - [x] finite-difference checks: `entmc`, `entlb` (pre-existing),
    `_gp_log_joint`, `_neg_elcbo`, `_vp_bound_loss`, `_soft_bound_loss`,
    `vp.pdf` (`plans/profile-and-gradient-checks.md` §6)
  - [ ] finite-difference checks: parameter transformer Jacobian, gpyreg
    kernel/mean/noise derivatives (`compute_vargrad` no longer applies:
    the dead path was deleted with Stage 2 item 1 on 2026-09-04; the
    variance gradient is unimplemented and raises)
  - [x] fixture generator script (`dev/scripts/make_oracle_fixtures.py`,
    2026-09-04: 8 snapshots as plain arrays under
    `pyvbmc/testing/oracles/fixtures/`, 1.4 MB; retiring the `.mat`
    fixtures is still open, `plans/fixture-generator-and-oracles.md`)
  - [x] stage-level oracles (`pyvbmc/testing/oracles/`, 16 oracles × 8
    snapshots since 2026-09-05 (`gp_nlZ`, the GP log marginal likelihood
    and log posterior with gradients, and `gp_fit`, a whole `train_gp`
    call, added for item 8), about 25 s; per-element tolerances with a
    robust floor, set from measured rounding floors on three BLAS builds
    (Windows, Ubuntu, macOS); the per-commit gate for Stage 2). Not built
    from the devlog's Stage 0 list: `GP.quad`, `kl_div_mvn`, `kde_1d`.
    Fixtures 1.5 MB.
  - [x] golden-trace harness over the benchmark target suite
    (`dev/scripts/golden_trace.py`; reference population `baseline_20260903`:
    20 seeds × 14 configs (D = 2–10, two noisy) on the code of `5020879`,
    run 2026-09-03/04 with the papers' procedure, 280 of 280 succeeded,
    null check clean over 56 KS tests; sidecars and `summary.md` in git
    under `dev/golden/baseline/`, traces gitignored under
    `dev/scripts/runs/golden/baseline_20260903/`; expand to 50 seeds and
    add the exhaust config; `plans/benchmark-suite-and-golden-traces.md`
    §Results (regenerated))
  - [ ] dtype canary
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
  §Results (regenerated)). Still to add: D = 6–8 for cigar and Student-t,
  the exhaust config in the golden set.
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
- [ ] **Stage 3 — pipeline features** (batched initial design,
  torch/jax target adapter docs, `vp.to_torch()`, ArviZ export).
- [ ] **Stage 4 — PyTorch port** (decision point, not default).

## Pickup point

0. ~~Regenerate the benchmark results~~ done 2026-09-03/04 in two sessions
   (profile campaign plain + cProfile, golden population 20 × 14, null check
   clean, sidecars published; `plans/benchmark-suite-and-golden-traces.md`
   §Results (regenerated)). The Stage 2 order above is confirmed; committed
   and pushed as `9206738`. The one wrong posterior in the population,
   `student_D4` seed 19, is a final-boost failure, written up in
   `2026-09-04-final-boost-failure.md`; the guard is deferred (see
   Deferred below).
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
3e. **Next** (2026-09-06, 00:20; code `bdaf322`, the ten commits since
   `2dcb51a` unpushed). (i) ~~**The 20-seed population is running**~~
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
   Every other config's median evaluations are within 10 %. Not promoted
   to the reference until the stage ends (pickup 5). Run on the PI's
   word, one process, code `bdaf322`
   (`python -u dev/scripts/golden_trace.py run --suite golden --seeds
   0-19 --workers 1 --out dev/scripts/runs/golden/item7_20260906`, then
   `summary`, `compare dev/golden/baseline`, `compare` with the item 1
   population and a per-config table of the sidecars' `peak_rss_mb`
   medians against item 1's, all chained by one script into
   `runs/golden_item7_20260906.log`; 4.75 h elapsed). Its sidecars say
   `dirty: true` because the record files of this commit were uncommitted
   when it started; no code differed. It was the statistical gate for
   items 8, 5, 6 and 7 together (none identity-preserving against the
   item 1 population, whose code predates the seam removal), and it
   passed. (ii) Push `dev-next` so the
   CI smoke runs on the package commits, and dispatch the full matrix
   once before anything else lands (the oracle tests run on three BLAS
   builds there). (iii) Then pickup 5 (the end-of-stage re-baseline of the
   golden population, 50 seeds, the exhaust config) and pickup 6 (the
   Stage 0 leftovers); Open question 8 of
   `plans/stage2-gpyreg-predict-and-sampler.md` is still the PI's call.
   Reading list for a fresh session: `dev/README.md`; this file's Stage 2
   bullet and pickup points 3c–3e; `plans/stage2-memory.md` (§Summary,
   §Decisions, §Results, the tracker's doublecheck entry); the population
   reports above.
4. ~~Run the `tests` workflow on `dev-next` for the package fix~~ done
   2026-09-03 (full matrix green, run 33715620257); pushes to `dev*` now
   run a smoke automatically.
5. Grow the golden population to 50 seeds (`golden_trace.py run --seeds
   20-49`, about 14 h on one process) and add cigar/Student-t at D = 6–8
   and the exhaust config (`plans/benchmark-suite-and-golden-traces.md`
   §Follow-ups). The reference sidecars live in git under
   `dev/golden/baseline/` (PI decision 2026-09-03); copy them over after
   every extension.
6. Stage 0 remaining after the oracles: finite-difference checks for the
   transformer Jacobian and the gpyreg derivatives (`compute_vargrad`
   dropped from the list: the path was deleted with Stage 2 item 1); retire
   the `.mat` fixtures once the oracles cover what they pin.
7. ~~gpyreg generator support (`GP.fit`, `SliceSampler`, `f_min_fill`,
   `GP.random_function`) on a gpyreg branch when convenient.~~ Done
   2026-09-05 in acerbilab/gpyreg#43 (item 8, point 3a); the PyVBMC seam is
   removed on `dev-next`; gpyreg v1.1.0 released 2026-09-05 and required
   by `pyproject.toml`.
8. One PR `dev-next` → `main` when the work is done.

## Deferred (devlog §12)

Per-component `lambd`, gradient-based acquisition optimization, batched
acquisition (parallel target evaluations within an iteration via local
penalization or Kriging believer, a research item; not the batched
acquisition *evaluation* of Stage 2 item 3), multi-chain slice sampling,
scaling to `N ≈ 2k–5k`, log-space
mixture sums, user-facing agent skill (`2026-09-02-user-agent-skill.md`),
porting MATLAB's diagonal approximation of the log-joint variance and its
gradient (`compute_var == 2` in `_gp_log_joint`, which raises "not
implemented"; it would allow the ELCBO gradient with `beta ≠ 0`, which no
option enables today; the dead accumulators were deleted 2026-09-04),
a guard for `final_boost` (`2026-09-04-final-boost-failure.md`: a small,
independent algorithmic tweak; alters one trace of the 280 in the golden
baseline).
