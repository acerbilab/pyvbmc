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
  - [x] stage-level oracles (`pyvbmc/testing/oracles/`, 14 oracles × 8
    snapshots, about 20 s; per-element tolerances with a robust floor,
    set from measured rounding floors; the per-commit gate for Stage 2).
    Not built from the devlog's Stage 0 list: oracles for the GP log
    marginal likelihood and its gradient (gpyreg-side), `GP.quad`,
    `kl_div_mvn`, `kde_1d`. Not yet exercised on another BLAS build: the
    first CI run after the commit is.
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
  (`plans/stage1-rng-generator.md`). Remaining seam: gpyreg still draws
  from the global state; see the follow-ups there.
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
  slice sampler 27–38 %): item 8.
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
   seed has a visible false-alarm rate on cigar. **Next: item 8 as one
   gpyreg PR** (sampler overhead plus `predict`'s per-sample loop and `sW`
   tiling from item 3), then item 5 (`entmc_vbmc`), then the memory items.
   The 20-seed population against `dev/golden/baseline` (the end-of-stage
   check of items 3, 1 and 2 together, about 7 h) was launched 2026-09-05
   01:05 on the final code (`a39e5ec`), output under
   `dev/scripts/runs/golden/item1_20260905/`; the same job writes
   `summary_item1.md` and `compare_vs_baseline.md` next to the traces
   when it finishes (the harness prints to stdout; the job redirects).
   **Read `compare_vs_baseline.md` first when picking up**; if the job
   died, rerun `python dev/scripts/golden_trace.py compare
   dev/golden/baseline dev/scripts/runs/golden/item1_20260905` (the `run`
   subcommand skips finished tasks, so it can be resumed with the same
   `--out`).
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
7. gpyreg generator support (`GP.fit`, `SliceSampler`, `f_min_fill`,
   `GP.random_function`) on a gpyreg branch when convenient. The PyVBMC seam
   can only go once gpyreg `main` has it: CI installs gpyreg from `main`,
   unpinned.
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
