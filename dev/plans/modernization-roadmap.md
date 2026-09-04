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
    kernel/mean/noise derivatives, `compute_vargrad`
  - [ ] fixture generator script (regenerable `.npz`/JSON; retire `.mat`)
  - [ ] stage-level oracles with pre-drawn randomness
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
- [ ] **Stage 2 — NumPy vectorization + memory fix.** Order measured
  2026-09-02 on the first suite and **confirmed 2026-09-03/04 on the
  regenerated suite** (papers' procedure; the shares below are the
  regenerated ones, `plans/benchmark-suite-and-golden-traces.md` §Results
  (regenerated)): (3) batched acquisition evaluation (`GP.predict` over
  `Ns`, `vp.pdf` over `K`, the CMA-ES population; active sampling 54–69 %
  of wall on every noiseless target, single-point `predict` 40–50 % of
  profiled time at D ≤ 10, `vp.pdf` 15 % at D = 15 with K ≈ 25), (8) gpyreg
  sampler overhead (GP training 13–20 % at D = 4, 22–28 % at D = 10, and
  41 % of an iteration in the late sampling regime at D = 15; the Cholesky
  is a small part of it; gpyreg PR), (1) `_gp_log_joint` einsum (variational
  stage 9–20 % at D = 4, largest on ridged posteriors and 32 % of an
  optimize-only iteration; PyVBMC-local, may land before (8) for
  logistics), (2) `_eval_full_elcbo` multi-RHS solve (also shrinks
  `final_boost`, 3–12 % of a short run), then `entmc_vbmc` (11 % at D = 15),
  drop per-candidate deepcopy, `GP.clean()` / stop retaining full GPs
  (memory only: deepcopy is < 1 % of time). On noisy targets the
  active-sampling bucket is the per-sample GP refits and VP re-optimizations,
  so (8) and (1) are what speed those up.
- [ ] **Stage 3 — pipeline features** (batched initial design,
  torch/jax target adapter docs, `vp.to_torch()`, ArviZ export).
- [ ] **Stage 4 — PyTorch port** (decision point, not default).

## Pickup point

0. ~~Regenerate the benchmark results~~ done 2026-09-03/04 in two sessions
   (profile campaign plain + cProfile, golden population 20 × 14, null check
   clean, sidecars published; `plans/benchmark-suite-and-golden-traces.md`
   §Results (regenerated)). The Stage 2 order above is confirmed; committed
   and pushed as `9206738`. The one failed run of the population,
   `student_D4` seed 19, is a final-boost failure, written up in
   `2026-09-04-final-boost-failure.md`; the guard is deferred (see
   Deferred below).
1. **Stage 2 item 3**: batch the acquisition evaluation (`GP.predict` over
   `Ns`, `vp.pdf` over `K`, the CMA-ES population). Gate every step with
   `golden_trace.py run --suite golden --seeds 0-19 --workers 1 --out
   dev/scripts/runs/golden/<label>` followed by `compare
   dev/golden/baseline dev/scripts/runs/golden/<label>` (about 10 h per
   population on one process), plus the test suite.
2. ~~Run the `tests` workflow on `dev-next` for the package fix~~ done
   2026-09-03 (full matrix green, run 33715620257); pushes to `dev*` now
   run a smoke automatically.
3. Grow the golden population to 50 seeds (`golden_trace.py run --seeds
   20-49`, about 14 h on one process) and add cigar/Student-t at D = 6–8
   and the exhaust config (`plans/benchmark-suite-and-golden-traces.md`
   §Follow-ups). The reference sidecars live in git under
   `dev/golden/baseline/` (PI decision 2026-09-03); copy them over after
   every extension.
4. Stage 0 remaining items: fixture generator, finite-difference checks for
   the transformer Jacobian, gpyreg derivatives and `compute_vargrad`.
5. gpyreg generator support (`GP.fit`, `SliceSampler`, `f_min_fill`,
   `GP.random_function`) on a gpyreg branch when convenient. The PyVBMC seam
   can only go once gpyreg `main` has it: CI installs gpyreg from `main`,
   unpinned.
6. One PR `dev-next` → `main` when the work is done.

## Deferred (devlog §12)

Per-component `lambd`, gradient-based acquisition optimization, batched
acquisition, multi-chain slice sampling, scaling to `N ≈ 2k–5k`, log-space
mixture sums, user-facing agent skill (`2026-09-02-user-agent-skill.md`),
a guard for `final_boost` (`2026-09-04-final-boost-failure.md`: a small,
independent algorithmic tweak; alters one trace of the 280 in the golden
baseline).
