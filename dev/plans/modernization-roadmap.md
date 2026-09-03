# Modernization roadmap (living plan)

Status tracker for the staged plan in
`dev/2026-09-02-modernization-discussion.md` §10. That devlog holds the
rationale and the decisions; this file holds only what is done, what is next,
and where to pick up. Update it in place; do not add status to the devlogs.

Branch: `dev-next`. No PR to `main` until the work here is done: one PR at
the end (decided 2026-09-02, superseding the per-stage PRs of devlog §11).
CI on `dev-next` is the manually dispatched `tests` workflow; run it after
each substantive commit.

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
    (`dev/scripts/golden_trace.py`; baseline population of 20 seeds × 11
    configs on the run path of `d76cdb6` (sidecars record
    `0056016`/`16369e5`), gitignored under `dev/scripts/runs/golden/
    baseline/`, null check clean; expand to 50 seeds and higher D;
    `plans/benchmark-suite-and-golden-traces.md` §Results)
  - [ ] dtype canary
- [x] **Stage 1 — `seed=` and `Generator` threading**
  (`plans/stage1-rng-generator.md`). Remaining seam: gpyreg still draws
  from the global state; see the follow-ups there.
- [x] **Benchmark target suite** (`dev/scripts/benchmark_targets.py`,
  2026-09-02/03): banana, cigar, lumpy, Student-t at D ≤ 6, a noisy VIQR
  target, a logistic regression, a budget-exhausting run; profiled
  (`plans/benchmark-suite-and-golden-traces.md`). Still to add: D = 8 and
  10, the exhaust config in the golden set.
- [ ] **Stage 2 — NumPy vectorization + memory fix.** Order confirmed
  2026-09-02 (evening) on the benchmark suite
  (`plans/benchmark-suite-and-golden-traces.md` §Results): (3) batched
  acquisition evaluation (`GP.predict` over `Ns`, `vp.pdf` over `K`, the
  CMA-ES population; 40–48 % of time is single-point `predict`), (8) gpyreg
  sampler overhead (`solve_triangular` wrappers 9–10 %, `__core_computation`;
  gpyreg PR), (1) `_gp_log_joint` einsum (close to (8); PyVBMC-local, may
  land first), (2) `_eval_full_elcbo` multi-RHS solve (also shrinks
  `final_boost`, 6–12 %), then `entmc_vbmc`, drop per-candidate deepcopy,
  `GP.clean()` / stop retaining full GPs (memory only: deepcopy is 0.4 % of
  time). On noisy targets (8) and (1) dominate.
- [ ] **Stage 3 — pipeline features** (batched initial design,
  torch/jax target adapter docs, `vp.to_torch()`, ArviZ export).
- [ ] **Stage 4 — PyTorch port** (decision point, not default).

## Pickup point

1. **Stage 2 item 3**: batch the acquisition evaluation (`GP.predict` over
   `Ns`, `vp.pdf` over `K`, the CMA-ES population; 40–48 % of run time is
   single-point `predict`). Gate every step with
   `golden_trace.py run --suite golden --seeds 0-19 --workers 1 --out
   dev/scripts/runs/golden/<label>` followed by `compare
   dev/scripts/runs/golden/baseline dev/scripts/runs/golden/<label>`
   (about 5 h per population on one process), plus the test suite.
2. Run the `tests` workflow on `dev-next` for the package fix `6f3f0ba`
   (`variational_optimization.py`, Ns = 1 variance shape) if not yet done.
3. Grow the golden population to 50 seeds and add D = 8/10 and the exhaust
   config (`plans/benchmark-suite-and-golden-traces.md` §Follow-ups);
   decide whether the reference sidecars (0.55 MB of JSON) should live in
   git so `compare` works from a fresh checkout.
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
mixture sums, user-facing agent skill (`2026-09-02-user-agent-skill.md`).
