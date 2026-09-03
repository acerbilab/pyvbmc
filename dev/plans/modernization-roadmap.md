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
- [ ] **Stage 2 — NumPy vectorization + memory fix.** Order measured
  2026-09-02 (evening) on the benchmark suite, **provisional again since
  the 2026-09-03 audit** (the suite's start points and boxes were
  truth-anchored; regeneration pending, pickup point 0)
  (`plans/benchmark-suite-and-golden-traces.md` §Results, §Audit): (3) batched
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

0. **Regenerate the benchmark results** (evening of 2026-09-03, PI's
   machine): `bash dev/scripts/regenerate_baseline.sh`, one process, about
   10–12 h. The 2026-09-02/03 profile and golden population were withdrawn
   after an audit against the papers found truth-anchored start points and
   plausible boxes (`plans/benchmark-suite-and-golden-traces.md` §Audit);
   the Stage 2 order below is therefore **provisional again** until the
   regenerated profile confirms it.
1. **Stage 2 item 3**: batch the acquisition evaluation (`GP.predict` over
   `Ns`, `vp.pdf` over `K`, the CMA-ES population). Gate every step with
   `golden_trace.py run --suite golden --seeds 0-19 --workers 1 --out
   dev/scripts/runs/golden/<label>` followed by `compare
   dev/golden/baseline dev/scripts/runs/golden/<label>` (about 10 h per
   population on one process), plus the test suite.
2. ~~Run the `tests` workflow on `dev-next` for the package fix~~ done
   2026-09-03 (full matrix green, run 33715620257); pushes to `dev*` now
   run a smoke automatically.
3. Grow the golden population to 50 seeds and add D = 8/10 and the exhaust
   config (`plans/benchmark-suite-and-golden-traces.md` §Follow-ups). The
   reference sidecars live in git under `dev/golden/baseline/` (PI decision
   2026-09-03); copy them over after every extension.
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
