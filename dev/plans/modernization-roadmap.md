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
  - [ ] golden-trace harness over the benchmark target suite (below)
  - [ ] dtype canary
- [x] **Stage 1 — `seed=` and `Generator` threading**
  (`plans/stage1-rng-generator.md`). Remaining seam: gpyreg still draws
  from the global state; see the follow-ups there.
- [ ] **Benchmark target suite** (devlog §10, "Benchmark target suite"):
  shared module under `dev/scripts/` used by `profile_run.py` and the
  golden-trace harness: banana, cigar, lumpy, Student-t at several `D`; a
  noisy VIQR-path target; one real likelihood with data; one
  budget-exhausting run reaching `N ≥ 200 + 10D`. Profile it before Stage 2.
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

1. Benchmark target suite as a shared module under `dev/scripts/`, wired
   into `profile_run.py`; profile it and confirm or revise the Stage 2 order.
2. Stage 0 remaining items: golden-trace harness over the suite, fixture
   generator, remaining finite-difference checks.
3. Stage 2 in the measured order.
4. gpyreg generator support (`GP.fit`, `SliceSampler`, `f_min_fill`,
   `GP.random_function`) on a gpyreg branch when convenient. The PyVBMC seam
   can only go once gpyreg `main` has it: CI installs gpyreg from `main`,
   unpinned.
5. One PR `dev-next` → `main` when the work is done.

## Deferred (devlog §12)

Per-component `lambd`, gradient-based acquisition optimization, batched
acquisition, multi-chain slice sampling, scaling to `N ≈ 2k–5k`, log-space
mixture sums, user-facing agent skill (`2026-09-02-user-agent-skill.md`).
