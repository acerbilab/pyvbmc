# Modernization roadmap (living plan)

Status tracker for the staged plan in
`dev/2026-09-02-modernization-discussion.md` §10. That devlog holds the
rationale and the decisions; this file holds only what is done, what is next,
and where to pick up. Update it in place; do not add status to the devlogs.

Branch: `dev-next` (PRs to `main` per stage, see devlog §11).

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
- [ ] **Stage 2 — NumPy vectorization + memory fix.** Provisional order
  (measured on two easy Gaussians only): batched acquisition evaluation,
  gpyreg sampler overhead (gpyreg PR), `_gp_log_joint` einsum,
  `_eval_full_elcbo` multi-RHS solve, `vp.pdf` over `K`, `entmc_vbmc`,
  drop per-candidate deepcopy, `GP.clean()` / stop retaining full GPs.
- [ ] **Stage 3 — pipeline features** (batched initial design,
  torch/jax target adapter docs, `vp.to_torch()`, ArviZ export).
- [ ] **Stage 4 — PyTorch port** (decision point, not default).

## Pickup point

1. Open the Stage 1 PR from `dev-next` to `main` (one-liners + generator
   threading + plans/devlog), after the full suite is green locally.
2. gpyreg PR: accept a generator (`GP.fit`, `SliceSampler`, `f_min_fill`,
   `GP.random_function`), then remove the PyVBMC seam.
3. Stage 0 remaining items, starting with the benchmark target suite and the
   golden-trace harness, then the remaining finite-difference checks.
4. Profile the benchmark suite; confirm or revise the Stage 2 order.

## Deferred (devlog §12)

Per-component `lambd`, gradient-based acquisition optimization, batched
acquisition, multi-chain slice sampling, scaling to `N ≈ 2k–5k`, log-space
mixture sums, user-facing agent skill (`2026-09-02-user-agent-skill.md`).
