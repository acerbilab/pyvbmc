# S-VBMC preparation and entropy speedups

Created 2026-09-13. Status: in progress on `dev-svbmc-speedups` (worktree
`../pyvbmc-stage3`, branched from `dev-next` at `83692ac`).

This plan owns the implementation contract, checks, measurement and worklog
of the performance follow-up decided in
[svbmc-integration.md](svbmc-integration.md) (PI, 2026-09-13). The
[prototype report](../results/2026-09-09-svbmc-numpy-prototype.md) owns the
measurements that motivated it: with the same preparation improvements,
NumPy and Torch were close to parity, so the gain is in the preparation,
not the backend, and Torch stays.

## Objective and boundaries

Speed up `SVBMC._stacked_entropy`, the cost of every `maximize_ELBO` step
and of the two final evaluations in `optimize`, without changing the
method, the sampled objective, the draws or the random stream. Three
changes, all taken from the prototype's shared-preparation Torch control:

1. One inverse transform per run maps the draws of all its components to
   the original space, instead of one call per component.
2. Each run's forward transform and log-Jacobian are evaluated once for all
   draws; the diagonal-Gaussian log densities of the run's components are
   formed in one broadcast over a workspace bounded at 32 MiB, replacing
   one `scipy.stats.norm.logpdf` call per component.
3. The per-component means and variances of the mixture log density are
   row reductions of a `(K_total, n_samples)` view in Torch, instead of a
   Python loop over boolean masks.

Preserved: one `standard_normal((n_samples, D))` draw per component from
the object's generator, in run and component order; the component-major
row layout; the density formula; the Torch autograd path through
`logsumexp`; the public signatures and return values of `stacked_entropy`,
`stacked_ELBO`, `maximize_ELBO` and `optimize`; float64; no new attributes
on the object. Excluded: the ELBO debiasing investigation, the stacking
objective, a NumPy backend, GPU work, `sample`, and the benchmark campaign
against the original S-VBMC, which
[svbmc-integration.md](svbmc-integration.md#benchmark-campaign-required-for-15)
tracks.

## Numerical contract

- The regression references (`pyvbmc/testing/svbmc/fixtures/references.*`,
  `rtol=1e-8`) are the gate and are not regenerated.
- `pyvbmc/testing/svbmc/test_entropy.py` compares the new preparation and
  reduction with literal per-component transcriptions of the previous
  implementation on the fixtures (D=1; bounded probit D=2 with three
  transforms; warped and unwarped D=3; three upstream D=2 runs, one
  warped): density matrix, entropy, weight gradient, stratified variance
  and generator state, at 1 and 3 draws per component, at `1e-12`. The
  accepted differences are floating-point: the order of the Torch row
  reductions, and the matrix product of a warped run rotation in the
  batched inverse transform, which rounds differently for a different
  number of rows and so can move the draws of warped runs by an ulp.
- Chunking must be exact: a one-row workspace and the default give
  bit-identical matrices.

## Measurement

`dev/scripts/svbmc_speedup_benchmark.py` runs two isolated, warmed worker
processes, one per source: the frozen detached checkout of `83692ac` under
`dev/scripts/runs/svbmc_speedups_20260913/before` and this branch. Each
worker has one BLAS and one Torch thread; the controller alternates which
source runs first in every cell and never runs both at once. A cell is one
complete `optimize(n_samples=20, n_samples_final=100, lr=0.1,
max_steps=500, version=mode)` on `load_group(group, rng=0)[0]` with
`SVBMC(vps, seed=seed)`: the three upstream ten-run K=50 groups, both
weight modes, seeds 0-2, 18 pairs. Recorded: wall and CPU time, number of
entropy evaluations, weights, ELBO, entropy, generator state digest, and
provenance (commits, imported module paths, library versions).

Acceptance: identical evaluation counts and generator states in every
pair, weights within `1e-8`, no cell slower after the change, and a
repeatable speedup on the ten-run stacks (the prototype control suggests
about 2.5x for these preparation-dominated fits). Laptop timing variation
is known to be large; the report gives per-cell values and no cross-machine
claim. Evidence goes to `dev/experiments/svbmc_speedups/` and the report to
`dev/results/2026-09-13-svbmc-speedups.md`.

## Execution checklist

- [x] 1. Baseline: S-VBMC suite green on the unchanged code in the worktree
  environment (173 passed, 67 s; Python 3.12.6, NumPy 2.5.2, SciPy 1.18.1,
  Torch 2.7.0+cpu); before-checkout frozen.
- [x] 2. `pyvbmc/svbmc/_entropy.py` preparation and the vectorized
  reduction in `_stacked_entropy`; docstrings.
- [x] 3. Equivalence tests; references and the full S-VBMC suite green
  (184 passed, 36.7 s, against 173 in 66.9 s before). On the worktree
  machine a one-off `np.array_equal` comparison at 3 draws per component
  found the density matrix bit-identical to the literal reference on
  every fixture group; at 1 draw the warped runs differ by up to
  2.4e-13 relative (the rotation matrix product), within the `1e-12`
  gate of the shipped tests.
- [x] 4. Benchmark campaign, evidence and report: two campaigns of 18
  pairs, all bit-identical in weights, ELBO, entropy, evaluation counts
  and generator states; aggregate 2.15x and 2.06x (Gaussian mixtures
  2.0-2.4x, Ring 1.8-2.0x), no cell slower. Below the prototype control's
  2.5x because Ring spends relatively more time in the Torch reduction.
  Evidence in `dev/experiments/svbmc_speedups/`, report in
  [results/2026-09-13-svbmc-speedups.md](../results/2026-09-13-svbmc-speedups.md).
- [x] 5. Records: `FIXTURES.md` tests table, `dev/README.md`, `TODO.md`,
  the pointer in `svbmc-integration.md`, the roadmap and the 1.5
  overview.
- [~] 6. Independent doublecheck completed 2026-09-13 (three static
  reviews: code and tests, records, benchmark and evidence; findings
  applied, none blocking). Remaining: formatting hooks on commit, the CI
  cell with Torch, merge into `dev-next`.

## Decisions

- **Per-component draws are kept** rather than one batched draw: the
  stream is preserved by construction, the generator stub of
  `test_elbo_reporting.py::test_entropy_variance_matches_fixed_stratified_draws`
  (one `(n_samples, D)` block per component) stays valid, and the five
  hundred small calls of a default stack cost milliseconds.
- **Transform and Jacobian once per run over all rows; only the
  `(rows, K_m, D)` density workspace is chunked**, so chunking cannot
  change a value.
- **No cached component arrays on the object.** Rebuilding the `(K_m, D)`
  arrays per call takes microseconds and adds no serialized state.
- **A private module** (`_entropy.py`) holds the NumPy preparation so it
  can be tested against a literal reference without Torch; the Torch
  reduction stays in `_stacked_entropy`.

## Risks and rollback

The `(S, K_total)` density matrix is unchanged (40 MB for a default
500-component stack, 200 MB at the final 100 draws per component); the new
workspace adds at most 32 MiB. Rollback is reverting the branch; no fixture
or reference changes.

## Worklog

- 2026-09-13: worktree `../pyvbmc-stage3` moved from `main` to
  `dev-svbmc-speedups` off `dev-next`; baseline suite green; before-checkout
  frozen at `83692ac`. (`dev/scripts/runs` in that worktree is a Windows
  junction into the sibling main checkout `../pyvbmc`, so the
  before-checkout and the raw benchmark output physically live under
  `../pyvbmc/dev/scripts/runs/svbmc_speedups_20260913/`; the relative
  path is the same from either checkout.)
- 2026-09-13: implementation, equivalence tests and the S-VBMC suite
  (184 passed); paired campaign 16:28-16:34, all pairs identical,
  aggregate 2.15x; evidence and report written.
- 2026-09-13: three independent static reviews (code and tests; records;
  benchmark and evidence) found no blocking issue. Applied: equivalence
  tests at 1 draw per component and on the retained runs, the warped-run
  rounding recorded in the contract, a hardened harness (true median,
  worker shutdown, provenance with fixture and script digests,
  construction timed separately, `no_cell_slower`), roadmap and overview
  pointers. The campaign was rerun with the final harness 16:51-16:59:
  identical outputs again, aggregate 2.06x on a slower machine state;
  both campaigns are kept as evidence. Final S-VBMC suite on the
  finished code: 192 passed in 33.6 s.
