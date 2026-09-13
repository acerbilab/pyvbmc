# S-VBMC preparation and entropy speedups: evidence

Machine-readable results of the paired before/after timing described in
[the plan](../../plans/svbmc-speedups.md); the reading of the numbers is in
[the report](../../results/2026-09-13-svbmc-speedups.md).

- `benchmark.json`: 18 pairs (three upstream ten-run K=50 groups, both
  weight modes, seeds 0-2) of `optimize(n_samples=20, n_samples_final=100,
  lr=0.1, max_steps=500)` calls, each pair run once from the source tree
  before the change (`dev-next` at `83692ac`) and once from the changed
  tree. The timed region is the `optimize()` call; loading the fixtures and
  constructing the object, which computes the Jacobian expectations, are
  timed separately as `load_and_construction_seconds`. Every cell records
  wall and CPU seconds, the number of entropy evaluations, the optimized
  weights, ELBO, entropy, ELBO standard deviation, the numeric ELBO details
  and a digest of the generator state after the fit; `comparison` holds
  the paired speedup and the agreement checks (weights within `1e-8`,
  identical evaluation counts and generator states), `summary` the
  per-configuration means, the aggregate, the paired-speedup range and
  median, `no_cell_slower` and `all_passed`. `sources` records both trees'
  commits, working-tree state of the package directory, imported module
  path, interpreter, library versions, thread settings and a digest of the
  fixture files each worker read; `harness` records the script's own
  digest. `status: complete` marks a finished campaign; the script exits 0
  only when `all_passed` and `no_cell_slower` both hold. The committed file
  is a verbatim copy of the raw output written under the ignored
  `dev/scripts/runs/svbmc_speedups_20260913/`.
- `benchmark_first.json`: the same 18 cells run 23 minutes earlier with
  an earlier revision of the harness, kept as a repeat measurement. Its
  outputs are identical to `benchmark.json` cell for cell; its summary
  lacks `no_cell_slower`, the construction times and the digests, and
  its `median_speedup` is the upper middle value rather than the median.

The controller keeps one worker process per tree, each importing its own
`pyvbmc` with one BLAS and one Torch thread and warmed by an unrecorded
two-step fit before any cell is timed; it alternates which tree runs first
in every cell and never runs the two at once. The before tree was a
detached worktree of `83692ac`. It lives under the main checkout's
`dev/scripts/runs/` because the measuring worktree's `dev/scripts/runs`
is a junction into it; the recorded `sources.before.path` shows that
location. Reproduce from a checkout of the changed tree with the before
tree checked out separately, then copy the output here:

```console
python dev/scripts/svbmc_speedup_benchmark.py --before <before-tree> --after . --output dev/scripts/runs/svbmc_speedups_<date>/benchmark.json
```

The script refuses to overwrite an existing output unless `--overwrite` is
passed. The recorded timings are one laptop's; the plan records the known
variation of such measurements and makes no cross-machine claim.
