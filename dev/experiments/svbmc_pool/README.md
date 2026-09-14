# S-VBMC run-pool benchmark campaign

Tracked evidence from the campaign described in
[`dev/plans/svbmc-benchmark-campaign.md`](../../plans/svbmc-benchmark-campaign.md),
which generates pools of independent VBMC runs on noisy targets and
compares the S-VBMC implementation integrated into PyVBMC against the
original standalone `svbmc` 0.1.1. Only curated, machine-readable
artifacts live here — the pool manifests, selections and summaries, the
comparison results and the identity records; the raw per-run artifacts
stay gitignored under `dev/scripts/runs/`. A pool contributes its
`manifest.json` (the allocation, the base options, the authorization and
the identity of the code the pool was generated with), the
`selection.json` that `svbmc_pool_run.py select` writes — per condition
the lowest-seed runs that pass the filters, up to the filtered target,
which is the authoritative definition of the filtered pool and the list
the stacking comparison stacks — and the `summarize` output. Every
identity the harness records comes in two halves: a `source` half (the
PyVBMC and gpyreg commits and working-tree state, the hashes of the suite
and harness modules, the Python, NumPy and SciPy versions) that every
process of one campaign must agree on, and a `host` half (hostname,
platform, interpreter, import paths, installed versions, thread settings)
that is recorded and never compared, so that any machine may generate any
case of one pool. `baseline_environment.json` is the
identity of the comparison's baseline arm: the commit and per-file
SHA-256 hashes of the pinned upstream checkout, the Torch overlay it runs
against, and the interpreter, library versions, PyVBMC and gpyreg
checkouts and thread settings of the machine that ran it. The checkout
and the overlay are machine-local, so that record is what makes the
baseline verifiable here and recoverable elsewhere; the stacking harness
re-verifies it before every campaign. This file gains a description of
every key of every JSON and the exact reproduction commands once the
campaign's runs are in.

## Pilot (2026-09-14)

`pilot/pool/` holds the campaign manifest of
`dev/scripts/runs/svbmc_pool_20260913/pool/` (allocation, base options,
identity with the pinned gpyreg worktree, authorization) and the
`summarize` output after the pilot sweep of three seeds per condition:
`summary.json` and `summary.md` give per condition the seeds run, the
filtered count, pass rate, usable fraction, wall-time and evaluation
quartiles and the single-run metric quartiles. `pilot/stack/` holds the
`svbmc_pool_stack.py` outputs of the `M = 3` measurement on those
artifacts (five cell seeds per condition, 500 Adam steps, both arms).

`results.json` holds one entry per cell: the subset drawn (its indices,
run names and seeds), the cell seed and the `entry_seeds` both arms
rebuilt the subset's posteriors with, which arm ran first, the maximum
weight difference between the two, and per arm the optimized weights,
every ELBO variant that arm reports, its own entropy and standard
deviations, the construction, optimization and sampling seconds, and the
quality of 100 000 draws against the target's truth. Each arm also
carries the terms its estimates are scored against: `e_log_joint_mc` and
`e_log_joint_mc_sd`, the mean of the target's noiseless log density over
a subsample of that arm's own draws; `entropy_ref` and `entropy_ref_sd`,
the entropy of the stacked mixture at that arm's final weights, estimated
by the harness for both arms with one estimator rather than taken from
the arm; `elbo_mc` and `elbo_mc_sd`, their sum, the Monte Carlo ELBO of
the posterior that arm's fit produced; one `bias` entry per variant
(`bias_<variant> = elbo_<variant> − elbo_mc`); and `kl_gap = ln Z −
elbo_mc`, how far the stacked posterior itself is from the target. The
file also carries the `M = 1` rows, which are the pool's own single-run
metrics, and a `settings` block with the draw counts, the stacking call,
the grid and the bootstrap of that run, which is what its summary
describes it by. `summary.json` / `summary.md` give the medians,
bootstrap intervals, paired differences and equivalence tests, and
`sources.json` both arms' environments and the baseline re-verification.
The plan's Phase 4 acceptance and worklog read from these files; the raw
artifacts stay under the ignored `dev/scripts/runs/`.
