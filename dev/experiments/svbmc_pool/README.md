# S-VBMC run-pool benchmark campaign

Tracked evidence from the campaign described in
[`dev/plans/svbmc-benchmark-campaign.md`](../../plans/svbmc-benchmark-campaign.md),
which generates pools of independent VBMC runs on noisy targets and
compares the S-VBMC implementation integrated into PyVBMC against the
original standalone `svbmc` 0.1.1. Only curated, machine-readable
artifacts live here — the pool manifests and summaries, the comparison
results and the identity records; the raw per-run artifacts stay
gitignored under `dev/scripts/runs/`. `baseline_environment.json` is the
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
artifacts (five cell seeds per condition, 500 Adam steps, both arms):
`results.json` with every cell (subset, seeds, per arm the weights, ELBO
variants, entropy, seconds, draw metrics and the Monte Carlo expected
log joint), `summary.json` / `summary.md` with the medians, bootstrap
intervals, paired differences and equivalence tests, and `sources.json`
with both arms' environments and the baseline re-verification. The
plan's Phase 4 acceptance and worklog read from these files; the raw
artifacts stay under the ignored `dev/scripts/runs/`.
