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
