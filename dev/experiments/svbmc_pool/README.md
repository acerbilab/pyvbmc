# S-VBMC run-pool benchmark campaign

Tracked evidence from the campaign described in
[`dev/plans/svbmc-benchmark-campaign.md`](../../plans/svbmc-benchmark-campaign.md),
which generates pools of independent VBMC runs on noisy targets and
compares the S-VBMC implementation integrated into PyVBMC against the
original standalone `svbmc` 0.1.1. Only curated, machine-readable
artifacts live here — the pool manifests, selections and summaries, the
comparison results and the identity records; the raw per-run artifacts
stay gitignored under `dev/scripts/runs/`. A pool contributes its
`manifest.json` (the allocation, the base options and the identity of the
code the pool was generated with), the
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
re-verifies it before every campaign. Each generated pool has its own
subdirectory with a README that describes every key of its JSON files
and the exact commands that generated it; this file indexes them and
owns the description of the comparison's outputs once the campaign's
stacking runs are in.

## Pilot (2026-09-14)

`pilot/pool/` holds the campaign manifest of
`dev/scripts/runs/svbmc_pool_20260913/pool/` (allocation, base options,
identity with the pinned gpyreg worktree) and the
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

## Phase 2 prototype (2026-09-14)

`pilot/phase2/` holds the outputs of `svbmc_honest_elbo.py` on the 25
cells of `pilot/stack/`: `results.json` (per cell the raw, capped and
honest expected log joints under every coverage rule and combination
with their biases against the cell's `e_log_joint_mc`, the Monte Carlo
and GP-side uncertainties, the covered weight, per-run own and cross-run
errors, calibration `z` statistics, effective training-point counts, the
decomposition of every run's heaviest component into what each run's
data and GP said about it, and the own-run checks; per run the self-check
row; the skipped cells; the settings), `cells.jsonl` (the same cell rows
as written during the sweep), `summary.json` and `summary.md` (medians
with bootstrap intervals per condition and rule, the decomposition
tables), `sources.json` (commits, import paths, versions, thread
settings, the hashes of the script and of the cells file) and `figures/`
(bias by condition, own against honest error per component, coverage and
bias under every rule, calibration, errors against effective training
points). `pilot/phase2_newconds_selfcheck/` is the script's
`--self-check` on the artifacts of the two later conditions under gpyreg
1.2.1. The per-cell component arrays stay with the raw artifacts. The
report is
[results/2026-09-14-svbmc-honest-elbo-pilot.md](../../results/2026-09-14-svbmc-honest-elbo-pilot.md).

## Cluster pool (2026-09-14)

`pool_20260914/` holds the tracked side of the eight-condition pool
generated on the University of Helsinki `kale` cluster with the Slurm
scripts under `dev/scripts/hpc/`: `manifest.json`, `selection.json` /
`selection.md` (700 selected runs, every condition at its filtered
target), `summary.json` / `summary.md` (1100 completed runs, 1035 passing
the filters) and `verification.json` (every artifact re-verified after
the array, none failed, partial, missing or stray). Its
[README](pool_20260914/README.md) describes every key of those files, the
commands and environment that generated the pool, its resource fit, and
the archive of the raw directory (`svbmc_pool_20260914.tar.zst`, an asset
of the draft release `svbmc-pool-20260914`, with its size and SHA-256).

## Stacking comparison, stage D (2026-09-14)

`stack_20260914/` holds the tracked side of the two-arm comparison on
that pool, run by `dev/scripts/svbmc_pool_stack.py` from `dev-next`
`744864a` on the analysis machine (560 cells: the decision-6 grid,
`M ∈ {2, 4, 8, 16}` with 20, 20, 20, 10 repetitions, both
implementations on every cell, all eight conditions from the pool's
`selection.json`, 8.1 hours). `summary.json` is the harness's summary:
`settings` (the stacking call, the draw counts, the entropy reference,
the bootstrap), `equivalence_tests` (one exact signed-rank test per
condition, `M` and metric, Holm-corrected per metric) and
`conditions`, each with its `single_run` medians (`M = 1`, from the
pool's records), an `all_M` aggregate of the runtime ratio and
`max_abs_dw`, `headline_bias_growth` (criterion 3's second clause) and
one entry per `M` holding both arms' medians with bootstrap intervals
(`arms`), the paired differences (`paired`), `criterion3` (the gate on
the headline bias), `max_abs_dw`, `runtime_ratio` and `tests`.
`summary.md` renders the same tables; the columns are defined in the
harness's module docstring and the campaign plan's "Stacking
comparison", "Metrics" and "Acceptance criteria" sections.
`sources.json` records both arms' environments (commits, import paths,
Torch and gpyreg, the checkout the pool's manifest names against the
`--gpyreg-source` used), the baseline re-verification, the pool's
identity and the harness's SHA-256. The comparison's `results.json`
(17.6 MB, every cell with its subset, seeds, weights, ELBO variants,
metrics and reference terms) and `cells.jsonl` are in the analyses
asset described under "Raw outputs of the analyses" below;
`results.json` is what `--summarize-only` and the Phase 2 estimator
read.

## Phase 2 scoring of stage D (2026-09-15)

`phase2_20260915/` holds the tracked side of `svbmc_honest_elbo.py` on
the 560 cells of stage D against the cluster pool (`--pool
dev/scripts/runs/svbmc_pool_20260914 --cells <stage D results.json>
--gpyreg-source <the 1.2.1 worktree>`, defaults otherwise: 100 draws per
component, seed 0, coverage ratios 1.5, 2, 3 and 5 with the headline at
2, launched from `dev-next` `cd59443`, 2.0 hours; its `sources.json`
records `d6a4fe6`, the head when the run ended, which differs from
`cd59443` in documentation only): `summary.json` / `summary.md`
(the headline table across conditions, then per condition and `M` the
biases of every estimate, the coverage, the calibration of the
self-reported SDs, the runs' own errors and the decomposition of the
heaviest components), `sources.json` and `figures/`. The pilot's
[report](../../results/2026-09-14-svbmc-honest-elbo-pilot.md) defines
every quantity; the reading of this run is in the
[stage D report](../../results/2026-09-15-svbmc-pool-comparison.md).
The 27.6 MB `results.json`, the 17.4 MB `cells.jsonl` and the per-cell
component arrays (`cells/`, 95 MB) are in the analyses asset described
under "Raw outputs of the analyses" below.

## Weight-aware caps (2026-09-15)

`cap_kappa_20260915/` is the one run of `svbmc_cap_kappa.py` on stage
D's cells: `cells.jsonl` (per cell, the stack's weighted expected log
joint `G`, entropy `H` and `elbo_mc`, and for every cap the level, the
number of components in its set, the bias and whether it binds),
`summary.json` and `summary.md` (per condition and `M`, the median bias
and median absolute bias over cells of the raw value, the class's cap,
each `kappa` in 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99 and 1, the
weighted median, and the run-level caps `E_max` (the largest run-level
expected log joint) and `E_top` (that of the run carrying the largest
stacking mass), next to the class's run-median cap, with the fraction
of cells each cap binds on and the count of cells on which a cap at the
largest component would bind, zero by construction) and
`sources.json` (the script's and the cells file's hashes, the pool,
the process's identity).

## Empirical-Bayes shrinkage (2026-09-15)

`shrink_20260915/` is the one run of `svbmc_shrink_elbo.py` on stage D's
cells: `cells.jsonl` (per cell, `G`, `H`, `elbo_mc`, the raw and
class-cap biases, per variant the shrunken `G`, its bias and the
weight-averaged shrinkage factor, and per run the population mean, the
excess variance `tau2`, the spread and the mean estimation variance),
`summary.json` and `summary.md` (per condition and `M`, the median bias
and median absolute bias over cells of the raw value, the class's cap
and the variants `within`, `within_full`, `stack`, `run_level`,
`two_level`, `two_level_full`, the anchored forms `anchored`,
`two_level_anchored`, `anchored_mean` and `two_level_anchored_mean`
(what the within-run shrinkage removes at each run's own weights
added back, mass-weighted or averaged over runs, so that a stack of
one run reports its own value; each cell's record carries the
per-run amount as `anchor`) and `hybrid`, with the shrinkage factor and the
noise shares, the mean estimation variance over the spread of the
estimates, within a run, over the stack's components and over the runs'
levels; each cell's record also carries every run's level, its
estimation variance and its shift), and `sources.json`. The noise in
a spread of correlated estimates is `(tr Σ − 1ᵀ Σ 1 / K) / (K − 1)`,
which the script uses for `tau2` and the shares. The directory (and
`shrink_M35_20260915/`) holds the rerun of 2026-09-15 that added the
anchored forms; the values of the earlier variants are unchanged, and
its `sources.json` hashes the script as run, before the formatting hook
reflowed it without changing its logic (the tracked cells reproduce
from the current script to 1e-13).

## The integrated arm at `M = 3` and `5` (2026-09-15)

`stack_M35_20260915/` holds the tracked side of `svbmc_pool_stack.py
--arms integrated --M 3,5 --repetitions 20,20` on the same pool, seed
and settings as stage D (320 cells, 15 minutes, from `dev-next`
`e95e85c`): `summary.json`, `summary.md` and `sources.json` in the
layout of stage D's, every cell carrying the integrated arm alone (no
paired quantity, no equivalence test). `shrink_M35_20260915/` and
`cap_kappa_M35_20260915/` are the shrinkage and cap scorings of its
cells, in the layouts described above. `stack_merged_20260915/` is the
summary of stage D and this run together, `svbmc_pool_stack.py
--summarize-only --from-results <stage D> --from-results <this run>`:
one table per condition over `M` = 2, 3, 4, 5, 8 and 16, the paired
quantities and equivalence tests on the cell sets both arms ran, and
the headline-bias growth across every `M`. The per-cell files of this
run (`results.json`, `cells.jsonl`) are the second asset of the
analyses release described below.

## The inputs' own bias (2026-09-15)

`single_run_20260915/` holds the outputs of
`svbmc_single_run_bias.py --pool <pool> --gpyreg-source <gpyreg 1.2.1>
--cells <stage D results.json> --cells <M = 3 and 5 results.json>
--shrink shrink_20260915/cells.jsonl --shrink
shrink_M35_20260915/cells.jsonl` (700 runs in 10.2 minutes, from
`dev-next` `0ca7dcd` with the script uncommitted; the join was then
regenerated with `--reuse` after the shrinkage rerun that added the
anchored forms and again after the doublecheck of 2026-09-15, and
`sources.json` describes that last pass, hashing the script, the runs
file and every input file): `runs.jsonl`, one
record per filtered run of the pool scored as a stack of one
(`condition`, `name`, `seed`, `K`, `noisy`; `elbo_vbmc`, the ELBO the
run reports; `elbo_raw`, the class's raw value for the run alone at
its own weights, with its `G` and `H`; `elbo_cap`, the
component-median cap at those weights; `e_log_joint_mc`,
`entropy_ref`, `elbo_mc` with `_sd` standard errors, computed as the
comparison computes a cell's reference; `bias_vbmc`, `bias_raw`,
`bias_cap` against `elbo_mc`; `kl_gap` where `ln Z` is known; the
run's own `elbo_sd`; `metrics` as `sample_metrics` returns them;
`seconds`); `summary.json` / `summary.md` (per condition, the median
with a bootstrap interval, the mean and the quartiles of each bias and
of the KL gap); `cells.jsonl`, one record per cell of the joined
comparisons (`inputs`: the runs' names, biases and stack masses, the
mean bias, the bias of the input with the highest reported ELBO and
the mass-weighted bias; `bias`: the cell's raw and capped biases from
the comparison and every shrinkage variant's from the shrinkage cells;
`added`: each minus the inputs' mean bias); `added.json` / `added.md`
(per condition and `M`, medians over cells of the inputs' biases, of
every estimate's bias and added bias, bootstrap intervals on the
added bias of `raw`, `run_level`, `two_level_full`,
`two_level_anchored` and `two_level_anchored_mean`, and the fraction
of cells whose raw added bias is positive); and
`sources.json`. The raw directory,
`dev/scripts/runs/svbmc_pool_20260914_single_run/`, holds the same
files and the controller's log. The runs' own scores are those of the
first pass; `cells.jsonl`, `added.*`, `summary.*` and `sources.json`
are the last pass's.

## Re-optimizing on the shrunken estimates (2026-09-15)

`shrink_opt_20260915/` holds the outputs of
`svbmc_shrink_optimize.py --pool <pool> --gpyreg-source <gpyreg
1.2.1> --cells <stage D results.json> --cells <M = 3 and 5
results.json> --shrink shrink_20260915/cells.jsonl --shrink
shrink_M35_20260915/cells.jsonl --single-runs
single_run_20260915/runs.jsonl --M 3,4,5` (480 cells in 23 minutes,
from `dev-next` `48f46f0` with the doublecheck's fixes to the script
uncommitted; a first run of 26 minutes that warm-started the
optimizer from the runs' unshrunk reported ELBOs gave the same
medians within 0.02 and is superseded): `cells.jsonl`, one record per
cell (`w`, the weights optimized on the two-level full shrinkage of
the corrected expected log joints, from a warm start shifted by
`level_shift`, the change of each run's own level under the
shrinkage; `G_shrunk` and
`G_raw_at_opt`, the shrunken and the raw expected log joint at those
weights; `H`; `elbo_mc` with its standard error, computed as the
comparison computes a cell's reference; `kl_gap` and
`kl_gap_recorded`; `bias` for `shrunk_opt`, `raw_at_opt`, `raw` (from
the cell's record) and `two_level_full` (from the shrinkage cells);
`metrics` with `gskl` and `mmtv` of the new stack and `_recorded` of
the raw optimization; `max_abs_dw` and `mass_shift`, the largest
weight change and half the L1 distance between the two weight
vectors; `inputs_mean_bias` and `added`; the timings), `summary.json`
/ `summary.md` (per condition and `M`, the medians over cells of the
biases and added biases with bootstrap intervals, the metrics of both
optimizations with the fraction of cells the new weights improve, the
KL gaps and the weight changes) and `sources.json`. The raw directory
`dev/scripts/runs/svbmc_pool_20260914_shrink_opt/` holds the same
files and the controller's log.

## Raw outputs of the analyses (release asset)

What stage D and the Phase 2 scoring wrote per cell is too large to
track and is, with the first run of the weight-aware caps,
`svbmc_pool_20260914_analyses.tar.gz` (113 585 523 bytes, SHA-256 `59ada698d573f40e733b598b4cc667b571b552d3da261b7168f2c18edd06aa3b`), the asset of the draft release `svbmc-analyses-20260915` of `acerbilab/pyvbmc` (target commit `24646b6`), as the pool
itself is the asset of `svbmc-pool-20260914`. It holds the three raw
directories in full, `svbmc_pool_20260914_stack/` (stage D:
`results.json`, `cells.jsonl`, the summaries, `sources.json`, the
controller's and the original arm's logs), `svbmc_pool_20260914_phase2/`
(the Phase 2 scoring: `results.json`, `cells.jsonl`, `cells/`, the
summaries, `figures/`, `sources.json`, the controller's log) and
`svbmc_pool_20260914_cap_kappa/` (the weight-aware caps), and unpacks
with `tar -xzf` to those directories, which belong under
`dev/scripts/runs/` on the machine that reads them (`gh release download
svbmc-analyses-20260915 --pattern svbmc_pool_20260914_analyses.tar.gz`).
`svbmc_pool_stack.py --summarize-only --from-results
<stack results.json>` rebuilds the stage D summaries from it, and the
Phase 2 estimator and `svbmc_cap_kappa.py` take that file as `--cells`.
The second asset of the same release,
`svbmc_pool_20260914_stack_M35.tar.gz` (1 849 978 bytes, SHA-256
`349e73f9e578782250bb06be3848b0e84da9d174f39991d75d9e3db6cafd34c3`),
is the raw directory of the `M = 3` and `5` run,
`svbmc_pool_20260914_stack_M35/` (`results.json`, `cells.jsonl`, the
summaries, `sources.json`, the controller's log), and unpacks the same
way; its `results.json` is the second `--from-results` file of the
merged summary and the `--cells` file of `shrink_M35_20260915/` and
`cap_kappa_M35_20260915/`.

## The numbers the documents quote

`python dev/scripts/svbmc_headline_numbers.py` prints, from the four
tracked `cells.jsonl` files of the shrinkage and cap scorings, every
number the [headline note](../../2026-09-15-svbmc-headline-shrinkage.md)
and the shrinkage and cap sections of the
[stage D report](../../results/2026-09-15-svbmc-pool-comparison.md)
quote: the per-condition tables, the worst and mean cases over the
noisy conditions per `M`, the noise-share percentiles, the hybrid
threshold sweep, the caps' bind fractions, the paired bootstrap
intervals, from `single_run_20260915/` the single-run biases and the
added-bias ranges, and from `shrink_opt_20260915/` the
re-optimization's rows. Check a sentence of those documents against
its output rather than against the raw directories. The tutorial's
worked example is printed by `dev/scripts/svbmc_shrink_worked_example.py`
from the pool and the stage D `results.json`.
