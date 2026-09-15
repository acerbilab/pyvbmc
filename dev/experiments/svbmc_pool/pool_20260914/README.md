# S-VBMC run pool generated on the cluster (2026-09-14)

The eight-condition run pool of the S-VBMC benchmark campaign
([`dev/plans/svbmc-benchmark-campaign.md`](../../../plans/svbmc-benchmark-campaign.md),
section "Cluster generation"; the brief was
[`dev/plans/svbmc-pool-handoff.md`](../../../plans/svbmc-pool-handoff.md)),
generated as a Slurm array on the University of Helsinki `kale` cluster
with the scripts under [`dev/scripts/hpc/`](../../../scripts/hpc/README.md).
This directory holds the tracked copies of the campaign's summary files;
the raw directory (1100 per-run artifacts with their completion records,
the task logs and the Slurm accounting) is the archive described under
"Archive" below, gitignored under `dev/scripts/runs/svbmc_pool_20260914/`
on the machine that holds it, which lists it in its `LOCAL.md`.

## How it was generated

- Code: commit `d63c477` of the branch `dev-svbmc-pool-hpc` (a descendant
  of `dev-next` at `613f2a8`), clean tree; gpyreg `v1.2.1`
  (`9e70e6ba53f7607d05c2d9cc2fa9f41cd12b8f3b`) from a detached clone at
  the harness default path, imported through `PYVBMC_GPYREG_SOURCE`. The
  manifest's `identity.source` half is what every worker matched.
- Environment: conda env `pyvbmc-pool` with Python 3.12.14, numpy 2.5.3,
  scipy 1.18.1 (OpenBLAS, single-threaded through the thread variables),
  PyVBMC installed editable from the checkout, GPyReg 1.2.1 from PyPI as
  the installed distribution (shadowed at run time by the pinned
  checkout), psutil 7.2.2. `manifest.identity.host` records the login
  node; each completion record records the compute node that ran it.
- Allocation: the approved defaults of `prepare` (100 filtered runs per
  noisy condition, 50 per noiseless control; seed caps 150, 200 for the
  ring, 75 for the controls; seeds from 1000), 1100 cases.
- Commands, on the login node `turso02` from the checkout root
  (`$POOL` = `dev/scripts/runs/svbmc_pool_20260914`, `$GPYREG` the pinned
  checkout):
  ```
  ARRAY=1 dev/scripts/hpc/svbmc_pool_submit.sh $POOL --gpyreg-source $GPYREG   # prepare + canary (job 75347909)
  ARRAY=2-1100 THROTTLE=200 dev/scripts/hpc/svbmc_pool_submit.sh $POOL         # job 75347910
  dev/scripts/hpc/svbmc_pool_finish.sh $POOL                                   # verify (job 75349018), select, summarize, archive
  ```
  No case failed and none went missing, so no reconciliation or
  resubmission was needed; the canary ran first (job 75347909, 6 min
  37 s), the other 1099 tasks as job 75347910 between 17:46 and 18:19.
  Every task ran with `--cpus-per-task=1 --mem=2G --time=00:30:00` on the
  `short` partition, no node constraint (the partition mixes AMD
  carrington/ukko3 and Intel kale/ukko2 nodes; runs on different node
  families are not bitwise reproducible against each other, which the
  campaign plan accepts for a pool). The Slurm scripts were revised
  after this run, in the review of the pull request that delivered it
  (`ARRAY` mapped onto the `MaxArraySize` chunks, a task exiting at once
  when its case has a completion record, one queue query per recorded
  job, the guards of the finish script), so the committed scripts are
  not the exact ones that ran these commands; the commands and the
  worker each task ran are the same.

## Outcome

All 1100 cases completed and every artifact passes the post-hoc checks of
`verify_run` (the stats keys, the recomputation gate, the float64 canary
and the hashes of its completion record; the comparisons against the live
run were made when each artifact was saved); 1035 runs pass the filters
and every condition reaches its filtered target, 700 selected runs in all
with no shortfall. Per condition, from `summary.json`,
`selection.json` and the Slurm accounting (task elapsed time, which
includes the interpreter start-up and the artifact's verification on
saving, so it exceeds the run's own wall in `summary.md` by a few
seconds; memory is the batch step's `MaxRSS`):

| condition | completed | pass the filters | selected / target | seeds scanned | elapsed min (median [IQR], max) | MaxRSS MB (median, max) |
|---|---|---|---|---|---|---|
| `multisensory_s1_D6_noise3_svbmc` | 150 | 149 (0.99) | 100 / 100 | 100 | 7.1 [6.5, 7.6], 8.9 | 231, 274 |
| `multisensory_s1_D6_noise1.3_svbmc` | 150 | 148 (0.99) | 100 / 100 | 102 | 5.0 [4.6, 5.7], 8.4 | 213, 259 |
| `rosenbrock_D2_noise3_svbmc` | 150 | 147 (0.98) | 100 / 100 | 103 | 3.5 [3.2, 4.0], 9.0 | 202, 263 |
| `gmm_D2_noise3_svbmc` | 150 | 145 (0.97) | 100 / 100 | 103 | 3.0 [2.5, 3.4], 5.3 | 195, 254 |
| `ring_D2_noise3_svbmc` | 200 | 146 (0.73) | 100 / 100 | 142 | 5.1 [3.8, 5.7], 7.3 | 210, 262 |
| `student_D8_noise3_svbmc` | 150 | 150 (1.00) | 100 / 100 | 100 | 9.3 [8.9, 9.8], 11.5 | 273, 302 |
| `gmm_D2_svbmc` | 75 | 75 (1.00) | 50 / 50 | 50 | 0.7 [0.6, 0.9], 1.6 | 0, 142 |
| `multisensory_s1_D6_svbmc` | 75 | 75 (1.00) | 50 / 50 | 50 | 2.4 [2.2, 2.8], 4.2 | 149, 210 |

The ring is the one condition where the filters bite (48 of its 200 runs
unstable, 6 more above `s_max`), which is why its seed cap was
over-provisioned; every other condition loses at most five runs. The
single-run quality columns of `summary.md` (`elbo_err`, `gskl`, `mmtv`
and the usable fraction under the house thresholds) are poor by
construction on the multimodal targets, as in the pilot: one run covers a
piece of the ring or some of the GMM's clusters, which is the regime
stacking is for.

Verification (`verification.json`, one `srun` task of 34 s on
`carrington-810`, job 75349018): 1100 verified, 0 failed, 0 partial,
0 missing, 0 stray. The recomputation gate reproduces the stored `I_sk`
and `J_sjk` bit for bit on 1094 of the 1100 artifacts; the six that
deviate do so by at most 3.0e-9 (`I_sk`) and 1.5e-9 (`J_sjk`) against
the tolerance of 1e-8, and they are the three tasks that ran on the Intel
`kale` nodes plus three of the 838 that ran on `ukko3`, so the AMD
`carrington` nodes (259 tasks, the verifying node among them) and almost
every `ukko3` node reproduce each other exactly. Each completion record
names its node in `identity.host`.

Resource fit (`slurm/sacct.txt` in the archive): 1100 tasks, all
`COMPLETED`, all `AllocCPUS=1`; elapsed median 4.6 min, quartiles 3.1
and 6.6 min, maximum 11.5 min (Student D8); `MaxRSS` median 210 MB,
maximum 302 MB (the noiseless GMM control's tasks are shorter than the
accounting sampling interval, hence its zero median); 90.5 CPU-hours in
33 minutes of wall time at 200 concurrent tasks. The requested
`--time=00:30:00 --mem=2G` had a margin of about 2.6× on time and 6.8× on
memory. Cluster wall times are about twice the laptop pilot's per run
(the canary took 6.5 min against the pilot's 3.3 min median on the same
condition), so the plan's 45 CPU-hour estimate became 90 on these cores.

## Files

In every file `campaign` is `svbmc_pool` and `generated` the time it was
written; `directory` is the bare directory name in `summary.json` and
the cluster's absolute path in `selection.json` and `verification.json`.

- `manifest.json` — written by `prepare`: `campaign` (`svbmc_pool`),
  `suite`, `options` (the base VBMC options every run used), `allocation`
  (per condition `label`, `seed_start`, `max_seeds`, `target_filtered`),
  `gpyreg_source` (the pinned checkout's absolute path on the cluster),
  `identity` (`source`: `python`, `pyvbmc_commit`, `pyvbmc_dirty`,
  `gpyreg_commit`, `gpyreg_clean`, `suite_module_dirty`,
  `suite_module_sha256`, `io_module_sha256`, `runner_sha256`, `numpy`,
  `scipy`; `host`: `hostname`, `platform`, `executable`,
  `pyvbmc_import`, `pyvbmc_version`, `gpyreg_import`, `gpyreg_version`,
  `threads`, `gpyreg_source`), `allow_dirty` (false), `created`,
  `allocation_history` (empty: the allocation was never revised).
- `selection.json` / `selection.md` — written by `select`, the
  authoritative filtered pool the stacking comparison reads: `campaign`,
  `directory`, `generated`, `target_override` (null), `identity` (the
  manifest's), `totals` (`selected`, `shortfall`) and per condition in
  `conditions`: `label`, `seed_start`, `seed_cap`, `target_filtered`,
  `selected`, `shortfall`, `seeds_scanned` (the prefix of seeds walked
  until the target was met), `failed_while_scanning`,
  `pass_rate_scanned` (selected over scanned, failures included, which
  is not the condition's pass rate), `last_seed` (the last selected
  seed), and `runs` (the selected `tag`, `seed` pairs in seed order).
- `summary.json` / `summary.md` — written by `summarize` over every case
  the directory holds: `campaign`, `directory`, `generated`,
  `pilot_seeds` (null), `identity`, `options`, `totals` (`seeds_run`,
  `filtered`, `failed`) and per condition: `label`, `seed_start`,
  `seed_cap`, `target_filtered`, `seeds_run`, `completed`, `failed`,
  `failures` (`tag`, `reason`), `filtered`, `pass_rate` (filtered over
  completed), `unstable`, `above_s_max`, `usable_fraction` (filtered runs
  under the house thresholds `elbo_err < 1`, `gskl < 1`, `mmtv < 0.2`),
  and quartile blocks (`n`, `median`, `q1`, `q3`) for `wall_minutes`,
  `func_count`, `K`, `max_J_sjk`, `elbo_err`, `gskl`, `mmtv`, `rmse`
  (`wall_minutes`, `func_count` and `max_J_sjk` over every completed run;
  `K` and the four metric blocks over the filtered runs, which is why
  their `n` is the `filtered` count).
- `verification.json` — written by `verify` after the array: `campaign`,
  `directory`, `generated`, `gpyreg_source` (the checkout used),
  `identity` (the manifest's), `counts` (`verified`, `failed`, `partial`,
  `missing`, `verify_failed`, `stray`), `conditions` (per condition the
  counts `verified`, `failed`, `partial` and `missing`; this file
  predates the per-condition `verify_failed` count), `stray` (tags of
  artifact files the allocation does not name) and `cases`, one entry
  per allocated case with its 1-based `index` (the line of `cases.txt`,
  the case index that `svbmc_pool_submit.sh` takes in `ARRAY`; a task's
  Slurm array index is this minus its chunk's offset), `tag`,
  `label`, `seed`, `status` and, for a verified case, `passes` (the
  filter verdict) and `differences` (the maximum absolute deviation of the
  recomputed `I_sk` and `J_sjk` from the stored statistics, the
  recomputation gate); a `failed` case carries the last line of its error
  file as `reason`.

## Archive

The whole campaign directory (369 MB unpacked: the 1100 `.npz` + `.json`
artifacts, `records/`, `cases.txt`, `slurm/` with every task's output,
`jobs.txt`, `sacct.txt` and the finish driver's log, `verification.json`,
`selection.*`, `summary.*`, `manifest.json`) is the asset
`svbmc_pool_20260914.tar.zst` of the draft release `svbmc-pool-20260914`
of `acerbilab/pyvbmc` (release target commit `d63c477`):

| asset | bytes | SHA-256 |
|---|---|---|
| `svbmc_pool_20260914.tar.zst` | 88740881 | `f2863d967a809e946f9b473b10ec4065ea79b4e466f631001535f51df3107bb1` |

It was written with `tar -C dev/scripts/runs -cf - svbmc_pool_20260914 | zstd -T2 -3`
and unpacks to `svbmc_pool_20260914/`, to be placed under `dev/scripts/runs/`
on the analysis machine and listed in that machine's gitignored
`dev/scripts/runs/LOCAL.md`. Every artifact's SHA-256 is in its
completion record under `records/`, so
`python dev/scripts/svbmc_pool_run.py verify --out dev/scripts/runs/svbmc_pool_20260914 --gpyreg-source <local gpyreg 1.2.1 checkout>`
re-checks the copy before anything reads it. Two things to know on
another machine: the manifest stores the cluster's absolute
`gpyreg_source`, which `verify`, `svbmc_pool_stack.py --pool` and
`svbmc_honest_elbo.py` each replace with `--gpyreg-source`, a local
clean checkout at the manifest's gpyreg commit (nothing hashes the
field; only a repeated `prepare` compares it); and that machine's BLAS
does not recompute the stored statistics bit for bit. The rounding it
introduces is amplified by the condition number of each run's GP,
which reaches 3e17 on the noisy Rosenbrock condition (training sets
with far-tail evaluations), so `verify` there holds every artifact to
the absolute gate plus its own amplified rounding and reports the
relative differences and condition numbers; the campaign plan's worklog
records what the analysis machine found.
