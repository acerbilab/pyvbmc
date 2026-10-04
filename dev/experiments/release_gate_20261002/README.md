# The release-gate campaigns on the cluster (2026-10-02)

The four campaigns of the PyVBMC 1.5 release gate (the brief:
[`dev/plans/release-gate-handoff.md`](../../plans/release-gate-handoff.md);
the procedure: [the operator's guide](../../scripts/hpc/README.md); the
design: [`dev/plans/slurm-benchmark-support.md`](../../plans/slurm-benchmark-support.md)),
run on 2026-10-02 on the University of Helsinki's Turso cluster, in the
operator's account, with the driver under `dev/scripts/hpc/`, in the
order the brief gives: the two reference-population arms together, the
S-VBMC run pools, the stacking comparison, then the two analyses of the
pools. Every campaign verified completely: no case failed, none went
missing, none was given up.

This directory holds the tracked copies of the four campaigns, redacted
by `campaign_redact.sh` as the guide's "The tracked copies, the archive
and the hand-back" describes (hosts named by the node family or `login`,
paths by the directory that holds them, the manifests without their
`site` block); each `redaction.json` records the SHA-256 of every copy
and of the campaign file it was made from. The raw campaign directories
are the archives of the draft releases listed under "Archives"; the
machine that holds them lists them in its gitignored
`dev/scripts/runs/LOCAL.md`. All times below are the cluster's (EEST).

## What ran, and from what

- **The release commit**, `ff3ed01463b3c2a031dd2225c9f487dc369d2397` on
  `dev-next`, in a clone checked out detached and clean, which did not
  move until the last analysis had finished; every task compared its
  source identity with the one its campaign's `prepare` recorded.
- **The source trees**, each a clean clone at its commit:

  | Tree | Commit | Used by |
  |---|---|---|
  | the harness checkout | `ff3ed01463b3c2a031dd2225c9f487dc369d2397` | every campaign |
  | gpyreg v1.4.0 | `682585f72d893824aa1c22951f49d2c52cc6942e` | the after arm, the pools, the stacking, the analyses |
  | gpyreg v1.2.1 | `9e70e6ba53f7607d05c2d9cc2fa9f41cd12b8f3b` | the before arm |
  | PyVBMC at `f91fdf0`, a detached worktree of the harness checkout | `f91fdf00894653f63a0be6b06cceb59ee373c283` | the before arm (`PYVBMC_SOURCE`) |
  | S-VBMC 0.1.1 | `13a78f6` | the stacking's original arm (`BASELINE_DIR`) |

- **The environment**, one conda environment built with
  `campaign_env.sh build` from the release commit's frozen
  `campaign_requirements.txt`: Python 3.12.14, numpy 2.5.2, scipy 1.18.1,
  cma 4.4.4, torch 2.14.0+cpu and the other 31 pins, with zstd 1.5.7,
  gh 2.102.0 and git 2.56.0 from conda-forge; `check-env` passed at the
  build and at every submission and finish. PyVBMC and gpyreg were
  imported from the source trees.
- **The environment check**, one batch job on the campaigns' node family
  (job 2282744): 18 min 57 s, 1.57 GB peak; the seven test modules passed
  139, 75, 60, 55, 76, 14 and 54 tests, none skipped; the interpreter was
  the environment's and the checkout clean afterwards.
- **The jobs.** Every job requested the node family's feature (`u8`,
  AMD EPYC 7452 nodes) with `--hint=nomultithread` and one
  CPU, so that each task had a physical core to itself (`verify` checks
  the affinity of every record); no partition was named. From 14:32 on,
  two nodes of the family that had failed their prolog twice each were
  excluded by name through `SBATCH_EXTRA` (see "Node failures and the
  throttle"); the Slurm job records of each campaign (`slurm/jobs.txt`
  in its archive) hold the setting of every submission.
- **The limits** were the brief's, unchanged; no task reached one. The
  throttle was 200 tasks a submission for the first population
  submissions and 75 afterwards.

## The order of events

| Time | Event |
|---|---|
| 12:20 | Both population arms prepared (the after arm paired to the before arm); canaries submitted, 24 tasks each (jobs 2289025, 2289049) |
| 12:44 to 12:48 | The canary looks: before arm `verify` 2290460 and `summarize` 2290461, after arm `verify` 2290722, `summarize` 2290723 and `rescore` 2290724; 24 cases verified in each arm, none failed, 48 rescored |
| 12:50 | The rest of both arms submitted at 200 a submission: before arm 2290911 (`cigar_D15_exhaust`) and 2291011 (the rest), after arm 2291012 and 2291013 |
| 12:50 to 12:56 | Four nodes of the family drained with a prolog error; the four arrays held at 12:55 |
| 13:53 | The held pending tasks cancelled (166 of the before arm, 188 of the after arm) |
| 13:54 to 14:15 | The same four submissions again at 75 a submission, a few minutes apart: 2295243, 2295345, 2295445, 2295730 |
| 14:32 | Two more nodes having drained, the pending tasks cancelled again (240 and 105) and the four submissions made once more with those two nodes excluded: 2296333, 2296409, 2296436, 2296539 |
| 16:33 | The last population task ended |
| 16:34 | Before arm's finish: `verify` 2301225, `summarize` 2301234, archive written 16:39. Pools prepared; canary: case 1 (2301226) and the first case of the other seven conditions (2301227) |
| 16:45 | After arm's finish: `verify` 2301239, `summarize` 2301247, `rescore` 2301254 (ended 17:53), archive written 17:56. Pools' look: `verify` 2301238, `select` 2301240, `summarize` 2301241; 8 cases verified |
| 16:49 to 16:56 | The rest of the pools in four interleaved chunks at 75 a submission: 2301255, 2301331, 2301409, 2301497 |
| 17:42 | The last pool task ended; finish: `verify` 2304232, `select` 2304234, `summarize` 2304235, archive written 17:46 |
| 17:59 | Stacking prepared; canary: task 1 (2304238) and the first task at `M = 32` (2304239, task 16) |
| 18:12 | Stacking look: `verify` 2304240 (2 tasks verified), `assemble` 2304241 exited 1, the expected end of a canary |
| 18:14 to 18:20 | The stacking subsets, a minute apart: `M2` 2304244, `M3` 2304252, `M4` 2304261, `M5` 2304269, `M8` 2304328, `M16` 2304336, `M32` 2304397 |
| 18:44 | The last stacking task ended; finish: `verify` 2304765, `assemble` 2304773, archive written 18:46 |
| 18:48 | The analyses submitted: `svbmc_shrink_elbo.py` 2304811, then `svbmc_single_run_bias.py` 2304812 after it |

## Node failures and the throttle

Four minutes after the first bulk submissions, the compute nodes began
failing their Slurm prolog as hundreds of tasks launched on them at once
(four submissions at 200 concurrent tasks, which Slurm packed a full
node at a time). Slurm drains a node whose prolog fails and requeues its
tasks: four nodes of the family were drained within six minutes, two
more later in the afternoon, and the administrators resumed them within
the hour. The arrays were held at 12:55 while the running tasks finished,
the pending tasks were cancelled and the same four submissions made
again at 75 concurrent tasks each, a few minutes apart, and once more at
14:32 with the two nodes that had failed twice excluded by name. The
submission script records each submission's throttle and `sbatch`
arguments in `slurm/jobs.txt`. At that rate the prolog of a node
receiving fifty-odd tasks took two to three minutes and then succeeded;
no node drained after 14:10. The pools and the stacking were submitted
at 75 a submission from the start, the pools' 2922 remaining cases as
four interleaved chunks (the case indices congruent to 0, 1, 2 and 3
modulo 4) so that four submissions ran together.

Nothing of the campaigns was damaged. A task whose node drained before
its worker started left nothing; eight tasks had started their run
(five in the before arm, three in the after arm), and each stopped on
SIGTERM, removed its claim and the partial files, and left its case to
the task that ran it later; four tasks of job 2295730 are accounted as
`NODE_FAIL` with exit code 1 and were requeued by Slurm. Tasks that Slurm
requeued in a held state were released by hand. The pending tasks
cancelled at 13:53 and 14:32 (406 of the before arm, 293 of the after
arm) never ran and hold no record. The accounting of every job, the
cancelled and requeued ones included, is `slurm/sacct.txt` in each
archive.

## Reference population, before arm (`population_before/`)

The 24 configurations of the `production` suite at seeds 0 to 99, 2400
cases, run with the package at `f91fdf0` and gpyreg v1.2.1 through the
release commit's harness. `verification.json`: 2400 verified, 0 verify
failed, 0 failed, 0 interrupted, 0 partial, 0 missing, 0 stray. No error
file was written at any point.

Accounting (`slurm/sacct.txt` of the archive; tasks that ran a case, the
ones that exited at once on finding a record excluded): elapsed median
2.5 min, quartiles 1.3 and 5.2 min, maximum 34.1 min
(`cigar_D15_exhaust`); about 180 CPU-hours; peak resident set 539 MB
(`cigar_D15_exhaust`), against limits of 1 h (1.5 h for
`cigar_D15_exhaust`) and 2 GB. The tasks ran on nine nodes of the
family. The finish's `verify` took 2 min 1 s and 1.17 GB, `summarize`
5 s.

The tracked copies are the manifest, the verification report, the
summary (`summary.md`) and, per configuration, every case's completion
record, sidecar and boost report; `rescored/` is not among them, since
the before arm does not rescore (the after arm's `rescored/` holds the
before arm's cases rescored with the release code).

## Reference population, after arm (`population_after/`)

The same 2400 cases with the release code and gpyreg v1.4.0, prepared
with `--pair` naming the before arm. `verification.json`: 2400 verified,
0 verify failed, 0 failed, 0 interrupted, 0 partial, 0 missing, 0 stray;
no error file.

Accounting as above: elapsed median 2.5 min, quartiles 1.3 and 5.5 min,
maximum 30.9 min (`cigar_D15_exhaust`); about 175 CPU-hours; peak
resident set 392 MB (`student_D8`); eleven nodes of the family. The
finish's `verify` took 2 min 35 s and 1.56 GB, `summarize` 5 s, and
`rescore` 1 h 4 min and 500 MB for the 4800 cases of both arms.

`rescore` reproduced the in-run metrics of every after-arm case exactly
(`elbo_err`, `gskl`, `mmtv` and `rmse` equal for 2400 of 2400 cases,
none non-finite). Scoring the before arm's 2400 cases with the release
code reproduces their in-run `elbo_err` for every case, `gskl` for 1699,
`rmse` for 1700 and `mmtv` for none, with no non-finite value: the port
review changed those scorings, which is what the rescored metrics in
`rescored/population_before.json` are for, and `analyze_population_run.py
--arms` reads them by default. The tracked copies are the before arm's
plus `rescored/population_after.json`, `rescored/population_before.json`
and `rescoring.json`.

## S-VBMC run pools (`pools/`)

The 8 conditions of the `svbmc_pool` suite, seeds from 1000, 350 a
condition and 480 for the ring, 2930 cases, with gpyreg v1.4.0;
`prepare` was given `--target 320 --max-seeds 350 --allocation
ring_D2_noise3_svbmc=320/480`. `verification.json`: 2930 verified, 0
failed, 0 missing, 0 partial, 0 stray; no error file. `select` reached
its target of 320 in every condition with no shortfall and nothing
unfinished, 2560 selected runs in all. From `summary.json` and
`selection.json`:

| Condition | Runs | Pass the filters | Usable fraction | Selected / target | Seeds scanned | Wall min, median [IQR] |
|---|---|---|---|---|---|---|
| `multisensory_s1_D6_noise3_svbmc` | 350 | 350 (1.00) | 0.04 | 320 / 320 | 320 | 7.1 [6.6, 7.5] |
| `multisensory_s1_D6_noise1.3_svbmc` | 350 | 349 (1.00) | 0.09 | 320 / 320 | 321 | 4.8 [4.4, 5.3] |
| `rosenbrock_D2_noise3_svbmc` | 350 | 349 (1.00) | 0.74 | 320 / 320 | 321 | 3.4 [3.0, 3.9] |
| `gmm_D2_noise3_svbmc` | 350 | 342 (0.98) | 0.00 | 320 / 320 | 328 | 2.9 [2.6, 3.3] |
| `ring_D2_noise3_svbmc` | 480 | 381 (0.79) | 0.00 | 320 / 320 | 405 | 4.9 [3.3, 5.7] |
| `student_D8_noise3_svbmc` | 350 | 349 (1.00) | 0.55 | 320 / 320 | 321 | 9.7 [9.2, 10.1] |
| `gmm_D2_svbmc` | 350 | 342 (0.98) | 0.12 | 320 / 320 | 326 | 0.5 [0.5, 0.7] |
| `multisensory_s1_D6_svbmc` | 350 | 341 (0.97) | 0.91 | 320 / 320 | 329 | 2.0 [1.7, 2.2] |

The usable fraction is the share of filtered runs under the house
thresholds for a single run. Accounting: elapsed
median 3.9 min, quartiles 2.5 and 6.2 min, maximum 11.8 min
(`student_D8_noise3_svbmc`); about 220 CPU-hours; peak resident set
349 MB, against 45 min and 2 GB; six nodes of the family. The finish's
`verify` took 1 min 59 s and 469 MB, `select` and `summarize` 3 s each.
The tracked copies are the manifest, the verification report, the
selection (`selection.json`, `selection.md`) and the summary
(`summary.json`, `summary.md`).

## Stacking comparison (`stacking/`)

Both arms (the integrated S-VBMC and the original package) at `M` = 2,
4, 8 and 16 and the integrated arm alone at 3, 5 and 32, 20 repetitions
below `M = 16` and 10 from it, on disjoint subsets of each condition's
320 selected runs: 960 cells in 200 tasks, a task holding every
repetition of one condition and `M` below 16 and one cell from 16 on.
`prepare` reported no condition short of repetitions.
`verification.json`: 200 verified, 0 failed, 0 missing, 0 partial, 0
stray; `assemble` merged 960 cells of 200 tasks into `results.json`,
`summary.json`, `summary.md` and `sources.json`, the tracked copies with
the manifest and the verification report. Per subset, from the
accounting:

| Subset | Tasks | Limits | Longest task | Peak resident set |
|---|---|---|---|---|
| `M2` | 8 | 1 h, 4 GB | 2 min 56 s | 1.09 GB |
| `M3` | 8 | 1 h, 4 GB | 1 min 27 s | 0.75 GB |
| `M4` | 8 | 1 h, 4 GB | 7 min 2 s | 1.09 GB |
| `M5` | 8 | 1 h, 4 GB | 2 min 45 s | 0.76 GB |
| `M8` | 8 | 1 h, 4 GB | 25 min 47 s | 1.34 GB |
| `M16` | 80 | 30 min, 4 GB | 8 min 48 s | 1.72 GB |
| `M32` | 80 | 1 h, 6 GB | 10 min 13 s | 2.82 GB |

The two longest tasks were the `M = 8` tasks of the two noisy
multisensory conditions; the 200 tasks took about 17 CPU-hours in all.
The finish's `verify` took 8 s and 209 MB, `assemble` 24 s and 508 MB;
the tasks ran on three nodes of the family.

## The two analyses of the pools

Two batch jobs from the harness checkout, the second waiting for the
first, with their outputs under one directory (`analyses/`) that is an
asset of the stacking's draft release, since its `sources.json`,
`summary.json` and `added.json` name paths and nodes that nothing
redacts.

- `svbmc_shrink_elbo.py` (job 2304811): 2 min 30 s and 637 MB;
  `960/960 cells`; `shrink/cells.jsonl`, `summary.json`, `summary.md`
  and `sources.json`.
- `svbmc_single_run_bias.py` (job 2304812, after the first): 46 min 54 s
  and 859 MB; `2560/2560 runs`; `single_run/runs.jsonl`, `cells.jsonl`,
  `summary.json`, `summary.md`, `added.json`, `added.md` and
  `sources.json`.

Both jobs exited 0. The directory went back whole as `analyses.tar.zst`
of the stacking's draft release.

## Files

Every campaign directory here holds `manifest.json` (without its `site`
block), `verification.json` and `redaction.json`, and what its harness
declares as tracked copies: for the population arms the summary and
every verified case's completion record, sidecar and boost report under
its configuration's directory, and for the after arm `rescored/` and
`rescoring.json`; for the pools the selection and the summary; for the
stacking `results.json`, the summaries and `sources.json`. The redaction
names the hosts that ran a job by the node family and any other host by
`login`, keeps the CPU model and `Linux x86_64` of each host part, and
writes a path under a named directory as that name (`$HARNESS_TREE/...`,
`$PYVBMC_GPYREG_SOURCE/...`, `$CAMPAIGN_PARENT/...`); task logs, the
accounting, claims and error files are not copied, and job ids stay.
`redaction.json` records, for each copy, the SHA-256 of the copy and of
the campaign's file, the archive parts with theirs, and `cases_not_verified`,
empty in every campaign; no `--path` name and no `--allow` exemption was
needed. `analyze_population_run.py --arms` reads the two population
directories as it reads the campaigns, and `golden_replay.py --sidecars`
reads the after arm's.

## Archives

Each campaign's whole directory, written by its finish as one
zstd-compressed part with its SHA-256 file, is an asset of a draft
release of `acerbilab/pyvbmc` of its own, the analyses' directory one
more asset of the stacking's; the public assets of the after
arm and of the pools, built by `campaign_public.sh` and checked with its
`--check`, are assets of `release-gate-public-20261002`. The archives
hold the site's details and the operator's paths, so these releases stay
drafts. `cat <name>.tar.zst.[0-9][0-9][0-9] | zstd -d | tar x` restores a
campaign directory.

| Release | Asset | Bytes | SHA-256 |
|---|---|---|---|
| `release-gate-population-before-20261002` | `population_before.tar.zst.000` | 414410851 | `41ab07227cd1f0f80c05bdfcd994ecb293df839fec37849c4139f12499ffea22` |
| `release-gate-population-after-20261002` | `population_after.tar.zst.000` | 825596327 | `cc66070d21456245dd2701d750f5045b0f42db8540018df682015bc218409ee0` |
| `release-gate-pools-20261002` | `pools.tar.zst.000` | 236170919 | `6a50da2b22c593ba8138ce072ff81b90f5e1e167b3ad690b1ca4d1f2c7ff1cc3` |
| `release-gate-stacking-20261002` | `stacking.tar.zst.000` | 15843438 | `968cdcf5a97e76e1c7c035cda82466fbff0f163626e61d3ecfdb9ada70784f54` |
| `release-gate-stacking-20261002` | `analyses.tar.zst` | 2997025 | `1eb9505c80a9a8b7f6999b4d7261980c24916c9b4e7adcdb380514957a96d5a8` |
| `release-gate-public-20261002` | `population_after.public.tar.gz.000` | 656433975 | `e7a7437d2813b608b976fbdc7cebfb2c917353793cc4bcd4baabab2aad544395` |
| `release-gate-public-20261002` | `pools.public.tar.gz.000` | 259769576 | `d36e0be85510866a6dabd9840d1987cbed0bc45a7531a452a62fe33adb685e05` |

The after arm's public asset holds 12008 files: the 4800 numeric files
(the trace and the posterior arrays of each of the 2400 verified cases),
the 7206 tracked copies, their `redaction.json`, and `public.json`,
which records the code that built the asset; the 2400 boost pickles are
left out. The pools' holds 8798 files: the 5860 run files (each run's
`.npz` and its `.json`, the latter redacted as the copies are), the 2930
completion records, the 6 tracked copies, their `redaction.json` and
`public.json`.
