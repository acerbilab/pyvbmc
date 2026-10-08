# The release gate's second batch on the cluster (2026-10-05)

The second batch of the PyVBMC 1.5 release gate (the brief:
[`dev/plans/release-gate-handoff-2.md`](../../plans/release-gate-handoff-2.md);
the procedure: [the operator's guide](../../scripts/hpc/README.md), its
sections "Arm 0: PyVBMC 1.0.4", "Arm 3: the port's end of warm-up" and
"The second batch's hand-back"; the designs:
[Arm 0's plan](../../plans/arm-1.0.4-comparison.md) and
[Arm 3's plan](../../plans/arm-3-warmup-comparison.md)), run on
2026-10-05 and 2026-10-06 on the University of Helsinki's Turso cluster,
in the operator's account, with the first batch's settings, environment
and node family, in the order the brief gives, but for one overlap: Arm
0's population, the slowest, was still running when Arm 3's finish, its
pools and its stacking ran, and its finish came after them; every step
that reads another's output ran after it (`fresh_release`'s finish
before `fresh_arm3`'s, `rescore-arms` after the finishes of Arm 0 and
Arm 3, the analyses after the stacking's `assemble`). Six campaigns: Arm 0
(`population_v104`), Arm 3's population (`population_arm3`), the fresh
seeds with the release code (`fresh_release`) and with Arm 3
(`fresh_arm3`), Arm 3's pools (`pools_arm3`) and its stacking
(`stacking_arm3`). Every campaign verified; two cases of Arm 0 failed in
PyVBMC 1.0.4 itself and are recorded as failed (see "The failed cases");
none went missing.

This directory holds the tracked copies of the six campaigns, redacted
as the first batch's were
([`release_gate_20261002/README.md`](../release_gate_20261002/README.md)),
with a `redaction.json` beside each. The raw campaign directories are the
archives of the draft releases listed under "Archives"; the machine that
holds them lists them in its gitignored `dev/scripts/runs/LOCAL.md`. All
times below are the cluster's (EEST).

## What ran, and from what

- **The two harness commits.** Arm 0's, `b196402b` on `dev-next`, to which
  the harness checkout moved from the first batch's `ff3ed014`; it ran
  `population_v104`, `fresh_release` and `rescore-arms`. Arm 3's,
  `ad63dd5a`, the head of `dev-arm3-port-warmup`, as a detached worktree of
  the harness checkout; it ran `population_arm3`, `pools_arm3`,
  `stacking_arm3`, `fresh_arm3` and the analyses, with its own package.
  The two commits differ in four files of `pyvbmc/` (the end of warm-up,
  an option's default and two tests) and hold the same files that build
  and score a run and the same frozen requirements as the after arm's
  commit, which `prepare --pair` and `prepare --compare` checked before
  any case ran. Both checkouts stayed at their commits, clean, throughout.
- **The source trees**, each clean at its commit:

  | Tree | Commit | Used by |
  |---|---|---|
  | the harness checkout | `b196402b` | Arm 0, `fresh_release`, `rescore-arms` |
  | `pyvbmc-arm3`, a detached worktree of it | `ad63dd5a` | Arm 3's five campaigns and the analyses |
  | `pyvbmc-v1.0.4`, a detached worktree of it | `0bb8b8f5` (tag v1.0.4) | Arm 0 (`PYVBMC_SOURCE`) |
  | gpyreg v1.0.4 | `3b1300a` | Arm 0 |
  | gpyreg v1.4.0 | `682585f72d893824aa1c22951f49d2c52cc6942e` | Arm 3, the fresh seeds, `rescore-arms`, the analyses |
  | S-VBMC 0.1.1 | `13a78f6` | the stacking's original arm |

- **The environment**: the first batch's, at the same path and unchanged;
  `check-env` passed at every submission and finish. The after arm's
  directory from the first batch was in place and was read, never
  finished again.
- **The environment checks**, one batch job in each checkout on the
  campaigns' node family, both with the 1.0.4 trees exported: job 2349473
  in the harness checkout (25 min 37 s, 1.27 GB) and job 2349474 in the
  Arm 3 checkout (25 min 14 s, 1.56 GB). Each ran the seven test modules
  with 139, 75, 81, 59, 76, 14 and 54 tests passed, none skipped, the
  environment's interpreter, and left its checkout clean. Nothing was
  submitted before both had passed.
- **The jobs.** Every job requested the node family's feature (`u8`) with
  `--hint=nomultithread` and one CPU; no partition was named; the two
  nodes of the family excluded by name in the first batch stayed
  excluded. Every submission ran at a throttle of 75 tasks, submissions a
  few minutes apart, and the tasks running at once were kept near 300 at
  most.
- **The limits** were the brief's, Arm 0's own among them; no task reached
  one. The longest task of the batch was an Arm 0 run of
  `cigar_D15_exhaust` at 3 h 0 min, under its 5 h.

## The order of events

| Time | Event |
|---|---|
| 18:22 | The trees set up; the two environment checks submitted (they ran 18:38 to 19:04) |
| 19:04 to 19:05 | The four campaigns prepared (Arm 0 with `--compare`, Arm 3 and `fresh_arm3` with `--pair`) and their canaries submitted: Arm 0 2355574 and Arm 3 2355575 (24 tasks each), `fresh_release` 2355597 and `fresh_arm3` 2355601 (1 each) |
| 19:10 to 19:13 | The fresh looks: `fresh_release` `verify` 2355678, `summarize` 2355683, `rescore` 2355689; `fresh_arm3` `verify` 2355694, `summarize` 2355695, `rescore` 2355700; 1 case verified in each |
| 19:20 to 19:21 | The rest of the fresh campaigns: 2355744, 2355787 |
| 19:30 to 19:35 | The fresh finishes: `fresh_release` `verify` 2355979, `summarize` 2355983, `rescore` 2355989, archive written 19:32; `fresh_arm3` `verify` 2355997, `summarize` 2356001, `rescore` 2356005, archive written 19:35 |
| 19:30 to 20:08 | Arm 3's look: `verify` 2355977, `summarize` 2355986, `rescore` 2355991 (36 min: Arm 3's 24 cases and the after arm's 2400) |
| 20:08 to 20:11 | The rest of Arm 3: 2358082 (`cigar_D15_exhaust`), 2358191 |
| 20:12 to 20:45 | Arm 0's look: `verify` 2358218 (33 min in the queue before it ran), `summarize` 2359116; 24 verified |
| 20:46 to 20:48 | The rest of Arm 0: 2359174 (`cigar_D15_exhaust`), 2359257 |
| 22:30 | Arm 3's last population task ended; its finish: `verify` 2362462, `summarize` 2362661, `rescore` 2362673 (36 min), archive written 23:11. The pools prepared; canary: case 1 (2362468) and the first case of the other seven conditions (2362469) |
| 22:40 to 22:42 | The pools' look: `verify` 2362758, `select` 2362776, `summarize` 2362781; 8 verified |
| 22:42 to 23:01 | The rest of the pools in four interleaved chunks (the case indices congruent to 0, 1, 2 and 3 modulo 4), the fourth once the running count allowed: 2362793, 2362861, 2362944, 2363623 |
| 00:37 | The last pool task ended; finish: `verify` 2367146, `select` 2367170, `summarize` 2367207, archive written 00:45 |
| 00:45 | The stacking prepared; canary: task 1 (2367217) and the first task at `M = 32` (2367223, task 16) |
| 00:52 | The stacking look: `verify` 2367263 (2 verified), `assemble` 2367265 exited 1, the expected end of a canary |
| 00:53 to 00:58 | The stacking subsets, a minute apart: `M2` 2367274, `M3` 2367286, `M4` 2367295, `M5` 2367307, `M8` 2367316, `M16` 2367324, `M32` 2367369 |
| 01:11 | Arm 0's last task ended; its finish: `verify` 2367641 (2398 verified, 2 failed), `summarize` 2367672, archive written 01:16 |
| 01:21 | `rescore-arms` submitted (job 2367702) |
| 01:22 | The last stacking task ended; finish: `verify` 2367720, `assemble` 2367727 (960 cells of 200 tasks), archive written 01:23 |
| 01:23 | The analyses submitted: `svbmc_shrink_elbo.py` 2367733, then `svbmc_single_run_bias.py` 2367734 after it |
| 02:11 | The analyses ended; their outputs archived and uploaded |
| 03:23 | `rescore-arms` ended (2 h 2 min); its archive part written and uploaded |

## The queue

The cluster was busier than in the first batch, with over a thousand
jobs of other users pending through the evening, and the account's fair
share stood low after the first batch, so the batch's tasks ranked
below most of the queue and started only as it drained ahead of them:
the tasks running at once stayed between 75 and 225 for most of the
evening against a ceiling of 300, and Arm 0's look waited 33 minutes
for its `verify` job to start. No node failed its prolog, no task was
requeued or cancelled, and no case went missing; each campaign's
`slurm/sacct.txt` in its archive holds the accounting of every job.

## Reference population, PyVBMC 1.0.4 (`population_v104`, Arm 0)

The 24 configurations of the `production` suite at seeds 0 to 99, 2400
cases, run with PyVBMC 1.0.4 and gpyreg 1.0.4 through the harness's
legacy profile, prepared with `--compare` naming the after arm.
`verification.json`: 2398 verified, 0 verify failed, 2 failed, 0
interrupted, 0 partial, 0 missing, 0 stray.

Accounting (tasks that ran a case): elapsed median 5.6 min, quartiles
2.8 and 10.6 min, maximum 3 h 0 min (`cigar_D15_exhaust`); about 387
CPU-hours; peak resident set 859 MB (`cigar_D15_exhaust`), against limits
of 2 h (5 h for `cigar_D15_exhaust`) and 3 GB; five nodes of the family.
The finish's `verify` took 2 min 57 s and 1.20 GB, `summarize` 4 s.

The tracked copies are the manifest, the verification report, the
summary and, per configuration, every verified case's completion record,
sidecar and boost report, with `rescored/` and `rescoring.json` written
by `rescore-arms` (below); `redaction.json` lists the two failed cases
under `cases_not_verified` and the finish's archive part alone.

## The failed cases

Two Arm 0 cases raised inside PyVBMC 1.0.4, `cigar_D15_exhaust` at seeds
5 (index 1506, task 2359174_1506, after 1 h 8 min) and 96 (index 1597,
task 2359174_1597, after 1 h 8 min), with the same traceback: in
`optimize()` (line 1205 of 1.0.4's `pyvbmc/vbmc/vbmc.py`), in `kl_div`
→ `moments` → `VariationalPosterior.sample`,
`np.repeat` raised `ValueError: repeats may not contain negative values`,
a negative sample count per component. Each run is fixed by NumPy's
global state seeded with the case's seed, so the failure is the
trajectory's.

That line is the symmetrized KL divergence between the iteration's
posterior and the previous one, which 1.0.4 computes right after the
iteration's variational optimization, before any final boost; the
configuration's other runs of 1.0.4 took up to 3 h. `sample` counts each
component's samples as `np.floor(w * N).astype(int)`, negative only for a
weight that is not finite (the weights are never negative): the
variational optimization returned a posterior whose weights were not
finite, and 1.0.4 went on with it. The release's variational optimization
passes over candidates whose ELBO is NaN and raises if every one is
(`CHANGELOG.md`). The legacy profile's compatibility patch acts in that
optimization once the GP holds a single hyperparameter sample, and
changes no value there: it takes the single element of the ELBO's
variance, which NumPy before 2.4 stored in its place, and the gradient of
the variance that it also reshapes is never computed there, since 1.0.4
optimizes with gradients only where it computes no variance. The PI ruled
(2026-10-06) that both cases stay failed, as 1.0.4's outcome on those
seeds, and count as failures in the statistics: they are not run again,
given up or patched, and the comparison with the after arm counts each as
a run of 1.0.4 that gave no usable posterior, an unusable run in the
McNemar test of usability of `cigar_D15_exhaust` and a failed case in Arm
0's counts, and leaves their seeds out of the signed-rank tests of the
metrics, which have no value of them to rank (Arm 0's plan, decision 9).

## Reference population, Arm 3 (`population_arm3`)

The same 2400 cases with Arm 3's code and gpyreg v1.4.0, prepared with
`--pair` naming the after arm across the two harness commits.
`verification.json`: 2400 verified, 0 failed, 0 missing, 0 partial, 0
stray; no error file.

Accounting: elapsed median 2.2 min, quartiles 1.1 and 5.5 min, maximum
30.8 min (`cigar_D15_exhaust`); about 165 CPU-hours; peak resident set
374 MB (`cigar_D15_exhaust`), against 1 h (1.5 h for `cigar_D15_exhaust`)
and 2 GB; three nodes of the family. The finish's `verify` took 2 min 8 s
and 1.52 GB, `summarize` 5 s, `rescore` 35 min 53 s and 1.05 GB for the
4800 cases of both arms (the canary's look had rescored the after arm's
2400 already, in 36 min 33 s).

`rescore` reproduced the in-run metrics of every Arm 3 case exactly
(`elbo_err`, `gskl`, `mmtv` and `rmse` equal for 2400 of 2400, none
non-finite), and Arm 3's code scored the after arm's 2400 cases as the
release code had for every case and metric (2400 of 2400 equal, none
non-finite). The tracked copies are the manifest, the verification
report, the summary and, per configuration, every case's completion
record, sidecar and boost report, with the `rescored/population_arm3.json`,
`rescored/population_after.json` and `rescoring.json` of its own
`rescore`.

## The fresh seeds (`fresh_release`, `fresh_arm3`)

`rosenbrock_D2_noise3_production` at seeds 100 to 199, 100 cases each,
with the release code from the harness checkout at Arm 0's commit and
with Arm 3's code from its checkout, `fresh_arm3` prepared with `--pair`
naming `fresh_release`. Both: 100 verified, 0 failed, 0 missing, 0
partial, 0 stray; no error file.

| Campaign | Elapsed, median [IQR], max | CPU-hours | Peak resident set | Finish steps |
|---|---|---|---|---|
| `fresh_release` | 3.4 [3.1, 3.9] min, 5.9 min | 6 | 301 MB | `verify` 7 s, `summarize` 3 s, `rescore` 42 s |
| `fresh_arm3` | 3.5 [3.2, 4.0] min, 5.8 min | 6 | 343 MB | `verify` 7 s, `summarize` 4 s, `rescore` 1 min 19 s |

`fresh_release`'s `rescore` reproduced its 100 cases exactly on every
metric. `fresh_arm3`'s reproduced its own 100 exactly and scored
`fresh_release`'s 100 cases as the release code had for every case and
metric. The tracked copies are each campaign's manifest, verification
report, summary and per-case records, sidecars and boost reports, with
`rescored/` and `rescoring.json`.

## S-VBMC run pools with Arm 3 (`pools_arm3`)

The 8 conditions of the `svbmc_pool` suite, seeds from 1000, 350 a
condition and 480 for the ring, 2930 cases, with Arm 3's code and gpyreg
v1.4.0; `prepare` was given `--target 320 --max-seeds 350 --allocation
ring_D2_noise3_svbmc=320/480`. `verification.json`: 2930 verified, 0
failed, 0 missing, 0 partial, 0 stray; no error file. `select` reached
its target of 320 in every condition with no shortfall and nothing
unfinished, 2560 selected runs in all:

| Condition | Selected / target | Seeds scanned | Pass rate over the scanned seeds |
|---|---|---|---|
| `multisensory_s1_D6_noise3_svbmc` | 320 / 320 | 322 | 0.99 |
| `multisensory_s1_D6_noise1.3_svbmc` | 320 / 320 | 321 | 1.00 |
| `rosenbrock_D2_noise3_svbmc` | 320 / 320 | 321 | 1.00 |
| `gmm_D2_noise3_svbmc` | 320 / 320 | 327 | 0.98 |
| `ring_D2_noise3_svbmc` | 320 / 320 | 420 | 0.76 |
| `student_D8_noise3_svbmc` | 320 / 320 | 320 | 1.00 |
| `gmm_D2_svbmc` | 320 / 320 | 324 | 0.99 |
| `multisensory_s1_D6_svbmc` | 320 / 320 | 326 | 0.98 |

Accounting: elapsed median 4.2 min, quartiles 2.5 and 6.2 min, maximum
11.7 min (`student_D8_noise3_svbmc`); about 222 CPU-hours; peak resident
set 592 MB (`multisensory_s1_D6_noise3_svbmc`), against 45 min and 2 GB;
five nodes of the family. The finish's `verify` took 2 min 28 s and
465 MB, `select` 2 min 59 s, `summarize` 3 s. The tracked copies are the
manifest, the verification report, the selection and the summary.

## Stacking comparison on Arm 3's pools (`stacking_arm3`)

The release gate's grid on Arm 3's pools: both arms at `M` = 2, 4, 8 and
16 and the integrated arm alone at 3, 5 and 32, 20 repetitions below
`M = 16` and 10 from it, 960 cells in 200 tasks; `prepare` reported no
condition short of repetitions. `verification.json`: 200 verified, 0
failed, 0 missing, 0 partial, 0 stray; `assemble` merged 960 cells of
200 tasks into `results.json`, `summary.json`, `summary.md` and
`sources.json`, the tracked copies with the manifest and the
verification report.

Correction to the preserved `stacking_arm3/summary.md` (2026-10-07): its
opening says every cell ran both implementations, and its aggregate
runtime line says "All 120 cells" of each condition. The runtime ratio
and weight comparison use **70 paired cells per condition, 560 in total**,
as their `n` fields in `summary.json` record. The other 50 cells per
condition ran the integrated arm alone. The hash-pinned campaign copies
retain their original text; the summary generator reports the paired
count explicitly.

Per subset, from the accounting:

| Subset | Tasks | Limits | Longest task | Peak resident set |
|---|---|---|---|---|
| `M2` | 8 | 1 h, 4 GB | 2 min 43 s | 1.03 GB |
| `M3` | 8 | 1 h, 4 GB | 1 min 36 s | 0.80 GB |
| `M4` | 8 | 1 h, 4 GB | 7 min 16 s | 1.10 GB |
| `M5` | 8 | 1 h, 4 GB | 2 min 53 s | 0.76 GB |
| `M8` | 8 | 1 h, 4 GB | 25 min 11 s | 1.31 GB |
| `M16` | 80 | 30 min, 4 GB | 12 min 15 s | 1.62 GB |
| `M32` | 80 | 1 h, 6 GB | 9 min 32 s | 2.78 GB |

The 200 tasks took about 16 CPU-hours; the finish's `verify` took 4 s,
`assemble` 25 s and 482 MB; the tasks ran on four nodes of the family.

## The rescoring of the three arms (`rescore-arms`)

One batch job from the harness checkout at Arm 0's commit (job 2367702,
2 h 2 min, 551 MB peak), after the finishes of Arm 0 and of Arm 3's
population, rescoring with the release code the after arm first, then
Arm 0 and Arm 3, into `population_v104/rescored/` and `rescoring.json`:

| Campaign rescored | Cases | Equal to their in-run metrics |
|---|---|---|
| `population_after` | 2400 | `elbo_err`, `gskl`, `mmtv`, `rmse`: 2400 each, none non-finite |
| `population_v104` | 2398 | `elbo_err` 2398, `gskl` 1697, `rmse` 1698, `mmtv` 0, none non-finite |
| `population_arm3` | 2400 | `elbo_err`, `gskl`, `mmtv`, `rmse`: 2400 each, none non-finite |

The after arm's line is the check the guide requires, that the rescoring
scores as the after arm's code did. Arm 0's rescored metrics, which the
comparison reads for it, differ from 1.0.4's in-run ones where scoring
draws samples, which 1.0.4 takes from NumPy's global state and the
release from the posterior's generator: `mmtv` in every case, and
`gskl` and `rmse` in the 700 cases of the seven configurations with
bounded parameters, whose moments are Monte Carlo estimates. Of the
1698 cases with exact moments, `rmse` is equal in every one and `gskl`
in all but one, `rosenbrock_D2_noise3_production_seed37`, whose in-run
`gskl` is 1.3160310748574509 and rescored `gskl` 1.3160310748574506;
the two versions' `kl_div_mvn` compute the log-determinant
differently. Arm 3's code scores every case as the release code does. The rescoring and its log went back as an
archive part of their own, since the finish had archived the campaign
before it.

## The two analyses of Arm 3's pools

Two batch jobs from the Arm 3 checkout, the second waiting for the
first, with their outputs under one directory (`analyses_arm3/`) that is
an asset of the stacking's draft release; like the first batch's, they
hold the cluster's details and no file of them enters the repository.

- `svbmc_shrink_elbo.py` (job 2367733): 2 min 24 s and 631 MB;
  `960/960 cells`; `shrink/cells.jsonl`, `summary.json`, `summary.md`
  and `sources.json`.
- `svbmc_single_run_bias.py` (job 2367734, after the first): 47 min 5 s
  and 872 MB; `2560/2560 runs`; `single_run/runs.jsonl`, `cells.jsonl`,
  `summary.json`, `summary.md`, `added.json`, `added.md` and
  `sources.json`.

Both jobs exited 0.

## Files

Every campaign directory here holds `manifest.json` (without its `site`
block), `verification.json` and `redaction.json`, and what its harness
declares as tracked copies, as the first batch's README describes; Arm
0's `rescored/` and `rescoring.json` are those of `rescore-arms`, and its
`redaction.json` records the finish's archive part and not the
rescoring's. The redaction names the hosts that ran a job by the node
family and any other host by `login`, keeps the CPU model and `Linux
x86_64` of each host part, writes a path under a named directory as that
name, and copies neither the task logs, the accounting, the claims nor
the error files; `cases_not_verified` is empty in every campaign but Arm
0's, where it lists the two failed cases. No `--path` name and no
`--allow` exemption was needed. The comparisons are the PI's, from the
tracked copies: `analyze_population_run.py --arms` on `population_v104`
and the first batch's `population_after`, on `population_after` and
`population_arm3`, and on `fresh_release` and `fresh_arm3`.

## Archives

Each campaign's whole directory, written by its finish as one
zstd-compressed part with its SHA-256 file, is an asset of a draft
release of `acerbilab/pyvbmc`; the two fresh campaigns share one, Arm 0's
also holds the rescoring's part, and the stacking's also holds the
analyses' outputs; the public assets of `population_arm3` and
`pools_arm3`, built by `campaign_public.sh` and checked with its
`--check`, are assets of `release-gate-public-arm3-20261005`. The archives
hold the site's details and the operator's paths, so every release stays
a draft. `cat <name>.tar.zst.[0-9][0-9][0-9] | zstd -d | tar x` restores
a campaign directory.

| Release | Asset | Bytes | SHA-256 |
|---|---|---|---|
| `release-gate-population-v104-20261005` | `population_v104.tar.zst.000` | 433252646 | `ee43fbb66ba5e4644dac758260761266c1eae43f37cec4299f181daa8a2d2271` |
| `release-gate-population-v104-20261005` | `population_v104.rescoring.tar.zst` | 2190891 | `0c830c91f84b5dfbaf54363c422f2072c90f4939051604d4c7126f6ddd2a8fc2` |
| `release-gate-population-arm3-20261005` | `population_arm3.tar.zst.000` | 820845590 | `f7d721c977f614fb98e39a66e4041a86e71e49bece1efa9ae818526dc8e9561c` |
| `release-gate-fresh-20261005` | `fresh_release.tar.zst.000` | 17398311 | `374b231a1330e485b58293db197448c6d67108ff0d208d1552bf4d28778a3c33` |
| `release-gate-fresh-20261005` | `fresh_arm3.tar.zst.000` | 16830922 | `ed340fe88e1116f3fd476914f2db2f742640cb2754c9965974200bc96e61e3cd` |
| `release-gate-pools-arm3-20261005` | `pools_arm3.tar.zst.000` | 233780131 | `59eb9a2cb4484b548b4cc26d4d884988bbac0981a9ba2bf0e8687a560a1a569b` |
| `release-gate-stacking-arm3-20261005` | `stacking_arm3.tar.zst.000` | 15839437 | `309d4ea4ea36a6f336c678c4a9328d4cdde5bcaa82c5ba72891736d28d8da937` |
| `release-gate-stacking-arm3-20261005` | `analyses_arm3.tar.zst` | 2994117 | `50f9dbc55dbc8a96b1d88c9a51b66c24fa05a5fa3e802d5827cc06b29425adc5` |
| `release-gate-public-arm3-20261005` | `population_arm3.public.tar.gz.000` | 653057679 | `dbda6df9a1686d4ff47aa63737a53c16b99db7ab0504c1933b11a751d5d8ab9a` |
| `release-gate-public-arm3-20261005` | `pools_arm3.public.tar.gz.000` | 257322902 | `ec922ee9e5b41b291c90cdfcd3014ff15d5f095564ab6fd06194860905868cf5` |

Arm 3's population public asset holds 12008 files: the 4800 numeric
files (the trace and the posterior arrays of each of the 2400 verified
cases), the 7206 tracked copies, their `redaction.json`, and
`public.json`, which records the code that built the asset; the 2400
boost pickles are left out. The pools' holds 8798 files: the 5860 run
files (each run's `.npz` and its `.json`, the latter redacted as the
copies are), the 2930 completion records, the 6 tracked copies, their
`redaction.json` and `public.json`. Both pass `campaign_public.sh
--check`.
