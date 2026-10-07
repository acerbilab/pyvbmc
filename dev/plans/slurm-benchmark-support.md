# Slurm benchmark support for the release gate

Created: 2026-09-25. Status: **reviewed by the PI on 2026-09-25; Phases 2
to 5, Phase 7's redaction, operator's guide and brief, and the tools of
Phase 9, the promotion and the public assets implemented on
`feat-slurm-campaigns`, doublechecked and merged into `dev-next` after the
PI's review; Phases 1 and 6 done on the cluster (2026-09-25 to 10-01);
Phase 8's campaigns run on 2026-10-02 and handed back (PR #181, merged on
2026-10-04); Phase 9 run on the developer's machine on 2026-10-04**.
What it assumes of the cluster rests on the surveys of 2026-09-25 and
2026-09-28 (Phase 1), on the smoke campaigns of Phase 6 and on the records
of the September pool. The harness sections below describe each harness as
it stood before this plan's work and what the work gave it.

## Live checklist

- [x] The PI's review of this plan (2026-09-25).
- [x] Phase 1a: the cluster survey (2026-09-25).
- [x] Phase 1b: the survey where the campaigns run, the environment and
  the source trees (2026-09-28/29).
- [x] Phase 2: the generic Slurm driver (2026-09-25).
- [x] Phase 3: the pool harness (2026-09-25).
- [x] Phase 4: the population harness and the arm comparison (2026-09-25).
- [x] Phase 5: the stacking harness (2026-09-25).
- [x] Phase 6: smoke campaigns on Turso (2026-09-29, and its last four
  checks on 2026-10-01; the worklog).
- [x] Phase 7: the operator guide, an independent review, the merge into
  `dev-next`, and the brief. The guide, the redaction, two review rounds
  (2026-09-26), a doublecheck of Phase 6's commits (2026-09-29) and a
  doublecheck of the whole branch before the merge, with its fixes
  (2026-09-30); the branch merged into `dev-next` after the PI's review;
  the [brief](release-gate-handoff.md), with the limits Phase 6 measured
  ("Resources and cost"), written on 2026-10-01.
- [x] Phase 8: the campaigns, run on 2026-10-02 from the release commit
  `ff3ed014` and handed back on 2026-10-04 (the worklog).
- [x] Phase 9: the replay fingerprints and the gate runs on the
  developer's machine (2026-10-04; the worklog).
- [x] The smaller fixes left for after the merge (the worklog of
  2026-09-30), on the branch `feat-slurm-small-fixes` (2026-10-07; the
  worklog):
  - [x] the stubs run the batch script as Slurm's spooled copy;
  - [x] the generated `golden/README.md`'s recipe for other machines;
  - [x] the promotion refuses a dirty checkout before the fingerprints'
    runs;
  - [x] the README check's blind spots (a hostname that a name the copies
    write begins or ends, host lists, the site's domain, features in
    prose);
  - [x] tests of the promotion's refusals and of the release grid;
  - [x] the note on `--sidecars` in the `golden_replay.py` entry of
    `dev/README.md`, with the promotion's template and guard;
  - [x] a doublecheck, its fixes, and every harness test module run.

**Pickup point.** The campaigns of Phase 8 ran on 2026-10-02 in the
operator's account, from the release commit `ff3ed014`, and every case of
the four verified; the operator handed them back on 2026-10-04 (PR #181).
Their redacted tracked copies are under
`dev/experiments/release_gate_20261002/`, whose README records the runs,
the node failures and resubmissions, the accounting and the archives (the
worklog, 2026-10-04). Phase 9 ran on the developer's machine on
2026-10-04 and met its acceptance (the worklog). The smaller fixes that
the worklog of 2026-09-30 left for after the merge were made on
2026-10-07 (the worklog), before the promotion, which took the after arm,
since MATLAB's end of warm-up stays (PI, 2026-10-06): the after arm is the
golden reference `reference_2400_20261007` since 2026-10-07 (the
[promotion record](../golden/promotion_20261007/README.md); the roadmap's
pickup 19). What remains of this plan is the review's optional findings,
which await the PI's ruling.

The arm comparison and the promotion are recorded by the roadmap's pickup
19, the reading of the pools and the stacking by the final large-scale
check, pickup 18 of the [roadmap](modernization-roadmap.md). The arm
comparison's noisy Rosenbrock target put
the end of warm-up in question
([the note](../results/2026-10-05-noisy-rosenbrock-warmup.md)); on the
campaigns of Arm 3 ([its plan](arm-3-warmup-comparison.md)) the PI kept
MATLAB's end (2026-10-06;
[the reading](../results/2026-10-06-arm-3-reading.md)), so the promotion
takes the after arm and Phase 9's runs stand. The PI accepted the
assessment on 2026-10-07 (the worklog). The PI's rulings of 2026-09-28 on the
promotion and on the release's assets are in "The populations", "Records
and hand-back", Phase 9 and the worklog. Two further arms run through this workflow in
one batch: PyVBMC 1.0.4 on the population cases (Arm 0,
[its plan](arm-1.0.4-comparison.md)), from a harness commit that holds
the after arm's run code, and the release code with the port's end of
warm-up on the population cases, pools and stacking (Arm 3).

## Purpose and scope

The release gate of PyVBMC 1.5 is a set of campaigns too large for one
machine: the S-VBMC run pools and their stacking comparison (the final
large-scale check, pickup 18 of the
[roadmap](modernization-roadmap.md)) and the reference populations that
replaced the golden references after the port review (the roadmap's pickup
19). This plan owns how they run on a Slurm cluster:
submission, resources, resumption, verification, collection, the
provenance of every run and what of it enters the repository. It also owns
the small set of exact replay traces that stays on the developer's
machine.

What other documents own, and what this plan hands them:

- The [campaign plan](svbmc-benchmark-campaign.md) owns the design and the
  acceptance criteria of the S-VBMC comparison; its decision 14 records the
  release pools and stacking decided here. This plan hands it the verified
  release pools and the assembled stacking results.
- The golden references item of `dev/TODO.md` owned the assessment of the
  new reference and its promotion, by the method of the
  [population plan](final-population-benchmark.md), with a promotion record
  under `dev/golden/` as for `reference_990_20260913`. This plan handed it
  the verified before and after populations, their rescored metrics and
  the replay fingerprints; the promotion of `reference_2400_20261007` on
  2026-10-07 closed it ([its record](../golden/promotion_20261007/README.md);
  the roadmap's pickup 19).
- The [headline note](../2026-09-15-svbmc-headline-shrinkage.md) owns the
  choice of the S-VBMC headline estimator.

The seed of the workflow is `dev/scripts/hpc/`, the scripts that generated
the September pool (`experiments/svbmc_pool/pool_20260914/README.md`) around
`scripts/svbmc_pool_run.py`: a manifest with a source identity, a case list,
one worker per array task with a hash-verified completion record, and a
`verify` that reconciles the allocation. This plan makes that shape a
contract that every harness of the release gate meets, and adds scripts
that drive any harness meeting it.

## What runs where

Decisions of the PI, 2026-09-25.

The gate runs two kinds of VBMC campaign. A **reference population** runs
each of the 24 configurations of the `production` suite at seeds 0–99 and
keeps every run; it measures the accuracy of the code and becomes the new
reference. An **S-VBMC run pool** runs each of the 8 conditions of the
`svbmc_pool` suite at every seed from 1000 up to a cap, and keeps for
stacking the first 320 runs, in seed order, that pass the filters of the
S-VBMC paper, which are also both implementations' defaults: the run
reports itself stable, and `sqrt(max J_sjk) < sqrt(5)`, a bound on the
GP's uncertainty about the components' expected log joints. The runs that
fail are kept with their verdict and never stacked. The stacking
comparison then stacks subsets of each pool.

| Campaign | Where | Code | Size |
|---|---|---|---|
| Reference population, after arm | Turso | the release code, gpyreg `v1.4.0` (`682585f`) | the 24 configurations of the `production` suite, seeds 0–99: 2400 runs |
| Reference population, before arm | Turso | PyVBMC `f91fdf0`, gpyreg `v1.2.1` (`9e70e6b`) | the same 2400 runs |
| S-VBMC run pools | Turso | the release code | the 8 conditions of `svbmc_pool`, seeds 1000–1349 (1000–1479 for the ring): 2930 runs, of which 320 per condition are stacked |
| Stacking comparison | Turso | the release code and the original S-VBMC 0.1.1 (`13a78f6`) | both arms at `M` = 2, 4, 8, 16 (560 cells); the integrated arm alone at 3, 5, 32 (400 cells); disjoint subsets |
| Replay fingerprints | the developer's machine | the release code | one seed of each `production` configuration, and the six seeded gate runs |

1. **One suite for new references: `production`.** `golden` and
   `production` share their 16 noiseless configurations and differ in the
   budget of the noisy ones: `golden` runs them at the 2020 paper's
   50 (D + 2) evaluations, `production` at the package's defaults for a
   specified-noise target (75 (D + 2) evaluations, a stability window of
   90 evaluations, VIQR). Both suites define `lumpy_D10_noise3`, which the
   golden reference holds no runs of. The paper budget served the
   trajectory identity of the existing references; that job passes to the
   replay fingerprints (decision 4), so no new reference runs at the paper
   budget. The existing references and their records stay as they are.
2. **100 seeds per configuration, 0–99**, in both arms, so that the two
   arms pair seed by seed and the noiseless configurations also pair with
   `reference_990_20260913` at its seeds. The port review's fixes move
   every CMA-ES search with `D > 1` and every sampled GP fit (the
   [ledger](../results/2026-09-23-port-correctness-review.md), "The fixes
   that move default trajectories"), so the noiseless runs are regenerated
   along with the noisy ones. The 100 seeds hold for the two costliest
   configurations as well: `cigar_D15_exhaust`, about a fifth of a
   population's cost, and `lumpy_D10_noise3_production`, whose seed
   count was to be revisited had its first smoke case taken hours; it
   took about 11 minutes (Phase 6).
3. **A before arm.** The only old-code runs of the noisy `production`
   configurations are the 80 runs of 2026-09-18/19 (six configurations,
   10 to 20 seeds each), too few to judge the fixes on those targets. The
   before arm runs the code at the start of the port review (`f91fdf0`,
   `dev-next` on 2026-09-19, with the gpyreg that code required) on the
   same seeds, node family and environment, so that the comparison is
   paired and the arms differ in their code alone. It measures every
   change to the default path since `f91fdf0`, the port review's fixes
   among them.
4. **Exact replay from one seed per configuration, on the developer's
   machine.** An exact replay baseline depends on the BLAS rounding of the
   machine that made it ([population plan](final-population-benchmark.md),
   "Run on the original benchmark machine"), so the statistics come from
   the cluster and exact identity from a small local set. One seed per
   configuration detects any change to the arithmetic on the paths the run
   takes; a further seed adds coverage only where it takes a branch no
   other run takes, and is added for that reason alone. The set is bound
   to its machine: on a new machine it is regenerated at the commit the old
   set certifies, checked as in Phase 9, and recorded in that machine's
   `dev/scripts/runs/LOCAL.md`.
5. **The pools: 320 stacked runs per condition.** Ten disjoint stacks of
   `M = 32` runs need 320 runs that pass the filters; the September
   pools, 100 runs per noisy condition and 50 per noiseless one, reuse
   each run 3.2 to 6.4 times at `M = 32`. The seed caps follow the
   September pass rates (`experiments/svbmc_pool/pool_20260914/README.md`):
   350 seeds for the seven conditions where 97 % or more of the runs
   passed, 480 for the ring, where 73 % did. The seeds start at 1000,
   continuing the campaign's range, so a release run and the campaign's
   run of a seed start from the same point.
6. **Disjoint subsets at every `M`**, with the campaign's repetition
   counts (20 at `M` = 2, 3, 4, 5 and 8; 10 at 16 and 32; `floor(320 / M)`
   exceeds each). Repetition `r` at `M` takes the `r`-th block of `M` runs
   of a permutation of the condition's selected runs drawn for that
   condition and `M` alone, so the subsets are disjoint within one `M` and
   independent across `M`. Repetitions that share runs are correlated, and
   their spread understates the uncertainty of a cell's median.
7. **Both arms of the stacking comparison run on the release pools**, at
   `M` = 2, 4, 8 and 16 as in the campaign: the release pools and their
   subsets are new cells, and criteria 1, 2 and 4 and the first clause of
   criterion 3 compare the two arms on the same cells. S-VBMC has also
   changed since the campaign, and the working rules require the
   comparison against the original for any change to it.
8. **Stacking runs on Turso**, as array tasks (the stacking harness below).
9. **`performance_calibration="off"` in every campaign run**, as both
   harnesses set it. `"cached"` would have every task read and write one
   calibration cache in the operator's home directory, so a run would
   depend on which task calibrated first and on how loaded its node was.
10. **The six seeded gate runs** of
    `experiments/port_review_20260919/verification/scripts/wave2_fixpass_gate_runs.py`
    are fingerprints of the same kind and join the local set (Phase 9).
11. **Operation.** A postdoc of the lab runs the campaigns in their own
    account and hands them back as draft-release assets and a pull
    request, as in September. The smoke campaigns run in the PI's account.
12. **Two campaigns, not one.** Four pool conditions are the problem of a
    `production` configuration, with its target, dimension, noise level
    and budget: noisy Rosenbrock, Student D8 and multisensory at noise
    1.3, and noiseless multisensory. Their runs are not shared, because
    each harness saves what its own analysis reads. A pool run saves,
    in about 300 KB, the returned posterior with its statistics `I_sk`
    and `J_sjk` (each component's expected log joint and the GP's
    uncertainty about it, which S-VBMC stacks), the GP that produced
    them, from which `verify` recomputes them, and the run's evaluations.
    A population run saves, in 0.4 to 1.1 MB, the trace of every
    iteration (the ELBO, the GP hyperparameters, the posterior and the
    transformer at each iteration, the initial design), which
    `golden_replay.py` compares step by step, and the state at the final
    boost, which the population analysis checks; it holds neither the
    statistics that S-VBMC stacks nor the GP they come from. The seeds
    differ too, 0–99 against 1000 upward, and a pool needs 320 passing
    runs, so one harness for both would save few runs; the duplicated
    ones are of targets that take minutes.
13. **The release code is the latest `dev-next` at the launch.** The
    campaigns of Phase 8 launch once no algorithmic work on 1.5 remains,
    from the latest commit of `dev-next` at that time. After the launch
    only the documentation changes, and the S-VBMC headline selection
    that the gate itself decides; a change to the numerics after it would
    need the campaigns it reaches to run again. From the launch to the
    promotion of the new reference no file of the package changes but
    its tests and S-VBMC, a docstring or a tip's wording included: the
    fingerprints, the gate runs and the promotion refuse a commit whose
    package differs from the after arm's in any file
    (`reference_promote.py`, `NUMERIC_PATHS`, which also lists the
    harness files that build a run). Such an edit lands before the
    launch or after the promotion.

## The cluster

The campaigns run on Turso, the University of Helsinki cluster the
September pool ran on. The scripts hold no value particular to a site: the
node feature, the modules, the location of the environment and anything
else a site needs are the operator's settings (below), and their values for
a site are kept in the operator's own notes, outside this repository.
What the design assumes of the cluster:

- **Slurm with job arrays.** The driver reads `MaxArraySize` from
  `scontrol show config`, splits a campaign into chunks below it and
  prints how many chunks the campaign takes.
- **One node family per campaign.** Runs on different node families are
  not bitwise reproducible against each other (September pool README,
  "Outcome"), so every task requests one node type with `-C`. The two
  population arms, the pools and the stacking use the same feature, so
  that the arms differ in their code alone and the stacking's runtime
  ratios come from one CPU model. A family that no single feature
  selects, one whose nodes carry only features that other families share,
  cannot hold a campaign.
- **Physical cores.** Where Slurm counts hardware threads as CPUs,
  `--cpus-per-task=1` is one thread and two tasks can share a core, so
  every task passes `--hint=nomultithread`; the accounting may then count
  both threads of the core. On Turso (Phase 1b) a task so placed holds
  both threads of one core, the job's cpuset equal to them, and the
  accounting counts one CPU.
- **Short tasks.** Every task of this plan takes minutes, and `TIME` is set
  from the measured durations (Phase 6).
- **Network on the login node only.** Environments and source trees are
  built there, after any setup the site needs (`LOGIN_SETUP`); tasks need
  no network.
- **Batch jobs only.** Every computation, `verify` and the finishing steps
  included, is a batch job, so that every step is in the accounting and
  none depends on the operator's session.
- **Storage without backup.** Campaign directories live on the shared
  filesystem, in the operator's home or a scratch area. Cluster storage is commonly neither backed
  up nor kept indefinitely, so a campaign is archived as soon as it
  verifies ("Records and hand-back"). Compute nodes may have no local
  disk, so `TMPDIR` goes under the campaign directory.
- **A shared parallel filesystem.** Each process writes its own files; no
  two processes append to one file; each harness writes its per-case files
  into one subdirectory per configuration or condition, so that no
  directory holds thousands of files; nothing runs `find` or a recursive
  listing on it.

In September 1100 tasks at 200 concurrent ran in 33 minutes of wall time;
a task took about twice the laptop's time (90.5 CPU-hours against the 45
estimated), `MaxRSS` peaked at 302 MB, and the longest task took 11.5
minutes (September pool README, "Resource fit").

## Operator settings

The driver reads its settings from the operator's environment; the
operator's guide (`dev/scripts/hpc/README.md`, "Settings") lists each
one, its default and whether it is fixed for a campaign. The driver
passes them to the harness's `prepare`, which records them in the `site`
block of the raw manifest, all but the limits of the verify and
finishing jobs, `STEP_POLL`, `STEP_WAIT_LIMIT` and `ARCHIVE_PART_SIZE`,
which each finish reads afresh. Only the raw campaign directory holds
that block ("Records and hand-back"). `HARNESS`, `CAMPAIGN_ENV`,
`NODE_FEATURE`, the source trees and `BASELINE_DIR` are fixed for a
campaign: a later submission and the finish refuse a value that differs
from the manifest's. The others may change from one submission to the
next, and `slurm/jobs.txt` records each submission's.

## The campaign contract

Each harness that the driver runs has these subcommands.

- **`prepare --out DIR ...`** writes `manifest.json`: the allocation, the
  run options, the source identity, the `site` block, the environment's
  `pip freeze` and the harness's finishing steps (`finishing_steps`, each
  an argument list). It runs on the login node and refuses to change the
  identity of a prepared directory.
- **`cases --out DIR [--subset NAME]`** prints one case per line, in a fixed
  order; the line numbers of the full list are the case indices every
  submission refers to, and `--subset` prints `<index> <line>` for each
  case of the subset, the index being its line number in the full list.
  Each line begins with the case's tag: components of letters, digits and
  `_.+=-`, none beginning with a dot, joined by `/`, the first naming the
  configuration or condition, so that a case's files lie in the
  subdirectory of its condition.
- **`worker --out DIR --case LINE`** runs one case, given as the text of its
  line, in a fresh process. It exits 0 at once when the case has a
  completion record or was given up (below); exits 64 on a line or a
  directory that is not the harness's, one without a readable manifest
  included; compares its source identity with the manifest's and refuses
  a difference, or an identity it cannot establish (exit 78; the node's
  features are asked of `scontrol` with retries); claims the case (below;
  exit 75 when the claim is refused); sets an earlier attempt's error file
  aside as `claims/<tag>.error.txt` and removes whatever an interrupted
  attempt left; runs it; and on success writes the artifacts and a
  completion record holding the SHA-256 of every file, the elapsed time
  and the identity with its host part, then removes its claim. None of
  the refusals touches the case's files. On failure it removes the case's
  partial files, writes `<tag>.error.txt` with the traceback, removes its
  claim and exits 1; a failure after the completion record is written
  leaves the case complete. A SIGTERM or SIGINT, which Slurm sends at the
  time limit and on `scancel`, makes it remove the case's partial files
  and its claim and exit 128 plus the signal's number without an error
  file, so that a killed case counts as missing, not failed.
- **`verify --out DIR`** re-checks every completed case against its record
  and the manifest, in the campaign's own environment and source trees; it
  checks that each record's node features include `NODE_FEATURE` and that
  it ran on one physical core: the affinity, or the job's cpuset, equals
  the hardware threads of one core (`core_threads` and `job_cpuset` of the
  host part). It reconciles the allocation case
  by case, each with its index: verified; failed (an error file, or a
  case the operator gave up, whatever its task left); in
  flight (a claim that is live by the rule below, which takes precedence
  over files without a record); interrupted (a stale claim without a
  record, with whatever files the task left, possibly none, as a task
  leaves it when it is killed without a SIGTERM, by SIGKILL or by its
  memory limit); partial (files without a record and without a claim);
  missing (never ran, or stopped by a SIGTERM that removed its claim, with
  an earlier attempt's error attached as `earlier_error` when one was set
  aside); and stray. It writes `verification.json`, and
  exits non-zero on a failed check, a partial case or a stray file.
- The harness's own **finishing steps**, which run only on a verified
  directory; the finish runs each with `--out DIR` appended, in order.
- **`campaign_contract.py give-up --out DIR --case TAG --reason TEXT`**,
  the operator's ruling on a case that stays missing or interrupted at
  every resubmission: it writes the case's error file with the line
  `given up by the operator: <reason>`, logs the ruling in
  `slurm/given_up.txt`, and refuses a case with a record, a live claim or
  a reason of more than one line.

A campaign directory holds, beside each case's artifacts,
`records/<tag>.complete.json`, `claims/<tag>` and `<tag>.error.txt`, and
`subsets/<name>.txt`, `slurm/` (the task logs, the job ids, the
accounting, the rulings of `give-up`) and `tmp/` (`TMPDIR`); `claims/`
also holds the retirement marks of the claim below.

**The claim.** A claim is `claims/<tag>`, holding the job id, the array
task id, the restart count (`SLURM_RESTART_COUNT`), the host and the start
time. The worker writes it to a temporary file and hard-links it into place
(`link()` fails when the claim exists), so a claim is never empty and two
workers never both hold one. When a claim exists:

- a task that finds its own job and array task id in it (a requeue) takes
  it over;
- otherwise the worker asks the Slurm accounting for the state of that
  task (`sacct -n -X -P -j <job>_<task> -o State`). While the task has not
  ended (pending, running, requeued, suspended), and whenever the query
  fails or answers nothing, the claim is live: the worker exits with code
  75 without touching the case's files;
- a claim whose task has ended (completed, failed, cancelled, timed out,
  node failure, out of memory, preempted) is stale. To retire it a worker
  first takes a retirement mark, a dot-file beside the claim created by
  the same hard link; holding it, the worker reads the claim again, and
  only if it is still the one it judged renames it to
  `claims/<tag>.stale.<job>_<task>.<key>` (`<key>` being eight hex digits
  of the claim's token) and creates its own. A worker that finds the mark
  held by a task that may still run refuses with 75; one whose holder's
  task has ended takes the mark at the next level. A requeue's takeover
  takes the mark too. No worker moves a claim it did not judge, so no
  interleaving of workers lets two hold one case.

A refusal never takes the failure path, which deletes the case's files.
A claim made outside Slurm, by `run` on a workstation, is live while its
process runs on the host that made it, and on any other host until the
operator removes it. The accounting keeps a job's state after the job ends. `squeue` answers
only for the jobs the controller still holds and can answer with an
error for one it has dropped, as it does when the query fails, so it
cannot tell an ended task from an unreachable controller. Where the
accounting does not answer from a compute node, every claim is live, and
a resubmission waits for the operator to clear the stale ones; on Turso
it answers (Phase 1b), and Phase 6's resubmission of a killed task took
over its stale claim.

**The source identity**, compared by every worker: the commit and the
clean state of each source tree (the whole harness checkout, the package
tree, gpyreg, and for the original arm the S-VBMC baseline); the SHA-256 of
the harness modules, the targets module and the data it reads
(`dev/scripts/data`); and the versions of Python, NumPy, SciPy, cma and,
where imported, Torch. **The host part**, recorded and not compared: the
hostname, the CPU model, the node's features, the BLAS library and its
thread count as `threadpoolctl` reports them, the CPU affinity
(`os.sched_getaffinity(0)`), the thread variables, and the Slurm job,
array task and restart ids.

Versions come from what is imported, not from the installed distribution.
With a tree pinned by path, the installed metadata can name another
version: the sidecars of `golden_trace.py` record the harness checkout's
commit (`profile_run.git_info`) and read the versions of `pyvbmc` and
`gpyreg` through `importlib.metadata`, which on a workstation name the
installed distributions whatever tree is imported. The campaign
environment installs neither package, so there those versions are null,
in the sidecars as in the pool artifacts' `installed_metadata_versions`.
The identity and the sidecars record each tree's commit and import path,
and label a version read from the installed distribution as such.

## The driver: `dev/scripts/hpc/`

New scripts, `campaign_env.sh`, `campaign_submit.sh`, `campaign_task.sbatch`
and `campaign_finish.sh`, drive any harness that meets the contract. The
September scripts (`svbmc_pool_env.sh`, `svbmc_pool_submit.sh`,
`svbmc_pool_task.sbatch`, `svbmc_pool_finish.sh`) stay as they are, since
that campaign's records cite them. From them the new scripts keep: the
refusal of a dirty tree, which the new scripts apply to every source
tree, `cases.txt` written
once and a later submission refused on a different list, the chunks below
`MaxArraySize` with an index offset, `ARRAY` naming case indices for a
canary or a resubmission, the job ids in `slurm/jobs.txt`, and a finish
that runs `sacct`, refuses while tasks are queued or running, runs
`verify`, prints the indices to resubmit, runs the harness's finishing
steps and writes the archive with its SHA-256.

What is new:

- **The environment** is a conda environment at `CAMPAIGN_ENV`, with
  Python 3.12 as in September, `zstd` for the archive and `gh` for the
  hand-back from conda-forge (a cluster need not have either), and the rest
  from pip at the versions pinned in
  `dev/scripts/hpc/campaign_requirements.txt`, into which Phase 1b froze
  every package the environment takes from PyPI, dependencies included
  (conda's `pip`, `setuptools` and `wheel` aside), so that the
  environments of the smoke campaigns and of the campaigns, built in
  different accounts, hold the same libraries. The submission compares the
  environment's installed versions with the file before `prepare` and
  refuses a difference. It is built on the login node after
  `LOGIN_SETUP`, from a site conda module or a Miniforge installation. The environment script reads
  the login profile (a non-interactive shell may not define `module`
  otherwise), activates the environment, exports single-threaded BLAS,
  `MPLBACKEND=Agg`, the source trees' paths and `TMPDIR` under the campaign
  directory.
- **Every task** passes `-C $NODE_FEATURE`, `--hint=nomultithread` and,
  when set, `-p $PARTITION`. The task script reads its line of `cases.txt`,
  exits at once when that case has a completion record, as in September,
  and otherwise runs `worker --case`.
- **Subsets.** With `CASES_SUBSET` the submission writes that subset's
  cases (`cases --subset`) and submits them as their own array with their
  own `TIME`; `verify` reconciles against the full allocation. The
  populations need it: by the laptop medians `cigar_D15_exhaust` takes
  about 86 times as long as `halfnormal_D2`.
- **The finish** writes the accounting of every recorded job to
  `slurm/sacct.txt` (removed when `sacct` fails), then checks the queue:
  `squeue` states that have ended do not count, and a job `squeue` cannot
  answer for is judged from `sacct.txt`; one that stays unresolved stops
  the finish unless `--allow-running`. It runs `verify` and every
  finishing step as step jobs, submitted with `--parsable`, recorded in
  `slurm/steps.txt` before it waits for them by polling the accounting, so
  that an interrupted finish leaves no step job unrecorded; a `verify`
  submitted while the campaign's tasks run may queue behind them. It
  counts in-flight cases apart from missing ones, so that
  `--allow-running` works without `--allow-missing`; a task still queued
  holds no claim, so the finish maps the queued and running array tasks
  (`squeue -r`) to their case indices and counts a missing case whose task
  is queued as in flight. It lists the interrupted and missing cases as
  the indices to resubmit, and archives only when the accounting shows
  every recorded task ended. It judges the accounting's allocation rows
  only, the step rows staying in `sacct.txt` for their `MaxRSS`; submits
  its step jobs with `--no-requeue`; and gives up waiting for one after
  `STEP_WAIT_LIMIT`. Under `--allow-running` it is a look: `verify` and
  its report, without finishing steps or archive.
- **The environment of a task** has `PYTHONPATH` unset, so that the
  source trees reach `sys.path` only through their own variables; the
  build installs `git` for the identity; `NODE_FEATURE` must be one
  feature name; and `slurm/jobs.txt` records each submission's variable
  settings, `SBATCH_EXTRA`, `CONDA_SETUP` and `LOGIN_PROFILE` among them.
- **Limits.** `TIME` and `MEM` per harness come from the smoke campaigns'
  accounting (Phase 6). The throttle applies per submission, so a campaign
  in several chunks runs up to `THROTTLE` times the number of chunks.

## The harnesses

### The pools: `svbmc_pool_run.py`, `svbmc_pool_io.py`

Before this plan's work the harness had `cases`, a standalone `worker`
and `verify`, but not the contract: its worker took `--label L --seed S`
and had no early exit (the September task script held it); its compared
identity lacked cma, the data directory (which the multisensory
conditions read) and the clean state of the whole harness checkout; its
host part lacked the CPU model, the node features, the BLAS library and
the affinity; and `verify` had no in-flight state. It gained:

- `--case`, the early exit, the claim, the identity and host fields, and
  the `verify` states and checks of the contract;
- one subdirectory per condition for artifacts and records, which `select`,
  `summarize`, `verify` and the stacking harness read (the release pools
  hold about 5,900 files);
- `--gpyreg-source` required at `prepare`, with no default path;
- the convergence fields: `convergence_status`, `message`, `r_index` and
  `iterations` in the completion record and in `summarize`, beside
  `success_flag`.
- a `select` that walks every seed in order and refuses when a seed below
  a condition's last selected run has no record and no error file
  (missing, interrupted or in flight), lists such seeds past a shortfall
  as `unfinished`, and records the SHA-256 of the manifest and of the
  verification it selected from; a case given up counts as a failed
  seed; `profile_run.py` among the hashed harness files. The stacking
  checks a selection against the allocation's first seed, seed cap and
  target (or the `select --target` value it records as
  `target_override`); a pool verified again after its selection is made
  stackable by selecting again.

Its test module `dev/scripts/test_svbmc_pool_run.py` reads the gpyreg
checkout from `PYVBMC_GPYREG_SOURCE`, which the campaign environment
exports, and skips only when it is unset; the operator's environment check
fails on any skipped test. `verify`, `select`, `summarize` and the scripts
that read a pool also read the September pool's flat layout, which its
manifest marks by the absence of `"contract"`.

The allocation of the release pools is set with the existing flags:
`--target 320 --max-seeds 350 --allocation ring_D2_noise3_svbmc=320/480`.

The scripts that read a pool (`svbmc_shrink_elbo.py`, `svbmc_cap_kappa.py`,
`svbmc_single_run_bias.py`, `svbmc_shrink_optimize.py`, and
`svbmc_honest_elbo.py`) took their gpyreg before this plan's work in
three ways: the first four defaulted to the September gpyreg path, which
`svbmc_single_run_bias.py` and `svbmc_shrink_optimize.py` checked
(`pinned_gpyreg_source`) and `svbmc_shrink_elbo.py` and
`svbmc_cap_kappa.py` did not; `svbmc_honest_elbo.py` took the path its
pools' manifests record, or `--gpyreg-source`, and refused a gpyreg that
resolved elsewhere. Since this work the first four take the gpyreg source as
a required argument, and `svbmc_honest_elbo.py` takes it or else the path
its pools' manifests record; all five check it against every pool's
manifest. `svbmc_headline_numbers.py` takes its directories as arguments
instead of the dated names it holds.

### The populations: `population_run.py`, `golden_trace.py`

Before this plan's work, `population_run.py` ran a manifest's cases one
fresh process at a time under one supervisor, which alone wrote
`launch.json`, `status.json` and `finished.json`; its worker needed
`launch.json` and recorded no failure. It gained:

- **Array mode.** `prepare` writes the manifest with the allocation (a
  suite, its labels and a seed range), the options and the expected
  identity, which the worker reads in place of `launch.json`; `cases`
  (with the subsets); a standalone `worker` with the early exit, the claim
  and an error file carrying the traceback of `golden_trace.run_task`;
  `verify` built on `validate_case`; and the population summary as a
  finishing step. `validate_case` loads the boost pickles, which only the
  package that wrote them reads, so a before-arm directory is verified in
  the before arm's environment and trees. The sequential `run` stays, for
  the laptop.
- **Two source trees.** The package comes from a detached worktree at the
  arm's commit, named by `PYVBMC_SOURCE` and put first on `sys.path`, as
  `PYVBMC_GPYREG_SOURCE` does for gpyreg. Only `population_run.py`, run
  as a script, reads `PYVBMC_SOURCE`: before it imports `golden_trace` and
  `profile_run` it puts the tree first on `sys.path` and imports the
  package from it, so that no other script and no process that imports
  the module, a gate among them, is redirected by the variable. The
  identity checks that `pyvbmc.__file__` resolves under that worktree,
  whose own checkout encloses it (a worktree nested inside another tree
  does not pass for it), and records its commit. Before this plan's work
  the identity required the package inside the harness checkout, which would
  make the before arm a branch of `f91fdf0` carrying the new harness with
  that commit's targets module. The harness, the targets module and its data
  come from the harness checkout and are the same files in both arms. The
  targets module has changed since `f91fdf0` (the uncertainty-level-1
  switch, the `oracle` suite, the smoke printout and its help text); every
  `production` configuration is defined identically in both.
- **The returned posterior as plain arrays**, so that the rescoring
  (below) reads the posteriors of both arms with NumPy and one code. The
  trace already holds the returned posterior's `w`, `mu`, `sigma` and
  `lambd` (`final_*`) and each iteration's transformer parameters
  (`pt_*`, whose entry at the sidecar's `best_iter` is the returned
  posterior's transformer). They lack the bounded-transform family and
  whether `scale` and `R_mat` are unset, which `<tag>.vp.npz` holds; with
  the configuration's bounds they rebuild the returned posterior exactly,
  which the worker checks for every case.
- The host part of the identity, and the provenance of the sidecars
  (the contract).
- The boost capture (`BoostCapture`, the `.boost.pkl` and `.boost.json` of
  every case) stays, since `analyze_population_run.py` checks every boost
  decision.

The harness works against `f91fdf0` as it does against the release code:
`VBMC.final_boost(vp, gp)` and the positional
`optimize_vp(options, optim_state, vp, gp, n_fast, n_slow, K_new)` call that
the boost capture patches are the same in both, the boost's
`weight_penalty` is 0 in both, the iteration-history and result keys that
`golden_trace.py` reads are the same, the options the harness sets
(`tol_elcbo_boost`, `vectorized_target`, `performance_calibration`) exist in
both, and the traces count iterations the same way (the length of the
iteration history).

`golden_replay.py` already takes the population whose accuracy envelope a
run that parts is judged against (`--sidecars`); what changes at the
promotion is its defaults, `DEFAULT_BASELINE` (the traces of
`reference_990_20260913`, in place of which come the replay fingerprints
of Phase 9) and `DEFAULT_CONFIGS` (which holds the golden label
`rosenbrock_D2_noise1`, in place of which comes
`rosenbrock_D2_noise1_production`). The promotion, which the golden
references item owns, is `reference_promote.py`'s, in the manner of
`golden/promotion_20260913/promote.py`, since `reference_join.py` extends a
reference and refuses any overlap with it: it copies the after arm's
verified sidecars into `dev/golden/baseline/` and the fingerprints into the
reference's local traces directory, and changes the two defaults together
with `AGENTS.md` ("Trajectories"), the current reference's section and the
`golden_replay.py` entry of `dev/README.md`, and `dev/golden/README.md`
(its steps are in `dev/README.md`). The fingerprints, the gate runs and the
promotion's own process run the after arm's code: equal in the package,
but for its tests and S-VBMC, which no VBMC run imports, and in the harness
files that build a run, which leaves the documentation and the S-VBMC
headline free to change after the launch (decision 13; PI, 2026-09-28).
The script copies itself into its record as `promote.py`, and the
promotion commit removes it and its test from `dev/scripts/`, as the
previous promotion's script lives in its record (PI, 2026-09-28).

### The arm comparison

- **Rescoring.** `benchmark_targets.metrics` takes `kl_div_mvn`, the
  posterior's moments and `mtv` from the package that is imported, and
  these changed after `f91fdf0` (`a37945a8` for `kl_div_mvn`, `23d962a1`
  and others for the posterior's sampling), so the in-run metrics of the
  two arms come from different code. A `rescore` step, run once in a
  release-code process over both arms, rebuilds each returned posterior
  from its plain arrays and recomputes gsKL, MMTV and RMSE with the release
  code, calling `benchmark_targets.metrics` as the run did, with its
  fixed-seed generators, so that the rescored metrics of the after arm
  equal its in-run metrics exactly; the evidence error uses each run's own
  ELBO. The in-run metrics stay in the sidecars.
- **Inputs.** The comparison reads the sidecars, the `.boost.json` files
  and the rescored metrics alone, so that no process imports both packages.
- **Method.** The population plan's: a KS screen per configuration and
  metric under a Holm correction, and paired families within each
  configuration (seeds paired across the arms): signed-rank tests of the
  evidence error, gsKL and MMTV, and McNemar tests of usability. The
  confirmatory family is fixed in the campaigns' manifests before the
  runs, as that plan requires, and keeps its size: a test that cannot be
  computed (no seed paired across the arms, or no finite difference)
  enters the Holm family with p = 1 and is listed as not computed. A pair
  with a non-finite rescored metric leaves that metric's test and is
  counted. The boost-stage usability counts come from each arm's in-run
  metrics, since only each arm's package reads its boost captures, and the
  report labels them so; a stage other than the returned posterior whose
  metrics could not be computed leaves the boost summary and is counted.
- **Binding.** `rescore` runs only with the source identity of the arm
  that rescores (the after arm, here), requires that arm to reproduce its
  in-run metrics exactly, resumes
  from per-configuration work files, and records the SHA-256 of what it
  writes in `rescoring.json`, which the analysis checks. `summarize`,
  `rescore` and the analysis refuse a `verification.json` older than its
  directory, so the before arm is finished completely before the after
  arm's finish, and the after arm is finished again after any later finish
  of the before arm. `prepare --pair` refuses two arms with the same
  `pyvbmc` commit or different environment versions, and, from one harness
  commit, with different harness files; arms of two harness commits pair
  only as clean checkouts, each of its own package, that hold the same
  files that build and score a run and different package code (Arm 3,
  [its plan](arm-3-warmup-comparison.md)). `rescore` without
  `PYVBMC_GPYREG_SOURCE`, and a worker whose package tree PyVBMC cannot be
  imported from, exit 78.
- **The tool.** `analyze_population_run.py` needs a signed-rank test valid
  for 100 pairs (before this plan's work its exact enumeration refused
  more than 62), a verification of the reference arm against that arm's own
  `verification.json` in place of the 870-entry manifest of
  `golden/noisy_extension_20260907/` that it asserts, and a reader for
  array-mode campaigns, which write no `launch.json`, `finished.json` or
  `comparison.md`. The September mode keeps that manifest's check, on the
  870 sidecars that `--reference DIR` names.

### The stacking: `svbmc_pool_stack.py`

Before this plan's work the comparison ran in one process. It verified
the original baseline at the paths its record holds, the Windows paths of
the machine that ran the campaign, at every start, even for the
integrated arm alone. It could not resume: a new run truncated
`cells.jsonl`, and `results.json` was written at the end. A run split by
condition or `M` and merged with `--summarize-only --from-results` took
the single-run rows from its first file only and drew its bootstrap in
another order than an unsplit run. Its repetitions drew independent
subsets, which could overlap. It gained:

- `prepare`, whose manifest holds the pools with their selections, the `M`
  values and repetition counts, the seed, the arms, the baseline's record
  and the identity; `cases`, one line per task; a `worker` with the claim;
  and `verify`.
- **Tasks.** A task computes all repetitions of one condition and `M` for
  `M` ≤ 8, and one cell for `M` = 16 and 32, writing one result file per
  cell. Each task runs the harness's `warm_up` before its first timed
  cell, so that first-call costs stay out of the timings of criterion 4,
  and runs both arms of a cell, the order alternating from repetition to
  repetition (the integrated arm first on even ones), so that a cell's
  runtime ratio is measured on one node.
- **Assembly**, a finishing step: `results.json` built from the cell files
  in their canonical order, the single-run rows computed from the pool,
  and the bootstrap drawn in that order, so that no partial results are
  ever merged.
- **Disjoint subsets** as decision 6 describes.
- **A verified pool.** `prepare` stacks a pool only when its
  `verification.json` passed and its `selection.json` was made from that
  verification and agrees with it, both in the pool's own directory, so
  that a copied pool is verified and selected again where it lies (PI,
  2026-09-30); it refuses a selected run whose files differ from its
  completion record, and records each pool's verification;
  it refuses a dirty harness checkout without `--allow-dirty`; the worker
  exits 64 on a line that is not a task of the manifest.
- **The baseline on the cluster**: the upstream S-VBMC at `13a78f6` in
  `BASELINE_DIR`, with CPU Torch 2.14.0 from the campaign environment,
  recreated as
  `experiments/svbmc_pool/baseline_environment.json` describes and verified
  by the SHA-256 of the committed content of every `svbmc/*.py` it records
  (`files_sha256_committed`; its `files_sha256` are those of a Windows
  working tree whose line endings git converted) and the Torch version,
  not by its paths. Both arms therefore import one Torch. A task of the
  integrated arm alone neither needs nor checks the baseline.

Its test module `dev/scripts/test_svbmc_pool_stack.py` reads the gpyreg
checkout from `PYVBMC_GPYREG_SOURCE` and the baseline from `BASELINE_DIR`,
and skips only when they are unset, under the same no-skip rule; Torch
comes from the environment, or else from the baseline's `deps/`. A pool
too small for the repetitions at an `M` gives `floor(n / M)` of them, and
`prepare` prints and records the shortfall. The analyses that read the
raw pools beside the stacking's assembled cells (`svbmc_shrink_elbo.py`,
`svbmc_single_run_bias.py`, each with `--pool`, `--cells` and the release
trees' gpyreg as `--gpyreg-source`) run where the pools lie, in the
operator's account, as batch jobs after the stacking's finish; the
[campaign plan](svbmc-benchmark-campaign.md), which owns the release
gate, owns what they compute and how their outputs are read, and the
brief gives their commands.

## Resources and cost

Phase 6 measured each harness's tasks on the campaigns' node family
(2026-09-29, and the stacking's finish and the pools' analyses on
2026-10-01), from the smoke campaigns' accounting: one task per core,
which the accounting counts as one CPU; where an accounting counts both
threads of a core, the accounted figures double. The figures are in the
logs that `dev/scripts/runs/LOCAL.md` lists, except those of the
populations' and the pools' step jobs, which were read from the
accounting (`sacct -j <job>`); the operator's notes list the job ids.

| Campaign | Cases | CPU-hours |
|---|---|---|
| Pools | 2930 | about 225, from the first seed of every condition and 40 seeds of `student_D8_noise3_svbmc` |
| Population, per arm | 2400 | about 155 (before arm) and 165 (after arm), from seed 0 of every configuration |
| Stacking | 560 two-arm cells and 400 integrated-only cells | not measured whole; about 25 by the September laptop times doubled, which the `M = 32` cell below makes a low figure |

- Populations, seed 0 of every `production` configuration (before arm /
  after arm): `cigar_D15_exhaust` 20.8 / 22.4 minutes, about a fifth of
  the arm; `lumpy_D10_noise3_production` 10.0 / 10.8, so it needs no
  subset of its own; `student_D8_noise3_production` 8.1 / 9.1; every
  other configuration 8.1 minutes or less. The 24 cases add up to 1.56
  and 1.65 CPU-hours. Peak resident memory 430 MB.
- Pools, the first seed of every condition and 40 seeds of
  `student_D8_noise3_svbmc`: the student condition about 10.5 minutes a
  run, `multisensory_s1_D6_noise3_svbmc` 7.3, the ring 5.9, the others 4.3
  or less. Peak 375 MB.
- Stacking, `student_D8_noise3_svbmc` alone: a task of 16 repetitions at
  `M = 2`, both arms, 1.3 minutes and 0.75 GB; 4 repetitions at `M = 8`
  2.8 minutes and 1.0 GB; one cell at `M = 16`, both arms, 2.8 minutes and
  1.4 GB; one cell at `M = 32` 4.3 minutes and 3.1 GB. A campaign's task
  below `M = 16` holds 20 repetitions, so its task at `M = 8` takes about
  14 minutes by this one. S-VBMC's final entropy evaluation reduces its
  matrix of log densities by chunks of rows (`ca4e8259`), with results
  identical bit for bit, so the peak of a task at `M = 32` is the
  gradient steps of its `optimize()`, which build the whole matrix at 20
  draws per component: 3.1 GB here, 2.1 GB on the developer's machine.
  Other conditions were not measured. The `M = 16` and `M = 32` tasks are
  submitted as their own subsets (`CASES_SUBSET=M16`, `M32`) with their
  own `MEM`.
- Step jobs: a pool `verify` of 47 cases 10 s and 128 MB; a population
  `verify` of 24 cases 11 to 18 s and up to 450 MB; `rescore` of 48 cases
  42 s and 241 MB; `select` and `summarize` a few seconds. The stacking's
  `verify` of 4 tasks 9 s and 231 MB, and its `assemble` of their 22
  cells 31 s (7 s of CPU) and 417 MB; the release's stacking holds 200
  tasks and 960 cells.
- The pools' analyses, on the smoke pool and the 22 cells of the smoke
  stacking, run twice: `svbmc_shrink_elbo.py` 16 and 17 s, 0.5 and
  0.6 GB; `svbmc_single_run_bias.py`, which scores every selected run
  (38 here, 2560 in the release's pools) at about 1 s a run, 45 and 54 s,
  0.7 and 1.0 GB.
- The environment check: 19 minutes and 1.5 GB at `2356f12d`, 25
  minutes and 1.5 GB at `edd3f633`.

The limits, with a margin for seeds slower than those measured and, for
the finishing steps, for campaigns a hundred times the size of the smoke
ones: `POP_TIME` 1 hour, `CIGAR_TIME` 1.5 hours, `POP_MEM` 2 GB;
`POOL_TIME` 45 minutes, `POOL_MEM` 2 GB; `STACK_TIME` 1 hour and
`STACK_MEM` 4 GB, `M16_TIME` 30 minutes and `M16_MEM` 4 GB, `M32_TIME`
1 hour and `M32_MEM` 6 GB; `POP_VERIFY_TIME` 2 hours and
`POP_VERIFY_MEM` 4 GB (2400 cases at about 0.75 s), `RESCORE_TIME` 3
hours and `RESCORE_MEM` 4 GB (4800 cases at about 0.9 s); the pools'
`verify` fits the default limits (2930 cases at about 0.2 s), and so
does the stacking's; `ASSEMBLE_TIME` 1 hour and `ASSEMBLE_MEM` 4 GB;
for the pools' analyses `SHRINK_TIME` 1 hour, `SINGLE_RUN_TIME` 3 hours
and `ANALYSIS_MEM` 4 GB; `CHECK_TIME` 1 hour, `CHECK_MEM` 4 GB. The
[brief](release-gate-handoff.md) tabulates them.

## Records and hand-back

- The harness checkout and every source tree are real git repositories at
  named commits, clean, and do not move while a campaign's tasks run; a
  fix to the harness is a new commit and a new campaign directory.
- Campaign directories live outside the checkout or under its gitignored
  `dev/scripts/runs/`, where the submission's clean-tree check does not see
  them.
- **The raw campaign directory** holds everything: the manifest with its
  `site` block and `pip freeze`, the task logs, the records with their host
  parts, the Slurm accounting. It is archived as soon as the finish passes,
  with zstd, in parts below GitHub's 2 GiB limit per asset: the September
  population campaigns hold 0.4 to 1.1 MB per case, the boost pickle
  dominating, so an arm is 1 to 2.6 GB. The parts go to a draft release of
  this repository per campaign (`release-gate-pools-<date>`,
  `release-gate-population-after-<date>`,
  `release-gate-population-before-<date>`, `release-gate-stacking-<date>`),
  uploaded with `gh` or from the operator's machine. An archive holds site
  details and the operator's paths, so it stays in its draft release,
  which only the repository's collaborators see; the public record is the
  redacted copies below. Nothing redacts an archive itself, so none is
  published: these draft releases stay drafts, since publishing one would
  make its archive public.
- **Public assets are built apart from the archives** (PI, 2026-09-28).
  The PyVBMC 1.5 release attaches the after arm's population and the
  pools, each as an archive of its redacted tracked copies and of every
  verified case's numeric files (the population's traces and posterior
  arrays, the pools' runs, `.npz` files that must hold numbers alone),
  completion record and other JSON files, redacted as the copies are,
  without the logs, the Slurm accounting, the `site` block and the
  pickles; each record ties its case's published files to the hashes of
  the files the campaign wrote (PI, 2026-09-30), and the stacking's
  tracked copies hold every cell already. `hpc/campaign_public.sh`
  (`campaign_public.py`) builds such an asset in parts below 2 GiB, in the
  operator's account after the redaction, since it redacts with that
  account's username and home, and refuses, writing nothing, a file that
  still holds what the copies may not; the asset records the code that
  built it and the exemptions it applied, and its check applies the rules
  that need no campaign and fails an asset that other code built. An
  asset is a record of its campaign and no campaign directory that
  `verify` accepts or the stacking reads, which 1.5 keeps (PI,
  2026-10-01). The operator uploads the parts to
  the draft release `release-gate-public-<date>`, from which the release
  attaches them. The before arm and the raw archives stay in their draft
  releases; the replay fingerprints and the traces of the earlier golden
  references, of which the developer's machine holds the only copies, are
  kept in a draft release of their own as a backup, not published.
- **The tracked copies are redacted.** They go under
  `dev/experiments/release_gate_<date>/` (`pools/`, `population_after/`,
  `population_before/`, `stacking/`, with a README) in a pull request to
  `dev-next`, as in September: the manifests, the verification reports, the
  summaries and, for the populations, every verified case's record,
  sidecar and boost report and the after arm's rescored metrics (about 30
  to 45 MB per arm), so that the assessment can be redone from a fresh
  checkout. The redaction (`hpc/campaign_redact.sh CAMPAIGN_DIR OUT_DIR`,
  `campaign_contract.py redact`) writes them: a command the operator runs
  on the login node in the account that ran the campaign, after the
  finish, since it removes that account's username and home, which it
  reads from its own process. Each harness declares its tracked copies in
  its manifest (`tracked_copies`). Hostnames reduce to the node family
  (`NODE_FEATURE`, which the copies keep so that the family of every run
  stays checkable) or to `login`, a host part's lists of node features to
  that feature alone (PI, 2026-09-29) and its platform to the operating
  system and machine type (PI, 2026-09-30); the CPU model stays. Paths
  under a path setting take its name; the identity trees and the
  campaign's parent get names of their own (`$<TREE>_TREE`,
  `$CAMPAIGN_PARENT`), under the home or not (PI, 2026-09-30); other paths
  under the home become `~`. The `site` block, the `pip freeze` paths and
  the Slurm accounting stay in the archive, and partitions are replaced in
  the fields that hold one. It refuses when a path or command setting, a named directory, the
  operator's username, the home, a hostname or another node feature
  remains anywhere in its output (paths, the home and the settings as
  substrings, the username and hostnames as whole names, hostnames in any
  case, also where a name the copies write begins or ends one, a host list
  that names a hostname, the domain of a hostname, a feature in the quotes
  a list of them holds, and unquoted in a README that `--check`
  searches), when an absolute
  path outside the system prefixes remains unnamed, when a username or
  hostname equals a name the copies write, and when a case is in flight.
  `--allow STRING` exempts a benign hit, which `redaction.json` records
  with its count; `redaction.json` records the SHA-256 of each copy beside its
  source's, which the population analysis checks when it reads the
  copies, and lists the cases not verified. `.gitattributes` keeps the
  bytes of `dev/experiments/release_gate_*/` as written. The operator
  writes the README of `release_gate_<date>/` and checks it with
  `campaign_redact.sh CAMPAIGN_DIR --check FILE`.
- **The README goes to the PI before the pull request opens** (PI,
  2026-10-01): the repository is public, so a pull request is public
  from the moment it opens, and the README check does not know every
  form a site detail can take in prose.
- **The outputs of the pools' analyses** (`svbmc_shrink_elbo.py`,
  `svbmc_single_run_bias.py`) are no campaign's tracked copies, and
  their `sources.json`, `summary.json` and `added.json` hold the
  operator's paths and the nodes' names. They go back whole as one more
  asset of the stacking's draft release, and no file of them enters the
  operator's pull request (PI, 2026-10-01).
- A machine that holds a raw directory lists it in its
  `dev/scripts/runs/LOCAL.md`.
- The tracked records of the September pool
  (`experiments/svbmc_pool/pool_20260914/`) and the `sources.json` of the
  analyses built on it predate the redaction and are kept as written; that
  pool's README says so. So are the other records of that campaign: the
  default partition of its scripts (`hpc/svbmc_pool_submit.sh`,
  `hpc/svbmc_pool_finish.sh`, and the operator's guide's account of them)
  and the node families that the
  [campaign plan](svbmc-benchmark-campaign.md) names in its record of the
  pool (PI, 2026-09-30).

## Phases

Executors: the orchestrator keeps the design, the integration and every
phase that runs on Turso; Opus sub-agents implement Phases 2 to 5 on the
branch `feat-slurm-campaigns` cut from `dev-next`, with their tests, the
formatting hooks and conventional commits. Phase 2 comes first, since the
harnesses build on its contract module; Phases 3, 4 and 5 then run in
parallel, each in its own worktree on a branch of `feat-slurm-campaigns`
that merges back into it, and at most one of them runs tests at a time.
Nothing heavy runs on the developer's machine beyond the tests a phase
owns. Every phase ends with its acceptance check.

### Phase 1: the cluster survey and the environment

In the PI's account. **1a** (done, 2026-09-25), on the login node:
`sinfo -s`, the partitions' limits and features
(`sinfo -o "%20P %10l %10L %6D %c %m %f"`), `MaxArraySize`, `MaxJobCount`,
`DefMemPerCPU` and `SelectTypeParameters` from `scontrol show config`, the
association (`sacctmgr show assoc`), a node's `CPUTot`, `CoresPerSocket`,
`Sockets` and `ThreadsPerCore`, the tools and the Python and conda modules,
and the storage quota. **1b**, where the campaigns will run: the same
survey, with the per-user limits, the node features that select the
campaigns' node family, and whether `sacct` answers from a compute node
(a batch job); then the conda environment and the source trees: the
harness checkout, gpyreg at `v1.4.0` and at `v1.2.1`, the package at
`f91fdf0` as a detached worktree, and the S-VBMC baseline. The first
environment built from the direct pins (`campaign_env.sh build`) is
frozen into `campaign_requirements.txt`: every package and Python to its
patch release (`python -m pip list --format=freeze --exclude-editable`),
leaving out `pip`, `setuptools` and `wheel`, which conda installs with
the interpreter. The file is committed and pulled into the harness
checkout before the first submission, since the submission refuses a
package the file does not pin. The site-specific results go into the
operator's notes; what bears on the design revises "The cluster".
**Acceptance:** the
environment built from the frozen file passes the environment check, and
every source tree is clean at its commit.

### Phase 2: the generic driver

The driver section, with tests against stub `sbatch`, `squeue`, `sacct`
commands, and the part of the contract the three harnesses share, as one
module, `dev/scripts/campaign_contract.py`, which each harness imports: the
claim (its creation, the takeover on a requeue, the staleness from the
accounting, its removal), the source identity (each tree's commit, clean
state and import path, the hashes of the harness files, the imported
versions), the host part, the completion record and its check, the
comparison of the environment with the pinned requirements, and the
reconciliation of `verify`'s states. `campaign_requirements.txt` holds the
direct requirements pinned; Phase 1b freezes the first environment built
from them into it, dependencies included. **Acceptance:** the tests cover the index mapping, the chunk
count, the subsets, every refusal, the reconciliation of in-flight and
missing cases, and a task that exits early on a completed case; `bash -n`
passes on every script.

### Phase 3: the pool harness

The pool harness section, with tests in `dev/scripts/test_svbmc_pool_run.py`.
**Acceptance:** the tests cover `--case`, the early exit, the claim (fresh,
live, stale, requeued, a failed query, and a refusal that leaves the
case's files untouched), the identity and host fields and every `verify`
state; a case run by the array-mode worker and by `run` on the same
machine gives identical artifacts.

### Phase 4: the population harness and the arm comparison

The population and arm-comparison sections, with tests in
`dev/scripts/test_population_run.py` and `test_analyze_population_run.py`.
**Acceptance:** a case run locally in each arm passes the two-tree identity
and records the right trees in its sidecar; the array-mode worker and `run`
give identical artifacts for one case on one machine; the rescoring of
after-arm cases reproduces their in-run metrics exactly; the analysis runs
on two small array-mode campaigns of 100 paired seeds.

### Phase 5: the stacking harness

The stacking section. **Acceptance:** a result assembled from tasks equals
an unsplit run's on a small pool; the subsets of a condition at one `M` are
disjoint; a recreated baseline verifies by content.

### Phase 6: smoke campaigns on Turso

In the PI's account. First the environment check: the test modules of
the contract, the driver and the three harnesses with their analyses, run
from the branch as a batch job in the campaign environment, pass with no
test skipped. Then each harness through the driver on the `smoke` suite
or a few cases, and the heaviest cases of each (`cigar_D15_exhaust`,
`lumpy_D10_noise3_production`, a stacking cell at `M = 32`): a canary; a
resubmission of a finished range; a task cancelled while running, which
`verify` must report as missing, then resubmitted; a task killed without
warning (by SIGKILL, or by exceeding its memory), which `verify` must
report as interrupted, then resubmitted, which must take over the stale
claim; a duplicate submission of a running case, which the claim must
refuse; a finish with and without `--allow-running`; the redaction, the
public asset (`campaign_public.sh`, whose tests run only its Python and
parse the script) and the archive. The before arm's worker on a few cases of `f91fdf0`, the boost
capture among them. **Acceptance:** every check above passes, and the
accounting (`sacct`, `seff`) gives `TIME` and `MEM` per harness and the
per-case times that replace the estimates above.

### Phase 7: the operator guide, the brief and a review

`dev/scripts/hpc/README.md` gains the operator's guide to the new driver
and the three harnesses, in site-neutral terms; the site values stay in the
operator's notes. The September scripts no longer drive the pool harness,
whose worker takes `--case`; the guide and the `scripts/hpc/` entry of
`dev/README.md` present them as the record of that campaign. The
redaction ("Records and hand-back") is implemented here, before the
hand-back needs it. The limits of the verify and finishing jobs of the
populations (a `verify` of 2400 cases, a `rescore` of 4800) and of the
pools are settings the guide names for every finish, whose values come
from the accounting of Phase 6. A
brief in the manner of [the September one](svbmc-pool-handoff.md) tells the
postdoc what to run, in what order, and what to hand back. It gives the
`TIME` and `MEM` of every job that Phase 6 measured, and names without
their values the settings particular to the site (the node feature, the
modules, the paths), which the operator's notes hold. A doublecheck by fresh reviewers reads the
branch against this plan. **Acceptance:** the review's findings are fixed
or ruled on, and the branch merges into `dev-next` after the PI's review.

### Phase 8: the campaigns

On the PI's instruction, once no algorithmic work on 1.5 remains
(decision 13) and Phase 1b's survey holds.
Each campaign runs whole in one environment on one node family, and the two
population arms run together, so that they differ in their code alone; the
pools and then the stacking follow, from a clean checkout at the release
commit. Each campaign is handed back as "Records and hand-back" says.
**Acceptance:** every campaign verifies with no case missing or failed, or
with the missing and failed cases named and ruled on.

### Phase 9: the replay fingerprints

On the developer's machine, one process, after the after arm's population
exists. One seed (seed 0) of each of the 24 `production` configurations,
with `reference_promote.py fingerprints`, which runs them through
`golden_replay.py`: about 70 to 80 minutes for the 23 configurations
whose laptop medians the golden references give, and about 11 minutes
for `lumpy_D10_noise3_production`, which they hold no runs of, by its
seed 0 on the cluster in Phase 6; about 1.5 hours in all (the 24 seeds 0
took about 1.6 CPU-hours per arm on the cluster). Each run is judged
against the after arm's accuracy envelope (`golden_replay.py --sidecars`,
which reads the per-configuration directories of the after arm or of its
tracked copies, counts verified cases only, and fails for a configuration
it finds no population of). Together the runs are a coarse check that the
machine and the cluster sample the same distribution: one run per
configuration can lie outside its envelope as the population's own runs
sometimes do (4.2 % of the runs of `reference_990_20260913` lie outside
their configuration's envelope), so the set is judged, not each run (PI,
2026-09-28). The six seeded gate runs are recorded twice by
`seeded_gate_runs.py`, whose record carries the commit, the gpyreg source,
the thread settings and the host, with `performance_calibration="off"` so
that no calibration cache enters them, and compared with the gate script's
own `compare`; they have no population, so no envelope applies. The set
goes to the promotion of the new reference
(`reference_promote.py prepare`), with the populations.
**Acceptance:** every fingerprint completes with finite metrics; no more of
them lie outside their envelopes than the after arm's own rate of runs
outside theirs makes plausible, a binomial tail of at least 0.01 (at the
rate of 4.2 %, four of 24 pass and five do not); and each gate run
reproduces bit for bit.

## Worklog

- 2026-09-25: plan written from the PI's decisions of the day. Phase 1a
  ran on the cluster: read-only queries in one session, and one query of
  the storage quota; its site-specific results are in the operator's notes.
  Three independent reviewers read the plan and its pointers the same day
  (privacy, the plan, the pointers); this text includes their findings.
- 2026-09-25: the PI ruled on the open questions: 100 seeds for
  `cigar_D15_exhaust` and `lumpy_D10_noise3_production` (decision 2), the
  September records kept as written ("Records and hand-back"), and the
  release code (decision 13), and a doublecheck for Phase 7's review.
  The claim reads the Slurm accounting in place of `squeue`, the
  environment is pinned, and the environment check moved from Phase 1b to
  Phase 6, where the harnesses' test modules first run without skipping.
  Phase 1b runs on whichever installation of the cluster is up; the
  scripts need no change between them.
- 2026-09-25: Phase 2 on `feat-slurm-campaigns` (`c1a77df2` to
  `a020cfc0`): `campaign_contract.py`, the four driver scripts, the pinned
  direct requirements, stub Slurm commands and a stub harness; 108 tests
  pass on the developer's machine with none skipped. The interface choices
  the plan had left open are written into "Operator settings", "The
  campaign contract" and "The driver", among them the tag-first case line,
  the exit codes, the stopped and interrupted cases and the queued tasks
  counted as in flight. The Linux-only paths (the host part's affinity,
  topology and BLAS, `conda activate`, the real output of `sacct`,
  `squeue` and `scontrol`, an external SIGTERM) first run in Phase 6.
- 2026-09-25: Phases 3, 4 and 5 ran in parallel on branches of
  `feat-slurm-campaigns` and merged into it. Phase 3: the pool harness on
  the contract, 78 tests; the September copy verifies 1100 of 1100 with
  every difference equal to the local record's, and the pool readers
  reproduce their tracked outputs. Phase 4: array mode, the two-tree
  identity and the rescoring, 76 tests; one short case of each arm ran
  through the harness, the before arm at `f91fdf0` with its boost capture,
  and the rescoring reproduced the after arm's in-run metrics exactly.
  Phase 5: the stacking as tasks, 44 tests; assembled tasks equal the
  single-process run's cells, rows and summaries. Two contract rules came
  from them: a stale claim without a record is interrupted whether or not
  the task left files, and `git status` lines are recorded unaltered. The
  rescoring calls the metrics' fixed-seed generators, since only they
  reproduce the in-run metrics. Phase 4 also found that
  `population_run.validate_case`, which `analyze_population_run.py` and
  `reference_join.py` call on the September campaigns, compared the
  transformers of the captured posteriors by their `__dict__`. Since
  `e610479e` (2026-09-20) every copy of a transformer builds its own
  bounded-transform functions, so the comparison fails on every case,
  bounded or not, the captured posteriors being separate copies; it
  compares their data instead.
- 2026-09-25: the PI asked for S-VBMC's final entropy evaluation to be
  chunked (`feat-svbmc-entropy-chunks`, merged into `dev-next` as
  `ca4e8259`): the draws, the generator's stream and every result are
  unchanged bit for bit (checked against the previous code on every
  fixture group and on whole `optimize()` runs), and the peak of the final
  evaluation at 32 runs of 50 components in `D = 6` falls from about
  5.9 GiB to 351 MiB. The gradient steps keep the whole matrix: chunking
  them changes the order in which the gradient sums over rows.
- 2026-09-25: Phase 7's redaction and operator's guide on
  `feat-slurm-campaigns` (`1f9a3538`, `4682b1a2`): the redaction is a
  command run after the finish, not a finishing step, since it needs the
  operator's account and writes into a checkout; every harness's test
  module passes with none skipped. The guide leaves `TIME` and `MEM` of
  every job to the accounting of Phase 6.
- 2026-09-25/26: a doublecheck of the branch by five fresh reviewers (the
  contract and driver, the pool and stacking harnesses, the population
  harness, the redaction and the documents, and one that ran every test
  module on Windows and, for the first time, the contract, driver,
  population and pool modules under Linux in WSL, where the host part's
  affinity, topology and BLAS paths and an external SIGTERM worked). The
  PI had every must-fix and should-fix finding fixed, on four branches
  merged into `feat-slurm-campaigns`: a failure after the completion
  record no longer deletes the case; retired claims carry a key; a stopped
  rerun of a failed case reads as missing; the one-core check reads the
  core's threads and the job's cpuset; a failed `squeue` never reads as an
  empty queue and step jobs are recorded before the finish waits; `select`
  refuses a gap below a selected run and the stacking's `prepare` requires
  a verified pool; a semantic failure of a population case is a failed
  case; the rescoring is bound, resumable and tied to the after arm; the
  confirmatory family keeps its size; `PYVBMC_SOURCE` redirects the
  population harness alone; the redaction's check is a substring search
  that also refuses unnamed absolute paths, and `.gitattributes` keeps the
  tracked copies' bytes; the chunked S-VBMC entropy concatenates its
  chunks so that `torch.func.vmap` works (`dev-next`, `d4add1b8`).
- 2026-09-26: a short re-review (the code fixes, the documents against
  the code, and every test module on Windows and in WSL) found two
  must-fix problems. The pool worker and the stacking's original arm read
  the installed metadata of `pyvbmc` and `gpyreg`, which the campaign
  environment does not install; and the finishing of a halfway claim
  retirement let two workers hold one case. Both are fixed, with the
  should-fix findings, in `3e7c64e7` to `3f1573af`, `fb9fa3a0` and
  `c714d88d`: the retirement mark; whole-name matching of usernames and
  hostnames in the redaction, with `--allow`; `--allow-running` as a look;
  `give-up`; the accounting's allocation rows alone; the limit of the
  wait for a step job; tests that run the pool and stacking with neither
  package's metadata present. The PI set this as the last review round:
  what only the cluster can show goes to the smoke campaigns of Phase 6.
- 2026-09-26: the final tests on the developer's machine, at the merged
  branch (`c714d88d`, then `da25e5dc`), all with none skipped:
  - on Windows: the contract (133), driver (74), population (59), analysis
    (55), pool (76), honest-ELBO (14) and stacking (54) modules;
  - under Linux in WSL: the contract (133) and driver (74) modules.

  No package code has changed since `17cac66c`, at which the S-VBMC suite
  (276), `pyvbmc/testing/test_golden_replay.py` (67) and the oracle gate
  (12 of 12 exact) passed. Not run: the population, analysis and pool
  modules under Linux at the final commit, which passed there at
  `17cac66c`, and the harnesses in a Linux environment without PyVBMC and
  gpyreg, which the environment check of Phase 6 is.
- 2026-09-27: `dev-next` merged into the branch (`aeb783bb`), bringing
  `1f9c207a`, with which PyVBMC leaves the root logger to the application,
  and `1bde36eb`, with which `display="off"` prints warnings only; the
  package tree of the branch is `dev-next`'s at `3e6b813e`. The same
  modules pass again with none skipped: on Windows the contract (133),
  driver (74), population (59), analysis (55), pool (76), honest-ELBO
  (14) and stacking (54) modules, under Linux in WSL the contract (133)
  and driver (74). The workers' logs still receive PyVBMC's output: no
  harness, and nothing a worker imports, configures the root logger, so
  PyVBMC's handler writes to the worker's standard output, which the
  sequential supervisors' `<tag>.log`, a task's `slurm/%A_%a.out` (which
  holds standard error too) and the stacking's `original_arm.log` (the
  original arm points its standard output at standard error) receive. A
  process that imported the harness modules as a worker does and ran a
  VBMC too short to converge at `display="off"` wrote the caution and a
  timer warning to its captured log, its root logger without a handler
  before and after the run. A converged run at `display="off"`, the
  harnesses' setting, leaves only the harness's own lines in its log.
- 2026-09-27/28: `dev-next` merged into the branch again (`52b16c96`),
  bringing the release sweep's fixes and records. The same modules pass
  again on Windows with none skipped: the contract (133), driver (74),
  population (59), analysis (55), pool (76), honest-ELBO (14) and stacking
  (54) modules. The tools of Phase 9 and of the promotion that follows it
  are on the branch, since the promotion reads the array-mode campaigns of
  `population_run.py` and the wrapper records its identity through
  `campaign_contract.py`, and both run after the branch's merge:
  `seeded_gate_runs.py` (`d6645999`), the wrapper of the six seeded gate
  runs, and `reference_promote.py` (`4efee154`), the promotion
  (`dev/README.md` describes both). Their test modules pass (8 and 7 tests);
  the promotion's runs a whole promotion, from the fingerprints to the
  rewritten documents, on two arms of `normal_D2` at seeds 0–2. The wrapper
  recorded the real six runs twice at `4efee154` from clean checkouts,
  with gpyreg at `d96d0d9`, in 9 minutes each: the 138 arrays of the two
  recordings are identical. That run checks the wrapper and enters no
  record; the records wait for the release code. The promotion copies the
  after arm's verified sidecars into `dev/golden/baseline/` as the earlier
  promotions did, so that `golden_trace.py compare dev/golden/baseline` and
  the default `--sidecars` of `golden_replay.py` read the new reference
  unchanged; the copies repeat the tracked copies' sidecars, about 12 MB for
  2400 cases of about 5 KB. The reference's local traces directory,
  `<name>_fingerprints`, holds its fingerprints and the gate runs; the
  population's traces stay in its archive.
- 2026-09-28: a doublecheck of the two tools by three fresh reviewers (the
  wrapper, the promotion's logic, the documents and templates), and the
  PI's rulings on what it raised. A fingerprint has no trace to be
  identical to, so golden_replay's exemption of identical runs never
  applies to it, and 4.2 % of the runs of `reference_990_20260913` lie
  outside their own configuration's envelope: a check of each of the 24
  fingerprints would refuse a correct promotion about two times in three,
  so the set is judged against the population's own rate (Phase 9). The
  code the checks compare leaves out the package's tests and S-VBMC, and
  the script leaves `dev/scripts` for its record at the promotion ("The
  populations"). What the release publishes is built apart from the draft
  releases ("Records and hand-back"). The fixes: `prepare` takes only the
  after arm's redacted tracked copies, since `publish` copies their
  sidecars into the repository; the after arm's trees must be clean;
  every step checks first that it runs the after arm's code and gpyreg;
  `publish` computes all it writes before the first write; tracked JSON is
  hashed with LF line endings; `publish` also rewrites the
  `regenerate_baseline.sh` paragraph of `dev/README.md` and the previous
  population's example in `golden_replay.py`, and names the documents to
  carry by hand; the generated READMEs say how another machine makes
  fingerprints of its own, who sees the draft archive and that the PI
  accepted the assessment. The wrapper takes an output directory in the
  checkout only where git ignores it, checks both recordings for clean
  trees, refuses a gpyreg that no checkout tracks, maps a crashed
  recording's exit code to 1, hashes the gate script with LF line endings
  and re-checks each run's digest. The test modules pass (11 and 9 tests),
  the promotion's now on redacted copies of arms made at a stand-in site.
- 2026-09-28: the public assets ("Records and hand-back"):
  `campaign_public.py` and `hpc/campaign_public.sh`, with
  `test_campaign_public.py` (6 tests, on a population campaign at a
  stand-in site, redacted; among them an asset in parts, a damaged part, a
  case's JSON that is no tracked copy, redacted, and a trace that holds
  text, refused). A trace of `golden_trace.py`, like a pool run's `.npz`,
  holds numbers alone; a pool run's JSON is no tracked copy, so the asset
  redacts it, which is why the asset is built in the operator's account.
  The operator's guide gives the step.
- 2026-09-28: Phase 1b's first session, on the cluster's new
  installation, in the PI's account; the site's values are in the
  operator's notes. The survey: `sacct -n -X -P -j <job>_<task> -o State`
  prints one state word; `squeue -h -r -j <job> -o "%i %T"` prints a line
  per task while the job runs, nothing with exit 0 once it has ended and
  the controller still holds it, and an error with exit 1 once the
  controller has dropped it; all three commands, `scontrol show node -o`
  among them, answer from a compute node, so the claim judges staleness
  there as the contract assumes. A task with `--cpus-per-task=1
  --hint=nomultithread` ran on both hardware threads of one core, its
  job's cpuset equal to them, and the accounting counted one CPU. One
  feature selects the campaigns' node family ("The cluster"). The five source
  trees are clean at their commits. The first environment built from the
  direct pins is frozen into `campaign_requirements.txt` (`f0d2789e`), and
  the environment built from the frozen file matches it. The freeze found
  that `check-env` did not know conda-forge's own Python packages as
  conda's: `setuptools` and `wheel` carry no `INSTALLER` file, and `pip`
  and `packaging` a `direct_url.json` naming their feedstock's build
  directory, which the check read as a source tree installed from a path.
  It now reads the environment's `conda-meta` records (`551e194f`), and
  the frozen file leaves `pip`, `setuptools` and `wheel` to conda. The
  first run of the environment check then failed in the tests alone,
  where the batch job's surroundings reached them: the site's conda on
  the job's `PATH`, which the driver's test of a submission without
  `CONDA_SETUP` must not find (1 test); the operator's exported
  `NODE_FEATURE`, which the population tests' `prepare` recorded and
  against which `verify` checked their hand-made records, and a fresh
  interpreter that imported PyVBMC without gpyreg (12 tests and 3
  errors); and a detached automatic `git gc` that repacked a test
  repository while the pool test copied it (1 test). The contract,
  analysis and honest-ELBO modules passed; the session ended before the
  stacking module's result. The fixes (`5c7b409d`) keep the population
  and stacking tests outside any Slurm task and away from the operator's
  settings, as the pool tests already were, take conda's directories and
  variables out of the driver's test environment, and run the tests' git
  without automatic `gc`. On the developer's machine the four modules pass
  with none skipped, with those surroundings reproduced (a failing
  `conda` first on the `PATH`, `NODE_FEATURE` and a Slurm job id
  exported).
- 2026-09-29: the first run's stacking module had failed one test
  (`test_an_interrupted_task_is_taken_over`) for the same reason, the
  job's Slurm variables and `NODE_FEATURE` reaching the worker and the
  `prepare` of its campaign, which `5c7b409d` removes. The environment
  check, run again at `2356f12d`, passes:
  the contract (135), driver (74), population (59), analysis (55), pool
  (76), honest-ELBO (14) and stacking (54) modules, none failed or
  skipped, the checkout clean after it; the job took 19 minutes and
  peaked at 1.5 GB. Phase 1b's acceptance is met. Step 3's canaries went
  out the same day, as smoke campaigns in the PI's account: the two
  population arms at the `production` allocation, seeds 0–99, with the
  subset `canary` submitted (the first seed of every configuration); the
  pools at the release allocation, with the first case of every
  condition and seeds 1001–1039 of `student_D8_noise3_svbmc`, from which
  the stacking campaign of the next entry took its runs.
- 2026-09-29: Phase 6's smoke campaigns, steps 3 and 4; the measurements
  are in "Resources and cost". The canary cases completed and verified:
  24 in each population arm, whose finishes ran `verify` and `summarize`
  and, in the after arm, `rescore`, which rescored the 48 cases and
  reproduced the after arm's in-run metrics exactly (the before arm's,
  from the code of `f91fdf0`, differ, as the rescoring expects); the
  before arm's `verify` loaded the boost captures that `f91fdf0`'s worker
  wrote. And 47 in the pools, of which `select --target 32` made the
  student condition stackable. A stacking campaign on that condition ran
  one task per `M` (2, 8 and 16 with both arms, 32 with the integrated
  arm); the four completed, and the campaign was neither verified nor
  assembled (the pickup point). The failure
  paths ran on a pool campaign of the `smoke` suite (12 cases): a task
  cancelled while it ran cleaned up and read as missing; a task whose
  worker was killed by SIGKILL read as interrupted, and its resubmission
  took over the stale claim; a duplicate submission of a running case was
  refused with 75; the finish with `--allow-running` counted the cases in
  flight and queued and exited 3 on the missing and interrupted ones, and
  without it named `ARRAY=6-8`; a resubmission of the finished range
  exited at once; the full finish verified 12 of 12 and wrote the
  archive, which matches its SHA-256; the redaction and the public asset
  passed their own checks. On the site's Slurm `scancel --full
  --signal=KILL` is an ordinary cancel, whose SIGTERM the worker handles,
  so the kill was a step inside the task's allocation (`srun
  --jobid=<job_id> --overlap kill -9 <pid>`, from the claim's `job_id` and
  `pid` fields; `job_id` is the task's own job, which for every task but
  an array's last differs from the array's `job`). Three findings, fixed
  on the branch: an independent search of the redacted copies found the
  site's other node features in a host part's lists, which the redaction
  kept, and the PI ruled that the copies keep the campaign's feature
  alone and the CPU model (`e293893c`). After the fix the redaction of the
  failure-paths pool, again, and of both population arms, for the first
  time, reduced the lists; its own search found none of the site's
  details in the copies or in the two public assets built from them (the
  failure-paths pool's and the after arm's), and an independent search,
  which showed at most ten hits per directory, found only benign ones
  (citations in the targets' descriptions, a chance byte pair in a
  compressed array); a whole search remains (the pickup point). The
  guide's pool canary submitted case 1 a second time, and that task
  exited 75 (`e293893c`), and `campaign_public.sh --check` refused a
  directory holding two assets, as the guide builds them (`eb7538de`). On
  the developer's machine the contract (135), public asset (6),
  promotion (9), population (59), analysis (55), pool (76), honest-ELBO
  (14) and stacking (54) modules pass at `e293893c`, and the public
  asset's (7) at `eb7538de`; the driver module, whose
  `test_array_refusals` failed once while another pytest process ran
  beside it, passes whole (74) at `6ff30208`, run alone.
- 2026-09-29: a doublecheck of the session's commits by three fresh
  reviewers (the environment check and the tests' isolation; the
  redaction and the public asset; the records and their privacy), and
  its fixes. The operator's guide gave one of the site's node features as
  its example of a quoted feature; it gives a placeholder. The two
  entries above claimed a verification of the smoke stacking campaign,
  which was not made, and more of the redaction's second pass than was
  done; they are corrected in place. `redaction.json`'s rules, the
  `redact` docstring and `dev/README.md` state the reduction of the node
  features; `campaign_public.py check` also fails on a part that no
  listing names; a test marks conda's distributions in a stand-in prefix
  through its `conda-meta` record, and another refuses a leftover feature
  in a JSON list; the honest-ELBO tests leave out the Slurm variables and
  the operator's settings as the other modules' do; the driver tests read
  a `PATH` entry they cannot open as holding no conda. The plan's
  estimates give way to Phase 6's measurements ("Resources and cost"),
  and decision 2, "The cluster", the driver's environment, Phase 1b and
  Phase 9 state what Phases 1b and 6 found. With the fixes the contract
  (136), public asset (7), honest-ELBO (14) and driver (74) modules pass
  on the developer's machine, none skipped. Three checks remain for the
  next cluster session (the pickup point).
- 2026-09-30: gpyreg 1.4.0 released, and PyVBMC requires it
  (`pyproject.toml`'s minimum and CI's pin, PR #178); `dev-next` merged
  into the branch (`fb034058`). "What runs where", Phase 1b and the
  operator's guide name gpyreg `v1.4.0` (`682585f`) for the after arm,
  the pools and the stacking. It reproduces 1.3.3 bit for bit on every
  oracle output, the GP-fit-history captures and the default golden
  replays (the [roadmap](modernization-roadmap.md), pickup 15). The
  cluster's trees hold `v1.3.3`; the pickup point gives the clone of
  `v1.4.0` and the environment check with it.
- 2026-09-30: a doublecheck of the whole branch before the merge, at
  `86477269`, by seven fresh reviewers, read-only (the contract; the
  driver; the population harness; the redaction, with a search of every
  tracked file for the site's details; the pools and the stacking; the
  promotion and the gate runs; the documents), while every harness test
  module, the gate runs' and `pyvbmc/testing/test_golden_replay.py` ran
  there on the developer's machine: 562 passed, none skipped. It found one
  leak: the redaction's comment on the username's boundary and a test of
  whole names held a cluster account's username and home path, since
  2026-09-26; stand-ins replace them (`33070164`), and the branch's
  history, which this plan, its worklog and `LOCAL.md` cite by hash,
  stays. The PI's rulings on the rest (2026-09-30): a copied or unpacked
  pool is verified and selected again where it lies before it is stacked,
  in code; the source trees and the campaign's parent take names of their
  own under the home too; the pools' public asset holds every verified
  case's completion record; a host part's platform keeps its operating
  system and machine type alone in the copies; the September campaign's
  records keep their partition and node families as written ("Records
  and hand-back"); and the branch merges after the PI's review, the brief
  following before Phase 8. The fixes:
  - the worker: a stop signal between its claim and its run is a stop,
    so a takeover's killed files go with the claim and the case is
    missing, not partial; `write_json` and the artifacts reach storage
    before the completion record; the clean-tree checks pass
    `--untracked-files=normal`, and `check-env` exempts a source tree but
    not a local archive (`b9ce5e5f`);
  - the finish takes the accounting again before the archive, which now
    requires an allocation row of every recorded job, the verify and
    finishing steps' included; `STEP_POLL` is above 0; the stub `sacct`
    answers with a row per requested job; `campaign_requirements.txt`
    keeps LF line endings in every checkout;
  - `--pair` compares the arms' node family and environment, and the
    population's `prepare` refuses a gpyreg tree that is not the release
    the package tree's `pyproject.toml` requires, exactly for an arm of
    other code and that release or a later commit for the release code;
  - the stacking's `prepare` refuses a pool whose verification or
    selection was made in another directory, and a selected run whose
    files differ from its completion record;
  - the redaction names trees and the campaign's parent under the home
    and reduces the platform; the public asset holds the completion
    records, records the code that built it and the exemptions it
    applied, and its check applies the rules that need no campaign and
    fails an asset that other code built;
  - `fingerprints` exits 0 once every run is made, and a metric of 0.0
    counts as finite;
  - the documents: the guide's environment check stops on a failed
    activation and prints its interpreter, its exit-code table gains 1
    without an error file and 2, its limits point at "Resources and
    cost" and the pools' `verify` runs at the default limits; this plan's
    settings table gives way to the guide's, its pickup point and
    checklist state the order the PI ruled, Phase 9's estimate is whole,
    and the pools' analyses have an owner.

  Left for after the merge and before Phase 9: the stubs running the
  batch script where it lies rather than as Slurm's spooled copy, the
  generated `golden/README.md`'s recipe for other machines, the
  promotion's refusal of a checkout that a tracked edit makes dirty,
  which comes only after the fingerprints' runs, the README check's
  blind spots (hostnames that begin a name the copies write, host lists,
  the site's domain, features in prose), the tests of the promotion's
  refusals and of the release grid, and the note on `--sidecars` in the
  `golden_replay.py` entry of `dev/README.md`, a passage the promotion
  rewrites from its template. For the next cluster session: the arm
  comparison (`analyze_population_run.py --arms`) on the smoke canaries,
  its only run on real before-arm output. Not done, with the reason: a
  task whose environment cannot be activated still exits 1 (a `source`
  in an `||` list runs without `errexit`, which the activation relies
  on), which the exit-code table now explains; the clean-tree checks
  keep the operator's global excludes file, whose removal could newly
  flag files that it hides in the S-VBMC tree the smoke finish needs.
  The review's optional findings not listed here await the PI's ruling.

  A fresh reviewer then read the fixes. The rules that each
  `redaction.json` records, the usage of `campaign_redact.sh` and the
  docstring of `redact` still gave the old order of the names and no
  platform rule; they give the new ones. The public asset is a record of
  its campaign and not a campaign directory that `verify` accepts, since
  its redacted `.json` artifacts no longer hold their records' hashes:
  the records tie each published file to its source hash, and the
  documents said more, which they no longer do. Whether the released
  pools should be verifiable and stackable from the asset, which would
  need the published records to hash the redacted files and the error
  files of the failed seeds published too, is for the PI. The asset's
  `built_by` hashes the modules with LF line endings, so that a Windows
  checkout checks an asset built on Linux; `check` reports a member that
  is no file and a `public.json` without `files`; and the guide, the
  population module's docstring, the task script's header, the
  requirements file's header, the stacking's docstrings and this plan
  describe the new checks.

  The test modules on the developer's machine, one at a time: on the
  fixes before that reviewer's, every harness test module, the gate runs'
  and `test_golden_replay.py`, 623 passed and none skipped, but for one
  fixture of the analysis module, which failed with a `MemoryError` while
  the machine ran short of memory (the module had passed whole, 55, on the
  same code before). After the reviewer's fixes the contract module passed
  (139), and at `c5f56bd3`, whose code is that of the fixes, the public
  asset's (9), analysis (55), pool (76) and driver (75) modules, none
  skipped, the checkout clean after them; so every harness test module
  passes on the fixes.
- 2026-10-01: the last checks of Phase 6 and the release's gpyreg, in the
  PI's account, with the harness checkout at `2356f12d` for the first and
  at `dev-next` (`edd3f633`) for the others.
  - The finish of the smoke stacking campaign verified its 4 tasks and
    assembled their 22 cells; its accounting gives the stacking's
    finishing limits ("Resources and cost").
  - The redaction ran again, into a new directory, on the failure-paths
    pool and both population arms, whose copies of 2026-09-29 predate
    the rules of 2026-09-30, and for the first time on the stacking
    campaign. All four passed: a host part holds the campaign's feature
    alone and `Linux x86_64`, and the trees and the campaign's parent
    carry their names. The public assets of the failure-paths pool and
    of the after arm, built from the new copies, pass
    `campaign_public.sh --check`, which fails the two assets of
    2026-09-29 as built by other code.
  - A search of the new copies (169 files) and of the unpacked assets
    (172 files) for the site's details, with no truncation, found the
    PI's surname in the citations that three targets' descriptions hold,
    and one chance byte triple in a compressed array; the assets' 60
    `.npz` files, decompressed, hold 1492 arrays of floats, integers and
    booleans and none of the strings.
  - The arm comparison ran on the canaries of the two smoke arms,
    validated their 24 cases each against the arms' verifications and
    wrote its assessment; with one seed per configuration its KS screen
    holds no test, and its confirmatory family of 96 tests rejects
    none.
  - gpyreg `v1.4.0` (`682585f`) is cloned beside `v1.3.3`, and the
    environment check with it passes at `edd3f633`: the contract (139),
    driver (75), population (60), analysis (55), pool (76), honest-ELBO
    (14) and stacking (54) modules, none failed or skipped, the
    interpreter the environment's and the checkout clean after it; the
    job took 25 minutes and peaked at 1.5 GB.
  - The two analyses of the pools, `svbmc_shrink_elbo.py` and
    `svbmc_single_run_bias.py`, ran on the cluster for the first time,
    as batch jobs on the smoke pool (38 selected runs) and the smoke
    stacking's 22 cells: once by hand and once by the block of the
    [brief](release-gate-handoff.md) as written, with the smoke pool's
    gpyreg and a directory of its own. The two passes give the same
    cells, summaries and added biases, and their runs differ in the
    recorded seconds alone.

  Two findings. The analyses' outputs hold site details: their
  `sources.json`, `summary.json` and `added.json` record the paths of
  the pool, the cells, the script and the interpreter, the node's name
  and its platform string, and one summary a login host's name. No
  harness declares them as tracked copies, so the redaction does not
  write them, and `campaign_redact.sh --check` refuses them (the
  operator's home, username and paths); their `.md` and `.jsonl` files
  hold none. How they are handed back was for the PI, whose ruling
  follows. And `campaign_public.py` names what an asset leaves out by
  `Path(name).suffixes`, which starts at the first dot of the file's
  name: the boost pickles of a configuration whose name holds a dot
  (`noise1.3`, `noise2.2`) are counted under keys such as
  `.3_production_seed0.boost.pkl`, one for each seed, beside
  `.boost.pkl`. The after arm's asset, with three such configurations
  at 100 seeds (two of them at `noise1.3`, which share their keys),
  would list 200 more keys in `public.json`'s `left_out` than the one
  that is meant; the files left out are the right ones.

  The brief is written, with the limits of "Resources and cost".

  The PI's rulings of the day, on what the campaigns' frozen checkout
  fixes at the launch. The analyses' outputs go back as an asset of the
  stacking's draft release alone, and the scripts stay as they are. The
  keys of `left_out` are fixed: `campaign_public.left_out_kind` names an
  artifact by what follows its case's name in the file's, with a test,
  and the public asset's module passes on the developer's machine (10).
  The pools' public asset stays a record that `verify` does not accept
  (the question the entry of 2026-09-30 left). The README check keeps
  its blind spots at the launch, and the operator sends the README to
  the PI before opening the pull request. The arm comparison's
  confirmatory family is the default, so neither arm's `prepare` takes
  `--confirmatory`. "Records and hand-back" and the brief say so.

  The PI named the release commit the same day: `ff3ed014`, the head of
  `dev-next` with the fix above and the brief. On the developer's
  machine the exact oracle check passes on it with gpyreg `v1.4.0`
  (`make_oracle_fixtures.py --check --exact`, one BLAS thread, 12 of
  12), and CI's full matrix, dispatched at it, passes in its nine jobs
  (Ubuntu, Windows and macOS under Python 3.10 to 3.12). Its package and
  harness modules are those of `edd3f633`, on which
  the environment check passed on the cluster; the operator runs that
  check again at the release commit in their own account.
- 2026-10-02 to 10-04: Phase 8. The operator ran the four campaigns on
  2026-10-02 from `ff3ed014`, in the brief's order, after the environment
  check at that commit (139, 75, 60, 55, 76, 14 and 54 tests, none
  skipped). Every case verified, none failed, went missing or was given
  up: 2400 in each population arm, whose `rescore` reproduced the after
  arm's in-run metrics exactly; 2930 in the pools, 320 selected in every
  condition with no shortfall; 200 stacking tasks of 960 cells, with no
  condition short of repetitions; and the two analyses covered 960 of 960
  cells and 2560 of 2560 runs. The first bulk submissions, four at the
  default `THROTTLE` of 200 (each arm's `cigar_D15_exhaust` subset and its
  rest), made nodes of the family fail their Slurm prolog and drain, four
  within six minutes and two more later. The operator held the arrays,
  cancelled the pending tasks and submitted again at 75 a submission,
  then without the two nodes that had failed twice; no node drained after
  that. The eight tasks that had started stopped on the SIGTERM, removed
  their claims and partial files, and left their cases to later tasks.
  The operator's guide gains a sentence on `THROTTLE`. Cost: about 180
  and 175 CPU-hours for the before and after arms, against the 155 and
  165 of "Resources and cost", 220 for the pools and 17 for the stacking;
  the longest population tasks, of `cigar_D15_exhaust`, took 34 and 31
  minutes, against 21 and 22 in Phase 6, and no task reached a limit. The operator sent
  the README of the hand-back to the PI before the pull request; its
  corrections were a count of pins, the file counts of the public assets,
  a reading of the pools' usable fractions, which went, and two figures
  of the cluster's configuration, which went too.

  On the developer's machine, before the merge (2026-10-04): every copy of
  the four campaigns has the SHA-256 and size its `redaction.json` records
  (14 421 files), and none lists a case not verified or an exemption; a
  search of every file of the hand-back found the hostnames `u8` and
  `login` alone, no absolute or home path, e-mail address, domain or Slurm
  variable, and the PI's surname in the targets' citations; the seven
  assets of the five draft releases, all drafts, have the sizes and
  digests the README gives; and `analyze_population_run.py --arms`
  validated the 2400 cases of each arm against their verifications and
  rescored metrics. The pull request merged into `dev-next` (`5924120b`).
  The arm comparison's record is
  `dev/experiments/release_gate_20261002_assessment/`.
- 2026-10-04: Phase 9, on the developer's machine, from a clean checkout
  at `dce4e18c`, whose package and run code are the release commit's
  (none of the paths the promotion compares differs from `ff3ed014`'s),
  with gpyreg `v1.4.0` and one BLAS thread. The six gate runs
  (`seeded_gate_runs.py run`, 16:22 to 16:30) reproduced bit for bit in
  their two recordings, 138 arrays of 138. The 24 fingerprints
  (`reference_promote.py fingerprints --after
  dev/experiments/release_gate_20261002/population_after`, 16:30 to
  17:13, 43 minutes against the 1.5 hours estimated) completed with
  finite metrics. One lies outside its envelope: `corr_D5`, on its
  evidence error, gsKL and MMTV (0.048, 0.012 and 0.018, against the
  population's fences of 0.033, 0.005 and 0.013), with 105 evaluations,
  inside the population's range. One of 24 is within what the after arm's
  own rate makes plausible (up to four pass), so the set meets the
  acceptance. The outputs are the machine's, which
  `dev/scripts/runs/LOCAL.md` lists ("The release gate's hand-back and
  Phase 9"); the promotion's `prepare` reads them.
- 2026-10-07: the PI accepted the assessment of the two arms,
  `dev/experiments/release_gate_20261002_assessment/`, whose
  `assessment.json` has the SHA-256, with LF line endings,
  `2625fc50cb035ccabe7e5ec50bc6ebc7743c515eec08240c15b361a14420898d`
  that the promotion's `--accepted-assessment` takes. None of its 96
  confirmatory tests rejects; its KS screen flags the evaluation counts of
  `cigar_D4` and `rosenbrock_D2_noise3_production`. The noisy Rosenbrock
  target's runs are usable 72 times in 100 against the before arm's 87,
  the cost of MATLAB's end of warm-up, which stays (2026-10-06); the
  promotion's record is to state that cost.
- 2026-10-07: the smaller fixes of 2026-09-30, on the branch
  `feat-slurm-small-fixes`:
  - the stub `sbatch` runs a step of the finish from a copy of the batch
    script in a spool directory of the job's own, and the driver's tests
    run each array task from such a copy, in the harness checkout that
    `campaign_submit.sh` submits from, as Slurm does; with the task
    script made to find `campaign_env.sh` by its own location, the
    finish's test fails;
  - `reference_promote.py`'s `fingerprints` and `replay` refuse a
    checkout with any tracked change before their runs, which would
    record it as dirty and which `prepare` and `publish` refuse;
  - the generated `golden/README.md` gives another machine the commands
    for fingerprints of its own: the check of its code against the
    reference's, `golden_replay.py` with the after arm's tracked copies as
    `--sidecars` (at the commits the promotion record names,
    `golden/baseline/` still holds the previous reference), the gate runs,
    and the rule by which the set is judged; its sentence on the
    fingerprints outside their envelopes takes the singular for one;
  - tests of the promotion's refusals that had none (the after arm's
    allocation, harness, options, package, trees and rescored metrics, the
    assessment's count of verified cases, the fingerprints' threads,
    budgets, configurations, failures, compared traces, sidecars, error
    files, packages, dirty checkouts and code, the record a replay needs,
    the gate runs' script, check, code and gpyreg, and `publish`'s checks
    of the prepared files, the documents' sources and the final replay),
    of the early refusal, and of the module's constants against the
    tracked after arm (the whole `production` suite at seeds 0-99);
  - the note on `--sidecars` of `golden_replay.py`'s entry in
    `dev/README.md`, with its template and guard and the script's
    docstring: the population's sidecar of a replayed seed is also the
    reference for the run's semantic finals where `--baseline` holds no
    trace of it, as for the fingerprints, and only a `verification.json`
    limits the envelopes to the verified cases;
  - the redaction's check refuses a hostname that a name the copies write
    begins or ends (the node `u8-12` of the family `u8`), which the
    blanking of those names had cut short, a Slurm host list that names a
    hostname, and the domain of a hostname that is a domain name, in the
    copies and in a README; in a README (`--check`), a node feature
    unquoted too, which its quoted `--allow` exempts in any letter case.

  A doublecheck by three fresh reviewers, read-only, found that searching
  the unblanked text alone lost a username that a dot joins to a name the
  copies write (`login.jdoe`), which the check again finds by searching
  both texts, and smaller points, all fixed: the expansion of a host list
  in arbitrary text (bounded, and only where its head can begin a
  hostname; the bound missed a list with a bracket that names no host,
  which the second doublecheck below found), a domain inside a host list,
  the allowance of a feature in a README, the working directory of the
  tests' array tasks, the number of the generated sentence above, the
  tests' phrases and the documents. At the branch's last
  commit every harness test module and `pyvbmc/testing/test_golden_replay.py`
  passed on the developer's machine, one at a time with one BLAS thread
  and none skipped (the pool, stacking and honest-ELBO modules with
  `PYVBMC_GPYREG_SOURCE` at gpyreg `v1.4.0` and the stacking's
  `BASELINE_DIR`, the population module with the 1.0.4 trees): the
  contract (141), driver (75), public asset (10), population (81),
  analysis (61), pool (76), stacking (55), honest-ELBO (14), promotion
  (18) and gate runs' (11) modules, and the replay's (67). The code that
  the promotion compares with the after arm's is unchanged.
- 2026-10-07: the promotion of the after arm to the golden reference
  `reference_2400_20261007` (the roadmap's pickup 19): `prepare` passed
  every check, with `corr_D5`'s fingerprint the one of 24 outside its
  envelope (a probability of 0.60 at the population's rate of 0.0375); the
  replay of the five defaults at `948bac0d` was identical; `publish`
  rewrote `dev/golden/baseline/` and the documents that name the current
  reference. The script and its test module left `dev/scripts/` with the
  promotion's commit, the script into the record as
  [`promote.py`](../golden/promotion_20261007/promote.py), which is what
  this plan's `reference_promote.py` names from then on. The record's
  [README](../golden/promotion_20261007/README.md) gives the evidence and
  the commands.
- 2026-10-07: a second doublecheck, by three fresh reviewers, read-only,
  of the promotion (its record and data, the documents and code it
  touched) and of the first doublecheck's fixes. The data held: every
  sidecar of `dev/golden/baseline/`, the summary, the local fingerprints
  and gate runs and the record's reports match the manifest and
  `validation.json`, and the sidecars are the after arm's tracked copies
  byte for byte; and over 70 000 generated texts the check's search of
  usernames and hostnames found what the searches before and after its
  first change found together, with no hit counted twice. Fixed: a host
  list with a bracket that names no host (`n[1-999999999]y[2-1]`) passed
  the bound, while its expansion walked the other brackets, up to most of
  an hour for one such list; a host list whose hosts hold a forbidden
  hostname before a domain or a suffix or after a dotted prefix
  (`n[1-3].example.net`, `x.n[1-3]`) escaped where only the short
  hostname is forbidden, and is now refused as the hostname search finds
  a hostname in its hosts; a quoted feature in a README counted as two
  hits; the whole-name rule in the header of `campaign_redact.sh`; the
  record's README (Arm 3's end of warm-up, the measures on which the noisy
  Rosenbrock target is worse, the table's columns, Phase 9's commands);
  the generated `golden/README.md`'s count of the runs that reached their
  budgets, its sentence on the fingerprints and the cluster's seed 0, and
  the population's rate; the flags of the two record-only scripts; and
  the documents that still named `reference_990_20260913` as the active
  reference or the TODO's closed items as open.
