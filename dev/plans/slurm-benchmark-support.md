# Slurm benchmark support for the release gate

Created: 2026-09-25. Status: **reviewed by the PI on 2026-09-25; Phases 2
to 5 and the code of Phase 7 implemented on `feat-slurm-campaigns`**, which
has not merged into `dev-next`. What it assumes of the cluster rests on a
survey of the cluster on 2026-09-25 and on the records of the September
pool. The harness sections below describe each harness as `dev-next` holds
it and what the branch gives it.

## Live checklist

- [x] The PI's review of this plan (2026-09-25).
- [x] Phase 1a: the cluster survey (2026-09-25).
- [ ] Phase 1b: the survey where the campaigns run, the environment and
  the source trees.
- [x] Phase 2: the generic Slurm driver (2026-09-25).
- [x] Phase 3: the pool harness (2026-09-25).
- [x] Phase 4: the population harness and the arm comparison (2026-09-25).
- [x] Phase 5: the stacking harness (2026-09-25).
- [ ] Phase 6: smoke campaigns on Turso.
- [ ] Phase 7: the operator guide, the brief, an independent review; merge
  into `dev-next`.
- [ ] Phase 8: the campaigns, on the PI's instruction.
- [ ] Phase 9: the replay fingerprints on the developer's machine.

## Purpose and scope

The release gate of PyVBMC 1.5 is a set of campaigns too large for one
machine: the S-VBMC run pools and their stacking comparison (the TODO's
"Final large-scale check") and the reference populations that replace the
golden references after the port review (the TODO's "The golden references
after the port review"). This plan owns how they run on a Slurm cluster:
submission, resources, resumption, verification, collection, the
provenance of every run and what of it enters the repository. It also owns
the small set of exact replay traces that stays on the developer's
machine.

What other documents own, and what this plan hands them:

- The [campaign plan](svbmc-benchmark-campaign.md) owns the design and the
  acceptance criteria of the S-VBMC comparison; its decision 14 records the
  release pools and stacking decided here. This plan hands it the verified
  release pools and the assembled stacking results.
- The golden references item of `dev/TODO.md` owns the assessment of the
  new reference and its promotion, by the method of the
  [population plan](final-population-benchmark.md), with a promotion record
  under `dev/golden/` as for `reference_990_20260913`. This plan hands it
  the verified before and after populations, their rescored metrics and
  the replay fingerprints.
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
| Reference population, after arm | Turso | the release code, gpyreg `v1.3.3` (`98ab5a4`) | the 24 configurations of the `production` suite, seeds 0–99: 2400 runs |
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
   configurations as well: `cigar_D15_exhaust`, about a quarter of a
   population's cost, and `lumpy_D10_noise3_production`, which has never
   run and whose seed count is revisited only if its first smoke case
   takes hours.
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
    need the campaigns it reaches to run again.

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
  ratios come from one CPU model.
- **Physical cores.** Where Slurm counts hardware threads as CPUs,
  `--cpus-per-task=1` is one thread and two tasks can share a core, so
  every task passes `--hint=nomultithread`; the accounting may then count
  both threads of the core.
- **Short tasks.** Every task of this plan takes minutes, and `TIME` is set
  from the measured durations (Phase 6).
- **Network on the login node only.** Environments and source trees are
  built there, after any setup the site needs (`LOGIN_SETUP`); tasks need
  no network.
- **Batch jobs only.** Every computation, `verify` and the finishing steps
  included, is a batch job, so that every step is in the accounting and
  none depends on the operator's session.
- **Storage without backup.** Campaign directories live in the operator's
  home on the shared filesystem. Cluster storage is commonly neither backed
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

| Variable | Meaning |
|---|---|
| `HARNESS` | the harness the driver runs: `dev/scripts/svbmc_pool_run.py`, `population_run.py` or `svbmc_pool_stack.py` |
| `CAMPAIGN_ENV` | the prefix of the campaign's conda environment |
| `NODE_FEATURE` | the Slurm feature that selects the campaign's node family |
| `PARTITION` | optional, no default; `-p` is omitted when it is unset |
| `LOGIN_SETUP` | optional commands the environment script runs on the login node before a step that needs the network |
| `PYVBMC_SOURCE`, `PYVBMC_GPYREG_SOURCE` | the package and gpyreg source trees of the campaign (of the arm, for a population) |
| `BASELINE_DIR` | the original S-VBMC checkout and its Torch, for the stacking's original arm |
| `CASES_SUBSET` | optional, a named subset of the cases, submitted as its own array |
| `THROTTLE`, `TIME`, `MEM`, `ARRAY`, `SBATCH_EXTRA` | as in the September scripts |
| `CONDA_SETUP` | optional commands that define `conda`, such as loading a site module |
| `LOGIN_PROFILE` | the login profile the environment script reads when `module` is not defined; default `/etc/profile` |
| `VERIFY_TIME`, `VERIFY_MEM`, `FINISH_TIME`, `FINISH_MEM` | the limits of the verify job and of each finishing step; default `01:00:00` and `2G` |
| `ARCHIVE_PART_SIZE` | the largest part of the archive; default `1900M` |
| `STEP_POLL` | seconds between the finish's accounting polls of a step job; default 30 |

The driver passes them to the harness's `prepare`, which records them in
the `site` block of the raw manifest. Only the raw campaign directory holds
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
  completion record; exits 64 on a line or a directory that is not the
  harness's; compares its source identity with the manifest's and refuses
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
  by case, each with its index: verified; failed (an error file); in
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

A campaign directory holds, beside each case's artifacts,
`records/<tag>.complete.json`, `claims/<tag>` and `<tag>.error.txt`, and
`subsets/<name>.txt`, `slurm/` (the task logs, the job ids, the
accounting) and `tmp/` (`TMPDIR`).

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
  node failure, out of memory, preempted) is stale: the worker renames it
  to `claims/<tag>.stale.<job>_<task>.<key>`, `<key>` being eight hex
  digits of the claim's token, so that of two workers only one succeeds,
  and then creates its own; a worker that finds the retired name already
  holding that very claim finishes the retirement.

A refusal never takes the failure path, which deletes the case's files.
A claim made outside Slurm, by `run` on a workstation, is live while its
process runs on the host that made it, and on any other host until the
operator removes it. The accounting keeps a job's state after the job ends. `squeue` answers
only for the jobs the controller still holds and can answer with an
error for one it has dropped, as it does when the query fails, so it
cannot tell an ended task from an unreachable controller. Phase 1b checks
that the accounting answers from a compute node; where it does not, every
claim is live, and a resubmission waits for the operator to clear the
stale ones.

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
`gpyreg` through `importlib.metadata`, and in the before arm the installed
distributions are the release's while the imported trees are `f91fdf0` and
gpyreg 1.2.1. The identity and the sidecars record each tree's commit and
import path, and label a version read from the installed distribution as
such.

## The driver: `dev/scripts/hpc/`

New scripts, `campaign_env.sh`, `campaign_submit.sh`, `campaign_task.sbatch`
and `campaign_finish.sh`, drive any harness that meets the contract. The
September scripts (`svbmc_pool_env.sh`, `svbmc_pool_submit.sh`,
`svbmc_pool_task.sbatch`, `svbmc_pool_finish.sh`) stay as they are, since
that campaign's records cite them. From them the new scripts keep: the
refusal of a dirty tree (now of every source tree), `cases.txt` written
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
  `dev/scripts/hpc/campaign_requirements.txt`, which pins every package the
  environment takes from PyPI, dependencies included, so that the
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
  every recorded task ended.
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

On `dev-next` the harness has `cases`, a standalone `worker` and `verify`,
but not the contract: its worker takes `--label L --seed S` and has no early exit (the
September task script holds it); its compared identity lacks cma, the data
directory (which the multisensory conditions read) and the clean state of
the whole harness checkout; its host part lacks the CPU model, the node
features, the BLAS library and the affinity; and `verify` has no in-flight
state. It gains:

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
  verification it selected from; `profile_run.py` among the hashed
  harness files.

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
`svbmc_honest_elbo.py`) default on `dev-next` to the September gpyreg path
and check nothing; on the branch they take the gpyreg source as a required
argument and check it against the pool's manifest. `svbmc_headline_numbers.py` takes its
directories and labels as arguments instead of the dated names it holds.

### The populations: `population_run.py`, `golden_trace.py`

On `dev-next`, `population_run.py` runs a manifest's cases one fresh
process at a time under one supervisor, which alone writes `launch.json`, `status.json` and
`finished.json`; its worker needs `launch.json` and records no failure. It
gains:

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
  does not pass for it), and records its commit. On `dev-next` the
  identity requires the package inside the harness checkout, which would
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
`reference_990_20260913`) and `DEFAULT_CONFIGS` (which holds the golden
label `rosenbrock_D2_noise1`). They change with the promotion, which the
golden references item owns, together with `AGENTS.md` ("Trajectories"),
the `golden_replay.py` entry of `dev/README.md` and `dev/golden/README.md`;
the promotion itself needs a script in the manner of
`golden/promotion_20260913/promote.py`, since `reference_join.py` extends a
reference and refuses any overlap with it.

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
  report labels them so.
- **Binding.** `rescore` runs only with the after arm's source identity,
  requires the after arm to reproduce its in-run metrics exactly, resumes
  from per-configuration work files, and records the SHA-256 of what it
  writes in `rescoring.json`, which the analysis checks. `summarize`,
  `rescore` and the analysis refuse a `verification.json` older than its
  directory, so the before arm is finished completely before the after
  arm's finish, and the after arm is finished again after any later finish
  of the before arm. `prepare --pair` refuses two arms with the same
  `pyvbmc` commit, or with different harness commits, harness files or
  environment versions.
- **The tool.** `analyze_population_run.py` needs a signed-rank test valid
  for 100 pairs (on `dev-next` its exact enumeration refuses more than 62),
  a
  verification of the reference arm against that arm's own
  `verification.json` in place of the 870-entry manifest of
  `golden/noisy_extension_20260907/` that it asserts, and a reader for
  array-mode campaigns, which write no `launch.json`, `finished.json` or
  `comparison.md`.

### The stacking: `svbmc_pool_stack.py`

On `dev-next` the comparison runs in one process. It verifies the original
baseline at the paths its record holds, the Windows paths of the machine
that ran the campaign, at every start, even for the integrated arm alone.
It cannot resume: a new run truncates `cells.jsonl`, and `results.json` is
written at the end. A run split by condition or `M` and merged with
`--summarize-only --from-results` takes the single-run rows from its first
file only and draws its bootstrap in another order than an unsplit run. Its
repetitions draw independent subsets, which can overlap. It gains:

- `prepare`, whose manifest holds the pools with their selections, the `M`
  values and repetition counts, the seed, the arms, the baseline's record
  and the identity; `cases`, one line per task; a `worker` with the claim;
  and `verify`.
- **Tasks.** A task computes all repetitions of one condition and `M` for
  `M` ≤ 8, and one cell for `M` = 16 and 32, writing one result file per
  cell. Each task runs the harness's `warm_up` before its first timed
  cell, so that first-call costs stay out of the timings of criterion 4,
  and runs both arms of a cell, the order alternating from cell to cell as
  it does today, so that a cell's runtime ratio is measured on one node.
- **Assembly**, a finishing step: `results.json` built from the cell files
  in their canonical order, the single-run rows computed from the pool,
  and the bootstrap drawn in that order, so that no partial results are
  ever merged.
- **Disjoint subsets** as decision 6 describes.
- **A verified pool.** `prepare` stacks a pool only when its
  `verification.json` passed and its `selection.json` was made from that
  verification and agrees with it, and records each pool's verification;
  it refuses a dirty harness checkout without `--allow-dirty`; the worker
  exits 64 on a line that is not a task of the manifest.
- **The baseline on the cluster**: the upstream S-VBMC at `13a78f6` and
  CPU Torch 2.14.0 in `BASELINE_DIR`, recreated as
  `experiments/svbmc_pool/baseline_environment.json` describes and verified
  by the SHA-256 of the committed content of every `svbmc/*.py` it records
  (`files_sha256_committed`; its `files_sha256` are those of a Windows
  working tree whose line endings git converted) and the Torch version,
  not by its paths. The integrated arm's environment holds the same Torch.
  A task of the integrated arm alone neither needs nor checks it.

Its test module `dev/scripts/test_svbmc_pool_stack.py` reads the gpyreg
checkout from `PYVBMC_GPYREG_SOURCE` and the baseline from `BASELINE_DIR`,
and skips only when they are unset, under the same no-skip rule; Torch
comes from the environment, or else from the baseline's `deps/`. A pool
too small for the repetitions at an `M` gives `floor(n / M)` of them, and
`prepare` prints and records the shortfall. The scripts that
read the assembled results (`svbmc_shrink_elbo.py`,
`svbmc_single_run_bias.py`) run as batch jobs.

## Resources and cost

Estimates from the September records; Phase 6 measures them.

| Campaign | Cases | CPU-hours on Turso |
|---|---|---|
| Pools | 2930 | about 220, from the September medians per condition |
| Population, per arm | 2400 | about 220, without `lumpy_D10_noise3_production`, which has never run |
| Stacking | 560 two-arm cells and 400 integrated-only cells | about 25, without the per-task overhead |

About 685 CPU-hours in all, before `lumpy_D10_noise3_production` in both
arms and the stacking's per-task overhead (Torch imports, the reference
draws of each condition, loading the pool); where the accounting counts
both threads of a core, the accounted figure doubles. The population
estimate adds the laptop medians of the 16 noiseless configurations
(`golden/baseline/summary.md`, 30 minutes for one seed of each), the
production-budget runs of 2026-09-18/19 for noisy Rosenbrock, logistic
regression and Student D8, and 1.5 times the paper-budget medians for the
timing and multisensory configurations, which ran at the production budget
without a tracked record of their time; that makes about 110 laptop hours
per arm, doubled for the cluster. `cigar_D15_exhaust` alone takes about
27.5 of them. The stacking estimate doubles the September laptop times
(8.1 hours for the two-arm cells, 15 minutes for the integrated cells at
`M` = 3 and 5, 237 minutes at `M = 32`). Phase 5 found one `M = 32` cell
taking about 150 s at three optimization steps on the developer's
machine, so the stacking may cost more than this estimate; Phase 6
measures it. S-VBMC's final entropy evaluation reduces its matrix of log
densities by chunks of rows (`ca4e8259`), with results identical bit for
bit, so its peak at `M = 32` is about 350 MiB; the peak of an
`optimize()` at `M = 32` is then its gradient steps, about 2.1 GB, which
build the whole matrix at 20 draws per component. The `M = 16` and
`M = 32` tasks are submitted as their own subsets (`CASES_SUBSET=M16`,
`M32`) with their own `MEM`, which Phase 6 sets from their accounting.

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
  published.
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
  stays checkable) or to `login`, paths under a path setting to its name
  and paths under the home to `~`; the `site` block, the `pip freeze`
  paths and the Slurm accounting stay in the archive. The identity trees
  and the campaign's parent get names of their own (`$<TREE>_TREE`,
  `$CAMPAIGN_PARENT`), and partitions are replaced in the fields that hold
  one. It refuses when a path or command setting, a named directory, the
  operator's username, the home or a hostname remains anywhere in its
  output (a substring search, hostnames in any case), when an absolute
  path outside the system prefixes remains unnamed, and when a case is in
  flight; `redaction.json` records the SHA-256 of each copy beside its
  source's, which the population analysis checks when it reads the
  copies, and lists the cases not verified. `.gitattributes` keeps the
  bytes of `dev/experiments/release_gate_*/` as written. The operator
  writes the README of `release_gate_<date>/` and checks it with
  `campaign_redact.sh CAMPAIGN_DIR --check FILE`.
- A machine that holds a raw directory lists it in its
  `dev/scripts/runs/LOCAL.md`.
- The tracked records of the September pool
  (`experiments/svbmc_pool/pool_20260914/`) and the `sources.json` of the
  analyses built on it predate the redaction and are kept as written; that
  pool's README says so.

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
harness checkout, gpyreg at `v1.3.3` and at `v1.2.1`, the package at
`f91fdf0` as a detached worktree, and the S-VBMC baseline. The first
environment built from the direct pins (`campaign_env.sh build`) is
frozen into `campaign_requirements.txt`, every package and Python to its
patch release (`python -m pip list --format=freeze --exclude-editable`),
and the file is committed and pulled into the harness checkout before the
first submission, since the submission refuses a package the file does
not pin. The site-specific results go into the operator's notes; what
bears on the design revises "The cluster". **Acceptance:** the
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

In the PI's account. First the environment check: the three harnesses'
test modules, run from the branch as a batch job, pass with no test
skipped. Then each harness through the driver on the `smoke` suite
or a few cases, and the heaviest cases of each (`cigar_D15_exhaust`,
`lumpy_D10_noise3_production`, a stacking cell at `M = 32`): a canary; a
resubmission of a finished range; a task cancelled while running, which
`verify` must report as missing, then resubmitted; a task killed without
warning (by SIGKILL, or by exceeding its memory), which `verify` must
report as interrupted, then resubmitted, which must take over the stale
claim; a duplicate submission of a running case, which the claim must
refuse; a finish with and without `--allow-running`; the redaction and the
archive. The before arm's worker on a few cases of `f91fdf0`, the boost
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
pools are set explicitly in the guide, from the accounting of Phase 6. A
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
with `golden_trace.py`: about 70 to 80 minutes by the laptop medians, and
`lumpy_D10_noise3_production`, which Phase 6 times. Each run must lie
inside the after arm's accuracy envelope (`golden_replay.py --sidecars`,
which reads the per-configuration directories of the after arm or of its
tracked copies, counts verified cases only, and fails for a configuration
it finds no population of), a coarse check that the machine and the cluster sample the same
distribution. The six seeded gate runs, wrapped so that their record
carries the commit, the gpyreg source, the thread settings and the host,
and run with `performance_calibration="off"` so that no calibration cache
enters them, are recorded twice and compared with the script's own
`--compare`; they have no population, so no envelope applies. The set goes
to the golden references item's promotion, with the populations.
**Acceptance:** every fingerprint lies inside its envelope, and each gate
run reproduces bit for bit.

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
