# Campaigns on a Slurm cluster

The scripts here run a campaign of VBMC runs, or of S-VBMC stacking
cells, as Slurm job arrays on a cluster. There are two sets.

- **The driver of the release gate**, `campaign_*`, runs any harness that
  meets the campaign contract of `campaign_contract.py`: the reference
  populations (`population_run.py`), the S-VBMC run pools
  (`svbmc_pool_run.py`) and the stacking comparison
  (`svbmc_pool_stack.py`). Its design, and the reasons behind each rule
  below, are in
  [plans/slurm-benchmark-support.md](../../plans/slurm-benchmark-support.md).
  The [operator's guide](#the-operators-guide) says how to run the
  campaigns with it.
- **The September pool scripts**, `svbmc_pool_*`, generated the S-VBMC
  run pool of 2026-09-14, whose records cite them. They no longer run
  against the pool harness, whose worker takes a case line (`--case`)
  where they pass a label and a seed, and they are kept as the
  [record of that campaign](#the-september-pool-scripts), which names the
  values of the site it ran on.

The driver's scripts hold no value particular to a site: the node
feature, the partition, the modules, the paths and everything else a site
needs are settings, whose values the operator keeps in their own notes,
outside this repository. The operator's guide names each setting and
never its value.

## The operator's guide

### The scripts

Each script's header documents it in full.

- `campaign_env.sh`: the environment of every process of a campaign.
  Sourced, it activates the conda environment `CAMPAIGN_ENV`, exports
  single-threaded BLAS and the source trees, unsets `PYTHONPATH` (saying
  so), so that no path of the calling shell reaches a task, and puts
  `TMPDIR` under the campaign directory. Run as
  `bash dev/scripts/hpc/campaign_env.sh build`, it builds that
  environment.
- `campaign_submit.sh CAMPAIGN_DIR [prepare flags]`: prepares a campaign
  once (the harness's `prepare`, with the flags), writes its case list
  once, and submits the cases, or a subset or the indices `ARRAY` names,
  as job arrays in chunks below the site's `MaxArraySize`.
- `campaign_task.sbatch`: one array task, which runs one case through the
  harness's `worker`; or, submitted by the finish, one step on the whole
  directory.
- `campaign_finish.sh CAMPAIGN_DIR [--no-archive] [--allow-missing]
  [--allow-running]`: the accounting, the queue check, `verify` and the
  harness's finishing steps as batch jobs, then the archive; with
  `--allow-running`, a look at the campaign as it stands, which stops
  after `verify` and its report.
- `campaign_redact.sh CAMPAIGN_DIR OUT_DIR [--path NAME=PATH ...]
  [--allow STRING ...]`: the tracked copies of a finished campaign,
  redacted for the repository; with `--check FILE ...` in place of
  `OUT_DIR`, the same search in files it did not write.

The submission and the finish run on the login node, from the harness
checkout (the clone of this repository whose code the campaign runs), and
refuse to go on when a source tree is dirty, when the environment differs
from `campaign_requirements.txt`, or when a fixed setting differs from the
one the campaign was prepared with. Every computation, `verify` and the
finishing steps included, is a batch job.

### Settings

| Setting | Meaning | Fixed for the campaign |
|---|---|---|
| `HARNESS` | the harness, relative to the checkout: `dev/scripts/population_run.py`, `dev/scripts/svbmc_pool_run.py` or `dev/scripts/svbmc_pool_stack.py` | yes |
| `CAMPAIGN_ENV` | the prefix of the campaign's conda environment | yes |
| `NODE_FEATURE` | the one Slurm feature that selects the campaign's node family, a name of letters, digits and `_.-` and never an expression; every job requests it with `-C` | yes |
| `PYVBMC_SOURCE` | the package tree of a population arm whose code is not the harness checkout's (the before arm's worktree); unset otherwise | yes |
| `PYVBMC_GPYREG_SOURCE` | the gpyreg checkout of the campaign, or of the arm | yes |
| `BASELINE_DIR` | the original S-VBMC checkout, for the stacking comparison | yes |
| `PARTITION` | `-p` of every job; omitted when unset | no |
| `LOGIN_SETUP` | commands that give the login node the network (see below) | no |
| `CONDA_SETUP` | commands that define `conda`, such as loading a site module or sourcing a Miniforge installation's `etc/profile.d/conda.sh` | no |
| `LOGIN_PROFILE` | the login profile the environment script reads when `module` is not defined; default `/etc/profile` | no |
| `CASES_SUBSET` | a named subset of the cases, submitted as its own array | no |
| `ARRAY` | the case indices to submit, as `1`, `17,233` or `2-1100`; default every case of the campaign or of the subset | no |
| `THROTTLE` | the tasks of one submission that run at once; default 200 | no |
| `TIME`, `MEM` | `--time` and `--mem` of each task; default `00:30:00` and `2G` | no |
| `SBATCH_EXTRA` | further `sbatch` arguments of every job, split into words at spaces, tabs and newlines, with no quoting and no pathname expansion | no |
| `VERIFY_TIME`, `VERIFY_MEM`, `FINISH_TIME`, `FINISH_MEM` | the limits of the verify job and of each finishing step; default `01:00:00` and `2G` | no |
| `STEP_POLL` | the seconds between two looks at the accounting while the finish waits for a step job; default 30 | no |
| `STEP_WAIT_LIMIT` | the seconds after which the finish stops waiting for a step job; default 86400, a day, and 0 for no limit | no |
| `ARCHIVE_PART_SIZE` | the largest part of the archive; default `1900M` | no |

`prepare` records every setting in the `site` block of the campaign's
manifest. A later submission and the finish refuse a fixed setting whose
value differs from the manifest's; the others may change from one
submission to the next, and `slurm/jobs.txt` records each submission's:
its job id, `array=`, `offset=`, `subset=`, `throttle=`, `time=`, `mem=`,
`partition=`, the date, and `sbatch_extra=`, `conda_setup=` and
`login_profile=` quoted as bash's `printf %q` quotes them, `-` standing for
an unset value.

`LOGIN_SETUP` is what a step that reaches the network from the login node
runs first: the environment's build runs it itself, and before any other
such step (the clones below, `gh`, `git push`) run it in your shell with
`eval "$LOGIN_SETUP"`.

Keep the site's values in a file of your notes that exports them, outside
every checkout, and source it in each shell you work in: `CAMPAIGN_ENV`,
`NODE_FEATURE`, and `PARTITION`, `CONDA_SETUP`, `LOGIN_SETUP` and
`SBATCH_EXTRA` where the site needs them. Keep one more such file per
campaign, with its `HARNESS` and its source trees, and work on each
campaign in a shell of its own, so that the fixed settings of one
campaign never reach another's commands. The commands below use a few
names of their own for values of your notes: `TREES`, the directory that
holds the source trees; `RUNS`, the directory that holds the campaign
directories; and the limits of [Limits](#limits).

### The source trees

Each tree is a clone of its repository at a named commit, with nothing
changed in it. Clone them on the login node, with `LOGIN_SETUP` run first:

```bash
# The harness checkout, at the release commit: the latest commit of
# dev-next when the campaigns launch (the plan's decision 13).
git clone https://github.com/acerbilab/pyvbmc "$TREES/pyvbmc"
git -C "$TREES/pyvbmc" checkout --detach <release commit>

# gpyreg at v1.3.3 (98ab5a4), for the after arm, the pools and the stacking.
git clone https://github.com/acerbilab/gpyreg "$TREES/gpyreg-v1.3.3"
git -C "$TREES/gpyreg-v1.3.3" checkout --detach v1.3.3

# gpyreg at v1.2.1 (9e70e6b), for the before arm.
git clone https://github.com/acerbilab/gpyreg "$TREES/gpyreg-v1.2.1"
git -C "$TREES/gpyreg-v1.2.1" checkout --detach v1.2.1

# The package at f91fdf0, the before arm's code, as a detached worktree.
git -C "$TREES/pyvbmc" worktree add --detach "$TREES/pyvbmc-f91fdf0" f91fdf0

# The original S-VBMC 0.1.1 at 13a78f6, the stacking's original arm.
git clone https://github.com/acerbilab/S-VBMC "$TREES/S-VBMC"
git -C "$TREES/S-VBMC" checkout --detach 13a78f6
```

Confirm each commit with `git -C <tree> rev-parse --short HEAD`. The
original S-VBMC checkout is `BASELINE_DIR`, and its arm imports the
campaign environment's Torch, the `+cpu` build of 2.14.0 that the
requirements pin; on a machine whose environment holds no Torch, the
recreation that `dev/experiments/svbmc_pool/baseline_environment.json`
records installs that build from the PyTorch CPU index beside the
checkout. A process of the original arm verifies the checkout by content
against that record: the commit, a clean tree, the SHA-256 of every file
of `src/svbmc` and the Torch version.

What checks the trees: the submission and the finish refuse a tree that is
not the top of its checkout or that holds any uncommitted or untracked
change (they print `git status --short` of it), and print each tree's
commit. `prepare` records in the manifest the source identity: each
tree's commit and clean state, the SHA-256 of the harness modules, the
targets module and its data, and the versions of Python, NumPy, SciPy, cma
and, where imported, Torch. Every worker compares its own with it and
refuses a difference (exit 78), so no tree may move until the campaigns
that use it are done. Campaign directories live outside every checkout,
under `RUNS`.

### The environment

Every process of every campaign runs in one conda environment, built on
the login node from the harness checkout's `campaign_requirements.txt`:
Python from conda-forge at the version its one `# python==` line pins (a
line of its own, from the first column; the build refuses a file with
none or two), `zstd` for the archive, `gh` for the hand-back and `git`,
which every worker runs on its compute node for its source identity, and
every package from PyPI at its pinned version. PyVBMC and gpyreg are not
installed in it; they come from the campaign's source trees, and the
versions that the records read from installed metadata
(`installed_metadata_versions`, in the identity's `imports` and in the
pool artifacts' host part) are null for both. Build it from
the harness checkout, with the site's settings exported and
`CAMPAIGN_ENV` naming a prefix that does not exist yet:

```bash
bash dev/scripts/hpc/campaign_env.sh build
```

The build prints the installed versions as pins and ends with the check
that every submission and every finish repeats:

```bash
"$CAMPAIGN_ENV/bin/python" dev/scripts/campaign_contract.py check-env \
    --requirements dev/scripts/hpc/campaign_requirements.txt
```

It passes when every pinned package is installed at its version and every
installed package is pinned, the ones conda installed aside.

**The frozen requirements.** The release commit holds a frozen
`campaign_requirements.txt`, which pins every package the environment
takes from PyPI, dependencies included, and Python to its patch release,
since the submission refuses an installed package that the file does not
pin. The file is frozen once, before the launch (the plan's Phase 1b), in
a clone of the branch that the release commit comes from, never in a
campaign's harness checkout: an environment built from the direct pins
the file first holds gives the pins and the patch release,

```bash
"$CAMPAIGN_ENV/bin/python" -c 'import platform; print(platform.python_version())'
"$CAMPAIGN_ENV/bin/python" -m pip list --format=freeze --exclude-editable
```

which replace the direct pins and the `# python==` line's version. Keep
the `--extra-index-url` line, from which the `+cpu` build of Torch comes,
and one `# python==` line, and say in the file's header that it is frozen.
Run the check above on the frozen file, commit it and push it to the
branch, so that the release commit holds it. Every environment of the
campaigns is then built from the frozen file, in whatever account, and
holds the same libraries.

### The environment check

The test modules of the contract, the driver and the three harnesses run
as one batch job on the campaigns' node family, from the harness checkout,
with the release trees: `REPO` the harness checkout, `PYVBMC_GPYREG_SOURCE`
the gpyreg v1.3.3 checkout, `BASELINE_DIR` the S-VBMC checkout, and
`PYVBMC_SOURCE` unset. The subshell keeps those settings, and the
`CAMPAIGN_DIR` whose `tmp/` becomes the job's `TMPDIR`, out of the shell
you work in:

```bash
(
    export REPO=$TREES/pyvbmc PYVBMC_GPYREG_SOURCE=$TREES/gpyreg-v1.3.3 \
        BASELINE_DIR=$TREES/S-VBMC CAMPAIGN_DIR=$RUNS/environment_check
    unset PYVBMC_SOURCE
    mkdir -p "$CAMPAIGN_DIR"
    cd "$REPO"
    read -r -d '' -a extra <<< "${SBATCH_EXTRA:-}" || true
    sbatch -C "$NODE_FEATURE" --hint=nomultithread \
        ${PARTITION:+-p "$PARTITION"} --cpus-per-task=1 \
        --time="$CHECK_TIME" --mem="$CHECK_MEM" \
        --output="$CAMPAIGN_DIR/%j.out" --export=ALL \
        ${extra[@]+"${extra[@]}"} <<'EOF'
#!/bin/bash
set -uo pipefail
cd "$REPO"
# Activates the environment, with TMPDIR under $CAMPAIGN_DIR.
source dev/scripts/hpc/campaign_env.sh
status=0
for module in campaign_contract campaign_driver population_run \
    analyze_population_run svbmc_pool_run svbmc_honest_elbo \
    svbmc_pool_stack; do
    python -m pytest "dev/scripts/test_$module.py" -q -rs \
        -p no:cacheprovider || status=1
done
git status --porcelain
exit $status
EOF
)
```

`SBATCH_EXTRA` is split as the driver splits it. The pool, honest-ELBO and
stacking modules generate short pools (the stacking module also stacks the
shipped S-VBMC fixtures), and `test_population_run.py` makes two short
real VBMC runs; the contract, driver and analysis modules run on stub
commands and hand-made records. The check passes
when the job exits 0, every module's summary line counts tests passed and
none failed or skipped (`grep -n skipped` on the output prints nothing;
`-rs` gives the reason of any skip), and `git status` printed nothing. A
module that skips proves nothing about the environment. The before arm's
trees are checked by its canary, which runs whole cases of `f91fdf0`
through the harness.

### The campaigns

The campaigns run in the order of the plan's Phase 8: the two population
arms together, then the pools, then the stacking comparison, which stacks
the pools' runs. Every command runs from the harness checkout, in the
campaign's own shell, with the site's settings and the campaign's
exported. Name each campaign's directory as its tracked copies will be
named: `$RUNS/population_after`, `$RUNS/population_before`,
`$RUNS/pools`, `$RUNS/stacking`.

The first submission of a campaign prepares it: the flags after the
campaign directory go to the harness's `prepare`. A campaign that has been
prepared takes no flags. A task whose case already has a completion record
exits at once, so a range that includes finished cases may be submitted
again. `squeue -u "$USER"` shows the queue. A submission that `sbatch`
refuses, or whose job id it does not print, stops with the `ARRAY=` still
to submit; the chunks before it are submitted and recorded
([When something goes wrong](#when-something-goes-wrong)).

#### The two population arms

The 24 configurations of the `production` suite at seeds 0 to 99, 2400
cases in each arm, with `HARNESS=dev/scripts/population_run.py`:

| Arm | `PYVBMC_SOURCE` | `PYVBMC_GPYREG_SOURCE` |
|---|---|---|
| before | `$TREES/pyvbmc-f91fdf0` | `$TREES/gpyreg-v1.2.1` |
| after | unset (the harness checkout's own package) | `$TREES/gpyreg-v1.3.3` |

Both arms use the same harness checkout, environment and node family, so
that they differ in their code alone. Prepare the before arm first: the
after arm's `--pair` names it, and refuses a campaign that is not its
other arm (another allocation, options or confirmatory family, another
harness commit, other harness files or environment versions, or a package
tree at the after arm's own commit). Both arms are prepared with the same
flags; with `--confirmatory FILE` where the PI fixes a family other than
the default, in both. The canary is the subset `canary`, the first seed
of every configuration, which includes `cigar_D15_exhaust` and so takes
its limit.

```bash
# before arm's shell
CASES_SUBSET=canary TIME=$CIGAR_TIME MEM=$POP_MEM \
    dev/scripts/hpc/campaign_submit.sh "$RUNS/population_before" \
    --suite production --seeds 0-99 --arm before

# after arm's shell
CASES_SUBSET=canary TIME=$CIGAR_TIME MEM=$POP_MEM \
    dev/scripts/hpc/campaign_submit.sh "$RUNS/population_after" \
    --suite production --seeds 0-99 --arm after \
    --pair "$RUNS/population_before"
```

**The order of the finishes.** The after arm's finishing step `rescore`
reads the before arm's `verification.json`, and `summarize`, `rescore` and
the analysis refuse a report older than the directory it describes (one
that places a case as missing or in flight that has a record since).
So the before arm is finished completely before every finish of the after
arm, and the after arm is finished again after any later finish of the
before arm; a stale report is mended by finishing its campaign again.

When the canary's tasks have ended, look at them with the finish, before
arm first:

```bash
dev/scripts/hpc/campaign_finish.sh "$RUNS/population_before" --allow-missing --no-archive
dev/scripts/hpc/campaign_finish.sh "$RUNS/population_after" --allow-missing --no-archive
```

Each must report 24 cases verified and the rest missing, none failed. The
after arm's `rescore` then rescores 48 cases, the canary of both arms, and
a later `rescore` reuses them. Then submit the rest in each arm's shell,
`ARM` being `before` or `after`. `cigar_D15_exhaust` goes as its own
subset with its own limit, and every other case in one more submission;
the two arms hold the same case list:

```bash
CASES_SUBSET=cigar_D15_exhaust TIME=$CIGAR_TIME MEM=$POP_MEM \
    dev/scripts/hpc/campaign_submit.sh "$RUNS/population_$ARM"
rest=$(awk -F/ '$1 != "cigar_D15_exhaust" { print NR }' \
    "$RUNS/population_$ARM/cases.txt" | paste -sd, -)
ARRAY=$rest TIME=$POP_TIME MEM=$POP_MEM \
    dev/scripts/hpc/campaign_submit.sh "$RUNS/population_$ARM"
```

A configuration whose smoke cases (Phase 6) ask for a limit of their own,
`lumpy_D10_noise3_production` perhaps, which had never run before them,
goes the same way: a subset of its own, and one more condition in the
`awk` that lists the rest. When the queue is empty, finish the before arm
and then the after arm:

```bash
VERIFY_TIME=$POP_VERIFY_TIME VERIFY_MEM=$POP_VERIFY_MEM \
    dev/scripts/hpc/campaign_finish.sh "$RUNS/population_before"
VERIFY_TIME=$POP_VERIFY_TIME VERIFY_MEM=$POP_VERIFY_MEM \
    FINISH_TIME=$RESCORE_TIME FINISH_MEM=$RESCORE_MEM \
    dev/scripts/hpc/campaign_finish.sh "$RUNS/population_after"
```

The before arm's finish runs `verify` and `summarize`, which tabulates the
verified cases; its `verify` runs in the before arm's environment and
trees, since the boost captures unpickle only in the package that wrote
them. The after arm's finish runs `verify`, `summarize` and `rescore`,
which rescores the verified cases of both arms with the release code and
writes `rescored/<name>.json` for each arm, `rescoring.json` beside them,
and work files under `rescored/<name>.parts/`, from which a later
`rescore` resumes. `rescore` exits 78 unless it runs with the after arm's
own source identity and `PYVBMC_GPYREG_SOURCE` set, and fails when a case
of the after arm does not reproduce its in-run metrics exactly. A failed
case of a population is one whose run raised, or whose artifacts fail the
harness's checks: finite metrics, the options requested and applied, a
boost report consistent with its capture, and 750 evaluations for
`cigar_D15_exhaust`, a run of which that stops short of them fails. A
boost stage other than the returned posterior whose scoring failed keeps
the error as its metrics and fails no case; the analysis counts and lists
such stages, apart from the boost summary's usability counts.

#### The pools

The 8 conditions of the `svbmc_pool` suite, seeds from 1000, 350 seeds
each and 480 for the ring, 2930 cases, with
`HARNESS=dev/scripts/svbmc_pool_run.py` and `PYVBMC_GPYREG_SOURCE` the
gpyreg v1.3.3 checkout. The first submission prepares the pool and runs
its first case; the canary then runs the first case of every condition:

```bash
ARRAY=1 TIME=$POOL_TIME MEM=$POOL_MEM \
    dev/scripts/hpc/campaign_submit.sh "$RUNS/pools" \
    --gpyreg-source "$PYVBMC_GPYREG_SOURCE" --target 320 --max-seeds 350 \
    --allocation ring_D2_noise3_svbmc=320/480
first=$(awk -F/ '!seen[$1]++ { print NR }' "$RUNS/pools/cases.txt" | paste -sd, -)
ARRAY=$first TIME=$POOL_TIME MEM=$POOL_MEM \
    dev/scripts/hpc/campaign_submit.sh "$RUNS/pools"
dev/scripts/hpc/campaign_finish.sh "$RUNS/pools" --allow-missing --no-archive
```

The finish must report 8 cases verified. Then the rest, and the finish,
which runs `verify` (which recomputes every run's statistics from its
stored GP), `select` and `summarize`:

```bash
TIME=$POOL_TIME MEM=$POOL_MEM dev/scripts/hpc/campaign_submit.sh "$RUNS/pools"
VERIFY_TIME=$POOL_VERIFY_TIME VERIFY_MEM=$POOL_VERIFY_MEM \
    dev/scripts/hpc/campaign_finish.sh "$RUNS/pools"
```

`select` walks every seed of each condition in order and takes the runs
that pass the filters, up to the target of 320. It stops the finish when a
seed below a condition's last selected run is missing, interrupted, in
flight or partial, since that seed would have come first had it finished:
resubmit the seeds it names and finish again. A seed that stays missing
or interrupted at every resubmission is given up
([Giving a case up](#giving-a-case-up)), and `select` then reads it as a
failed seed, whatever files its task left. `selection.md` shows each
condition's selected runs against its target, and under `unfinished` the
seeds above its last selected run that have not finished; those are
resubmitted first, and a shortfall means that the condition's seeds ran
out only when it lists none, which goes to the PI. `selection.json`
records the SHA-256 of the manifest and of the verification report it was
made after, and the stacking comparison stacks a pool only when that
report passed and the selection agrees with it: a `verify` run by hand on
the pool afterwards needs a new `select`, which the finish, run again,
makes.

#### The stacking comparison

Both arms at `M` = 2, 4, 8 and 16 and the integrated arm alone at 3, 5 and
32, on disjoint subsets of each condition's 320 selected runs, with
`HARNESS=dev/scripts/svbmc_pool_stack.py`, `PYVBMC_GPYREG_SOURCE` the
gpyreg v1.3.3 checkout and `BASELINE_DIR` the S-VBMC checkout, from a
finished pool: `prepare` refuses a pool whose `verification.json` is
absent or did not pass, or whose `selection.json` is absent, was made
before its last `verify`, disagrees with it or was not made with the
allocation's first seed, seed cap and target (or the target that `select
--target` gave, which the stacking's manifest records as
`target_override`), and a dirty harness
checkout (the release gate never passes `--allow-dirty`). The grid below
is `prepare`'s default, written out. A task holds every repetition of one
condition and `M` below 16, and one cell from 16 on. The first submission
prepares the campaign and runs task 1, both arms at `M = 2` for the first
condition; the canary then runs the first task at `M = 32`:

```bash
ARRAY=1 TIME=$STACK_TIME MEM=$STACK_MEM \
    dev/scripts/hpc/campaign_submit.sh "$RUNS/stacking" \
    --pool "$RUNS/pools" --M 2,3,4,5,8,16,32 \
    --repetitions 20,20,20,20,20,10,10 \
    --arms both,integrated,both,integrated,both,both,integrated
first32=$("$CAMPAIGN_ENV/bin/python" dev/scripts/svbmc_pool_stack.py cases \
    --out "$RUNS/stacking" --subset M32 | head -n 1 | cut -d' ' -f1)
CASES_SUBSET=M32 ARRAY=$first32 TIME=$M32_TIME MEM=$M32_MEM \
    dev/scripts/hpc/campaign_submit.sh "$RUNS/stacking"
dev/scripts/hpc/campaign_finish.sh "$RUNS/stacking" --allow-missing --no-archive
```

This finish reports the two tasks verified and the rest missing, and then
stops at its finishing step, `assemble`, which merges a complete campaign
only: it exits 1 with `the finishing step 'assemble' failed`, the end to
expect of a canary. `prepare` prints a line for any condition and `M`
whose pool gives fewer repetitions than the grid asks; there should be
none. The tasks at `M = 16` and `M = 32` hold the largest S-VBMC fits and
take limits of their own, so each `M` goes as its own subset:

```bash
for M in 2 3 4 5 8; do
    CASES_SUBSET=M$M TIME=$STACK_TIME MEM=$STACK_MEM \
        dev/scripts/hpc/campaign_submit.sh "$RUNS/stacking"
done
CASES_SUBSET=M16 TIME=$M16_TIME MEM=$M16_MEM \
    dev/scripts/hpc/campaign_submit.sh "$RUNS/stacking"
CASES_SUBSET=M32 TIME=$M32_TIME MEM=$M32_MEM \
    dev/scripts/hpc/campaign_submit.sh "$RUNS/stacking"
```

A task whose error names the original arm's subprocess killed by SIGKILL
(exit code -9) met its memory limit: resubmit it with a larger `MEM`. The
finish runs `verify` and `assemble`, which builds `results.json`, the
summaries and `sources.json` from the cell files and draws the bootstrap:

```bash
FINISH_TIME=$ASSEMBLE_TIME FINISH_MEM=$ASSEMBLE_MEM \
    dev/scripts/hpc/campaign_finish.sh "$RUNS/stacking"
```

### Reading the finish's report, and resubmitting

The finish goes in this order.

1. **The accounting** of every job that `slurm/jobs.txt` and
   `slurm/steps.txt` record, into `slurm/sacct.txt`, which is removed when
   `sacct` fails. It holds the rows of the jobs' steps (`.batch`,
   `.extern`), whose `MaxRSS` sizes `MEM` ([Limits](#limits)); the checks
   below judge the allocation rows alone, since a step's row can stay
   `RUNNING` after its node failed.
2. **The queue check** (`campaign_contract.py queue-check`): `squeue` for
   every recorded job. A task it lists in a state that has not ended is
   queued; a job it cannot answer for is judged from `sacct.txt`, and one
   that the accounting does not show ended may still run. Either stops the
   finish unless `--allow-running`, and a finish that sees one never
   archives.
3. **`verify`, then each finishing step**, as batch jobs: each is
   submitted with `--parsable` and `--no-requeue`, its job id recorded in
   `slurm/steps.txt` (`<job> step=<name> submitted <date>`) before the
   finish waits for it, looking at the accounting every `STEP_POLL`
   seconds, and its exit code recorded when it ends (`<job> step=<name>
   rc=<code> <date>`); a submission that `sbatch` refused is recorded as
   `? step=<name> sbatch=<rc> <date>`. The wait prints a notice after every
   20 looks in a row that find the job unknown to the accounting, and
   after `STEP_WAIT_LIMIT` seconds it gives up, records `rc=124` and stops
   the finish. Interrupting the finish while it waits (Ctrl-C) is safe: the
   job keeps running, and the next finish's queue check sees it, as it
   sees a job that the wait gave up on.
4. **The archive**, unless `--no-archive`, only when nothing is queued, may
   still run or is in flight, and the accounting shows every recorded task
   ended (`campaign_contract.py archive-check`).

After `verify`, the finish prints one line of counts and the case indices
of every state but `verified`:

- **failed**: the case raised, or its artifacts failed the harness's
  checks; `<tag>.error.txt` in its condition's directory holds the
  traceback. Rerun it with `ARRAY=<index>`. A rerun sets the earlier error
  file aside as `claims/<tag>.error.txt` when it starts. A case that the
  operator gave up is failed too, its error file starting with the ruling
  ([Giving a case up](#giving-a-case-up)), and no task runs it again.
- **missing**: the case never ran, or Slurm stopped its task with SIGTERM
  (its time limit, `scancel`) and the worker cleaned up. A rerun of a
  failed case that is stopped so reads as missing, with the earlier
  attempt's reason as `earlier_error`.
- **interrupted**: Slurm killed its task outright (out of memory, SIGKILL,
  a lost node), leaving a stale claim and no completion record, with or
  without files. It is resubmitted like a missing case, and the worker
  that takes over the stale claim removes what the killed task left.
- **in flight**, **queued**: a task holds the case's live claim, or is
  still queued for it.
- **verify failed**, **partial**, **stray**: a record that no longer
  matches its files, files without a record or a claim, files that no
  case owns. All three are fatal, and none arises from the contract's own
  paths; find out what changed the directory before anything else.

The finish exits 1 on a failed verification, a partial case or a stray
file, 4 on cases in flight (unless `--allow-running`) and 3 on missing or
interrupted cases (unless `--allow-missing`), printing the `ARRAY=...` to
resubmit. With `--allow-running` the finish is a look at the campaign as
it stands: `verify` and its report, and neither the finishing steps nor
the archive; it exits 0 when nothing but cases in flight remains, and 3
on missing or interrupted cases without `--allow-missing`. Before
resubmitting, see in the accounting what stopped them:

```bash
grep -E 'TIMEOUT|OUT_OF_MEMORY|NODE_FAIL' "$RUNS/<campaign>/slurm/sacct.txt"
```

A time limit asks for a larger `TIME`, the memory limit for a larger
`MEM`; resubmit with them (`ARRAY=<indices> TIME=... MEM=...
campaign_submit.sh "$RUNS/<campaign>"`) and run the finish again. It runs
`verify` and the finishing steps again, and writes the archive only once
no task is queued, may still run or is in flight.

#### Giving a case up

A case that stays missing or interrupted at every resubmission (its time
limit stops it every time, say) is ruled out, from the harness checkout,
with its tag, the first field of line `<index>` of `cases.txt`:

```bash
"$CAMPAIGN_ENV/bin/python" dev/scripts/campaign_contract.py give-up \
    --out "$RUNS/<campaign>" --case <tag> --reason "<one line>"
```

Its error file then starts with `given up by the operator: <reason>`, and
`slurm/given_up.txt` logs the ruling: the time, the tag, the user and the
reason. `verify` places the case as failed with that reason, the worker
of any later task for it exits 0 at once, and the pool's `select` reads it
as a failed seed, whatever files a killed task left. `give-up` refuses a
case that has a record or is given up already, a tag that is not a case
of the campaign, a reason that is empty or longer than one line, and a
case whose claim is live. The reason enters `verification.json`, and so
the tracked copies, whose search for site details covers it: keep
hostnames, paths and usernames out of it.

### Limits

`TIME` and `MEM` are settings of each submission, for each harness and for
each subset that needs its own, and `VERIFY_TIME`, `VERIFY_MEM`,
`FINISH_TIME` and `FINISH_MEM` settings of each finish. In the commands
above they are: `POP_TIME`, `CIGAR_TIME` and `POP_MEM` for the population
tasks, `POP_VERIFY_TIME` and `POP_VERIFY_MEM` for their `verify`, and
`RESCORE_TIME` and `RESCORE_MEM` for the after arm's finishing steps;
`POOL_TIME` and `POOL_MEM` for the pool tasks, `POOL_VERIFY_TIME` and
`POOL_VERIFY_MEM` for their `verify`; `STACK_TIME`, `STACK_MEM`,
`M16_TIME`, `M16_MEM`, `M32_TIME` and `M32_MEM` for the stacking tasks,
and `ASSEMBLE_TIME` and `ASSEMBLE_MEM` for its finishing step; and
`CHECK_TIME` and `CHECK_MEM` for the environment check. Their values come
from the accounting of the smoke campaigns (the plan's Phase 6, `sacct`
and `seff`), and the operator's notes hold them; `MEM` comes from the
`MaxRSS` of the accounting's step rows. The population sidecars hold each
case's peak resident set as `max_rss_mb`, and as `peak_rss_mb`, on Linux,
the resident set at the end of the run, which is not a peak. For the
stacking, the developer's machine gives an order of size: an S-VBMC
`optimize()` at `M = 32` peaks near 2.1 GB, in its gradient steps, and its
final entropy evaluation near 350 MiB; a task's `MEM` is still Phase 6's.
`THROTTLE` applies to each submission, so a campaign submitted in several
chunks or subsets, and two campaigns that run together, run that many
tasks at once for each submission.

### The tracked copies, the archive and the hand-back

**The archive.** The finish writes the whole campaign directory beside
it, as `$RUNS/<campaign>.tar.zst.000`, `.001`, ..., parts below GitHub's
2 GiB per release asset, with their SHA-256 in
`$RUNS/<campaign>.tar.zst.sha256`. It holds everything, the site's
details and your paths among them, so it goes to a draft release, which
only those with write access to the repository see.
`cat <campaign>.tar.zst.[0-9][0-9][0-9] | zstd -d | tar x` restores it.

**The tracked copies.** The files of a campaign that enter the repository
are the ones its harness declares in the manifest (`tracked_copies`),
beside its manifest and verification report: for the pools the selection
and the summary; for each population arm the summary and every verified
case's completion record, sidecar and boost report, and for the after arm,
which rescores, also `rescored/*.json` and `rescoring.json` (the before arm
holds no `rescored/`); for the stacking `results.json`, the summaries and
`sources.json`. `analyze_population_run.py --arms` reads the copies of the
two arms as it reads the campaigns, the rescored metrics from the
candidate's by default (`--rescoring`), and `golden_replay.py --sidecars`
reads the per-configuration directories of the after arm or of its copies,
counting its verified cases alone. `campaign_redact.sh` writes the copies,
redacted, into the hand-back clone:

- every host that ran a Slurm job of the campaign is named by the node
  family, the value of `NODE_FEATURE`, and any other host (the login node)
  by `login`;
- a path under a named directory starts with its name: a path setting of
  the site block (`$PYVBMC_GPYREG_SOURCE/...`), a directory given as
  `--path NAME=PATH` (`$NAME/...`), your home (`~/...`), and, where none of
  those holds it, a source tree of the campaign's identities
  (`$HARNESS_TREE/...`, `$<TREE>_TREE`) or the directory that holds the
  campaign directory (`$CAMPAIGN_PARENT/...`);
- a field that holds a partition is `$PARTITION`;
- the manifest keeps no `site` block, and its `pip freeze` lines that name
  a path keep their package's name alone;
- the task logs, the Slurm accounting, the claims and the error files are
  not copied; job ids stay.

`redaction.json` beside the copies records the SHA-256 of each copy and of
the campaign's file it was made from, which the records hash, so that the
population analysis on the copies gives the numbers it gives on the
campaign; the archive's parts with theirs; and the cases the verification
report does not place as verified (`cases_not_verified`: failed,
interrupted, missing). A report that places a case in flight is refused.
The redaction then searches every copy: as plain substrings, for each
value of the site block that is a path or a command, every named
directory and your home directory; as whole names, which no letter,
digit, `_` or `-` flanks, for your username and every hostname that the
campaign's records, manifest, claims, logs and accounting hold, a
hostname in any letter case; and for any absolute path outside the
system's directories (`/bin`, `/dev`, `/etc`, `/lib`, `/lib64`,
`/opt/conda`, `/proc`, `/sbin`, `/sys`, `/tmp`, `/usr`, `/var/tmp`) that
no name covers. If any remains, it writes nothing and names the file and
the string. A hit that is benign, such as a short username that is also a
word of a copy, is exempted with `--allow STRING`, repeatable and taken by
`--check` too, and `redaction.json` records each allowed string under
`allowed` with the hits it cleared. A username, or a hostname that the
copies name otherwise, that is itself a name the copies write (`login`,
the node family) is refused, since the search could not tell the two
apart.

**Where it runs.** On the login node, in the account that ran the
campaign, after the finish has passed. The username and the home directory
it removes are those of the process that redacts (`USER`, `LOGNAME`,
`HOME` and the password database), never read from a file, so only that
account knows what to remove; the campaign directory and its archive are
there; and it reads the campaign's records and writes text, which needs
no batch job.

**The hand-back**, once per campaign as it finishes, with `LOGIN_SETUP`
run first for the clone, `gh` and the push:

```bash
# once: a clone for the hand-back, apart from the harness checkout,
# which stays clean while later campaigns use it
git clone https://github.com/acerbilab/pyvbmc "$TREES/handback"
git -C "$TREES/handback" switch -c release-gate-<date> origin/dev-next

# the tracked copies, from the harness checkout
dev/scripts/hpc/campaign_redact.sh "$RUNS/population_after" \
    "$TREES/handback/dev/experiments/release_gate_<date>/population_after"

# the archive's parts and their SHA-256, to a draft release of its own,
# with the environment's gh (once: "$CAMPAIGN_ENV/bin/gh" auth login)
"$CAMPAIGN_ENV/bin/gh" release create release-gate-population-after-<date> \
    --draft --repo acerbilab/pyvbmc \
    --title "Release gate: population, after arm" \
    --notes "The raw campaign directory; see dev/experiments/release_gate_<date>/"
"$CAMPAIGN_ENV/bin/gh" release upload release-gate-population-after-<date> \
    "$RUNS"/population_after.tar.zst.* --repo acerbilab/pyvbmc
```

and likewise `population_before`, `pools` and `stacking`, with the
releases `release-gate-population-before-<date>`,
`release-gate-pools-<date>` and `release-gate-stacking-<date>`. The
automatic names cover a campaign directory outside your home, on a
scratch area say; `--path NAME=PATH` gives a directory a name of your
choosing, ahead of the automatic ones, and a name may be no setting's,
not `CAMPAIGN_PARENT` and not one ending in `_TREE`.

Then write `dev/experiments/release_gate_<date>/README.md` in the
hand-back clone, in the manner of
[experiments/svbmc_pool/pool_20260914/README.md](../../experiments/svbmc_pool/pool_20260914/README.md):
the release commit and the trees' commits, the dates, the job ids, the
node family, each campaign's counts from its `verification.json`, the
resubmissions and every case that failed or went missing, the time and
memory the accounting shows, and each archive part with its size and
SHA-256. Keep hostnames, the username, paths and the site's names out of
it, and search it as the copies are searched, once for each campaign,
since each has its own hosts and settings:

```bash
for campaign in population_after population_before pools stacking; do
    dev/scripts/hpc/campaign_redact.sh "$RUNS/$campaign" --check \
        "$TREES/handback/dev/experiments/release_gate_<date>/README.md"
done
```

Commit the copies and the README, push the branch and open a pull request
to `dev-next`. List the raw campaign directories and the archives in
`dev/scripts/runs/LOCAL.md` of the harness checkout, which git ignores.

### When something goes wrong

A task's exit code, which the accounting records (the `ExitCode` column
of `slurm/sacct.txt`), is its worker's, one of:

| Code | Meaning |
|---|---|
| 0 | the case is complete: the worker completed it, it had a record already, or something failed after its record was written, which leaves the case complete; or the operator gave it up, and the worker exits at once |
| 1 | the case failed; its `<tag>.error.txt` holds why |
| 64 | the case line or the campaign directory is not this harness's, or the directory holds no readable manifest; nothing was touched |
| 75 | a live task holds the case's claim; nothing was touched |
| 78 | the source identity differs from the manifest's, or could not be established; nothing was touched |
| 128 + n | signal n stopped the run (143 for SIGTERM); one that came after the completion record leaves the case complete |

- **A live claim** (exit 75, `refused, claimed by ...`): another task
  holds the case, and the accounting does not show it ended, or could not
  be asked. A duplicate submission leaves these, harmlessly. The claim is
  `claims/<tag>` and names its owner; a stale one is retired as
  `claims/<tag>.stale.<owner>.<key>`. A worker removes a stale claim, or
  replaces one for a requeue, only while it holds the claim's retirement
  mark, the dot-file `.<name>.retiring.<owner>.<key>.<level>` beside it
  (`<name>` the last component of the tag). A worker that finds the mark
  held by a task that may still run refuses with 75 (`... is retiring it,
  and may still run`); one whose holder's task has ended takes the mark's
  next level. A `.retiring.` file that remains is the harmless trace of a
  retirer that was killed. Where the accounting cannot be asked from
  a compute node, every claim of an ended task stays live: check its owner
  with `sacct -j <job>_<task> -o State`, remove the claim once that task
  has ended, and resubmit the case.
- **An identity refusal** (exit 78, `refused, the source identity differs
  from the manifest's in [...]` or `refused, the identity cannot be
  established`): the named part of a tree, a hashed file or an imported
  version is not what `prepare` recorded, or a fact of the identity is
  missing: a tree that git cannot read, or the node's features, which
  `scontrol show node` still did not give after its retries at 2, 5
  and 10 s. A tree moved or was edited, or the environment changed: put it
  back to the recorded commit or pins. A campaign's identity never
  changes, and a fix to the harness is a new commit and a new campaign
  directory. A population task that exits 78 with `population_run.py
  refused: PyVBMC cannot be imported from the campaign's package tree`
  could not import PyVBMC from the tree that `PYVBMC_SOURCE` names, or
  from the harness checkout where it is unset; the traceback before that
  line says why.
- **Exit 64**: the task gave the harness a line that is not a case of its
  allocation, a directory that is not a campaign of that harness
  (`HARNESS` names another harness than the one the campaign was prepared
  with), or one that holds no readable manifest (`holds no readable
  manifest.json`).
- **A submission that stops** (`sbatch exited ...`, or `sbatch printed no
  job id`): the refusal names the `ARRAY=` still to submit, the chunks
  before it being submitted and recorded in `slurm/jobs.txt`. Without a job
  id, a job may exist that no file records: look for it with `squeue -n
  <campaign directory's name>` before submitting the rest.
- **A dirty tree** (`refusing: ... has uncommitted or untracked changes`):
  the submission or the finish lists the change. Remove it; nothing is
  committed to, or written into, a tree a campaign uses.
- **An environment that differs from the pins** (`refusing: the
  environment ... differs from ...campaign_requirements.txt`, and the
  differences): something was installed into the environment, or it was
  built from another file. Never install into a campaign's environment by
  hand; build a new one from the file, at a new prefix, for a new
  campaign.
- **A finish that refuses**: tasks still queued or running, or a job that
  neither the queue nor the accounting shows ended (wait, or pass
  `--allow-running` for a look, which runs no finishing step and never
  archives); a step job that the wait gave up on (`rc=124` in
  `slurm/steps.txt`; the next finish's queue check sees the job while it
  may still run); a fixed setting that differs from the manifest's (the
  shell holds another campaign's settings; the manifest's `site` block
  shows the ones it was prepared with); a step whose submission failed (`? step=` in `slurm/steps.txt`;
  look for its job with `squeue -n <campaign directory's name>_<step>`); a
  failed verification, or missing and interrupted cases (see
  [Reading the finish's report](#reading-the-finishs-report-and-resubmitting)).
- **A record whose core `verify` refuses** (`the CPU affinity ... is not
  the hardware threads ... of core ...`): `verify` requires that each task
  ran on one physical core with the whole core to itself, its CPU affinity
  or its job's cpuset being exactly that core's hardware threads, which
  the record's host part holds (`core_threads`, `job_cpuset`). A task
  that shared its core breaks that rule; how the site gives a task a whole
  core is a setting of the site, `SBATCH_EXTRA`, and goes to the PI.
- **A redaction that refuses**: the file and the string it names are what
  the rules did not remove, or an absolute path that no name covers; or,
  before any search, a username or hostname that is a name the copies
  write. A directory that holds nothing to hide once named can be given a
  name with `--path NAME=PATH`, and a hit where the string stands for no
  site detail is exempted with `--allow STRING`; anything else goes to the
  PI, since it is a site detail the redaction does not know.

## The September pool scripts

`svbmc_pool_env.sh`, `svbmc_pool_submit.sh`, `svbmc_pool_task.sbatch` and
`svbmc_pool_finish.sh` generated the S-VBMC run pool of 2026-09-14
([experiments/svbmc_pool/pool_20260914/README.md](../../experiments/svbmc_pool/pool_20260914/README.md),
"How it was generated"), for the cluster half of the S-VBMC run-pool
campaign
([plans/svbmc-benchmark-campaign.md](../../plans/svbmc-benchmark-campaign.md),
section "Cluster generation"; the brief was
[plans/svbmc-pool-handoff.md](../../plans/svbmc-pool-handoff.md)).
Written for the University of Helsinki's Turso cluster, they keep every
site-specific value in an environment variable. Their commands ran at
`d63c477`, and the scripts were revised after them in the review of the
pull request that delivered the pool. They no longer run against the pool
harness, whose worker takes a case line where their task script passes a
label and a seed; the harness and its test module have changed since in
other ways too, and the sections below describe them as they were for
that campaign. Its prerequisites install gpyreg 1.2.1, which the
`pyproject.toml` of the release (gpyreg 1.3.3 or later) excludes.

### Prerequisites

- A clone of this repository at the commit the pool is to be generated
  with, **clean**, with the package installed editable in a conda
  environment together with the `test` extra, `gpyreg==1.2.1` and
  `psutil` (`pip install -e ".[test]" "gpyreg==1.2.1" psutil`), and with
  `zstd` for the archive (`conda install -n pyvbmc-pool zstd`; activating
  the environment drops the base environment's binaries from the PATH).
  The environment's name goes in `POOL_CONDA_ENV` (default `pyvbmc-pool`)
  and conda's hook in `CONDA_SH` (default
  `~/miniconda3/etc/profile.d/conda.sh`).
- A gpyreg checkout at the pinned tag (`v1.2.1`, commit `9e70e6b`) that is
  a real, clean git repository. Put it at the harness default
  `dev/scripts/runs/svbmc_pool_20260913/gpyreg_1.2.1` (gitignored): the
  test module `test_svbmc_pool_run.py` skips every test when that path is
  absent, so a checkout elsewhere makes the environment check vacuous.
  ```
  git clone https://github.com/acerbilab/gpyreg dev/scripts/runs/svbmc_pool_20260913/gpyreg_1.2.1
  git -C dev/scripts/runs/svbmc_pool_20260913/gpyreg_1.2.1 checkout --detach v1.2.1
  ```
- The environment check, under `srun` rather than on the login node
  (it fits two small VBMC runs):
  ```
  srun -p short -c 1 --mem=2G -t 00:20:00 bash -c \
    'REPO=$PWD; source dev/scripts/hpc/svbmc_pool_env.sh; python -m pytest dev/scripts/test_svbmc_pool_run.py -q'
  ```
  It must report tests passed and none skipped.

### Scripts

- `svbmc_pool_env.sh` — sourced by the others with `REPO` set: activates
  the conda environment, exports single-threaded BLAS and `MPLBACKEND=Agg`,
  and, when `POOL_DIR` holds a manifest, `PYVBMC_GPYREG_SOURCE` from it.
- `svbmc_pool_submit.sh POOL_DIR [prepare flags]` — refuses a dirty
  checkout, runs `prepare` when the directory holds no manifest (the
  flags go to `prepare`, whose suite defaults to `svbmc_pool`; a prepared
  campaign is never revised here), writes `cases.txt` once and refuses to
  overwrite it with a different list, and submits the array in chunks
  that stay below the site's `MaxArraySize` (the largest valid array
  index is one less, so chunk `k` carries `INDEX_OFFSET = k × (MaxArraySize − 1)`
  and array indices from 1). `ARRAY` names case indices, the lines of
  `cases.txt`, as an sbatch-style list of numbers and ranges and is
  mapped onto those chunks, one submission per chunk touched, so
  `ARRAY=1` is a canary and `ARRAY=17,233` a resubmission on any site.
  A task whose case already has a completion record exits at once, so
  resubmitting a range that includes finished cases is safe. `THROTTLE`
  (200), `TIME` (00:30:00), `MEM` (2G), `PARTITION` (short) and
  `SBATCH_EXTRA` tune the submission. Job ids go to
  `POOL_DIR/slurm/jobs.txt`.
- `svbmc_pool_task.sbatch` — the array task: line
  `INDEX_OFFSET + SLURM_ARRAY_TASK_ID` of `cases.txt`, one `worker` call
  with the required environment. Its output is
  `POOL_DIR/slurm/<jobid>_<array index>.out`; in a chunk with an offset
  the array index is the case index minus the offset, and the task's
  first line names its case.
- `svbmc_pool_finish.sh POOL_DIR [--no-archive] [--allow-missing] [--allow-running]` —
  `sacct` for every recorded job into `slurm/sacct.txt`, `verify` under
  `srun` (`verification.json`; `VERIFY_TIME`, default 01:00:00, and
  `VERIFY_MEM`, default 2G, size that step), then `select`, `summarize`
  and the archive `<parent>/<name>.tar.zst` with its SHA-256 and size. It
  stops while tasks of the recorded jobs are still queued or running,
  when verification fails, or when cases are missing, printing the
  indices to resubmit; `--allow-missing` lets it go on past the missing
  cases, `--allow-running` past the queue check (a look at a campaign
  in flight, whose selection and summary leave out the runs in flight;
  it is never archived while tasks run).

### A campaign, step by step

```
REPO=~/pyvbmc                             # a clean checkout at the campaign's commit
GPYREG=$REPO/dev/scripts/runs/svbmc_pool_20260913/gpyreg_1.2.1
POOL=$REPO/dev/scripts/runs/svbmc_pool_20260914
cd $REPO

# smoke test of the machinery (three short cases, a scratch directory)
SMOKE=$REPO/dev/scripts/runs/svbmc_pool_smoke
dev/scripts/hpc/svbmc_pool_submit.sh $SMOKE --gpyreg-source $GPYREG \
    --suite smoke --only normal_D2 --target 2 --max-seeds 3 --seed-start 4000
# ... wait for the three tasks ...
dev/scripts/hpc/svbmc_pool_finish.sh $SMOKE --no-archive
rm -rf $SMOKE

# the campaign: prepare, one canary task, then the rest
ARRAY=1 dev/scripts/hpc/svbmc_pool_submit.sh $POOL --gpyreg-source $GPYREG
# ... the canary's record appears under $POOL/records/ ...
ARRAY=2-1100 dev/scripts/hpc/svbmc_pool_submit.sh $POOL
squeue -u $USER -n svbmc_pool

# when the queue is empty
dev/scripts/hpc/svbmc_pool_finish.sh $POOL --no-archive --allow-missing
# resubmit what verify lists as missing (raise TIME/MEM if Slurm killed them)
ARRAY=<indices> TIME=01:00:00 dev/scripts/hpc/svbmc_pool_submit.sh $POOL
dev/scripts/hpc/svbmc_pool_finish.sh $POOL
```

### Rules that protect the data

- **Do not move the checkout while the array runs.** Every worker
  compares the identity fixed by `prepare`: the checkout's HEAD, the
  hashes of the suite module and both pool scripts, the gpyreg commit and
  the Python, NumPy and SciPy versions. A commit checked out, a pull, or
  an edit to those files makes every later task refuse; a harness fix
  therefore means a new commit and a fresh campaign directory (`prepare`
  refuses to change the identity of an existing one).
- **Only light commands on the login node**: `prepare`, `cases`,
  `select`, `summarize`, `sbatch`, `sacct`, `tar`, `zstd` at its light
  setting and `sha256sum`. The tests and `verify` run under `srun`.
- **A failed case** leaves `<tag>.error.txt` and no artifact, exits
  non-zero, and is skipped by `select`; rerun it (`ARRAY=<index>`; a
  success clears the error file) or leave it out. **A task Slurm kills**
  (time limit, memory) leaves nothing; `verify` lists it as missing with
  its index. `select` and `summarize` cannot see such a case, so run
  `verify` before them.
- `--save-vbmc` is deliberately off: the comparison reads the `.npz` and
  `.json` artifacts only, and the pickles would multiply the archive.
- The machine holding the raw directory lists it in the gitignored
  `dev/scripts/runs/LOCAL.md`, with the two commands above that recreate
  the gpyreg checkout.

### Hand-back

The archive is uploaded as an asset of a draft release of this repository
and its name, size and SHA-256 recorded in the README under
`dev/experiments/svbmc_pool/pool_<date>/`, together with the tracked
copies of `manifest.json`, `selection.json`, `selection.md`,
`summary.json`, `summary.md` and `verification.json`. The manifest stores
the cluster's absolute gpyreg path; on another machine
`svbmc_pool_run.py verify`, the stacking comparison
(`svbmc_pool_stack.py --pool`) and the Phase 2 estimator each take
`--gpyreg-source PATH`, a local clean checkout at the manifest's gpyreg
commit, in its place. That machine's BLAS does not reproduce the
cluster's recomputation bit for bit; `verify` allows each artifact the
rounding its own GP's condition number amplifies and reports both.
