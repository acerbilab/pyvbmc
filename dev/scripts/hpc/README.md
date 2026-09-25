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
  [record of that campaign](#the-september-pool-scripts).

The scripts hold no value particular to a site: the node feature, the
partition, the modules, the paths and everything else a site needs are
settings, whose values the operator keeps in their own notes, outside
this repository. This guide names each setting and never its value.

## The operator's guide

### The scripts

Each script's header documents it in full.

- `campaign_env.sh`: the environment of every process of a campaign.
  Sourced, it activates the conda environment `CAMPAIGN_ENV`, exports
  single-threaded BLAS and the source trees, and puts `TMPDIR` under the
  campaign directory. Run as `bash dev/scripts/hpc/campaign_env.sh build`,
  it builds that environment.
- `campaign_submit.sh CAMPAIGN_DIR [prepare flags]`: prepares a campaign
  once (the harness's `prepare`, with the flags), writes its case list
  once, and submits the cases, or a subset or the indices `ARRAY` names,
  as job arrays in chunks below the site's `MaxArraySize`.
- `campaign_task.sbatch`: one array task, which runs one case through the
  harness's `worker`; or, submitted by the finish, one step on the whole
  directory.
- `campaign_finish.sh CAMPAIGN_DIR [--no-archive] [--allow-missing]
  [--allow-running]`: the accounting, `verify` and the harness's finishing
  steps as batch jobs, then the archive.
- `campaign_redact.sh CAMPAIGN_DIR OUT_DIR [--path NAME=PATH]`: the
  tracked copies of a finished campaign, redacted for the repository.

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
| `NODE_FEATURE` | the Slurm feature that selects the campaign's node family; every job requests it with `-C` | yes |
| `PYVBMC_SOURCE` | the package tree of a population arm whose code is not the harness checkout's (the before arm's worktree); unset otherwise | yes |
| `PYVBMC_GPYREG_SOURCE` | the gpyreg checkout of the campaign, or of the arm | yes |
| `BASELINE_DIR` | the original S-VBMC checkout, for the stacking comparison | yes |
| `PARTITION` | `-p` of every job; omitted when unset | no |
| `LOGIN_SETUP` | commands that the environment's build runs first on the login node, for the network | no |
| `CONDA_SETUP` | commands that define `conda`, such as loading a site module or sourcing a Miniforge installation's `etc/profile.d/conda.sh` | no |
| `LOGIN_PROFILE` | the login profile the environment script reads when `module` is not defined; default `/etc/profile` | no |
| `CASES_SUBSET` | a named subset of the cases, submitted as its own array | no |
| `ARRAY` | the case indices to submit, as `1`, `17,233` or `2-1100`; default every case of the campaign or of the subset | no |
| `THROTTLE` | the tasks of one submission that run at once; default 200 | no |
| `TIME`, `MEM` | `--time` and `--mem` of each task; default `00:30:00` and `2G` | no |
| `SBATCH_EXTRA` | further `sbatch` arguments, split into words | no |
| `VERIFY_TIME`, `VERIFY_MEM`, `FINISH_TIME`, `FINISH_MEM` | the limits of the verify job and of each finishing step; default `01:00:00` and `2G` | no |
| `ARCHIVE_PART_SIZE` | the largest part of the archive; default `1900M` | no |

`prepare` records every setting in the `site` block of the campaign's
manifest. A later submission and the finish refuse a fixed setting whose
value differs from the manifest's; the others may change from one
submission to the next, and `slurm/jobs.txt` records each submission's.

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

### The environment

Every process of every campaign runs in one conda environment, built on
the login node from `campaign_requirements.txt`: Python from conda-forge
at the version its `# python==` line pins, `zstd` and `gh`, and every
package from PyPI at its pinned version. PyVBMC and gpyreg are not
installed in it; they come from the campaign's source trees. Build it from
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

**Freezing the requirements.** The file first holds the direct
requirements alone. The first environment built from them is frozen into
it once, before the first submission of any campaign, since the
submission refuses an installed package that the file does not pin:

```bash
"$CAMPAIGN_ENV/bin/python" -c 'import platform; print(platform.python_version())'
"$CAMPAIGN_ENV/bin/python" -m pip list --format=freeze --exclude-editable
```

The first command gives the patch release for the `# python==` line; the
second, the pins that replace the file's direct ones, dependencies
included. Keep the `--extra-index-url` line, from which the `+cpu` build
of Torch comes, and say in the file's header that it is frozen. Run the
check above on the frozen file, commit it, and pull it into the harness
checkout. Every environment of the campaigns is then built from the
frozen file, in whatever account, and holds the same libraries.

### The source trees

Each tree is a clone of its repository at a named commit, with nothing
changed in it. Clone them on the login node:

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
original S-VBMC arm imports the campaign environment's Torch, CPU Torch
2.14.0, which the requirements pin; its checkout is `BASELINE_DIR`. A
process of the original arm verifies it by content against
`dev/experiments/svbmc_pool/baseline_environment.json`: the commit, a
clean tree, the SHA-256 of every file of `src/svbmc` and the Torch
version.

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

### The environment check

The test modules of the contract, the driver and the three harnesses run
as one batch job on the campaigns' node family, from the harness checkout,
with the release trees: `REPO` the harness checkout, `PYVBMC_GPYREG_SOURCE`
the gpyreg v1.3.3 checkout, `BASELINE_DIR` the S-VBMC checkout, and
`PYVBMC_SOURCE` unset.

```bash
export REPO=$TREES/pyvbmc PYVBMC_GPYREG_SOURCE=$TREES/gpyreg-v1.3.3 \
    BASELINE_DIR=$TREES/S-VBMC CAMPAIGN_DIR=$RUNS/environment_check
unset PYVBMC_SOURCE
mkdir -p "$CAMPAIGN_DIR"
cd "$REPO"
sbatch -C "$NODE_FEATURE" --hint=nomultithread ${PARTITION:+-p "$PARTITION"} \
    --cpus-per-task=1 --time="$CHECK_TIME" --mem="$CHECK_MEM" \
    --output="$CAMPAIGN_DIR/%j.out" --export=ALL <<'EOF'
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
```

It passes when the job exits 0, every module's summary line counts tests
passed and none failed or skipped (`grep -n skipped` on the output prints
nothing; `-rs` gives the reason of any skip), and `git status` printed
nothing. A module that skips proves nothing about the environment. The
before arm's trees are checked by its canary, which runs whole cases of
`f91fdf0` through the harness.

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
again. `squeue -u "$USER"` shows the queue.

#### The two population arms

The 24 configurations of the `production` suite at seeds 0 to 99, 2400
cases in each arm, with `HARNESS=dev/scripts/population_run.py`:

| Arm | `PYVBMC_SOURCE` | `PYVBMC_GPYREG_SOURCE` |
|---|---|---|
| before | `$TREES/pyvbmc-f91fdf0` | `$TREES/gpyreg-v1.2.1` |
| after | unset (the harness checkout's own package) | `$TREES/gpyreg-v1.3.3` |

Both arms use the same environment and node family, so that they differ
in their code alone. Prepare the before arm first: the after arm's
`--pair` names it, and the after arm's finishing step `rescore` rescores
both arms with the release code. The two arms must be prepared with the
same allocation, options and confirmatory family; with `--confirmatory
FILE` where the PI fixes a family other than the default, in both. The
canary is the subset `canary`, the first seed of every configuration,
which includes `cigar_D15_exhaust` and so takes its limit.

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

When the canary's tasks have ended, look at them with the finish, before
arm first:

```bash
dev/scripts/hpc/campaign_finish.sh "$RUNS/population_before" --allow-missing --no-archive
dev/scripts/hpc/campaign_finish.sh "$RUNS/population_after" --allow-missing --no-archive
```

Each must report 24 cases verified and the rest missing, none failed. Then
submit the rest in each arm's shell, `ARM` being `before` or `after`.
`cigar_D15_exhaust` goes as its own subset with its own limit, and every
other case in one more submission; the two arms hold the same case list:

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
and then the after arm, whose `rescore` reads the before arm's
verification:

```bash
VERIFY_TIME=$POP_VERIFY_TIME VERIFY_MEM=$POP_VERIFY_MEM     dev/scripts/hpc/campaign_finish.sh "$RUNS/population_before"
VERIFY_TIME=$POP_VERIFY_TIME VERIFY_MEM=$POP_VERIFY_MEM     FINISH_TIME=$RESCORE_TIME FINISH_MEM=$RESCORE_MEM     dev/scripts/hpc/campaign_finish.sh "$RUNS/population_after"
```

The before arm's finish runs `verify` and `summarize`; the after arm's,
`verify`, `summarize` and `rescore`, which rescores the 4800 cases of
both arms and writes `$RUNS/population_after/rescored/`. `verify` of the before arm runs in its
own environment and trees, since the boost captures unpickle only in the
package that wrote them.

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
ARRAY=$(awk -F/ '!seen[$1]++ { print NR }' "$RUNS/pools/cases.txt" | paste -sd, -) \
    TIME=$POOL_TIME MEM=$POOL_MEM \
    dev/scripts/hpc/campaign_submit.sh "$RUNS/pools"
dev/scripts/hpc/campaign_finish.sh "$RUNS/pools" --allow-missing --no-archive
```

The finish must report 8 cases verified. Then the rest, and the finish,
which runs `verify`, `select` (the first 320 runs of each condition, in
seed order, that pass the filters) and `summarize`:

```bash
TIME=$POOL_TIME MEM=$POOL_MEM dev/scripts/hpc/campaign_submit.sh "$RUNS/pools"
VERIFY_TIME=$POOL_VERIFY_TIME VERIFY_MEM=$POOL_VERIFY_MEM     dev/scripts/hpc/campaign_finish.sh "$RUNS/pools"
```

Its `verify` recomputes every run's statistics from its stored GP.

`selection.md` shows each condition's selected runs against its target of
320; a shortfall means the condition's seeds ran out, and goes to the PI.

#### The stacking comparison

Both arms at `M` = 2, 4, 8 and 16 and the integrated arm alone at 3, 5 and
32, on disjoint subsets of each condition's 320 selected runs, with
`HARNESS=dev/scripts/svbmc_pool_stack.py`, `PYVBMC_GPYREG_SOURCE` the
gpyreg v1.3.3 checkout and `BASELINE_DIR` the S-VBMC checkout, from a
finished pool. The grid below is `prepare`'s default, written out. A
task holds every repetition of one condition and `M` below 16, and one
cell from 16 on. The first submission prepares the campaign and runs task
1, both arms at `M = 2` for the first condition; the canary then runs the
first task at `M = 32`:

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

`prepare` prints a line for any condition and `M` whose pool gives fewer
repetitions than the grid asks; there should be none. The tasks at
`M = 16` and `M = 32` need more memory and time than the others, so each
`M` goes as its own subset:

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

The finish runs `verify` and `assemble`, which builds `results.json`, the
summaries and `sources.json` from the cell files and draws the bootstrap:

```bash
FINISH_TIME=$ASSEMBLE_TIME FINISH_MEM=$ASSEMBLE_MEM \
    dev/scripts/hpc/campaign_finish.sh "$RUNS/stacking"
```

### Reading the finish's report, and resubmitting

The finish first stops while any task of the campaign is queued or
running, unless `--allow-running`. It writes the accounting of every job
to `slurm/sacct.txt`, runs `verify`, which writes `verification.json`,
and prints one line of counts and the case indices of every state but
`verified`:

- **failed**: the case raised; `<tag>.error.txt` in its condition's
  directory holds the traceback. Rerun it with `ARRAY=<index>`; leaving
  a case out is the PI's ruling.
- **missing**: the case never ran, or Slurm stopped its task with SIGTERM
  (its time limit, `scancel`) and the worker cleaned up.
- **interrupted**: Slurm killed its task outright (out of memory, SIGKILL,
  a lost node) and it left its claim. It is resubmitted like a missing
  case, and the worker that takes over its stale claim removes what the
  killed task left.
- **in flight**, **queued**: a task holds the case's live claim, or is
  still queued for it.
- **verify failed**, **partial**, **stray**: a record that no longer
  matches its files, files without a record or a claim, files that no
  case owns. None of them arises from the contract's own paths; find out
  what changed the directory before anything else.

The finish exits 1 on a failed verification, a partial case or a stray
file, 4 on cases in flight (unless `--allow-running`) and 3 on missing
or interrupted cases (unless `--allow-missing`), printing the
`ARRAY=...` to resubmit. Before
resubmitting, see in the accounting what stopped them:

```bash
grep -E 'TIMEOUT|OUT_OF_MEMORY|NODE_FAIL' "$RUNS/<campaign>/slurm/sacct.txt"
```

A time limit asks for a larger `TIME`, the memory limit for a larger
`MEM`; resubmit with them (`ARRAY=<indices> TIME=... MEM=...
campaign_submit.sh "$RUNS/<campaign>"`) and run the finish again. It runs
`verify` and the finishing steps again, and writes the archive only once
no task is queued, running or in flight.

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
`CHECK_TIME` and `CHECK_MEM` for the environment check. Their values
come from the accounting of the smoke campaigns (the plan's Phase 6, `sacct` and
`seff`), and the operator's notes hold them. The one measurement so far
is of the stacking's tasks, on the developer's machine: one at `M = 16`
peaked near 2.9 GiB and one at `M = 32` near 6.1 GiB. `THROTTLE`
applies to each submission, so a campaign submitted in several chunks
or subsets, and two campaigns that run together, run that many tasks at
once for each submission.

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
and the summary; for each population arm the summary, the rescored
metrics, and every verified case's completion record, sidecar and boost
report, which `analyze_population_run.py --arms` reads; for the stacking
`results.json`, the summaries and `sources.json`. `campaign_redact.sh`
writes them, redacted, into the hand-back clone:

- every host that ran a Slurm job of the campaign is named by the node
  family, the value of `NODE_FEATURE`, and any other host (the login node)
  by `login`;
- a path under a path setting of the site block starts with the setting's
  name, `$PYVBMC_GPYREG_SOURCE/...`, and a path under your home with `~`;
- a partition is `$PARTITION`;
- the manifest keeps no `site` block, and its `pip freeze` lines that
  name a path keep their package's name alone;
- the task logs, the Slurm accounting, the claims and the error files are
  not copied; job ids stay.

`redaction.json` beside the copies records the SHA-256 of each copy and of
the campaign's file it was made from, which the records hash, so that the
population analysis on the copies gives the numbers it gives on the
campaign, and the archive's parts with theirs. The redaction then
searches every copy for each value of the site block that is a path, a
command or a partition, for your username and home directory, and for
every hostname that the campaign's records, manifest, logs and accounting
hold; if any remains, it writes nothing and names the file and the string.

**Where it runs.** On the login node, in the account that ran the
campaign, after the finish has passed. The username and the home directory
it removes are those of the process that redacts (`USER`, `LOGNAME`,
`HOME` and the password database), never read from a file, so only that
account knows what to remove; the campaign directory and its archive are
there; and the redaction reads and writes text alone, as `prepare` does,
so it needs no batch job.

**The hand-back**, once per campaign as it finishes:

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
`release-gate-pools-<date>` and `release-gate-stacking-<date>`. A campaign
directory outside your home, on a scratch area say, is given to the
redaction as `--path NAME=PATH`, so that its paths become `$NAME/...`.

Then write `dev/experiments/release_gate_<date>/README.md` in the
hand-back clone, in the manner of
[experiments/svbmc_pool/pool_20260914/README.md](../../experiments/svbmc_pool/pool_20260914/README.md):
the release commit and the trees' commits, the dates, the job ids, the
node family, each campaign's counts from its `verification.json`, the
resubmissions and every case that failed or went missing, the time and
memory the accounting shows, and each archive part with its size and
SHA-256. It is written by hand, so no redaction checks it: keep hostnames,
the username, paths and the site's names out of it, and search it for
them before you commit. Commit the copies and the README, push the branch
and open a pull request to `dev-next`. List the raw campaign directories
and the archives in `dev/scripts/runs/LOCAL.md` of the harness checkout,
which git ignores.

### When something goes wrong

- **A live claim** (a task's log says `refused, claimed by ...`, exit 75):
  another task holds the case, and the accounting does not show it ended,
  or could not be asked. The case is left as it is. A duplicate submission
  leaves these, harmlessly. Where the accounting cannot be asked from a
  compute node, every claim of an ended task stays live: check its owner
  with `sacct -j <job>_<task> -o State`, which the claim file
  `claims/<tag>` names, remove the claim once that task has ended, and
  resubmit the case.
- **An identity refusal** (`refused, the source identity differs from the
  manifest's in [...]`, exit 78): the named part of a tree, a hashed file
  or an imported version is not what `prepare` recorded. A tree moved or
  was edited, or the environment changed. Put it back to the recorded
  commit or pins; a campaign's identity never changes, and a fix to the
  harness is a new commit and a new campaign directory.
- **A dirty tree** (`refusing: ... has uncommitted or untracked changes`):
  the submission or the finish lists the change. Remove it; nothing is
  committed to, or written into, a tree a campaign uses.
- **An environment that differs from the pins** (`refusing: the
  environment ... differs from ...campaign_requirements.txt`, and the
  differences): something was installed into the environment, or it was
  built from another file. Never install into a campaign's environment by
  hand; build a new one from the file, at a new prefix, for a new
  campaign.
- **A finish that refuses**: tasks still queued or running (wait, or pass
  `--allow-running` for a look); a fixed setting that differs from the
  manifest's (the shell holds another campaign's settings; the manifest's
  `site` block shows the ones it was prepared with); a failed
  verification, or missing and interrupted cases (see
  [Reading the finish's report](#reading-the-finishs-report-and-resubmitting)).
- **A redaction that refuses**: the file and the string it names are
  what the rules did not remove. A path outside your home takes
  `--path NAME=PATH`; anything else goes to the PI, since it is a site
  detail the redaction does not know.

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
