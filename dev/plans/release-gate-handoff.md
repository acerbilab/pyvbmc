# Hand-off: the release-gate campaigns on the cluster

Written 2026-10-01 for the developer (and their coding agents) who runs
the campaigns of the PyVBMC 1.5 release gate on the Slurm cluster, in
their own account. Everything referenced is on `dev-next`. The procedure,
with every command, is the operator's guide,
[scripts/hpc/README.md](../scripts/hpc/README.md); this brief says what
runs, in what order and with which limits, and what comes back. The
design and its reasons are in the [Slurm plan](slurm-benchmark-support.md).

## What this is and why

PyVBMC 1.5 is released once four campaigns have run with the release code
and their results have been read:

| Campaign | Size | Measured cost |
|---|---|---|
| Reference population, before arm | the 24 configurations of the `production` suite at seeds 0 to 99, 2400 VBMC runs, with the code from before the port review (`f91fdf0`) | about 155 CPU-hours |
| Reference population, after arm | the same 2400 runs with the release code | about 165 CPU-hours |
| S-VBMC run pools | the 8 conditions of the `svbmc_pool` suite, 2930 VBMC runs, of which 320 per condition are stacked | about 225 CPU-hours |
| Stacking comparison | 200 tasks, 960 cells: the integrated S-VBMC and the original package at `M` = 2, 4, 8 and 16, the integrated one alone at 3, 5 and 32 | 25 CPU-hours or more |

The after arm becomes the package's reference population, whose accuracy
envelopes its trajectory checks use, and its paired comparison with the
before arm measures what the port review's fixes did to accuracy. The
pools and their stacking are the gate of S-VBMC and decide which estimate
of the stacked ELBO the package reports. Two analyses of the pools follow
the stacking. Reading the results is the PI's work: the operator's job
ends with campaigns that verify and are handed back.

Every task is one VBMC run or one group of stacks on one physical core,
and takes minutes. The costs are those of the smoke campaigns of the
plan's Phase 6, which ran each harness on the campaigns' node family; the
plan's "Resources and cost" gives the measurements.

## Before the launch

1. **The release commit.** The campaigns run from one commit of
   `dev-next`, which the PI names at the launch. The harness checkout is a
   clone at that commit and stays there, clean, until the last campaign
   is finished.
2. **The site's settings.** The scripts hold no value particular to a
   site. The node feature that selects the campaigns' node family, the
   commands that define `conda` and that give the login node the network,
   and the partition where one is needed are settings (the guide's
   "Settings"), and the PI gives their values outside this repository.
   Keep them in a file of your own that exports them, outside every
   checkout, with the directories of your choosing for the source trees
   (`TREES`), the campaign directories (`RUNS`) and the environment
   (`CAMPAIGN_ENV`).
3. **The source trees** (the guide's "The source trees"), each a clean
   clone at its commit:

   | Tree | Commit | Used by |
   |---|---|---|
   | `pyvbmc`, the harness checkout | the release commit | every campaign |
   | `gpyreg-v1.4.0` | `682585f` (tag `v1.4.0`) | the after arm, the pools, the stacking, the analyses |
   | `gpyreg-v1.2.1` | `9e70e6b` (tag `v1.2.1`) | the before arm |
   | `pyvbmc-f91fdf0`, a detached worktree of the harness checkout | `f91fdf0` | the before arm |
   | `S-VBMC` | `13a78f6` | the stacking's original arm |

4. **The environment** (the guide's "The environment"): one conda
   environment, built with `bash dev/scripts/hpc/campaign_env.sh build`
   from the frozen `dev/scripts/hpc/campaign_requirements.txt` of the
   release commit. Nothing is installed into it afterwards; PyVBMC and
   gpyreg are not installed in it and come from the source trees.
5. **The environment check** (the guide's "The environment check"), a
   batch job of 20 to 25 minutes that runs the harnesses' test modules in
   that environment with the release trees. It passes when the job exits
   0, no test is skipped, the interpreter it names is the environment's
   and the checkout is clean after it. Nothing is submitted before it
   passes.

## The campaigns, in order

The guide's "The campaigns" gives the commands; each campaign runs in a
shell of its own, with its `HARNESS` and its trees exported. Every
campaign starts with a canary, a look at it with the finish, and then
the rest.

1. **The two population arms, together** ("The two population arms").
   Prepare the before arm first, then the after arm with `--pair`. The
   canary is the first seed of every configuration: each arm must report
   24 cases verified and none failed. Then the rest of each arm, with
   `cigar_D15_exhaust` as its own subset. Finish the before arm
   completely before every finish of the after arm, whose `rescore`
   reads the before arm's verification.
2. **The pools** ("The pools"). The canary is the first case of every
   condition: 8 cases verified. Then the rest, and the finish, whose
   `select` takes the first 320 runs of each condition that pass the
   filters. `selection.md` must show 320 selected runs for every
   condition.
3. **The stacking comparison** ("The stacking comparison"), from the
   finished pools. The canary is the first task and the first task at
   `M = 32`; its finish verifies the two and stops at `assemble`, as the
   guide says. Then every `M` as its own subset, and the finish.
4. **The two analyses of the pools**, after the stacking's finish, as two
   batch jobs from the harness checkout, the second waiting for the
   first:

   ```bash
   (
       export REPO=$TREES/pyvbmc PYVBMC_GPYREG_SOURCE=$TREES/gpyreg-v1.4.0 \
           POOL=$RUNS/pools CELLS=$RUNS/stacking/results.json \
           OUT=$RUNS/analyses
       unset PYVBMC_SOURCE BASELINE_DIR CAMPAIGN_DIR
       mkdir -p "$OUT"
       cd "$REPO"
       shrink=$(sbatch --parsable -J shrink -C "$NODE_FEATURE" \
           --hint=nomultithread ${PARTITION:+-p "$PARTITION"} \
           --cpus-per-task=1 --no-requeue \
           --time="$SHRINK_TIME" --mem="$ANALYSIS_MEM" \
           --output="$OUT/shrink_%j.out" --export=ALL \
           --wrap 'cd "$REPO" && source dev/scripts/hpc/campaign_env.sh \
               && python -u dev/scripts/svbmc_shrink_elbo.py \
               --pool "$POOL" --cells "$CELLS" --out "$OUT/shrink" \
               --gpyreg-source "$PYVBMC_GPYREG_SOURCE"')
       sbatch -J single_run -C "$NODE_FEATURE" --hint=nomultithread \
           ${PARTITION:+-p "$PARTITION"} --cpus-per-task=1 --no-requeue \
           --dependency="afterok:${shrink%%;*}" \
           --time="$SINGLE_RUN_TIME" --mem="$ANALYSIS_MEM" \
           --output="$OUT/single_run_%j.out" --export=ALL \
           --wrap 'cd "$REPO" && source dev/scripts/hpc/campaign_env.sh \
               && python -u dev/scripts/svbmc_single_run_bias.py \
               --pool "$POOL" --out "$OUT/single_run" \
               --gpyreg-source "$PYVBMC_GPYREG_SOURCE" \
               --cells "$CELLS" --shrink "$OUT/shrink/cells.jsonl"'
   )
   ```

   Each job must exit 0. The first prints `960/960 cells` and writes
   `shrink/cells.jsonl` with its summaries; the second prints `2560/2560
   runs` and writes `single_run/runs.jsonl`, `cells.jsonl` and the
   summaries `summary.md` and `added.md`.

## Limits

`TIME` and `MEM` of each job, from the smoke campaigns' accounting with a
margin. A task that Slurm stops at a limit is resubmitted with a larger
one (the guide's "Reading the finish's report, and resubmitting").

| Setting | Value | Job | Measured |
|---|---|---|---|
| `CHECK_TIME`, `CHECK_MEM` | 1 h, 4 GB | the environment check | 19 and 25 min, 1.5 GB |
| `POP_TIME`, `POP_MEM` | 1 h, 2 GB | a population run | 11 min or less, 430 MB |
| `CIGAR_TIME` | 1.5 h | a run of `cigar_D15_exhaust`, and the canary | 22 min |
| `POP_VERIFY_TIME`, `POP_VERIFY_MEM` | 2 h, 4 GB | a population arm's `verify` | 0.75 s a case, 450 MB |
| `RESCORE_TIME`, `RESCORE_MEM` | 3 h, 4 GB | the after arm's finishing steps | 0.9 s a case, 241 MB |
| `POOL_TIME`, `POOL_MEM` | 45 min, 2 GB | a pool run | 11 min or less, 375 MB |
| `STACK_TIME`, `STACK_MEM` | 1 h, 4 GB | a stacking task at `M` = 2, 3, 4, 5 or 8 | about 14 min at `M = 8`, 1.0 GB |
| `M16_TIME`, `M16_MEM` | 30 min, 4 GB | a stacking cell at `M = 16` | 2.8 min, 1.4 GB |
| `M32_TIME`, `M32_MEM` | 1 h, 6 GB | a stacking cell at `M = 32` | 4.3 min, 3.1 GB |
| `ASSEMBLE_TIME`, `ASSEMBLE_MEM` | 1 h, 4 GB | the stacking's `assemble` | 31 s and 417 MB for 22 cells |
| `SHRINK_TIME` | 1 h | the first analysis | 16 s for 22 cells |
| `SINGLE_RUN_TIME` | 3 h | the second analysis | about 1 s a run |
| `ANALYSIS_MEM` | 4 GB | both analyses | 1.0 GB |

The `verify` of the pools and of the stacking fits the default limits of
the finish (1 hour, 2 GB). The stacking was measured on one condition,
`student_D8_noise3_svbmc`; another condition's tasks may need more, which
the margins of `M16_MEM` and `M32_MEM` are for.

## What to hand back

The guide's "The tracked copies, the archive and the hand-back" gives the
commands. For each campaign, as soon as its finish has passed, since the
cluster's storage has no backup:

1. **The archive** that the finish wrote, the whole campaign directory in
   parts below 2 GiB, uploaded with its SHA-256 file to a draft release
   of this repository of its own: `release-gate-population-before-<date>`,
   `release-gate-population-after-<date>`, `release-gate-pools-<date>`,
   `release-gate-stacking-<date>`. An archive holds the site's details and
   your paths, so these releases stay drafts: never publish one.
2. **The tracked copies**, which `campaign_redact.sh` writes into a
   hand-back clone under `dev/experiments/release_gate_<date>/`, with the
   site's details and your username and paths removed. It runs in your
   account, on the login node, and refuses when anything it knows of
   remains.
3. **The public asset** of the after arm and of the pools
   (`campaign_public.sh`, then its `--check`), uploaded to the draft
   release `release-gate-public-<date>`, from which the PyVBMC release
   attaches them.

Then, once for the four campaigns:

4. **The analyses' outputs.** `$RUNS/analyses` goes back whole as one
   more asset of `release-gate-stacking-<date>`, with its SHA-256. Its
   `sources.json`, `summary.json` and `added.json` hold your paths and
   the names of the nodes, and nothing redacts them, so no file of it
   enters the pull request.

   ```bash
   tar -C "$RUNS" -cf - analyses | "$CAMPAIGN_ENV/bin/zstd" -q -c \
       > "$RUNS/analyses.tar.zst"
   (cd "$RUNS" && sha256sum analyses.tar.zst) > "$RUNS/analyses.tar.zst.sha256"
   "$CAMPAIGN_ENV/bin/gh" release upload release-gate-stacking-<date> \
       "$RUNS/analyses.tar.zst" "$RUNS/analyses.tar.zst.sha256" \
       --repo acerbilab/pyvbmc
   ```
5. **A pull request to `dev-next`** from the hand-back clone's branch
   `release-gate-<date>`: the tracked copies of the four campaigns and
   `dev/experiments/release_gate_<date>/README.md`, which the guide
   describes and `campaign_redact.sh --check` searches. The repository
   is public, and a pull request is public from the moment it opens, so
   send the README to the PI before opening it: the check does not know
   every form a site detail can take in prose. The PI reviews the pull
   request.

## Things to know

- **Nothing moves under a campaign.** From a campaign's first submission
  to its last finish, its trees stay at their commits with nothing
  changed in them, and the environment stays as built: every worker
  compares its code with what `prepare` recorded and refuses a
  difference. A fix to the harness is a new commit and new campaign
  directories, and both population arms run from one harness commit, so
  a problem in the harness goes to the PI before anything is changed.
- **Site details stay out of the repository**: hostnames, your username,
  paths, module names, the partition and every node feature but the
  campaigns' family appear in no tracked file, commit message, pull
  request or issue. The redaction covers the tracked copies and the
  public assets; the README of the hand-back is yours, and
  `campaign_redact.sh --check` searches it.
- **What goes to the PI before you act on it:** a failed case, and a case
  that stays missing or interrupted after a resubmission with larger
  limits (giving a case up is the PI's ruling); a `verify failed`,
  `partial` or `stray` in a finish's report; a `rescore` that does not
  reproduce the after arm's metrics; a condition short of 320 selected
  runs with no seed left to resubmit; a line from the stacking's
  `prepare` about a pool that gives fewer repetitions than the grid
  asks; and a redaction that refuses a string it does not name a rule
  for.
- **Bitwise reproduction** holds within the campaigns' node family and
  environment alone, which is why every job requests the family's
  feature and why the two population arms share everything but their
  code.
- Questions about the procedure go to the operator's guide and to each
  script's header; questions about the design to the
  [Slurm plan](slurm-benchmark-support.md); questions about what the
  campaigns are for to `dev/TODO.md` ("Final large-scale check before the
  release", "The golden references after the port review") and the
  [campaign plan](svbmc-benchmark-campaign.md). The working rules of the
  repository are in `AGENTS.md`.
