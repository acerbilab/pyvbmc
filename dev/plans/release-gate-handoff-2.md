# Hand-off: the release gate's second batch on the cluster

Written 2026-10-05 for the developer (and their coding agents) who ran
the release gate's first batch of campaigns on the Slurm cluster, in their
own account, and runs the second the same way. Everything referenced is
on `dev-next` but Arm 3's code, which is on the branch
`dev-arm3-port-warmup`. The procedure, with every command, is the
operator's guide,
[scripts/hpc/README.md](../scripts/hpc/README.md): its sections "Arm 0:
PyVBMC 1.0.4", "Arm 3: the port's end of warm-up" and "The second batch's
hand-back", and for the pools and the stacking the release gate's own.
This brief says what runs, in what order and with which limits, and what
comes back. The designs and their reasons are in
[Arm 0's plan](arm-1.0.4-comparison.md) and
[Arm 3's plan](arm-3-warmup-comparison.md); the first batch's brief is
[release-gate-handoff.md](release-gate-handoff.md).

## What this is and why

Two arms join the release gate's populations. Each is compared seed by
seed with the first batch's after arm, which is read and not run again.

| Campaign | Size | Expected cost |
|---|---|---|
| `population_v104` (Arm 0) | PyVBMC 1.0.4 as released, with gpyreg 1.0.4, on the 2400 cases of the population arms | about 500 CPU-hours |
| `population_arm3` (Arm 3) | the release code with the port's end of warm-up on the same 2400 cases | about 175 CPU-hours |
| `pools_arm3` | the 8 conditions of the `svbmc_pool` suite, 2930 runs, with Arm 3's code | about 220 CPU-hours |
| `stacking_arm3` | 200 tasks, 960 cells, on Arm 3's pools | about 17 CPU-hours |
| `fresh_release`, `fresh_arm3` | `rosenbrock_D2_noise3_production` at seeds 100 to 199, with the release code and with Arm 3's | under 10 CPU-hours |

Arm 0 measures PyVBMC 1.5 against 1.0.4, the release its users upgrade
from. Arm 3 is the release code with the end of warm-up that the port had
before two fixes of the port review moved it to MATLAB's: by a rule that
the PI fixed before the runs, its campaigns decide which end of warm-up
the release keeps, and its stacking which estimate of the stacked ELBO
S-VBMC reports. Reading the results is the PI's work: the operator's job
ends with campaigns that verify and are handed back.

Arm 0's cost scales the after arm's runs by 1.0.4's time on the
developer's machine (Arm 0's plan, "Limits from the local runs"); the
others are the first batch's accounting
([its README](../experiments/release_gate_20261002/README.md)).

## Before the launch

1. **The two commits**, which the PI names at the launch: Arm 0's, on
   `dev-next`, to which the harness checkout `$TREES/pyvbmc` moves, and
   Arm 3's, on the branch `dev-arm3-port-warmup`, whose checkout is a
   detached worktree of the harness checkout, `$TREES/pyvbmc-arm3`. Both
   hold the same files that build and score a run as the after arm's
   commit and the same frozen requirements, and both stay where they are,
   clean, until the hand-back, since the redaction, the public assets and
   any later `rescore-arms` run from them. Arm 0's commit also runs
   `fresh_release` and Arm 0's `rescore-arms`; Arm 3's runs every other
   campaign of Arm 3.
2. **The source trees** (the guide's "The source trees"): the first
   batch's, with the harness checkout moved, and the new ones.

   | Tree | Commit | Used by |
   |---|---|---|
   | `pyvbmc`, the harness checkout | Arm 0's commit | Arm 0, `fresh_release`, `rescore-arms` |
   | `pyvbmc-arm3`, a detached worktree of it | Arm 3's commit | `population_arm3`, `pools_arm3`, `stacking_arm3`, `fresh_arm3`, the analyses |
   | `pyvbmc-v1.0.4`, a detached worktree of it | `0bb8b8f5` (tag `v1.0.4`) | Arm 0 |
   | `gpyreg-v1.0.4` | `3b1300a` (tag `v1.0.4`) | Arm 0 |
   | `gpyreg-v1.4.0` | `682585f` (tag `v1.4.0`) | Arm 3, the fresh seeds, `rescore-arms`, the analyses |
   | `S-VBMC` | `13a78f6` | the stacking's original arm |

3. **The environment**: the first batch's, at the same path
   (`CAMPAIGN_ENV`) and unchanged; the pairing of Arm 3 with the after
   arm compares the environment's path and versions.
4. **The environment check**, twice, as a batch job each: in the harness
   checkout at Arm 0's commit, and in the Arm 3 checkout
   (`REPO=$TREES/pyvbmc-arm3`, `CAMPAIGN_DIR=$RUNS/environment_check_arm3`),
   both with the 1.0.4 trees exported as the guide gives them. Each
   passes as in the first batch: the job exits 0, no test is skipped, the
   interpreter it names is the environment's and the checkout is clean
   after it. Nothing is submitted before both pass.
5. **The after arm's directory**, `$RUNS/population_after`, in place or
   restored from its archive (the guide's Arm 0 section, "The after arm's
   directory"), and never finished again.

## The campaigns, in order

Each campaign runs in a shell of its own, from its harness checkout, with
its `HARNESS` and trees exported, and starts with a canary and a look at
it with the finish. Submit at a `THROTTLE` of 75 a submission and keep
the tasks running at once near 300, the first batch's peak once its nodes
no longer failed their prolog: a submission that would take the batch
well past that waits for an earlier one to drain, and submissions go a
few minutes apart.

1. **The canaries** of Arm 0 (24 cases), of Arm 3's population (24
   cases), of `fresh_release` and then of `fresh_arm3` (1 case each),
   each looked at with its finish as the guide gives it. Arm 3's look
   rescores the after arm's 2400 cases beside its 24, about 35 minutes,
   with the rescoring's limits.
2. **The rest of the two populations and of the fresh seeds**: each
   population's `cigar_D15_exhaust` as its own subset, Arm 0's with
   `ARM0_CIGAR_TIME` and the rest with `ARM0_TIME`, Arm 3's with
   `CIGAR_TIME` and `POP_TIME`. Then the finishes: `fresh_release`
   before `fresh_arm3`, whose `rescore` reads its verification; Arm 0's;
   Arm 3's.
3. **Arm 3's pools**, then **its stacking comparison**, from the Arm 3
   checkout, as the release gate's ("The pools", "The stacking
   comparison"), with `$RUNS/pools_arm3` and `$RUNS/stacking_arm3`.
   `selection.md` must show 320 selected runs for every condition.
4. **The two analyses of Arm 3's pools**, after the stacking's finish, as
   two batch jobs from the Arm 3 checkout, the second waiting for the
   first:

   ```bash
   (
       export REPO=$TREES/pyvbmc-arm3 PYVBMC_GPYREG_SOURCE=$TREES/gpyreg-v1.4.0 \
           POOL=$RUNS/pools_arm3 CELLS=$RUNS/stacking_arm3/results.json \
           OUT=$RUNS/analyses_arm3
       unset PYVBMC_SOURCE BASELINE_DIR CAMPAIGN_DIR
       mkdir -p "$OUT"
       cd "$REPO"
       read -r -d '' -a extra <<< "${SBATCH_EXTRA:-}" || true
       shrink=$(sbatch --parsable -J shrink -C "$NODE_FEATURE" \
           --hint=nomultithread ${PARTITION:+-p "$PARTITION"} \
           --cpus-per-task=1 --no-requeue \
           --time="$SHRINK_TIME" --mem="$ANALYSIS_MEM" \
           --output="$OUT/shrink_%j.out" --export=ALL \
           ${extra[@]+"${extra[@]}"} \
           --wrap 'cd "$REPO" && source dev/scripts/hpc/campaign_env.sh \
               && python -u dev/scripts/svbmc_shrink_elbo.py \
               --pool "$POOL" --cells "$CELLS" --out "$OUT/shrink" \
               --gpyreg-source "$PYVBMC_GPYREG_SOURCE"')
       sbatch -J single_run -C "$NODE_FEATURE" --hint=nomultithread \
           ${PARTITION:+-p "$PARTITION"} --cpus-per-task=1 --no-requeue \
           --dependency="afterok:${shrink%%;*}" \
           --time="$SINGLE_RUN_TIME" --mem="$ANALYSIS_MEM" \
           --output="$OUT/single_run_%j.out" --export=ALL \
           ${extra[@]+"${extra[@]}"} \
           --wrap 'cd "$REPO" && source dev/scripts/hpc/campaign_env.sh \
               && python -u dev/scripts/svbmc_single_run_bias.py \
               --pool "$POOL" --out "$OUT/single_run" \
               --gpyreg-source "$PYVBMC_GPYREG_SOURCE" \
               --cells "$CELLS" --shrink "$OUT/shrink/cells.jsonl"'
   )
   ```

   Each job must exit 0. The first prints `960/960 cells`, the second
   `2560/2560 runs`, as in the first batch.
5. **Arm 0's `rescore-arms`**, after the finishes of Arm 0 and of Arm 3's
   population: it rescores the after arm, Arm 3 and Arm 0 with the release
   code, and the after arm's 2400 cases must reproduce their in-run
   metrics exactly (the guide's Arm 0 section, "The rescoring"), and its
   own archive part.
6. **The hand-back** (below).

## Limits

The first batch's, with Arm 0's of its own, from the measurements its
brief and Arm 0's plan give; nothing in the first batch reached a limit
(its longest population task took 34 minutes, of `cigar_D15_exhaust`).

| Setting | Value | Job |
|---|---|---|
| `CHECK_TIME`, `CHECK_MEM` | 1 h, 4 GB | each environment check |
| `ARM0_TIME`, `ARM0_MEM` | 2 h, 3 GB | an Arm 0 run |
| `ARM0_CIGAR_TIME` | 5 h | an Arm 0 run of `cigar_D15_exhaust`, and Arm 0's canary |
| `POP_TIME`, `POP_MEM` | 1 h, 2 GB | a run of Arm 3's population or of the fresh seeds |
| `CIGAR_TIME` | 1.5 h | an Arm 3 run of `cigar_D15_exhaust`, and Arm 3's canary |
| `POP_VERIFY_TIME`, `POP_VERIFY_MEM` | 2 h, 4 GB | a population's or fresh campaign's `verify` |
| `RESCORE_TIME`, `RESCORE_MEM` | 3 h, 4 GB | the finishing steps of Arm 3's population (its canary's among them) and of `fresh_arm3`, and `rescore-arms`; 0.8 s a case in the first batch, so about 65 and 100 minutes for the 4800 and 7200 cases of Arm 3's finish and of `rescore-arms` |
| `POOL_TIME`, `POOL_MEM` | 45 min, 2 GB | a pool run |
| `STACK_TIME`, `STACK_MEM` | 1 h, 4 GB | a stacking task at `M` = 2, 3, 4, 5 or 8 |
| `M16_TIME`, `M16_MEM` | 30 min, 4 GB | a stacking cell at `M = 16` |
| `M32_TIME`, `M32_MEM` | 1 h, 6 GB | a stacking cell at `M = 32` |
| `ASSEMBLE_TIME`, `ASSEMBLE_MEM` | 1 h, 4 GB | the stacking's `assemble` |
| `SHRINK_TIME` | 1 h | the first analysis |
| `SINGLE_RUN_TIME` | 3 h | the second analysis |
| `ANALYSIS_MEM` | 4 GB | both analyses |

Arm 3's noisy runs end their warm-up a few iterations later than the
release code's, and run a little longer; the first batch's limits hold
that margin.

## What to hand back

The guide's "The second batch's hand-back" gives the commands, `<date>`
being the batch's launch.

1. **The archives, as each finish passes**, since the cluster's storage
   has no backup: each campaign's archive parts, with their SHA-256 files,
   to its draft release; after `rescore-arms`, its archive part to Arm 0's
   draft release; and the analyses' outputs, `$RUNS/analyses_arm3`, as
   one archive with its SHA-256, as a further asset of
   `release-gate-stacking-arm3-<date>`. Like the first batch's, the
   analyses' outputs hold the cluster's details, and no file of them
   enters the pull request.
   A finish run again rewrites its archive, whose parts go up again
   (`--clobber`), and the rescorings that name its verification run again.
2. **The tracked copies of the six campaigns**, once the batch's last
   finish and `rescore-arms` have passed, once each: `population_v104`'s
   declare the rescoring that `rescore-arms` writes, a later finish calls
   for a rescoring again, and the redaction writes into an empty directory
   alone. Each is redacted in its own shell, from the harness checkout it
   ran from, into a branch `release-gate-<date>` of the hand-back clone,
   under `dev/experiments/release_gate_<date>/`.
3. **The public assets** of `population_arm3` and `pools_arm3`, built from
   the Arm 3 checkout into `$RUNS/public_arm3` and uploaded to the draft
   release `release-gate-public-arm3-<date>`, which stays a draft: the
   PyVBMC release attaches what the PI's decision on the end of warm-up
   names.
4. **A pull request to `dev-next`** from that branch: the tracked copies
   and `dev/experiments/release_gate_<date>/README.md`, searched with
   `campaign_redact.sh --check` once for each campaign and sent to the PI
   before the pull request opens. Every draft release stays a draft.

## Things to know

- **Nothing moves under a campaign**, as in the first batch: the two
  checkouts stay at their commits, clean, and the environment as built,
  from the first submission to the last finish. A problem in the harness
  goes to the PI before anything is changed.
- **The two checkouts are paired.** Arm 3's campaigns pair with the after
  arm and with `fresh_release` across harness commits, which `prepare
  --pair` allows only while the files that build and score a run are the
  same in both (the guide's "The pairing across harness commits"). The
  hash of `dev/scripts/data/` takes every file in it, those git ignores
  among them, so nothing is added to either checkout's `dev/scripts/data/`.
- **What goes to the PI before you act on it:** what the first brief lists (a
  failed case, a case missing or interrupted after a resubmission with larger
  limits, a failed `verify`, a `partial` or `stray`, a condition short of 320
  selected runs, a line from the stacking's `prepare` about a pool that gives
  fewer repetitions than the grid asks, a redaction that refuses a string it
  names no rule for); a `prepare --pair` or `prepare --compare` that refuses; a
  `rescore` or `rescore-arms` whose own arm does not reproduce its in-run
  metrics (Arm 3's in the finishes of `population_arm3` and `fresh_arm3`,
  `fresh_release`'s in its own, the after arm's in `rescore-arms`).
- **Site details stay out of the repository**, as in the first batch.
- Questions about the procedure go to the guide and to each script's
  header; questions about the design to the two arms' plans; questions
  about what the campaigns are for to `dev/TODO.md` ("Arm 3: the port's
  end of warm-up", "1.5 against 1.0.4 (Arm 0)").
