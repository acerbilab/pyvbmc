#!/bin/bash
# Account for, verify, finish and archive a campaign of a harness that
# meets the campaign contract (dev/plans/slurm-benchmark-support.md).
#
#   campaign_finish.sh CAMPAIGN_DIR [--no-archive] [--allow-missing]
#                                   [--allow-running]
#
# On the login node, with the operator settings of the submission (see
# campaign_submit.sh); like it, refuses a dirty source tree, an environment
# that differs from campaign_requirements.txt and a fixed setting that
# differs from the manifest's site block. Then:
#
# 1. The accounting of every job of slurm/jobs.txt and slurm/steps.txt,
#    into slurm/sacct.txt (removed when sacct fails).
# 2. The queue: `squeue` for every recorded job (campaign_contract.py
#    queue-check). A task it lists in a state that has not ended (pending,
#    running, completing, suspended, requeued and the like) is queued; a
#    job it cannot answer for, as it cannot for one the controller has
#    dropped, is looked up in the accounting, and one that the accounting
#    does not show ended may still run. It stops while any task is queued
#    or may still run, unless --allow-running; the case indices of the
#    queued array tasks go to slurm/queued.txt.
# 3. `$HARNESS verify` as a batch job on the campaign's node family
#    (slurm/verify_<job>.out), which writes verification.json; a verify
#    submitted while the campaign's tasks run may queue behind them. It
#    stops when verify fails (a record that fails its check, a stray file,
#    or artifacts with neither a record nor a claim), when cases are in
#    flight (a live claim, or a task still queued) unless --allow-running,
#    and when cases are missing or interrupted unless --allow-missing. A
#    case is missing when its task never ran, or stopped on SIGTERM (its
#    time limit, scancel) and cleaned up; it is interrupted when its task
#    was killed outright (SIGKILL, out of memory) and left its claim, with
#    or without files. Both are resubmitted alike: the finish prints their
#    indices for `ARRAY=... campaign_submit.sh CAMPAIGN_DIR` (raise TIME or
#    MEM when the accounting shows that a limit stopped them). Cases in
#    flight are counted apart from missing ones, so --allow-running works
#    without --allow-missing.
# 4. The harness's finishing steps, the manifest's "finishing_steps", each
#    a batch job in turn (slurm/<step>_<job>.out).
# 5. Unless --no-archive, and never while a task is queued, may still run
#    or is in flight, or while the accounting of step 1 shows a recorded
#    task that has not ended or is missing (campaign_contract.py
#    archive-check): the whole directory as <parent>/<name>.tar.zst.000,
#    .001, ..., zstd-compressed parts of at most ARCHIVE_PART_SIZE (default
#    1900M, below GitHub's 2 GiB per release asset), with their SHA-256 in
#    <parent>/<name>.tar.zst.sha256.
#    `cat <name>.tar.zst.[0-9][0-9][0-9] | zstd -d | tar x` restores it.
#
# Each batch job of steps 3 and 4 is submitted, its job id recorded in
# slurm/steps.txt ("<job> step=<name> submitted <date>"), and then waited
# for in the accounting (campaign_contract.py wait-job), which gives its
# exit code ("<job> step=<name> rc=<code> <date>"), so that a finish
# stopped while it waits leaves a job that the next finish's queue check
# sees.
#
# Environment (optional, beyond the submission's): VERIFY_TIME (01:00:00)
# and VERIFY_MEM (2G) size the verify job, FINISH_TIME (01:00:00) and
# FINISH_MEM (2G) each finishing step; STEP_POLL (30) is the seconds between
# two looks at the accounting while a step job runs.
set -euo pipefail

usage() {
    sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//' | head -n -1 >&2
    exit 64
}
refuse() {
    echo "refusing: $*" >&2
    exit 1
}

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
CONTRACT="$REPO/dev/scripts/campaign_contract.py"
export REPO

if [ $# -lt 1 ] || [ ! -f "$1/manifest.json" ]; then
    usage
fi
CAMPAIGN_DIR=$(cd "$1" && pwd -P)
shift
ARCHIVE=1
FLAGS=()
ALLOW_RUNNING=0
for arg in "$@"; do
    case $arg in
        --no-archive) ARCHIVE=0 ;;
        --allow-missing) FLAGS+=(--allow-missing) ;;
        --allow-running)
            FLAGS+=(--allow-running)
            ALLOW_RUNNING=1
            ;;
        *) usage ;;
    esac
done

for name in HARNESS CAMPAIGN_ENV NODE_FEATURE; do
    if [ -z "${!name:-}" ]; then
        refuse "$name is not set (the operator settings of" \
            "dev/plans/slurm-benchmark-support.md)"
    fi
done
# The pattern of campaign_contract.FEATURE_PATTERN.
[[ $NODE_FEATURE =~ ^[A-Za-z0-9_.-]+$ ]] \
    || refuse "NODE_FEATURE=$NODE_FEATURE is not one node feature:" \
        "letters, digits and _.- alone"
PARTITION=${PARTITION:-}
SBATCH_EXTRA=${SBATCH_EXTRA:-}
STEP_POLL=${STEP_POLL:-30}
[[ $STEP_POLL =~ ^[0-9]+([.][0-9]+)?$ ]] \
    || refuse "STEP_POLL=$STEP_POLL is not a number of seconds"
export HARNESS CAMPAIGN_ENV NODE_FEATURE PARTITION LOGIN_SETUP CONDA_SETUP \
    LOGIN_PROFILE PYVBMC_SOURCE PYVBMC_GPYREG_SOURCE BASELINE_DIR SBATCH_EXTRA
unset CAMPAIGN_STEP INDEX_OFFSET
export CAMPAIGN_DIR

# shellcheck source=campaign_env.sh
source "$HERE/campaign_env.sh"
campaign_check_trees
cd "$REPO"
python "$CONTRACT" check-env --requirements "$CAMPAIGN_REQUIREMENTS" \
    || refuse "the environment $CAMPAIGN_ENV differs from $CAMPAIGN_REQUIREMENTS"
python "$CONTRACT" check-site --manifest "$CAMPAIGN_DIR/manifest.json" \
    || refuse "the settings differ from those $CAMPAIGN_DIR was prepared with"

SLURM_DIR="$CAMPAIGN_DIR/slurm"
mkdir -p "$SLURM_DIR"
touch "$SLURM_DIR/jobs.txt" "$SLURM_DIR/steps.txt"
JOB_NAME=$(basename "$CAMPAIGN_DIR" | tr -c 'A-Za-z0-9_.+=\n-' '_')

# 1. The accounting of every recorded job.
JOBS=$(cat "$SLURM_DIR/jobs.txt" "$SLURM_DIR/steps.txt" \
    | awk '$1 ~ /^[0-9]+$/ && !seen[$1]++ { print $1 }' | paste -sd, -)
if [ -n "$JOBS" ]; then
    if sacct -P --units=M -j "$JOBS" \
        --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,AllocCPUS,NodeList \
        > "$SLURM_DIR/sacct.txt" < /dev/null; then
        echo "accounting in $SLURM_DIR/sacct.txt"
    else
        rm -f "$SLURM_DIR/sacct.txt"
        echo "sacct failed; the finish has no accounting of the campaign" >&2
    fi
fi

# 2. The queue, one query per job: an id that has aged out of the queue
#    makes squeue fail, which must not hide another job's tasks.
QUEUED="$SLURM_DIR/queued.txt"
queue=$(python "$CONTRACT" queue-check --slurm "$SLURM_DIR") \
    || refuse "the queue check of $SLURM_DIR failed"
if [ "$queue" != clear ] && [ "$ALLOW_RUNNING" = 0 ]; then
    echo "pass --allow-running for a look at the campaign as it stands" >&2
    exit 1
fi

PARTITION_ARGS=()
if [ -n "$PARTITION" ]; then
    PARTITION_ARGS=(-p "$PARTITION")
fi
# SBATCH_EXTRA as words, split without pathname expansion.
SBATCH_EXTRA_ARGS=()
if [ -n "$SBATCH_EXTRA" ]; then
    read -r -d '' -a SBATCH_EXTRA_ARGS <<< "$SBATCH_EXTRA" || true
fi

# step NAME TIME MEM WORDS...: one harness step as a batch job, its job id
# recorded before the wait for it.
step() {
    local name=$1 time=$2 mem=$3 jid rc=0
    shift 3
    jid=$(CAMPAIGN_STEP="$*" sbatch --parsable -J "${JOB_NAME}_$name" \
        -C "$NODE_FEATURE" --hint=nomultithread \
        ${PARTITION_ARGS[@]+"${PARTITION_ARGS[@]}"} \
        --cpus-per-task=1 --mem="$mem" --time="$time" \
        --output "$SLURM_DIR/${name}_%j.out" --export=ALL \
        ${SBATCH_EXTRA_ARGS[@]+"${SBATCH_EXTRA_ARGS[@]}"} \
        "$HERE/campaign_task.sbatch" < /dev/null) || rc=$?
    jid=${jid%%;*}
    if [ "$rc" != 0 ] || ! [[ $jid =~ ^[0-9]+$ ]]; then
        echo "? step=$name sbatch=$rc $(date +%FT%T)" >> "$SLURM_DIR/steps.txt"
        echo "step '$*': sbatch exited $rc and printed '$jid', no job id;" \
            "a job it may have submitted is not in slurm/steps.txt: look for" \
            "it with squeue -n ${JOB_NAME}_$name" >&2
        return 1
    fi
    echo "$jid step=$name submitted $(date +%FT%T)" >> "$SLURM_DIR/steps.txt"
    echo "step '$*': job $jid submitted; waiting for it"
    python -u "$CONTRACT" wait-job --job "$jid" --poll "$STEP_POLL" || rc=$?
    echo "$jid step=$name rc=$rc $(date +%FT%T)" >> "$SLURM_DIR/steps.txt"
    echo "step '$*': job $jid exited $rc (log $SLURM_DIR/${name}_$jid.out)"
    return "$rc"
}

# 3. Verification.
rm -f "$CAMPAIGN_DIR/verification.json"
rc=0
step verify "${VERIFY_TIME:-01:00:00}" "${VERIFY_MEM:-2G}" verify || rc=$?
if [ ! -f "$CAMPAIGN_DIR/verification.json" ]; then
    echo "verify wrote no $CAMPAIGN_DIR/verification.json (exit $rc)" >&2
    exit 1
fi
decision=0
in_flight=$(python "$CONTRACT" finish-check \
    --verification "$CAMPAIGN_DIR/verification.json" --queued "$QUEUED" \
    ${FLAGS[@]+"${FLAGS[@]}"}) || decision=$?
if [ "$rc" -ne 0 ]; then
    echo "verify exited $rc; fix or remove the reported cases first" >&2
    exit "$rc"
fi
if [ "$decision" -ne 0 ]; then
    exit "$decision"
fi

# 4. The harness's finishing steps.
steps_text=$(python "$CONTRACT" finishing-steps \
    --manifest "$CAMPAIGN_DIR/manifest.json") \
    || refuse "the manifest's finishing steps break the contract"
while read -r line; do
    [ -n "$line" ] || continue
    # The step is its words, which the contract limits to plain ones.
    # shellcheck disable=SC2086
    step "${line%% *}" "${FINISH_TIME:-01:00:00}" "${FINISH_MEM:-2G}" $line \
        || { echo "the finishing step '$line' failed" >&2; exit 1; }
done <<< "$steps_text"

# 5. The archive: the whole directory, next to it.
if [ "$ARCHIVE" = 1 ]; then
    if [ "$queue" != clear ] || [ "${in_flight:-0}" != 0 ]; then
        echo "not archiving a campaign whose tasks are queued, may still" \
            "run or are in flight; run again when they are done" >&2
        exit 1
    fi
    if ! python "$CONTRACT" archive-check --slurm "$SLURM_DIR"; then
        echo "not archiving: the accounting does not show every recorded" \
            "task ended; run again when it does" >&2
        exit 1
    fi
    if ! command -v zstd > /dev/null 2>&1; then
        refuse "zstd is not on the PATH of the environment $CAMPAIGN_ENV"
    fi
    NAME=$(basename "$CAMPAIGN_DIR")
    PARENT=$(dirname "$CAMPAIGN_DIR")
    OUT="$PARENT/$NAME.tar.zst"
    rm -f "$OUT".[0-9][0-9][0-9] "$OUT.sha256"
    # The artifacts are compressed already, so a light setting is enough and
    # the shared login node is not loaded for nothing.
    tar -C "$PARENT" -cf - "$NAME" | zstd -T2 -3 -q -c \
        | split -b "${ARCHIVE_PART_SIZE:-1900M}" -d -a 3 - "$OUT."
    (cd "$PARENT" && sha256sum "$NAME.tar.zst".[0-9][0-9][0-9]) > "$OUT.sha256"
    cat "$OUT.sha256"
    ls -l "$OUT".[0-9][0-9][0-9]
fi
