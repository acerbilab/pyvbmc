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
# 1. The queue: `squeue` for every job of slurm/jobs.txt and
#    slurm/steps.txt. It stops while any task is queued or running, unless
#    --allow-running; the case indices of the queued and running array
#    tasks go to slurm/queued.txt.
# 2. The accounting of every recorded job, into slurm/sacct.txt.
# 3. `$HARNESS verify` as a batch job on the campaign's node family
#    (sbatch --wait; slurm/verify_<job>.out), which writes
#    verification.json; a verify submitted while the campaign's tasks run
#    may queue behind them. It stops when verify fails (a record that fails
#    its check, a stray file, or artifacts with neither a record nor a
#    claim), when cases are in flight (a live claim, or a task still
#    queued) unless --allow-running, and when cases are missing or
#    interrupted unless --allow-missing. A case is missing when its task
#    never ran, or stopped on SIGTERM (its time limit, scancel) and cleaned
#    up; it is interrupted when its task was killed outright (SIGKILL, out
#    of memory) and left its claim, with or without files. Both are
#    resubmitted alike: the finish prints their indices for `ARRAY=...
#    campaign_submit.sh CAMPAIGN_DIR` (raise TIME or MEM when the
#    accounting shows that a limit stopped them). Cases in flight are
#    counted apart from missing ones, so --allow-running works without
#    --allow-missing.
# 4. The harness's finishing steps, the manifest's "finishing_steps", each
#    a batch job in turn (slurm/<step>_<job>.out).
# 5. Unless --no-archive, and never while tasks are queued, running or in
#    flight: the whole directory as <parent>/<name>.tar.zst.000, .001, ...,
#    zstd-compressed parts of at most ARCHIVE_PART_SIZE (default 1900M,
#    below GitHub's 2 GiB per release asset), with their SHA-256 in
#    <parent>/<name>.tar.zst.sha256.
#    `cat <name>.tar.zst.[0-9][0-9][0-9] | zstd -d | tar x` restores it.
#
# Environment (optional, beyond the submission's): VERIFY_TIME (01:00:00)
# and VERIFY_MEM (2G) size the verify job, FINISH_TIME (01:00:00) and
# FINISH_MEM (2G) each finishing step.
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
PARTITION=${PARTITION:-}
SBATCH_EXTRA=${SBATCH_EXTRA:-}
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

# 1. The queue, one query per job: an id that has aged out of the queue
#    makes squeue fail, which must not hide another job's tasks.
QUEUED="$SLURM_DIR/queued.txt"
: > "$QUEUED"
running=""
while read -r job fields; do
    [ -n "$job" ] || continue
    tasks=$(squeue -h -r -j "$job" -o "%i %T" 2>/dev/null < /dev/null || true)
    [ -n "$tasks" ] || continue
    running+="$tasks"$'\n'
    offset=$(echo "$fields" | sed -n 's/.*offset=\([0-9][0-9]*\).*/\1/p')
    if [ -n "$offset" ]; then
        # "<job>_<array index> <state>" -> "<case index> <state>"
        echo "$tasks" | awk -v o="$offset" '{
            n = split($1, p, "_")
            if (n == 2 && p[2] ~ /^[0-9]+$/) print p[2] + o, $2
        }' >> "$QUEUED"
    fi
done < <(cat "$SLURM_DIR/jobs.txt" "$SLURM_DIR/steps.txt")
if [ -n "$running" ]; then
    echo "tasks of the recorded jobs are still queued or running:" >&2
    printf '%s' "$running" >&2
    if [ "$ALLOW_RUNNING" = 0 ]; then
        exit 1
    fi
fi

# 2. The accounting of every recorded job.
JOBS=$(cat "$SLURM_DIR/jobs.txt" "$SLURM_DIR/steps.txt" \
    | awk '$1 ~ /^[0-9]+$/ { print $1 }' | paste -sd, -)
if [ -n "$JOBS" ]; then
    sacct -P --units=M -j "$JOBS" \
        --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,AllocCPUS,NodeList \
        > "$SLURM_DIR/sacct.txt" < /dev/null || echo "sacct failed" >&2
    echo "accounting in $SLURM_DIR/sacct.txt"
fi

PARTITION_ARGS=()
if [ -n "$PARTITION" ]; then
    PARTITION_ARGS=(-p "$PARTITION")
fi

# step NAME TIME MEM WORDS...: one harness step as a batch job, waited for.
step() {
    local name=$1 time=$2 mem=$3 jid rc=0
    shift 3
    # SBATCH_EXTRA is meant to word-split.
    # shellcheck disable=SC2086
    jid=$(CAMPAIGN_STEP="$*" sbatch --parsable --wait -J "${JOB_NAME}_$name" \
        -C "$NODE_FEATURE" --hint=nomultithread \
        ${PARTITION_ARGS[@]+"${PARTITION_ARGS[@]}"} \
        --cpus-per-task=1 --mem="$mem" --time="$time" \
        --output "$SLURM_DIR/${name}_%j.out" --export=ALL \
        $SBATCH_EXTRA "$HERE/campaign_task.sbatch" < /dev/null) || rc=$?
    jid=${jid%%;*}
    echo "${jid:-?} step=$name rc=$rc $(date +%FT%T)" >> "$SLURM_DIR/steps.txt"
    echo "step '$*': job ${jid:-?} exited $rc (log $SLURM_DIR/${name}_${jid:-?}.out)"
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
    # The step is its words.
    # shellcheck disable=SC2086
    step "${line%% *}" "${FINISH_TIME:-01:00:00}" "${FINISH_MEM:-2G}" $line \
        || { echo "the finishing step '$line' failed" >&2; exit 1; }
done <<< "$steps_text"

# 5. The archive: the whole directory, next to it.
if [ "$ARCHIVE" = 1 ]; then
    if [ -n "$running" ] || [ "${in_flight:-0}" != 0 ]; then
        echo "not archiving a campaign whose tasks are queued, running or" \
            "in flight; run again when they are done" >&2
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
