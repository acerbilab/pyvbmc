#!/bin/bash
#SBATCH --job-name=svbmc_pool
# One case of the S-VBMC run pool per array task: line
# INDEX_OFFSET + SLURM_ARRAY_TASK_ID of POOL_DIR/cases.txt, run by
# `svbmc_pool_run.py worker`. Submitted by svbmc_pool_submit.sh, which
# passes POOL_DIR, REPO and INDEX_OFFSET through `--export` and puts the
# resources, partition and output path on the command line (an #SBATCH
# line cannot take a variable). A failing case leaves <tag>.error.txt and
# exits non-zero, so Slurm records FAILED; a task Slurm kills leaves
# nothing, which `svbmc_pool_run.py verify` reports as missing.
set -euo pipefail

: "${POOL_DIR:?POOL_DIR must be exported by the submitter}"
: "${REPO:?REPO must be exported by the submitter}"
: "${SLURM_ARRAY_TASK_ID:?this script runs as a Slurm array task}"

LINE=$((SLURM_ARRAY_TASK_ID + ${INDEX_OFFSET:-0}))
CASE=$(sed -n "${LINE}p" "$POOL_DIR/cases.txt")
case "$CASE" in
    *" "*) ;;
    *)
        echo "no case on line $LINE of $POOL_DIR/cases.txt" >&2
        exit 2
        ;;
esac
LABEL=${CASE% *}
SEED=${CASE#* }
TAG="${LABEL}_seed${SEED}"
# A completed case is never regenerated: resubmitting a range that
# includes finished cases costs nothing but these lines.
if [ -f "$POOL_DIR/records/$TAG.complete.json" ]; then
    echo "task ${SLURM_ARRAY_JOB_ID:-?}_${SLURM_ARRAY_TASK_ID} line $LINE: $TAG already has a completion record; nothing to do"
    exit 0
fi

# shellcheck source=svbmc_pool_env.sh
source "$REPO/dev/scripts/hpc/svbmc_pool_env.sh"
cd "$REPO"
echo "task ${SLURM_ARRAY_JOB_ID:-?}_${SLURM_ARRAY_TASK_ID} line $LINE: $LABEL seed $SEED on $(hostname)"
exec python -u dev/scripts/svbmc_pool_run.py worker \
    --out "$POOL_DIR" --label "$LABEL" --seed "$SEED"
