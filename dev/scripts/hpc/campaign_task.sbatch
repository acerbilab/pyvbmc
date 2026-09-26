#!/bin/bash
# One Slurm task of a campaign that campaign_submit.sh or campaign_finish.sh
# submitted, for a harness that meets the campaign contract
# (dev/plans/slurm-benchmark-support.md). The submitter passes CAMPAIGN_DIR,
# REPO, HARNESS and the operator settings through `--export=ALL` and puts
# the resources, the node feature and the output path on the command line
# (an #SBATCH line cannot take a variable).
#
# As an array task it runs one case: line INDEX_OFFSET + SLURM_ARRAY_TASK_ID
# of CAMPAIGN_DIR/cases.txt. When that case has a completion record,
# records/<tag>.complete.json with the tag the line's first field, it
# exits 0 at once, before the environment is even activated; otherwise it
# runs `$HARNESS worker --out CAMPAIGN_DIR --case <line>` and exits with the
# worker's code: 0 done, 1 failed (the case's <tag>.error.txt holds the
# traceback), 64 a case line or a directory that is not the harness's, 75
# claimed by a live task, 78 an identity that cannot be established (a
# source tree git cannot read, the node's features that `scontrol show
# node` does not give after its retries) or whose source part differs from
# the manifest's (these three leaving the case's files as they are) and 128
# plus the signal's number when a signal stopped it. The script execs
# the worker, so that the signals Slurm sends reach it. At the time limit
# or on scancel Slurm sends SIGTERM, and the worker removes the case's
# partial files and its claim before it exits: `verify` reports the case as
# missing. A task killed outright (SIGKILL after KillWait, an out-of-memory
# kill) leaves its claim and whatever files it had written, possibly none:
# `verify` reports the case as interrupted, and the next worker finds the
# claim stale in the accounting, takes it over and removes the files
# before it runs.
#
# With CAMPAIGN_STEP set it runs one step on the whole directory instead,
# `$HARNESS $CAMPAIGN_STEP --out CAMPAIGN_DIR` (verify, a finishing step),
# as campaign_finish.sh submits it.
#
# The redaction of the campaign's tracked copies (campaign_redact.sh) finds
# the host of every task and step in the two lines below that end in
# "on $(hostname)" (campaign_contract.LOG_HOST_LINES), so a change to their
# form changes those patterns too.
set -euo pipefail

: "${CAMPAIGN_DIR:?CAMPAIGN_DIR must be exported by the submitter}"
: "${REPO:?REPO must be exported by the submitter}"
: "${HARNESS:?HARNESS must be exported by the submitter}"

if [ -n "${CAMPAIGN_STEP:-}" ]; then
    # shellcheck source=campaign_env.sh
    source "$REPO/dev/scripts/hpc/campaign_env.sh"
    cd "$REPO"
    echo "step '$CAMPAIGN_STEP' of $CAMPAIGN_DIR: job ${SLURM_JOB_ID:-?} on $(hostname)"
    # The step is its words.
    # shellcheck disable=SC2086
    exec python -u "$REPO/$HARNESS" $CAMPAIGN_STEP --out "$CAMPAIGN_DIR"
fi

: "${SLURM_ARRAY_TASK_ID:?this script runs as a Slurm array task, or with CAMPAIGN_STEP set}"
LINE=$((SLURM_ARRAY_TASK_ID + ${INDEX_OFFSET:-0}))
CASE=$(sed -n "${LINE}p" "$CAMPAIGN_DIR/cases.txt")
if [ -z "$CASE" ]; then
    echo "no case on line $LINE of $CAMPAIGN_DIR/cases.txt" >&2
    exit 2
fi
TAG=${CASE%% *}
TASK="${SLURM_ARRAY_JOB_ID:-?}_${SLURM_ARRAY_TASK_ID}"
# A completed case is never regenerated: resubmitting a range that includes
# finished cases costs nothing but these lines.
if [ -f "$CAMPAIGN_DIR/records/$TAG.complete.json" ]; then
    echo "task $TASK, case $LINE: $TAG already has a completion record; nothing to do"
    exit 0
fi

# shellcheck source=campaign_env.sh
source "$REPO/dev/scripts/hpc/campaign_env.sh"
cd "$REPO"
echo "task $TASK, case $LINE: $CASE, on $(hostname)"
exec python -u "$REPO/$HARNESS" worker --out "$CAMPAIGN_DIR" --case "$CASE"
