#!/bin/bash
# Prepare (once) and submit an S-VBMC run-pool campaign as a Slurm array.
#
#   svbmc_pool_submit.sh POOL_DIR [prepare flags...]
#
# On the login node. With no manifest in POOL_DIR, runs
# `svbmc_pool_run.py prepare --out POOL_DIR <flags>` (the suite defaults
# to svbmc_pool; pass `--gpyreg-source PATH`, and for a smoke test
# something like `--suite smoke --only normal_D2 --target 2 --max-seeds 3`);
# with a manifest, refuses any flag, since a prepared campaign is not
# revised here. Then writes POOL_DIR/cases.txt once (a later call
# regenerates it to a temporary file and refuses on any difference: the
# line numbers are the case indices of every earlier submission) and
# submits svbmc_pool_task.sbatch. A task whose case already has a
# completion record exits at once, so resubmitting any range is safe.
#
# Case indices are the lines of cases.txt. Slurm's MaxArraySize caps the
# largest array index (the largest valid one is MaxArraySize-1), so the
# cases go out in chunks of that many: chunk k carries
# INDEX_OFFSET=k*(MaxArraySize-1) and array indices 1..chunk size, and
# the task adds the offset to its array index to find its line.
#
# Environment (all optional):
#   ARRAY         explicit case indices to submit, as an sbatch-style list
#                 of numbers and ranges (`ARRAY=1` for a canary,
#                 `ARRAY=17,233` or `ARRAY=2-1100` for a resubmission),
#                 mapped onto the chunks above, one submission per chunk
#                 touched. Default: every line of cases.txt.
#   THROTTLE      concurrent tasks per submission, the %N of --array
#                 (default 200)
#   TIME          --time per task (default 00:30:00)
#   MEM           --mem per task (default 2G; DefMemPerCPU is 512M)
#   PARTITION     -p (default short)
#   SBATCH_EXTRA  further sbatch arguments, e.g. --constraint=amd
#   POOL_CONDA_ENV, CONDA_SH   see svbmc_pool_env.sh
#
# The identity every worker is compared against is fixed at `prepare`,
# which does not refuse a dirty tree, so this script refuses one: the
# checkout must be clean and must not move until the array is done.
set -euo pipefail

usage() {
    sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//' | head -n -1 >&2
    exit 64
}

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
export REPO

case "${1:-}" in
    "" | -h | --help) usage ;;
esac
mkdir -p "$1"
POOL_DIR=$(cd "$1" && pwd)
export POOL_DIR
shift

THROTTLE=${THROTTLE:-200}
TIME=${TIME:-00:30:00}
MEM=${MEM:-2G}
PARTITION=${PARTITION:-short}
SBATCH_EXTRA=${SBATCH_EXTRA:-}

if [ -n "$(git -C "$REPO" status --porcelain)" ]; then
    echo "refusing: $REPO has uncommitted or untracked changes:" >&2
    git -C "$REPO" status --short >&2
    exit 1
fi
echo "checkout $(git -C "$REPO" rev-parse --abbrev-ref HEAD) at $(git -C "$REPO" rev-parse HEAD)"

# shellcheck source=svbmc_pool_env.sh
source "$HERE/svbmc_pool_env.sh"
cd "$REPO"

if [ -f "$POOL_DIR/manifest.json" ]; then
    if [ $# -gt 0 ]; then
        echo "refusing: $POOL_DIR is prepared; a campaign is not revised" \
            "here (run svbmc_pool_run.py prepare yourself)" >&2
        exit 1
    fi
else
    python -u dev/scripts/svbmc_pool_run.py prepare --out "$POOL_DIR" "$@"
    PYVBMC_GPYREG_SOURCE=$(python -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["gpyreg_source"])' "$POOL_DIR/manifest.json")
    export PYVBMC_GPYREG_SOURCE
fi
echo "gpyreg source: $PYVBMC_GPYREG_SOURCE"

if [ -f "$POOL_DIR/cases.txt" ]; then
    python dev/scripts/svbmc_pool_run.py cases --out "$POOL_DIR" \
        > "$POOL_DIR/cases.txt.new"
    if ! cmp -s "$POOL_DIR/cases.txt" "$POOL_DIR/cases.txt.new"; then
        echo "refusing: the allocation no longer matches $POOL_DIR/cases.txt," \
            "whose line numbers are the indices of earlier submissions" >&2
        diff "$POOL_DIR/cases.txt" "$POOL_DIR/cases.txt.new" >&2 || true
        rm -f "$POOL_DIR/cases.txt.new"
        exit 1
    fi
    rm -f "$POOL_DIR/cases.txt.new"
else
    python dev/scripts/svbmc_pool_run.py cases --out "$POOL_DIR" \
        > "$POOL_DIR/cases.txt"
fi
N=$(wc -l < "$POOL_DIR/cases.txt")
# A fresh campaign has no records/ yet (the first worker creates it), and
# under `set -o pipefail` a failing `ls` piped into `wc` would end the
# script here.
DONE=0
if [ -d "$POOL_DIR/records" ]; then
    DONE=$(ls "$POOL_DIR/records" | wc -l)
fi
echo "$N cases in $POOL_DIR/cases.txt; $DONE already have a completion" \
    "record (their tasks exit at once)"
mkdir -p "$POOL_DIR/slurm"

MAXA=$(scontrol show config 2>/dev/null | sed -n 's/^MaxArraySize *= *//p' || true)
MAXA=${MAXA:-1001}
LIMIT=$((MAXA - 1)) # the largest valid array index, hence the chunk size

submit() {
    local spec=$1 offset=$2 jid
    # SBATCH_EXTRA is meant to word-split.
    # shellcheck disable=SC2086
    jid=$(sbatch --parsable -J svbmc_pool -p "$PARTITION" \
        --cpus-per-task=1 --mem="$MEM" --time="$TIME" \
        --output "$POOL_DIR/slurm/%A_%a.out" \
        --array="${spec}%${THROTTLE}" \
        --export=ALL,POOL_DIR="$POOL_DIR",REPO="$REPO",INDEX_OFFSET="$offset" \
        $SBATCH_EXTRA "$HERE/svbmc_pool_task.sbatch")
    jid=${jid%%;*}
    echo "$jid array=$spec offset=$offset throttle=$THROTTLE time=$TIME mem=$MEM $(date +%FT%T)" \
        >> "$POOL_DIR/slurm/jobs.txt"
    echo "submitted job $jid: --array=$spec (index offset $offset)"
}

# "1,5-7" -> one case index per line.
expand_indices() {
    local -a items
    local item a b
    IFS=, read -ra items <<< "$1"
    for item in "${items[@]}"; do
        case $item in
            "") ;;
            *-*)
                a=${item%-*}
                b=${item#*-}
                seq "$a" "$b"
                ;;
            *) echo "$item" ;;
        esac
    done
}

# Sorted integers on stdin -> the shortest sbatch list, "a-b,c,d-e".
compress_ranges() {
    awk 'NR == 1 { s = $1; p = $1; next }
         $1 == p + 1 { p = $1; next }
         { out = out (out ? "," : "") (s == p ? s : s "-" p); s = $1; p = $1 }
         END { if (NR) print out (out ? "," : "") (s == p ? s : s "-" p) }'
}

if [ -n "${ARRAY:-}" ]; then
    # Case indices, mapped onto the chunks: chunk k holds the case indices
    # k*LIMIT+1 .. (k+1)*LIMIT as array indices 1..LIMIT with offset k*LIMIT.
    indices=$(expand_indices "$ARRAY" | sort -n | uniq)
    if [ -z "$indices" ]; then
        echo "ARRAY=$ARRAY names no case" >&2
        exit 1
    fi
    if [ "$(echo "$indices" | head -n 1)" -lt 1 ] \
        || [ "$(echo "$indices" | tail -n 1)" -gt "$N" ]; then
        echo "ARRAY=$ARRAY is outside the $N cases of $POOL_DIR/cases.txt" >&2
        exit 1
    fi
    for chunk in $(echo "$indices" | awk -v L="$LIMIT" '{ print int(($1 - 1) / L) }' | sort -un); do
        offset=$((chunk * LIMIT))
        spec=$(echo "$indices" | awk -v L="$LIMIT" -v c="$chunk" -v o="$offset" \
            'int(($1 - 1) / L) == c { print $1 - o }' | compress_ranges)
        submit "$spec" "$offset"
    done
else
    offset=0
    while [ "$offset" -lt "$N" ]; do
        k=$((N - offset))
        if [ "$k" -gt "$LIMIT" ]; then
            k=$LIMIT
        fi
        submit "1-$k" "$offset"
        offset=$((offset + k))
    done
fi
echo "watch with: squeue -u $USER -n svbmc_pool"
