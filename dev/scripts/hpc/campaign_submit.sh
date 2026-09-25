#!/bin/bash
# Prepare (once) and submit a campaign of a harness that meets the campaign
# contract (dev/plans/slurm-benchmark-support.md) as Slurm job arrays.
#
#   campaign_submit.sh CAMPAIGN_DIR [prepare flags...]
#
# On the login node. Refuses unless every source tree is clean
# (campaign_env.sh, campaign_check_trees) and the environment matches
# campaign_requirements.txt (campaign_contract.py check-env). With no
# manifest in CAMPAIGN_DIR, runs `$HARNESS prepare --out CAMPAIGN_DIR
# <flags>`, which records the operator settings in the manifest's site
# block; with one, refuses any flag (a prepared campaign is not revised
# here) and any fixed setting that differs from the site block
# (campaign_contract.py check-site). Then writes CAMPAIGN_DIR/cases.txt
# from `$HARNESS cases` once: a later call regenerates the list and refuses
# on any difference, since its line numbers are the case indices of every
# earlier submission. A task whose case has a completion record exits at
# once, so resubmitting any range is safe.
#
# Slurm's MaxArraySize (from `scontrol show config`) caps the largest
# array index, the largest valid one being MaxArraySize-1, so the cases go
# out in chunks of that many: chunk k holds the case indices
# k*(MaxArraySize-1)+1 .. (k+1)*(MaxArraySize-1) as array indices from 1,
# with INDEX_OFFSET=k*(MaxArraySize-1), which the task adds back. One
# submission goes out per chunk that the submitted indices touch, and the
# throttle applies per submission.
#
# Operator settings (environment):
#   HARNESS        the harness, relative to the checkout, e.g.
#                  dev/scripts/svbmc_pool_run.py (required)
#   CAMPAIGN_ENV   the environment's prefix (required; campaign_env.sh)
#   NODE_FEATURE   the Slurm feature of the campaign's node family, which
#                  every task requests with -C (required)
#   PARTITION      -p for every task; omitted when unset
#   PYVBMC_SOURCE, PYVBMC_GPYREG_SOURCE, BASELINE_DIR
#                  the campaign's source trees, where the harness takes them
#   CASES_SUBSET   a subset the harness names (`cases --subset NAME`),
#                  written to CAMPAIGN_DIR/subsets/NAME.txt once like
#                  cases.txt and submitted as its own array, with its own
#                  TIME and MEM
#   ARRAY          case indices to submit, as an sbatch-style list of numbers
#                  and ranges (`ARRAY=1` for a canary, `ARRAY=17,233` or
#                  `ARRAY=2-1100` for a resubmission); with CASES_SUBSET, every
#                  index must be in the subset. Default: every case (of the
#                  subset)
#   THROTTLE       concurrent tasks per submission, the %N of --array
#                  (default 200)
#   TIME, MEM      --time and --mem per task (default 00:30:00 and 2G)
#   SBATCH_EXTRA   further sbatch arguments, split into words
#   CONDA_SETUP, LOGIN_PROFILE   see campaign_env.sh
#
# Every task passes -C $NODE_FEATURE, --hint=nomultithread and
# --cpus-per-task=1, writes CAMPAIGN_DIR/slurm/<job>_<array index>.out, and
# its job id and settings go to CAMPAIGN_DIR/slurm/jobs.txt.
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

case "${1:-}" in
    "" | -h | --help) usage ;;
esac
TARGET=$1
shift

for name in HARNESS CAMPAIGN_ENV NODE_FEATURE; do
    if [ -z "${!name:-}" ]; then
        refuse "$name is not set (the operator settings of" \
            "dev/plans/slurm-benchmark-support.md)"
    fi
done
case $HARNESS in
    /* | ../* | */../* | *..) refuse "HARNESS=$HARNESS must be a path inside" \
        "the checkout, relative to it" ;;
esac
[ -f "$REPO/$HARNESS" ] || refuse "there is no harness at $REPO/$HARNESS"
THROTTLE=${THROTTLE:-200}
TIME=${TIME:-00:30:00}
MEM=${MEM:-2G}
PARTITION=${PARTITION:-}
SBATCH_EXTRA=${SBATCH_EXTRA:-}
CASES_SUBSET=${CASES_SUBSET:-}
ARRAY=${ARRAY:-}
[[ $THROTTLE =~ ^[1-9][0-9]*$ ]] \
    || refuse "THROTTLE=$THROTTLE is not a positive integer"
if [ -n "$CASES_SUBSET" ] && ! [[ $CASES_SUBSET =~ ^[A-Za-z0-9_.+=-]+$ ]]; then
    refuse "CASES_SUBSET=$CASES_SUBSET is not a subset name"
fi
if [ -n "$ARRAY" ] \
    && ! [[ $ARRAY =~ ^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$ ]]; then
    refuse "ARRAY=$ARRAY is not a list of case indices and ranges"
fi
# The harness's prepare records these in the manifest's site block, and
# every task receives them through --export=ALL.
export HARNESS CAMPAIGN_ENV NODE_FEATURE PARTITION LOGIN_SETUP CONDA_SETUP \
    LOGIN_PROFILE PYVBMC_SOURCE PYVBMC_GPYREG_SOURCE BASELINE_DIR \
    CASES_SUBSET THROTTLE TIME MEM ARRAY SBATCH_EXTRA
unset CAMPAIGN_DIR CAMPAIGN_STEP INDEX_OFFSET

# shellcheck source=campaign_env.sh
source "$HERE/campaign_env.sh"
campaign_check_trees
cd "$REPO"
python "$CONTRACT" check-env --requirements "$CAMPAIGN_REQUIREMENTS" \
    || refuse "the environment $CAMPAIGN_ENV differs from $CAMPAIGN_REQUIREMENTS"

# The campaign directory: outside the checkout, or ignored by it, so that
# the clean-tree check never sees it.
created=0
[ -d "$TARGET" ] || created=1
mkdir -p "$TARGET"
CAMPAIGN_DIR=$(cd "$TARGET" && pwd -P)
case "$CAMPAIGN_DIR/" in
    "$(cd "$REPO" && pwd -P)/"*)
        if ! git -C "$REPO" check-ignore -q "$CAMPAIGN_DIR"; then
            [ "$created" = 0 ] || rmdir "$CAMPAIGN_DIR"
            refuse "$CAMPAIGN_DIR is inside the checkout and not ignored" \
                "by it; use dev/scripts/runs/ or a directory outside it"
        fi
        ;;
esac
export CAMPAIGN_DIR
campaign_tmpdir
mkdir -p "$CAMPAIGN_DIR/slurm"

if [ -f "$CAMPAIGN_DIR/manifest.json" ]; then
    if [ $# -gt 0 ]; then
        refuse "$CAMPAIGN_DIR is prepared; a campaign is not revised here," \
            "so it takes no prepare flags"
    fi
else
    python -u "$REPO/$HARNESS" prepare --out "$CAMPAIGN_DIR" "$@"
    [ -f "$CAMPAIGN_DIR/manifest.json" ] \
        || refuse "$HARNESS prepare wrote no $CAMPAIGN_DIR/manifest.json"
fi
python "$CONTRACT" check-site --manifest "$CAMPAIGN_DIR/manifest.json" \
    || refuse "the settings differ from those $CAMPAIGN_DIR was prepared with"

# write_once FILE: puts FILE.new in place as FILE, or refuses when FILE
# exists with other content.
write_once() {
    local file=$1
    if [ -f "$file" ]; then
        if ! cmp -s "$file" "$file.new"; then
            echo "refusing: the harness's list no longer matches $file," \
                "whose line numbers are the case indices of earlier" \
                "submissions:" >&2
            diff "$file" "$file.new" >&2 || true
            rm -f "$file.new"
            exit 1
        fi
        rm -f "$file.new"
    else
        mv "$file.new" "$file"
    fi
}

CASES="$CAMPAIGN_DIR/cases.txt"
python "$REPO/$HARNESS" cases --out "$CAMPAIGN_DIR" > "$CASES.new" \
    || { rm -f "$CASES.new"; refuse "$HARNESS cases failed"; }
python "$CONTRACT" check-cases --cases "$CASES.new" \
    || { rm -f "$CASES.new"; refuse "$HARNESS cases breaks the contract"; }
write_once "$CASES"
N=$(($(wc -l < "$CASES")))

SUBSET_INDICES=""
if [ -n "$CASES_SUBSET" ]; then
    mkdir -p "$CAMPAIGN_DIR/subsets"
    SUBSET="$CAMPAIGN_DIR/subsets/$CASES_SUBSET.txt"
    if ! python "$REPO/$HARNESS" cases --out "$CAMPAIGN_DIR" \
        --subset "$CASES_SUBSET" > "$SUBSET.new"; then
        rm -f "$SUBSET.new"
        refuse "$HARNESS has no subset $CASES_SUBSET"
    fi
    if ! SUBSET_INDICES=$(python "$CONTRACT" check-cases --cases "$CASES" \
        --subset "$SUBSET.new"); then
        rm -f "$SUBSET.new"
        refuse "the subset $CASES_SUBSET breaks the contract"
    fi
    write_once "$SUBSET"
fi

# "1,5-7" -> one case index per line.
expand_indices() {
    local -a items
    local item a b
    IFS=, read -ra items <<< "$1"
    for item in "${items[@]}"; do
        case $item in
            *-*)
                a=${item%-*}
                b=${item#*-}
                if [ "$a" -gt "$b" ]; then
                    refuse "ARRAY=$ARRAY holds the empty range $item"
                fi
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

if [ -n "$ARRAY" ]; then
    indices=$(expand_indices "$ARRAY" | sort -n | uniq)
    [ -n "$indices" ] || refuse "ARRAY=$ARRAY names no case"
    if [ "$(echo "$indices" | head -n 1)" -lt 1 ] \
        || [ "$(echo "$indices" | tail -n 1)" -gt "$N" ]; then
        refuse "ARRAY=$ARRAY is outside the $N cases of $CASES"
    fi
    if [ -n "$CASES_SUBSET" ]; then
        outside=$(awk 'NR == FNR { s[$1] = 1; next } !($1 in s)' \
            <(echo "$SUBSET_INDICES") <(echo "$indices") | compress_ranges)
        if [ -n "$outside" ]; then
            refuse "ARRAY=$ARRAY names cases $outside, which are not in" \
                "the subset $CASES_SUBSET"
        fi
    fi
elif [ -n "$CASES_SUBSET" ]; then
    indices=$SUBSET_INDICES
else
    indices=$(seq 1 "$N")
fi

MAXA=$({ scontrol show config 2>/dev/null || true; } \
    | sed -n 's/^MaxArraySize *= *\([0-9][0-9]*\).*/\1/p' | head -n 1)
if ! [[ ${MAXA:-} =~ ^[0-9]+$ ]] || [ "$MAXA" -lt 2 ]; then
    refuse "cannot read MaxArraySize from 'scontrol show config'"
fi
LIMIT=$((MAXA - 1)) # the largest valid array index, hence the chunk size

COUNT=$(echo "$indices" | wc -l)
DONE=0
while read -r tag; do
    if [ -f "$CAMPAIGN_DIR/records/$tag.complete.json" ]; then
        DONE=$((DONE + 1))
    fi
done < <(awk 'NR == FNR { want[$1] = 1; next } (FNR in want) { print $1 }' \
    <(echo "$indices") "$CASES")
echo "$N cases in $CASES; with MaxArraySize $MAXA the campaign takes" \
    "$(((N + LIMIT - 1) / LIMIT)) chunk(s) of at most $LIMIT cases"
echo "submitting $((COUNT)) cases${CASES_SUBSET:+ of the subset $CASES_SUBSET};" \
    "$DONE of them already have a completion record (their tasks exit at once)"

JOB_NAME=$(basename "$CAMPAIGN_DIR" | tr -c 'A-Za-z0-9_.+=\n-' '_')
PARTITION_ARGS=()
if [ -n "$PARTITION" ]; then
    PARTITION_ARGS=(-p "$PARTITION")
fi

submit() {
    local spec=$1 offset=$2 jid
    # SBATCH_EXTRA is meant to word-split.
    # shellcheck disable=SC2086
    jid=$(INDEX_OFFSET=$offset sbatch --parsable -J "$JOB_NAME" \
        -C "$NODE_FEATURE" --hint=nomultithread \
        ${PARTITION_ARGS[@]+"${PARTITION_ARGS[@]}"} \
        --cpus-per-task=1 --mem="$MEM" --time="$TIME" \
        --output "$CAMPAIGN_DIR/slurm/%A_%a.out" \
        --array="${spec}%${THROTTLE}" --export=ALL \
        $SBATCH_EXTRA "$HERE/campaign_task.sbatch" < /dev/null)
    jid=${jid%%;*}
    [[ $jid =~ ^[0-9]+$ ]] || refuse "sbatch printed no job id: $jid"
    echo "$jid array=$spec offset=$offset subset=${CASES_SUBSET:--}" \
        "throttle=$THROTTLE time=$TIME mem=$MEM partition=${PARTITION:--}" \
        "$(date +%FT%T)" >> "$CAMPAIGN_DIR/slurm/jobs.txt"
    echo "submitted job $jid: --array=${spec}%${THROTTLE} (index offset $offset)"
}

chunks=$(echo "$indices" | awk -v L="$LIMIT" '{ print int(($1 - 1) / L) }' \
    | sort -un)
for chunk in $chunks; do
    offset=$((chunk * LIMIT))
    spec=$(echo "$indices" | awk -v L="$LIMIT" -v c="$chunk" -v o="$offset" \
        'int(($1 - 1) / L) == c { print $1 - o }' | compress_ranges)
    submit "$spec" "$offset"
done
echo "watch with: squeue -u \$USER -n $JOB_NAME"
