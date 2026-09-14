#!/bin/bash
# Account for, verify, select, summarize and archive a finished S-VBMC
# run-pool campaign.
#
#   svbmc_pool_finish.sh POOL_DIR [--no-archive] [--allow-missing]
#                                 [--allow-running]
#
# On the login node. Writes the Slurm accounting of every submitted job
# to POOL_DIR/slurm/sacct.txt, runs `svbmc_pool_run.py verify` under
# `srun` (every artifact re-checked, the allocation reconciled, the
# result in POOL_DIR/verification.json), and stops if verification fails
# or if cases are missing, printing the array indices to resubmit with
# `ARRAY=... svbmc_pool_submit.sh POOL_DIR` (raise TIME or MEM when Slurm
# killed them). It also stops while tasks of the recorded jobs are still
# queued or running. --allow-missing goes on past the missing cases;
# --allow-running goes on past the queue check, for a look at a campaign
# still in flight, whose selection and summary then leave out the runs
# in flight and which is never archived. Then `select` and `summarize`,
# and, unless --no-archive, the whole directory as
# <parent>/<name>.tar.zst with its SHA-256 and size.
#
# Environment (all optional): PARTITION (short), VERIFY_TIME (01:00:00)
# and VERIFY_MEM (2G) for the srun that runs `verify`, POOL_CONDA_ENV,
# CONDA_SH (see svbmc_pool_env.sh).
set -euo pipefail

usage() {
    sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//' | head -n -1 >&2
    exit 64
}

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
export REPO

if [ $# -lt 1 ] || [ ! -f "$1/manifest.json" ]; then
    usage
fi
POOL_DIR=$(cd "$1" && pwd)
export POOL_DIR
shift
ARCHIVE=1
ALLOW_MISSING=0
ALLOW_RUNNING=0
for arg in "$@"; do
    case $arg in
        --no-archive) ARCHIVE=0 ;;
        --allow-missing) ALLOW_MISSING=1 ;;
        --allow-running) ALLOW_RUNNING=1 ;;
        *) usage ;;
    esac
done

# shellcheck source=svbmc_pool_env.sh
source "$HERE/svbmc_pool_env.sh"
cd "$REPO"

# 1. Accounting of every job the submit script recorded.
if [ -s "$POOL_DIR/slurm/jobs.txt" ]; then
    JOBS=$(awk '{print $1}' "$POOL_DIR/slurm/jobs.txt" | paste -sd,)
    # One query per job: an id that has aged out of the queue makes
    # squeue fail, which must not hide another job's running tasks.
    running=""
    for job in $(awk '{print $1}' "$POOL_DIR/slurm/jobs.txt"); do
        tasks=$(squeue -h -j "$job" -o "%i %T %M" 2>/dev/null || true)
        if [ -n "$tasks" ]; then
            running+="$tasks"$'\n'
        fi
    done
    if [ -n "$running" ]; then
        echo "tasks of the recorded jobs are still queued or running:" >&2
        printf '%s' "$running" >&2
        if [ "$ALLOW_RUNNING" = 0 ]; then
            exit 1
        fi
    fi
    sacct -P --units=M -j "$JOBS" \
        --format=JobID,State,ExitCode,Elapsed,MaxRSS,AllocCPUS,NodeList \
        > "$POOL_DIR/slurm/sacct.txt" || echo "sacct failed" >&2
    echo "accounting in $POOL_DIR/slurm/sacct.txt"
fi

# 2. Verification under srun: it rebuilds every artifact's posterior and
#    GP and re-runs the recomputation gate, which is compute.
rc=0
srun -p "${PARTITION:-short}" -c 1 --mem="${VERIFY_MEM:-2G}" \
    -t "${VERIFY_TIME:-01:00:00}" -J svbmc_pool_verify \
    python -u dev/scripts/svbmc_pool_run.py verify --out "$POOL_DIR" || rc=$?
if [ "$rc" -ne 0 ]; then
    echo "verify exited $rc; fix or remove the reported cases first" >&2
    exit "$rc"
fi
MISSING=$(python - "$POOL_DIR/verification.json" <<'EOF'
import json, sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
counts = report["counts"]
print(
    "verified {verified}, failed {failed}, partial {partial}, "
    "missing {missing}, stray {stray}".format(**counts),
    file=sys.stderr,
)
for status in ("missing", "failed"):
    indices = [c["index"] for c in report["cases"] if c["status"] == status]
    if indices:
        print(
            f"{status} indices: " + ",".join(str(i) for i in indices),
            file=sys.stderr,
        )
print(counts["missing"])
EOF
)
if [ "$MISSING" -gt 0 ] && [ "$ALLOW_MISSING" = 0 ]; then
    echo "$MISSING cases are missing (tasks Slurm killed or never ran);" \
        "resubmit them with ARRAY=<indices> svbmc_pool_submit.sh $POOL_DIR" \
        "(raise TIME or MEM if a limit killed them), or pass --allow-missing" >&2
    exit 3
fi

# 3. The filtered pool and the summary (light; the login node is fine).
python -u dev/scripts/svbmc_pool_run.py select --out "$POOL_DIR"
python -u dev/scripts/svbmc_pool_run.py summarize --out "$POOL_DIR"

# 4. The archive: the whole directory, next to it. The artifacts are
#    already compressed, so a light zstd setting is enough and the shared
#    login node is not loaded for nothing.
if [ "$ARCHIVE" = 1 ]; then
    if [ -n "${running:-}" ]; then
        echo "not archiving a campaign whose tasks are still queued or" \
            "running; run again when the queue is empty" >&2
        exit 1
    fi
    if ! command -v zstd >/dev/null 2>&1; then
        echo "zstd is not on the PATH of the $POOL_CONDA_ENV environment" \
            "(conda activate drops the base environment's bin); install it" \
            "there with 'conda install -n $POOL_CONDA_ENV zstd'" >&2
        exit 1
    fi
    NAME=$(basename "$POOL_DIR")
    PARENT=$(dirname "$POOL_DIR")
    OUT="$PARENT/$NAME.tar.zst"
    tar -C "$PARENT" -cf - "$NAME" | zstd -T2 -3 -q -f -o "$OUT"
    sha256sum "$OUT"
    ls -l "$OUT"
fi
