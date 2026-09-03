#!/usr/bin/env bash
# Regenerate the benchmark profile campaign and the golden-trace reference
# population from scratch, as ONE sequential process (never more than one
# VBMC running; eight concurrent ones hard-crashed the laptop on 2026-09-02).
#
# Usage, from the repo root, in Git Bash or any POSIX shell:
#
#     bash dev/scripts/regenerate_baseline.sh            # everything
#     bash dev/scripts/regenerate_baseline.sh golden     # golden sweep only
#     bash dev/scripts/regenerate_baseline.sh profile    # profile campaign only
#
# Estimated wall time on the 2026-09-02 laptop: profile campaign about 1.5 h
# (plain + cProfile, including the 15-D cigar exhaust run twice), golden
# sweep 14 configs x 20 seeds about 8-10 h. Resumable: rerunning skips
# finished runs. Outputs go to dev/scripts/runs/ (gitignored); at the end the
# reference sidecars and summary are copied into dev/golden/baseline/ (in
# git). Log: dev/scripts/runs/regenerate_<stamp>.log (this script tees).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
[ -x "$PY" ] || PY=python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
STAMP=${STAMP:-$(date +%Y%m%d)}
WHAT=${1:-all}
mkdir -p dev/scripts/runs
LOG=dev/scripts/runs/regenerate_${STAMP}.log
exec > >(tee -a "$LOG") 2>&1
echo "=== regenerate_baseline $WHAT start $(date '+%F %T') stamp $STAMP ==="

echo "--- target checks ---"
$PY dev/scripts/benchmark_targets.py --check --suite all

if [ "$WHAT" = all ] || [ "$WHAT" = profile ]; then
  echo "--- profile campaign, plain $(date '+%T') ---"
  $PY -u dev/scripts/profile_suite.py --suite profile --mode plain \
      --out "dev/scripts/runs/profile_${STAMP}"
  echo "--- profile campaign, cProfile $(date '+%T') ---"
  $PY -u dev/scripts/profile_suite.py --suite profile --mode cprof \
      --out "dev/scripts/runs/profile_${STAMP}"
fi

if [ "$WHAT" = all ] || [ "$WHAT" = golden ]; then
  OUT="dev/scripts/runs/golden/baseline_${STAMP}"
  echo "--- golden sweep $(date '+%T') -> $OUT ---"
  $PY -u dev/scripts/golden_trace.py run --suite golden --seeds 0-19 \
      --workers 1 --out "$OUT"
  echo "--- summary and null check $(date '+%T') ---"
  $PY dev/scripts/golden_trace.py summary "$OUT"
  $PY dev/scripts/golden_trace.py compare --split "$OUT" || true
  echo "--- publish sidecars to dev/golden/baseline ---"
  mkdir -p dev/golden/baseline
  rm -f dev/golden/baseline/*.json dev/golden/baseline/summary.md
  cp "$OUT"/*.json "$OUT"/summary.md dev/golden/baseline/
  ls dev/golden/baseline | wc -l
fi

echo "=== regenerate_baseline $WHAT end $(date '+%F %T') ==="
