# shellcheck shell=bash
# Environment shared by the S-VBMC pool scripts in this directory. Source it
# with REPO set: the submit and finish scripts derive REPO from their own
# location, and the array task receives it through `sbatch --export`,
# because Slurm runs a copy of the batch script from its spool directory,
# so a path relative to the batch script does not reach this file.
#
#   REPO            the PyVBMC checkout (required)
#   POOL_DIR        campaign directory; when it holds a manifest, the
#                   manifest's gpyreg_source is exported as
#                   PYVBMC_GPYREG_SOURCE (optional)
#   POOL_CONDA_ENV  conda environment with the editable PyVBMC install
#                   (default pyvbmc-pool)
#   CONDA_SH        conda's shell hook
#                   (default ~/miniconda3/etc/profile.d/conda.sh)
#
# Every process of a campaign runs single-threaded BLAS with the Agg
# backend, as the campaign plan requires.

if [ -z "${REPO:-}" ]; then
    echo "svbmc_pool_env.sh: REPO must be set before sourcing" >&2
    return 1 2>/dev/null || exit 1
fi
CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}
POOL_CONDA_ENV=${POOL_CONDA_ENV:-pyvbmc-pool}
if [ ! -f "$CONDA_SH" ]; then
    echo "svbmc_pool_env.sh: $CONDA_SH not found; set CONDA_SH" >&2
    return 1 2>/dev/null || exit 1
fi

# conda's activation reads variables that may be unset; relax `set -u`
# around it and restore it afterwards.
_pool_had_u=0
case $- in *u*) _pool_had_u=1 ;; esac
set +u
# shellcheck source=/dev/null
. "$CONDA_SH"
conda activate "$POOL_CONDA_ENV"
if [ "$_pool_had_u" = 1 ]; then
    set -u
fi
unset _pool_had_u

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg

if [ -n "${POOL_DIR:-}" ] && [ -f "$POOL_DIR/manifest.json" ]; then
    PYVBMC_GPYREG_SOURCE=$(python -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["gpyreg_source"])' "$POOL_DIR/manifest.json")
    export PYVBMC_GPYREG_SOURCE
fi
