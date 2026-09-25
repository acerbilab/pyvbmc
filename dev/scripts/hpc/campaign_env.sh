# shellcheck shell=bash
# The environment of every process of a campaign that the scripts in this
# directory drive (dev/plans/slurm-benchmark-support.md, "The driver").
# Sourced with REPO set: campaign_submit.sh and campaign_finish.sh derive
# REPO from their own location, and campaign_task.sbatch receives it
# through `sbatch --export=ALL`, because Slurm runs a copy of a batch script
# from its spool directory, so a path relative to the batch script does not
# reach this file. Executed as
#
#   bash dev/scripts/hpc/campaign_env.sh build
#
# on the login node, it builds the environment instead: it runs
# LOGIN_SETUP, defines conda as below, creates CAMPAIGN_ENV (refusing an
# existing prefix) from conda-forge alone with the Python that the
# `# python==` line of campaign_requirements.txt pins, zstd and gh, installs
# campaign_requirements.txt with pip, prints the installed versions as
# pins (`pip list --format=freeze --exclude-editable`) and checks the
# environment against the file (campaign_contract.py check-env).
#
# Settings read here; every value particular to a site is one of them:
#   CAMPAIGN_ENV   the prefix of the campaign's conda environment (required)
#   CONDA_SETUP    optional commands that define `conda`, such as loading a
#                  site's conda module or sourcing a Miniforge installation's
#                  etc/profile.d/conda.sh
#   LOGIN_PROFILE  the login profile, read when `module` is not defined, as
#                  in a non-interactive shell (default /etc/profile)
#   LOGIN_SETUP    optional commands the build runs first, on the login
#                  node, for the steps that need the network
#   PYVBMC_SOURCE, PYVBMC_GPYREG_SOURCE, BASELINE_DIR
#                  the campaign's source trees, exported for the harness
#   CAMPAIGN_DIR   the campaign directory; TMPDIR goes to its tmp/, since a
#                  compute node may have no local disk
#
# Sourcing activates the environment with `conda activate` when conda is
# defined, and otherwise, saying so, by putting CAMPAIGN_ENV/bin first on
# the PATH; it fails unless `python` is then the environment's. It exports
# single-threaded BLAS, MPLBACKEND=Agg and PYTHONNOUSERSITE=1, so that no
# package in the operator's user site shadows the environment's. It also
# defines campaign_check_trees, the refusal of a dirty source tree that the
# submission and the finish share, and campaign_tmpdir.

_campaign_env_build=0
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    set -euo pipefail
    if [ "${1:-}" != build ]; then
        sed -n '2,/^_campaign_env_build=0/p' "$0" | sed 's/^# \{0,1\}//' \
            | head -n -2 >&2
        exit 64
    fi
    _campaign_env_build=1
    REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
fi

campaign_env_fail() {
    echo "campaign_env.sh: $*" >&2
    return 1
}

if [ -z "${REPO:-}" ]; then
    campaign_env_fail "REPO must be set before sourcing"
    return 1 2>/dev/null || exit 1
fi
if [ -z "${CAMPAIGN_ENV:-}" ]; then
    campaign_env_fail "CAMPAIGN_ENV, the prefix of the campaign's conda" \
        "environment, is not set"
    return 1 2>/dev/null || exit 1
fi
CAMPAIGN_REQUIREMENTS="$REPO/dev/scripts/hpc/campaign_requirements.txt"

# The login profile, the conda setup, LOGIN_SETUP and conda itself may read
# unset variables or fail on the way; they run with errexit and nounset
# relaxed, and the caller's options are restored after each.
campaign_relax() {
    _campaign_flags=$-
    set +eu
}
campaign_restore() {
    case $_campaign_flags in *e*) set -e ;; esac
    case $_campaign_flags in *u*) set -u ;; esac
    return 0
}
campaign_relax
_campaign_rc=0
if [ "$_campaign_env_build" = 1 ] && [ -n "${LOGIN_SETUP:-}" ]; then
    echo "campaign_env.sh: running LOGIN_SETUP"
    eval "$LOGIN_SETUP" || _campaign_rc=$?
fi
if [ "$_campaign_rc" = 0 ] && ! type module >/dev/null 2>&1; then
    _campaign_profile=${LOGIN_PROFILE:-/etc/profile}
    if [ -r "$_campaign_profile" ]; then
        # shellcheck source=/dev/null
        . "$_campaign_profile"
    fi
fi
if [ "$_campaign_rc" = 0 ] && [ -n "${CONDA_SETUP:-}" ]; then
    eval "$CONDA_SETUP" || _campaign_rc=$?
fi
campaign_restore
if [ "$_campaign_rc" != 0 ]; then
    campaign_env_fail "LOGIN_SETUP or CONDA_SETUP failed (exit $_campaign_rc)"
    return 1 2>/dev/null || exit 1
fi

if [ "$_campaign_env_build" = 1 ]; then
    if [ -e "$CAMPAIGN_ENV" ]; then
        campaign_env_fail "$CAMPAIGN_ENV exists; the build makes a new" \
            "environment and never changes one"
        exit 1
    fi
    if ! type conda >/dev/null 2>&1; then
        campaign_env_fail "conda is not defined; set CONDA_SETUP"
        exit 1
    fi
    _campaign_python=$(sed -n 's/^# *python==\([0-9.]*\) *$/\1/p' \
        "$CAMPAIGN_REQUIREMENTS" | head -n 1)
    if [ -z "$_campaign_python" ]; then
        campaign_env_fail "$CAMPAIGN_REQUIREMENTS has no '# python==' line"
        exit 1
    fi
    conda create -y -p "$CAMPAIGN_ENV" --override-channels -c conda-forge \
        "python=$_campaign_python" zstd gh
fi

campaign_relax
if type conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook 2>/dev/null)"
    conda activate "$CAMPAIGN_ENV"
    _campaign_rc=$?
else
    echo "campaign_env.sh: conda is not defined (set CONDA_SETUP);" \
        "activating $CAMPAIGN_ENV by putting its bin first on the PATH" >&2
    if _campaign_bin=$(cd "$CAMPAIGN_ENV/bin" 2>/dev/null && pwd); then
        PATH="$_campaign_bin:$PATH"
    fi
    CONDA_PREFIX=$CAMPAIGN_ENV
    export PATH CONDA_PREFIX
    _campaign_rc=0
fi
campaign_restore
if [ "$_campaign_rc" != 0 ]; then
    campaign_env_fail "conda activate $CAMPAIGN_ENV failed"
    return 1 2>/dev/null || exit 1
fi
_campaign_python=$(command -v python || true)
if [ -z "$_campaign_python" ] \
    || [ "$(cd "$(dirname "$_campaign_python")" && pwd -P)" \
        != "$(cd "$CAMPAIGN_ENV/bin" 2>/dev/null && pwd -P)" ]; then
    campaign_env_fail "python is ${_campaign_python:-not on the PATH}," \
        "not the environment's $CAMPAIGN_ENV/bin/python"
    return 1 2>/dev/null || exit 1
fi
unset _campaign_flags _campaign_rc _campaign_profile _campaign_python \
    _campaign_bin

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONNOUSERSITE=1
for _campaign_tree in PYVBMC_SOURCE PYVBMC_GPYREG_SOURCE BASELINE_DIR; do
    if [ -n "${!_campaign_tree:-}" ]; then
        export "${_campaign_tree?}"
    fi
done
unset _campaign_tree

# TMPDIR under the campaign directory.
campaign_tmpdir() {
    TMPDIR="$CAMPAIGN_DIR/tmp"
    mkdir -p "$TMPDIR"
    export TMPDIR
}
if [ -n "${CAMPAIGN_DIR:-}" ]; then
    campaign_tmpdir
fi

# campaign_check_tree NAME PATH: refuses unless PATH is the top of a git
# checkout with no uncommitted or untracked change; prints its commit.
campaign_check_tree() {
    local name=$1 path=$2 cdup status
    if ! cdup=$(git -C "$path" rev-parse --show-cdup 2>/dev/null); then
        echo "refusing: $name=$path is not a git checkout" >&2
        return 1
    fi
    if [ -n "$cdup" ]; then
        echo "refusing: $name=$path is not the top of its checkout" >&2
        return 1
    fi
    if ! status=$(git -C "$path" status --porcelain); then
        echo "refusing: git status fails in $name=$path" >&2
        return 1
    fi
    if [ -n "$status" ]; then
        echo "refusing: $name=$path has uncommitted or untracked changes:" >&2
        git -C "$path" status --short >&2
        return 1
    fi
    echo "$name: $(git -C "$path" rev-parse HEAD) $path"
}

# campaign_check_trees: every source tree of the campaign must be clean and
# must not move until its tasks are done, since every worker compares the
# identity that `prepare` fixed. The trees are REPO (the harness checkout)
# and, where set, PYVBMC_SOURCE, PYVBMC_GPYREG_SOURCE and BASELINE_DIR, or,
# when BASELINE_DIR is not itself a checkout, every checkout directly
# inside it.
campaign_check_trees() {
    local name dir cdup found
    campaign_check_tree REPO "$REPO" || return 1
    for name in PYVBMC_SOURCE PYVBMC_GPYREG_SOURCE; do
        if [ -n "${!name:-}" ]; then
            campaign_check_tree "$name" "${!name}" || return 1
        fi
    done
    if [ -z "${BASELINE_DIR:-}" ]; then
        return 0
    fi
    if cdup=$(git -C "$BASELINE_DIR" rev-parse --show-cdup 2>/dev/null) \
        && [ -z "$cdup" ]; then
        campaign_check_tree BASELINE_DIR "$BASELINE_DIR"
        return
    fi
    found=0
    for dir in "$BASELINE_DIR"/*/; do
        if [ -d "$dir" ] \
            && cdup=$(git -C "$dir" rev-parse --show-cdup 2>/dev/null) \
            && [ -z "$cdup" ]; then
            campaign_check_tree BASELINE_DIR "${dir%/}" || return 1
            found=1
        fi
    done
    if [ "$found" = 0 ]; then
        echo "refusing: BASELINE_DIR=$BASELINE_DIR holds no git checkout" >&2
        return 1
    fi
}

if [ "$_campaign_env_build" = 1 ]; then
    python -m pip install -r "$CAMPAIGN_REQUIREMENTS"
    python -m pip list --format=freeze --exclude-editable
    python "$REPO/dev/scripts/campaign_contract.py" check-env \
        --requirements "$CAMPAIGN_REQUIREMENTS"
fi
unset _campaign_env_build
