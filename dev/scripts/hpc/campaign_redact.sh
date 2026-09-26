#!/bin/bash
# Write a finished campaign's tracked copies, redacted for the repository
# (dev/plans/slurm-benchmark-support.md, "Records and hand-back"), or check
# other files as the copies are checked.
#
#   campaign_redact.sh CAMPAIGN_DIR OUT_DIR [--path NAME=PATH ...]
#   campaign_redact.sh CAMPAIGN_DIR --check FILE [FILE ...] \
#       [--path NAME=PATH ...]
#
# On the login node, in the account that ran the campaign, once
# campaign_finish.sh has passed. It runs `campaign_contract.py redact` in
# the campaign's environment (campaign_env.sh, with CAMPAIGN_ENV set), from
# this checkout, the harness checkout of the campaign. OUT_DIR is a new or
# empty directory in another checkout, the one the hand-back is committed
# from: it may not lie inside CAMPAIGN_DIR or inside any source tree of the
# campaign, which stay clean while later campaigns use them. It receives
# manifest.json, verification.json and the files the harness declares in
# the manifest's tracked_copies, at their paths in the campaign directory,
# and redaction.json, which records the SHA-256 of each copy and of the
# campaign's file it was made from, and the cases the verification report
# does not place as verified (failed, interrupted, missing). A report that
# places a case in flight is refused. In the copies:
#
# - every host that ran a Slurm job of the campaign (in a completion
#   record, a task's or a step's log, the accounting) is named by the node
#   family, the value of NODE_FEATURE, and every other host (the login
#   node) by "login";
# - a path under a path setting of the site block starts with the
#   setting's name ($CAMPAIGN_ENV, $PYVBMC_GPYREG_SOURCE, ...), one under a
#   directory given with --path NAME=PATH (a scratch area that holds the
#   campaign, say) with $NAME, and one under the home directory with ~; a
#   path under a source tree of the campaign's identity, or under the
#   directory that holds the campaign directory, that none of those names
#   starts with $<TREE>_TREE ($HARNESS_TREE, ...) or $CAMPAIGN_PARENT;
# - a field that holds a partition (a record's SLURM_JOB_PARTITION) is
#   $PARTITION;
# - the manifest keeps no site block, and its pip freeze lines that name a
#   path keep their package's name alone;
# - the task logs, the Slurm accounting, the claims and the error files
#   are not copied.
#
# The username and the home directory are this process's (USER, LOGNAME,
# HOME and the password database), never read from a file, which is why it
# runs in the account that ran the campaign. It refuses, and writes
# nothing, when a copy still holds, as a plain substring, a value of the
# site block that is a path or a command, a named directory, the username,
# the home directory or a hostname of the campaign (in any letter case),
# or an absolute path outside the system's directories (/usr, /etc, /tmp
# and the like) that no name covers, naming the file and the string.
#
# With --check, it searches the FILEs, which it did not write (the README
# beside the tracked copies, say), for the same strings, and refuses when
# any holds one.
set -euo pipefail

usage() {
    sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//' | head -n -1 >&2
    exit 64
}

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
export REPO

if [ $# -lt 2 ] || [ ! -f "$1/manifest.json" ]; then
    usage
fi
CAMPAIGN=$1
shift
if [ "$1" = --check ]; then
    [ $# -ge 2 ] || usage
    TARGET=()
else
    TARGET=(--out "$1")
    shift
fi
if [ -z "${CAMPAIGN_ENV:-}" ]; then
    echo "refusing: CAMPAIGN_ENV, the campaign's environment, is not set" >&2
    exit 1
fi
unset CAMPAIGN_DIR CAMPAIGN_STEP INDEX_OFFSET

# shellcheck source=campaign_env.sh
source "$HERE/campaign_env.sh"
exec python -u "$REPO/dev/scripts/campaign_contract.py" redact \
    --campaign "$CAMPAIGN" ${TARGET[@]+"${TARGET[@]}"} "$@"
