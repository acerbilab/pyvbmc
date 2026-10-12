#!/bin/bash
# Build the public asset of a finished campaign, what the PyVBMC release
# publishes of it (dev/plans/slurm-benchmark-support.md, "Records and
# hand-back"), or check one.
#
#   campaign_public.sh CAMPAIGN_DIR COPIES_DIR OUT_DIR \
#       [--path NAME=PATH ...] [--allow STRING ...] [--part-size BYTES]
#   campaign_public.sh --check OUT_DIR
#
# On the login node, in the account that ran the campaign, after
# campaign_redact.sh has written COPIES_DIR, the campaign's tracked copies.
# It runs `campaign_public.py build` in the campaign's environment
# (campaign_env.sh, with CAMPAIGN_ENV set), from this checkout, the harness
# checkout of the campaign. OUT_DIR receives
# <campaign>.public.tar.gz.000, .001, ... and their SHA-256 in
# <campaign>.public.tar.gz.sha256: the tracked copies, the .npz artifacts
# of every verified case, which must hold numbers alone, and its
# completion record and other .json artifacts redacted as the copies are,
# but no pickle and no log. The redaction's rules are campaign_redact.sh's,
# with this process's username and home, and it takes the same --path and
# --allow; a string the copies may not hold refuses the asset, which is
# then not written. Give the same --path and --allow as to
# campaign_redact.sh. The asset's public.json records the code that built
# it and the exemptions it applied.
#
# With --check, it re-reads every asset in OUT_DIR, one per campaign built
# into it: every part against its SHA-256 and every file against the
# asset's public.json; then it fails an asset that other code than this
# checkout's built, and applies the rules that need no campaign (numbers
# alone in the .npz files, the copies and .json files alone beside them,
# no absolute path that no name covers).
set -euo pipefail

usage() {
    sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//' | head -n -1 >&2
    exit 64
}

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
export REPO

if [ $# -eq 2 ] && [ "$1" = --check ]; then
    COMMAND=(check "$2")
    shift 2
elif [ $# -ge 3 ] && [ -f "$1/manifest.json" ]; then
    COMMAND=(build --campaign "$1" --copies "$2" --out "$3")
    shift 3
else
    usage
fi
if [ -z "${CAMPAIGN_ENV:-}" ]; then
    echo "refusing: CAMPAIGN_ENV, the campaign's environment, is not set" >&2
    exit 1
fi
unset CAMPAIGN_DIR CAMPAIGN_STEP INDEX_OFFSET

# shellcheck source=campaign_env.sh
source "$HERE/campaign_env.sh"
exec python -u "$REPO/dev/scripts/campaign_public.py" "${COMMAND[@]}" "$@"
