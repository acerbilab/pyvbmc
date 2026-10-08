# The date of the latest release in this source tree, as an ISO date
# ("YYYY-MM-DD"), or None before the first release that sets it. The release
# pull request sets it to the date of the release's heading in CHANGELOG.md,
# "## [X.Y.Z] - YYYY-MM-DD", and a test checks that the two agree. The
# old-release reminder (pyvbmc/vbmc/_release_reminder.py) reads it; with None
# it stays silent.
RELEASE_DATE: str | None = "2026-10-13"
