# Update reminders: an old-release reminder and `check_for_updates()`

Created: 2026-09-30. Status: **PLANNED** — scope settled with the PI; the
decisions below, and the wording of the reminder, await the PI's rulings.
Executors: Sol implements phases 1 to 3; a fresh Sol reviewer runs the check
of phase 4. `dev/TODO.md` ("Update reminders") links here.

## Goal and settled scope

Users who installed PyVBMC once and never updated should learn that a newer
release may exist, without PyVBMC making a network request they did not ask
for. PyVBMC makes no network request today, and it runs inside users'
scripts, in CI and on cluster nodes without internet access. The PI chose
(2026-09-30) two mechanisms, and no automatic network check:

1. **An old-release reminder, with no network access.** The package ships
   the date of its release. When a new run starts and the installed release
   is older than a threshold, PyVBMC prints one line, at most once per Python
   session, saying that a newer version may exist and how to find out.
2. **`pyvbmc.check_for_updates()`, on request.** It asks PyPI for the latest
   release, compares it with the installed version, and says how to update.
   It is the only code in the package that opens a network connection, and
   only when the user calls it.

Both only print and return; neither touches a run's results, its random
stream, NumPy's global state or any file. The rejected option, a check of
PyPI made automatically in the background (the pattern of pip's own
update check), stays out: an unsolicited network request from a library
called inside other code, which some users and institutions forbid.

Whatever is shipped helps only from the release that contains it, so the
work belongs in 1.5. The release gate regenerates its run pools with "the
latest `dev-next` at the launch, after which only the documentation and the
headline selection change" (`dev/TODO.md`), so this work merges before the
launch of the Slurm plan's Phase 8, unless the PI rules it a presentation
change that may follow (decision D6).

## Decisions for the PI

Each has a recommendation; the phases below assume it.

- **D1. Threshold.** Remind when the release is more than 365 days old.
  Releases of PyVBMC are infrequent, so a shorter threshold would remind
  users of the latest release too; the wording (D4) stays true either way,
  since it says a newer version *may* exist.
- **D2. What silences the reminder.** The same switches as the tips:
  `options={"show_tips": False}` and `display="off"`. The description of
  `show_tips` then names both. An option of its own is the alternative; it
  adds an option for one line per session.
- **D3. S-VBMC.** `SVBMC` shows no reminder: a session that stacks runs has
  usually made them in the same session, where the reminder has shown once.
- **D4. Wording.** Proposed, for the PI to edit as the tips' wording was:
  `PyVBMC 1.5.0 was released more than a year ago. Run
  pyvbmc.check_for_updates() to see whether a newer version is available.`
  with `https://pypi.org/project/pyvbmc/` on its own line, rendered through
  the tips' emitter. The version and the "more than a year" follow the
  installed release and D1.
- **D5. No new dependency.** Both mechanisms count as a release only a
  version of the form `X.Y.Z`, parse it with a regular expression and compare
  versions as integer tuples; they treat any other installed version (a
  `.dev` build, a local `+g<hash>` suffix) as a development install. The
  alternative is `packaging.version`, installed today through Matplotlib;
  using it means declaring `packaging` in `pyproject.toml`, and the
  conda-forge recipe with it.
- **D6. Sequencing.** Merge before the Phase 8 launch (recommended), or rule
  that a change which only prints may follow the launch.
- **D7. Where the release date comes from.** A tracked constant,
  `RELEASE_DATE` in `pyvbmc/_release.py`, set in the release pull request to
  the date of the changelog heading `## [X.Y.Z] - YYYY-MM-DD`, with a test
  that the two agree. It survives every build path: the wheel, the sdist,
  and conda-forge's build from the sdist, which has no git history.
  Rejected: a date written at build time by `setuptools_scm` (a build from the
  sdist has no git history to date, and what the template receives there
  would have to be verified on every path); the modification time of the
  installed files (it dates the installation, and containers and conda
  rewrite it); the upload date on PyPI (a network request).
  Between releases the constant holds the date of the last release, which a
  development install never reads (D5); before the first release that sets
  it, it is `None` and the reminder is off.

## Design

### The old-release reminder

- `pyvbmc/_release.py` holds `RELEASE_DATE: str | None` (ISO date) and
  nothing else, with a comment saying that the release pull request sets it
  to the date of the changelog heading.
- A private module `pyvbmc/vbmc/_release_reminder.py` holds the policy:
  `consider_release_reminder(*, display, enabled, slot_taken, today=None,
  installed=None, release_date=None) -> bool`. The keyword arguments with
  `None` defaults read the real clock (`datetime.date.today()`), the installed
  version (`importlib.metadata.version("pyvbmc")`, as
  `VBMC._create_result_dict` reads it) and `_release.RELEASE_DATE`; tests pass
  their own. It returns whether it printed.
- It prints nothing when display is off, when `enabled` is false, when the
  slot is taken, when the installed version is not a final `X.Y.Z` (D5), when
  the release date is `None` or unreadable, when the release is not older than
  the threshold (D1), or when it has printed already in this Python session.
  A process-local flag, under a lock as `_runtime_tips.py` keeps its state,
  records that it printed; nothing is written to disk and nothing travels
  with a saved run. A start at which it cannot print (display off, slot
  taken) leaves it eligible for the next start.
- It prints through `pyvbmc._user_hints.emit_user_hint`, with the PyPI URL
  as its `urls`, so it reaches standard output as the tips and the
  calibration reminder do and stays out of the log file.
- **The start-of-run slot.** At the first `optimize()` of a new run
  (`vbmc.py`, the block that calls `consider_runtime_tip`, guarded by
  `_runtime_tip_handled`), at most one hint prints, in this order: the
  calibration reminder, then the old-release reminder, then a tip. When the
  old-release reminder prints, the tip is skipped for that run and the tips'
  cadence does not advance, as for a calibration reminder
  (`dev/plans/runtime-tips.md`, "Approved user experience"). Resumed and
  continued runs consider neither, as now.
- `consider_runtime_tip` learns of the reminder through its slot argument;
  generalizing `calibration_reminder_emitted` to a slot flag, or adding a
  second flag, is the implementer's choice, with the tips' tests kept green
  and their behavior unchanged when no reminder prints.

### `pyvbmc.check_for_updates()`

- Signature: `check_for_updates(*, timeout: float = 5.0) -> UpdateCheck`,
  where `UpdateCheck` is a `NamedTuple` of `installed` (`str` or `None`),
  `latest` (`str` or `None` when PyPI could not be read) and
  `update_available` (`bool` or `None`). It prints one message and returns
  the tuple; it raises only for an invalid `timeout`, never for a network or
  parse failure.
- It lives in a private module, `pyvbmc/_update_check.py`, and is exported
  as `pyvbmc.check_for_updates` from `pyvbmc/__init__.py` beside `calibrate`.
  `urllib.request` and `json` are imported inside the function, so that
  `import pyvbmc` imports no networking code.
- The request: a GET of `https://pypi.org/pypi/pyvbmc/json` with
  `urllib.request`, the given timeout, and a `User-Agent` of
  `pyvbmc/<installed version> (check_for_updates)`. Nothing else is sent.
  `urllib` honors the proxy environment variables.
- The latest release is the highest final `X.Y.Z` among the response's
  `releases` whose files are not all yanked. Pre-releases, development
  releases and yanked releases are ignored.
- The message covers:
  - a newer release: the two versions and the update command;
  - the latest release installed: say so;
  - a development install (D5): the installed version and the latest
    release, and that a development install is updated from its checkout;
  - PyPI unreachable or its reply unreadable: say so in one line, with the
    reason, and give `https://pypi.org/project/pyvbmc/`.
- The update command follows the installer, read from the `INSTALLER` file of
  the installed distribution (`importlib.metadata.distribution("pyvbmc")`):
  `pip` gives `python -m pip install --upgrade pyvbmc`; `conda` gives
  `conda update --channel=conda-forge pyvbmc`, with a note that the
  conda-forge package can follow PyPI by a few days; anything else gives
  both.

## Phases

### Phase 0 — the PI's rulings (PI)

- [ ] D1 to D7 ruled, and the wording of D4 edited or approved. Record the
  rulings in "Decisions for the PI", marked with the date.

### Phase 1 — the release date and the reminder (Sol)

Work on a branch `feat-update-reminders` cut from `dev-next`.

1. Add `pyvbmc/_release.py` with `RELEASE_DATE = None`.
2. Add `pyvbmc/vbmc/_release_reminder.py` as designed above, and wire it
   into `VBMC.optimize()` at the start-of-run slot. Add a private
   `_reset_release_reminder_state()` for tests, as `_runtime_tips.py` has.
3. If D2 stands, edit the description of `show_tips` in
   `pyvbmc/vbmc/option_configs/basic_vbmc_options.ini` to name the reminder.
4. Add `pyvbmc/testing/conftest.py` with an autouse fixture that resets the
   reminder's state and makes it inert (the session flag set, or the release
   date `None`), so that no shipped test's output depends on the calendar.
   Without it, the test suite of a release would print the reminder, and
   fail where a test captures startup output, a year after the release.
5. Tests in `pyvbmc/testing/vbmc/test_release_reminder.py`, with injected
   dates and versions:
   - [ ] under and over the threshold, and on its boundary;
   - [ ] printed once per session, and eligible again after the state reset;
   - [ ] nothing for a development version, a local version, `None` or an
     unreadable release date;
   - [ ] nothing with `display="off"` or (D2) `show_tips=False`, and such a
     start does not use up the session's reminder;
   - [ ] a calibration reminder takes the slot and the old-release reminder
     waits for the next start; when the old-release reminder prints, no tip
     prints and the tips' cadence does not advance;
   - [ ] resumed and continued runs print neither;
   - [ ] the random streams (the run's generator, NumPy's global state, the
     tips' private `random.Random`) are unchanged, as the tips' tests check;
   - [ ] a short seeded run gives the same results with the reminder printed
     and without it, sharing an existing fixture rather than adding an
     `optimize()` run (`AGENTS.md`, "Tests and their traps");
   - [ ] the release-date test: when `CHANGELOG.md` has a released section
     `## [X.Y.Z] - YYYY-MM-DD`, the first such heading's date equals
     `RELEASE_DATE`; with none, `RELEASE_DATE` is `None`. The test reads
     `CHANGELOG.md` from the repository root and skips where the file is
     absent.

### Phase 2 — `check_for_updates()` (Sol)

1. Add `pyvbmc/_update_check.py` and the export, as designed above.
2. Tests in `pyvbmc/testing/test_update_check.py`, with
   `urllib.request.urlopen` patched (`pytest-mock`), so that no test opens a
   connection:
   - [ ] newer release available; latest installed; installed newer than
     PyPI's latest; development install;
   - [ ] pre-releases, development releases and fully yanked releases are
     ignored, a partly yanked release is not;
   - [ ] `URLError`, `HTTPError`, a timeout, malformed JSON and a JSON without
     `releases` each give the failure message and a tuple with
     `latest=None`, and raise nothing;
   - [ ] the update command for `INSTALLER` of `pip`, `conda`, another value
     and a missing file;
   - [ ] the request's URL, timeout and `User-Agent`; nothing else sent;
   - [ ] an invalid `timeout` raises `ValueError`;
   - [ ] `import pyvbmc` leaves `pyvbmc._update_check` unimported, or imports
     it without importing `urllib.request` through it.

### Phase 3 — documentation and records (Sol)

1. `docsrc/source/api/functions/check_for_updates.rst`, in the style of
   `calibrate.rst`, and its entry in the toctree of
   `docsrc/source/api/functions/functions.rst` (`AGENTS.md`: nothing generates
   the API pages).
2. `docsrc/source/faq.md`: a question under "Installing PyVBMC", "How do I
   know whether a newer version of PyVBMC exists?", with its anchor and its
   line in the table of contents: the reminder, `check_for_updates()`, and
   the two update commands.
3. The "Startup tips" section of `docsrc/source/api/classes/vbmc.rst` and the
   tips paragraph of `docsrc/source/quickstart.rst`: the reminder, its place
   in the start-of-run slot, and what silences it.
4. `CHANGELOG.md`, under Added (`AGENTS.md`, "Changelog"), one entry: a run
   of a release more than a year old says once per session that a newer
   version may exist, and `pyvbmc.check_for_updates()` asks PyPI. No
   Upgrading line: nothing a script relies on changes.
5. `AGENTS.md`, "Conventions": the package opens a network connection only in
   `check_for_updates()`, which the user calls; the old-release reminder reads
   only `pyvbmc/_release.py`. This is a rule nothing enforces, and an agent
   adding a convenience could break it.
6. The pre-release checklist of `dev/plans/modernization-roadmap.md`: a step
   to set `RELEASE_DATE` in the release pull request to the date of the
   changelog heading, which the release-date test then checks.
7. `skills/pyvbmc/SKILL.md`: whether to point agents to
   `check_for_updates()` when a user reports a problem is the PI's call; the
   skill links documentation pages by name, so a new page is a candidate.

### Phase 4 — verification and delivery

- [ ] Sol: the focused tests (the two new modules, `test_runtime_tips.py`, the
  calibration and options tests, `test_vbmc_seed.py`), then the whole suite
  once (`python -m pytest --reruns=5 -x -vv`), one heavy process at a time.
- [ ] Sol: `python dev/scripts/make_oracle_fixtures.py --check --exact` on the
  machine that generated the fixtures (`dev/scripts/runs/LOCAL.md`), BLAS
  single-threaded: the change must move nothing.
- [ ] Sol: the Sphinx build with the example notebooks copied in, as
  `make github` does; the new API page, the FAQ entry and the tips sections
  render, and the build adds no warning.
- [ ] Sol: one call of `pyvbmc.check_for_updates()` against the real PyPI from
  the development environment; before 1.5 is on PyPI it reports a development
  install and 1.0.4 as the latest release.
- [ ] A fresh Sol reviewer, read-only: the diff against this plan, the
  decisions as ruled, the invariants (no network request outside
  `check_for_updates()`, no effect on results or random streams, no file
  written), the tests' independence from the calendar and the network, and
  the documentation. Findings resolved, affected checks rerun.
- [ ] The CI test matrix on the feature branch, as the tips work ran it
  (`dev/plans/runtime-tips.md`, "Delivery checklist"); merge into `dev-next`
  with the PI's approval, before the Phase 8 launch (D6); remove the branch.

## Acceptance

- A run of a final release older than the threshold prints the reminder once
  per session, in the start-of-run slot, and nothing else changes: results,
  random streams, files, the log file and the tips' cadence when no reminder
  prints.
- `check_for_updates()` reports correctly in every case above, never raises on
  a network or parse failure, and is the only network access in the package.
- No test depends on the date or on the network.
- `RELEASE_DATE` and the changelog's latest release heading cannot disagree
  on a commit that passes the tests.
