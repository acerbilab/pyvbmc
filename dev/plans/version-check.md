# Update reminders: an old-release reminder and `check_for_updates()`

Created: 2026-09-30. Status: **IN PROGRESS** — every decision ruled by the
PI on 2026-09-30; implementation on the branch `feat-update-reminders`.
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
   is older than a threshold, PyVBMC prints one line saying that a newer
   version may exist and how to find out: at most three times for each
   installed version, at least 90 days apart, and at most once per Python
   session (D8).
2. **`pyvbmc.check_for_updates()`, on request.** It asks PyPI for the latest
   release, compares it with the installed version, and says how to update.
   It is the only code in the package that opens a network connection, and
   only when the user calls it.

Neither touches a run's results, its random stream or NumPy's global state.
The reminder writes one small file in PyVBMC's user cache directory, to count
its showings; `check_for_updates()` writes nothing. The rejected option, a
check of PyPI made automatically in the background (the pattern of pip's own
update check), stays out: an unsolicited network request from a library
called inside other code, which some users and institutions forbid.

Whatever is shipped helps only from the release that contains it, so the
work belongs in 1.5. The release gate regenerates its run pools with "the
latest `dev-next` at the launch, after which only the documentation and the
headline selection change" (`dev/TODO.md`), so this work merges before the
launch of the Slurm plan's Phase 8, unless the PI rules it a presentation
change that may follow (decision D6).

## Prior art

A survey of update notices on 2026-09-30 (primary sources: each tool's source
code or documentation) found:

- Offline reminders from the age of a release are rare: yt-dlp warns when
  its version is more than 90 days old (the command line only; its library
  interface stays quiet unless asked), browserslist and
  baseline-browser-mapping when their bundled data is 6 and 2 months old,
  Chrome when its build is 8 weeks old and the clock is not behind the build.
  None stops by itself: each shows on every run until the user updates or
  silences it.
- Network checks (pip, npm's `update-notifier`, `gh`, `huggingface_hub`,
  conda) keep a timestamp in the user's cache or configuration directory and
  check every 1 to 7 days. pip writes its state to a temporary file, renames
  it into place and swallows every error. The command-line tools of npm and
  GitHub stay quiet in CI (`CI` set) and without a terminal. Each has an
  opt-out environment variable of its own (`PIP_DISABLE_PIP_VERSION_CHECK`,
  `NO_UPDATE_NOTIFIER`, `GH_NO_UPDATE_NOTIFIER`,
  `HF_HUB_DISABLE_UPDATE_CHECK`); no convention exists for Python libraries.
- No tool caps its notice per version on its own. The nearest are Sparkle's
  "Skip This Version", a user's choice that a manual check ignores, and
  Homebrew's notice on analytics, shown once and counted as shown only when
  it reached a terminal.
- The complaints are about notices the user cannot act on (installs managed
  by a system package manager or an institution, a notice from a dependency),
  noise in CI, output that cannot be filtered, and unsolicited network
  requests (streamlit removed its check); not about repetition as such.

## Decisions for the PI

The PI ruled every decision on 2026-09-30: D1, D4 and D8 as stated, the
others as recommended.

- **D1. Threshold: 12 months** (365 days; PI, 2026-09-30). The reminder cannot
  know whether a newer release exists, only how old the installed one is, so
  a gap of more than 12 months between releases gives users of the latest
  release reminders they cannot act on. The threshold therefore stands on a
  release at least once a year, a small maintenance release included (PI,
  2026-09-30); the cap (D8) bounds the cost of a longer gap to three lines
  per version. The wording (D4) says that a newer version *may* exist.
- **D2. What silences the reminder** (PI, 2026-09-30). The same switches as
  the tips: `options={"show_tips": False}` and `display="off"`. The
  description of `show_tips` then names both. An option of its own was the
  alternative; it adds an option for one line per session.
- **D3. S-VBMC** (PI, 2026-09-30). `SVBMC` shows no reminder: a session that
  stacks runs has usually made them in the same session, where the reminder
  has shown once.
- **D4. Wording** (approved by the PI, 2026-09-30). A template, filled when
  the run starts from the installed release:
  `Note: PyVBMC {version} was released {age}. Run
  pyvbmc.check_for_updates() to see whether a newer version is available.`
  On the third and last showing for a version (D8) the sentence
  `This is the last reminder for PyVBMC {version}.` follows. The line is
  followed by `https://pypi.org/project/pyvbmc/` on its own line, rendered
  through the tips' emitter. `{version}` is the installed version
  (`importlib.metadata.version("pyvbmc")`), and `{age}` is computed from
  `RELEASE_DATE` and the date of the run: "more than a year ago" up to two
  years, then "more than N years ago", N the whole number of years. With
  1.5.0 installed two and a half years after its release, the line reads
  `Note: PyVBMC 1.5.0 was released more than 2 years ago. Run ...`. The
  templates are constants at the top of `_release_reminder.py`, so that a
  wording edit is a text edit. The messages of `check_for_updates()`, below,
  were approved with it.
- **D5. No new dependency** (PI, 2026-09-30). Both mechanisms count as a
  release only a version of the form `X.Y.Z`, parse it with a regular
  expression and compare versions as integer tuples; they treat any other
  installed version (a `.dev` build, a local `+g<hash>` suffix) as a
  development install. The
  alternative was `packaging.version`, installed today through Matplotlib;
  using it means declaring `packaging` in `pyproject.toml`, and the
  conda-forge recipe with it.
- **D6. Sequencing** (PI, 2026-09-30). The work merges before the launch of
  Phase 8; the alternative was to rule that a change which only prints may
  follow the launch.
- **D7. Where the release date comes from** (PI, 2026-09-30). A tracked
  constant, `RELEASE_DATE` in `pyvbmc/_release.py`, set in the release pull
  request to the date of the changelog heading `## [X.Y.Z] - YYYY-MM-DD`,
  with a test that the two agree. It survives every build path: the wheel,
  the sdist, and conda-forge's build from the sdist, which has no git
  history.
  Rejected: a date written at build time by `setuptools_scm` (a build from the
  sdist has no git history to date, and what the template receives there
  would have to be verified on every path); the modification time of the
  installed files (it dates the installation, and containers and conda
  rewrite it); the upload date on PyPI (a network request).
  Between releases the constant holds the date of the last release, which a
  development install never reads (D5); before the first release that sets
  it, it is `None` and the reminder is off.
- **D8. Cap: at most three times per installed version, at least 90 days
  apart** (PI, 2026-09-30). The offline reminders found elsewhere never stop
  by themselves ("Prior art"); PyVBMC's users pin versions for
  reproducibility, so the reminder stops for a version after its third
  showing. With the 12-month threshold it shows at about 12, 15 and 18
  months after the release. The count and the spacing are judgements; no
  prior art sets either.

## Design

### The old-release reminder

- `pyvbmc/_release.py` holds `RELEASE_DATE: str | None` (ISO date) and
  nothing else, with a comment saying that the release pull request sets it
  to the date of the changelog heading.
- A private module `pyvbmc/vbmc/_release_reminder.py` holds the policy:
  `consider_release_reminder(*, display, enabled, slot_taken, today=None,
  installed=None, release_date=None, interactive=None, environ=None,
  state_path=None) -> bool`. The keyword arguments with `None` defaults read
  the real clock (`datetime.date.today()`), the installed version
  (`importlib.metadata.version("pyvbmc")`, as `VBMC._create_result_dict`
  reads it), `_release.RELEASE_DATE`, whether the session is interactive,
  `os.environ` and the path of the state file; tests pass their own. It
  returns whether it printed.
- It prints nothing when display is off, when `enabled` is false, when the
  slot is taken, when the installed version is not a final `X.Y.Z` (D5), when
  the release date is `None` or unreadable, when the date of the run is
  earlier than the release date (a wrong clock), when the release is not
  older than the threshold (D1), when the session is not interactive, when
  `CI`, `PYVBMC_NO_UPDATE_REMINDER` or `NO_UPDATE_NOTIFIER` is set to a
  value other than empty, `0` or `false` (read as CI detection reads `CI`),
  when the cap forbids it (D8, below), or when it has printed already in
  this Python session. A process-local flag, under a
  lock as `_runtime_tips.py` keeps its state, records that it printed in
  this session; nothing travels with a saved run. A start at which it does
  not print writes nothing and leaves it eligible for the next start.
- The session is interactive when standard output is a terminal
  (`sys.stdout.isatty()`) or the code runs in an IPython kernel, as in a
  Jupyter notebook; the log of a batch job is neither. The test suite stays
  quiet through the fixture of phase 1, not through a check for pytest in the
  package.
- **The cap (D8)** is kept in `update_reminder.json` in PyVBMC's user cache
  directory, the one the calibration cache uses
  (`platformdirs.user_cache_dir("pyvbmc", appauthor=False, opinion=False)`:
  `~/.cache/pyvbmc` on Linux, `~/Library/Caches/pyvbmc` on macOS,
  `%LOCALAPPDATA%\pyvbmc` on Windows). The file maps each installed version
  to the dates of its showings, as in `{"1.5.0": ["2027-11-02",
  "2028-02-14"]}`, and holds nothing else. Before printing, the reminder
  reads it: when the installed version has three dates, or its last date is
  less than 90 days before the date of the run, nothing prints. After
  printing, it appends the date and writes the file as pip writes its own
  state: to a temporary file in the same directory, renamed over the old
  one, every error swallowed. The file first appears when the reminder first
  shows, so nothing is written in a release's first year. A file whose
  content is malformed (not a mapping from `X.Y.Z` versions to lists of ISO
  dates, or larger than 64 KiB) counts as empty and is replaced at the next
  showing, so that corruption cannot turn the cap into a reminder in every
  session. A file that cannot be read or written makes the reminder fall
  back to once per session. The file honors `PYVBMC_CACHE_DIR`, as the
  calibration cache does. A newly installed version has a
  list of its own, and deleting the file resets every count. Parallel starts
  may each print before any of them writes; later starts see their dates.
- It prints through `pyvbmc._user_hints.emit_user_hint`, with the PyPI URL
  as its `urls`, so it reaches standard output as the tips and the
  calibration reminder do and stays out of the log file.
- **The start-of-run slot.** At the first `optimize()` of a new run
  (`vbmc.py`, the block that calls `consider_runtime_tip`, guarded by
  `_runtime_tip_handled`), at most one hint prints, in this order: the
  calibration reminder, then the old-release reminder, then a tip. When the
  old-release reminder prints, the tip is skipped for that run and the tips'
  cadence does not advance, as for a calibration reminder
  (`dev/plans/runtime-tips.md`, "Approved user experience"): the tip that
  start would have shown comes at the next start instead of being lost.
  Resumed and continued runs consider neither, as now.
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
  `pyvbmc/<installed version> (check_for_updates)` and an `Accept` of
  `application/json`; nothing else about the installation or the user is
  sent. `urllib` honors the proxy environment variables.
  A release counts only when at least one of its files is not yanked, so a
  release with no files does not count.
- The latest release is the highest final `X.Y.Z` among the response's
  `releases` whose files are not all yanked. Pre-releases, development
  releases and yanked releases are ignored.
- The message, one of (wording approved by the PI with D4):
  - a newer release: `PyVBMC {latest} is available; you have {installed}.
    Update with: {command}`, where the command is the installer's (below);
  - the latest release installed: `PyVBMC {installed} is the latest
    release.`;
  - a development install (D5): `PyVBMC {installed} is a development
    version; the latest release is {latest}.`;
  - PyPI unreachable or its reply unreadable: `Could not reach PyPI
    ({reason}); see https://pypi.org/project/pyvbmc/.`, the reason a few
    words (`timed out`, `HTTP 503`, `unreadable reply`, the reason of a
    `URLError`).
  Written in the same style during the implementation, and awaiting the PI's
  reading:
  - a final installed version newer than PyPI's latest, which happens only
    before a release reaches PyPI: `PyVBMC {installed} is newer than the
    latest release on PyPI, {latest}.`;
  - an installed version that cannot be read: `PyVBMC's installed version
    is unknown; the latest release is {latest}.`;
  - a readable reply with no release that counts: the failure message with
    the reason `no release found`;
  - an installer other than pip or conda: `... Update with: python -m pip
    install --upgrade pyvbmc, or with conda: conda update
    --channel=conda-forge pyvbmc (the conda-forge package can follow PyPI
    by a few days)`.
- The update command follows the installer, read from the `INSTALLER` file of
  the installed distribution (`importlib.metadata.distribution("pyvbmc")`):
  `pip` gives `python -m pip install --upgrade pyvbmc`; `conda` gives
  `conda update --channel=conda-forge pyvbmc`, with a note that the
  conda-forge package can follow PyPI by a few days; anything else gives
  both.

## Live checklist

- [x] Phase 0: every decision ruled (2026-09-30).
- [x] Phase 1: the release date and the reminder (tests pass; a malformed
  state file starts afresh, and `0`/`false` leave an opt-out unset).
- [x] Phase 2: `check_for_updates()` (61 tests pass; the messages written
  during the implementation await the PI, in the design above).
- [~] Phase 3: documentation and records.
- [ ] Phase 4: verification and delivery.

## Phases

### Phase 0 — the PI's rulings (PI)

- [x] D1 (12 months) and D8 (three times per version, 90 days apart) ruled
  on 2026-09-30.
- [x] D4, the wording of the reminder with its "Note:" label and its
  last-reminder sentence, and the messages of `check_for_updates()`,
  approved on 2026-09-30.
- [x] D2, D3, D5, D6 and D7 ruled as recommended, and the pointer in the
  user skill (phase 3, step 7) approved, on 2026-09-30.

### Phase 1 — the release date and the reminder (Sol)

Work on a branch `feat-update-reminders` cut from `dev-next`.

1. Add `pyvbmc/_release.py` with `RELEASE_DATE = None`.
2. Add `pyvbmc/vbmc/_release_reminder.py` as designed above, and wire it
   into `VBMC.optimize()` at the start-of-run slot. Add a private
   `_reset_release_reminder_state()` for tests, as `_runtime_tips.py` has.
3. If D2 stands, edit the description of `show_tips` in
   `pyvbmc/vbmc/option_configs/basic_vbmc_options.ini` to name the reminder.
4. Add `pyvbmc/testing/conftest.py` with an autouse fixture that resets the
   reminder's state, points its state file into pytest's temporary
   directory, and makes it inert (the session flag set, or the release date
   `None`), so that no shipped test's output depends on the calendar and no
   test writes the user's cache directory. Without it, the test suite of a
   release would print the reminder, and fail where a test captures startup
   output, a year after the release.
5. Tests in `pyvbmc/testing/vbmc/test_release_reminder.py`, with injected
   dates and versions:
   - [ ] under and over the threshold, and on its boundary;
   - [ ] the line starts with `Note:` and names the installed version and
     the age computed from the injected dates: "more than a year ago"
     between one and two years, "more than N years ago" beyond;
   - [ ] the third showing for a version, and only it, ends with the
     last-reminder sentence;
   - [ ] printed once per session, and eligible again after the state reset;
   - [ ] the cap: printed at the first eligible start, not again within 90
     days, again after 90, never after the third time; a new version starts
     a list of its own; the file holds only versions and dates;
   - [ ] a state file with malformed content: counted as empty and
     replaced at the showing; one that cannot be read or written: once per
     session; nothing raised in either case; a write goes through a
     temporary file and a rename;
   - [ ] nothing printed and nothing written for a development version, a
     local version, a release date that is `None` or unreadable, a date of
     the run earlier than the release date, a session that is not
     interactive, and each of `CI`, `PYVBMC_NO_UPDATE_REMINDER` and
     `NO_UPDATE_NOTIFIER` set;
   - [ ] nothing with `display="off"` or (D2) `show_tips=False`, and such a
     start does not use up the session's reminder or write the file;
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
     PyPI's latest; development install; each prints its message of the
     design, word for word;
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
   line in the table of contents: the reminder (when it shows, how often,
   where it records its showings, what silences it), `check_for_updates()`,
   and the two update commands.
3. The "Startup tips" section of `docsrc/source/api/classes/vbmc.rst` and the
   tips paragraph of `docsrc/source/quickstart.rst`: the reminder, its place
   in the start-of-run slot, and what silences it.
4. `CHANGELOG.md`, under Added (`AGENTS.md`, "Changelog"), one entry: in an
   interactive session, a run of a release more than a year old says that a
   newer version may exist, at most three times per version, and
   `pyvbmc.check_for_updates()` asks PyPI. No Upgrading line: nothing a
   script relies on changes.
5. `AGENTS.md`, "Conventions": the package opens a network connection only in
   `check_for_updates()`, which the user calls; the old-release reminder reads
   only `pyvbmc/_release.py` and its state file, and writes only the latter.
   This is a rule nothing enforces, and an agent adding a convenience could
   break it.
6. The pre-release checklist of `dev/plans/modernization-roadmap.md`: a step
   to set `RELEASE_DATE` in the release pull request to the date of the
   changelog heading, which the release-date test then checks.
7. `skills/pyvbmc/SKILL.md`: one line telling an agent to run
   `pyvbmc.check_for_updates()` when a user reports a problem, since the fix
   may be released already, linking the new API page (PI, 2026-09-30).

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
  written but the reminder's state file), the tests' independence from the
  calendar, the network and the user's cache directory, and the
  documentation. Findings resolved, affected checks rerun.
- [ ] The CI test matrix on the feature branch, as the tips work ran it
  (`dev/plans/runtime-tips.md`, "Delivery checklist"); merge into `dev-next`
  with the PI's approval, before the Phase 8 launch (D6); remove the branch.

## Acceptance

- In an interactive session, a run of a final release older than the
  threshold prints the reminder in the start-of-run slot, at most once per
  session and three times per installed version, at least 90 days apart.
  Nothing else changes: results, random streams, the log file, files other
  than the reminder's state file, and the tips' cadence when no reminder
  prints.
- `check_for_updates()` reports correctly in every case above, never raises on
  a network or parse failure, and is the only network access in the package.
- No test depends on the date or on the network, or writes the user's cache
  directory.
- `RELEASE_DATE` and the changelog's latest release heading cannot disagree
  on a commit that passes the tests.
