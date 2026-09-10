# Optional runtime tips

Created: 2026-09-10. Status: **COMPLETE — implemented and verified**.
Astra owns design/integration; Sol implements and independently reviews.
This plan owns the tip policy, wording, startup integration and acceptance
checks for maintainers. Roadmap pickup 13 links here; no separate worklog.

## Goal and settled scope

Occasionally show one short, useful tip when a new VBMC run starts, helping
users discover guidance without crowding the optimization output.
The PI's shared-slot rule is fixed: **if the calibration reminder is shown,
skip the tip**. This includes a reminder emitted by an early call on that
run's `vbmc.vp`, before `optimize()`.
The PI also confirmed that history stays within the Python session, with no
additional persistent storage, and requested shared hint-output code where useful.
Some tips may introduce complementary methods, framed around a useful task.
The PI suggested at most roughly 25-30% of the catalog for these recommendations.
Use a factual **low-frequency** category, also available for other tips we want
to appear less often. Shuffle so short sessions can encounter them too; avoid
consecutive low-frequency tips when possible, without blocking progress.
Developer editability and URL support are essential requirements: changing,
replacing or adding either category of tip must be a catalog edit, not a change
to scheduling or rendering code. Any tip may include documentation URLs.

Preserve quiet mode, all inference settings and both instance/global RNG
states. No target evaluation, timing campaign or dependency is introduced.
Implementation authorized on 2026-09-10 after the PI edited the messages below.
The authorization includes the recorded defaults, cadence and first-start policy.
Implementation branch: `feat/runtime-tips`, merged to `dev-next` as `fc64381`
and removed after the merge.

## Live implementation checklist

- [x] Sol: implement the catalog, shared emitter, selector, lifecycle integration,
  option validation and focused tests (phase 1 below).
- [x] Astra: update Quickstart, VBMC API guidance and the 1.5 overview; verify
  that the catalog preserves the PI's edited messages and useful links.
- [x] Astra: run focused tests, the required package suite and the docs build.
- [x] Independent Sol review via doublecheck; resolve findings and rerun affected checks.
- [x] Reconcile acceptance checks and update this plan, TODO and roadmap with the outcome.

## Delivery checklist

Commit/push, CI-gated merge to `dev-next`, and a post-merge full matrix were
authorized by the PI on 2026-09-10.

- [x] Commit and push `feat/runtime-tips` (`1c3d4f2`; all pre-commit hooks passed).
- [x] Run the feature branch test matrix and package build; require green CI before merging.
- [x] Merge to `dev-next`, push, and remove the merged feature branch.
- [x] Run and monitor the full matrix on the merged `dev-next` commit.

Feature CI: [test matrix](https://github.com/acerbilab/pyvbmc/actions/runs/34474881485)
and [package build](https://github.com/acerbilab/pyvbmc/actions/runs/34474884265)
both passed for `1c3d4f2`; all nine OS/Python combinations were green before merge.

Post-merge [full matrix](https://github.com/acerbilab/pyvbmc/actions/runs/34476040012)
passed all nine OS/Python combinations on `fc64381` on `dev-next`. The automatic
[smoke check](https://github.com/acerbilab/pyvbmc/actions/runs/34476039794) also
passed. The merge tree exactly matches the feature tree tested before merge.
Delivery is complete; subsequent tracker updates change only developer Markdown.

Catalog verification: all eight message strings and URL tuples exactly match
the PI-edited wording below (checked by extracting the plan entries and comparing
them with `_tip_catalog.TIPS`). The linked official guidance was checked during
the wording review in this session.

Docs build: `.venv/Scripts/python.exe -m sphinx -M html source _build`
from `docsrc`, with the example notebooks copied in as for `make github`,
succeeded. The sole warning is the existing notebook-2 Plotly MIME warning.
The rendered Quickstart, VBMC API and basic-options reference include the new
guidance. Local HTML output is in ignored `docs/`; copied source examples were removed.

Focused verification: runtime tips, calibration, options, VBMC save/load and
existing seeded-run modules passed: **151 passed in 11.35 s**, with no reruns.
The first attempt hit sandbox restrictions in pytest's default temp/cache
directories; using task-specific paths inside `.venv` resolved that environment
issue. BLAS was single-threaded. The seeded comparison printed a real tip in
one existing run and matched the tips-disabled run exactly.

Full package suite: **1170 passed, 35 skipped, 3 warnings in 367.46 s**, with
no failures or reruns. Command: `.venv/Scripts/python.exe -m pytest --reruns=5
-x -vv --basetemp=.venv/runtime-tips-pytest-full -o
cache_dir=.venv/runtime-tips-pytest-cache`, with BLAS single-threaded. The three
warnings are the existing VP gradient division warnings from oracle cases.
Optional torch/ArviZ dependencies are absent in this environment. New modules
and focused tests also pass Black formatting, and changed Python files pass
isort and compilation checks; `git diff --check` is clean.

Independent fresh-context Sol review completed with no findings. The feature
preserves the approved messages and URLs, calibration priority, quiet/resume
behavior, and inference randomness. All in-scope implementation and verification
work is complete; changes are merged to `dev-next`. Delivery CI is tracked above.

## Approved user experience

- Basic option `show_tips=True`; `options={"show_tips": False}` disables tips.
  This option does not disable the calibration reminder. `display="off"`
  suppresses both, as it already suppresses ordinary startup output.
- Consider tips only on the first `optimize()` entry for a new run, after
  calibration resolution and the performance-settings summary, before the
  iteration headings. Resumed/continued runs do not show another tip.
- Frequency: first eligible new run, then every third eligible run
  (opportunities 1, 4, 7, ...). Eligibility requires tips enabled, nonquiet
  startup, an available unseen tip, and no calibration reminder for this run.
  Suppressed runs do not advance the counter or consume a tip.
- Shuffle the catalog once per Python session; show each tip at most once,
  then stop. A fresh interpreter resets the history; notebook cells share it.
  The shuffle uses a private RNG independent of inference and module-level RNGs.
  There is no filesystem I/O and no need to seed inference to control tip order.
- The initial catalog has six ordinary tips and two low-frequency tips (25%).
  A low-frequency tip can appear first, including in a one-run session. After
  one is shown, prefer an ordinary tip if one remains; otherwise show the next
  remaining tip. Do not enforce a percentage on short prefixes, require ordinary
  tips first, or stall because only low-frequency tips remain. Calibration
  reminders do not consume an entry or reset the last-tip category.
- Print one `Tip: ...` message through the same small output helper as calibration
  reminders, preserving their current terminal/notebook stdout behavior without
  adding hints to PyVBMC's log file.
  Explain the disable option in the options reference and a brief Quickstart
  note, rather than attaching boilerplate to every tip.

Session-only history and the cadence are approved.
Do not add tip history to the calibration store or create another persistent store.

## Developer-editable catalog and URLs

Keep all tip content in one data-only module, `pyvbmc/vbmc/_tip_catalog.py`,
separate from scheduling in `_runtime_tips.py`. Each record has a stable `id`,
message `text`, `frequency` (`"normal"` or `"low_frequency"`), and optional
`urls` (an empty tuple when there are none). Both categories use the same format.
For example, one entry would contain:

```python
Tip(
    id="posterior_plot",
    text="Use vp.plot() to inspect parameter uncertainty and trade-offs "
         "after fitting. Broad marginal distributions indicate uncertainty; "
         "tilted or curved joint contours can reveal parameter trade-offs.",
    frequency="normal",
    urls=("https://acerbilab.github.io/pyvbmc/api/classes/variational_posterior.html",),
)
```

Developers edit the text or URLs in place, replace an entry, or add/remove a
record. Keep its ID for wording edits; use a new ID for a different topic.
Changing frequency is a field edit. Catalog length, IDs and category membership
must not be hard-coded in the selector or tests. The six/two split is only the
initial proposed content, not a fixed requirement on future catalog edits.
Add a short maintenance comment above the catalog explaining these operations.
Ordinary source edits take effect in a fresh Python process; live reloading or
user-supplied catalogs are not required.

Render `Tip: <text>` and, when present, each full URL on its own following line.
Keep URLs separate from prose so they are easy to replace and never manually
split or truncated. Plain stdout is the supported output: a terminal or notebook
may make the URL clickable, but clickability depends on that frontend. Do not
add HTML, terminal hyperlink escapes or a display dependency for this feature.
The same optional URL support belongs in the shared emitter; calibration can
continue calling it without URLs and retain its existing output exactly.

## Approved initial wording

Keep a small catalog with stable internal IDs. Messages are listed here by
topic rather than presentation order. The linked guidance following each message
is its `urls` content: render the full URL on a separate line at runtime,
using the shared emitter. These are reader-facing links, not just sources for
maintainers. All eight messages below are approved following the PI's edits
and instruction to implement. Preserve this wording in the initial catalog.

1. `multiple_runs` (wording approved): "Tip: Run VBMC 3–4 times with different
   starting points. Compare the posterior plots across runs; large differences
   can indicate that a run missed part of the posterior."
   [Multiple-run validation tutorial](https://acerbilab.github.io/pyvbmc/_examples/pyvbmc_example_4_validation.html)
2. `evidence_uncertainty`: "Tip: When comparing models, report results['elbo']
   together with results['elbo_sd']. The ELBO is a lower bound on the true log
   model evidence; its SD measures uncertainty in estimating that bound. A small
   SD does not guarantee that the bound is close to the true evidence!"
   [Interpreting the ELBO and its uncertainty](https://acerbilab.github.io/pyvbmc/quickstart.html)
3. `plausible_bounds`: "Tip: Use PLB and PUB to guide PyVBMC's initial search
   toward likely parameter values. If you have no better information, use each
   prior's 16th and 84th percentiles: roughly mean minus and plus one SD for a
   Gaussian prior. These guide the search without restricting the posterior."
   [Choosing plausible bounds](https://github.com/acerbilab/vbmc/wiki#how-do-i-choose-plb-and-pub)
4. `pybads` (low-frequency, wording approved): "Tip: A maximum-likelihood (MLE)
   or maximum a posteriori (MAP) estimate can provide a good starting point for
   PyVBMC. You can use PyBADS to find one, then pass it as x0."
   [PyBADS](https://acerbilab.github.io/pybads/)
5. `posterior_plot`: "Tip: Use vp.plot() to inspect parameter uncertainty and
   trade-offs after fitting. Broad marginal distributions indicate uncertainty;
   tilted or curved joint contours can reveal parameter trade-offs."
   [Posterior plotting and other methods](https://acerbilab.github.io/pyvbmc/api/classes/variational_posterior.html)
6. `save_resume`: "Tip: Save your run with vbmc.save('fit.pkl') so you can
   continue later without starting over. If it needs more evaluations, load it
   with VBMC.load('fit.pkl', new_options={'max_fun_evals': N}), then call
   optimize() on the loaded object. Set N to the larger total evaluation budget."
   [Saving and continuing a run](https://acerbilab.github.io/pyvbmc/api/classes/vbmc.html)
7. `noisy_target`: "Tip: PyVBMC can account for noise in simulated log-likelihood
   estimates if you supply its size. Set options={'specify_target_noise': True}
   and return (log_density, noise_sd) from your target, where noise_sd estimates
   the standard deviation of the noise in that log-density evaluation."
   [Noisy-likelihood tutorial](https://acerbilab.github.io/pyvbmc/_examples/pyvbmc_example_6_noisy_likelihoods.html)
8. `svbmc` (low-frequency): "Tip: Using S-VBMC, you can improve your posterior
   estimates by combining multiple independent PyVBMC runs on the same model and
   data. S-VBMC re-optimizes mixture weights to form a combined posterior without
   new model evaluations. See the tutorial to combine your runs and draw samples."
   [S-VBMC usage and posterior sampling](https://github.com/acerbilab/svbmc#how-to-use-s-vbmc)

Editorial requirement (PI): a tip must help the reader make a useful choice in
their PyVBMC workflow, explain why it helps, and link to guidance when useful.
A command alone, a description of an API, or instructions for another tool do
not meet that standard without the connection to the reader's task. In
particular, the PyBADS tip is about finding a good PyVBMC starting point; the
optimization recipe belongs in the linked documentation. Keep messages
concrete, clear and actionable. Use
specific quantities or useful ranges (such as 3–4), not vague quantities such
as "a few". Avoid unnecessary seed-management language: independent runs do
not require users to set seeds manually. The multiple-run wording above is
approved and follows the [VBMC FAQ](https://github.com/acerbilab/vbmc/wiki#how-can-vbmc-fail-how-do-i-find-out-and-how-do-i-fix-it);
the revised PyBADS message was accepted in the subsequent wording discussion.
The remaining six messages were edited and approved by the PI before
implementation. Each connects a useful action to its
purpose in PyVBMC and includes a link for the next step.
The prior-percentile suggestion follows the FAQ's 15.87–84.13% interval,
rounded to 16th and 84th at the PI's request, and is conditional on having no
better information. Prefer useful rounded quantities to unnecessary precision.

Populate the URL field for each complementary-method tip so the
reader can act on it: [PyBADS](https://acerbilab.github.io/pybads/) and
[S-VBMC](https://github.com/acerbilab/svbmc). Official documentation confirms
PyBADS minimizes objectives and S-VBMC combines VBMC posteriors without new
model evaluations. State the use case, avoid superlatives or promises of a global
optimum, and do not imply a method is required for an ordinary PyVBMC run.
S-VBMC is available separately; its pending PyVBMC integration does not prevent
mentioning it, but the tip/link must describe the interface actually released.

These tips give general or explicitly conditional guidance without inspecting
the target or guessing model intent. The Quickstart already explains plausible
bounds and `elbo_sd`; the VP API describes plotting. The linked multiple-run
and noisy-target examples were checked during this wording pass. `log_density`
in the noisy tip means the log joint, or the log likelihood when the prior is
supplied separately, as explained in Example 6; it must not suggest dropping
the prior. `N` in the resume tip is the total evaluation budget, not the number
of additional evaluations, and `new_options` is the supported load-time override.
The S-VBMC link explains its separate object and sampling interface; do not
promise a PyVBMC `VariationalPosterior` return value. Recheck released guidance
and links before shipping.
Context-dependent selection can be added when there is a clear applicability
rule; do not build a recommendation system now.
Keep wording consistent with the future user-agent skill.

## Integration findings and design

- Add a lightweight `pyvbmc/_user_hints.py::emit_user_hint(message, *, display, urls=())`
  that prints the supplied complete message and optional URL lines once if enabled
  and returns whether it emitted text. It must not import calibration, VBMC or RNG machinery. Both
  calibration's existing suggestion function and the tip selector use it.
  Preserve calibration's exact message, boolean return, fingerprint suppression
  and stdout behavior; this is an extraction of shared output, not a redesign of
  its cache policy. Callers own eligibility, prefix and once-only decisions.
- `pyvbmc/calibration/_cache.py::suggest_calibration_once` already returns
  whether it printed a reminder, and suppresses repeats per process/fingerprint.
  `VariationalPosterior._resolve_calibration` currently discards that boolean.
- Record a private boolean on the VP when a reminder is actually emitted;
  use `getattr(..., False)` for older objects. Never infer it from profile
  source/status: a cache miss can be silent because a different run already
  showed the reminder. Keep the resolver's profile return type unchanged.
  A private initialization/migration path must preserve an already recorded
  true value. A reminder on an unrelated VP does not suppress this run's tip.
- `VBMC.optimize()` restores the historical VP on continuation before resolving
  calibration. Leave this ordering intact. Run the tip decision once after
  resolution; no tip handling in PDF/entropy hot paths beyond recording the
  existing reminder outcome at the one-time resolution.
- A private VBMC startup-handled flag prevents repeated consideration, even
  when the first start was quiet or calibration took the slot. Save/deepcopy
  preserves it. Legacy saves with an existing iteration count are treated as
  already started; legacy pre-run saves remain eligible. Loading alone emits
  nothing. Do not add catalog/cadence data or new result/history fields; the
  VP's calibration-emission marker naturally follows its existing copies.
- Add a small private `pyvbmc/vbmc/_runtime_tips.py` importing the immutable
  catalog from `_tip_catalog.py` and holding the process-local
  counter/seen IDs, last-emitted category and selector. Keep all process state
  out of pickles; only the small VBMC/VP flags travel with a save.
  The selector should accept explicit eligibility inputs and be testable
  without constructing or optimizing a full model. Check display/option/slot
  eligibility before consuming a tip; commit the seen ID only after successful
  emission. The emitter's boolean also supplies the calibration VP marker.
- Build the shuffled order lazily on the first tip opportunity using a private
  stdlib `random.Random()` instance; inject a seeded private instance in tests.
  Shuffle the full catalog once. Normally take its first remaining entry. If
  both that entry and the last emitted tip are low-frequency, take the first
  ordinary entry still in the shuffled list instead; if none remains, take the
  first entry anyway. An empty list produces no tip. This finite scan needs no
  retries, gap construction or quota state, and cannot deadlock on one category.
  Do not use NumPy RNGs or stdlib module-level `random` draws. Initial tips are generally or
  explicitly conditionally worded; no applicability filtering is needed.
- Validate `show_tips` as a boolean on construction and load-time option
  overrides, following neighboring option-validation patterns. Old saved
  options missing the key receive the default before merging `new_options`;
  validate the merged value afterward so an explicit override is preserved.

## Implementation and verification

### 1. Startup policy and shared slot — Sol implementation

1. Read the relevant startup, VP resolver and save/load paths above. Extract the
   shared emitter into `pyvbmc/_user_hints.py`; use it in the calibration reminder
   and tip selector. Add the catalog in `vbmc/_tip_catalog.py`, the selector in
   `vbmc/_runtime_tips.py`, and
   the basic option in `vbmc/option_configs/basic_vbmc_options.ini`.
2. Propagate the actual reminder outcome to the private VP marker. Integrate
   the VBMC first-start flag and selector without changing calibration caching,
   reminder frequency, fixed profiles or historical continuation ordering.
3. Add focused cases in `pyvbmc/testing/vbmc/test_runtime_tips.py`, extending
   existing calibration/lifecycle tests where they already cover the seam.
   Keep tests deterministic and restore process-local tip/reminder state.

Acceptance checks:

- [x] First/fourth/seventh eligible starts, reproducible shuffled ordering with
  an injected RNG, exhausted catalog, and no repeated IDs in a process; inference,
  NumPy global and stdlib module-level RNG states remain unchanged.
- [x] Every catalog entry can be emitted exactly once, a low-frequency tip can
  appear first, and consecutive low-frequency tips are avoided whenever an
  ordinary entry remains. Verify progress with empty, single-entry, all-ordinary,
  all-low-frequency and exhausted-ordinary lists without statistical tolerances;
  quiet, cadence and calibration skips leave
  the next entry and last-emitted category unchanged.
- [x] Real calibration reminder at optimize startup excludes a tip; an early
  `vbmc.vp.pdf()` reminder also excludes it; a silent cache miss, cached profile,
  explicit profile or calibration-off mode does not by itself exclude tips.
- [x] Quiet, disabled tips and calibration-occupied slots do not consume cadence
  or catalog entries. Disabling tips leaves calibration feedback unchanged.
- [x] First start only, repeated optimize, completed/unfinished resume, current
  pre-run saves and legacy saves; load overrides validate and do not emit text.
- [x] Shared emitter's enabled/disabled return, unchanged calibration text,
  `iter` and `full` display, captured notebook-style stdout, and log-file exclusion;
  no new calls to target functions, numerical kernels or cache reads. Importing
  the helper introduces no cache I/O or heavy dependency imports.
- [x] Catalog edits need no selector changes: test adding, replacing and removing
  records of either frequency with small injected catalogs. Check unique IDs,
  nonempty text and recognized categories without fixing the catalog size or wording.
- [x] Normal and low-frequency tips render correctly with zero, one or multiple
  URLs; preserve each URL in full on its own line. Quiet output suppresses the
  message and URLs together. No network access, automatic opening or rich-output
  dependency is introduced; frontend clickability is not a test requirement.

### 2. Public guidance and review — Sol implementation/review; Astra integration

1. Add a short Quickstart note explaining occasional tips, session scope and
   `show_tips=False`. The basic option's comment owns its reference description.
   Add a concise startup-tips paragraph to the existing VBMC API page explaining
   calibration priority and continuation behavior. No new public API/page is needed.
2. Review wording against existing guidance; retain the distinction between
   ELBO uncertainty and evidence error. Verify complementary-method links and
   released interfaces; keep recommendations useful and nonpromotional. Update
   the overview only to reflect the feature's user benefit/availability once
   implemented, without delivery history.
3. Astra runs focused tips/calibration/options/save-load/seed tests sequentially,
   then the required package suite and docs build. Use existing short seeded-run
   fixtures for tips-on/off reproducibility; do not add full optimize runs.
   Static Sol review checks the final diff, output and lifecycle cases. Resolve
   findings and rerun only affected checks. No calibration campaign or population
   benchmark is needed for this presentation feature.

## Approved decisions

- **Session shuffle with best-effort spacing:** follows the PI's choice and gives
  short sessions access to low-frequency tips. A private RNG keeps presentation
  randomness separate from inference; restarting Python resets history and shuffles again.
- **First eligible start, then every third:** visible enough to discover, sparse
  in notebooks/batches. No public frequency knob initially; the boolean is enough
  unless actual usage shows a need for more control.
- **One consideration per run:** repeated optimization and resume stay focused
  on that run. A calibration reminder uses the slot even if shown before optimize.
- **Tips-only disable option:** lets users opt out of general guidance while
  preserving the existing calibration behavior. Quiet mode suppresses both.
- **Shared stdout emitter:** reuse calibration's current output behavior for
  both hints. Avoid a generic notification system or a second persistence layer.
- **Separate data-only catalog:** one place to edit text, categories and URLs,
  with scheduling and formatting independent of the current catalog contents.
- **Small low-frequency group:** the initial six/two catalog puts complementary
  methods at 25%, within the PI's suggested ceiling. The category can also hold
  other tips; it describes frequency, not promotion. Prefer separation when an
  ordinary tip remains, with no prefix quota or retry loop.

## Approval and review

- The PI authorized implementation after editing the messages on 2026-09-10.
  Cadence, option name/default, first-start-only behavior and the current wording
  are approved alongside shared-slot priority, session-only history, code reuse,
  shuffling and best-effort low-frequency spacing.
- Independent Sol structural plan review complete; no findings remain. Review clarified
  legacy option-override ordering and removed unnecessary future applicability
  cases. The final shuffled, best-effort spacing policy was reviewed separately.
  The PI's final wording edits are reflected above. Source implementation and
  verification are complete; the editorial pass did not change the reviewed
  scheduling or integration design.
  The subsequent catalog-editability/URL requirements are recorded above;
  they extend the planned catalog and emitter without changing startup policy.

Implementation is complete with `show_tips=True`, the 1/4/7 cadence and
first-start-only behavior. Future edits should keep the shared slot, session history and shuffled,
best-effort low-frequency spacing; do not restore deterministic catalog order
or a requirement for three ordinary tips before a low-frequency one. Concrete
commands and quantities/ranges are preferred to vague advice. The source
locations and implementation/test steps above are sufficient to resume from a
clean checkout; no ignored calibration artifacts or session jobs are needed.
