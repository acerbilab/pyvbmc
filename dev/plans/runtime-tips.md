# Optional runtime tips

Created: 2026-09-10. Status: **DRAFT — pending approval**.
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

Preserve quiet mode, all inference settings and both instance/global RNG
states. No target evaluation, timing campaign or dependency is introduced.
Only planning is authorized so far. Work on this plan directly on `dev-next`;
create a source feature branch only after implementation approval.

## Proposed user experience

- Basic option `show_tips=True`; `options={"show_tips": False}` disables tips.
  This option does not disable the calibration reminder. `display="off"`
  suppresses both, as it already suppresses ordinary startup output.
- Consider tips only on the first `optimize()` entry for a new run, after
  calibration resolution and the performance-settings summary, before the
  iteration headings. Resumed/continued runs do not show another tip.
- Proposed frequency: first eligible new run, then every third eligible run
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

Session-only history is settled. The cadence remains a recommendation for review.
Do not add tip history to the calibration store or create another persistent store.

## Initial wording for review

Keep a small catalog with stable internal IDs. Proposed messages, listed here
by topic rather than presentation order:

1. `multiple_runs` (wording approved): "Tip: Run VBMC 3–4 times with different
   starting points. Compare the posterior plots across runs; large differences
   can indicate that a run missed part of the posterior."
2. `evidence_uncertainty`: "Tip: Report results['elbo'] together with
   results['elbo_sd']. The SD measures uncertainty in the estimated ELBO,
   not its distance from the true log model evidence."
3. `plausible_bounds`: "Tip: If you have no better estimate of where the
   posterior lies, set PLB and PUB to the 15.9th and 84.1st percentiles of
   each parameter's prior. For a Gaussian prior, these are its mean minus
   and plus one standard deviation."
4. `pybads` (low-frequency): "Tip: To find a maximum-likelihood estimate with
   PyBADS, minimize -log_likelihood(x). For a MAP estimate, minimize
   -(log_likelihood(x) + log_prior(x))."
5. `posterior_plot`: "Tip: Call vp.plot() after fitting. The diagonal panels
   show uncertainty in each parameter; the off-diagonal panels show how pairs
   of parameters vary together."
6. `save_resume`: "Tip: Save a resumable run with vbmc.save('fit.pkl'). To
   continue it later, use vbmc = VBMC.load('fit.pkl'), then vbmc.optimize()."
7. `noisy_target`: "Tip: For a noisy log likelihood, return (log_likelihood,
   noise_sd) from your target and set options={'specify_target_noise': True}.
   Supply the standard deviation of the log-likelihood estimate, not its variance."
8. `svbmc` (low-frequency): "Tip: Combine the posteriors from your 3–4 runs
   on the same model and data with S-VBMC. It reweights their mixture components
   to produce one posterior without new model evaluations."

Editorial requirement (PI): tips must be concrete, clear and actionable. Use
specific quantities or useful ranges (such as 3–4), not vague quantities such
as "a few". Avoid unnecessary seed-management language: independent runs do
not require users to set seeds manually. The multiple-run wording above is
approved and follows the [VBMC FAQ](https://github.com/acerbilab/vbmc/wiki#how-can-vbmc-fail-how-do-i-find-out-and-how-do-i-fix-it);
the remaining messages above have been rewritten against this standard and
remain proposed for wording approval. Each gives a specific action, an exact
command/setting or useful quantity where relevant, and enough context to use it.
The prior-percentile suggestion follows the FAQ's 15.87–84.13% interval,
rounded for readability, and is conditional on having no better information.

Append a short documentation URL to each complementary-method tip so the
reader can act on it: [PyBADS](https://acerbilab.github.io/pybads/) and
[S-VBMC](https://github.com/acerbilab/svbmc). Official documentation confirms
PyBADS minimizes objectives and S-VBMC combines VBMC posteriors without new
model evaluations. State the use case, avoid superlatives or promises of a global
optimum, and do not imply a method is required for an ordinary PyVBMC run.
S-VBMC is available separately; its pending PyVBMC integration does not prevent
mentioning it, but the tip/link must describe the interface actually released.

These tips give general or explicitly conditional guidance without inspecting
the target or guessing model intent. The Quickstart already explains plausible
bounds and `elbo_sd`; the VP API describes plotting. Check the multiple-run
and noisy-target wording against the existing FAQ/examples before shipping.
Context-dependent selection can be added when there is a clear applicability
rule; do not build a recommendation system now.
Keep wording consistent with the future user-agent skill.

## Integration findings and design

- Add a lightweight `pyvbmc/_user_hints.py::emit_user_hint(message, *, display)`
  that prints the supplied complete message once if enabled and returns whether
  it emitted text. It must not import calibration, VBMC or RNG machinery. Both
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
- Add a small private `pyvbmc/vbmc/_runtime_tips.py` holding the immutable
  catalog (with an ordinary/low-frequency classification), process-local
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
   and tip selector. Add the catalog/selector in `vbmc/_runtime_tips.py` and
   the basic option in `vbmc/option_configs/basic_vbmc_options.ini`.
2. Propagate the actual reminder outcome to the private VP marker. Integrate
   the VBMC first-start flag and selector without changing calibration caching,
   reminder frequency, fixed profiles or historical continuation ordering.
3. Add focused cases in `pyvbmc/testing/vbmc/test_runtime_tips.py`, extending
   existing calibration/lifecycle tests where they already cover the seam.
   Keep tests deterministic and restore process-local tip/reminder state.

Acceptance checks:

- [ ] First/fourth/seventh eligible starts, reproducible shuffled ordering with
  an injected RNG, exhausted catalog, and no repeated IDs in a process; inference,
  NumPy global and stdlib module-level RNG states remain unchanged.
- [ ] Every catalog entry can be emitted exactly once, a low-frequency tip can
  appear first, and consecutive low-frequency tips are avoided whenever an
  ordinary entry remains. Verify progress with empty, single-entry, all-ordinary,
  all-low-frequency and exhausted-ordinary lists. Check the six/two catalog split
  without statistical tolerances; quiet, cadence and calibration skips leave
  the next entry and last-emitted category unchanged.
- [ ] Real calibration reminder at optimize startup excludes a tip; an early
  `vbmc.vp.pdf()` reminder also excludes it; a silent cache miss, cached profile,
  explicit profile or calibration-off mode does not by itself exclude tips.
- [ ] Quiet, disabled tips and calibration-occupied slots do not consume cadence
  or catalog entries. Disabling tips leaves calibration feedback unchanged.
- [ ] First start only, repeated optimize, completed/unfinished resume, current
  pre-run saves and legacy saves; load overrides validate and do not emit text.
- [ ] Shared emitter's enabled/disabled return, unchanged calibration text,
  `iter` and `full` display, captured notebook-style stdout, and log-file exclusion;
  no new calls to target functions, numerical kernels or cache reads. Importing
  the helper introduces no cache I/O or heavy dependency imports.

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

## Decisions for approval

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
- **Small low-frequency group:** the initial six/two catalog puts complementary
  methods at 25%, within the PI's suggested ceiling. The category can also hold
  other tips; it describes frequency, not promotion. Prefer separation when an
  ordinary tip remains, with no prefix quota or retry loop.

## Open questions and review

- Approve or adjust cadence, option name/default and wording. Shared-slot priority,
  session-only history, code reuse, shuffling and best-effort low-frequency spacing are
  settled; the six/two catalog is proposed within the PI's suggested ceiling.
- Independent Sol structural plan review complete; no findings remain. Review clarified
  legacy option-override ordering and removed unnecessary future applicability
  cases. The final shuffled, best-effort spacing policy was reviewed separately.
  The subsequent wording rewrite is proposed for PI review; only `multiple_runs`
  has explicit wording approval. No source implementation has begun.

Pickup: review the seven remaining proposed messages above, then settle
`show_tips=True`, the 1/4/7 cadence and first-start-only behavior before approving
implementation. Keep the settled shared slot, session history and shuffled,
best-effort low-frequency spacing; do not restore deterministic catalog order
or a requirement for three ordinary tips before a low-frequency one. Concrete
commands and quantities/ranges are preferred to vague advice. The source
locations and implementation/test steps above are sufficient to resume from a
clean checkout; no ignored calibration artifacts or session jobs are needed.
