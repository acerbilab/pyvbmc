# The release sweep of PyVBMC 1.5: ledger

Opened 2026-09-27 on `dev-next` at `05dc5dc9`. This is the sweep that
`dev/TODO.md` ("Release documentation and validation") asks for before the
release. It searched the tracked documents, records and docstrings for two
kinds of statement:

- (a) an acknowledgment that a value or a defect in some file is wrong,
  while that file carries neither the correction nor a flag;
- (b) a statement that was true when written and is stale now.

Any other false or misleading statement met on the way is also recorded.
The rule (PI, 2026-09-19): an error is fixed in the file that holds it;
where a record cannot change, the flag goes into the record, or as close to
it as its format allows; a note somewhere else is not a fix.

## Scope and method

Four read-only Opus reviewers each covered what one kind of reader reads:

- **A, users:** `README.md`, `CHANGELOG.md`, the Sphinx sources, the
  agent skill, the markdown of the example notebooks, and the catalogue of
  deliberate differences in the porting log `pyvbmc/vbmc/README.md`.
- **B, the API:** the docstrings and comments of `pyvbmc/` and the option
  descriptions of the two `.ini` files.
- **C, contributors and agents:** `AGENTS.md`, `dev/README.md`,
  `dev/TODO.md`, the READMEs under `dev/golden/` and `dev/experiments/`, and
  the `FIXTURES.md` files. Of the plans and results, C read only the status
  headers, live checklists, pickup points and headline conclusions; the
  worklogs are records.
- **D, the dated devlogs** `dev/*.md`, read in full.

No reviewer covered kind (a) as a whole. Each reviewer followed the
acknowledgments it met. The orchestrator searched all tracked text for
phrases of acknowledgment and checked each hit's target (findings E-n).
Findings that several reviewers noted outside their areas are X-n.

Out of scope:

- the documents on `feat-slurm-campaigns`, which a delta pass covers after
  their merge;
- the raw artifacts under `dev/scripts/runs/`.

Verification:

- The reviewers verified their findings in the code and the records.
- The orchestrator re-verified A-1 by two quadratures, and A-2, B-6 and
  D-2 in the code.
- Duplicates are merged under one ID, and each finding names its twins.
- Line numbers refer to `05dc5dc9`.

**Not a finding.** B noted that the pruning loop of `optimize_vp`
(`variational_optimization.py:377-379`) calls `np.delete` on `w` and `sigma`
without an axis. The setters of `VariationalPosterior` reshape both to
`(1, K)` on assignment, so they keep their shape (checked 2026-09-27).

## Rulings

Each finding carries a recommendation:

- **fix:** the proposed text, or its substance;
- **decide:** a choice for the PI;
- **leave:** no change needed.

The PI's ruling goes on its **Ruling** line. The fixes go on the branch
`docs-release-sweep`, and this ledger records each fix's commit. Changes to
code under `pyvbmc/` that a user can notice also get their `CHANGELOG.md`
line.

The findings that need a decision: A-2, A-14, A-18, B-1, B-2, B-23, C-3,
C-24, C-37, E-1, E-3 and X-3.

## A: what users read

**A-1** · fix, at the release's re-execution of the notebooks · no one ·
high
- **Where:** `examples/pyvbmc_example_2_inputs_outputs.ipynb`, cell 18;
  `examples/scripts/pyvbmc_example_2_full_code.py:58`.
- **Finding:** Example 2 gives the true log evidence of its target as
  −1.836. It is −1.8396 (adaptive quadrature and a fine grid, both computed
  by the orchestrator). The difference, 0.004 nats, is far below anything
  the example or its records compare, so no conclusion depends on it.
- **Fix:** `lml_true = -1.8396`, with the script regenerated, when the
  notebooks are re-executed for the release. The TODO and the rotoscaling
  report can keep −1.836.
- **Ruling:**

**A-2** · decide · users · high
- **Where:** `docsrc/source/faq.md:573`, `:586`.
- **Finding:** The FAQ says that noisy inference needs
  `specify_target_noise` and that VBMC does not infer the noise level.
  `uncertainty_handling=True` without `specify_target_noise` is level 1:
  the target returns the log density alone, and VBMC infers the noise
  (`vbmc.py:1362-1364`).
- **Fix:**
  - :573: "No. To perform inference with a noisy target function that
    estimates its own noise, you need to:"
  - :586: "Only when asked to. With `options={"uncertainty_handling":
    True}` and without `specify_target_noise`, the target returns the log
    density alone and VBMC infers the noise level from the evaluations. A
    target that can estimate the SD of each evaluation should return it,
    as described above."
- **Decision:** whether the FAQ keeps recommending that a target supply
  its SDs.
- **Ruling:**

**A-3** · fix · users · high
- **Where:** `examples/pyvbmc_example_7_stacking.ipynb`, Conclusions.
- **Finding:** "the object cannot check this". `SVBMC` refuses runs that
  differ in dimension or in their hard bounds (`svbmc.py:68-183`).
- **Fix:** "The runs must describe the same model and data, which the
  object cannot check; it refuses runs that differ in the number of
  parameters or in the hard bounds."
- **Ruling:**

**A-4** · fix · users · high
- **Where:** `examples/pyvbmc_example_8_pymc.ipynb`, Conclusions, second
  bullet.
- **Finding:** The bullet says one-sided variables keep a
  "dimension-preserving" transform and that the adapter rejects transforms
  that change dimension. The adapter accepts only the log, log-odds and
  interval transforms (`pymc/_target.py:18`), and refuses every other
  transform, dimension-preserving ones included.
- **Fix:** "The adapter supports continuous float64 free variables that
  are unbounded, or that carry PyMC's log, log-odds or interval transform;
  a one-sided variable keeps its log transform and its Jacobian. It rejects
  discrete variables, any other transform (an ordered, a simplex or a
  custom one), unbounded distributions it does not recognize, and bounds
  that depend on other random variables."
- **Ruling:**

**A-5** · fix · users · high
- **Where:** `CHANGELOG.md:1084-1087`.
- **Finding:** The entry says a list of one-dimensional priors that holds
  a `UserFunction` works as a prior. `Product` refuses a marginal whose
  `D != 1`, and `UserFunction` defaults to `D=None`.
- **Fix:** "… that holds a `UserFunction` made with `D=1` works as a
  prior …"
- **Ruling:**

**A-6** · fix · users · high
- **Where:** `CHANGELOG.md:752-754`.
- **Finding:** The entry says a saved run "still contains the target
  function". A saved `VBMC` always holds the bytecode of the options whose
  value is a function, and the target's only when it cannot be pickled by
  name.
- **Fix:** "A saved *run* (`VBMC.save`) still holds Python bytecode, that
  of the options whose value is a function and often that of the target:
  under another Python version it can be loaded and inspected, but should
  not be continued or saved again there."
- **Ruling:**

**A-7** · fix · users · high
- **Where:** `CHANGELOG.md:140-146`, the "Upgrading" lead.
- **Finding:** The lead omits a change that its entry at :986-987 lists:
  `ParameterTransformer` refuses a `rotation_matrix` that is not
  orthogonal.
- **Fix:** Add "a `rotation_matrix` that is not orthogonal" to the lead's
  list.
- **Ruling:**

**A-8** (= B-21) · fix · users · high
- **Where:** `docsrc/source/api/classes/priors.rst:97`.
- **Finding:** The page names a class `UserPrior`, which does not exist.
  The class is `UserFunction`.
- **Fix:** "… it is wrapped in a ``UserFunction`` prior."
- **Ruling:**

**A-9** · fix · users · high
- **Where:** `docsrc/source/index.rst:59`.
- **Finding:** The page says the points of the current iteration are
  green. They are orange, as the README says correctly.
- **Fix:** "(*hollow blue*: previously sampled points, *orange*: points
  sampled in the current iteration)"
- **Ruling:**

**A-10** · fix · contributors · high
- **Where:** `docsrc/source/development.rst:104-117`.
- **Finding:** The section lists API-page sections (port status, MATLAB
  reference, known issues) that no page has, and says each new file goes
  in the index page.
- **Fix:** The rule of `AGENTS.md`: a hand-written `.rst` under
  `docsrc/source/api/`, with an entry in `documentation.rst` for a
  headline page and in `api/classes/classes.rst` or
  `api/functions/functions.rst` otherwise.
- **Ruling:**

**A-11** (= B-3, D-13) · fix · users · high
- **Where:** `pyvbmc/vbmc/_tip_catalog.py:115-126`.
- **Finding:** The runtime tip on S-VBMC links the standalone package's
  README, whose interface differs from `pyvbmc.SVBMC`. D-13 notes that
  `dev/2026-09-12-svbmc-elbo-optimism.md:285-287` recorded this change as
  to do, and no tracker carries it.
- **Fix:** "… See Example 7 to combine your runs with pyvbmc.SVBMC and
  draw samples.", with the URL
  `https://acerbilab.github.io/pyvbmc/_examples/pyvbmc_example_7_stacking.html`.
  The devlog records that it is done.
- **Ruling:**

**A-12** · fix · users · high
- **Where:** `CHANGELOG.md:1075-1076`.
- **Finding:** The entry calls `prior.a` and `prior.b` "read-only arrays".
  They are setter-less properties that return a fresh array at each
  access, so writing into the array returned succeeds and has no effect.
- **Fix:** "… are read-only attributes, recomputed from the distribution
  at each access: a script can no longer assign to them, and a change made
  to the array one of them returns does not reach the prior."
- **Ruling:**

**A-13** · fix · contributors · high
- **Where:** `pyvbmc/vbmc/README.md:455-461`.
- **Finding:** The entry says `precomputed_evaluations` rows are charged
  a `setup_cost`. The rows are not charged. `initialization_cost` is
  charged once, and a `PyMCTarget` passes its `setup_cost` as that cost.
- **Fix:** As stated.
- **Ruling:**

**A-14** · decide · contributors · high
- **Where:** `pyvbmc/vbmc/README.md:1071-1073`.
- **Finding:** The note says the logging's shortcomings "are not recorded
  in this repository". `1f9c207a` records and fixes one of them.
- **Fix:** Name `1f9c207a` and the "Logging" convention of `AGENTS.md`.
- **Decision:** whether a logging redesign is still wanted.
- **Ruling:**

**A-15** · fix · contributors · medium
- **Where:** `pyvbmc/vbmc/README.md:388-392`.
- **Finding:** The entry says that with
  `tol_stable_warmup <= fun_evals_per_iter` the warm-up check leaves the
  flag false. That holds at the first check only, and at every check only
  when `tol_stable_warmup = 0`.
- **Fix:** As stated.
- **Ruling:**

**A-16** · fix · users · high
- **Where:** `docsrc/source/api/functions/calibrate.rst:66-68`.
- **Finding:** The page says a calibration suggestion accompanies every
  missing or incompatible cache record. It is printed once per Python
  session.
- **Fix:** "… and the first such run of a Python session suggests
  calibrating unless ``display="off"``."
- **Ruling:**

**A-17** · fix · users · high
- **Where:** `CHANGELOG.md:106-107`, the lead.
- **Finding:** The lead is broader than its entry: 1.0.4 truncated a
  fractional number of samples in `kl_div` only with `gauss_flag=True`.
- **Fix:** "`vp.moments` and `vp.kl_div(gauss_flag=True)` refuse …"
- **Ruling:**

**A-18** · decide · no one · medium
- **Where:** the last cell of all nine notebooks.
- **Finding:** The cells credit the funding to FCAI alone. The README,
  `index.rst` and `about_us.rst` name the Research Council of Finland
  grants and its Flagship programme FCAI.
- **Fix:** Use the README's wording.
- **Decision:** whether the notebooks should match.
- **Ruling:**

**A-19** · fix · users · high
- **Where:** `README.md:223`, the suggested citation sentence.
- **Finding:** "(PyVBMC; Acerbi, 2018, 2020)" attributes the software to
  the algorithm papers. `index.rst:137` has the corrected form.
- **Fix:** "… using Variational Bayesian Monte Carlo (VBMC; Acerbi, 2018,
  2020) via the PyVBMC software (Huggins et al., 2023) …"
- **Ruling:**

**A-20** (= B-19) · fix · no one · high
- **Where:** `calibrate.rst:41-43`.
- **Finding:** "Thirty seconds is an estimate": nothing on the page or in
  the printed output gives thirty seconds, and the watchdog stops a
  campaign at 300 s.
- **Fix:** "Expect a duration in the tens of seconds, depending on your
  machine. The duration is not capped: the campaign completes its
  measurements and validation, and a watchdog stops a campaign that runs
  longer than five minutes. Run it when the machine is otherwise quiet for
  useful timings."
- **Ruling:**

**A-21** (twin B-12) · fix · no one · high
- **Where:** `CHANGELOG.md:1050-1051`.
- **Finding:** The entry says `set_parameters` requires the entries to be
  positive. It refuses negative values only, so zero passes.
- **Fix:** "… refuses a negative value in the entries that hold `sigma`,
  `lambd` and the weights, and checks those alone." B-12 fixes the
  docstring.
- **Ruling:**

**A-22** · fix · no one · high
- **Where:** `pyvbmc/vbmc/README.md:808-812`.
- **Finding:** The entry says weights that do not sum to one raise. That
  holds only in the unbalanced draw of a posterior with several
  components.
- **Fix:** Say so.
- **Ruling:**

## B: the API

**B-1** · decide · users · high
- **Where:** `advanced_vbmc_options.ini:248-255`: `fitness_shaping` and
  `out_warp_thresh_*`.
- **Finding:** Output warping is not ported (`pyvbmc/vbmc/README.md:66-69`
  acknowledges it), and the descriptions do not say so.
  `fitness_shaping=True` also draws 20000 points at each check and so
  advances the run's random stream.
- **Options:** say "not used, output warping is not ported" in each
  description; or refuse `fitness_shaping=True` with
  `NotImplementedError`, as for `noise_shaping`, which changes behavior and
  needs a changelog entry.
- **Ruling:**

**B-2** · decide · users · high
- **Where:** `advanced_vbmc_options.ini:212-215` (`det_entropy_alpha`,
  `update_random_alpha`) and `:188` (`moments_run_weight`).
- **Finding:** These options are read, but they reach no computation.
  MATLAB removed the entropy interpolation in 2021, and nothing reads the
  running moments. `update_random_alpha` still advances the random stream.
- **Options:** say so in the descriptions; or make them inert, which
  changes behavior.
- **Ruling:**

**B-3:** merged into A-11.

**B-4** · fix · users · high
- **Where:** `vbmc.py:2751-2794`, the `final_boost` docstring.
- **Finding:** The docstring has:
  - a wrong `changed_flag` (it is False when the guard rejects the boost);
  - an incomplete Raises section (ValueError for `tol_elcbo_boost`, and
    RuntimeError);
  - an unstated guard rule, `min(dE, dE - 5 dSD) > -tol`;
  - "entropy annealing switched off", which is inert: the boost actually
    sets `tol_weight=0`, the `ns_ent*_boost` sizes,
    `max_iter_stochastic=inf` and, under the guard, `weight_penalty=0`;
  - `gp : GaussianProcess`, which should be `gpyreg.GP`.
- **Fix:** B's full text.
- **Ruling:**

**B-5** · fix · users · high
- **Where:** `vbmc.py:297-298` (docstring), `:826-827` (message).
- **Finding:** Both say the hard bounds must equal the "model support". The
  code compares them with `target.lb` and `target.ub`, which are infinite
  for a variable that keeps its PyMC transform.
- **Fix:**
  - docstring: "Supplied hard bounds must equal ``target.lb`` and
    ``target.ub``, which are in the target's flat coordinates: the support
    of a variable the adapter leaves untransformed, and infinite for a
    variable that keeps its PyMC transform."
  - message: "{name} must equal the PyMCTarget's hard bounds (target.lb
    and target.ub, in its flat coordinates)."
- **Ruling:**

**B-6** · fix (code) · users · high
- **Where:** `iteration_history.py:173-177`.
- **Finding:** `str(vbmc.iteration_history)` reports "num. iterations =
  25" for every run: `len()` counts the keys.
- **Fix:** Report the length of the longest recorded entry (B's code).
- **Ruling:**

**B-7** · fix · users · high
- **Where:** `vbmc.py:422-428`, the class's Raises section.
- **Finding:** The section omits the `NotImplementedError` for the
  unported features, and the `ValueError` for refused options,
  `precomputed_evaluations`, `initialization_cost`, a prior that does not
  cover the bounds, and a design without spread.
- **Fix:** B's text.
- **Ruling:**

**B-8** · fix · users · high
- **Where:** `priors/prior.py:96-98`, and the private `_support` of each
  prior.
- **Finding:** The docstring says returning "a box which bounds the
  support" is acceptable. `VBMC` now checks the hard bounds against this
  box, so a box looser than the support defeats the check.
- **Fix:** Require the smallest box that contains the support (B's text).
- **Ruling:**

**B-9** · fix (code) · users · high
- **Where:** `create_vbmc_animation.py:30-33`, `:76-89`.
- **Finding:** With `suptitle="full"`, the default, the branch that adds
  the logging action is unreachable. The Returns section also omits the
  returned figure.
- **Fix:** Test for "full" first, and document the return.
- **Ruling:**

**B-10** · fix · users · high
- **Where:** `variational_posterior.py:1939-1953`, `VariationalPosterior.load`.
- **Finding:** The `calibration` override is described loosely, and the
  `ValueError` it raises is not documented.
- **Fix:** B's text.
- **Ruling:**

**B-11** · fix · users · high
- **Where:** `variational_posterior.py:1741`, `:1768`, `:1777`, `plot`.
- **Finding:** `highlight_data` is documented as a list, which raises;
  it is an ndarray. The default figure size is `(6, 6)`, not `(8, 8)`.
- **Fix:** As stated.
- **Ruling:**

**B-12** (twin A-21) · fix · users · high
- **Where:** `variational_posterior.py:1259-1263`, `:1281-1284`.
- **Finding:** The docstring and the message say "positive"; the check
  refuses negative values only.
- **Fix:** "Raised if `raw_flag` is ``False`` and a scale or a weight in
  `theta` is negative." Change the message to match.
- **Ruling:**

**B-13** · fix · users · high
- **Where:** `variational_posterior.py`.
- **Findings:**
  - (a) `x0`: only a 1-by-1 array starts every coordinate.
  - (b) `mtv`: the result has shape `(1, D)`, and the argument is `vp2`.
  - (c) `kl_div`: `gauss_flag=True` with `N=0` raises, undocumented.
  - (d) `mode`: the stored mode is read and written only in the original
    space without `n_opts`.
  - (e) `df`: a negative value gives products of univariate t densities.
- **Fix:** B's texts.
- **Ruling:**

**B-14** · fix · users · high
- **Where:** `whitening.py:182-195`, `warp_input`.
- **Finding:** The docstring omits the `options` parameter and the
  `ValueError` for `warp_cov_reg`.
- **Fix:** Document both.
- **Ruling:**

**B-15** · fix · users · high
- **Where:** `advanced_vbmc_options.ini:128`, `stochastic_optimizer`.
- **Finding:** Only "adam" is implemented, and another value raises only
  at the first stochastic optimization.
- **Fix:** Say so, and correct "varational".
- **Ruling:**

**B-16** · fix · contributors · high
- **Where:** `active_sample.py:680-724`, `:120-122`.
- **Finding:** A "Missing port" block of commented MATLAB repeat logic,
  and a TODO on level-2 cache values, are both superseded: repeats are
  selected in the sieve, and level-2 values come through
  `precomputed_evaluations`.
- **Fix:** B's replacement comments.
- **Ruling:**

**B-17** (also A) · fix · contributors · high
- **Where:** `vbmc.py:510-513`.
- **Finding:** The comment says the generator is handed to "the CMA-ES
  noise handler". The search has had no noise handler since `d617d993`.
- **Fix:** B's comment.
- **Ruling:**

**B-18** · fix (code) · users · high
- **Where:** `vbmc.py:4514-4532`, `__repr__`.
- **Finding:** The repr lists `log_prior`, `sample_prior` and `K`, which
  `VBMC` does not have, so it prints `log_prior = None` for a run with a
  prior.
- **Fix:** Use `"prior"` and `"vp.K"`.
- **Ruling:**

**B-19:** merged into A-20.

**B-20** · fix · no one · high
- **Where:** `calibrate.rst:29-30`.
- **Finding:** The sample output splits one printed line in two.
- **Fix:** Join the two lines.
- **Ruling:**

**B-21:** merged into A-8.

**B-22** · fix · users · high
- **Where:** `advanced_vbmc_options.ini:124`.
- **Finding:** The description calls the option the "size of cache". It is
  the initial number of rows, and the arrays grow as needed.
- **Fix:** "Initial number of rows of the arrays that store the fcn
  evaluations (they grow as needed)"
- **Ruling:**

**B-23** (= X-2) · decide · agents, users · high on the facts
- **Where:** `pyvbmc/pymc/_target.py:20`; `pyvbmc/_logging.py`;
  `AGENTS.md`, "Logging".
- **Finding:** The PyMC adapter takes its logger from `logging.getLogger`
  and not from `get_logger`. With no logging configured, its warnings go
  to stderr through the last-resort handler, while the rest of PyVBMC
  writes to stdout. `AGENTS.md` says every package logger comes from
  `get_logger`.
- **Options:** change the code to `get_logger("pyvbmc.pymc")`, which moves
  those warnings to stdout (a changelog line); or record the exception in
  `AGENTS.md` and `_logging.py`.
- **Ruling:**

**B-24** · fix · contributors · high
- **Where:** `svbmc.py:7-8`; `svbmc.rst:238-239`.
- **Finding:** Both say the module "is" the standalone package moved in.
  It has been largely rewritten since.
- **Fix:** "derives from", with a pointer to the changelog's S-VBMC entry.
- **Ruling:**

**B-25** · fix · users · high: rendered docstrings of lower impact
- `function_logger.add` and `__call__`: document the space of `x`, the
  Jacobian in `f_val`, and that `SD` is None at level 0.
- `parameter_transformer.py:29-36`: the order of the bounds is
  `lb <= plb < pub <= ub`, and the plausible bounds default to the hard
  bounds.
- `parameter_transformer.py:275-278`: the math markup is broken.
- `acq_fcn_viqr.py:476-478`, `acq_fcn_imiqr.py:215-217`: the formula is
  `log[2 sinh(u f_s)]`, and `f_s` is the SD, not the variance.
- `acq_fcn_imiqr.py:65-72`: "at the candidate points".
- `active_sample.py:66-72`, `:101-102`: `gp` may be None.
- `entlb_vbmc.py:31-34`: the Raises section describes an unreachable
  branch.
- `handle_0D_1D_input.py:9-29`: document the method-only contract.
- `smooth_box.py:42-45`: document the default and scalar form of `scale`.
- `tile_inputs.py:24-25`: the default of `size`, and a Returns section.
- `product.py:116-119`: a `UserFunction` marginal does not share the
  generator.
- `timer.py:40-41`: "stopped". `timer.py:97`: "Default `True`".
- Rendered typos: "likelhood", "one-dimenional", "nd"
  (`vbmc.py:284`, `:355`, `:1196`, `:1395`; `vbmc.rst:12`).
- **Ruling:**

**B-26** · fix · contributors · high: comments and private docstrings
- `variational_optimization.py:587-589`: the penalty is in `_neg_elcbo`.
- `variational_optimization.py:1178-1208`: state when the outputs are
  returned.
- `variational_optimization.py:140-145`: the `stochastic_optimizer` path
  (see B-15).
- `minimize_adam.py:34-53`: four parameters are described as "?", and the
  returns are means over the trailing window.
- `active_importance_sampling.py:384-386`: a nonexistent `vp` parameter.
- `active_importance_sampling.py:93-97`: three undocumented ValueErrors.
- `whitening.py:350`: the comment labels the wrong reset.
- `abstract_acq_fcn.py:337-338`: "rows", not "columns".
- `abstract_acq_fcn.py:387`: a stale comment.
- `variational_posterior.py:1180-1182`, `:1255-1257`: the weights are raw
  too.
- `variational_posterior.py:919-927`: the comment describes a former loop.
- `parameter_transformer.py:113`, `:137`, `:379`: stale comments.
- `iteration_history.py:201`: a missing `f` prefix prints the literal
  `{self.check_keys}` (code).
- **Ruling:**

**B-27** · fix (code) · no one · high
- **Where:** `_bounds.py:112-113`.
- **Finding:** The log message reads "Estimatingplausible bounds".
- **Fix:** Add the missing space.
- **Ruling:**

## C: what contributors and agents read

**C-1** · fix · agents · high
- **Where:** `dev/plans/modernization-roadmap.md:645-651` (3f "Next"),
  `:452-458` (3a "Open"), `:703-757` (point 9), `:808-809` (point 10).
- **Finding:** The roadmap's pickup point names work done since
  2026-09-13, and a branch that no longer exists.
- **Fix:** Open "Pickup point" with a pointer to TODO's "In scope for
  1.5", and mark 3f, 3a, 9 and 10 done or settled, with their records
  (C's texts).
- **Ruling:**

**C-2** · fix · agents · medium-high
- **Where:** `AGENTS.md:307-311`, "Trajectories".
- **Finding:** The gate replays against `reference_990_20260913`. That
  reference predates the port review's fixes, which the TODO,
  `dev/README.md` and `dev/golden/README.md` say, and `AGENTS.md` does not.
  So the default configurations part from it whatever a change does.
- **Fix:** Say so, and describe the interim gate: replay at the parent
  commit, then pass that replay's `--out` as `--baseline` (C's text; C did
  not run it).
- **Ruling:**

**C-3** · decide · agents · high
- **Where:** `AGENTS.md:328-331`; `dev/README.md:479`, `:483`, `:489`,
  `:495`.
- **Finding:**
  - Merged `feat-*` and `dev-*` branches remain, locally and one on
    origin (`dev-port-review`).
  - `dev-final-boost`, which holds the parked boost scripts, is local only.
  - `dev-boost-campaign` and `dev-acq-regularization-stage` are the
    checked-out branches of frozen worktrees.
- **Options:** push `retain/final-boost` at `764a177` and point
  `dev/README.md` at it; delete the merged branches; detach the two frozen
  worktrees. Or qualify the `AGENTS.md` sentence.
- **Ruling:**

**C-4** · fix · agents · high
- **Where:** `modernization-roadmap.md:40-54`.
- **Finding:** "The remaining numerical work … pending": the efficiency
  workstream closed on 2026-09-19 without a change of default.
- **Fix:** C's text.
- **Ruling:**

**C-5** · fix · agents · high
- **Where:** `modernization-roadmap.md:163`, `:281`, `:297-298`,
  `:305-306`, `:355-356`.
- **Finding:** Items that are done are still marked `[~]`, and the text
  names branches that no longer exist.
- **Fix:** C's texts.
- **Ruling:**

**C-6** · fix · agents · high
- **Where:** `modernization-roadmap.md:694`, `:1010-1011`, `:1112-1113`.
- **Finding:** The text gives "870 traces across 19 configurations", says
  "S-VBMC integration is in progress", and calls the Slurm design "for the
  PI's review", which took place on 2026-09-25.
- **Fix:** C's texts.
- **Ruling:**

**C-7** · fix · agents · high
- **Where:** `dev/plans/noisy-acquisition-efficiency.md:9-11`;
  `dev/README.md:121-125`.
- **Finding:** The status and branch are stale. The README entry says
  "conditional adaptation" ran, but E4 was never entered.
- **Fix:** C's texts.
- **Ruling:**

**C-8** · fix · agents · high
- **Where:** `dev/plans/stage3-pipeline-features.md:3-4`;
  `dev/README.md:271-278`.
- **Finding:** "reference campaign awaits start instruction": it ran on
  2026-09-06/07. The plan also names a branch that is gone.
- **Fix:** C's texts.
- **Ruling:**

**C-9** · fix · agents · high
- **Where:** `dev/plans/stage2-gpyreg-predict-and-sampler.md:3`,
  `:650-651`, `:1281`.
- **Finding:** "IN PROGRESS", an open question and an open population
  item: all were done by 2026-09-06.
- **Fix:** C's texts.
- **Ruling:**

**C-10** · fix · agents · high (medium for the run ID)
- **Where:** `stage2-entmc.md:349`, `:420-422`, `:695`;
  `stage2-gp-log-joint-einsum.md:421`, `:446`.
- **Finding:** The population runs and the CI smoke run are marked
  pending; they ran and passed.
- **Fix:** C's texts.
- **Ruling:**

**C-11** · fix · agents · high
- **Where:** `dev/plans/latent-bug-fixes.md:1-4`, `:73`.
- **Finding:** The plan has no status line, and says the matrix and the
  population "remain pending". The work is complete, and was promoted on
  2026-09-13.
- **Fix:** C's status line.
- **Ruling:**

**C-12** · fix · agents · high
- **Where:** `dev/plans/fixture-generator-and-oracles.md:3-5`.
- **Finding:** "commit pending; the CI matrix has not yet run": both were
  done.
- **Fix:** C's text.
- **Ruling:**

**C-13** · fix · agents · high
- **Where:** `dev/README.md:143-144`.
- **Finding:** The entry says calibration delivery is tracked in the TODO,
  which has no calibration item.
- **Fix:** "Merged into `dev-next` at `46c16b7` with the full CI matrix
  passed; the recipe was revised (v2) on 2026-09-23 after the port
  review."
- **Ruling:**

**C-14** · fix · agents · high
- **Where:** `dev/README.md`, the plans index.
- **Finding:** `plans/final-population-benchmark.md`, which the TODO
  links, and `plans/runtime-tips.md` have no entries.
- **Fix:** Add C's two entries.
- **Ruling:**

**C-15** · fix · agents · medium
- **Where:** `dev/plans/svbmc-benchmark-campaign.md:3`.
- **Finding:** "complete", but decisions 12 and 14 put a rerun at the
  release gate.
- **Fix:** Append the pointer to the TODO's gate item.
- **Ruling:**

**C-16** · fix · agents · high
- **Where:** `dev/README.md:835-837`.
- **Finding:** The entry dates the re-baselining of the oracle references
  to the end of Stage 2. They have been re-baselined repeatedly since,
  most recently on 2026-09-24.
- **Fix:** C's text.
- **Ruling:**

**C-17** · fix · contributors · high
- **Where:** `dev/README.md:323`.
- **Finding:** As a shell command, `export
  OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` sets one
  variable, to a string.
- **Fix:** `export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  MKL_NUM_THREADS=1`
- **Ruling:**

**C-18** · fix · contributors · high
- **Where:** `dev/results/2026-09-07-final-boost-comparison.md:3-16`.
- **Finding:** "the decision is deferred": the PI chose the 0.1 guard on
  2026-09-08.
- **Fix:** A dated decision line, pointing at the boost analysis.
- **Ruling:**

**C-19** · fix · contributors · high
- **Where:** `dev/results/2026-09-04-final-boost-failure.md:3-8`.
- **Finding:** "No code changed": the guard was implemented on 2026-09-08.
- **Fix:** C's text.
- **Ruling:**

**C-20** · fix · contributors · high
- **Where:** `dev/results/2026-09-11-noisy-follow-up-seeds-15-29.md:24-26`;
  `dev/results/2026-09-11-overnight-population.md:19-27`.
- **Finding:** Both recommend against further runs. The PI had the full
  870 run, and they were promoted on 2026-09-13.
- **Fix:** A dated line in both headers.
- **Ruling:**

**C-21** · fix · agents · high
- **Where:** `dev/experiments/machine_calibration/integrated/README.md:3-4`;
  `dev/results/2026-09-09-machine-local-calibration.md:5`.
- **Finding:** Both name the deleted branch `dev-machine-calibration`.
- **Fix:** Name the merge (`f2f99e36`; `46c16b7`).
- **Ruling:**

**C-22** · fix · agents · high
- **Where:** `dev/experiments/svbmc_speedups/README.md:28-32`.
- **Finding:** The README describes `benchmark_first.json`, which no
  commit holds.
- **Fix:** Say that the output is not kept, and that the report quotes
  its numbers; or commit the file if it survives locally.
- **Ruling:**

**C-23** · fix · agents · high
- **Where:** `dev/experiments/noisy_acq_20260908/README.md:9-12`,
  `:30-35`.
- **Finding:** The README names a deleted branch. Its EIG and
  `var_reduction` arms need acquisitions that were removed on 2026-09-14.
- **Fix:** C's note, pointing at `retain/experimental-acquisitions`.
- **Ruling:**

**C-24** · decide · contributors · medium
- **Where:** `dev/README.md:849-852`; the three
  `oracles/fixtures/gp_fit_history/*.json`.
- **Finding:** The flag on the leftover `hyp_dict_logp` arrays lives in
  `dev/README.md` and the port-review ledger, not in the captures' own
  sidecars.
- **Options:** add a `meta["notes"]` entry to each sidecar, a hand-edit of
  generator-written files; or leave the flag where it is. Either way,
  change "no sidecar describes them" to "no reference names them".
- **Ruling:**

**C-25** · fix · agents · high
- **Where:** `dev/golden/promotion_20260913/README.md:6-8`, about
  `previous_reference_README.md`.
- **Finding:** Nothing says that the file is a verbatim snapshot. Its
  "Current reference" is the previous one, and its links resolve from
  `dev/golden/`.
- **Fix:** C's sentence.
- **Ruling:**

**C-26** · fix · agents · high
- **Where:** `dev/golden/extension_20260907/README.md:75`, `:108-109`,
  `:118-120`; `dev/golden/noisy_extension_20260907/README.md:40`.
- **Finding:** These dated records link the live `baseline/summary.md`,
  which the files have replaced twice since.
- **Fix:** Cite the commits `2a09fcd7` and `b2ea859` (C's texts).
- **Ruling:**

**C-27** · fix · agents · high
- **Where:** `dev/golden/realdata_extension_20260912/README.md:155-157`.
- **Finding:** "with the current code": the text holds as of 2026-09-12.
- **Fix:** C's text.
- **Ruling:**

**C-28** · fix · contributors · high
- **Where:** `dev/README.md:805-810`; `dev/scripts/svbmc_parity_check.py:12`.
- **Finding:** "Runs only from `a8ae260`": that commit lacks the script,
  which `6103be8` added.
- **Fix:** C's text.
- **Ruling:**

**C-29** · fix · contributors · high (medium for the environment)
- **Where:** `dev/README.md:796-801`, `pymc_setup_probe`.
- **Finding:** The entry omits the `coverage` and `coverage_summary`
  subcommands, and names the feasibility environment, removed on
  2026-09-23.
- **Fix:** C's text.
- **Ruling:**

**C-30** · fix · contributors · high
- **Where:** `pyvbmc/testing/svbmc/FIXTURES.md:136`, `:138`.
- **Finding:** The table lists what `test_elbo_reporting.py` reads
  incompletely, and omits `test_elbo_shrinkage.py`.
- **Fix:** C's rows.
- **Ruling:**

**C-31** · fix · contributors · high
- **Where:** `pyvbmc/testing/variational_posterior/FIXTURES.md:31-36`.
- **Finding:** Four tests use `get_matlab_vp()`, not two.
- **Fix:** C's text.
- **Ruling:**

**C-32** · fix · contributors · high
- **Where:** `pyvbmc/testing/whitening/FIXTURES.md:19-23`.
- **Finding:** Two more tests read `test_warp_input_rands.txt`.
- **Fix:** C's text.
- **Ruling:**

**C-33** · fix · contributors · medium
- **Where:** `dev/experiments/gp_box_20260916/README.md:33`, `:38-39`.
- **Finding:** The README runs the uniform arm with `--source-root .`, and
  does not name the fix commit `40a6f18c`.
- **Fix:** C's text.
- **Ruling:**

**C-34** · fix · agents · high
- **Where:** `dev/golden/noisy_extension_20260907/README.md:96`;
  `dev/experiments/svbmc_pool/pool_20260914/README.md:17`.
- **Finding:** Both name deleted branches.
- **Fix:** Name them as merged and deleted.
- **Ruling:**

**C-35** · fix · agents · high
- **Where:** `dev/plans/svbmc-numpy-prototype.md:7`;
  `dev/results/2026-09-09-svbmc-numpy-prototype.md:15`.
- **Finding:** "Integration remains parked": it was implemented on
  2026-09-11.
- **Fix:** C's text.
- **Ruling:**

**C-36** · fix · contributors · high
- **Where:** `modernization-roadmap.md:1177-1179`.
- **Finding:** "Recorded and not fixed": it was fixed as W6-37
  (`e657eba8`).
- **Fix:** C's text.
- **Ruling:**

**C-37** · decide · agents · low
- **Where:** `modernization-roadmap.md:1119-1126`.
- **Finding:** An open item, "Make the distinction between reference
  generation and candidate checking explicit", that no TODO item owns.
- **Options:** tick it, citing `final-population-benchmark.md`
  "Purpose"; or assign it to a TODO item.
- **Ruling:**

**C-38** · fix · agents · high
- **Where:** `stage4-torch-feasibility.md:405-406`;
  `profile-and-gradient-checks.md:293-313`;
  `modernization-roadmap.md:1141-1142`.
- **Finding:** Pickup statements for work that is done.
- **Fix:** C's texts.
- **Ruling:**

**C-39** · fix · agents · high
- **Where:** the headers of `dev/results/2026-09-08-main-loop-fixes.md`,
  `2026-09-08-boost-analysis.md`, `2026-09-08-eta-bound-comparison.md`,
  `2026-09-08-svbmc-compatibility.md` and
  `2026-09-14-svbmc-honest-elbo-pilot.md`.
- **Finding:** Each names as pending work that has since been done.
- **Fix:** One dated line per header (C's texts). Where C's text for
  main-loop-fixes says the step-out repair "was deferred to gpyreg #44",
  add that gpyreg fixed it on 2026-09-13 (D-3).
- **Ruling:**

**C-40** · fix · contributors · high
- **Where:** `dev/scripts/golden_trace.py:17-18`.
- **Finding:** The docstring names "the gpyreg/cma global random state",
  which no run touches now.
- **Fix:** Keep only the timer singleton.
- **Ruling:**

**C-41** · fix · agents · high
- **Where:** `dev/experiments/pymc_setup_probe/README.md:7-8`.
- **Finding:** "the package remain[s] unchanged": `pyvbmc/pymc/` has been
  added since.
- **Fix:** "The experiment changed neither the historical feasibility
  prototype nor the package."
- **Ruling:**

**C-42** · fix · no one · high
- **Where:** `dev/scripts/data/README.md:53-57`;
  `benchmark_targets.py:58-60`.
- **Finding:** Both say the truths exist once the generator has run. They
  have been tracked since `98254c0`.
- **Fix:** C's text.
- **Ruling:**

**C-43** · fix · no one · high
- **Where:** `dev/scripts/regenerate_baseline.sh:13-14`.
- **Finding:** "14 configs … 8-10 h": the golden suite has 24
  configurations.
- **Fix:** C's text.
- **Ruling:**

**C-44** · fix · no one · high
- **Where:** `dev/experiments/svbmc_pool/README.md:32-34`.
- **Finding:** "once the campaign's stacking runs are in": they are in.
- **Fix:** C's text.
- **Ruling:**

**C-45** · fix · no one · high
- **Where:** `dev/TODO.md:6-8` against its last section.
- **Finding:** "Completed 1.5 work is not listed here", while the last
  section lists some.
- **Fix:** C's text.
- **Ruling:**

## D: the dated devlogs

**D-1** · fix · agents · high
- **Where:** `dev/2026-09-02-modernization-discussion.md` §9 (lines
  529-780), §6:402, §3:241-243.
- **Finding:** The defect list that `AGENTS.md` sends agents to still says
  "not fixed" for about twenty defects that have since been fixed or ruled
  on. It also contradicts itself on the resume self-comparison (733-735
  against 1021-1027).
- **Fix:** A dated note at the head of §9, pointing at
  `plans/latent-bug-fixes.md` and the port-review ledger, and a "Fixed"
  mark on each entry (D's list).
- **Ruling:**

**D-2** · fix · contributors · high (orchestrator re-verified)
- **Where:** `dev/2026-09-08-numerical-campaigns.md:198-200`.
- **Finding:** The key mismatch is stated backwards. Before `90d08d3d`
  the writer used `acqfcn` and the reader `acq_fcn`.
- **Fix:** D's text.
- **Ruling:**

**D-3** · fix · contributors · high
- **Where:** `dev/2026-09-06-pyvbmc-1.5-overview.md:148-149`;
  `2026-09-08-numerical-campaigns.md:244-246`;
  `2026-09-02-modernization-discussion.md:775-780`.
- **Finding:** All three call the gpyreg step-out issue deferred. gpyreg
  fixed it in `f610e11` (#48, 2026-09-13), released in 1.2.1. The same
  holds for `plans/latent-bug-fixes.md:67-69`, `:1576-1590` and
  `plans/modernization-roadmap.md:803-804`.
- **Fix:** D's texts, and a dated line in the two plans.
- **Ruling:**

**D-4** · fix · agents · high
- **Where:** `2026-09-02-modernization-discussion.md:1096-1132`,
  `:1177-1183`, `:1238-1243`.
- **Finding:** The devlog still intends a Torch solver for 1.5. The PI
  kept NumPy/SciPy on 2026-09-09, which other records carry and this
  devlog does not.
- **Fix:** D's "Superseded 2026-09-09" note.
- **Ruling:**

**D-5** · fix · contributors, PI · high
- **Where:** `2026-09-06-pyvbmc-1.5-overview.md:207-208`, `:140-146`,
  `:235-236`.
- **Finding:** The living release overview:
  - lists as remaining the efficiency work, which closed on 2026-09-19;
  - never mentions the port review, or that the 990 reference predates
    its fixes;
  - omits the open items;
  - calls the release notes' account of the random streams and the
    history to be done, which the changelog has written.
- **Fix:** D's texts, and update the "refreshed on" date.
- **Ruling:**

**D-6** · fix · contributors · high
- **Where:** `2026-09-08-numerical-campaigns.md:7`, `:248-252`.
- **Finding:** "the final population benchmark remains pending": it was
  completed and promoted on 2026-09-13.
- **Fix:** D's texts.
- **Ruling:**

**D-7** · fix · contributors · high
- **Where:** `2026-09-12-svbmc-elbo-optimism.md:391-397` (also `:422`,
  `:447-448`).
- **Finding:** The note says the GP behind the returned posterior is
  `vbmc.gp`. It is `vbmc.get_gp(results["best_iter"])`; the campaign plan
  says so, and the note carries no correction.
- **Fix:** D's "Corrected 2026-09-14" note.
- **Ruling:**

**D-8** · fix · users, contributors · high
- **Where:** `2026-09-08-ecosystem-integration.md:5-11`, `:44`, `:55-58`,
  `:132-140`.
- **Finding:** The status line omits the float `elbo` and
  `SVBMC.save`/`load`. The snippet `stacked.elbo["estimated"]` now raises
  `TypeError`.
- **Fix:** Extend the status line, and mark the snippet as the proposal
  it was.
- **Ruling:**

**D-9** · fix · contributors · high
- **Where:** `2026-09-12-svbmc-elbo-optimism.md:369-372`, `:446-449`.
- **Finding:** The Phase 2 outcome, scored on 2026-09-15, is not recorded
  here.
- **Fix:** D's "Outcome (2026-09-15)" note.
- **Ruling:**

**D-10** · fix · contributors · high
- **Where:** `2026-09-13-pymc-integration.md:150-151`, and
  `plans/pymc-target-adapter.md:17`.
- **Finding:** "`dev/TODO.md` carries the decision": it no longer does.
- **Fix:** Point at the adapter plan.
- **Ruling:**

**D-11** · fix · contributors · high
- **Where:** `2026-09-08-noisy-acquisitions.md:27-28`, `:162-198`.
- **Finding:** The devlog names a deleted branch, and says nothing of what
  shipped or of what became of its suggestions.
- **Fix:** D's header note.
- **Ruling:**

**D-12** · fix · contributors · high
- **Where:** `2026-09-12-svbmc-elbo-optimism.md:244-246`.
- **Finding:** The devlog says `reference_870_20260907` has its sidecars
  in `dev/golden/baseline/`. That directory has held the 990 reference
  since 2026-09-13.
- **Fix:** Cite `b2ea8597`.
- **Ruling:**

**D-13:** merged into A-11.

**D-14** · fix · agents · medium
- **Where:** `2026-09-02-modernization-discussion.md:3-4`.
- **Finding:** "No code changed": the devlog itself records code changes
  and dated addenda.
- **Fix:** D's status line.
- **Ruling:**

**D-15** · fix · contributors · high
- **Where:** `2026-09-15-svbmc-headline-shrinkage.md:36-38`.
- **Finding:** "100 independent runs were generated": they were selected
  by the filters out of up to 150 seeds (200 for the ring).
- **Fix:** D's text.
- **Ruling:**

**D-16** · fix · no one · high
- **Where:** `2026-09-15-svbmc-shrinkage-explained.md:193-195`.
- **Finding:** "smaller otherwise": ν is smaller only when the errors are
  positively correlated on average.
- **Fix:** D's text.
- **Ruling:**

## E: errors acknowledged elsewhere (orchestrator's search)

**E-1** · decide (gpyreg) · agents · high
- **Where:** gpyreg's `AGENTS.md:94`.
- **Finding:** The file still maps `gplite_sample.m` to `slice_sample.py`
  and `gplite_fmin.m` to `f_min_fill.py`. Neither MATLAB file has a Python
  counterpart: `slice_sample.py` ports `private/slicesamplebnd.m`, and
  `f_min_fill.py` ports `private/fminfill.m`. PyVBMC's
  `dev/experiments/port_review_20260919/prep_report.md` (§3) and
  `counterpart_map.md` have called both mappings wrong since 2026-09-19.
  The third mapping they named, `gplite_quad.m`, has been corrected.
- **Fix:** In gpyreg, with the work before its final release.
- **Ruling:**

**E-2** · fix · contributors · high
- **Where:** `dev/scripts/torch_vi_step.py`.
- **Finding:** The port-review ledger (its "kept" items) records that the
  module's source hashes are stale; the module does not. It fails closed.
- **Fix:** One sentence in the module docstring: the hashes are those of
  the code of the Stage 4 experiment, the production functions have
  changed since, and the module refuses to run until they are refreshed.
- **Ruling:**

**E-3** · decide · contributors · high
- **Where:** `dev/experiments/port_review_20260919/reviews/P7_internal.md:197`.
- **Finding:** The report claims the default fraction 0.8 gives a tie at
  `N = 5, 15, 25`. `4N/5` is never a half-integer, which the port-review
  plan records in its wave-5 entry. The report's header flags another of
  its corrected claims (W7-3) but not this one.
- **Fix:** Add the flag to the header.
- **Decision:** whether to check every raw reviewer report's header
  against the refutations in the verification ledgers, which would take
  one more targeted reviewer.
- **Ruling:**

## X: noted outside the reviewers' areas

**X-1** (A, B, C) · fix · contributors · high
- **Where:** `pyproject.toml:63-65`, the comment on the `test` extra.
- **Finding:** The comment gives the command as `pytest --pyargs pyvbmc
  --reruns=3`. The tests are not in the wheel, and CI runs
  `python -m pytest --reruns=5 -x` from a checkout.
- **Fix:** Correct the comment.
- **Ruling:**

**X-2:** merged into B-23.

**X-3** (C) · decide · contributors · high
- **Where:** `dev/scripts/analyze_population_run.py`.
- **Finding:** `--out` defaults to the tracked record directory
  `dev/experiments/population_extension_20260911`, so a run without
  arguments writes into a record.
- **Options:** require `--out`; or default it under `scripts/runs/`.
- **Ruling:**
