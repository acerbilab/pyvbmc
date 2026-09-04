# Developer notes

Dated developer logs for PyVBMC. One file per working session or decision,
named `YYYY-MM-DD-short-slug.md`, newest at the bottom of the list below.

These are working notes for maintainers, not user documentation. They record
what was investigated, what was decided and why, and what was deliberately
left open, so that a future contributor (or the same person six months later)
can reconstruct the reasoning without re-deriving it.

Why `dev/` and not `docs/dev/`: `docs/` is the Sphinx HTML build output
directory. It is gitignored and its contents are copied verbatim to the
`gh-pages` branch by `.github/workflows/docs.yml`, so nothing placed there
would be committed, and anything that was would be published to the docs site.

## Index

- `2026-09-02-modernization-discussion.md` — Assessment of porting the
  numerical core from NumPy to a tensor backend. Hot-path analysis, gradient
  inventory, gpyreg audit, backend decision (PyTorch), staged plan, repo
  process decisions, list of latent bugs found along the way.
- `2026-09-02-user-agent-skill.md` — Deferred idea: ship an Agent Skills
  folder so users' coding agents set up and troubleshoot PyVBMC correctly.
  What BayesFlow did, the spec, and a design that lives in-repo and ships in
  the wheel. Build after the 1.5 API settles.
- `2026-09-04-final-boost-failure.md` — `final_boost` can turn a converged
  posterior into an unusable one when the GP mean function has gone flat
  (golden population, `student_D4` seed 19; 4 of 6 boost reruns fail;
  inherited from MATLAB). Mechanism, evidence, three candidate guards,
  reproduction; decision deferred.

## Plans, worklogs and task files

Devlogs (above) are the record of discussions and decisions that involved a
person, most often the PI. Everything an agent produces on its own along the
way lives in `plans/`: plans, checklists and status trackers, worklogs of a
work session, measurement reports, task files. They are updated in place
while the work is open and kept afterwards as the record of what was done;
status and next steps never go into the devlogs.

- `plans/modernization-roadmap.md` — living tracker of the staged plan in
  `2026-09-02-modernization-discussion.md` §10: stage status, pickup point.
- `plans/profile-and-gradient-checks.md` — dev environment, baseline test
  run, first measured profile (D=5, D=10) and the first Stage 0
  finite-difference gradient checks, which found the reshape-order bug in
  `_vp_bound_loss`.
- `plans/stage1-rng-generator.md` — Stage 1 worklog: `VBMC(seed=)`,
  `vbmc.rng`, the gpyreg/cma global-state seam, the random-state save
  format, what the tests had to change, review findings, follow-ups.
- `plans/benchmark-suite-and-golden-traces.md` — plan and worklog for the
  benchmark target suite (`scripts/benchmark_targets.py`), the profile
  campaign on it that fixed the Stage 2 order, and the first golden-trace
  population; target definitions and ground truth, the measured profile,
  the harness design, results and follow-ups.
- `plans/fixture-generator-and-oracles.md` — plan and worklog for the
  Stage 0 fixture generator and stage-level oracles
  (`pyvbmc/testing/oracles/`): snapshot format, regime coverage, the
  oracle list and tolerances, decisions, tracker. The per-commit gate for
  Stage 2.
- `plans/stage2-batched-acquisition.md` — plan and worklog for Stage 2
  item 3: the replay gate (`scripts/golden_replay.py`), the batched CMA-ES
  acquisition evaluation, the broadcast `vp.pdf`, the targeted re-baseline
  of the step oracle, the measured speedup.
- `plans/stage2-gp-log-joint-einsum.md` — plan and worklog for Stage 2
  items 1 and 2: `_gp_log_joint` vectorized over hyperparameter samples
  and mixture components (one `(Ns, K, D, N)` tensor, `einsum`
  contractions), the log-joint variance from multi-RHS solves, the latent
  defects of the function fixed on the way, the bit-checks against the
  loop, the sensitivity experiment that explains why the replay parts at
  iteration 0 on cigar, the initial-design certificate added to the
  replay (`X_init` in the traces), the `--probe` speed probe of the
  profile suite, the measured speedup.

Naming: `plans/` files are named by slug only, never by date (the date is in
the file header), so that they cannot be mistaken for copies of the dated
devlogs.

## Scripts

`scripts/` holds developer tooling that is not part of the package or the
test suite. Output directories under it (e.g. `scripts/runs/`) are gitignored;
results that matter get summarized in the relevant `plans/` worklog (and
decisions taken with a person in a dated devlog), not committed raw. Run the
scripts from the repo root with the project venv; they import each other by
plain module name, so run them as `python dev/scripts/<name>.py`. They need
`psutil` (not a package dependency; `pip install psutil`). Keep to **one
heavy process at a time** on a laptop (`golden_trace.py run --workers 1`,
the default; eight concurrent VBMC processes hard-crashed the machine on
2026-09-02) and export `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`
before profiling if wall times are to be compared with the golden baseline,
which was made single-threaded. The golden reference population's sidecars
(JSON only, under 1 MB) live under `golden/baseline/` together with its
`summary.md`, so `python dev/scripts/golden_trace.py compare
dev/golden/baseline <new_dir>` works from a fresh checkout; the full traces
(`.npz`) stay gitignored under `scripts/runs/golden/`.
`scripts/regenerate_baseline.sh` regenerates everything (target checks,
profile campaign plain and cProfile, golden sweep, summary, null check,
publishing the sidecars) as one sequential process, about 10–12 hours;
see `golden/README.md` for the population's status. The benchmark follows
the VBMC papers' procedure: each run's start point is drawn uniformly inside
the plausible box from a stream spawned off the run seed, and the plausible
box is the papers' prior box (family mean ± 3 marginal SD); see the audit
in `plans/benchmark-suite-and-golden-traces.md` for every deviation and its
reason.

- `scripts/benchmark_targets.py` — the benchmark target suite: nine targets
  with ground truth (normal, corr, halfnormal, rosenbrock, banana, cigar,
  lumpy, student, logreg), a generic noise wrapper, the `smoke` / `profile`
  / `golden` suites, shared posterior-moment and metric helpers, and
  `--list` / `--check` / `--smoke` self-tests. Every other script takes its
  targets from here.
- `scripts/profile_run.py` — run VBMC on one target or suite config under a
  fixed seed and report per-stage timers, truth-based metrics and, with
  `--cprofile`, a cProfile attribution of the hot paths.
- `scripts/profile_suite.py` — run `profile_run.py` over a whole suite
  (plain and/or cProfile, resumable) and aggregate the summaries into one
  markdown table. `--probe CONFIG` runs a short reference config plain
  before and after the campaign and prints the ratio of the two walls: a
  speed probe, because a laptop under sustained load can slow down by
  1.5× partway through a campaign (2026-09-04) and the untouched stages
  are otherwise the only tell.
- `scripts/golden_trace.py` — the golden-trace regression harness: `run` a
  suite over many seeds (one process by default), storing one compact
  `.npz` trace and a JSON sidecar per run; `summary` a population; `compare`
  two populations with KS tests under a Holm family correction (`--split`
  for a null check). Populations live under `scripts/runs/golden/`.
- `scripts/golden_replay.py` — the per-change trajectory gate of Stage 2:
  replays a few golden configurations in-process with the current code
  (about 7 minutes for the default set) and compares each run with its
  stored trace: the first iteration at which the ELBO path parts (iteration
  0 identical certifies the initial design), the live points identical,
  and the finals against the baseline population's `Q3 + 3 IQR` envelope.
  An arithmetic-preserving change is expected to part once a CMA-ES
  ranking flips (a change to the ELBO arithmetic parts at iteration 0);
  a parted run's finals must stay inside the envelope (an identical run
  is exempt: its own seed may be the outlier). The initial design is
  certified from the traces: exactly where both store it (`X_init`,
  written by `golden_trace.py` since commit `9d92c7f`), against the 2026-09-03
  baseline by finding a design point of the new run among the reference's
  live rows, and "not certifiable" without a flag where warm-up trimming
  removed the whole design (cigar). Needs the baseline `.npz` traces for
  the horizons (finals only without them); `--report-only` re-renders a
  finished run. Flags: `--configs`, `--seeds`
  (default seed 0 only), `--baseline` (the traces directory; the default
  `scripts/runs/golden/baseline_20260903/` exists only on the machine that
  made the baseline), `--sidecars`, `--out`, `--threads` (1, as the
  baseline). Exit code 1 if anything is flagged or nothing was compared.
- `scripts/regenerate_baseline.sh` — the whole benchmark regeneration as
  one sequential process (see above).
- `scripts/make_oracle_fixtures.py` — generates the stage-level oracle
  fixtures under `pyvbmc/testing/oracles/fixtures/`: short seeded runs on
  the benchmark targets with regime-forcing options, the state at chosen
  iterations saved as plain arrays, and the reference outputs of every
  numerical stage computed from the rebuilt state (`--list`, `--only`,
  `--check`; about six minutes, one process). Regenerating replaces the
  references: only for a deliberate new baseline. `--rebaseline ORACLE
  --reason "..."` replaces one oracle's references from the stored state
  without rerunning the source runs (every other reference stays
  bit-identical, asserted): for the CMA-ES step oracle after a change
  that the `acq_*` oracles have cleared. It rewrites the whole `.npz`
  (git shows a full binary change), appends an audit entry under
  `meta["rebaselined"]` in the `.json` (oracle, date, git SHA, reason,
  per-output max change: the thing to look for when reviewing such a
  diff), refuses the step oracle off the generating platform, and runs
  one process at a time.
