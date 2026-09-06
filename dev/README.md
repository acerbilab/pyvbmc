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

For the release overview, start with
[PyVBMC 1.5: the big picture](2026-09-06-pyvbmc-1.5-overview.md).

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

- [2026-09-06-pyvbmc-1.5-overview.md](2026-09-06-pyvbmc-1.5-overview.md) —
  Human-readable release outline: benefits, scope, validation, and the path
  to 1.5. A dated overview of the agreed direction, not an execution checklist.

`TODO.md` is a scratch reminder of the ongoing work and the current pickup
point, rewritten at each handoff. It is not a record: the roadmap and the
plans are.

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
- `plans/stage2-gpyreg-predict-and-sampler.md` — plan and worklog for
  Stage 2 item 8: the gpyreg PR (acerbilab/gpyreg#43; `predict` and the
  slice sampler's log-posterior evaluation without scipy's wrapper layers,
  the Cholesky factor reused across mean-hyperparameter moves, generator
  support), identity-preserving and gated by exactness (the `gp_nlZ` and
  `gp_fit` oracles, a dump of the pre-change oracle outputs, the replay
  reporting `identical`), the PyVBMC seam removal (every draw of a run
  through `vbmc.rng`), the measured speedup, the review findings.
- `plans/stage2-gp-log-joint-einsum.md` — plan and worklog for Stage 2
  items 1 and 2: `_gp_log_joint` vectorized over hyperparameter samples
  and mixture components (one `(Ns, K, D, N)` tensor, `einsum`
  contractions), the log-joint variance from multi-RHS solves, the latent
  defects of the function fixed on the way, the bit-checks against the
  loop, the sensitivity experiment that explains why the replay parts at
  iteration 0 on cigar, the initial-design certificate added to the
  replay (`X_init` in the traces), the `--probe` speed probe of the
  profile suite, the measured speedup.
- `plans/stage2-entmc.md` — plan and worklog for Stage 2 item 5:
  `entmc_vbmc` vectorized over components and samples (one draw for every
  component's antithetic samples, the mixture density and the
  reparameterization gradients as a broadcast over a `(components,
  samples, D, K)` tensor in 2^16-element blocks), the per-call profile
  that put the time in the density loop rather than the draws, the
  bit-checks against the loop, the tensor layouts and the GEMM expansion
  that were measured and not taken, the speedup.
- `plans/stage2-memory.md` — plan and worklog for Stage 2 items 6 and 7
  (memory): the sieve candidates of `_vb_init` built as shells that share
  the run's generator and transformer instead of a `copy.deepcopy` each
  (done; bit-identical candidates, about 0.1 % of a run), what
  `iteration_history` retains (the GPs' Cholesky factors, Σ Ns N² doubles,
  323 MB on the exhaust run) and the history re-copying its whole past on
  every record, the readers of the stored GPs, the decisions taken with the
  PI (what can be rebuilt from the record is never stored; what cannot is
  dropped by default and kept under `record_full_history_details`), and
  the four steps that followed the same night (the resume test made real,
  the history growing without re-copying its past, lean GP records
  restored by the public `VBMC.get_gp`, the importance samples out of the
  recorded `optim_state`), the three code steps each replayed and
  measured.
- `plans/stage0-dtype-canary.md` — plan and worklog for the last Stage 0
  item, the dtype canary (tests only): what the value-comparing oracles
  can and cannot see of a float32 regression, the raw-output and
  rebuilt-state checks inside `test_oracles.py`, the walk of a live run
  in `test_vbmc_seed.py` and the manifest of load-bearing arrays
  (`pyvbmc/testing/_dtype.py`), the float32/float16 constructor inputs
  found to keep their dtype (pinned as a strict `xfail`), and the
  `active_sample_step` oracle's need for single-threaded BLAS on the
  machine that generated the fixtures.

- `plans/stage3-pipeline-features.md` - approved Stage 3 plan and live
  worklog: connect torch/JAX models through opt-in initial-design batching,
  use fitted posteriors through torch and current ArviZ DataTree exports,
  optional dependencies, documentation, CI wiring and verification gates.
  Implementation is complete on `dev-next-stage3` at code `4ee612d`
  (records/docs `285cd74`); merged into `dev-next` at `4bff1a5`, with
  branch/full-matrix/integrated CI and all local integration checks passed.
  Reference snapshot: `reference/stage3-20260906`.

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
which was made single-threaded. The long runs (a profile campaign, about
an hour; a golden population, about 6.5 h) measure absolute time, so
desktop use distorts them (item 8 reran three configs for that reason):
they start only when the PI has said the laptop is free, never on a
timer or a guess; short gates (oracles, a module's tests, the replay)
can run at any time as one process. The golden reference population's sidecars
(JSON only, under 1 MB) live under `golden/baseline/` together with its
`summary.md`, so `python dev/scripts/golden_trace.py compare
dev/golden/baseline <new_dir>` works from a fresh checkout; the full traces
(`.npz`) stay gitignored under `scripts/runs/golden/`. For the current
two-night extension, first merge Stage 3 into `dev-next`, pass the exact
oracles, identical replay and full local suite on the integrated tree, then
freeze that code for both nights. Keep `vectorized_target=False` in every
reference configuration. Preserve the existing `18a236c` sidecars and
traces; new sidecars record the actual frozen integrated SHA. Neither
trajectory-neutral nor trajectory-moving latent fixes land until both
nights have finished. The nights may be one explicitly authorized chain
(about 7 h and 5 h, plus check overhead); its second batch starts only after
the first and its checks succeed.
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
- `scripts/profile_compare.py BASE NEW` — compare two campaigns config by
  config: wall and per-stage seconds with ratios, whether each trajectory
  is the same, cProfile buckets and per-call times, and a per-config
  machine-speed control (the ratio of a stage the change does not touch,
  `--control variational_fit` by default). A control far from 1.0 marks a
  config the machine slowed, not the code; rerun those alone by deleting
  their run directories and repeating `profile_suite.py` with the same
  `--out` (item 8, 2026-09-05: three configs slowed by desktop use, clean
  on the fourth attempt).
- `scripts/golden_trace.py` — the golden-trace regression harness: `run` a
  suite over many seeds (one process by default), storing one compact
  `.npz` trace and a JSON sidecar per run; `summary` a population; `compare`
  two populations with KS tests under a Holm family correction (`--split`
  for a null check). Populations live under `scripts/runs/golden/`.
- `scripts/golden_replay.py` — the per-change trajectory gate of Stage 2:
  replays a few golden configurations in-process with the current code
  (about 7 minutes for the default set) and compares each run with its
  stored trace: the first iteration at which the ELBO path parts, the live
  points identical, the initial design (see below), and the finals against
  the baseline population's `Q3 + 3 IQR` envelope.
  An arithmetic-preserving change is expected to part once a CMA-ES
  ranking flips (a change to the ELBO arithmetic parts at iteration 0);
  a parted run's finals must stay inside the envelope (an identical run
  is exempt: its own seed may be the outlier). The initial design is
  certified from the traces: exactly where both store it (`X_init`,
  written by `golden_trace.py` since commit `9d92c7f`), against the 2026-09-03
  baseline by finding a generator-drawn design point of the new run among
  the reference's live rows (the start point `x0` comes from the run seed
  and is identical by construction, so it does not count), and "not
  certifiable" without a flag where warm-up trimming removed the whole
  design (cigar). Needs the baseline `.npz` traces for
  the horizons (finals only without them); `--report-only` re-renders a
  finished run. Flags: `--configs`, `--seeds`
  (default seed 0 only), `--baseline` (the traces directory; the default
  `scripts/runs/golden/item7_20260906/`, the reference population since
  2026-09-06, exists only on the machine that made it), `--sidecars`,
  `--out`, `--threads` (1, as the baseline). Exit code 1 if anything is
  flagged or nothing was compared.
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
  diff), refuses the platform-bound oracles (`active_sample_step`,
  `gp_fit`) off the generating platform, and runs one process at a time.
  Since 2026-09-05 (item 8): `--expect-moving A,B` names the other
  oracles a change moves so the post-write check does not fail on them
  (a random-stream change moves every oracle that draws); `--add-oracle
  NAME --reason "..."` adds a newly registered oracle's references to the
  existing fixtures from their stored state (audit entry under
  `meta["oracles_added"]`); `--check --exact` compares the working tree
  with the committed references bit for bit, the gate for an
  identity-preserving refactor since the references were re-baselined to
  the current numerics at the end of Stage 2 (2026-09-06); `--dump-outputs
  DIR` writes the current code's outputs of every oracle on every snapshot
  and `--check --exact --against DIR` compares with such a dump, for a
  change made while the references are known to lag.
