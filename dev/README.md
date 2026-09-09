# Developer notes

`dev/` is for human review: major findings, proposals, discussions and
consolidated decisions. Use dated names (`YYYY-MM-DD-short-slug.md`) for
these notes. Update or consolidate a related narrative instead of creating
a top-level file for each agent, working session or experiment phase.
Separate notes are appropriate for genuinely separate topics.

Detailed findings and experiment writeups belong directly in `results/`,
with no additional date or campaign subdirectory. A top-level summary should
explain the important evidence and decisions and link to those full reports.
Keep execution plans, checklists and ongoing status in `plans/`, updating
existing files in place. Machine-readable evidence belongs in `experiments/`;
ignored raw runs and logs remain under `scripts/runs/`.

These are maintainer records, not user documentation. `docs/` is gitignored
Sphinx HTML output published to `gh-pages`, so it cannot hold source notes.

## Index

For the release overview, start with
[PyVBMC 1.5: the big picture](2026-09-06-pyvbmc-1.5-overview.md).

- [Modernization discussion](2026-09-02-modernization-discussion.md) —
  Hot paths, gradient inventory, latent bugs, backend direction and staged plan.
- [User-facing agent skill proposal](2026-09-02-user-agent-skill.md) —
  Guidance, FAQ/reference material, helpers and packaging after the API settles.
- [PyVBMC 1.5 overview](2026-09-06-pyvbmc-1.5-overview.md) —
  Human-readable release scope, benefits and validation approach.
- [Numerical campaigns](2026-09-08-numerical-campaigns.md) —
  Consolidated boost and eta evidence and PI decisions, main-loop repairs,
  and cross-platform validation. Links to the full September 4, 7 and 8
  reports in `results/`; the final integrated population remains pending.
- [Ecosystem integration proposal](2026-09-08-ecosystem-integration.md) —
  High-level delivery/API discussion for S-VBMC and related methods, completed
  compatibility evidence, and the subsequent NumPy investigation direction.
  Integration remains parked.

`TODO.md` is the current pickup reminder. The roadmap and plans retain
execution status; the summaries above retain the human discussion and decisions.

## Plans, worklogs and task files

`plans/` holds implementation plans, checklists and execution worklogs.
Keep these current while work is open and retain them afterwards. Detailed
standalone measurement reports belong in `results/`, linked from the relevant
plan and consolidated human summary.

- `plans/modernization-roadmap.md` — living tracker of the staged plan in
  `2026-09-02-modernization-discussion.md` §10: stage status, pickup point.
- [plans/latent-bug-fixes.md](plans/latent-bug-fixes.md) — pickup 9
  implementation plan: verified candidate dispositions, numerical and
  compatibility contracts, PI-selected boost/eta fixes, and regression gates
  against the completed 870-run reference. Final integrated population
  validation remains pending.
  Phase 0 records the reduced 60-run noisy extension; the original 150-run
  preparation remains as historical evidence.
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

- [plans/stage4-torch-feasibility.md](plans/stage4-torch-feasibility.md) -
  Completed bounded PyTorch feasibility prototype: complete variational fits,
  current estimator/optimizer semantics, float64 CPU/GPU evidence,
  final-boost workloads, setup/transfer costs, the recommendation to retain
  NumPy and the pending explicit backend decision.

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
which was made single-threaded. Long campaigns start only on explicit PI
instruction. Light browser/email use is compatible with reference-quality
checks, but timings collected under mixed use are not controlled speed
benchmarks. Short gates (oracles, a module's tests, the replay) can run at
any time as one process.

Default pytest discovery is limited by `pyproject.toml` to the shipped suite
under `pyvbmc/testing`; developer experiment checks in this directory run
only when named explicitly. In particular,
`python -m pytest dev/scripts/test_eta_bound_variants.py -vv` must be run from
historical checkout `c4c692c`, which contains the checks and retains the
`03650a2` numerical source pinned by `scripts/eta_bound_variants.py`. Its
source-digest guard is expected to reject later production changes.

The current golden reference is `reference_870_20260907`: **870 runs across
19 configurations, including 160 noisy runs, with 76 population KS tests**.
Its JSON sidecars and `summary.md` live under `golden/baseline/`,
so `python dev/scripts/golden_trace.py compare dev/golden/baseline <new_dir>`
works from a fresh checkout. Full `.npz` traces stay gitignored
under `scripts/runs/golden/reference_870_20260907/` until release-asset
publication. The historical 810 pairs remain unchanged: the original 280
retain `18a236c`, and 530 record frozen `7314a6a`. The 60 additions were made
from pinned source `623f5cd` with gpyreg `a2f8ddc`, 30 seeds each for
`rosenbrock_D2_noise3` and `student_D8_noise3`; 53 converged and seven noisy
Rosenbrock runs reached their evaluation budget. See the historical
[810-run extension record](golden/extension_20260907/README.md) and the
[870-run integration record](golden/noisy_extension_20260907/README.md) for
commands, provenance, hashes and validation reports. All 870 archives passed
integrity checks, the 76-test even/odd population check had no flags, and the
default five-case replay matched every stored loop and final value plus each
initial design, with zero flags. The historical traces do not store the
returned posterior's transformer, so that field remains explicitly
uncertifiable.

[`golden/README.md`](golden/README.md) is the human-facing description of the
current golden runs: purpose, target coverage, metrics, files and usage. Keep
it about the current snapshot; batch chronology, launch details and release
stage history belong in the separate execution records.

`scripts/regenerate_baseline.sh` is the legacy whole-campaign helper. Its
seed allocation differs from the extended reference and it masks comparison
failures, so it must not be used to republish this reference. Follow the
extension record's commands and fail-fast checks instead. The benchmark follows
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
- `scripts/boost_comparison.py` (parked at `764a177` on `dev-final-boost`) — reads stored pre/final boost scores and
  compares tolerances 0.1/0.2 without optimization. Optional
  `--metrics-tags selected` reconstructs paired accuracy diagnostics for
  rejected/near-threshold candidates and the new noisy configurations.
- `scripts/boost_replay.py` (parked at `764a177` on `dev-final-boost`) — runs Phase 2's seven specified trajectories,
  with explicit `--out`, one worker and one BLAS thread. Each trajectory
  generates one unpenalized boost candidate; both tolerances are evaluated
  on it. Authentic restart state and raw candidates are retained under
  `captures/`. Results and limitations are in the
  [Phase 2 evidence note](results/2026-09-07-final-boost-comparison.md).
- `scripts/boost_reconstruction.py` (parked at `764a177` on `dev-final-boost`) — rebuilds boost inputs from compact
  traces and compares them with authentic restart captures. Optional
  `--population-numerics` checks all stored pre-boost SDs; `--replay TAG...`
  compares authentic/reconstructed boosts with a common RNG. These are
  reconstruction checks, not a penalty-on/off campaign. See the
  [reconstruction findings](results/2026-09-08-boost-reconstruction.md).
- `scripts/boost_penalty_pilot.py` (parked at `764a177` on `dev-final-boost`) — times three fixed seed-0 pairs with
  weight penalty 0.1 versus zero, starting from identical reconstructed
  states and fresh RNGs. Requires `--out`; retains full paired captures and
  separate optimization/diagnostic/save timings. See the
  [pilot results](results/2026-09-08-boost-penalty-pilot.md).
- `scripts/golden_replay.py` — the per-change trajectory gate of Stage 2:
  replays a few golden configurations in-process with the current code
  (about 7 minutes for the default set) and compares each run with its
  stored trace: exact shapes and values of every non-timer NPZ array and
  all semantic final-result fields, the ELBO/live-point agreement horizons,
  the initial design (see below), and final accuracy against the baseline
  population's `Q3 + 3 IQR` envelope. It reports "same loop, changed final"
  separately and applies the accuracy fences to that case. Old traces omit
  the returned posterior's transformer; that state is explicitly reported
  as not certifiable. Sidecar counts must agree with their NPZ arrays.
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
  `scripts/runs/golden/reference_870_20260907/`, the current reference
  population, exists only on the machine that made it), `--sidecars`,
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
