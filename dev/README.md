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
- [User-facing agent skill](2026-09-02-user-agent-skill.md) —
  A thin wrapper pointing coding agents to the maintained documentation;
  scope and maintenance of the first version.
- [PyVBMC 1.5 overview](2026-09-06-pyvbmc-1.5-overview.md) —
  Human-readable release scope, benefits and validation approach. Includes the
  [compiled-Torch follow-up](results/2026-09-09-torch-compile-follow-up.md),
  with cold/warm complete-fit timings and numerical limitations.
- [Numerical campaigns](2026-09-08-numerical-campaigns.md) —
  Consolidated boost and eta evidence and PI decisions, main-loop repairs,
  and cross-platform validation. Links to the full September 4, 7 and 8
  reports in `results/`; the completed integrated population assessment is
  in the [promotion record](golden/promotion_20260913/README.md).
- [Ecosystem integration proposal](2026-09-08-ecosystem-integration.md) —
  Human-review proposal for integrating the existing S-VBMC implementation:
  preserved workflow, optional Torch, code placement and migration, with
  links to completed compatibility and backend evidence. Approved and
  implemented 2026-09-11; record in
  [plans/svbmc-integration.md](plans/svbmc-integration.md).
- [Noisy-target acquisitions](2026-09-08-noisy-acquisitions.md) —
  Where noisy (VIQR) runs spend their time, what was tried on the
  acquisition, where the acquisition search loses evaluations, and the
  interim conclusions and suggestions; the evidence is in the
  [acquisition experiments](results/2026-09-08-noisy-acquisition-experiments.md)
  and the [search analysis](results/2026-09-09-acquisition-search-analysis.md).
- [GP robustness on noisy unbounded targets](results/2026-09-16-gp-tail-acquisition.md) —
  Completed investigation of tail acquisitions and GP conditioning. The
  [box-sampler study](results/2026-09-16-gp-box-sampler.md) found and fixed
  the normal-versus-uniform porting bug. Exact batch replays and 50-digit
  checks explain the remaining VIQR/refit behavior; algorithmic remedies
  are deferred beyond 1.5. Both reports link to the preserved evidence.
- [S-VBMC ELBO optimism](2026-09-12-svbmc-elbo-optimism.md) —
  What the stacking implementation reports today, why the ELBO optimism
  on noisy targets is a cross-run selection effect, the decision to leave
  the objective alone, and the two agreed phases: corrections, reporting
  and tips from the stored posteriors alone, then an exploration of a
  cross-run honest estimate using the runs' GPs. Phase 1 is implemented,
  verified and merged into `dev-next`; the execution record is
  [plans/svbmc-elbo-reporting.md](plans/svbmc-elbo-reporting.md).

- [Scoped PyMC integration](2026-09-13-pymc-integration.md) —
  PR #73 compared with 1.5, the proposed split between model/export adapters
  and automatic fitting, and the feasibility questions for release scope;
  the adapter and structured export are implemented. The
  [feasibility report](results/2026-09-14-pymc-feasibility.md) records the
  prototype; the [implementation plan](plans/pymc-target-adapter.md)
  records the final contract and verification.
- [PyMC setup budget and evaluation reuse](results/2026-09-16-pymc-setup-probe.md) —
  Preapproval measurements of a capped gradient search and exact Hessian,
  stopping rules, the prior-location guard, and three ways to use setup
  observations at equal total evaluation budgets. The resulting adapter
  and generic precomputed-evaluation interface are implemented; the plan
  records their verification and integration status.
- [S-VBMC headline shrinkage](2026-09-15-svbmc-headline-shrinkage.md) —
  Two questions about the S-VBMC port, answered from the run-pool
  benchmark: it matches the standalone package (same weights and
  posterior quality, 1.9 to 4 times faster), and the candidate for the
  number it should report on a noisy stack is a two-level
  empirical-Bayes shrinkage of the components' expected log joints by
  the estimation covariance saved with each posterior (`J_sjk`; the
  class's inputs do not change), in place of the component-median
  cap, which over-corrects on a heavy-tailed target; the candidates
  side by side, the decision and what
  implementing it entails. The evidence is the
  [stage D report](results/2026-09-15-svbmc-pool-comparison.md).
- [S-VBMC shrinkage explained](2026-09-15-svbmc-shrinkage-explained.md) —
  Tutorial companion to the headline note, for a reader who knows what
  a GP and an ELBO are: what the stacked ELBO is built from, why it is
  optimistic on a noisy target, the empirical-Bayes model and its
  moments, why the correlation of a run's estimation errors changes the
  noise term and the shrinkage, the second level over runs, a worked
  example on three stacks from the run pool, what the method cannot
  fix, and the computation step by step.

`TODO.md` contains only current actions, constraints and links. Do not
accumulate completed handoffs there; the roadmap and plans retain execution
status, and the summaries above retain the human discussion and decisions.

## Plans, worklogs and task files

`plans/` holds implementation plans, checklists and execution worklogs.
Keep these current while work is open and retain them afterwards. Detailed
standalone measurement reports belong in `results/`, linked from the relevant
plan and consolidated human summary.

- `plans/modernization-roadmap.md` — living tracker of the staged plan in
  `2026-09-02-modernization-discussion.md` §10: stage status, pickup point.
- [plans/teaching-material.md](plans/teaching-material.md) — Example 9's
  psychometric-model design, Torch/JAX targets, posterior-export workflow,
  consistency review and notebook/script/rendered-documentation verification.
- [plans/noisy-acquisition-efficiency.md](plans/noisy-acquisition-efficiency.md)
  — completed guarded-sum and kernel-reuse optimizations, and the completed
  integration/search experiment: fixed-state comparisons, independent
  judging, conditional adaptation and a ten-seed paired inference
  comparison, with the E6 assessment and proposed follow-ups.
  Its [experiment report](results/2026-09-16-noisy-acquisition-integration-search.md)
  records sources, coverage, verification and measured results.
- [plans/port-correctness-review.md](plans/port-correctness-review.md) —
  independent correctness review of PyVBMC and gpyreg before the release
  freeze: internal-correctness and MATLAB-comparison tracks per slice,
  third readers on the gradient and integral formulas, the reviewer
  brief and verification into a findings ledger, against the latest
  MATLAB `master`; MATLAB itself was to be run only if a finding's
  disposition depended on it, and none was. Complete (2026-09-19 to 09-23):
  the [closing ledger](results/2026-09-23-port-correctness-review.md) holds
  every finding with its disposition, and its fix commit where there is one
  (those the close left without a ruling were ruled on the same day), and
  the porting log `pyvbmc/vbmc/README.md` the catalogue of deliberate
  differences from MATLAB.
- [plans/machine-local-calibration.md](plans/machine-local-calibration.md) —
  implemented package integration for explicit PDF/entropy calibration with
  progress, a machine/environment cache and fixed per-run settings. Local
  numerical, lifecycle, documentation and distribution validation is complete;
  delivery status and remaining CI gates are tracked in the plan and `TODO.md`.
  The [first results](results/2026-09-09-machine-local-calibration.md) retain
  current defaults in both balanced sweeps and establish discovery costs.
- [plans/latent-bug-fixes.md](plans/latent-bug-fixes.md) — pickup 9
  implementation plan: verified candidate dispositions, numerical and
  compatibility contracts, PI-selected boost/eta fixes, and regression gates
  against the historical 870-run reference. Integrated population validation
  is complete; the [promotion record](golden/promotion_20260913/README.md)
  records acceptance and the active 990-run reference.
  Phase 0 records the reduced 60-run noisy extension; the original 150-run
  preparation remains as historical evidence.
- [plans/svbmc-integration.md](plans/svbmc-integration.md) — S-VBMC
  integration (complete): the settled decisions (Torch retained,
  independent sampling with a balanced option, `seed`, float64, snapshot
  fixtures plus a generated D=1, bounded and warped set), layout, the
  parity gate against upstream, the test suite, verification, the
  `SVBMC.save` and `SVBMC.load` methods (2026-09-24) and the remaining
  follow-up (forwarding release after 1.5); the speedups have their own
  plan below.
- [plans/svbmc-speedups.md](plans/svbmc-speedups.md) — S-VBMC
  preparation and entropy speedups: per-run transforms and Jacobians,
  broadcast component densities, vectorized per-component reduction;
  the numerical contract (references unchanged, equivalence tests at
  1e-12), the paired before/after measurement and its evidence in
  [results/2026-09-13-svbmc-speedups.md](results/2026-09-13-svbmc-speedups.md).
- [plans/svbmc-benchmark-campaign.md](plans/svbmc-benchmark-campaign.md) —
  the S-VBMC run-pool campaign: pools of independent VBMC runs on noisy
  targets saved with their posteriors and GPs for the cross-run estimator
  investigation, the matched comparison of the integrated S-VBMC against
  the original standalone package, the harness scripts, allocation, gates
  and execution worklog. Run on 2026-09-14/15, extended to `M = 32` for
  the integrated arm on 2026-09-15/16, and assessed in
  [results/2026-09-15-svbmc-pool-comparison.md](results/2026-09-15-svbmc-pool-comparison.md).
- [plans/svbmc-shrinkage-estimator.md](plans/svbmc-shrinkage-estimator.md) —
  integration of full-covariance two-level shrinkage as an additional
  S-VBMC ELBO estimate, using existing posterior statistics; numerical
  boundary cases, campaign recording and parity checks. Headline selection
  remains a decision after the final release campaign.
- [plans/svbmc-pool-handoff.md](plans/svbmc-pool-handoff.md) — the brief
  under which the S-VBMC run pools were generated on the cluster
  (2026-09-14): what the pools are for, the steps (`prepare`, `cases`, a
  Slurm array of `worker` calls, `select`, `summarize`), what to hand
  back and the branch-and-PR flow; carried out, and retained as written.
- [plans/slurm-benchmark-support.md](plans/slurm-benchmark-support.md) —
  the Slurm workflow of the release gate on the Turso cluster: what runs
  where (the reference populations of the `production` suite with a
  before arm, the S-VBMC pools and stacking on the cluster; exact replay
  fingerprints on the developer's machine), the campaign contract every
  harness meets, the generic driver grown from `scripts/hpc/`, the
  harness changes, what it assumes of the cluster, costs and phases. For
  the PI's review (2026-09-25).
- [plans/benchmark-realistic-targets.md](plans/benchmark-realistic-targets.md) —
  the real-data benchmark targets from benchflow (Bayesian timing,
  multisensory causal inference on two subjects): the decisions, the
  target definitions and bounds, the ground-truth generation method
  (slice sampling, Geyer's estimator with a mixture proposal), the
  reference campaigns, the work breakdown and the smoke-run evidence.
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

- [plans/pymc-target-adapter.md](plans/pymc-target-adapter.md) — the
  PyMC target adapter for 1.5: the design settled from the feasibility
  check (`PyMCTarget`, coordinates and bounds, the Laplace box and its
  fallbacks, the structured `to_arviz` export, the `pymc >= 6.3` floor
  and the capability guards, save and load), what it rests on, the
  phased work with its executors, decisions and the execution record.
  Approved and implemented 2026-09-16; includes the executed Example 8,
  independent reviews, numerical regression evidence and CI records.

- [plans/stage4-torch-feasibility.md](plans/stage4-torch-feasibility.md) -
  Completed bounded PyTorch feasibility prototype: complete variational fits,
  current estimator/optimizer semantics, float64 CPU/GPU evidence,
  final-boost workloads and setup/transfer costs. The PI accepted the
  recommendation to retain the NumPy/SciPy solver for 1.5; no full Torch
  solver port is planned for this release.

- [plans/svbmc-numpy-prototype.md](plans/svbmc-numpy-prototype.md) -
  Completed optimized NumPy stacking prototype, actual upstream comparison,
  shared-preparation Torch control and matched float64 optimization checks.
  Detailed findings are in
  [results/2026-09-09-svbmc-numpy-prototype.md](results/2026-09-09-svbmc-numpy-prototype.md).
  The [integration proposal](2026-09-08-ecosystem-integration.md) uses this
  evidence to recommend retaining Torch and the existing S-VBMC workflow.

Naming: `plans/` files are named by slug only, never by date (the date is in
the file header), so that they cannot be mistaken for copies of the dated
devlogs.

## Scripts

`scripts/` holds developer tooling that is not part of the package or the
test suite. Output directories under it (e.g. `scripts/runs/`) are gitignored;
results that matter get summarized in the relevant `plans/` worklog (and
decisions taken with a person in a dated devlog), not committed raw. A
machine that holds raw artifacts under `scripts/runs/` lists them, with the
commands that recreate its frozen worktrees, in the gitignored
`scripts/runs/LOCAL.md`; if that file is absent, nothing is stored locally.
Tracked documents never say "this machine": they point at that file. Run the
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

The golden suite's noisy configurations pin the 2020 paper's budget of
50 (D + 2) evaluations, which suppresses the package's own 75 (D + 2)
default for a specified-noise target; everything else is a package
default. Since 2026-09-18 release checks and new experiments use the
`production` suite instead: the same configurations with the noisy entries
freed of that pin under `production`-tagged labels. Its noiseless entries
are the golden ones. The golden reference below is the record of the
golden suite; the reference that replaces it after the port review is to
run the whole `production` suite on the cluster, with exact replay from
one seed per configuration generated on the developer's machine
([plans/slurm-benchmark-support.md](plans/slurm-benchmark-support.md)).

The current golden reference is `reference_990_20260913`: **990 runs across
23 configurations, including 250 noisy runs, with 92 population KS tests**.
Its JSON sidecars and `summary.md` live under `golden/baseline/`, so
`python dev/scripts/golden_trace.py compare dev/golden/baseline <new_dir>`
works from a fresh checkout. Full `.npz` traces stay gitignored under
`scripts/runs/golden/reference_990_20260913/` until release-asset publication.
The 870 synthetic-target pairs use frozen treatment `68a43db`; the 120
real-data pairs retain `fc50ee1` and are byte-identical to their previous
reference. Both use gpyreg `a2f8ddc`. The previous 990- and 870-run references
remain preserved for historical comparisons.

The [promotion record](golden/promotion_20260913/README.md) contains the
assessment, provenance, hashes and verification. All 990 archives passed
integrity checks and the 92-test even/odd check had no flags. The five
default cases replayed exactly under the code at promotion (2026-09-13) in
every non-timer NPZ loop/final array, semantic final-result field and
initial design; the port review's fixes have moved default trajectories
since (`TODO.md`, "The golden references after the port review"). The four real-data seed-0 replays
retain their earlier exact certification. The returned posterior's
transformer is absent from the traces and remains uncertifiable.

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

- `scripts/benchmark_targets.py` — the benchmark target suite: eleven
  synthetic targets with ground truth (normal, corr, halfnormal,
  rosenbrock, banana, cigar, lumpy, student, logreg, and gmm and ring,
  the S-VBMC paper's two two-dimensional targets, ported from
  `src/svbmc/targets.py` of the standalone `svbmc` package at `13a78f6`;
  the ring is defined without the `log r` term of that source, which the
  paper's own fitted posteriors rule out, see
  [plans/svbmc-benchmark-campaign.md](plans/svbmc-benchmark-campaign.md)),
  three real-data targets (the Bayesian timing model and the multisensory
  causal-inference model on two subjects, from the 2020 noisy-VBMC paper,
  with spline-trapezoidal priors; see
  [plans/benchmark-realistic-targets.md](plans/benchmark-realistic-targets.md)),
  a generic noise wrapper, whose target returns the noise SD beside the
  value (uncertainty level 2) or, with `provide_noise=False`, the value
  alone (level 1; the label then carries `_level1`), the `smoke` / `profile`
  / `golden` / `svbmc_pool` / `production` / `oracle` suites
  (`svbmc_pool` the pool conditions of the S-VBMC campaign, at PyVBMC's
  default budget and tagged `svbmc`; `production` the golden suite with
  its noisy entries at the package's defaults; `oracle` the configs that
  only the oracle fixtures read), shared
  posterior-moment and metric helpers, and `--list` / `--check` / `--smoke`
  self-tests. Every other script takes its targets from here.
- `scripts/export_benchflow_data.py` — exports the real-data targets' data
  from a benchflow checkout into the plain `.npz` archives under
  `scripts/data/` (layout and provenance in `scripts/data/README.md`).
- `scripts/make_benchmark_truths.py` — regenerates the real-data targets'
  ground truths under `scripts/data/truths/`: slice sampling in the
  transformed space (four chains, whitened by the Laplace covariance at the
  MAP), a variational Gaussian mixture proposal with a broad defensive
  component fitted to the chains, Geyer's estimator of the log normalizing
  constant against it, and the importance-weighted proposal draws as the
  stored population (their effective sample size is far above the
  chains'); importance-sampling and Laplace cross-checks, ESS and R-hat;
  `--check` verifies stored files. Resumable per chain; run overnight
  (timing costs about 40 to 50 ms per evaluation).
- `scripts/profile_run.py` — run VBMC on one target or suite config under a
  fixed seed and report per-stage timers, truth-based metrics and, with
  `--cprofile`, a cProfile attribution of the hot paths. It runs this
  checkout's package, or the tree that `PYVBMC_SOURCE` names.
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
  for a null check). Populations live under `scripts/runs/golden/`. It runs
  this checkout's package, or the tree that `PYVBMC_SOURCE` names, which
  goes ahead of the checkout on `sys.path`; each sidecar's `meta` records
  the checkout's commit, the path and commit of the imported PyVBMC and
  gpyreg, and the versions their installed distributions name, labelled as
  such.
- `scripts/population_run.py` — the population harness of the release gate
  (`plans/slurm-benchmark-support.md`), meeting the campaign contract of
  `scripts/campaign_contract.py`: `prepare` fixes the allocation (a suite,
  its labels, one seed range), the options, the identity and the
  confirmatory family of the comparison of two arms; `cases` and its
  subsets; `worker` runs one case in a fresh process, wrapping the
  production final boost to retain the pre-boost posterior, the raw
  candidate and the decision without changing the calculation or the
  random stream, and writes, beside the trace and the sidecar, the arrays
  that rebuild the returned posterior exactly; `verify` reconciles the
  contract's states and re-checks every case in the campaign's own trees;
  `summarize` and `rescore` are its finishing steps, the second
  recomputing the metrics of both arms with the release code; its tracked
  copies (`TRACKED_COPIES`) are the summary, the rescored metrics and every
  verified case's record, sidecar and boost report; `run` is the
  same campaign one case after another on a workstation. `PYVBMC_SOURCE`
  names the package tree of an arm of other code and `PYVBMC_GPYREG_SOURCE`
  the gpyreg checkout; the harness, the targets module and its data are
  this checkout's in every arm. `validate_case` checks the records of the
  campaigns of September 2026, which ran before array mode.
  `test_population_run.py` checks the capture, the contract's states,
  `verify`, `rescore` and the rebuilt posterior without inference.
- `scripts/analyze_population_run.py` — assesses finished campaigns without
  inference. By default, campaigns of one treatment against the golden
  reference: it revalidates every case and the reference sidecars, pools a
  first-stage campaign with its extensions, recomputes the KS screen and a
  within-configuration paired family (exact signed-rank tests by dynamic
  programming over midranks, exact at any number of pairs, and exact
  McNemar tests of usability), checks every boost decision against the
  guard, and reports each extension on its own with the confirmatory family
  fixed in its manifest. With `--arms REFERENCE CANDIDATE`, two arms of
  `population_run.py`'s array mode, each checked against its own
  `verification.json` and compared seed by seed on the metrics that
  `rescore` recomputed, with the confirmatory family of their manifests;
  either arm may be a campaign directory or its redacted tracked copies,
  which give the same report. Writes `assessment.json` and `comparison.md`
  (and, by default, the campaign manifests) under `--out`.
  `test_analyze_population_run.py` checks the statistics and the
  comparison of two arms on campaign directories it writes and on their
  redacted copies.
- `scripts/reference_join.py` — joins a finished `population_run.py`
  campaign to the golden reference as one command (`join`): it repeats the
  launcher's completion check on every case, verifies the previous
  reference against its tracked SHA256 manifest and the ZIP integrity of
  every trace, refuses any overlap, copies both populations byte for byte
  into a new combined directory, writes the combined summary, copies the
  new sidecars and the summary into `golden/baseline/`, and writes the new
  SHA256 manifest, the even/odd null check and a validation record under
  the given record directory (nonzero exit if the null check flags a
  configuration). `record-replay` adds a finished `golden_replay.py`
  report of the new configurations against the combined traces to that
  record. The README of the record is written by hand from its output.
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
  (this checkout's package, or the tree that `PYVBMC_SOURCE` names, which
  the report then names), about 7 minutes for the default set, and
  compares each run with its stored trace: exact shapes and values of every non-timer NPZ array and
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
  `scripts/runs/golden/reference_990_20260913/`, the current reference
  population, exists only on the machine that made it), `--sidecars`,
  `--out`, `--threads` (1, as the baseline), `--calibration-budget` (pin
  all three chunk budgets to this integer for a nondefault-profile check;
  omitted means historical defaults, independent of the local cache).
  Replay reports retain this setting, including on `--report-only`.
  Exit code 1 if anything is
  flagged or nothing was compared.
- `scripts/regenerate_baseline.sh` — the whole benchmark regeneration as
  one sequential process (see above).
- `scripts/make_svbmc_fixtures.py` — writes the S-VBMC test fixtures under
  `pyvbmc/testing/svbmc/fixtures/`: `convert` turns the thirty posteriors
  shipped with S-VBMC 0.1.1 (pickles in the pinned, ignored checkout) into
  plain-array snapshots with SHA-256 provenance; `generate` adds nine
  posteriors from short seeded VBMC runs (D=1; bounded D=2 with a different
  plausible box per run; correlated D=3 mixing warped and unwarped runs);
  `references` records the seeded three-step optimization of every group
  and mode as the regression gate (needs Torch); `saved-stack` writes one
  stack with `SVBMC.save`, the file that every CI cell loads, and a sidecar
  of its state and seeded draws (needs Torch). Every written posterior
  is rebuilt and compared with its source. `pyvbmc/testing/svbmc/FIXTURES.md`
  documents the files.
- `scripts/svbmc_speedup_benchmark.py` — paired before/after timing of
  complete S-VBMC fits from two source trees, each imported by its own
  warmed single-threaded worker process, alternating order per cell;
  checks weights, evaluation counts and generator states agree. Written
  for `plans/svbmc-speedups.md`; the evidence is in
  `experiments/svbmc_speedups/`.
- `scripts/svbmc_pool_run.py` — the run-pool generator of the S-VBMC
  benchmark campaign (`plans/svbmc-benchmark-campaign.md`), a harness of
  the campaign contract (`plans/slurm-benchmark-support.md`), which the
  driver under `scripts/hpc/` runs with
  `HARNESS=dev/scripts/svbmc_pool_run.py`: `prepare` fixes the allocation
  (every condition of the named suite, which for `svbmc_pool` is the
  campaign's eight pool conditions, `POOL_LABELS`, with a first seed, a
  seed cap and a filtered target each; `--only` allocates a subset; the
  release pools are `--target 320 --max-seeds 350 --allocation
  ring_D2_noise3_svbmc=320/480`, 2930 cases), the run options, the
  contract's identity over the harness checkout and the gpyreg checkout
  that `--gpyreg-source` names (required), the `site` block, the `pip
  freeze` and the finishing steps (`select`, `summarize`); it refuses a
  dirty gpyreg checkout, and a dirty harness checkout without
  `--allow-dirty`, and re-running it on a prepared directory revises the
  targets and seed caps alone, the previous allocation kept in
  `allocation_history`; `cases` prints one tag
  `<label>/<label>_seed<seed>` per case over every seed of each range
  (`--subset LABEL`, one condition's as `<index> <line>`); `worker --case
  LINE` runs one case through `campaign_contract.run_worker` (the early
  exit, the identity and claim refusals, the clean-up after a failure or a
  SIGTERM, a completion record holding the run's filter verdict, metrics,
  `success_flag`, `convergence_status`, `message`, `r_index` and
  `iterations`); `run` is the workstation supervisor, walking the
  conditions in manifest order, seeds upward, one `worker` process at a
  time, stopping each condition once its filtered target or its seed cap
  is reached (`--pilot-seeds K` runs exactly K seeds per condition
  instead; `--save-vbmc` also pickles the whole `VBMC` object); `select`
  then defines the filtered pool post hoc, per condition the lowest-seed
  runs that pass the filters up to the target, in `selection.json`, which
  the comparison reads, and reports its pass rate over the seeds it
  scanned before the target was met, which is not `summarize`'s over every
  completed case; `summarize` writes the per-condition pass rates,
  convergence, wall times and metric quartiles from the cases the
  directory holds; `verify` re-checks every stored artifact post hoc
  against its completion record and the manifest (the source identity,
  the SHA-256 of every file, the node feature and one physical core where
  the `site` block names `NODE_FEATURE`), the recomputation gate
  included, reconciles the allocation with `campaign_contract.reconcile`
  into `verification.json`, and takes `--gpyreg-source` for a pool copied
  to another machine (a clean checkout at the manifest's gpyreg commit,
  checked by `pinned_gpyreg_source`, which the stacking comparison
  shares). On such a machine the recomputation gate is met up to the other
  BLAS's rounding amplified by each run's GP condition number, which
  `verify_run` allows and reports (`--rounding-factor` scales the
  allowance); the report also records the verifying checkout's identity
  next to the pool's. Each condition's artifacts, error files and logs lie
  in its own directory, its records and claims under `records/<label>/`
  and `claims/<label>/`. A pool prepared before the contract, such as
  `pool_20260914`, keeps its flat layout (artifacts and records named
  `<label>_seed<seed>`, the flat identity of `identity()`), which the
  September scripts under `scripts/hpc/` generated: `verify`, `select`,
  `summarize` and `cases` read it with its own checks, and `prepare`,
  `worker` and `run` refuse it. `test_svbmc_pool_run.py` generates a short
  campaign and checks the artifact, resume, revision, selection, summary,
  the contract's refusals, claims, clean-up and `verify` states, the
  identity of a case run by the array worker and by `run`, the flat
  layout, and one campaign through the driver; it reads the gpyreg
  checkout from `PYVBMC_GPYREG_SOURCE` and skips when that is unset.
- `scripts/svbmc_pool_io.py` — the campaign's per-run artifact: `save_run`
  stores one finished run through the oracle snapshot codec (the returned
  posterior with all of `stats`, the GP that produced those statistics,
  the transformer, every evaluation, the run's state, options and
  metadata) and verifies it against the live objects, posterior,
  statistics and evaluations alike; `load_run` rebuilds every object
  through the public constructors; `verify_run` re-runs the checks on a
  stored artifact, including the recomputation gate (`_gp_log_joint`
  reproduces the stored `I_sk` and `J_sjk` from the rebuilt posterior and
  GP alone, exactly on the machine that generated the run and, on
  another, within that machine's rounding amplified by the condition
  number of the GP's kernel matrix, which the gate allows as a fixed
  multiple of `eps` times that number and reports) without the live run,
  and post hoc also against the hashes of the completion record, in the
  contract's shape or the flat layout's (`recorded_hashes`);
  `filter_verdict` applies the pool's stability and `J_sjk` filters.
- `scripts/hpc/` — Slurm tooling ([README](scripts/hpc/README.md), which
  holds the operator's guide to the release gate's campaigns: the
  settings, the environment and its frozen pins, the source trees, the
  environment check, each campaign command by command in the order of the
  plan's Phase 8, the finish's report, the limits, the redaction and the
  hand-back, and what to do when something refuses). The generic driver of
  the release gate (`plans/slurm-benchmark-support.md`, "The driver"),
  which each script's header documents, runs any harness that meets the
  campaign contract (`campaign_contract.py`) and holds no value particular
  to a site: `campaign_submit.sh` refuses a dirty source tree, an
  environment that differs from `campaign_requirements.txt` and a fixed
  operator setting that differs from the manifest's, prepares a campaign
  once, writes its case list once and submits `campaign_task.sbatch` in
  chunks below `MaxArraySize`, a named subset of the cases as its own
  array; `campaign_finish.sh` refuses while tasks are queued or running,
  runs `verify` and the harness's finishing steps as batch jobs, counts
  the cases in flight apart from the missing ones and writes the archive
  in parts with their SHA-256; `campaign_redact.sh` writes a finished
  campaign's tracked copies, redacted, on the login node in the account
  that ran it (`campaign_contract.py redact`); `campaign_env.sh`
  activates the environment, or builds it (`build`).
  `test_campaign_driver.py` runs them against stub Slurm commands
  (`campaign_slurm_stubs.py`) and a stub harness
  (`campaign_stub_harness.py`). Beside them, `svbmc_pool_submit.sh`,
  `svbmc_pool_task.sbatch`, `svbmc_pool_finish.sh` and `svbmc_pool_env.sh`
  are the record of the September pool (`pool_20260914`), which they
  generated on the University of Helsinki's Turso cluster; they no longer
  run against the pool harness, whose worker takes a case line where they
  pass a label and a seed.
- `scripts/campaign_contract.py` — the part of the campaign contract of
  `plans/slurm-benchmark-support.md` that the Slurm-driven harnesses
  share: the layout of a campaign directory (a case line starts with its
  tag; `records/<tag>.complete.json`, `claims/<tag>`, `<tag>.error.txt`);
  the claim of a case, hard-linked into place and judged stale only when
  the Slurm accounting shows that its task has ended (`acquire_claim`);
  the identity, whose source part (each tree's commit and clean state,
  file and directory hashes, the imported modules' versions) every worker
  compares with the manifest's, and whose import paths, installed-metadata
  versions and host part (CPU model, node features, BLAS threads, CPU
  affinity and its physical cores, Slurm ids) are recorded only
  (`identity`); the completion record and its check; the worker sequence
  with its refusals, and its clean-up when Slurm's SIGTERM stops a case
  (`run_worker`); the environment check against a pinned requirements
  file; the reconciliation of `verify`'s states, among them the
  `interrupted` case that a task killed outright leaves, which is
  resubmitted like a missing one (`reconcile`); and the tracked copies of
  a finished campaign, which its harness declares in the manifest
  (`tracked_copies`) and `redact` writes for the repository: hostnames
  reduced to the node family, paths under the operator's home to `~` and
  under a path setting to its name, the site block, the `pip freeze` paths
  and the Slurm accounting left in the archive, and every copy searched
  for what must not remain, with `redaction.json` recording each copy's
  SHA-256 beside its source file's, which `source_sha256` gives the
  readers that check the records' hashes. Run as a script, it offers the
  checks the driver's shell scripts call and `redact`.
  `test_campaign_contract.py` checks it.
- `scripts/svbmc_pool_stack.py` — the stacking comparison of the same
  campaign: for every condition, every `M` on a grid and every repetition,
  one subset of the filtered pool is stacked by both the integrated
  `pyvbmc.svbmc.SVBMC` (in process) and the original standalone `svbmc`
  0.1.1 (in a long-lived subprocess whose `PYTHONPATH` alone carries the
  pinned checkout's `src`, so no other process can import it), never at
  the same time and alternating which goes first, both arms rebuilding the
  subset's posteriors with one seed per entry derived from the cell's
  seed. The subsets of one condition and `M` are disjoint (repetition `r`
  takes the `r`-th block of `M` runs of a permutation drawn for that
  condition and `M`), so a pool of `n` runs gives at most `n // M`
  repetitions, and the cells beyond are listed as skipped. A
  condition's filtered pool is what its directory's `selection.json`
  names, or every passing completion record when the directory holds no
  selection; the printed lines and `sources.json` say which. A run's
  files are `<pool>/<tag>.npz` and `.json` whatever its tag holds, flat
  or in one subdirectory per condition, and their SHA-256, taken when the
  pool is read, is checked before a posterior is rebuilt. Each cell
  records the weights, every ELBO variant, the entropy, the seconds and
  the quality of 100 000 draws (`benchmark_targets.sample_metrics`, plus
  the Monte Carlo expected log joint over a random subsample of them).
  Every reported ELBO is scored by its bias against `elbo_mc`, the stacked
  posterior's own Monte Carlo ELBO, whose entropy term the script
  estimates for both arms with one estimator (the integrated class's
  `stacked_entropy`, 200 draws per component in 8 batches) from equally
  seeded generators, so that the reference is arm-independent — the same
  component draws for both arms, differing only through the weights —
  rather than taking each arm's own; the KL gap `ln Z - elbo_mc` is
  reported per cell. The outputs are `results.json`, `summary.json` and
  `summary.md` (medians with bootstrap intervals, the biases with
  criterion 3's gates, paired differences with exact signed-rank tests
  Holm-corrected over every condition and `M`, `max |dw|`, runtime ratio)
  and `sources.json`. It needs Torch importable (the campaign
  environment's, or the overlay `<BASELINE_DIR>/deps` on `PYTHONPATH`).
  A run of the original arm needs `BASELINE_DIR`, the upstream checkout
  or the directory holding it, and refuses to start unless that checkout
  verifies by content against
  `experiments/svbmc_pool/baseline_environment.json`, wherever it lies:
  the recorded commit, a clean tree, the committed content of every file
  of `src/svbmc` (CRLF read as LF) and the recorded Torch version; a run
  of the integrated arm alone needs no baseline. Both arms import gpyreg
  from `--gpyreg-source`, `PYVBMC_GPYREG_SOURCE` or the path the pool's
  manifest names, which must be a clean checkout at the gpyreg commit
  every pool's manifest records. `--arms` names the arms of every `M`,
  `both` or `integrated` (the integrated class alone, for the larger-`M`
  regime where the original's cost, quadratic in `M` and two to four
  times the integrated arm's, is not worth paying; such cells carry no
  paired quantity), once or once per `M`. The subcommands `prepare`,
  `cases`, `worker`, `verify` and the finishing step `assemble` meet the
  campaign contract of `plans/slurm-benchmark-support.md`, for the
  driver under `scripts/hpc/`: a task computes every repetition of one
  condition and `M` below `--split-from` (16) and one cell from it on,
  warms both implementations up first and writes one file per cell;
  `prepare` defaults to the release gate's grid; `assemble` refuses a
  campaign that does not verify whole and writes the outputs of the
  single-process run from the cell files, in the plan's order, with the
  same numbers but for the seconds. `--summarize-only --out DIR` rebuilds
  the summaries from a finished `results.json` without running a cell or
  needing Torch, describing that comparison by the settings it recorded
  rather than by the script's current constants, and with several
  `--from-results` files summarizes their cells together (both arms up
  to one `M`, the integrated arm beyond it), the paired quantities and
  equivalence tests covering the cell sets both arms ran. `--fixtures
  GROUP` compares the shipped S-VBMC posterior fixtures instead of a pool,
  which is what `test_svbmc_pool_stack.py` runs, in one process and as a
  campaign split into tasks, whose assembly it checks against the single
  process; the test module skips unless `PYVBMC_GPYREG_SOURCE` and
  `BASELINE_DIR` are set.
- `scripts/svbmc_honest_elbo.py` — the Phase 2 estimator of the S-VBMC
  ELBO optimism note on the pool artifacts: for every component of a
  stacked cell (a pool directory plus the comparison's `results.json`),
  the other runs' GPs that cover it estimate its expected log joint from
  draws mapped into each run's transformed space (predict mean minus
  log-Jacobian; the mean predictive variance drives the coverage rule
  against the component's own run), combined by median, precision
  weighting or mean, and every estimate (raw, the class's caps, honest
  under a grid of rules) is scored by its bias against the cell's
  `e_log_joint_mc`; the same draws give per-component truths,
  calibration `z` statistics of the self-reported SDs, and for every
  evaluating run and component the data it had there (effective and
  within-two-SD training-point counts, their offset from the truth, the
  GP mean function), written out for every run's heaviest component as a
  decomposition record, plus own-run consistency checks. Writes
  `results.json`, `cells.jsonl` as the sweep goes (a failed cell is
  recorded as skipped and the sweep continues), per-cell component
  arrays, summaries with the decomposition tables, figures and
  `sources.json` with the thread settings and the hashes of the script
  and of the cells file; `--headline-ratio` moves the headline within
  `--ratios`; `--self-check` runs the run-level checks on a pool without
  cells; `--summarize-only` rebuilds summaries and figures. Needs no
  Torch. It reads the artifacts of a pool of either layout of
  `svbmc_pool_run.py`. `test_svbmc_honest_elbo.py` (the gpyreg checkout
  from `PYVBMC_GPYREG_SOURCE`, skipped when that is unset) checks the
  contracts on a generated two-run pool and, synthetically, the combination rules, the checks and
  the mapping between two different transformers; the first run, on the
  pilot artifacts, is
  [results/2026-09-14-svbmc-honest-elbo-pilot.md](results/2026-09-14-svbmc-honest-elbo-pilot.md).
- `scripts/svbmc_cap_kappa.py` — weight-aware variants of the S-VBMC
  component-median cap, scored on a comparison's recorded cells without
  refitting: each cell's stack is rebuilt from the pool artifacts with
  the recorded seeds and constructed (the rebuild is checked against the
  cell's raw value and its recorded cap), and the cap is placed at the
  median expected log joint of the top-weight components carrying mass
  `kappa` (the crossing component included; `kappa = 1` is the class's
  cap) and at the weighted median, every variant's bias scored against
  the cell's `elbo_mc`. Needs Torch on `PYTHONPATH` and the pool's
  gpyreg through `--gpyreg-source`, a clean checkout at the manifest's
  gpyreg commit; writes `cells.jsonl`, `summary.json` and `summary.md`. Its runs are
  `experiments/svbmc_pool/cap_kappa_20260915/`, `cap_kappa_M35_20260915/`
  and `cap_kappa_M32_20260916/`, read in the
  [stage D report](results/2026-09-15-svbmc-pool-comparison.md).
- `scripts/svbmc_shrink_elbo.py` — empirical-Bayes shrinkage of the
  stacked expected log joint on a comparison's recorded cells: each
  component's estimate is shrunk toward its population mean by the share
  of the population's spread that is estimation noise (from the `J_sjk`
  and `I_sk` statistics saved with each posterior; no GP object), within each
  run with the diagonal or the full estimation covariance, or over the
  whole stack, at the run level (each run's own value shrunk toward the
  runs' mean by its run-level estimation variance, composed with the
  within-run forms), in anchored forms that add back what the
  within-run shrinkage removes at each run's own weights (mass-weighted
  or averaged over runs, so that a stack of one run reports its own
  value and only the stacking's selection is removed), and
  re-evaluated at the recorded weights; a `hybrid`
  rule applies the class's cap when the cell's noise share (the share of
  the components' spread the GP attributes to estimation noise) is at
  least 0.2 and the within-run shrinkage otherwise. Every variant's bias
  is scored against the cell's `elbo_mc`. Same rebuild, inputs and
  outputs as `svbmc_cap_kappa.py`; its runs are
  `experiments/svbmc_pool/shrink_20260915/`, `shrink_M35_20260915/` and
  `shrink_M32_20260916/`, read in the same report and in the
  [headline note](2026-09-15-svbmc-headline-shrinkage.md); the method
  is explained in the
  [tutorial note](2026-09-15-svbmc-shrinkage-explained.md).
- `scripts/svbmc_single_run_bias.py` — the bias of a single VBMC run's
  ELBO and what stacking adds to it: every filtered run of a pool is
  scored as a stack of one against its own Monte Carlo ELBO (the
  comparison's reference estimator at the run's own weights: the run's
  reported ELBO, the class's raw value for the run alone and the
  component-median cap), and with `--cells` (the comparison's
  `results.json`, repeatable) and `--shrink` (a shrinkage `cells.jsonl`,
  repeatable) every stacked estimate's bias is restated as the **added
  bias**, the stack's bias minus the mean bias of its input runs, the
  yardstick of the campaign plan's decision 11. The pool's gpyreg comes
  through `--gpyreg-source`, checked against the manifest's gpyreg
  commit, as for every script below that reads a pool. Its run is
  `experiments/svbmc_pool/single_run_20260915/`, read in the
  [stage D report](results/2026-09-15-svbmc-pool-comparison.md)
  (section "The inputs' own bias") and the
  [headline note](2026-09-15-svbmc-headline-shrinkage.md).
- `scripts/svbmc_shrink_optimize.py` — re-optimizes the stacking
  weights on the shrunken expected log joints: for every recorded cell
  the stack is rebuilt, the class's corrected expected log joints are
  replaced by the two-level full shrinkage of `svbmc_shrink_elbo.py`,
  the weights are optimized with the comparison's settings, and the
  new stack is scored as a cell is scored (its own `elbo_mc`, the
  biases, gsKL, MMTV, the KL gap) next to the raw optimization's
  record, the value-only shrinkage's bias (`--shrink`) and the added
  bias against the input runs (`--single-runs`). Its run is
  `experiments/svbmc_pool/shrink_opt_20260915/` (`M` = 3, 4 and 5),
  read in the [stage D report](results/2026-09-15-svbmc-pool-comparison.md)
  (section "Re-optimizing on the shrunken estimates").
- `scripts/svbmc_headline_numbers.py` — prints the numbers the headline
  note and the report's shrinkage, cap, single-run, re-optimization
  and `M = 32` sections quote (per-condition tables, worst and mean
  cases over the noisy conditions per `M`, noise-share percentiles,
  the hybrid threshold sweep, the caps' bind fractions, paired
  bootstrap intervals, the single-run biases and added-bias ranges,
  the re-optimization's rows; every aggregate over `M` both for the
  grid through 16 and for every `M`) from the `cells.jsonl`,
  `summary.json` and `added.json` files of the directories it is given
  (`--shrink`, `--caps`, `--single-run`, `--shrink-opt`; the quoted
  numbers are those of the tracked directories under
  `experiments/svbmc_pool/`, which its docstring's command names); NumPy
  only, no pool or raw directory needed. The check for a quoted number.
- `scripts/svbmc_shrink_worked_example.py` — prints the worked example
  of the [tutorial note](2026-09-15-svbmc-shrinkage-explained.md)
  (section 7): one recorded `M = 4` stack per condition rebuilt from
  the pool with the shrinkage script's own functions, with each run's
  spread, estimation standard deviation, error correlation, noise
  share, shrinkage factors and level shift and the stack's biases.
  Needs the pool and Torch, like the scoring scripts.
- `scripts/pymc_feasibility.py` — the bounded feasibility check of a PyMC
  model adapter (`2026-09-13-pymc-integration.md`): a PyMC model becomes a
  box-bounded log joint for `VBMC` (a variable bounded on one side keeps
  PyMC's transform and Jacobian, the others are handed over in their own
  coordinates with their bounds), the fitted posterior is exported with
  the model's names, shapes and coordinates, and PyMC computes
  deterministics and posterior predictions on it; five models with
  hand-written densities and evidences (unbounded, one-sided through the
  log and interval transforms, two-sided per coordinate), four rejected
  ones, density and Jacobian checks, and two routes to the starting point
  and plausible box (`--plausible laplace|prior`, the Laplace one with
  recorded fallbacks). Needs a Python environment with PyMC and
  ArviZ (the project venv has neither; the machine that ran it lists the
  environment in its gitignored `scripts/runs/LOCAL.md`). Outputs under
  `experiments/pymc_feasibility/`; the write-up is
  [results/2026-09-14-pymc-feasibility.md](results/2026-09-14-pymc-feasibility.md).
- `scripts/pymc_setup_probe.py` — preapproval setup and evaluation-reuse
  experiments for the [PyMC adapter plan](plans/pymc-target-adapter.md).
  Subcommands `a`, `references`, `b`, `check` and `summarize` compare
  capped gradient searches, generate and assess sequential NUTS references,
  run the three reuse arms at matched total budgets, verify derivatives
  and cached observations, and publish compact evidence. It uses the
  feasibility environment in `scripts/runs/LOCAL.md` and imports the
  historical prototype's coordinate mapping without editing it. Raw
  paths and draws stay under `scripts/runs/`; summaries belong under
  `experiments/pymc_setup_probe/`. See the
  [setup report](results/2026-09-16-pymc-setup-probe.md).
- `scripts/svbmc_parity_check.py` — historical: the moved
  `pyvbmc.svbmc.SVBMC` against the pinned upstream package on the thirty
  posteriors with matched draws (upstream's `testing=True` mode). Runs only
  from the source-move commit `a8ae260`, the last where the moved class
  carries that flag; its result (every difference exactly zero) is in
  `plans/svbmc-integration.md`.
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
  `gp_fit`, `gp_fit_history`) off the generating platform, where `--check`
  skips them as the tests do (`PYVBMC_ORACLES_ALL=1` forces them), and runs
  one process at a time.
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
  change made while the references are known to lag. The authentic
  captures of the hyperparameter fit under `fixtures/gp_fit_history/` have
  their own modes: `--capture-gp-fit-history` writes them and refuses to
  replace one, `--check-gp-fit-history` compares them, and
  `--rebaseline-gp-fit-history NAME --reason "..."` replays one capture on
  its stored inputs after a deliberate change of the fit and replaces the
  outputs of the fit and the sampler widths alone, on the generating
  platform only, with the captured inputs and the portable references
  asserted bit-identical and an audit entry in the capture's `.json`.
  The three captures still hold the arrays `capture/ref/fit/hyp_dict_logp`,
  the fit's output under the `logp` key that the port review's W6-23
  removed from its hyperparameter dictionary; no sidecar describes them,
  and a capture written anew leaves them out.
