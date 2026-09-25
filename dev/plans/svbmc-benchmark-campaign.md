# S-VBMC run-pool benchmark campaign

Created 2026-09-13. Status: **complete**. The design was approved by
the PI on 2026-09-13
and revised on 2026-09-14 (decisions 7–10); the harness was merged
into `dev-next` on 2026-09-14 (`647698b`); the pools were generated on
the cluster the same day (draft release `svbmc-pool-20260914`); stage
D, the two-arm comparison on all eight conditions at `M` = 2, 4, 8
and 16, ran on 2026-09-14/15, the integrated arm alone at `M` = 3
and 5 on 2026-09-15 and at `M = 32` overnight on 2026-09-15/16.
Criteria 1, 2, 4 and 5 hold on every condition;
criterion 3's gate fails on Student D8, where the capped headline
over-corrects (the
[stage D report](../results/2026-09-15-svbmc-pool-comparison.md) and
the worklog). The Phase 2 scoring and the estimates scored afterwards
on the same cells are in the report and summarized in the
[headline note](../2026-09-15-svbmc-headline-shrinkage.md). The
`M = 32` run of the integrated arm (80 cells, 4 hours) closed the
campaign on 2026-09-16; the report's section "The integrated arm at
`M = 32`" reads it. The pools and the comparison are long campaigns
under the working rules of [dev/README.md](../README.md#scripts): one
heavy process at a time, started only on explicit PI instruction, in
stages that are authorized separately.

This plan owns the design, harnesses, allocation, gates and worklog of the
campaign. The [integration plan](svbmc-integration.md#benchmark-campaign-required-for-15)
owns the requirement; the [optimism note](../2026-09-12-svbmc-elbo-optimism.md#phase-2-measuring-instead-of-capping-with-the-runs-surrogates)
owns the Phase 2 estimator that consumes the pools; the
[reporting plan](svbmc-elbo-reporting.md) owns the Phase 1 corrections
that the integrated class ships.

## Purpose

Two open items of PyVBMC 1.5 need the same input:

- **Phase 2 of the ELBO debiasing** evaluates every stacked component's
  draws under the GPs of the *other* runs. It needs pools of independent
  VBMC runs on noisy targets, each saved with its final posterior and the
  GP that produced the posterior's expected-log-joint statistics.
- **The comparison against the original S-VBMC** (standalone `svbmc`
  0.1.1, pinned at `13a78f6`) needs matched groups of input runs on noisy
  targets, stacked at different numbers of runs by both implementations,
  scored for posterior quality, evidence accuracy and runtime.

No retained artifact serves either as it stands. The
population campaigns left 990 boost captures under
`dev/scripts/runs/population_*/results/*.boost.pkl` whose final posteriors
carry `I_sk` and `J_sjk` and could be stacked today, but they have no GP,
predate the `uncertainty_handling_level` stat (added 2026-09-12 in
`483a8da`; the captures are from `68a43db` and `fc50ee1`, so every noisy
stack of them would fall back to the `elbo_sd > 0.1` proxy), and were run
with the 2020 paper's pinned budget. The 870 dills of the 2026-09-08 boost
campaign pair posteriors with GPs, but the GPs were reconstructed from
traces at an older code state and there are no real-data targets. The
golden traces hold hyperparameters and training sets from which GPs could
be rebuilt next to the capture posteriors, at the price of an audit of the
reconstruction. The S-VBMC paper's own corpus (30 posteriors in the pinned
checkout, converted to `pyvbmc/testing/svbmc/fixtures/`) has no GPs. No
whole `VBMC` object was saved anywhere.

The campaign therefore generates fresh pools with a per-run artifact both
consumers read, builds the harnesses (pool generator, baseline
environment, stacking comparison), measures a pilot, and fixes the
allocation from the pilot.

## What the comparison tests

The two implementations optimize the same stacking objective on the same
inputs, so posterior quality is expected to coincide up to Monte Carlo
noise and, on bounded targets, up to the deterministic Jacobian correction
the integrated class applies (the reporting plan measured a bounded
all-weights L1 weight change of about 0.01 against a seed-to-seed spread
of about 0.12). The comparison is therefore three things:

- an **equivalence check** that the source move, the preparation and
  entropy speedups and the Phase 1 corrections did not regress the
  stacked posterior (criteria 1 and 2 below);
- a **measurement** of what the reporting change does to evidence
  accuracy as the number of stacked runs grows: the original reports the
  raw stacked ELBO, the integrated class the capped headline, and the
  Phase 2 honest estimator is later evaluated on the same cells
  (criterion 3);
- a **measurement** of runtime (criterion 4), with reproducibility as the
  fifth criterion.

The intended reporting differences are documented next to the results,
as the integration plan requires.

## Design

### Conditions

| # | Label | D | Noise SD | Origin | Filtered pool | Seed cap | Role |
|---|---|---|---|---|---|---|---|
| 1 | `multisensory_s1_D6_noise3_svbmc` | 6 | 3 | new configuration; the paper's real-data condition | 100 | 150 | real data, bounded; paper parity |
| 2 | `multisensory_s1_D6_noise1.3_svbmc` | 6 | 1.3 | the 2020 paper's IBS noise level; the hardest golden configuration ([summary](../golden/baseline/summary.md)) | 100 | 150 | real data at the realistic noise level; the noise-scaling point |
| 3 | `rosenbrock_D2_noise3_svbmc` | 2 | 3 | the golden target at the default budget | 100 | 150 | PI priority target |
| 4 | `gmm_D2_noise3_svbmc` | 2 | 3 | ported from the pinned upstream `targets.py` | 100 | 150 | the paper's multimodal target; Phase 2 needs the per-mode coverage case |
| 5 | `ring_D2_noise3_svbmc` | 2 | 3 | ported from upstream | 100 | 200 | paper parity; the largest residual bias after capping in the paper |
| 6 | `student_D8_noise3_svbmc` | 8 | 3 | golden target at the default budget | 100 | 150 | heavy tails at higher dimension |
| 7 | `gmm_D2_svbmc` | 2 | none | ported | 50 | 75 | noiseless synthetic control: no ELBO overshoot expected |
| 8 | `multisensory_s1_D6_svbmc` | 6 | none | golden target at the default budget | 50 | 75 | noiseless real-data control |

Every pool configuration is a separate `Config(..., tag="svbmc")` entry
in the suite `svbmc_pool` at PyVBMC's default evaluation budget
(75 (D + 2) for noisy targets, 50 (D + 2) otherwise), the S-VBMC paper's
convention and what users get (PI, 2026-09-13). The tag keeps
`find_config` from confusing a pool configuration with a golden one, which
pins 50 (D + 2) on noisy targets; `M = 1` baselines come from the pool
itself. Pool sizes are the paper's 100 filtered runs per noisy condition
and 50 per noiseless control (PI, 2026-09-14), which supports `M` up to 40
with diverse subsets. The seed caps scale the paper's over-provisioning
(Table A.2 of `papers/silvestrin2025stacking_appendix.md`: 150 runs for
145 filtered on GMM, 149 for 100 on the ring, 150 for 149 on
multisensory, all at noise 3) with margin. The pools are generated on a
cluster (section "Cluster generation"); the analyses that consume them
run on a laptop within an overnight budget of 8–10 hours.

Conditions considered and left out (PI, 2026-09-14): multisensory
subject 2 (the same model on a second dataset), the timing model (real
data is covered, and its likelihood costs about 50 times more per
evaluation), Rosenbrock at noise 1 (redundant with noise 3), logistic
regression at noise 3 (bounded posteriors are covered by multisensory),
lumpy at D = 10 and noise 3 (never validated in a campaign), and the
noiseless ring (one synthetic control suffices).

Noise is the suite's generic wrapper (homoskedastic Gaussian noise on the
log density, known SD returned to VBMC, `specify_target_noise=True`), as
in both papers. The paper's neuronal (NEURON) benchmark is not
reproducible from this repository and is not substituted. `dev/plans/benchmark-realistic-targets.md`
lists a "bimodal ring" among rejected candidates; that rejection was about
real-data coverage and does not bear on a synthetic pool target. Noisy
runs at noise 1.3 terminate on the reliability index at about 205
evaluations, well inside the budget; at noise 3 the budget may bind.

### Seeds, filters and stopping rule

Seeds are contiguous from 1000 per condition, disjoint from the golden
population's 0–49 so the two populations are never conflated. One integer
seeds the target's noise and start-point streams (`Config.make(seed=)`)
and VBMC (`VBMC(seed=)`), as `golden_trace.run_task` does.

A run enters the filtered pool when it passes the paper's two filters,
which are also both implementations' defaults: `stats["stable"]` is true
and `sqrt(max J_sjk) < sqrt(5)`, the maximum taken over the whole
`(Ns, K, K)` array. Every run is kept and recorded with its verdict and
`max_J_sjk = float(np.max(vp.stats["J_sjk"]))`; the seeds run in order and
a condition stops when the filtered count reaches its target or the seed
cap is reached, the paper's "lowest indices" rule. The pilot stage runs a
fixed number of seeds per condition regardless of the filters.

### Per-run artifact

One `<label>_seed<seed>.npz` plus `.json` sidecar per run, written with the
oracle snapshot codec (`pyvbmc/testing/oracles/_state.py`,
`snapshot_from_objects` / `save_snapshot`), holding as plain arrays:

- the returned posterior `vbmc.vp` with all of `stats` (`I_sk`, `J_sjk`,
  `elbo`, `elbo_sd`, `e_log_joint`, `e_log_joint_sd`, `entropy`,
  `entropy_sd`, `stable`, `uncertainty_handling_level`);
- the GP that produced those statistics, `vbmc.get_gp(results["best_iter"])`:
  training inputs and targets (log joint with the log-Jacobian folded in,
  in the run's transformed space), `s2`, every hyperparameter sample, and
  the covariance, mean and noise specification. `optimize()` boosts the
  best iteration's posterior with exactly this GP
  (`pyvbmc/vbmc/vbmc.py`, `final_boost(self.vp, self.get_gp(idx_best))`);
  when the boost candidate is rejected the returned posterior is the best
  iteration's, whose statistics that iteration's GP produced, so the
  choice holds in both branches. The optimism note speaks of "the final
  GP of each run, `vbmc.gp`"; `vbmc.gp` is the last iteration's GP, which
  differs from the best iteration's whenever the best iteration is not
  the last, and the recomputation gate below makes the choice checkable;
- the parameter transformer;
- the live function logger (every evaluation of the run, `X_orig`,
  `y_orig`, `S`, `n_evals`);
- `vbmc._optim_state_record()` (the live state without the noisy
  acquisitions' importance samples, a private method the two pool scripts
  depend on) and the user options. The codec reads `gp_cov_fun`,
  `gp_mean_fun`, `gp_noise_fun`, `plb_orig` and `pub_orig` from this
  state; the three function specifications are set once at
  initialization, and the plausible bounds only seed a transformer whose
  `mu`, `delta`, `type`, `R_mat` and `scale` the codec then overwrites
  from the saved transformer, so a warp during the run does not corrupt
  the rebuilt state;
- `meta`: label, seed, target name, `D`, noise SD, requested and effective
  options, `results` passed through `profile_run.jsonable` (ELBO, ELBO SD,
  `best_iter`, `success_flag`, `func_count`, and the nested `rng_state`
  and `performance_calibration` dicts), `K`, wall and target-evaluation
  seconds, metrics against the truth (`elbo_err`, `gskl`, `mmtv`, `rmse`;
  the moment arrays through `jsonable`), the filter verdict, `pyvbmc` and
  `gpyreg` commits and import paths, library and Python versions, thread
  settings and hostname.

`build_state` rebuilds every object through public constructors. Saving
verifies the artifact against the live objects: posterior arrays and
stats equal (`equal_nan=True`, since non-finite floats round-trip through
a tag), GP predictions at the training inputs equal within 1e-10, and the
**recomputation gate**: `_gp_log_joint(vp, gp, False, True, True, True, True)`
(`pyvbmc/vbmc/variational_optimization.py`) returns
`(G, dG, varG, dvarG, var_ss, I_sk, J_sjk)` from the rebuilt posterior and
GP alone, and both arrays must match the stored statistics within 1e-8.
They are weight-independent, so a pruned posterior recomputes exactly.
The gate also runs post hoc without the live objects. That absolute
tolerance is met exactly on the machine that generated a run; on another
machine the BLAS rounds differently, and the rebuilt posterior factors
carry that rounding amplified by the condition number of the GP's
kernel matrix, which the noisy Rosenbrock condition's far-tail
evaluations push to 3e17 (worklog, 2026-09-14). The gate therefore
allows each artifact a relative deviation of 100 times machine epsilon
times its own condition number on top of the absolute tolerance, and
reports the deviation and the condition number. A
`records/<tag>.complete.json` holds SHA-256 hashes of both files, elapsed
time, identity and verdict, and gates resumption as in
`population_run.py`. `--save-vbmc` additionally writes `<tag>.vbmc.pkl`
through `VBMC.save` for the pilot runs only (a few MB each, the whole
iteration history), so the Phase 2 interface question (VBMC objects,
posterior-GP pairs, or the GP attached to the posterior) can be
prototyped against real objects.

About 300 KB per run (the `J_sjk` array dominates; the noisy VIQR oracle
fixture with four iterations of state is 150 KB); the 395 runs at the
seed caps are about 120 MB, gitignored under
`dev/scripts/runs/svbmc_pool_<date>/`. Curated summaries, manifests and
comparison outputs are tracked under `dev/experiments/svbmc_pool/` with a
README that explains every key.

### Pool runtime (measured in the pilot)

Pilot of 2026-09-14, three seeds per condition, one process, BLAS
single-threaded, harness commit `e2aaef5`, gpyreg at the pin. Every one
of the 15 runs passed the filters and every artifact verified post hoc
with exact zeros on the recomputation gate.

| Condition | Wall per run (median) | Evaluations (median) | `K` | Single-run `elbo_err` / gsKL / MMTV (median) |
|---|---|---|---|---|
| `multisensory_s1_D6_noise3_svbmc` | 3.3 min | 265 | 50 | 0.72 / 5.85 / 0.343 |
| `rosenbrock_D2_noise3_svbmc` | 1.6 min | 180 | 50 | 0.03 / 0.04 / 0.083 |
| `gmm_D2_noise3_svbmc` | 1.5 min | 175 | 50 | 0.46 / 1.17 / 0.317 |
| `ring_D2_noise3_svbmc` | 1.9 min | 170 | 50 | 1.91 / 154 / 0.651 |
| `gmm_D2_svbmc` | 0.25 min | 75 | 50 | 0.71 / 11.0 / 0.363 |

The single-run quality of the ring and the two GMM conditions is poor by
construction (one run covers a piece of the ring or some of the four
clusters), which is the regime stacking is for. Per-run costs are well
below the paper's server timings (about 9 min for the ring and 14 min for
multisensory at noise 3) and below the pre-pilot assumptions. For the two
conditions the pilot did not run, the `reference_990_20260913` sidecars
give 3.16 min for the noise-1.3 multisensory configuration and 5.3 min
for Student D8 at noise 3, both at the pinned budget, and 1.25 min for
noiseless multisensory. Projected cost of the eight-condition pool at
the seed caps: about 1200 runs and roughly 45 CPU-hours (150 × 3.3 +
150 × 3.2 + 150 × 1.6 + 150 × 1.5 + 200 × 1.9 + 150 × 5.5 + 75 × 0.25 +
75 × 1.25 min), about 30 CPU-hours if every condition reaches its
filtered target at the pilot's pass rates: an afternoon as a cluster
array job, or several nights on a laptop one process at a time. The
pools are generated on the cluster (section "Cluster generation"); the
comparison (stage D) and the Phase 2 analyses run locally. The worker is
invocable per case, which is what the array job calls.

### GP library pin

Every pool run and both comparison arms import gpyreg from a frozen,
detached worktree at a released tag, through the environment variable
`PYVBMC_GPYREG_SOURCE` prepended to `sys.path` before PyVBMC is imported,
the mechanism `population_run.py` uses. The identity check of every
process refuses to run when `gpyreg.__file__` does not resolve under that
directory. The pin is `v1.2.1`, commit
`9e70e6ba53f7607d05c2d9cc2fa9f41cd12b8f3b`, worktree
`dev/scripts/runs/svbmc_pool_20260913/gpyreg_1.2.1/` (PI, 2026-09-14,
the day 1.2.1 was released; the CI pin in `test-matrix.yml` moved with
it). Under 1.2.1 the exact oracle check is bit-identical on 10 of the 11
fixtures and the VIQR acquisition oracle moves by one ulp (8.9e-16), the
footprint of that release's heteroskedastic quadrature-variance fix, with
the CMA-ES search oracle downstream of it still exact; the 15 pilot
artifacts recompute their `I_sk` and `J_sjk` with zero difference, so the
GP numerics S-VBMC depends on are unchanged. The pilot itself ran against
`v1.2.0` (`39536b0`); its worktree was removed once the pilot comparison
had been regenerated, so that no process can pick up the old version by
habit, and its records name the commit. To re-run anything against the
pilot pool, recreate it with
`git -C ../gpyreg worktree add --detach dev/scripts/runs/svbmc_pool_20260913/gpyreg v1.2.0`.
The pools generated for the campaign use 1.2.1.
At the time of Phase 1 the sibling checkout `../gpyreg`, which the venv's
editable install points at, was an in-progress release branch with
uncommitted changes to the GP core, which is why the campaign never
imports it; `baseline_environment.json` records that state. If a gpyreg
change to the GP numerics lands before 1.5, the pools must be regenerated
against it, as the working rules require for golden references.

### Cluster generation

The pools are generated on a Slurm cluster by another developer as a
once-in-a-while golden-fixture job (decision 8); the short brief handed
to them is [the hand-off brief](svbmc-pool-handoff.md),
which asks for the campaign directory as an archive and a pull request
to `dev-next` from a branch off it. What the harness provides for that,
and what the hand-over needs:

- **A clean checkout at a named commit** of this repository (the pool's
  identity records it and every worker refuses a different one) with
  the package installed, `psutil` and `filelock`; a gpyreg checkout at
  the tag of the "GP library pin" section, named through
  `PYVBMC_GPYREG_SOURCE` or `prepare --gpyreg-source`; no Torch and no
  original-svbmc checkout are needed on the cluster (both belong to the
  local comparison only). The targets' data and ground truths are
  tracked files.
- **`prepare`** on the login node writes the manifest for the eight
  conditions with their targets, seed ranges and caps; the identity's
  *source* part (PyVBMC commit and clean state, gpyreg commit and clean
  state, the hashes of the suite, io and runner modules, and the Python,
  NumPy and SciPy versions) is what workers are compared against, its
  *host* part (hostname, platform, interpreter, import paths, installed
  distribution versions, thread settings) is recorded only (decision 9).
  `identity()` runs `git` on both checkouts, so the cluster needs `git`
  and real repositories, not exported tarballs.
- **`cases`** prints every `(label, seed)` of the allocation over the
  full seed range, one per line, so a Slurm array maps its index to one
  `worker --out DIR --label L --seed S` call. The default allocation is
  1100 cases and Slurm's default `MaxArraySize` is 1001, which caps the
  largest array index, so on such a site the cases go out in chunks
  whose tasks add an index offset to their array index (a throttle alone
  does not get past the cap); `dev/scripts/hpc/` implements this. Each
  task sets
  `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`,
  `MPLBACKEND=Agg` and `PYVBMC_GPYREG_SOURCE`, runs one case, writes its
  artifact and its hash-verified completion record, and needs one core
  and under 2 GB of memory for a few minutes. A failed task removes its
  partial artifact, leaves `<tag>.error.txt` with the traceback, exits
  non-zero, and is rerun or left out; `select` and `summarize` count it.
  The manifest stores the gpyreg source as an absolute path, so
  `prepare` runs on the cluster, never on another machine for a
  directory that is then copied there.
- **`verify`** runs first on an array's output: every artifact is
  re-checked against its record and the manifest's source identity, the
  recomputation gate re-run, and the allocation reconciled, which is the
  only way a task Slurm killed shows up (it leaves neither a record nor
  an error file), each such case named with the index to resubmit.
  **`select`** then defines the filtered pool post hoc, per condition
  the lowest-seed runs that pass the filters up to the target, written
  to `selection.json`, which the comparison reads; **`summarize`**
  writes the pool summary. The sequential `run` supervisor, its
  `FileLock`, `status.json` and stopping rule are the laptop path and
  are not used on the cluster.
- **Hand-back**: the campaign directory (artifacts, records,
  `manifest.json`, `selection.json`, summaries) is copied back under
  `dev/scripts/runs/` on the analysis machine, which lists it in its
  gitignored `dev/scripts/runs/LOCAL.md`, and its manifest, selection,
  summaries and verification report into
  `dev/experiments/svbmc_pool/pool_<date>/` with a README. Pool runs on Linux with the
  cluster's BLAS will not reproduce laptop runs bit for bit; that is
  expected for a pool, and the records carry the platform.

### Baseline: original S-VBMC 0.1.1

The pinned checkout `dev/scripts/runs/svbmc_compat_20260908/source/` (git
HEAD `13a78f6c4a3ffe9c4557fee4f4b8f67c98b29e01`, clean tree) is the
baseline implementation. It declares `pyvbmc>=1.0.4`, `GPyReg>=1.0.2`,
`torch>=2.7`, `numpy>=2.3`, `scipy>=1.16`, `corner` and `matplotlib`; it
imports PyVBMC and gpyreg from the project's editable installs (this
checkout and the sibling gpyreg checkout), and the sibling `deps/`
overlay supplies CPU Torch 2.14.0, which is **not** installed in the
project venv. The compatibility campaign's `environment.json` records
exactly that layout. Two path sets are used throughout this plan:

- `TORCH_PATH` = `dev/scripts/runs/svbmc_compat_20260908/deps`: needed by
  anything that imports the integrated `pyvbmc.svbmc` (tests, the
  comparison controller, the pool generator's stacking check);
- `BASELINE_PATH` = `TORCH_PATH` joined with
  `dev/scripts/runs/svbmc_compat_20260908/source/src` by `os.pathsep`:
  the original arm's worker only. The controller must not carry
  `source/src`, or a stray `import svbmc` would pick up upstream. In Git
  Bash a two-entry value must be quoted (`PYTHONPATH="…/deps;…/source/src"`).
  Subprocesses set `env["PYTHONPATH"]` to the joined value explicitly;
  `svbmc_speedup_benchmark.py` replaces the variable with a single path
  and must not be copied for this. The overlay also shadows a few venv
  packages (`filelock`, `setuptools`, `typing_extensions`, `jinja2`,
  `networkx`, `sympy`, `fsspec`, `mpmath`) with its own versions.

The original constructor is `SVBMC(vp_list, s_max=np.sqrt(5), M_min=2/3, testing=False)`
and `optimize(n_samples=20, lr=0.1, max_steps=500, version="all-weights")`;
it reports `elbo["estimated"]` (raw), `elbo["debiased_I_median"]` and
`elbo["debiased_E_median"]` from the optimization's own 20-draw estimate
(no final re-evaluation), and `sample(n)` returns approximately `n` rows
by deep-copying each input posterior and drawing `round(n · ω_m)` from it.
The integrated class reports the capped headline for noisy stacks with
`elbo_details["raw"]`, `raw_sd`, `cap_amount` and `noise_status_source`,
re-evaluates with `n_samples_final=100`, and `sample(n)` returns exactly
`n` rows.

**Randomness.** Neither implementation draws from Torch; Torch supplies
autodiff and Adam only, and `torch.manual_seed` controls nothing. The
integrated class draws from its own NumPy generator, `SVBMC(seed=)`. The
original draws its entropy samples from NumPy's global legacy state
(`scipy.stats.multivariate_normal.rvs` without `random_state`) and its
posterior samples through the input posteriors' own generators, which
`VariationalPosterior.__deepcopy__` shares. The original arm is therefore
seeded with `np.random.seed(cell_seed)` and receives posteriors rebuilt
afresh for the cell, each with its own generator seeded from
`np.random.SeedSequence(cell_seed).spawn(M)` (the same per-entry seeds in
both arms, recorded in the cell), so the original's per-run sample blocks
are independent of one another; reusing loaded posteriors across cells
would make its results order-dependent, and one shared seed would couple
the blocks.

The baseline is machine-local and gitignored. Its identity is preserved
in `dev/experiments/svbmc_pool/baseline_environment.json` (commit, clean
tree, SHA-256 of every `svbmc/*.py`, Torch, Python, NumPy, SciPy, PyVBMC
and gpyreg commits), and it is recoverable elsewhere by cloning the
upstream repository (`acerbilab/S-VBMC`) at `13a78f6` and installing CPU
Torch 2.14.0 into a `deps/` directory with `pip install --target`; the
recorded hashes verify the recreation. The stacking harness re-verifies
the environment before every campaign.

### Stacking comparison

For every condition, `M` on a grid capped at the filtered pool size, and
`R(M)` repetitions, a subset of `M` runs is drawn without replacement from
the filtered pool by `np.random.default_rng([seed, condition_index, M, r])`.
The same subset goes to both implementations, run one after the other,
never concurrently, alternating which arm runs first. Each cell records
the optimized weights, every ELBO variant, entropy, construction and
optimization wall seconds, metrics of 100 000 draws from the stacked
posterior, and the Monte Carlo ELBO of the stacked posterior, `elbo_mc`,
the reference every reported estimate is scored against:
`e_log_joint_mc`, the mean of the noiseless `problem.log_density_vec`
over a seeded random subsample of 10 000 of the draws, plus
`entropy_ref`, the entropy of the stacked mixture at the arm's final
weights estimated afresh by the harness with the integrated class's
entropy machinery (a few hundred draws per component, a dedicated
generator, the same estimator for both arms), never an arm's own
reported entropy; both terms carry their Monte Carlo standard deviation.
Per cell and arm the harness records `bias_<variant> = elbo_<variant> −
elbo_mc` for every ELBO variant the arm reports and the KL gap
`ln Z − elbo_mc`. The multisensory likelihood costs about 0.6 ms per
evaluation, so the reference adds seconds per cell and gives Phase 2 the
`ELBO_MC` reference on every cell without a rerun. `M = 1` rows are the
filtered pool's own single-run metrics.

**Cost.** The entropy Monte Carlo evaluates every component against every
draw at every Adam step, so a cell costs about `M²`. Measured in the pilot
(2026-09-14, `M = 3`, 500 Adam steps, five repetitions per condition,
single-threaded): the integrated arm's `optimize` takes 0.5–0.7 s on every
condition, the original's 1.1 s (noiseless GMM), 2.0–2.5 s (noisy D = 2)
and 2.8 s (multisensory); the runtime ratio is 0.20–0.51. Extrapolating
with `M²`, a cell at `M = 16` costs about 17 s for the integrated arm
and 70–80 s for the original, at `M = 32` four times that. The grid of
decision 6 (`M ∈ {2, 4, 8, 16}`, `R = 20, 20, 20, 10`, both arms) costs
about 35 minutes per condition; over the eight conditions of decision 8
that is under 5 hours, and the entropy reference adds a few minutes.
With pools of 100 the grid can follow the paper further: the integrated
arm at `M = 32` with `R = 10` adds about 15 minutes per condition, the
original arm about an hour, so the proposal for stage D is the integrated
arm on `M ∈ {2, 4, 8, 16, 32}` over all eight conditions and the original
arm on `M ≤ 16`, about 7 hours in all, inside the overnight budget of
decision 8; `M = 40` for the integrated arm on the four paper conditions
at `R = 10` would add about an hour. The grid is fixed when stage D is
authorized. The paper's timings (about 2300 s at `M = 40` on multisensory
for the original) had put the paper's full protocol near 100 hours on
the pilot machine; that was an overestimate.

Summaries report the median over repetitions with a 95 % bootstrap
interval (10 000 resamples), the paired differences integrated minus
original, the maximum weight difference, and the runtime ratio.

### Metrics

The house triple, as `benchmark_targets.metrics` computes it for single
runs: `elbo_err = |ELBO − ln Z|`, `gskl = 0.5 · Σ kl_div_mvn(both directions)`
(the 2020 paper's convention, no `1/D` factor), `mmtv` (mean marginal
total variation against exact reference draws through `kde_1d`), with the
usability thresholds evidence error < 1, gsKL < 1, MMTV < 0.2. For stacked
posteriors the same quantities are computed from draws (`sample` in both
implementations): moments from the draws for gsKL, the
`VariationalPosterior.mtv` procedure applied to two sample sets for MMTV,
and `elbo_err` for every reported ELBO variant. The S-VBMC paper's
normalized GsKL, `(1/2D) Σ KL`, equals the house value divided by `D`
and is stored as `gskl_normalized` (paper threshold 1/8) so
paper-comparable figures can be drawn; every figure names its convention.

### Acceptance criteria for the comparison

1. **Agreement (equivalence).** For every condition, the paired weight
   difference between the two implementations, `max |Δw|` per cell, lies
   within the within-arm seed-to-seed spread measured on the same inputs
   (Phase 1 measures it on unbounded, bounded and warped fixture groups
   at five seeds per arm; Phase 4 repeats it on the pilot posteriors of
   every condition), and the integrated capped headline agrees with the
   original's `debiased_I_median` within their Monte Carlo standard
   deviations. On bounded targets a deterministic offset of the order
   the reporting plan measured is expected and is not a failure. A
   difference beyond the spread stops the campaign for investigation.
   *Agreement tolerance.* Phase 1 (2026-09-13, three Adam steps, five
   seeds per arm): paired `max |Δw|` at most 0.0085 on the unbounded
   upstream groups (`M = 10`, `K = 50`) against within-arm spreads of
   0.0086–0.0096, and at most 0.0076 on the bounded and warped
   supplementary groups (`M = 3`) against spreads of 0.0077–0.0096, so
   the paired difference is below the within-arm spread everywhere
   (ratios 0.79–1.0). The capped headline agreed with
   `debiased_I_median` within 0.25 `elbo_sd` on the one group where
   both caps were active; on `bounded_D2` and `corr_D3` only the
   original's cap is active (their `elbo_sd` is below the 0.1 proxy),
   which is a reporting difference, not a disagreement. Phase 4
   (2026-09-14, the three pilot posteriors of every condition at `M = 3`,
   500 Adam steps, five cell seeds): paired `max |Δw|` 0.028–0.044
   against within-arm spreads of 0.018–0.044 (integrated) and
   0.027–0.048 (original); the paired difference is at most 1.3 times the
   larger within-arm spread (rosenbrock 1.03, ring 1.29, the others
   below 1). The raw stacked ELBO of the integrated class sits
   0.04–0.19 nats below the original's on every condition, and its
   entropy 0.02–0.16 nats below: the original reports the optimizer's
   own 20-draw entropy at the selected weights, the integrated class a
   fresh 100-draw evaluation, so the offset is the intended reporting
   difference, not a disagreement of the posteriors. The capped headline
   and `debiased_I_median` differ by 0.03–0.18 nats, within one
   integrated `elbo_sd` (0.17–0.24) on every noisy condition.
2. **Posterior quality (equivalence).** For every condition and `M`, the
   paired differences in MMTV and gsKL are not significantly worse for
   the integrated class (exact signed-rank tests, Holm-corrected across
   conditions and `M` at α = 0.05), and both implementations improve on
   the `M = 1` medians as the paper reports.
3. **Evidence accuracy (measurement).** The yardstick is the stacked
   posterior's own ELBO, `ELBO(q) = E_q[log p] + H[q] = ln Z − KL(q‖p)`,
   estimated per cell as `elbo_mc` (see "Stacking comparison"); every
   reported estimate is scored by its bias `estimate − elbo_mc`. At
   every `M`, the absolute value of the median bias of the integrated
   headline is at most that of the original's raw estimate; and the
   median headline bias at the largest `M` exceeds its value at the
   smallest `M` by less than 0.5 nats, the bound the paper reports for
   the capped bias. The raw
   bias curve with `M` and the honest estimator's later bias on the same
   cells are reported, not gated. The error against `ln Z`,
   `|estimate − ln Z|`, is reported as a descriptive column only: it
   mixes the estimator's bias with the stack's KL gap, so a small value
   can arise by cancellation and cannot rank estimators (PI, 2026-09-14).
   The KL gap `ln Z − elbo_mc` is reported per cell as the
   posterior-quality measure in evidence units.
4. **Runtime (measurement).** Per condition, the median over cells of
   the paired ratio integrated / original optimization seconds is below 1
   with its 95 % bootstrap interval below 1, on the same machine, one
   process at a time, single-threaded BLAS and Torch.
5. **Reproducibility.** Every artifact is hash-recorded; rerunning the
   analysis from the tracked JSON reproduces every table; the pool
   manifest, identity and environment records name every source.

### What Phase 2 receives

The filtered pools (posterior plus GP per run, with the run's transformer
and log-Jacobian available), the filter records, and the comparison cells
(subsets, weights, raw and capped values, `elbo_mc`). Everything the
honest estimator needs (per-run GP with predictive variance, transformer,
`J_sjk`) is in the saved artifact. The estimator script and report of
Phase 2 stay under `dev/scripts/` and `dev/results/`, with the package
untouched until its result is in, as the optimism note decided. The
script is `dev/scripts/svbmc_honest_elbo.py`: it reads a pool directory
and the comparison's `results.json` and scores every cell, so stage D's
cells are its input as they are; its first run, on the pilot artifacts,
is [results/2026-09-14-svbmc-honest-elbo-pilot.md](../results/2026-09-14-svbmc-honest-elbo-pilot.md).

## Execution

### Phase 1: baseline environment, agreement spread and save-contract probe

**Executor**: Opus sub-agent. Light compute: two short VBMC runs (one
about 1.5 minutes) are the heaviest steps; nothing else runs
concurrently.

**Goal**: establish that the original implementation runs in its recorded
environment against current PyVBMC posteriors, measure the weight
agreement between the two implementations and the within-arm spread, and
confirm that the snapshot codec captures a finished run so that the
recomputation gate of the artifact contract is feasible.

**Steps**:

1. From the repository root (Git Bash), run the integrated S-VBMC tests
   with Torch from the overlay:
   `PYTHONPATH="dev/scripts/runs/svbmc_compat_20260908/deps" .venv/Scripts/python.exe -m pytest pyvbmc/testing/svbmc -q`.
   Expected: all pass, none skipped for a missing Torch. If Torch does not
   import, stop and report the traceback.
2. Agreement and spread, throwaway script in the scratchpad run with
   `PYTHONPATH` set to `BASELINE_PATH` (quoted, `;`-joined). Import
   `svbmc` and `torch`; print `svbmc.__version__` (expected `0.1.1`, via
   the module constant since it is not installed), `svbmc.__file__`
   (under `source/src`), `torch.__version__` (expected `2.14.0+cpu`) and
   `pyvbmc.__file__` (this checkout). For each fixture group in
   `("upstream_GMM_noisy", "upstream_Ring", "bounded_D2", "corr_D3")`
   (unbounded, unbounded, probit-bounded, rotoscale-warped) and each seed
   `s` in 0–4: rebuild the posteriors with
   `vps, metas = pyvbmc.testing.svbmc._fixtures.load_group(group, rng=s)`
   (a 2-tuple); original arm: `np.random.seed(s)` then
   `svbmc.SVBMC(vps).optimize(n_samples=20, lr=0.1, max_steps=3, version="all-weights")`;
   integrated arm, on posteriors loaded again with `rng=s`:
   `pyvbmc.svbmc.SVBMC(vps, seed=s).optimize(n_samples=20, lr=0.1, max_steps=3, n_samples_final=100)`.
   Record per group: the paired `max |Δw|` at each seed, the within-arm
   maximum pairwise `max |Δw|` across the five seeds for each arm, both
   raw ELBOs, and the original's `debiased_I_median` against the
   integrated headline. Report the table; it becomes the first entry of
   the agreement tolerance in criterion 1. If on an unbounded group the
   paired difference exceeds the within-arm spread by more than a factor
   of two, do not tune anything: report which quantities differ and stop.
3. Write `dev/experiments/svbmc_pool/baseline_environment.json`: the
   checkout's `git rev-parse HEAD` (must equal
   `13a78f6c4a3ffe9c4557fee4f4b8f67c98b29e01`) and `git status --porcelain`
   (must be empty), SHA-256 of every file under `source/src/svbmc/`,
   Torch version and path, Python version and executable, NumPy and SciPy
   versions, PyVBMC commit and import path, gpyreg commit (`git -C ../gpyreg rev-parse HEAD`)
   and version, thread environment variables, hostname, and the date.
   Create the directory with a one-paragraph `README.md` naming this
   plan; the README grows in Phase 6.
4. Save-contract probe, throwaway script in the scratchpad, BLAS threads
   set to 1: for the smoke configuration `normal_D2` and seed 0, build the
   problem with `find_config("normal_D2").make(seed=0)`, set options
   `display="off", plot=False, print_iteration_header=False, performance_calibration="off"`,
   run `vp, results = VBMC(*args, options=options, seed=0).optimize()`,
   then
   `snapshot_from_objects(vbmc.vp, vbmc.get_gp(results["best_iter"]), vbmc.function_logger, vbmc._optim_state_record(), vbmc.options, meta={"probe": True})`,
   `save_snapshot`, `load_snapshot`, `build_state`. Check: (a) rebuilt
   `vp.stats` equal to the live stats with `equal_nan=True`; (b) rebuilt
   GP predictions at the logger's live `X` equal the live
   `get_gp(best_iter)` predictions within 1e-10; (c) the recomputation
   gate: `_gp_log_joint(vp, gp, False, True, True, True, True)` on the
   rebuilt pair, last two outputs within 1e-8 of the stored `I_sk` and
   `J_sjk`; (d) `pyvbmc.testing._dtype.assert_float64` on
   `load_bearing_arrays(vp=, gp=, logger=, pt=)` of the rebuilt state.
   Repeat (a)–(d) for `rosenbrock_D2_noise1` seed 0 (a VIQR run; verify
   that the encoded `optim_state` has `active_importance_sampling` absent
   or `None` and that the `.npz` is under 1 MB). Report the outcome of
   every check, the file sizes, and any exception verbatim. If the codec
   raises (an unencodable `optim_state` entry, or `_encode_user_options`
   rejecting an array-valued option), report the key and stop; do not
   extend the codec.

**Verification**:

- [x] Step 1 passes with Torch from the overlay (2026-09-13).
- [x] Step 2 table reported for four groups and five seeds; unbounded
      groups agree within the within-arm spread.
- [x] `baseline_environment.json` written with the fields above; commit
      equals the pin; tree clean.
- [x] Step 4 checks (a)–(d) pass on both configurations, or the mismatch
      is documented.
- [x] Nothing is committed except the new experiments directory.

If any check contradicts an assumption above, stop and report rather
than adapting the contract.

### Phase 2: targets

**Executor**: Opus sub-agent. Light compute (`--check` quadratures, Monte
Carlo ELBO recomputations on fixture posteriors, a two-iteration smoke).

**Goal**: make every pool condition buildable from a clean checkout by
`find_config(label).make(seed)` with a ground truth the metrics can use,
and settle which ring definition the paper's runs used.

**Steps**:

1. In `dev/scripts/benchmark_targets.py`, add registry entries `gmm` and
   `ring` (both require `D == 2`, raise otherwise) whose **log densities**
   reproduce `dev/scripts/runs/svbmc_compat_20260908/source/src/svbmc/targets.py`
   at `13a78f6` pointwise; say so in each docstring with the commit. The
   upstream functions take one point and use the NumPy global stream for
   sampling, so the port is vectorized over `(n, D)` rows and has its own
   samplers. GMM: the 20 component means `DEFAULT_MUS` and covariances
   `DEFAULT_SIGMAS` (unit variances, correlation ±0.5), unnormalized log
   density `logsumexp_k(−½ quad_k − ½ log|Σ_k| − log 2π)` with no
   `−log K`, so `ln Z = log 20` on the unbounded plane; `true_mean` is the
   mean of the component means, `true_cov` the mixture covariance,
   `sampler` picks a component and draws from it with the passed
   generator. Ring: `R = 8`, `σ = 0.1`, centre `(1, −2)`, log density
   `log r − (r − R)² / (2σ²)` with `r` the distance to the centre (clamped
   away from zero as upstream does). Its normalizer in polar coordinates
   is `2π ∫₀^∞ r² exp(−(r − R)²/(2σ²)) dr`: compute `ln Z` with
   `scipy.integrate.quad` and check it against the closed form
   `log(2π σ √(2π) (R² + σ²))` (4.6133 at these parameters; the mass
   below `r = 0` is negligible); `true_mean` is the centre;
   `true_cov = (E[r²]/2) I` with `E[r²] = ∫ r⁴ e(r) dr / ∫ r² e(r) dr`;
   `sampler` draws the radius by inverse CDF on a fine table of the
   radial density `∝ r² e(r)` over `[max(0, R − 10σ), R + 10σ]` and the
   angle uniformly. Upstream's `Ring.sample` draws `r ~ N(R, σ)`, which
   is not the target's radial marginal, so the sampler is a deliberate
   departure. Give each target an independent `reference_logpdf` (GMM
   through `scipy.stats.multivariate_normal`, ring through `np.hypot`).
   Bounds: unbounded. Plausible box `[−10, 10]²` for both, the box the
   upstream runs used: the upstream notebook
   `source/examples/svbmc_example_1_basic_usage.ipynb` sets
   `PLB = [−10, −10]`, `PUB = [10, 10]`, and the fixtures'
   `pt/mu = [0, 0]`, `pt/delta = [20, 20]` agree (their `pt/plb_orig` and
   `pt/pub_orig` are placeholders written by the converter and must not
   be read).
2. Pointwise port check and pins: load the upstream module by file path
   (`importlib.util.spec_from_file_location` on `targets.py`, so the
   `svbmc` package and Torch are not imported) and compare
   `log_density_vec` against `GMM().log_pdf` and `Ring().log_pdf` on a
   21 × 21 grid over `[−12, 12]²` plus 200 random points, to 1e-12
   absolute. Pin three of those values per target into `Problem.pins`
   (`(x, expected, "logp", 1e-10)`) so the check survives in a clean
   checkout; `--check` enforces pins but never verifies `ln_Z`, which
   step 1's quadrature does.
3. Which ring did the paper's runs use? For every posterior of the
   fixture groups `upstream_Ring`, `upstream_GMM` and `upstream_GMM_noisy`
   (`load_group(group, rng=0)`), recompute its ELBO under the ported
   target by Monte Carlo, `mean(log p(x)) − mean(log q(x))` over 100 000
   draws `x = vp.sample(...)` in the original space with `vp.log_pdf`,
   and compare with `stats["elbo"]`. For the ring do it under two
   variants: the upstream definition (with the `log r` term, `ln Z`
   4.613) and the same density without that term (`ln Z` 2.534). VBMC's
   own ELBO standard deviations are at most 0.02 on the ring group, and
   the two variants differ by about `log 8 ≈ 2.1` nats on the ring, so
   the recomputation identifies the variant: the one whose recomputed
   ELBOs sit within about 0.3 nats of the stored values. If the variant
   without `log r` matches, port that definition instead (its normalizer
   is `log(2π σ √(2π) R)`, `true_cov = (E[r²]/2) I` with the radial
   density `∝ r e(r)`), and record in the docstring and the worklog that
   the pinned upstream `targets.py` differs from the target of the
   paper's runs, with the numbers. If neither variant matches within
   0.5 nats for most posteriors, stop and report. For GMM the
   recomputation must match the stored ELBOs within about 0.3 nats.
   Also require every stored ELBO to be at most `ln Z + 3 elbo_sd + 0.5`
   under the chosen definition. Record min, median and max stored ELBO
   per group next to `ln Z` in the worklog (0.31 / 1.15 / 1.25 on the
   ring, 1.61 / 2.29 / 2.70 on GMM, 1.23 / 1.74 / 2.28 on noisy GMM).
4. Add the suite `SUITES["svbmc_pool"]` with the five conditions of the
   table plus the extension condition, every entry
   `Config(name, D, noise_sd=..., tag="svbmc")` with no `_paper_budget`
   (the labels then end in `_svbmc`; the extension entry is present in
   the suite but not in the pool manifest). Then check, in a one-liner,
   that `find_config(label)` returns the pool entry for every pool label
   and that no two configurations in `suite_configs("all")` share a label
   while differing in `options` (`suite_configs("all")` keeps the first
   configuration per label silently).
5. Run `python dev/scripts/benchmark_targets.py --check --only gmm,ring`
   (target names, since the labels carry the suffix) and require it to
   pass, then
   `python dev/scripts/benchmark_targets.py --smoke --suite svbmc_pool`
   (two VBMC iterations per configuration).
6. Update the module docstring, the `dev/README.md` entry for
   `benchmark_targets.py` (eleven synthetic targets, naming `gmm` and
   `ring` and their upstream origin, and the `svbmc_pool` suite). Run
   `pre-commit run --files` on the changed files.

**Verification**:

- [x] Pointwise port check passes; pins added; `--check --only gmm,ring`
      passes.
- [x] Ring variant identified from the ELBO recomputation and recorded;
      no stored ELBO above the truth beyond noise.
- [x] `find_config` returns the pool entries; no shadowed labels;
      `--smoke --suite svbmc_pool` passes.
- [x] `python -m pytest pyvbmc/testing/oracles -q` still passes (the
      `active_sample_step` oracle imports the suite module).
- [x] Docs updated; pre-commit clean.

### Phase 3: pool generator

**Executor**: Opus sub-agent. Light compute except one manual single-seed
run of a noisy condition at the end (about 3 minutes, the only heavy
process at that time).

**Goal**: a resumable, hash-verified generator that writes the per-run
artifact of the contract above and a summary, following
`population_run.py`.

**Steps**:

1. Create `dev/scripts/svbmc_pool_io.py` with:
   `save_run(out_dir, tag, vbmc, results, problem, cfg, seed, timing, meta_extra)`
   building the snapshot exactly as the contract says (`vbmc.vp`,
   `vbmc.get_gp(results["best_iter"])`, `vbmc.function_logger`,
   `vbmc._optim_state_record()`, `vbmc.options`, the listed `meta`
   fields with `profile_run.jsonable` on nested values), calling
   `save_snapshot`, then `verify_run` against the live objects;
   `load_run(path, rng=None)` returning
   `dict(vp, gp, pt, logger, optim_state, options, meta)` through
   `load_snapshot` and `build_state(fun=None, rng=rng)`;
   `verify_run(path, vbmc=None, results=None)` implementing checks
   (a)–(d) of Phase 1 step 4 and, without live objects, checks (c) and
   (d) plus the hash record; `filter_verdict(vp, s_max=np.sqrt(5))`
   returning `dict(stable, max_J_sjk, passes)` with
   `max_J_sjk = float(np.max(vp.stats["J_sjk"]))`.
2. Create `dev/scripts/svbmc_pool_run.py` with subcommands:
   - `prepare --out DIR --suite svbmc_pool [--only LABELS] --target N --max-seeds M --seed-start 1000 [--control-target 50 --control-max-seeds 75]`
     writing `manifest.json`: allocation per condition (`label`,
     `seed_start`, `max_seeds`, `target_filtered`), base options
     (`display="off"`, `plot=False`, `print_iteration_header=False`,
     `performance_calibration="off"`), identity (PyVBMC commit, import
     path and whether `pyvbmc/` or `dev/scripts/benchmark_targets.py`
     has uncommitted changes; the gpyreg source directory from
     `--gpyreg-source DIR` (default the frozen worktree named under "GP
     library pin"), its commit and clean state; NumPy, SciPy, Python
     versions; thread environment; hostname) and the time it was
     written. Whether a campaign may start is a decision recorded in this
     plan, not a manifest field: the manifest carries provenance only.
     This identity is a deliberate relaxation of
     `population_run.identity()`, which demands a frozen PyVBMC checkout
     as well; gpyreg is pinned the same way as there.
   - `run --out DIR [--pilot-seeds K] [--save-vbmc]`: refuse when the
     package directory or the suite module
     has uncommitted changes; set `PYVBMC_GPYREG_SOURCE` to the
     manifest's gpyreg source for itself and every child, prepend it to
     `sys.path` before importing PyVBMC, and refuse when
     `gpyreg.__file__` resolves elsewhere; `FileLock` on `campaign.lock`;
     identity must equal the manifest's; BLAS thread variables set to 1
     and `MPLBACKEND=Agg` in the child environment; Windows idle-sleep
     prevention as in `population_run.py`; for each condition in
     manifest order, seeds from `seed_start` upward; skip a seed whose
     `records/<tag>.complete.json` re-verifies (hashes and identity);
     stop the condition when the filtered count reaches
     `target_filtered` or `max_seeds` seeds have run; with
     `--pilot-seeds K`, run exactly the first `K` seeds of every
     condition instead. One worker subprocess per case
     (`sys.executable -u <this file> worker ...`), stdout and stderr to
     `<tag>.log`, `START`/`DONE`/`FAILED` lines with `flush=True`,
     `status.json` rewritten before and after every case (active case,
     per-condition counts, totals, timestamps); a failing case writes
     `<tag>.error.txt` and the sweep continues; a partial artifact
     without a valid record stops the sweep for inspection.
   - `worker --out DIR --label L --seed S [--save-vbmc]`: identity check;
     `cfg = find_config(label)`; `prob = cfg.make(seed=seed)`; options from
     `prob.vbmc_args()` updated with the base options; `VBMC(*args,
     options=options, seed=seed)`; `vp, results = vbmc.optimize()` timed
     with `perf_counter`; `metrics(prob, vp, results["elbo"])`;
     `save_run`; `filter_verdict`; `records/<tag>.complete.json` with
     hashes, elapsed seconds, identity, verdict and metrics; with
     `--save-vbmc`, `vbmc.save(out / f"{tag}.vbmc.pkl")`.
   - `summarize --out DIR`: `summary.json` and `summary.md` per condition:
     seeds run, filtered count, pass rate, failure reasons, wall median
     and IQR, evaluation-count median, `K`, metric medians and IQRs, and
     the usable fraction under the house thresholds.
3. Create `dev/scripts/test_svbmc_pool_run.py` (explicit path, outside
   default discovery, modelled on `dev/scripts/test_population_run.py`):
   a manifest for `normal_D2` with `target 2`, `max_seeds 3`, run end to
   end into a temporary directory (about one minute); assert both
   artifact files and the record exist per run, `load_run` returns
   objects whose `vp.stats["I_sk"]` matches the sidecar, `verify_run`
   passes post hoc, a second `run` skips every completed case,
   `summarize` reports the right counts, and a corrupted record (edit
   one hash) makes the second `run` stop for inspection rather than
   rerun silently.
4. Manual gate, the one heavy process: `prepare` into a scratch
   directory for `rosenbrock_D2_noise3_svbmc` with
   `--max-seeds 1 --target 1`, `run`, `summarize`; then
   `verify_run` post hoc on the artifact; then, with `PYTHONPATH` set to
   `TORCH_PATH`, rebuild the posterior with `load_run(path, rng=0)` and
   construct `pyvbmc.svbmc.SVBMC([vp, vp], noisy=True)` to confirm the
   rebuilt posterior passes `_validate_posteriors` (two copies suffice).
5. Add both scripts to the `dev/README.md` scripts list (one bullet each,
   in the style of the existing entries) and run pre-commit.

**Verification**:

- [x] `python -m pytest dev/scripts/test_svbmc_pool_run.py -vv` passes.
- [x] Manual gate: artifact under 1 MB, `verify_run` passes post hoc,
      rebuilt posterior accepted by the integrated class.
- [x] `git status` shows only the new scripts, the test and the README
      change.

If the codec rejects a live object, report the key and stop; do not
filter the state ad hoc.

### Phase 4: pilot

**Executor**: Fable (orchestrator), holding the single heavy slot in the
main thread.

**Goal**: measured wall times and filter pass rates on the pilot machine for
every condition, stacking times of both arms at `M = 3`, artifacts proven
to reload and stack in both implementations, and a fixed allocation for
the pools and the comparison grid.

**Constraints**: three seeds per condition (`--pilot-seeds 3`), one
process, BLAS single-threaded, `--save-vbmc` on; the pool runs and the
stacking timings are taken with nothing else computing (Phase 5's test
and dry run wait until they finish); timings under other load are
informational only. Nothing in the package changes.

**Failure path**: a run that errors is recorded and the pilot continues;
an artifact that fails `verify_run` stops the pilot for investigation
(the contract, not the run, is in question); a condition whose three
seeds all fail the filters is reported with the reasons and its seed cap
and pass-rate assumption are revised before stage B.

**Acceptance**:

- [x] 15 artifacts (three seeds of five conditions) verify post hoc
      (2026-09-14, exact zeros on the recomputation gate, 0.2 s in all).
- [x] For every condition, both arms ran a full `optimize` at `M = 3`
      from the rebuilt posteriors, five cell seeds each; the paired
      `max |Δw|`, the within-arm spreads and both arms' seconds are
      recorded under criterion 1 and in the cost paragraph
      (`dev/scripts/runs/svbmc_pool_20260913/pilot_stack/`, copied to
      `dev/experiments/svbmc_pool/pilot/`).
- [x] The pool runtime table and the comparison cost paragraph carry
      the measured values; every pilot run passed the filters, so the
      seed caps and the stage split stand and no allocation revision is
      needed. A revised allocation, if one is ever needed, is applied to
      the same campaign directory by running `prepare` again with the
      new `--allocation` values (the manifest accepts a change of the
      allocation alone and records the previous one in
      `allocation_history`); the pilot's artifacts are the first runs of
      stages B and C.
- [x] The PI authorizes stage B, stage C and the comparison grid in a
      dated worklog entry before any further pool run starts (stages B
      and C became the cluster pool of decision 8, generated 2026-09-14;
      the yardstick is decision 7; stage D was authorized on 2026-09-14
      on the decision-6 grid, see the worklog; `M = 32` for the
      integrated arm alone remains a separate decision, since the
      harness runs both arms on one grid).

### Phase 5: stacking comparison harness

**Executor**: Opus sub-agent. Light compute; authoring proceeds alongside
Phase 4, its test and dry run wait until the pilot's timed runs finish.

**Goal**: `dev/scripts/svbmc_pool_stack.py`, the matched comparison of the
design above, tested on the upstream fixture groups.

**Steps**:

1. Add to `dev/scripts/benchmark_targets.py` a function
   `sample_metrics(problem, samples, elbos)` returning `mmtv`, `gskl`
   (house convention), `gskl_normalized = gskl / D`, and
   `elbo_err_<name>` for every entry of the `elbos` dict, plus the sample
   moments; MMTV mirrors `VariationalPosterior.mtv` (2^13-point `kde_1d`
   per dimension, ranges from the samples widened by a tenth and clipped
   to the problem's bounds) between `samples` and
   `problem.sampler(100_000, rng)` with a dedicated generator.
2. Create `dev/scripts/svbmc_pool_stack.py`:
   - inputs: one or more pool directories, `--conditions`, `--M` grid
     (default `2,4,8,16`), `--repetitions` per `M` (default
     `20,20,20,10`), `--seed 0`, `--max-steps 500`, `--out DIR`,
     `--overwrite`; refuses to start unless
     `baseline_environment.json` re-verifies (commit, clean tree,
     file hashes, Torch version);
   - for each condition, the filtered runs ordered by seed; subsets from
     `np.random.default_rng([seed, condition_index, M, r]).choice(n, M, replace=False)`;
     `cell_seed` derived from the same tuple;
   - integrated arm in-process (controller started with `PYTHONPATH`
     = `TORCH_PATH`): posteriors rebuilt for the cell with
     `load_run(path, rng=entry_seed)`, one seed per entry from
     `SeedSequence(cell_seed).spawn(M)`, `SVBMC(vps, seed=cell_seed)`,
     `optimize(n_samples=20, lr=0.1, max_steps=..., version="all-weights", n_samples_final=100)`;
     record `w`, `elbo`, `elbo_sd`, `elbo_details`, `entropy`,
     construction and optimization seconds, `sample_metrics` of
     `sample(100_000)`, and `e_log_joint_mc` / `elbo_mc` from 10 000 of
     those draws through the noiseless `problem.log_density_vec`;
   - original arm in a long-lived worker subprocess whose environment
     sets `PYTHONPATH` to `BASELINE_PATH` (`os.pathsep.join`), receiving
     the artifact paths, the subset indices and `cell_seed`; the worker
     rebuilds the posteriors with the same per-entry seeds, calls
     `np.random.seed(cell_seed)`, then
     `svbmc.SVBMC(vps, s_max=np.sqrt(5), M_min=2/3)` and
     `optimize(n_samples=20, lr=0.1, max_steps=..., version="all-weights")`;
     records `w`, the three `elbo` entries, `entropy`, seconds,
     `sample_metrics` of `sample(100_000)` (approximately that many rows)
     and `e_log_joint_mc` / `elbo_mc` the same way;
   - the two arms never run at the same time; alternate which arm runs
     first in every cell;
   - `M = 1` rows from the pool sidecars' metrics;
   - outputs `results.json` (every cell), `summary.json` and `summary.md`
     (per condition and `M`: medians with 10 000-resample bootstrap 95 %
     intervals of the median, paired differences with exact signed-rank
     p-values Holm-corrected across all condition-and-`M` cells at
     α = 0.05 for MMTV and gsKL, the bias of every ELBO variant relative
     to `elbo_mc` with the criterion 3 gate (the absolute median bias of
     the integrated headline at most that of the original's raw estimate
     at every `M`, headline-bias growth below 0.5 nats) and the KL gap,
     `max |Δw|`, runtime ratio with its interval,
     per-condition aggregates over all `M` of the runtime ratio and
     `max |Δw|`, and the `M = 1` medians), and `sources` (both trees'
     commits, working-tree state, the suite module's hash, import paths,
     versions, thread settings, the baseline environment record, the
     harness's own SHA-256). A `--summarize-only` mode rebuilds the
     summaries from an existing `results.json` without running a cell,
     which is what criterion 5 requires.
3. Create `dev/scripts/test_svbmc_pool_stack.py`: run the harness on the
   fixture groups `upstream_GMM_noisy` and `upstream_Ring` (through a
   small adapter that presents fixtures as a pool) with `--M 2,3
   --repetitions 2,2 --max-steps 3`; assert the output schema, that both
   arms produced every cell, and that `max |Δw|` is within the Phase 1
   spread for those groups. Skip cleanly when Torch is not importable.
4. Dry run on the pilot artifacts (`--M 2,3 --repetitions 2,2`) once the
   pilot's timed runs are done; report the summary table in the worklog.
5. Add the script to the `dev/README.md` list; pre-commit.

**Verification**:

- [x] `PYTHONPATH="<TORCH_PATH>" .venv/Scripts/python.exe -m pytest dev/scripts/test_svbmc_pool_stack.py -vv` passes.
- [x] Dry run on the pilot completes with the two arms agreeing within
      the recorded spread (run as the pilot's `M = 3` measurement,
      25 cells; worklog of 2026-09-14).

### Phase 6: campaign, comparison and report

**Executor**: Fable (orchestrator) for the runs (single heavy slot) and
the assessment; Opus sub-agents for the report drafting and the
experiments README if delegated.

**Goal**: the authorized pools generated and summarized, the comparison
run against the acceptance criteria, and the results recorded.

**Work**:

- Stage B then stage C with `svbmc_pool_run.py run`, each on explicit PI
  instruction, resuming from the pilot directory; `summarize` after
  each; copy `manifest.json`, `summary.json`, `summary.md` and the
  records index into `dev/experiments/svbmc_pool/`.
- Stage D: `svbmc_pool_stack.py` over the filtered pools with the
  authorized grid; copy `results.json`, `summary.json`, `summary.md` and
  `sources` into `dev/experiments/svbmc_pool/`.
- Assess criteria 1–5; write `dev/results/<date>-svbmc-pool-comparison.md`
  (lead paragraph with the headline and links, comparison design,
  results tables, agreement, runtime, limitations); complete
  `dev/experiments/svbmc_pool/README.md` (every key of every JSON, the
  exact reproduction commands, the raw location under
  `dev/scripts/runs/`).
- Update the documents listed below and this plan's status and worklog.

**Acceptance**:

- [x] Every filtered pool reaches its target or its seed cap, with the
      shortfall recorded (every condition at its target, no
      shortfall; worklog of 2026-09-14).
- [x] Criteria 1–5 assessed with the numbers in the report; any failure
      is stated as such (criterion 3 fails on Student D8; the report
      and the worklog of 2026-09-15). The `M = 32` cells, an
      extension of the grid for the integrated arm, are pending.
- [x] Phase 2 hand-off paragraph in the worklog names the pool directory,
      the filtered counts and the comparison cells (the Phase 2
      scoring itself ran on the cells; worklog of 2026-09-15).

## Documentation

- `dev/TODO.md`: the "S-VBMC benchmark campaign" item links to this plan
  and drops "remain to be designed" once the design is approved.
- `dev/plans/svbmc-integration.md`, section "Benchmark campaign required
  for 1.5": one sentence pointing here as the owner of the design and
  execution.
- `dev/2026-09-12-svbmc-elbo-optimism.md`, "Where and on what": one
  sentence pointing here for the pools and the `elbo_mc` cells.
- `dev/plans/modernization-roadmap.md`: the S-VBMC workstream entry
  names this plan.
- `dev/README.md`: the plans index lists this plan; the scripts list
  gains `svbmc_pool_run.py`, `svbmc_pool_io.py` and
  `svbmc_pool_stack.py`; the `benchmark_targets.py` entry names the two
  new targets and the `svbmc_pool` suite.
- `dev/experiments/svbmc_pool/README.md` (new): indexes the
  machine-readable evidence and owns the description of the comparison's
  outputs and of the baseline environment; each generated
  pool's subdirectory (`pool_<date>/`) carries its own README with every
  key of its files, the commands that generated it and its archive's
  name, size and SHA-256.
- `dev/scripts/hpc/README.md` (new, with the cluster PR): owns how the
  pool is generated on a Slurm cluster with the scripts beside it
  (submit, array task, finish), the identity rule that freezes the
  checkout while the array runs, and the reconciliation of missing cases
  with `svbmc_pool_run.py verify`.
- `dev/results/<date>-svbmc-pool-comparison.md` (new, Phase 6): owns the
  reading of the comparison numbers.
- Not needed: `MANIFEST.in` (nothing under `dev/` ships; the packages are
  enumerated in `pyproject.toml`) and `dev/golden/README.md` (the pools
  are not golden traces).
- The Phase 2 report is not part of this campaign.

## Decisions

Choices made while drafting that could reasonably have gone the other
way and would be costly to reverse.

- **Generate fresh pools rather than reuse retained captures, on every
  axis.** The captures carry no GP, predate the recorded noise level,
  and were run at an older code state under the pinned budget; and the
  comparison must run on the same inputs Phase 2 uses so that raw,
  capped, honest and Monte Carlo values land on the same cells.
  Rejected: rebuilding GPs from the golden traces next to the capture
  posteriors (zero VBMC compute, but a reconstruction audit and an
  older code state); rejected: stacking the captures for the
  posterior-quality axis alone (different inputs from Phase 2, noise
  level inferred by proxy).
- **Plain-array snapshots, not pickled `VBMC` objects.** The oracle codec
  already captures posterior, GP, transformer, logger and state
  independently of class layout, rebuilds through public constructors,
  and is exercised by the oracle tests; the shipped `.pkl` fixtures show
  how renamed attributes break pickles. `VBMC.save` is used only for the
  pilot runs to prototype the Phase 2 interface.
- **The saved GP is `get_gp(best_iter)`, not the last iteration's
  `vbmc.gp`.** It is the GP behind the returned statistics in both boost
  branches, and the recomputation gate checks it.
- **GMM and ring are ported into the developer benchmark suite.** The
  integration decision kept the toy targets out of the package, which
  this respects; importing them from the gitignored checkout would make
  the pool irreproducible from a clean clone. Log densities are
  reproduced pointwise against the pinned source and pinned; samplers
  are the suite's own.
- **Seeds disjoint from the golden population.** A cross-check against
  the golden traces would be weak (the guarded-sinh change moved one of
  the 18 replayed trajectories, and the budget convention may differ),
  and disjoint ranges keep the populations from being conflated.
- **Matched subsets, fresh posteriors and one cell seed per cell for both
  arms.** Paired cells turn the comparison into differences on identical
  inputs; the original's use of the global NumPy stream and of the input
  posteriors' generators is contained by seeding and rebuilding per cell.
- **Criterion 1 is an equivalence within measured spread, not a fixed
  tolerance.** The Phase 1 corrections intentionally move weights on
  bounded targets; a fixed tolerance measured on unbounded fixtures
  would stop the campaign on multisensory by design.
- **House gsKL convention as the gate, normalized variant stored.** The
  repository's analyses all use the 2020 convention with threshold 1;
  the paper's variant is kept for paper-comparable figures only.
- **One artifact contract for both consumers.** The comparison runs on
  the pools Phase 2 uses, so raw, capped, honest and Monte Carlo values
  land on the same cells; the `elbo_mc` field is recorded during the
  comparison because the draws are already in hand.
- **Cost drivers named and bounded.** The cell cost grows as `M²` and a
  D = 6 stack costs about five D = 2 stacks; the grid, the pool sizes and
  the condition set were sized together (see the PI decisions) so the
  comparison stays near 5 hours.
- **The plan lives in `dev/plans/`.** The repository convention keeps
  execution plans there, slug-named; the reporting plan already records
  executor-labelled phases in this form.

## PI decisions (2026-09-13)

The design above was approved with these choices, each of which the
draft had left open:

1. **Evaluation budget**: PyVBMC defaults (75 (D + 2) for noisy targets),
   the S-VBMC paper's convention; every pool configuration is a tagged
   suite entry. Rejected: the golden suite's pinned 50 (D + 2), which
   would have made conditions 2 and 3 directly comparable with the golden
   single-run baseline at the price of a budget users never see.
2. **Condition set**: one multisensory noise level (3, paper parity);
   the noise-1.3 condition is an extension, since its only unique
   contribution is a second point on the noise scaling of the optimism.
   The noiseless GMM control stays.
3. **Pool sizes**: 60 filtered runs per noisy condition, 40 for the ring
   (the most expensive condition), 30 for the control; `M` capped at 16.
   Rejected: the paper's 100, about a third more pool compute for `M` up
   to 40.
4. **Compute**: staged on the PI's laptop, each stage on a separate
   instruction; the per-case worker stays reusable for a Slurm array
   job. Rejected: waiting for the Slurm support item.
5. **Phase 5 timing**: the comparison harness is built now, alongside
   the pilot, since it is needed whatever Phase 2 decides.
6. **Comparison grid**: `M ∈ {2, 4, 8, 16}` with `R = 20, 20, 20, 10`,
   both arms on every cell, about 5 hours from the paper's timings.
   Rejected: the paper's `2–40` grid with 20 repetitions (about 100
   hours), whose cost sits in the `M ≥ 32` cells that the pool size no
   longer supports.
7. **Evidence yardstick (2026-09-14, after the pilot)**: every reported
   ELBO is scored by its bias relative to the stacked posterior's own
   Monte Carlo ELBO, whose entropy term the harness estimates
   independently of both arms; the error against `ln Z` is descriptive
   only, and the KL gap `ln Z − ELBO(q)` is reported as the
   posterior-quality measure in evidence units. Rejected: the draft's
   error against `ln Z` as the gate, which mixes estimator bias with the
   stack's KL gap and can rank a more optimistic estimator higher by
   cancellation.
8. **Pools on the cluster, analyses on the PI's laptop (2026-09-14)**: pool
   generation is a once-in-a-while golden-fixture job handed to another
   developer for the HPC cluster; everything that consumes the pools
   (the two-arm comparison, the Phase 2 estimator study, any later
   regression check) must fit an overnight laptop run of 8–10 hours.
   Consequently the pool returns to the paper's 100 filtered runs per
   noisy condition with `M` up to 40, and the condition set is the eight
   of the table (decision 2's one-level, five-condition allocation is
   superseded). Rejected: adding conditions merely because the cluster
   can afford them; each of the six left out is named with its reason.
9. **Source identity, not host identity (2026-09-14)**: a pool must be
   generated by one code and library state, so the worker still refuses
   a case whose PyVBMC or gpyreg commit, harness module hashes or library
   versions differ from the manifest's, but hostname, interpreter path,
   thread settings and platform are recorded without being compared, so
   any cluster node may run any case. Rejected: dropping the refusal and
   only flagging mixed identities in the summary (a mixed pool is a
   corrupted fixture, and the refusal is cheap).
10. **gpyreg 1.2.1 everywhere (2026-09-14)**: the campaign pin, the CI
    pin and PyVBMC's minimum version all move to the released 1.2.1;
    the `acq_AcqFcnVIQR` oracle reference is re-baselined to it (one
    ulp). Rejected: staying on 1.2.0 for the pool while the package
    moves on, which would only invite a forgotten mismatch.
11. **The headline's yardstick is the bias of the inputs (2026-09-15,
    after stage D)**: S-VBMC is not asked to remove the optimism
    VBMC's own optimization builds into each run's ELBO, only not to
    add to it, so a stacked estimate is judged by its bias relative to the
    mean bias of the runs it was built from, each run scored against
    its own `elbo_mc` as a stack of one (`svbmc_single_run_bias.py`),
    and a correction by the part of the stack's bias it removes beyond
    the inputs' own. Debiasing the VBMC ELBO itself is a separate
    question (`dev/TODO.md`, outside 1.5). Rejected: zero bias as the
    target for the stacked headline, which criterion 3 and the day's
    analyses had used without saying so, and which would put the
    stacked ELBO below the ELBOs of its own inputs.
12. **The headline switch is confirmed at the release gate, on fresh
    pools (2026-09-16, after the `M = 32` run)**: the PI leans to
    promoting the two-level shrinkage to the headline of a noisy stack
    (the noiseless case open), but the switch is confirmed, and the
    user-facing wording written, at the final large-scale check before
    the release of 1.5: VBMC pools regenerated on the cluster by the
    consolidated release code (about 100 runs per condition, several
    hundred where `M = 32` is to be scored), S-VBMC at several `M`,
    and this campaign's results holding. That check gates 1.5 as a
    whole. The present pools (100 noisy and 50 noiseless runs per
    condition) reuse each run 1.6 to 3.2 times at `M = 16` and 3.2 to
    6.4 times at `M = 32`, so the `M = 32` cells of a control are
    closer to one measurement than to ten and the `M = 32` control
    observation is a direction to confirm, not an interval. Rejected:
    switching the headline on this campaign's evidence alone.
13. **Package the candidate before selecting the headline (2026-09-16):**
    implement the full-covariance two-level estimate as
    `elbo_details["shrunk_two_level"]` for both noise statuses, alongside
    a noise-share diagnostic, under the
    [estimator integration plan](svbmc-shrinkage-estimator.md). Record it
    in subsequent campaign cells and summaries while preserving the
    current headline and its acceptance criteria. Promote it to headline
    after the final campaign only if that comparison confirms it is best.
    The inputs remain fitted posteriors with their existing statistics;
    this estimator needs no GP or full VBMC object, and the objects
    returned by `VBMC.optimize()` remain unchanged.
14. **The release gate's pools and stacking (2026-09-25)**: the release
    pools hold 320 filtered runs per condition (seed caps 350, and 480 for
    the ring; seeds from 1000, continuing this campaign's range), which
    gives ten disjoint subsets at `M = 32`. Every `M` is stacked in
    disjoint subsets: repetition `r` takes the `r`-th block of `M` runs of
    a permutation drawn for the condition and `M`, with the repetition
    counts of decision 6 and of the `M` = 3, 5 and 32 runs. Both arms run
    again at `M` = 2, 4, 8 and 16, since the release pools make new cells,
    and the integrated arm alone at 3, 5 and 32. Generation, stacking and
    the analyses of the gate all run on the cluster, through the
    [Slurm plan](slurm-benchmark-support.md). This supersedes decision 8
    where it keeps everything that consumes a pool within an overnight
    laptop run, and decision 12's allocation of about 100 runs per
    condition. Rejected: stacking the release pools on the laptop, which
    their size and the rerun of the original arm would stretch well past a
    night.

## Risks and rollback

- Noise-3 multisensory or ring runs may pass the filters rarely; the
  seed caps bound the cost, and the shortfall is recorded rather than
  chased.
- A snapshot that does not reproduce the stored statistics would mean
  the GP behind the posterior is not the one the contract assumes; Phase
  1 tests this before any harness is written.
- The original implementation prints warnings and draws from the global
  NumPy stream and the input posteriors' generators; the harness seeds
  the stream and rebuilds the posteriors per cell and never runs both
  arms at once.
- The comparison's cost estimate rests on the paper's hardware; the
  pilot's `M = 3` timings and the `M²` extrapolation replace it before
  stage D is authorized.
- Rollback: the harness lives in three new scripts and one suite entry;
  removing them and the `gmm`/`ring` registry entries restores the tree.
  Pool artifacts are gitignored.

## Worklog

- 2026-09-13: plan drafted after an inventory of retained artifacts
  (population captures, boost-campaign dills, golden traces, fixtures,
  the pinned upstream checkout), the paper protocols and the campaign
  conventions, then revised after a two-reviewer check: the randomness
  of both implementations is NumPy's, not Torch's; the agreement
  criterion became an equivalence within measured spread because the
  Phase 1 corrections move weights on bounded targets by design; the
  comparison grid was sized (`M²` cost) and reduced; the ring's
  normalizer under the upstream definition is 4.613 against a paper
  figure reading near 2.25, to be settled by ELBO recomputation on the
  upstream posteriors.
- 2026-09-13: design approved by the PI with the six decisions recorded
  above (default budget, one multisensory level, pools of 60/40/30, this
  laptop in stages, comparison harness now, grid `{2, 4, 8, 16}`).
  Harness implementation may start; the pilot waits for its own go.
- 2026-09-13: the PI authorized the pilot to start once Phases 1–3 are
  complete and verified. Phase 1 complete: 192 integrated S-VBMC tests
  pass with Torch from the overlay; the original runs at the pin against
  current posteriors; agreement and spread measured (criterion 1);
  `baseline_environment.json` written; the snapshot probe on `normal_D2`
  and the VIQR run `rosenbrock_D2_noise1` round-trips with exact zeros
  on the GP predictions and the recomputation gate, 35 KB and 61 KB per
  artifact, `active_importance_sampling` present as `None` on the noisy
  run and absent on the noiseless one. The sibling gpyreg checkout was
  found on an in-progress branch with uncommitted GP-core changes, so
  the campaign pins gpyreg to a frozen worktree at the CI pin (section
  "GP library pin").
- 2026-09-13: Phase 2 complete. `gmm` and `ring` are in the suite with
  pins; the GMM log density is bit-identical to the upstream module on
  641 points, `ln Z = log 20`. The ring question of Phase 2 step 3 is
  settled: recomputing every upstream ring posterior's ELBO by Monte
  Carlo reproduces the stored values within 0.23–0.24 nats under the
  density **without** the `log r` term and misses by 2.30–2.32 nats
  (`log R` exactly, the posteriors sit on the ridge) with it, against
  stored `elbo_sd` at most 0.02; the noiseless GMM control reproduces to
  a median of 0.011 nats. The ported ring therefore omits the term,
  `ln Z = 2.5337` (quadrature and closed form agree to 1e-15),
  `true_cov = 32.015 I`, and the `_ring` docstring records that the
  pinned upstream `targets.py` differs from the target of the paper's
  runs. `--check` passes for both targets, `--smoke --suite svbmc_pool`
  passes for all six entries, the oracle suite still passes (143 passed,
  the platform-bound oracles included), `_mixture_moments` now accepts
  full covariances (the `lumpy` check is unchanged), and no stored
  upstream ELBO exceeds its truth beyond noise. Phase 3 was interrupted
  by the usage limit while reading files and resumed at 02:31 on
  2026-09-14.
- 2026-09-14: Phase 3 complete. `svbmc_pool_io.py`, `svbmc_pool_run.py`
  and `test_svbmc_pool_run.py` are in place (16 tests, 35 s, on
  `normal_D2`); the manual gate on `rosenbrock_D2_noise3_svbmc` seed 1000
  took 1.6 min, 170 evaluations, `K = 50`, passes the filters
  (`max_J_sjk` 0.66), 73 KB per artifact, `verify_run` post hoc with
  exact zeros on the recomputation gate, and two rebuilt copies stack in
  the integrated class as a noisy pair; `--save-vbmc` writes a 742 KB
  pickle for a small run. Every worker imports gpyreg from the frozen
  worktree (`sys.path.insert(1, …)` suffices because the editable
  finder appends itself to `sys.meta_path`). Departures from the Phase 3
  text, all recorded in the scripts' docstrings: `prepare` has
  `--allow-dirty` (used only by the test and the gate while the tree
  carried uncommitted work; a campaign manifest is prepared from a
  committed tree without it) and a repeatable `--allocation LABEL=T/M`
  for per-condition pool sizes; the identity also hashes the suite
  module; metadata goes through the codec's own `encode` because a
  `results` field is NaN and the sidecar is written with
  `allow_nan=False`; a failed case leaves `<tag>.error.txt` and a resume
  continues past it, while a partial artifact without a record or error
  file stops the sweep. `prepare --suite svbmc_pool` must be given
  `--only` with the five pool labels, since the extension condition is
  a suite entry. Note for harness code: the integrated class is imported
  as `from pyvbmc.svbmc import SVBMC`; `pyvbmc.svbmc` is not an
  attribute of the top-level package.
- 2026-09-14: Phase 5 complete except the dry run on pilot artifacts.
  `sample_metrics` sits next to `metrics` in the suite module and agrees
  with it to Monte Carlo noise on fixture posteriors; `svbmc_pool_stack.py`
  takes `--pool DIR` or `--fixtures GROUP` (the fixture adapter lives in
  the harness because the original arm's subprocess must load the same
  entries; groups map to the ported targets), re-verifies the baseline
  record before any cell, warms both arms with two discarded Adam steps,
  runs the arms one at a time alternating order, and writes
  `results.json`, `summary.json`, `summary.md`, `sources.json`,
  `cells.jsonl` and `original_arm.log`. Its 7 tests pass in 18 s on the
  upstream GMM-noisy and ring groups at `M = 2, 3`, three Adam steps:
  paired `max |Δw|` at most 0.015, both arms improving on the `M = 1`
  medians, both arms importing gpyreg from the frozen worktree. A dry
  run on the two-run `normal_D2` pool of Phase 3's test exercised the
  `--pool` path end to end (`noise_status_source` recorded). Guards
  checked: a tampered baseline record and an upstream path on the
  controller are refused; an `M` above the filtered pool is skipped and
  recorded.
- 2026-09-14: doublecheck of Phases 1, 2, 3 and 5 by three fresh
  reviewers, every finding fixed and re-verified (29 pool-generator
  tests, 11 comparison tests, the target checks, 143 oracle tests and 192
  S-VBMC tests pass). Corrections of substance: the comparison's Monte
  Carlo log joint took the first 10 000 draws, which for the original
  arm (unshuffled per-run blocks) measured one run rather than the stack;
  it now takes a seeded random subsample in both arms. The fixture
  adapter matched groups by name prefix and would have merged the noisy
  GMM fixtures into `upstream_GMM`; it now reads the sidecar group.
  Every posterior of a cell was rebuilt with the same seed, coupling the
  original arm's sample blocks; each entry now gets its own seed spawned
  from the cell seed (the Randomness paragraph records this). `prepare`
  could allocate the extension condition by default; it now allocates
  exactly the five pool labels unless `--only` names the extension. A
  stale `<tag>.log` blocked resumption; only artifact suffixes count as
  partial now. The allocation may be revised in place with a recorded
  history (Phase 4 needs this). The identity pins the io and runner
  modules as well as the suite module; the comparison records
  working-tree state for both arms and checks the controller's gpyreg at
  start. Criterion 2's exact signed-rank tests with Holm correction and
  criterion 5's `--summarize-only` mode exist. The ring's `--check` now
  uses an independent reference route and machine-checks the three
  upstream `log_pdf` values (the port plus `log r`), and the S-VBMC
  paper's author list was corrected in two citations. `sample_metrics`
  and `metrics` now draw the same reference set by default.
  `baseline_environment.json`'s gpyreg note states the pin. Phase 4 was
  cleared to start once the tree is committed.
- 2026-09-14: Phase 4, the pilot, complete (harness commit `e2aaef5`,
  campaign directory `dev/scripts/runs/svbmc_pool_20260913/pool/`,
  manifest authorized "Luigi Acerbi (PI, 2026-09-13)"). Fifteen runs in
  27 minutes, every one passing the filters, every artifact verifying
  post hoc with exact zeros; per-run wall times and single-run metrics
  are in the pool runtime table, the projected pool cost is about 8
  hours (12.5 at the seed caps). The `M = 3` stacking measurement
  (`pilot_stack/`, 25 cells, five cell seeds per condition, 500 Adam
  steps): runtime ratios 0.20–0.51, paired `max |Δw|` within 1.3 times
  the within-arm spread, no equivalence flag on MMTV or gsKL, comparison
  grid cost about 3 hours. Two observations for the PI's decision before
  stage D: (1) **criterion 3's yardstick.** Its first clause compares
  every estimate with `ln Z`, but a stack of three runs has a genuine
  KL gap to the target (the Monte Carlo ELBO of the stacked posterior,
  `elbo_mc`, sits 0.55 nats below `ln Z` on noisy GMM and 0.6–0.7 on
  multisensory), and the raw estimates are already optimistic by
  0.1–0.7 nats relative to `elbo_mc` at `M = 3` (largest on noisy GMM
  and multisensory), so the raw error against `ln Z` is small by
  cancellation while the capped headline, which removes most of the
  optimism, lands 0.4–0.9 nats below `ln Z`. Measured against `elbo_mc`,
  the capped headline is closer than the raw estimate on all four noisy
  conditions (bias of at most ±0.33 nats against +0.11 to +0.67). The
  clause as written would fail at `M = 3` on three noisy conditions for
  the wrong reason. Proposed amendment: state criterion 3 as the bias of
  each estimate relative to `elbo_mc` (the quantity Phase 2 compares
  against), require `|headline − elbo_mc| ≤ |raw − elbo_mc|` in the
  median at every `M`, keep the growth bound on the capped bias, and
  add `bias_<variant> = elbo_<variant> − elbo_mc` columns to the
  comparison summary (a small harness change, to be reviewed before
  stage D). (2) **Grid.** The measured cost leaves room for `M = 32` at
  `R = 10` (about an hour per condition more) if the paper's larger-`M`
  regime is wanted; the pool of 60 supports it with less subset
  diversity than the paper's 100. Stages B, C and D await the PI's
  instruction.
- 2026-09-14: the PI accepted the three-part proposal on the evidence
  yardstick (decision 7): criterion 3 restated as bias relative to the
  stack's Monte Carlo ELBO, that reference hardened with an
  arm-independent entropy estimate and Monte Carlo standard deviations,
  and the KL gap reported as its own column. Implemented the same day:
  the reference entropy is the integrated class's `stacked_entropy` at
  each arm's final weights, four batches of 50 draws per component from
  a generator seeded by the cell, its standard error from the batch
  spread; `e_log_joint_mc` carries its own standard error; every
  `bias_<variant>`, `kl_gap` and the two gates are in the cell records
  and the summaries, and `--summarize-only` reproduces them. The pilot
  measurement was regenerated with the final harness (25 cells, 1.9
  min; eight batches of 25 reference draws shared by both arms). At
  `M = 3`, medians over five cells: headline bias 0.03 (Rosenbrock),
  0.02 (noiseless GMM, no cap active), −0.30 (noisy GMM), 0.00 (ring),
  0.27 (multisensory) against raw biases of 0.13, 0.02, 0.69, 0.12, 0.55
  for the integrated class and 0.20, 0.06, 0.79, 0.17, 0.63 for the
  original's raw estimate; `headline_bias_not_worse` holds on every
  condition with the paired difference's interval below zero; `elbo_mc`
  standard errors are 0.01–0.02 nats. The KL gaps of the three-run
  stacks are 0.03 (Rosenbrock), 0.33 (noiseless GMM), 0.56 (noisy GMM),
  1.19 (ring) and 0.67 (multisensory). The reference confirms the reason
  for the change:
  the original's own reported entropy sits 0.02–0.08 nats above the
  arm-independent estimate on every condition, the integrated class's
  fresh evaluation within ±0.015. Stage D note: the entropy machinery
  allocates a `(K_total · draws, K_total)` array, about 256 MB per
  reference batch and about 512 MB for the class's own 100-draw final
  evaluation at `M = 16`; to be checked against the machine's memory
  before stage D. Stages B, C and D remain unauthorized.
- 2026-09-14: three further PI decisions (8–10 above). The pools become
  a cluster job for another developer and the local budget applies to
  the analyses only, so the allocation returns to 100 filtered runs per
  noisy condition over the eight conditions of the table (six noisy, two
  noiseless controls); the identity check compares source only; gpyreg
  moves to the released 1.2.1 everywhere. Done the same day: editable
  gpyreg reinstalled at 1.2.1, a frozen worktree at the tag, the CI pin
  and `pyproject.toml` minimum moved, the full test suite green under
  1.2.1 (1288 passed, 47 skipped, one rerun, 6.5 minutes), the exact
  oracle check 10 of 11 with the VIQR acquisition one ulp off, that
  reference re-baselined with the sanctioned tool and the exact check 11
  of 11 again, the 15 pilot artifacts recomputing exactly under 1.2.1,
  and the two new suite entries (`student_D8_noise3_svbmc`,
  `multisensory_s1_D6_svbmc`). The review of the evidence-bias change
  found no must-fix; its should-fix items (the same reference draws for
  both arms, eight batches of 25, a component-count pairing guard, a
  direct unit test, wording) are being applied in one harness pass
  together with the cluster changes: the source/host identity split, a
  `select` step that defines the filtered pool post hoc, and a `cases`
  enumeration for array jobs.
- 2026-09-14: the harness pass is complete and reviewed. Implemented:
  gpyreg default at the 1.2.1 worktree; identity split into a compared
  source half and a recorded host half, with a fallback that reads the
  pilot's flat records (the pilot directory stays readable but cannot be
  extended under the new pin, as intended); `select` writing
  `selection.json` and `selection.md`, read by the comparison and named
  in its `sources.json`; `cases` with `lines` and `json` formats and a
  three-part sbatch sketch in the runner's docstring (login-node
  commands, the array script, the post-array `select` and `summarize`),
  chunked for Slurm's default array limit; `prepare` defaults of
  100/150, controls 50/75, the ring's cap 200, with the precedence
  defaults → `--target`/`--max-seeds` → `--control-*` → `--allocation`;
  the eight `POOL_LABELS`; the evidence-review fixes (the same reference
  draws for both arms, eight batches of 25, a component-count pairing
  guard, a direct unit test of the reference, `headline_bias_growth`,
  a clear error for results files without the reference fields, GFM
  header escaping). The fresh review found two must-fix items, both
  fixed: the array-path worker now records its own failure
  (`<tag>.error.txt`, partial artifact removed, non-zero exit, stale
  file cleared on success) and the tracked pilot comparison was
  regenerated with the final harness; also fixed: `--summarize-only`
  describes a run from its recorded settings, `cases` and `worker`
  refuse an unauthorized manifest, `select`'s pass rate is named as the
  prefix quantity it is, `growth_within_bound`, `pool_entry` through
  `record_path`, the experiments README. Tests: 38 pool-generator and
  16 comparison tests pass. The superseded 1.2.0 worktree was removed.
- 2026-09-14: everything merged into `dev-next` (`1903506`) and pushed;
  the feature branch deleted. End-to-end check of the two conditions
  the pilot had not run, one seed each, from the clean committed tree
  through `prepare` (no `--allow-dirty`), `cases`, `run`, `select` and
  `summarize`: `student_D8_noise3_svbmc` 3.8 min, 280 evaluations,
  `K = 50`, passes the filters, 95 KB artifact; `multisensory_s1_D6_svbmc`
  0.9 min, 150 evaluations, `K = 50`, passes, 118 KB. Both artifacts
  verify post hoc with exact zeros on the recomputation gate under
  gpyreg 1.2.1, and two rebuilt copies of each stack in the integrated
  class (the noisy one with the capped headline, the noiseless one raw).
  All eight conditions have now run end to end on the pilot machine. The hand-off note
  for the cluster developer is
  [svbmc-pool-handoff.md](svbmc-pool-handoff.md).
- 2026-09-14: the launch-authorization gate of the pool generator
  (`launch_ready`, `--ready`, `--authorized-by`, and the refusals in
  `run`, `cases` and `worker`) was removed (PI). It had grown out of the
  laptop runner's guard against starting a half-written manifest and a
  reviewer's request to record who flipped it, and on the cluster it
  protected nothing: the person preparing the manifest is the person
  submitting the array, and whether a campaign may start is a decision
  recorded in this plan. The manifest keeps its provenance (identity,
  allocation, timestamp); the source-identity comparison, the
  clean-tree refusal, the hash-verified records, the partial-artifact
  stop and the worker's error file stay, since they protect the data.
  The pool archive comes back as a draft-release asset with its hash in
  the PR, since the PI has no direct access to the cluster.
- 2026-09-14: the optimism note's Phase 2 estimator is prototyped on the
  pilot artifacts while the pools are generated on the cluster
  (`dev/scripts/svbmc_honest_elbo.py` and `test_svbmc_honest_elbo.py`,
  12 tests; developed on `dev-svbmc-honest-elbo`, merged into
  `dev-next`, then revised on `dev-svbmc-honest-elbo-review` after an
  independent review of the code and of the record). It scores every
  cell of a comparison's `results.json` against the pool artifacts: the
  covering other runs' GP estimates of each component's expected log
  joint under a grid of coverage rules and combinations, the bias of raw,
  capped and honest values against the cell's `e_log_joint_mc`,
  per-component truths, calibration `z` statistics, the data each run
  had at each component, and own-run consistency checks. Run on the 25
  pilot cells (15 runs, 41 s) and, as a self-check, on the two later
  conditions' artifacts under gpyreg 1.2.1; every check passes. At
  `M = 3` the honest value is accurate where other runs cover a noiseless
  target, closer to the reference than raw and capped on multisensory,
  and *low* on the noisy GMM and Rosenbrock by more than raw is high;
  the component-median cap is closer than the honest estimate on four of
  the five conditions. The report
  [results/2026-09-14-svbmc-honest-elbo-pilot.md](../results/2026-09-14-svbmc-honest-elbo-pilot.md)
  records the numbers, the decomposition of the cross-run bias into the
  own run's data and the other run's fit, and the open questions for the
  full pools; tracked outputs are under
  `dev/experiments/svbmc_pool/pilot/phase2/` and
  `pilot/phase2_newconds_selfcheck/`. The pilot artifacts were scored
  under the campaign's gpyreg pin 1.2.1 (`--gpyreg-source`), not the
  1.2.0 their manifest names; a scoring under 1.2.0 gave every number
  identically, and that worktree is not kept (decision 10).
- 2026-09-14: the pool was generated on the University of Helsinki `kale`
  cluster by the cluster developer (PR from `dev-svbmc-pool-hpc`), as a
  Slurm array of `worker` tasks from commit `d63c477` of that branch (a
  descendant of `dev-next` `613f2a8`; the branch adds the `verify`
  subcommand to the runner and the Slurm scripts under
  `dev/scripts/hpc/`) with gpyreg 1.2.1 (`9e70e6b`) in a dedicated conda
  environment (Python 3.12.14, numpy 2.5.3, scipy 1.18.1, OpenBLAS,
  single-threaded): a one-task canary, then `--array=2-1100%200`; 1100
  tasks, all completed on one core each, 33 minutes of wall time and
  90.5 CPU-hours (elapsed median 4.6 min, maximum 11.5 min on Student D8,
  peak memory at most 302 MB, so the 30-minute and 2 GB requests were
  ample; nodes of three families, recorded per case). `verify`: 1100
  verified, no failed, partial, missing or stray case; the recomputation
  gate is exact on 1094 artifacts and within 3.0e-9 (`I_sk`) and 1.5e-9
  (`J_sjk`) on six (three generated on the Intel `kale` nodes and three
  on AMD `ukko3` nodes; the verifying node was an AMD `carrington`).
  `select`: every condition at its
  target, 700 selected runs, no shortfall; the ring needed 142 seeds
  (pass rate 0.73 over its 200 runs: 48 unstable, 6 above `s_max`), every
  other condition 100–103 seeds (pass rates 0.97–1.00). Cluster wall
  times per run are about twice the laptop pilot's. The archive
  `svbmc_pool_20260914.tar.zst` (88,740,881 bytes, SHA-256
  `f2863d96…7bb1`) is an asset of the draft release `svbmc-pool-20260914`
  (target `d63c477`); the tracked copies and a README with every key are
  under `dev/experiments/svbmc_pool/pool_20260914/`. For the analysis
  machine: the manifest stores the cluster's absolute `gpyreg_source`,
  which `svbmc_pool_stack.py --pool` reads and does not let
  `--gpyreg-source` override, so that one field must be pointed at a
  local 1.2.1 checkout before stage D; `verify --gpyreg-source` re-checks
  the unpacked copy first. Stage D and the Phase 2 study can start on the
  pool once the PR is reviewed.
- 2026-09-14: the pool PR (#177) reviewed and merged into `dev-next`
  (`2738a2f`). The review found one must-fix, a records count in the
  submit script that ends the script on a fresh campaign under
  `pipefail` (added after the cluster run, never exercised), and the
  should-fix items were applied in a follow-up commit on the branch
  before the merge; the module's 43 tests pass on the analysis machine.
  The archive was downloaded, its SHA-256 checked, and unpacked to
  `dev/scripts/runs/svbmc_pool_20260914/` (4411 files, 121 MB; the
  cluster's `verification.json` is kept beside the local one as
  `verification.cluster.json`). `verify --gpyreg-source` there, on
  Windows with NumPy 2.5.2 and scipy-openblas 0.3.34, did **not** pass
  the absolute recomputation gate: 902 artifacts within 1e-8 and 198
  beyond it (126 Rosenbrock, 70 ring, 2 noiseless multisensory). A
  recomputation of all 1100 artifacts on this machine explains it: the
  relative deviation of the recomputed `I_sk` and `J_sjk` from the
  stored ones is at most 1e-8 on six conditions (typically 1e-11), at
  most 6e-7 on the ring, and up to 0.29 on Rosenbrock at noise 3, where
  80 of the 100 selected runs deviate by more than 1e-6 and 67 by more
  than 1e-3; case by case the deviation equals machine epsilon times
  the condition number of the GP's kernel matrix (condition 3.9e15
  gives an observed 0.13 against a predicted 0.88, condition 465 gives
  1e-13 against 1e-13). The Rosenbrock GPs are ill-conditioned because
  their training sets hold far-tail evaluations, log densities down to
  -1.5e6 among the selected runs (-6.5e9 in an unselected one), 17 to
  100 points below the maximum minus 1e3, with mean-function
  hyperparameters of 1e3 to 5e5; the laptop pilot's three Rosenbrock
  runs have the same structure (conditions 1e12 to 1e14), so this is
  how the noisy Rosenbrock configuration behaves and not a cluster
  effect, and the pilot only recomputed exactly because it was checked
  on the machine that generated it. The PI's decision: lenient as
  needed if it is rounding, unless a better fix exists. Implemented as a
  conditioning-aware gate rather than a blanket tolerance:
  `verify_run` allows each artifact 100 times epsilon times its own
  condition number, relative to the largest stored value, on top of the
  absolute 1e-8, and reports the relative deviation and the condition
  number (`--rounding-factor` scales the allowance, 0 restores the
  absolute gate); `verify` records the verifying checkout's identity
  next to the pool's and the per-condition maxima. Under it the copy
  verifies, 1100 of 1100, with per-condition maximum relative
  deviation and condition number: multisensory noise 3 2.2e-11 /
  7.1e4, noise 1.3 2.4e-11 / 2.2e5, Rosenbrock 0.29 / 2.9e17, noisy GMM
  7.5e-11 / 3.0e5, ring 5.9e-7 / 1.9e9, Student 6.6e-13 / 1.4e3,
  noiseless GMM 1.3e-10 / 8.8e7, noiseless multisensory 2.6e-8 /
  4.7e10. In the same pass `svbmc_pool_stack.py --pool` accepts
  `--gpyreg-source` at the manifest's gpyreg commit through the check
  `verify` uses (`pinned_gpyreg_source`), recording in `sources.json`
  which of the two named the checkout, and the comparison keeps the
  laptop awake as the pool runner does; 45 pool-generator and 17
  comparison tests pass. What this means for the analyses: both arms
  stack the stored statistics, the cluster's, so stage D is internally
  consistent on every condition; the Phase 2 estimator re-evaluates the
  runs' GPs on this machine, where the Rosenbrock GPs' predictions
  carry the same platform-level sensitivity (relative deviations up to
  0.3 in `I_sk`-like quantities), which its report must state; whether
  the Rosenbrock noise-3 condition is kept with that caveat,
  regenerated under a change that keeps far-tail evaluations out of the
  GP, or dropped from the evidence analysis is open for the PI.
- 2026-09-14: stage D authorized by the PI once the checks above were
  clean, and launched from `dev-next` `744864a` (clean tree) into
  `dev/scripts/runs/svbmc_pool_20260914_stack/`:
  `svbmc_pool_stack.py --pool dev/scripts/runs/svbmc_pool_20260914
  --gpyreg-source dev/scripts/runs/svbmc_pool_20260913/gpyreg_1.2.1
  --seed 0` with the harness defaults, which are the decision-6 grid
  (`M ∈ {2, 4, 8, 16}`, `R = 20, 20, 20, 10`, 500 Adam steps), both arms
  on all eight conditions from their `selection.json`, 560 cells, Torch
  from the overlay, BLAS and Torch single-threaded, one process. The
  `M = 32` cells for the integrated arm alone, which the cost paragraph
  proposed, are not part of this run: the harness stacks both arms on
  one grid, and running the original arm at `M = 32` would add about an
  hour per condition; a per-arm cap is a small harness change if that
  regime is wanted. The first cells take about 5 s (`M = 2`), in line
  with the pilot's timings, so the run is expected to take under five
  hours; the controller's log is `controller_<timestamp>.log` in the
  output directory and `cells.jsonl` grows one line per finished cell.
- 2026-09-15: stage D complete, 560 cells in 485.8 minutes (8.1 hours;
  the plan's 5-hour figure extrapolated the pilot's `M = 3` timings
  with `M²`, and the measured cells cost about 2 s at `M = 2`, 8 s at
  `M = 4`, 47 s at `M = 8` and 240 s at `M = 16`, of which the
  integrated arm takes 57 s, the original 162 s and the entropy
  reference 23 s, with the multisensory conditions the slowest at about
  95 minutes each). Against the acceptance criteria: **1** holds, the
  all-`M` median paired `max |Δw|` is 0.007 (ring) to 0.028 (Student)
  per condition, inside Phase 4's within-arm spread of 0.018–0.048;
  **2** holds, no Holm rejection in either direction on any of the 32
  condition-and-`M` cell sets for MMTV or gsKL; **4** holds, the median
  runtime ratio integrated / original is 0.25 (multisensory noise 3) to
  0.54 (noiseless GMM) with every upper interval at or below 0.60;
  **5** holds, `--summarize-only` on the recorded `results.json`
  reproduces `summary.md` exactly and `summary.json` up to the
  timestamp and the `arms_present` field the harness gained the same
  day. **Criterion 3 fails on one condition.** Its gate (the median
  bias of the integrated headline no larger in magnitude than the
  original's raw estimate's) holds on seven conditions at every `M`,
  with the raw bias growing with `M` on every noisy condition
  (Rosenbrock 0.25 → 0.74, noisy GMM 0.30 → 0.73, ring 0.41 → 0.76,
  multisensory noise 3 0.77 → 1.31 from `M = 2` to `16`) while the
  capped headline stays within 0.05–0.56 nats of the reference; on
  Student D8 the gate fails at every `M`: the capped headline sits
  0.95 (`M = 2`) to 1.78 (`M = 16`) nats **below** the stack's own
  Monte Carlo ELBO while the original's raw estimate is within ±0.41
  and the integrated raw within ±0.27, so the component-median cap
  over-corrects there and deepens with `M` (the growth bound, which
  only limits an increase, holds everywhere; Student's growth is
  −0.84). The stacked posteriors improve with `M` on every condition
  (MMTV from the single-run 0.28 to 0.12 at `M = 16` on multisensory
  noise 3, 0.63 to 0.13 on the ring, 0.48 to 0.11 on noisy GMM; the
  KL gap of the ring from 1.01 to 0.15 nats). The tracked copies are
  `dev/experiments/svbmc_pool/stack_20260914/` (`summary.json`,
  `summary.md`, `sources.json`; the 17.6 MB `results.json` stays with
  the raw directory until it is decided whether to track it or publish
  it as a release asset like the pool). The Phase 2 scoring of the 560
  cells was launched from `dev-next` `cd59443` into
  `dev/scripts/runs/svbmc_pool_20260914_phase2/` once the per-arm
  harness change had merged (`svbmc_pool_stack.py --arms integrated`
  and `--summarize-only` over several `--from-results` files, 19
  comparison tests, the integrated-only run reproducing the two-arm
  run's integrated fits exactly); the `M = 32` cells for the integrated
  arm alone follow it, 10 repetitions on all eight conditions, about
  6 minutes per cell from the measured `M = 16` cost, so about 8 hours
  rather than the 2 the cost paragraph's extrapolation gave.
- 2026-09-15: the Phase 2 scoring of stage D's cells is done (695 runs
  checked, 560 cells, 2.0 hours; 100 draws per component, coverage
  ratio 2, median combination; tracked under
  `dev/experiments/svbmc_pool/phase2_20260915/`, the 27.6 MB
  `results.json` and the per-cell arrays with the raw directory). The
  cross-run estimate of the expected log joint is within 0.01 nats on
  the noiseless GMM and 0.035 on the noiseless multisensory control at
  every `M`, and within 0.07 on Rosenbrock (where nine runs are flagged
  by the own-run check, the platform sensitivity of that condition's
  GPs); it under-predicts on noisy GMM (−0.21 to −0.29 at `M ≥ 4`) and
  on both multisensory conditions (−0.15 to −0.29 at noise 1.3, −0.30
  to −0.71 at noise 3, growing with `M`), the pilot's systematic
  under-prediction at scale, where the component-median cap is closer
  (+0.005 to +0.07 and +0.11 to +0.56); on the ring the two are
  comparable (honest +0.22 to −0.10, cap +0.20 to +0.14); and on Student
  D8 it is as low as the cap, −1.18 to −1.49 against the cap's −0.91 to
  −1.75, with the raw within 0.27. One reading fits both failures on
  Student: the GPs' negative-quadratic mean falls off faster than a
  Student-t log density, so predictions away from a run's own data,
  which is where the tail components and the other runs' evaluations
  sit, are well below the truth, and any estimator built on them (the
  median over all components, the cross-run estimates) is dragged down
  while the raw value, built on the selected central components with
  their own data, is not; the estimator's decomposition records are
  where to test that. The weight-aware cap the PI proposed the same day
  was tried on the recorded cells without refitting
  (`dev/scripts/svbmc_cap_kappa.py`, tracked under
  `dev/experiments/svbmc_pool/cap_kappa_20260915/`): order the
  components by weight, take the shortest prefix carrying mass `κ`, the
  crossing component included, and cap at the median of that prefix's
  expected log joints; `κ = 1` is the class's cap and reproduces it
  exactly. For `κ ≤ 0.8` the cap binds on at most 30 % of the cells and
  the headline is the raw value on every condition (multisensory noise
  3 at `M = 16`: +1.30 against raw +1.31); `κ = 0.9` to `0.99` binds on
  most cells but leaves +0.3 to +1.0 of the optimism on the typical
  noisy conditions while still over-correcting Student by −0.3 to −1.2;
  the weighted median of the expected log joints is the raw value
  everywhere. So the cap's debiasing comes entirely from the low-weight
  components it includes, which serve as a control group for the
  winner's curse of the selected ones, and on Student that control
  group is worse than the selected components for a real reason, not
  by luck; no `κ` separates the two cases. The stage D report carries
  both sections. The `M = 32` run for the integrated arm alone is ready
  (`--arms integrated`, the merged summary through several
  `--from-results`) and, by the PI's instruction of 2026-09-15, waits
  for an overnight slot rather than a working day; it costs about 8
  hours.
- 2026-09-15: the PI's decisions on the day's results: the `M = 32`
  integrated-arm run happens later, overnight on the analysis machine
  or on the cluster; the per-cell outputs of stage D, the Phase 2
  scoring and the weight-aware caps, too large to track, are published
  as `svbmc_pool_20260914_analyses.tar.gz` (113 585 523 bytes, SHA-256 `59ada698d573f40e733b598b4cc667b571b552d3da261b7168f2c18edd06aa3b`), the asset of the draft release `svbmc-analyses-20260915` (target `24646b6`), the experiments README's "Raw outputs
  of the analyses" saying what it holds and how it unpacks; and the
  decision on the reported estimator is postponed, 1.5 keeping the
  current headline (the component-median cap on a noisy stack) with
  every variant in `elbo_details` and a documented caveat for
  heavy-tailed targets until it is made.
- 2026-09-15: three further estimates scored on stage D's recorded
  cells, each in minutes without refitting (the scripts rebuild every
  cell's stack from the artifacts and re-evaluate at the recorded
  weights): run-level caps (`E_max`, `E_top`, in `svbmc_cap_kappa.py`),
  empirical-Bayes shrinkage of the components' expected log joints by
  their estimation covariance, from the `J_sjk` and `I_sk` statistics
  saved with each posterior (`svbmc_shrink_elbo.py`: within a
  run, over the stack, and a two-level form with each run's level
  shrunk toward the runs' mean), and a `hybrid` rule that applies the
  cap only where the noise share is at least 0.2. Because users stack
  three to five runs, the integrated arm was also run at `M = 3` and
  `5` (320 cells, 15 minutes) and scored the same way, and the
  harness's merged `--from-results` gives one summary over every `M`.
  The stage D report's sections "Weight-aware caps" and
  "Empirical-Bayes shrinkage" carry every number and the reading; the
  tracked directories are `shrink_20260915/`, `stack_M35_20260915/`,
  `shrink_M35_20260915/`, `cap_kappa_M35_20260915/` and
  `stack_merged_20260915/`. The PI endorsed the two-level shrinkage as
  the headline candidate for noisy stacks and the hybrid as the one to
  revisit with more conditions; the
  [headline note](../2026-09-15-svbmc-headline-shrinkage.md) is the
  summary and the decision record.
- 2026-09-15: doublecheck of the day's work by three independent
  reviewers (write-ups, code, methodology with a recomputation of every
  table from the tracked cells). Code: the shrinkage's `τ²` subtracted
  the mean diagonal of the estimation covariance, which is the noise in
  a spread of independent estimates only; for one GP's correlated
  estimates the noise term is `(tr Σ − 1ᵀ Σ 1 / K) / (K − 1)`, 20 to
  50 % smaller, so `τ²` was biased low and the noise shares high (by
  about a factor of two on Student). Corrected and re-scored: the
  shrinkage biases moved by a few hundredths, the corrected noise
  shares are 0.21 to 1.1 on the typical noisy conditions and 0.05 on
  Student, and the hybrid's worst case is unchanged with its plateau
  now at thresholds 0.08 to 0.20. Also fixed: the reported factor of
  the full-covariance forms (row sums of the shrinkage matrix), a
  singular solve at `τ² = 0`, undefined noise shares, merged summaries
  refusing mixed arm sets and comparing every shared setting, `κ = 1`
  including every component, the gate's rounding term at a factor of
  0, and provenance records (`sources.json`) for the two scoring
  scripts. Write-ups: every table reproduced from the tracked evidence;
  several summarizing sentences did not (the `κ ≤ 0.8` claim, the
  "within 0.06 nats" claim about the run-level term at `M = 4`, the
  Student single-run error, "halves", the hybrid's control cells and
  threshold plateau) and were corrected from the regenerated numbers;
  the inference from the run-level caps, the Student mechanism and the
  headline-to-raw "range" were restated as the evidence supports
  (paired bootstrap intervals over cells added where a comparison rests
  on them, with the caveat that cells share runs). Tests: 45
  pool-generator and 19 comparison tests pass.
- 2026-09-15: hand-off. The `M = 3` and `5` run's per-cell outputs are
  the second asset of `svbmc-analyses-20260915`
  (`svbmc_pool_20260914_stack_M35.tar.gz`; the experiments README has
  the size and hash); `dev/scripts/svbmc_headline_numbers.py`
  regenerates every number the headline note and the report's
  shrinkage and cap sections quote from the tracked cells. The
  per-phase checklists above are reconciled with this worklog (every
  check of Phases 1–3 and 5 passed when its phase completed; Phase 6's
  acceptance is met except the `M = 32` extension), and the status
  paragraph and `dev/TODO.md` state what remains: the `M = 32` run,
  and the two decisions of the debiasing item (whether the two-level
  shrinkage ships in 1.5, and the headline's user-facing caveat).
- 2026-09-15: the yardstick restated (decision 11) and the baseline
  measured. `svbmc_single_run_bias.py` scores every filtered run of
  the pool as a stack of one against its own `elbo_mc` (the
  comparison's reference estimator at the run's own weights; 700 runs
  in 10.2 minutes) and joins the biases to the 880 cells of stage D
  and the `M = 3` and `5` run; tracked under
  `dev/experiments/svbmc_pool/single_run_20260915/`. A single run's
  reported ELBO is optimistic by 0.13 (Rosenbrock) to 0.74 nats
  (multisensory noise 3) in the median on the typical noisy
  conditions, pessimistic by 0.19 on Student D8, within 0.03 on the
  controls; the class's raw value for the run alone agrees within
  0.02. Added bias (the stack's bias minus its inputs' mean, medians
  over cells): raw +0.02 to +0.15 at `M = 2`, +0.13 to +0.36 at
  `M = 5`, +0.27 to +0.60 at `M = 16`, tracking the highest-reported
  input within 0.25; the cap +0.00 to −0.38 on the typical conditions
  (at the inputs' level on Rosenbrock) and −0.76 to −1.54 on Student;
  `run_level` alone removes 15 to 51 %
  of the addition at `M = 5`; `two_level_full` within −0.23 and
  +0.15 everywhere. The report's section "The inputs' own bias" and
  the headline note carry the reading; the anchored variant (add
  back per run what the shrinkage removes at the run's own weights)
  is the untested next candidate.
- 2026-09-15: the anchored variants tested. `svbmc_shrink_elbo.py`
  gained `anchored` and `two_level_anchored` (add back, per run and
  weighted by its stack mass, what the within-run shrinkage removes
  at the run's own weights, so that a stack of one run reports its
  own value) and `anchored_mean`, `two_level_anchored_mean` (the mean
  over runs added back); both scorings were rerun (the earlier
  variants' values unchanged to the last digit) and the single-run
  join regenerated. Added bias over the noisy conditions: `anchored`
  +0.06 to +0.32 at `M = 5` against raw's +0.13 to +0.36, so the
  stacking's addition is not a reweighting within runs;
  `two_level_anchored` +0.01 to +0.18 at `M` = 3 to 5 and +0.05 to
  +0.42 at `M = 16`, the mean form within 0.06 of it, the two
  together removing 16 to 114 % of the raw addition at `M ≥ 3`.
  Neither meets the yardstick; the two-level
  shrinkage without the add-back (−0.23 to +0.15) stays the
  candidate. Reading: the run-level term's error variance, `wᵀ Σ w`
  at fixed weights, leaves out the run-to-run scatter of the runs'
  own optimism, which is what the stacking selects on; a run-level
  error that includes it is the refinement to try.
- 2026-09-15: re-optimizing on the shrunken estimates tested
  (`svbmc_shrink_optimize.py`: the class's corrected expected log
  joints replaced by the two-level full shrinkage before `optimize`,
  the comparison's settings, the new stack scored against its own
  `elbo_mc`; `M` = 3, 4 and 5 on every condition, 480 cells in 23
  minutes, tracked under
  `dev/experiments/svbmc_pool/shrink_opt_20260915/`; the tracked run
  shifts the optimizer's warm start by the change of each run's level
  under the shrinkage, a first run with the unshrunk warm start gave
  the same medians within 0.02). The re-optimized headline is 0.05 to
  0.40 nats more optimistic than the value-only shrinkage on the two
  multisensory conditions, Rosenbrock and noisy GMM (added bias +0.07
  to +0.29 on the last two) and within 0.11 of it on the ring and
  Student: the optimizer selects on the shrinkage's own errors. The
  posterior improves on the multisensory conditions and the ring
  (gsKL ratios 0.71 to 0.83, the KL gap smaller by 0.02 to 0.09), is
  a little worse on Student (ratios 1.05 to 1.07, KL gap +0.03) and
  worse on Rosenbrock (gsKL ratio 5.8, KL gap +0.09) and on GMM's KL
  gap (+0.06); a tenth to a half of the mass moves in the median, the
  largest single weight change exceeds 0.08 on a tenth of the cells
  (45 % on Rosenbrock, up to 0.98). Rejected: the shrinkage stays a
  correction of the reported value at the raw weights.
- 2026-09-15: doublecheck of the single-run baseline, the anchored
  variants, the re-optimization and the day's documents by three
  independent reviewers (code, numbers, prose). Code: the `added.md`
  delimiter row had one cell too many (no table rendered); the
  single-run `--reuse` pass left `sources.json` and the summaries
  describing an earlier pass and shared its bootstrap stream with the
  full pass; the re-optimization warm-started from the unshrunk run
  ELBOs and lacked the rebuilt-stack value check; the anchored forms
  were described as reporting a run's own value at `M = 1`, which
  holds at the run's own weights only. All fixed, the join and the
  re-optimization rerun, the worked example of the tutorial made a
  tracked script. Write-ups: every table reproduced from the tracked
  evidence; the prose ranges were narrower than the tables in a dozen
  places, always dropping the edge condition (Rosenbrock for the cap,
  which lands at the inputs' level there; the ring for the anchored
  forms and for one paired bootstrap; Student for the raw value; the
  original arm's 0.9 quoted as the port's), one "untested" sentence
  survived the test it referred to, the note said the `M` = 3 and 5
  subsets ran both arms, and the hybrid's resolvable gains and losses
  were incomplete; all corrected from recomputation. Recorded as a
  choice with its sensitivity: the added-bias baseline is the plain
  mean over the inputs; against the mass-weighted mean every
  estimate sits lower.
- 2026-09-16: the `M = 32` extension. The integrated arm alone at
  `M = 32`, 10 repetitions per condition (`svbmc_pool_stack.py --M 32
  --repetitions 10 --arms integrated`, stage D's other settings, from
  `dev-next` `1888688`), authorized by the PI for the night of
  2026-09-15 and run on the analysis machine: 80 cells in 237 minutes
  (the fit 2.4 to 3.9 times its `M = 16` time, under the `M²`
  extrapolation that had put the run at 8 hours). Scored with
  `svbmc_shrink_elbo.py` and `svbmc_cap_kappa.py`, joined to the
  single-run baseline (`--reuse`, the third `--cells` and `--shrink`
  files) and merged into the one summary over every `M`; tracked
  under `stack_M32_20260915/`, `shrink_M32_20260916/`,
  `cap_kappa_M32_20260916/` and the regenerated `single_run_20260915/`
  and `stack_merged_20260915/`; the per-cell files are the third asset
  of `svbmc-analyses-20260915` (`svbmc_pool_20260914_stack_M32.tar.gz`,
  2 943 060 bytes, SHA-256
  `11780b8d7998d2e6742d47b1a92980eeb89a6872813062f303d6ba65cc41d171`).
  `svbmc_headline_numbers.py` reads the new cells and prints every
  aggregate over `M` both for the grid through 16 and for every `M`,
  so the sentences written before this run stay checkable. Findings
  (the report's section "The integrated arm at `M = 32`"): posterior
  quality still improves (MMTV down on six conditions, the KL gap
  within 0.2 nats of the target on seven); the raw bias reaches +0.71
  to +1.45 on the typical noisy conditions and +0.38 on Student,
  adding +0.36 to +0.79 to the inputs'; the cap stays at +0.04 to
  +0.45 on the typical conditions (headline growth from `M = 2` at
  most +0.11) and −1.85 on Student; `two_level_full` +0.86 / +0.43 /
  +0.06 / +0.21 / +0.10 / −0.06, adding −0.15 to +0.19, the cap closer
  on the multisensory conditions by 0.41 and 0.24; the run-level term
  removes 15 to 55 % of the addition; the hybrid's worst case 0.45;
  and on the noiseless multisensory control stacking 32 runs adds
  0.11 nats of optimism to the 0.03 its inputs carry, which no
  estimate removes. The campaign item of `dev/TODO.md` closes; the two
  decisions of the debiasing item remain. The PI's reading of the
  same day: the two contributions to a reported value's bias are to
  be stated separately in every document, VBMC's own (what each run
  carries, the separate item) and what S-VBMC adds; the headline
  note's decision paragraph and the TODO's decision (2) were restated
  that way. Decision 12, the same day: the two-level shrinkage is the
  PI's leaning for the noisy headline, confirmed at the release gate
  on fresh pools rather than on this campaign; the run reuse of the
  present pools at `M` = 16 and 32 is quantified in the report's
  Limitations and the gate is a TODO item of its own.

## Execution tracking

Live status of the phases above (`[ ]` not started, `[~]` in progress,
`[x]` complete, `[!]` blocked). Implementation runs on `dev-svbmc-pool`
(branched from `dev-next` at `4216b6d`, 2026-09-13).

- [x] Phase 1: baseline environment, agreement spread, save-contract probe (Opus sub-agent; 2026-09-13, all checks pass; see worklog)
- [x] Phase 2: targets (Opus sub-agent; 2026-09-13, all checks pass; ring ported without the log-radius term, see worklog)
- [x] Phase 3: pool generator (Opus sub-agent; 2026-09-14, 16 tests pass, manual gate on `rosenbrock_D2_noise3_svbmc` verified and stacked; see worklog)
- [x] Phase 4: pilot (Fable; authorized by the PI on 2026-09-13; run 2026-09-14 from the harness commit `e2aaef5`, campaign directory `dev/scripts/runs/svbmc_pool_20260913/pool/`, three seeds per condition, `--save-vbmc`; 15/15 runs pass the filters, all artifacts verify, `M = 3` stacking measured in both arms; stages B–D await authorization)
- [x] Phase 5: stacking comparison harness (Opus sub-agent; 2026-09-14, steps 1–3 and 5 done, 7 tests pass; step 4, the dry run on the pilot artifacts, waits for Phase 4)
- [x] Phase 6: campaign, comparison and report (stages B and C became the cluster pool of decision 8; stage D authorized by the PI and launched 2026-09-14 from `dev-next` `744864a` into `dev/scripts/runs/svbmc_pool_20260914_stack/`, the decision-6 grid on all eight conditions, both arms, 560 cells, finished 2026-09-15 in 8.1 hours: criteria 1, 2, 4 and 5 hold on every condition and criterion 3's gate fails on Student D8 at every `M`, see the worklog; the integrated arm alone at `M` = 3 and 5 (320 cells) on 2026-09-15 and at `M = 32` (80 cells, 4 hours) overnight on 2026-09-15/16; the Phase 2 scoring, the weight-aware caps and the shrinkage estimates are done and reported in `dev/results/2026-09-15-svbmc-pool-comparison.md`, the acceptance checks above are met)
- [x] Documentation updates listed above (the experiments README with its per-pool and per-analysis sections, the hpc README, the stage D report and the `dev/README.md` entries; 2026-09-15)
- [x] Doublecheck of the implemented phases (three fresh reviewers on 2026-09-14; every finding fixed and re-verified, see worklog)
- [x] Evidence yardstick change (decision 7): `elbo_mc` with an arm-independent entropy reference, bias and KL-gap columns, criterion 3 gates, `--summarize-only`; reviewed, no must-fix (2026-09-14)
- [x] Harness pass for the cluster (decisions 8–10): gpyreg default at the 1.2.1 worktree, source/host identity split, `select` and `cases` subcommands, approved defaults with an explicit precedence, the bias-review fixes; reviewed, every finding fixed; 38 + 16 tests pass; pilot comparison regenerated with the final harness (2026-09-14)
- [x] Hand-over of the pool generation to the cluster developer (the "Cluster generation" section and the runner docstring are the brief); stage D and the Phase 2 analyses run on the PI's laptop once the pools are back (pool generated on the cluster and handed back 2026-09-14: 1100 runs, 700 selected, every artifact verified there, draft release `svbmc-pool-20260914`; PR #177 reviewed and merged the same day; the copy on the analysis machine verified under the conditioning-aware gate, see the worklog)
- [x] The optimism note's Phase 2 estimator, prototyped on the pilot artifacts (Fable; 2026-09-14; `svbmc_honest_elbo.py`, 12 tests pass, 25 cells scored and the two later conditions self-checked, report under `results/`, reviewed; the study on the full pools waits for them)
