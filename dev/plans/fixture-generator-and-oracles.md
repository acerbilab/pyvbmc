# Fixture generator and stage-level oracles (Stage 0)

Created: 2026-09-04 05:10. Status: **DONE 2026-09-04 07:50** (commit
pending; the CI matrix has not yet run the oracles on another BLAS build,
see the tolerance paragraph). Decided with the PI on
2026-09-04: an arithmetic-preserving refactor (Stage 2) is gated by
fixed-state oracles that run in seconds, not by the 10-hour statistical
golden run, which becomes the end-of-stage check and the Stage 4 gate.
Roadmap pickup point 1 (`plans/modernization-roadmap.md`); rationale in
`dev/2026-09-02-modernization-discussion.md` §10 "Stage 0 — Test oracle"
("regenerable `.npz` fixtures", "stage-level oracles with tight
tolerances", "randomness injected, not drawn"). This file is the plan now
and the worklog afterwards.

## Summary

A generator script runs a handful of short, seeded VBMC runs with
regime-forcing options, snapshots the algorithm state at chosen iterations
**as plain arrays** (never pickles), and stores next to each snapshot the
reference outputs of every numerical stage that Stage 2 will touch. A test
module rebuilds the state through public constructors and compares the
current code's outputs with the stored references at about 1e-10. The same
fixtures serve the PyTorch port later, since they contain no Python
objects.

## Scope

- **In**: `pyvbmc/testing/oracles/` (package: state codec and rebuilders,
  oracle registry, the test module, `fixtures/`), the generator
  `dev/scripts/make_oracle_fixtures.py` (`--list`, `--check`, regenerate),
  `MANIFEST.in`, `AGENTS.md` and `dev/README.md` entries, roadmap ticks.
- **Out**: any change to `pyvbmc/` package code outside `testing/`
  (findings are recorded, not fixed); retiring the `.mat` fixtures (a later
  step, once these oracles cover what they pin); the finite-difference
  checks still missing from Stage 0; the per-PR deterministic replay of
  golden traces (roadmap pickup 2, separate small script); four oracles
  from the devlog's Stage 0 list that Stage 2 does not need yet: the GP
  log marginal likelihood and its gradient (gpyreg-side), `GP.quad`,
  `kl_div_mvn`, `kde_1d`.

## Findings the design rests on

Verified against the code; a read-only Opus review (2026-09-04, 28
findings) corrected the first draft of this section and the design tables,
and its corrections are folded in below and in the code.

- **GP rebuild.** `gp.update(X_new, y_new, s2_new, compute_posterior=False)`
  stores the data without factorizing; `gp.set_hyperparameters(H,
  compute_posterior=True)` with `H` of shape `(Ns, hyp_N)` builds `Ns`
  posteriors through gpyreg's core computation, which reads only
  `(X, y, s2, hyp)`; the Cholesky ladder (`sn2_mult`) and the
  `L_chol` switch are deterministic. `predict(Xs, separate_samples=True)`
  returns `(Nc, Ns)` arrays, the averaged form `(Nc, 1)`. The noise
  function's switches are recorded from `gp.noise.parameters`, not from
  `optim_state["gp_noise_fun"]`: with `uncertainty_handling` but no
  `specify_target_noise` the two disagree (`GaussianNoise` applies the
  scale flag only inside the user-provided branch).
- **`gp.temporary_data` is empty on every recorded GP**: `train_gp` builds
  a new GP each iteration and `active_sample` fills `sn2_new` /
  `X_rescaled` (and `optim_state["gp_length_scale"]`) on the previous one.
  The acquisition oracle therefore *recomputes* them from the rebuilt
  `(gp, logger)` by replaying that block (`_oracles.prepare_gp_for_acq`).
- **Acquisitions.** `AbstractAcqFcn.__call__(Xs, gp, vp, function_logger,
  optim_state)` reads `integer_vars`, `variance_regularized_acq_fcn`,
  `tol_gp_var`, `lb_eps_orig`/`ub_eps_orig`, `gp_length_scale` and the
  temporary data; the subclasses read `function_logger.y_max`; VIQR/IMIQR
  read `optim_state["active_importance_sampling"]` (`X`, `f_s2`,
  `ln_weights`, `K_Xa_X`, `C_tmp`, all arrays), which is the pre-drawn
  randomness of the noisy acquisitions and is stored as recorded.
  `_real2int` mutates its input, so the oracle passes a copy of the
  candidates. Two latent bugs surfaced (devlog §9): the option key is set
  as `variance_regularized_acqfcn`, so the regularization branch is dead
  and the oracle pins the dead path; and `_gp_log_joint(...,
  compute_var=False, separate_K=True)` raises `UnboundLocalError`.
- **`active_sample`** draws from `vp.rng` (acquisition choice, search
  points, the CMA-ES population through a custom `randn`, importance
  sampling) and from the legacy global state (`cma.NoiseHandler`; gpyreg's
  slice sampler when a per-sample refit happens); the target itself may
  hold a third stream (noisy problems). It reads `iteration_history` only
  when the per-sample full update is on, and then pulls whole GP objects
  out of it through `train_gp`; `specify_target_noise` turns that update
  on through `Options.update_defaults`, so the `active_sample_step` oracle
  is defined only for snapshots with the full update off (all noiseless
  ones), with a dict stand-in for the history that carries `r_index`.
- **Transformer.** Constructor `(D, lb, ub, plb, pub, scale,
  rotation_matrix, transform_type)`; `plb`/`pub` are consumed, not
  stored, so they come from `optim_state["plb_orig"/"pub_orig"]`. After a
  rotoscale warp `whitening.py` sets `R_mat`, `scale` **and resets `mu` to
  0 and `delta` to 1**, so `mu, delta, type, R_mat, scale` are all assigned
  from the snapshot after construction. Methods: `__call__`, `inverse`,
  `log_abs_det_jacobian`.
- **VP.** `eta` is *not* derived from `w`: it is written by `_neg_elcbo`
  and `_sieve` and read by the entropies and the softmax Jacobians, and can
  differ from `log w` after `optimize_vp`; it is stored. `bounds` is
  cumulative across `get_bounds` calls and is stored. `get_parameters()`
  renormalizes `lambd`, rescales `sigma` and `w`: oracles call it on a
  copy. `_neg_elcbo` shifts the eta block of `theta` in place: each call
  gets its own copy.
- **Logger.** `add(x, ...)` takes *transformed* `x` and averages repeated
  points, so replaying rows through it cannot reproduce a run's logger;
  the rebuilder assigns the preallocated arrays directly (`X_orig, y_orig,
  X, y, S, n_evals, fun_eval_time, X_flag` for all rows up to `Xn`,
  including rows warm-up trimming switched off, plus `Xn, y_max,
  func_count, cache_count`). `optim_state["N"]` is `Xn + 1`, not the live
  count.
- **Recording seam.** `iteration_history` deep-copies `vp, gp,
  function_logger` right after the variational fit and `optim_state` at
  the end of the iteration, after warm-up trimming (which flips `X_flag`)
  and the termination checks. On the trimming iteration the two describe
  different data; the generator asserts `N == Xn + 1` and `n_eff ==
  Σ n_evals[X_flag]` on every snapshot and refuses the iteration otherwise.
- **Options** are rebuilt as `VBMC.__init__` does (basic `.ini` with the
  user options, advanced `.ini`, `update_defaults`, validation); user
  options win. `update_defaults` flips `max_fun_evals`,
  `tol_stable_count`, `active_sample_*_update` and `search_acq_fcn` when
  `specify_target_noise` is set; the same path runs on rebuild. The
  round-trip check compared all 178 options of a real run: identical.
- **Regimes.** GP hyperparameter sampling stops once `N >=
  stable_gp_sampling` (no warm-up guard; also `vp_K >= stable_gp_vp_k`):
  `Ns_gp` then reads 0 and gpyreg's `fit` with zero samples builds exactly
  one posterior, so the check is `len(gp.posteriors) == 1`, and
  `stable_gp_sampling = 1` forces it from the first fit. The rotoscale
  warp needs `iteration > 0`, `not warmup`, `K >= warp_min_k` (5),
  `r_index < warp_tol_reliability` (3), `D > 1` and the `warp_every_iters`
  delay, and `warp_undo_check` can undo it: seed 0 of `banana_D2` logs a
  warp that leaves the identity, while `corr_D5` and `cigar_D4` end with a
  rotation in place. `entmc_vbmc` has no `K = 1` branch; at `K = 1` the
  sieve sets the Monte Carlo sample count to 0 and `entlb_vbmc`'s exact
  single-component expression runs. Production passes the entropy sample
  count *per component*, `ceil(ns_ent(K) / K)`.
- **`final_boost`** re-optimizes against `iteration_history["gp"][best]`,
  the *best* iteration's GP, and `K = max(K, 50)`.
- **`optim_state`** holds ndarrays, numpy and Python scalars (some
  non-finite), strings, booleans, `None`, empty lists, a list of ndarrays
  (`vp_repo`) and nested dicts of arrays and `None` (`hyp_dict`, `cache`,
  `iter_list`, `active_importance_sampling`); no callables. The codec
  handles exactly these and fails loudly on anything else.

## Design

### Snapshot format

One `<name>.npz` (arrays, `allow_pickle=False`) plus `<name>.json`
(scalars, strings, booleans, `None`, lists and the tree structure, with
the marker `"@array:<key>"` where an array belongs) per snapshot.
Namespaces:

- `gp/X, gp/y, gp/s2` (absent when noiseless), `gp/hyp (Ns, hyp_N)`; JSON:
  `Ns`, covariance id, mean name, `noise_parameters` (the noise function's
  own switches) and the option triple for reference.
- `vp/w, vp/eta, vp/mu, vp/sigma, vp/lambd`; JSON: `K, D`, `optimize_*`;
  `stats` and `bounds` through the codec.
- `pt/lb_orig, pt/ub_orig, pt/plb_orig, pt/pub_orig, pt/mu, pt/delta,
  pt/type` and, when set, `pt/R_mat, pt/scale`; JSON: `transform_type`.
- `logger/X_orig, y_orig, X, y, S (noisy), n_evals, fun_eval_time, X_flag`
  for rows `0..Xn`; JSON: `Xn, y_max, func_count, cache_count,
  total_fun_eval_time, noise_flag, uncertainty_handling_level, cache_size`.
- `optim_state/...` through the codec, recursively.
- `options`: the run's user options (acquisition objects as class names).
- `cand/Xs (Nc, D)`: `Nc = 512` points, a fixed stride through the seeded
  `2^13` sieve of `_get_search_points`.
- `ref/<oracle>/<output>`: the reference outputs.
- JSON `meta`: recipe, config, problem `(name, D, noise_sd, seed)`, the
  iteration and the run's `best_iter`, `r_index`, `K`, `Ns`, `N`, the
  oracle seed, git SHA and dirty flag, package versions, timestamp.

Rebuilding: `build_transformer`, `build_vp`, `build_gp`, `build_logger`,
`build_options`, `build_optim_state` in `pyvbmc/testing/oracles/_state.py`,
each a plain function of the decoded dict; `snapshot_from_vbmc(vbmc, i)`
does the inverse from a live `VBMC` (or from `iteration_history[i]`).

### Snapshots (regime coverage)

All from `dev/scripts/benchmark_targets.py` problems, seed 0,
`display="off"`; recipe options are merged over the problem's own. Runs
total about six minutes.

| name | source run | options forced | iteration | covers |
|---|---|---|---|---|
| `normal_D2_warmup` | `normal_D2` | — | 2 (warm-up, K = 2, Ns = 8) | baseline; K small |
| `normal_D2_K1` | same run | synthetic VP with K = 1 on the iteration-2 GP | — | K = 1: `entlb`'s exact expression, `pdf`, `_gp_log_joint` |
| `normal_D2_singlesample` | `normal_D2` | `stable_gp_sampling = 1`, `min_iter = 0`; assert one posterior | last | the single-posterior (optimize-only) regime of the 2026-09-02 crash |
| `corr_D5_warped` | `corr_D5` | — | last iteration with `R_mat` set (assert) | rotoscale warp at D = 5 |
| `cigar_D4_largeK` | `cigar_D4` | — | last (assert warped) | large K, correlated posterior, warp |
| `cigar_D4_boosted` | same run | — | final VP (K = 50) with the best iteration's GP | `final_boost`-sized VP for `_gp_log_joint`, entropies, the ELCBO |
| `halfnormal_D2_bounded` | `halfnormal_D2` | — | last | probit transform, finite bounds, non-constant log-Jacobian |
| `rosenbrock_D2_noise1_viqr` | `rosenbrock_D2_noise1` | `max_iter = 4`, `min_iter = 0`, `min_fun_evals = 0` | last | noisy path: `S`, `s2`, VIQR/IMIQR with pre-drawn importance samples (no `active_sample_step`: the per-sample full update needs history GPs) |

### Oracles (reference outputs per snapshot; seed = snapshot seed)

| oracle | call | stored | tolerance |
|---|---|---|---|
| `gp_predict` | `gp.predict(Xs, separate_samples=True)` and averaged | `fmu_samples, fs2_samples (Nc, Ns)`, `fmu, fs2 (Nc, 1)` | mean 1e-4 + 1e-10 abs (bit-identical across thread counts; across BLAS builds the per-sample mean moved by 1.2e-6 per element on the bounded snapshot, whose candidates reach far into the probit tails, against 1e-6 for the analytic expectations); variance 1e-3 + 1e-8 abs (floor 2e-5: `k** − vᵀv` cancels near the training points, where the variance is tiny) |
| `vp_pdf` | `vp.pdf(Xs, orig_flag=False, log_flag∈{F,T}, grad_flag=True)`, `vp.pdf(Xs_orig, orig_flag=True)` | values `(Nc, 1)`, gradients `(Nc, D)` | 1e-10 (floor 0; no GP involved) |
| `acq_<name>` | after `prepare_gp_for_acq`: each of `AcqFcnLog, AcqFcn, AcqFcnVanilla, AcqFcnNoisy` on every snapshot; `AcqFcnVIQR, AcqFcnIMIQR` on noisy snapshots, with the importance samples redrawn from the rebuilt state under the oracle seed (the recorded ones are stale for the recorded GP) | `acq (Nc,)` | log forms and VIQR/IMIQR 1e-5 per element (floor 3e-7: they carry the variance); exponential forms `AcqFcn`, `AcqFcnNoisy`, `AcqFcnVanilla` 1e-3 (floor 1e-5: exp of the log form, so its absolute error becomes their relative error); no absolute tolerance |
| `gp_log_joint` | `_gp_log_joint(vp, gp, True, True, True, False)` (gradients, no variance), the same with `avg_flag=False` (per sample), and `(vp, gp, False, True, True, True, separate_K=True)` (variance, `I_sk`, `J_sjk`) | `G, dG, G_samples, dG_samples, G_var_call, varG, var_ss, I_sk, J_sjk` | GP-solve class, 1e-6 + 1e-10 abs, for `G*`, `dG*`, `I_sk` (Ubuntu floors on `cigar_D4_boosted`: `dG` 1.2e-8, `dG_samples` 2.9e-8, `I_sk` 3.3e-10, `G_samples` 1.5e-10); variance class, 1e-3 + 1e-8 abs, for `varG`, `var_ss`, `J_sjk` (Ubuntu: `J_sjk` 2e-11 absolute on cigar, 2.3e-10 on corr); all zero across thread counts on Windows |
| `neg_elcbo` | `theta = vp.get_parameters()` on a copy, `theta_bnd = vp.get_bounds(gp.X, options, K)`, `Ns = ceil(ns_ent(K)/K)`; gradient calls without variance (MC entropy, then `Ns = 0`), and the `_eval_full_elcbo` call shape (no gradient, full variance, `separate_K`, `ns_ent_fine` samples); fresh `theta` copy and seeded `vp.rng` per call | `theta, F, dF, G, H, F_detent, dF_detent, H_detent, F_full, G_full, H_full, varF, varG_ss, varG, varH` | GP-solve class 1e-6 + 1e-10 abs for everything that carries `G`; `H`, `H_detent` 1e-10 and `theta` 1e-12 (no GP) |
| `entlb` | `entlb_vbmc(vp)` | `H, dH` | 1e-10 |
| `entmc` | `entmc_vbmc(vp, ceil(ns_ent(K)/K), rng=default_rng(seed))` | `H, dH` | 1e-10 while the draw order is unchanged; item 5 (vectorized `entmc`) re-baselines this oracle deliberately after its own finite-difference and statistical checks |
| `transform` | `pt(X_orig)`, `pt.inverse(U)`, `pt.log_abs_det_jacobian(U)` on the live rows | arrays | 1e-12 |
| `active_sample_step` | legacy state and `vp.rng` seeded, `active_sample(gp, fun_evals_per_iter, …)` on deep copies of the rebuilt state, history stand-in `{"r_index": [r]}` | the chosen points `X_new (n, D)` in original space, their `y_new` | atol 1e-8 on points, **on the platform that generated the fixture only** (`meta["platform"]`; `PYVBMC_ORACLES_ALL=1` forces it elsewhere). The first CI run showed why: on Ubuntu's BLAS the CMA-ES search on `cigar_D4_boosted` ended 1.3 away from the reference points while every acquisition oracle before it passed. It is a same-machine determinism check; a rounding change in the acquisition can flip a CMA-ES ranking even locally, and then the `acq_*` oracles are the arbiter and this oracle is re-baselined with a note |

**Tolerances are per element with a robust floor** (for every entry
`|out − ref| ≤ rtol · max(|ref|, q25(|ref|)) + atol`, with `q25` the lower
quartile of the finite `|ref|`; `nan`/`inf` patterns equal; `rtol` may
differ per output of one oracle) and were set from a measured floor, not
chosen: the references are generated with one BLAS thread, and
recomputing them with the default thread count (a change of summation
order, the same kind of difference another BLAS or OS introduces) left the
predictive mean, the densities, `_gp_log_joint`, `_neg_elcbo`, both
entropies and the transformer bit-identical, moved the predictive variance
by ≤ 2e-5 per element, the log acquisition by ≤ 3.2e-7 and the
exponential-form acquisitions by ≤ 1e-5. Each tolerance sits 30–100× above
its floor and still catches a fivefold change of one median-sized variance
entry, a 1e-7 change of one mean entry, and a 1e-9 change of the density
(checked by perturbing stored references). If the CI matrix (other BLAS
builds) exceeds a floor, re-measure it there and record the new value; do
not loosen by guesswork. To measure a floor: run
`OMP_NUM_THREADS=<n> OPENBLAS_NUM_THREADS=<n> MKL_NUM_THREADS=<n> python
dev/scripts/make_oracle_fixtures.py --check --verbose` on the machine in
question (the generator only *defaults* the thread variables to 1); the
`scaled` column is each output's worst per-element error against the
stored single-threaded reference. The perturbation check is three lines:
load a fixture's `.npz`, multiply one median-sized entry of
`ref/gp_predict/fs2` by 5 (or a `ref/gp_predict/fmu` entry by `1 + 1e-7`,
or all of `ref/vp_pdf/pdf` by `1 + 1e-9`), save, run the test, restore.

The test module `test_oracles.py` is parametrized over
`(snapshot, oracle)`, skips oracles not applicable to a snapshot (noisy
acquisitions on noiseless snapshots, the `active_sample_step` when
`dev/scripts` is unavailable), and reports per-output max relative error.
The generator's `--check` runs the same comparison outside pytest and
prints a table, so a refactor branch can be checked in one command.

### Generator

`dev/scripts/make_oracle_fixtures.py`:
- `--list`: the snapshot recipes and the oracle registry.
- default: run the recipes (one process, BLAS single-threaded), write
  `pyvbmc/testing/oracles/fixtures/<name>.{npz,json}` (overwriting), then
  immediately reload and re-run every oracle from the rebuilt state and
  assert equality with what was just stored (round-trip check: the codec
  loses nothing).
- `--only <name>`: one snapshot.
- `--check`: reload the stored fixtures, recompute, compare, print a
  table, nonzero exit on any mismatch.

## Steps

1. `_state.py`: codec (`encode`/`decode`, typed `optim_state` dump),
   rebuilders, `snapshot_from_vbmc`; round-trip check (state → files →
   state → identical oracle outputs) run by the generator after every
   write, with `test_fixture_complete` guarding against half-written
   fixtures.
2. `_oracles.py`: the registry; each oracle a function
   `(state_objects, seed) -> dict[str, ndarray]` with an `applies(meta)`
   predicate and a tolerance.
3. Generator with the recipes; run it; inspect sizes (target: fixtures
   under 2 MB in total) and the per-oracle timings.
4. `test_oracles.py`; `__init__.py`; run it; `MANIFEST.in`.
5. Sanity: perturb one output on purpose (e.g. rescale a stored `fmu` by
   `1 + 1e-9`) and confirm the test fails; restore.
6. Records: `AGENTS.md` (testing conventions: how to regenerate and when a
   re-baseline is legitimate), `dev/README.md`, roadmap ticks, this
   file's tracker. Commit on `dev-next`; the smoke CI runs because
   `pyvbmc/testing/` changes.

## Verification

- [x] Round trip: every oracle output identical before and after
      save/load for every snapshot (the generator asserts it with zero
      tolerance).
- [x] `pytest pyvbmc/testing/oracles -q` green in 18 s (99 passed, 15
      skipped); the `active_sample_step` oracle takes 0.4–4 s per snapshot.
- [x] Deliberate perturbation of a stored reference makes the test fail
      (1e-7 on the predictive mean, 1e-9 on the density).
- [x] Fixtures 1.4 MB in total (8 snapshot pairs, 100–330 KB each);
      `MANIFEST.in` lists `fixtures/*.npz` and `*.json` (they reach the
      sdist; the testing package is not in the wheel).
- [x] Regime assertions hold: one posterior on the single-sample
      snapshot, `R_mat` on the two warped ones, `S`/`s2` on the noisy one,
      K = 50 on the boosted one, K = 1 on the synthetic one; every
      snapshot passes the program-point check (`N == Xn + 1`, `n_eff ==
      Σ n_evals[X_flag]`).
- [x] Full suite still green (`pytest --reruns=5 -x`, one BLAS thread,
      2026-09-04 07:35–07:50): passed, see the tracker.

## Decisions

- **Arrays, not pickles**, for the state: pickled fixtures cannot outlive
  an attribute rename, and a generator that must be re-run under the new
  code to reload them would also regenerate the references it is meant to
  check against. Arrays plus public constructors keep the reference
  independent of class layout and usable by a torch implementation.
- **Reference outputs are stored with the state**, not recomputed from a
  pinned git revision, so the check needs no second checkout; the
  generator's round-trip check guards the codec.
- **Seeded generators stand in for injected samples** where a function
  draws internally (`entmc_vbmc`, `active_sample`), with the caveat
  recorded in the table; injecting samples would need signature changes
  in the package, out of scope here.
- **`Nc = 512` candidates**, a fixed subsample of the seeded sieve set,
  to keep fixtures small; the acquisition is pointwise, so coverage does
  not depend on `Nc`.
- **Snapshots from the benchmark targets**, so profile, golden and oracle
  fixtures share one definition of every problem.
- **Fixtures live under `pyvbmc/testing/oracles/fixtures/`** with the
  tests that use them, as the existing `.txt`/`.mat` fixtures do, and are
  listed in `MANIFEST.in`.
- **Recomputed, not recorded: `gp.temporary_data`, `gp_length_scale` and
  the importance samples.** All three are produced inside `active_sample`
  for the GP of the *previous* fit and are stale for the GP recorded at
  the end of the iteration (found when the noisy snapshot's stored
  `C_tmp` had one training point fewer than its GP). The oracles rebuild
  them from the rebuilt state, the importance samples under the oracle
  seed; the snapshot stores `active_importance_sampling = None`, which
  also keeps the noisy fixture small (its `K_Xa_X`/`C_tmp` are
  `(Ns, Na, N)`).
- **Per-element tolerances with a robust floor, set from a measured
  rounding floor** (see the paragraph under the oracle table), instead of
  a blanket 1e-10 that the machine itself cannot meet for
  cancellation-prone quantities. A first version scaled every entry by
  the array's maximum; the doublecheck showed that this let 40–70 % of
  the entries of the predictive variance and of the exponential-form
  acquisitions (arrays spanning ten orders of magnitude) vary freely,
  and was loosest exactly near the training data where the cancellation
  happens. The criterion now divides each entry's error by
  `max(|ref|, lower quartile of |ref|)`: every entry is load-bearing and
  the floor only protects the genuinely tiny ones.
- **`build_state` works on a deep copy of the snapshot**, so that oracles
  that mutate what they are handed (`get_bounds`, `prepare_gp_for_acq`)
  cannot leak into another test of the same module-scoped fixture.

## Open questions (defaults in bold)

1. Should the oracle tests run in CI by default? **Yes** (seconds; the
   `active_sample_step` oracle is the slowest at a few seconds each).
2. Tolerance for `active_sample_step` points: **atol 1e-8** in original
   space; revisit after the first Stage 2 PR shows how often rankings
   flip.

## Risks

- `optim_state` may hold a type the codec does not expect (found by the
  loud failure; extend the codec).
- ~~`active_sample_step` on the noisy snapshot refits the GP per sample~~
  — resolved by design: the oracle applies only where the per-sample full
  update is off, so the noisy snapshot never runs it.
- Rebuilt `Options` must equal the run's: checked once by hand at Step 1
  (all 178 options of a real run identical, acquisition objects compared
  by class); not automated.

## Execution tracker

Legend: `[ ]` not started, `[~]` in progress, `[x]` done, `[!]` needs
attention. Times are wall clock on 2026-09-04.

- [x] Plan written — 05:10; roadmap pickup point rewritten — 05:00
- [x] Read-only Opus review of the plan (28 findings) — 05:15–05:30;
  folded into §Findings, the design tables and the code (noise switches
  from `gp.noise.parameters`, candidates copied before `_real2int`, fresh
  `theta` per `_neg_elcbo` call, per-component entropy samples, warped
  snapshot from `corr_D5`, boosted VP with the best iteration's GP,
  program-point assertion on every snapshot)
- [x] Step 1 `_state.py` + round trip — 05:40 on the saved `student_D4`
  seed-19 object: GP posteriors, VP, logger, `optim_state` and all 178
  options rebuilt bit-for-bit, transformer equal
- [x] Step 2 `_oracles.py` — 06:00; two unimplemented paths avoided
  (variance gradient with the full variance, `separate_K` without
  variance)
- [x] Step 3 generator + fixtures — first pass 06:40, regenerated 07:05
  after the review's fixes and again for the noisy snapshot (stale
  importance samples: recomputed at oracle time); 8 snapshots, 1.4 MB in
  total, about six minutes of runs; every regime assertion holds (one
  posterior on `normal_D2_singlesample`, `R_mat` on `corr_D5_warped` and
  `cigar_D4_largeK`, K = 50 on `cigar_D4_boosted`, `S`/`s2` on the noisy
  one, K = 1 on `normal_D2_K1`); round trip exact on every snapshot
- [x] Step 4 tests + `MANIFEST.in` — `pytest pyvbmc/testing/oracles`: 99
  passed, 15 skipped (VIQR/IMIQR on noiseless snapshots,
  `active_sample_step` on the noisy one), 18 s, identical with one and
  with the default number of BLAS threads after the tolerance floors were
  measured (07:20) and the criterion made scale-relative (superseded by
  the doublecheck entry below: per element with a floor, 100 tests)
- [x] Step 5 perturbation check — a 1e-7 relative change of a stored
  predictive mean and a 1e-9 change of a stored density both fail the
  test; restored
- [x] Step 6 records — `AGENTS.md`, `dev/README.md`, roadmap, devlog §9
  (two latent bugs from the review); full test suite `pytest --reruns=5
  -x` (one BLAS thread): **516 passed, 15 skipped, 1 rerun, 10:47**
  (07:35–07:50; the 15 skips are the oracle tests not applicable to a
  snapshot)
- [x] **Doublecheck (2026-09-04, 08:00–08:50)**: three read-only Opus
  reviews (code and generator; documentation and the §9 bullets; the two
  commits against the run artifacts), 49 findings, all folded in.
  Code: the comparison criterion (a global scale let 40–70 % of the
  entries of the variance and exponential-acquisition arrays vary freely
  → per element with a lower-quartile floor, `atol = 0` on acquisitions,
  per-output tolerances re-measured: variance 1e-3, log-form acquisitions
  1e-5, exponential forms 1e-3, everything else 1e-10 or exact); the
  global legacy RNG is restored after the two oracles that seed it and is
  never touched by `build_state`; sidecars are strict JSON (tagged
  non-finite floats, LF); array-valued user options are refused instead
  of dropped; the array marker is `@@npz:` and checked against the
  archive; snapshot names are not truncated at a dot; the generator
  asserts transformer/GP/logger consistency on every snapshot and rejects
  unknown `--only` names; `test_fixture_complete` guards against
  half-written fixtures; the third `neg_elcbo` call uses `ns_ent_fine` as
  `_eval_full_elcbo` does; `applicable()` shared by generator and tests;
  fixtures regenerated (1.44 MB). Records: the final-boost devlog's
  over-prediction is about 5 nats in the GP's units, not 15 (a mixed
  coordinate space), with the seed comparisons tightened; the §9 bullets'
  reasons; profile readings (4–13 %, 1.1–1.6×, 41k, 24 %, `f_min_fill`
  under GP training); golden-section counts and rates (12 of 280 runs
  touch the single-posterior regime); the `EST_MINUTES` comment; the
  AGENTS.md wheel wording; sizes and runtime made consistent. Checks
  after the fixes: `pytest pyvbmc/testing/oracles` 100 passed, 15
  skipped, 26 s, identical with one and with the default number of BLAS
  threads; `--check` 8 of 8; three perturbations detected (a fivefold
  change of one median-sized variance entry, 1e-7 on one mean entry, 1e-9
  on the density); full suite rerun after the fixes (`pytest --reruns=5
  -x`, one BLAS thread): green, 517 passed, 15 skipped.
- [x] **Commit and push** (handoff, 2026-09-04): the oracle package (4
  modules + 16 fixture files), `dev/scripts/make_oracle_fixtures.py`,
  `MANIFEST.in`, `AGENTS.md`, `dev/README.md`, this plan, the roadmap and
  the modernization devlog in one commit; the doublecheck corrections to
  the earlier records (final-boost devlog, benchmark plan §Results
  wording, golden README, `golden_trace.py` comment) in a second. The push
  touches `pyvbmc/`, so it runs the reduced CI smoke (Ubuntu / 3.12): the
  first test of the tolerance floors on another BLAS build.
- [x] **First CI run (33841593231, Ubuntu / 3.12): one failure**,
  `cigar_D4_boosted / active_sample_step` (chosen points 1.3 away, 5
  identical reruns), after every acquisition oracle on that snapshot had
  passed; `-x` stopped the run there, so the other snapshots' floors on
  Ubuntu are not yet known. Fix (2026-09-04): the step oracle is gated on
  the fixture's recorded `platform` (skipped elsewhere unless
  `PYVBMC_ORACLES_ALL=1`), the generator records the platform, fixtures
  regenerated (references bit-identical).
- [x] **Second CI run (33862592547): one failure**,
  `cigar_D4_boosted / gp_log_joint`, the first cross-BLAS measurement of
  the GP-solve class: `dG` 1.2e-8, `dG_samples` 2.9e-8, `I_sk` 3.3e-10,
  `G_samples` 1.5e-10 per element, `J_sjk` 2e-11 absolute (`G`, `varG`,
  `var_ss` within tolerance); all zero across thread counts on Windows, so
  a different BLAS build is the larger perturbation, amplified by the
  cigar GP's conditioning. Fix: two tolerance classes, GP-solve outputs
  (prediction mean, expected log joint and gradients, ELCBO and its
  gradient, variances) at 1e-6 + 1e-10 absolute, GP-free outputs
  (densities, entropies, theta, transformer) at 1e-10; recorded in the
  `_oracles.py` docstring and the table above. `-x` again stopped the
  run at the first failure, so later snapshots' floors on Ubuntu are still
  unmeasured.
- [x] **Third CI run (33862923626): one failure**, `corr_D5_warped /
  gp_log_joint`, `J_sjk` only: 2.3e-10 absolute on entries of order 1e-6
  (2.8e-4 of the per-element denominator); every acquisition oracle on
  both cigar snapshots and on corr passed at 1e-5 / 1e-3, and the whole
  cigar_D4_boosted and cigar_D4_largeK sets passed. `J_sjk`, `varG`,
  `var_ss` and the ELCBO variances are differences of nearly equal terms
  like the predictive variance, so they join the **variance class** (1e-3
  relative + 1e-8 absolute); a third class alongside GP-solve (1e-6 +
  1e-10) and GP-free (1e-10). Remaining unmeasured on Ubuntu after corr's
  log joint: corr's prediction, ELCBO, transformer and density, then the
  well-conditioned snapshots.
- [x] **Fourth CI run (33863234640): one failure**, `halfnormal_D2_bounded
  / gp_predict`, `fmu_samples` only: 1.2e-6 per element (7.5e-6 absolute),
  after the whole of `corr_D5_warped` and halfnormal's acquisitions,
  entropies and log joint had passed. The predictive mean at arbitrary
  candidate points, which reach far into the probit tails on the bounded
  snapshot, is worse-conditioned than the analytic expectations under the
  VP (those passed at 1e-6 on all three ill-conditioned snapshots). Fix:
  the prediction mean alone to 1e-4; the log joint and the ELCBO stay at
  1e-6. Remaining unmeasured on Ubuntu: halfnormal's ELCBO, transformer
  and density, then the well-conditioned `normal_D2_*` snapshots and the
  noisy one.
- [x] **Fifth CI run (33863547190): green.** 510 passed, 22 skipped (the
  15 by-design oracle skips plus the 7 platform-gated `active_sample_step`
  tests), 1 rerun, 18:48 on Ubuntu / 3.12. Every oracle on every snapshot
  now holds on a second BLAS build. The tolerance classes as they stand
  after four measurements: GP-free 1e-10; GP-solve (log joint, gradients,
  ELCBO) 1e-6 + 1e-10 abs; predictive mean 1e-4; variance-type 1e-3 +
  1e-8 abs; log-form acquisitions 1e-5, exponential-form 1e-3; the step
  oracle platform-bound. macOS (Accelerate) has not run them yet: the
  full-matrix dispatch before the eventual PR is the next data point, to
  be handled the same way (measure, then set).
